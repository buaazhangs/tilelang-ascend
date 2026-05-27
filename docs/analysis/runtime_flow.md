# TileLang-Ascend 运行与编译流程

## 1. 一句话流程

用户写 Python DSL -> `@T.prim_func` 解析成 TVM `PrimFunc` -> `tilelang.jit/compile` 触发 Lower -> TileLang/TVM Pass 把高层语义降级 -> NPUIR codegen 生成 MLIR/NPUIR -> TileLangIR pass pipeline 继续处理 -> Ascend 编译器产出目标二进制 -> 生成 Python launcher -> 返回可调用 JIT kernel。

## 2. 从用户代码出发

典型入口有两种：

### 方式 A：显式编译

```python
func = vec_add(...)
kernel = tilelang.compile(func, target="npuir")
kernel(a, b, c, shape)
```

### 方式 B：装饰器 JIT

```python
@tilelang.jit(target="npuir")
def matmul(...):
    @T.prim_func
    def main(...):
        ...
    return main

kernel = matmul(...)
kernel(a, b, c)
```

两种方式本质上都会落到 NPU 专用编译器逻辑。

## 3. 前端：Python 函数如何变成 PrimFunc

### 第一步：`@T.prim_func`

`tilelang/language/tir/entry.py` 中的 `prim_func` 本质上是对 TVM Script parser 的一层封装：

- 接收 Python 函数
- 捕获函数闭包里的符号
- 调用 `parse(...)`
- 生成 TVM `tir.PrimFunc`

也就是说，TileLang 的很多前端语法，其实建立在 TVM Script/TIR parser 基础上，再叠加自己的扩展 DSL。

### 第二步：DSL 语义写进 PrimFunc

像这些语句最终都会变成 TIR / 自定义 intrinsic：

- `with T.Kernel(...):`
- `T.alloc_ub(...)`
- `T.copy(...)`
- `T.vadd(...)`
- `T.gemm(...)`
- `for i, j in T.Parallel(...):`
- `for k in T.Pipelined(...):`

其中一部分是 TVM 标准结构，一部分是 TileLang 自定义 op，通过 `_ffi_api` 注册和 Lower。

## 4. JIT 入口如何分流

### `tilelang.compile`

`tilelang/jit/__init__.py` 中的 `compile(...)` 会先看 target：

- `target != "npuir"`：走通用 `JITKernel`
- `target == "npuir"`：走 `compiler_npu().compile(...)`

### `@tilelang.jit`

`@tilelang.jit(...)` 的工作方式是：

1. 缓存被装饰的 Python 工厂函数
2. 真正调用时，先用入参构造出 `PrimFunc`
3. 再调用 `compile(...)`
4. 把结果缓存成 kernel 对象

所以 `@tilelang.jit` 更像“按参数生成并缓存 kernel 的工厂”。

## 5. Lower 主流程

通用 Lower 入口在 `tilelang/engine/lower.py`。

### 5.1 输入整理

如果输入是 `PrimFunc`：

- 从参数列表和 `buffer_map` 里抽出 kernel 参数信息
- 包成 `IRModule`
- 规范化 target / target_host

### 5.2 Phase 1: LowerAndLegalize

在 `tilelang/engine/phase.py` 中定义。

对于 `target.kind.name == "npuir"`，这个阶段较简洁：

1. `tir.transform.BindTarget(target)`
2. `tir.transform.Simplify()`
3. `tir.transform.RemoveNoOp()`

和 CUDA 路径不同，NPUIR 路径不会在这一步做完整的 TileLang frontend legalize 套路，而是保留更多结构给后续 NPU pass。

### 5.3 Phase 2: OptimizeForTarget

对 NPUIR 的核心 pass 顺序大致是：

1. `tilelang.transform.NpuLoopVectorize()`
2. `tilelang.transform.LegalizeNpuirBF16()`，必要时触发
3. `tilelang.transform.PlanAndUpdateBufferAllocationLocation()`
4. `tir.transform.LowerOpaqueBlock()`
5. `tilelang.transform.LowerNpuirBlock()`
6. `tir.transform.RemoveNoOp()`

这一步的目标是：

- 把 NPU 并行/向量化模式落成更低层的 NPU 友好 IR
- 处理 BF16 等目标相关合法化
- 调整 buffer 分配位置，兼顾复用与后续 codegen
- 清掉高层 block 结构，准备进 NPUIR codegen

## 6. NPUIR codegen 流程

### 6.1 TVM IR -> NPUIR/MLIR 字符串

`tilelang/engine/lower.py` 中：

- `target == "npuir"` 时直接走 `device_codegen(mod, target)`
- 根据 `TILELANG_ASCEND_MODE` 选择：
  - Expert 模式：`target.build.tilelang_npuir_apis`
  - Developer 模式：`target.build.tilelang_npuir_dev`

对应 C++ 实现在：

- `src/target/codegen_npuir_api.cc`
- `src/target/codegen_npuir_dev.cc`

返回值不再是普通 runtime module，而是 NPUIR/MLIR 文本。

### 6.2 TileLangIR pass pipeline

拿到 MLIR 字符串后，`lower.py` 会继续用 `tilelang.tladapter.utils.Pipeline` 串额外 pass：

1. `transforms.mlir.canonicalize`
2. `transforms.bishengir.adapt_triton_kernel`

这部分说明项目不是“生成 MLIR 就结束”，而是还会用 tilelangir native module 再加工一次，适配后续 Ascend 编译器。

## 7. NPU 专用 JIT 编译流程

这部分集中在 `tilelang/jit/jit_npu.py` 的 `compiler_npu.compile(...)`。

### 7.1 参数与动态 shape 处理

先做几件准备工作：

- `_extract_param_info`
  - 推导每个参数的 dtype、shape、是否输出
- `_symbolic_var_promoter_pass`
  - 把动态 shape 用到的符号变量提升成显式参数
- `check_debug_op`
  - 检测是否包含 debug intrinsic
- `_parse_grid`
  - 从 PrimFunc 中抽取 `blockIdx.x` 对应的 grid 表达式

### 7.2 调用 Lower 得到 MLIR

这里会调用 `lower(self.mod)`。

返回值可能是：

- 直接 MLIR 字符串
- 或一个 `.mlir` 文件路径

随后会进一步解析：

- kernel 名称
- mix mode（aic / aiv）
- 参数签名
- workspace 信息

### 7.3 调 Ascend 编译器

`_npuir_to_bin_enable_npu_compile()` 会：

1. 把 MLIR/NPUIR 写到临时文件
2. 找到 NPU 编译器路径
3. 组装编译命令
4. 调用外部编译器
5. 读取产出的目标对象文件

这里真正依赖 Ascend 工具链，所以在没有卡/没有完整工具链的机器上不适合尝试运行。

### 7.4 生成 launcher stub

编译出 kernel 后，还要把它包装成 Python 能调用的形式。

核心步骤：

1. 生成 `npu_launcher.h` 对应的 wrapper 源码
2. 编译 launcher stub 为 `.so`
3. 构造 `JitKernel_NPU`

这个对象最终支持：

- `kernel(...)`
- `kernel.get_kernel_source()`
- `kernel.benchmark(...)`
- profiler 等能力

## 8. 执行阶段

JIT 完成后，用户调用：

```python
kernel(a, b, c, ...)
```

运行时主要做：

- 处理输入 tensor 指针
- 处理动态 shape 附加参数
- 分配 workspace / lock 等 runtime 资源
- 启动 Ascend kernel
- 按 `out_idx` 返回输出张量

## 9. 以一个向量加例子串起来

参考 `examples/elementwise/vec_add_1d.py`，实际链路可以概括为：

1. 用户定义 `vec_add(...)`
2. `@T.prim_func` 把 `main` 变成 `PrimFunc`
3. `tilelang.compile(main, target="npuir")`
4. `compiler_npu.compile(main)`
5. `lower(main)` 进入 NPUIR pass 流程
6. C++ codegen 生成 NPUIR/MLIR
7. TileLangIR pipeline 做 canonicalize / adapt
8. 外部 Ascend 编译器生成目标对象
9. Python wrapper + launcher `.so` 生成
10. `compiled_kernel(v1, v2, v3, seq_len)` 启动执行

## 10. Developer 模式与 Expert 模式

从源码看，NPUIR 路径至少分两种工作模式：

- `TILELANG_ASCEND_MODE=Expert`
  - 更偏直接使用 NPU API/codegen 路径
- `TILELANG_ASCEND_MODE=Developer`
  - 更强调编程一致性、自动 buffer reuse、自动向量化、自动 pipeline 等开发体验

这也是 README 和文档里频繁提到两种模式的原因。

## 11. 建议的“理解调试点”

如果以后要继续排查某个 kernel 为什么没按预期 Lower，可以按这条链断点式排：

1. `@T.prim_func` 后的 `PrimFunc.script()`
2. `LowerAndLegalize` 后的 TVM IR
3. `OptimizeForTarget` 后的 TVM IR
4. `device_codegen(...).get_source()` 输出的初始 NPUIR/MLIR
5. `Pipeline.run(...)` 后的最终 MLIR
6. `jit_npu.py` 里解析出的 signature/grid/kernel_name/mix_mode
7. Ascend 编译器的 stdout/stderr

## 12. 本次文档的约束

本文是静态流程梳理，不包含真实运行日志。

原因：

- 当前机器不具备 NPU 卡
- 用户明确说明“不必要试着跑”

所以这份流程文档更适合：

- 代码阅读
- 新人 onboarding
- 后续线程恢复上下文

而不是作为“已验证构建手册”。
