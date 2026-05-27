# TileLang-Ascend 项目总览

## 1. 项目是做什么的

`tilelang-ascend` 是 TileLang 在华为 Ascend NPU 上的一个专用变体。

它的目标是：

- 用 Python 风格 DSL 描述高性能算子
- 基于 TVM/TIR 做前端解析、IR 变换和部分代码生成
- 面向 Ascend 走 `npuir` 编译链，把 TileLang 程序降到 AscendNPU IR / MLIR，再继续编译成可执行 kernel
- 提供 JIT 运行时，把编译后的 kernel 包装成可直接从 Python / PyTorch 调用的对象

从代码结构上看，它不是单纯的“算子例子仓库”，而是一个完整的小型编译器工程，包含：

- Python DSL 前端
- TVM 扩展与 TileLang 自定义 Pass
- NPU 专用 codegen 和运行时封装
- TileLangIR / MLIR pass 基础设施
- 示例、测试、benchmark、安装脚本

## 2. 仓库分层理解

可以把仓库粗分成 4 层：

### A. 用户编程层

用户写的代码通常长这样：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(target="npuir")
def kernel_factory(...):
    @T.prim_func
    def main(A: T.Tensor(...), B: T.Tensor(...), C: T.Tensor(...)):
        with T.Kernel(..., is_npu=True) as (cid, _):
            ...
    return main
```

这一层主要由 `tilelang/language/` 和 `tilelang/jit/` 提供能力。

### B. TileLang / TVM IR 变换层

`@T.prim_func` 会把 Python 函数解析成 TVM `PrimFunc`。之后 `tilelang.engine.lower()` 会调用一串 TileLang 自定义 pass 和 TVM pass，把高层 DSL 语义逐步降级。

这一层的核心在：

- `tilelang/engine/`
- `tilelang/transform/`
- `src/transform/`

### C. NPUIR / MLIR 编译层

当 target 是 `npuir` 时，Lower 后不会直接得到 CUDA/C++ 源码，而是进入 NPUIR 路径：

- C++ 端 `src/target/codegen_npuir*.cc` 生成 NPUIR/MLIR
- Python 端 `tilelang/tladapter/` 对 MLIR 字符串继续串 pass
- `tilelang/jit/jit_npu.py` 调 Ascend 编译器把 `.npuir` 编译成二进制对象和 launcher

### D. 运行时与验证层

JIT 后得到的对象最终能像 Python 函数一样调用；示例、测试、benchmark 则覆盖这条链路。

相关目录：

- `examples/`
- `testing/`
- `unittest/`
- `benchmark/`

## 3. 目录结构梳理

### 根目录

- `.github/`
  - CI、PR 模板、工作流等仓库协作配置。
- `3rdparty/`
  - 第三方依赖源码，重点是 `tvm`，以及 CUDA/ROCm 相关依赖。当前工作区里这部分有未提交变更，需要避免误改。
- `benchmark/`
  - 性能基准脚本，侧重 matmul、block sparse attention 等。
- `docker/`
  - 容器化环境。
- `docs/`
  - 用户文档、开发文档、TileLang language 操作文档。
- `examples/`
  - 面向用户的示例算子，最适合理解“怎么写 TileLang Ascend kernel”。
- `images/`
  - README 和文档配图。
- `maint/`
  - 维护脚本、格式化、开发辅助内容。
- `src/`
  - C++ 核心实现。TileLang 自定义 IR、算子定义、变换 pass、target codegen、runtime 都在这里。
- `testing/`
  - 主要测试目录。包含 Python 层、NPUIR 层、MLIR pass、autotune 等测试。
- `tilelang/`
  - Python 包主目录。对用户最重要，负责 DSL、JIT、cache、autotune、transform 封装等。
- `tilelangir/`
  - TileLangIR / MLIR 子项目，提供 native pass pipeline 和相关工具。
- `unittest/`
  - 较早期/更直接的 NPUIR 用例与 mlir 产物比对材料。

### `tilelang/` 目录

- `language/`
  - DSL 入口。定义 `T.prim_func`、`T.Kernel`、`T.Parallel`、`T.Pipelined`、内存分配、拷贝、归约和大量 NPUIR intrinsic。
- `jit/`
  - JIT 编译入口。`tilelang.jit` 和 `tilelang.compile` 在这里。
- `engine/`
  - lower 主流程。连接 pass pipeline、codegen 和 artifact 组织。
- `transform/`
  - Python 侧 pass 包装器，对应 C++ `_ffi_api` 暴露的 pass。
- `tladapter/`
  - TileLangIR/MLIR 适配层，把 NPUIR 字符串送进 MLIR pass pipeline。
- `cache/`
  - 编译缓存。
- `autotuner/`
  - 自动调优基础设施。
- `primitives/`
  - 更底层原语封装。
- `layout/`
  - 布局相关抽象，如 `Layout`、`Fragment`。
- `profiler/`
  - benchmark/profiler 支持。
- `utils/`
  - 目标设备、路径、NPU 工具、编译辅助函数等。
- `contrib/`
  - 外部编译器桥接，如 nvcc、hipcc 等。
- `intrinsics/`, `math/`, `common/`, `tools/`
  - 语言辅助、通用工具、数学封装等。

### `src/` 目录

- `ir.cc`
  - TileLang 自定义 IR 注册/绑定入口之一。
- `op/`
  - TileLang 自定义算子/intrinsic 定义，如 `copy`、`gemm`、`parallel`、Ascend 相关 op。
- `transform/`
  - 最关键的一层。实现 TileLang 自定义 lowering/optimization pass。
  - NPU 相关重点 pass 包括：
    - `npu_loop_vectorize.cc`
    - `legalize_npuir_bf16.cc`
    - `lower_npuir_block.cc`
    - `plan_update_buffer_allocation_location.cc`
- `target/`
  - 多后端 codegen/runtime。
  - 对 Ascend/NPUIR 重点是：
    - `codegen_npuir.cc`
    - `codegen_npuir_api.cc`
    - `codegen_npuir_dev.cc`
    - `rt_mod_npuir.cc`
- `runtime/`
  - 运行时支持。
- `layout/`
  - 布局和 swizzle 相关实现。
- `tl_templates/`
  - 各 target 的模板头文件和内核模板。

### `tilelangir/` 目录

- `include/`, `lib/`
  - TileLangIR/MLIR pass 定义与实现。
- `python/`
  - Python 绑定。
- `tools/tilelangir-opt`
  - 类似 MLIR opt 的 pass 调试工具。

### `examples/` 目录

- `elementwise/`
  - 向量加、广播、atomic add 等最基础例子。
- `gemm/`, `gemv/`
  - 线性代数核心例子。
- `norm/`
  - RMSNorm 等归一化。
- `mixcv/`
  - Cube/Vector 混合算子。
- `torch_tl_ops/`
  - 一个更接近真实工程集成的示例，展示如何把 TileLang kernel 打包成 torch 扩展风格接口。
- `flash_attn_npuir.py`, `sparse_mla_fwd.py`
  - 较复杂的大模型算子样例。

### `testing/` 与 `unittest/`

- `testing/python/`
  - 通用 Python 层能力测试，不只覆盖 NPU。
- `testing/npuir/`
  - Ascend / NPUIR 核心测试主阵地，按 op 类别分文件夹。
- `testing/mlir/`
  - TileLangIR / MLIR pass 测试。
- `unittest/npuir/`
  - 一批偏底层的 NPUIR case 和对应 `.mlir` 期望文件，适合理解产物长什么样。

## 4. 代码结构重点

### 入口 API

- `tilelang.__init__`
  - 暴露 `jit`、`compile`、`transform`、`language`、`lower` 等主 API。
- `tilelang.language.__init__`
  - 把 DSL 能力统一暴露到 `import tilelang.language as T`。
- `tilelang.jit.__init__`
  - 提供 `tilelang.compile(...)` 与 `@tilelang.jit(...)`。

### 编译主流程骨架

- `tilelang.jit.__init__.compile`
  - 普通 target 走通用 JITKernel；`target="npuir"` 特判走 `compiler_npu()`
- `tilelang.engine.lower`
  - 通用 Lower 主入口
- `tilelang.engine.phase`
  - 定义 Lower 阶段的 pass 顺序
- `tilelang.jit.jit_npu`
  - NPU 专用 JIT 编译、MLIR 解析、wrapper 生成、launcher 构建

## 5. 读代码建议顺序

如果后续还要继续深入这个仓库，推荐按下面顺序读：

1. `README.md`
2. `examples/elementwise/vec_add_1d.py`
3. `examples/gemm/matmul.py`
4. `tilelang/language/__init__.py`
5. `tilelang/language/kernel.py`
6. `tilelang/language/allocate.py`
7. `tilelang/language/copy.py`
8. `tilelang/language/customize_npuir.py`
9. `tilelang/jit/__init__.py`
10. `tilelang/engine/lower.py`
11. `tilelang/engine/phase.py`
12. `tilelang/jit/jit_npu.py`
13. `src/target/codegen_npuir*.cc`
14. `testing/npuir/*`

## 6. 当前理解边界

这次分析没有实际跑编译或执行，符合“机器上没有卡，不必尝试运行”的约束。

因此以下内容是基于源码和文档的静态理解：

- 目录职责
- 编译/运行流程
- DSL 语义和 API 能力
- NPUIR 路径的总体架构

但以下内容没有在本机做动态验证：

- 实际 Ascend 编译器版本兼容性
- JIT 运行时细节是否受环境变量影响
- 某些文档示例在当前分支是否仍完全可运行
