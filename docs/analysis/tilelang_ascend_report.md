# TileLang-Ascend 项目分析汇报

## 1. 项目概述

`tilelang-ascend` 是 TileLang 面向华为 Ascend NPU 的专用实现。它不是一个单纯的算子样例仓库，而是一套较完整的编译与运行体系，目标是让开发者用 Python 风格的 DSL 编写高性能 NPU kernel，再通过 TVM、TileLang 自定义 Pass、NPUIR/MLIR 和 Ascend 编译器，把这些算子编译成可执行的 NPU 内核，并以 JIT 方式从 Python / PyTorch 直接调用。

从定位上看，这个项目要解决的是两件事：

- 提供一套相对高层、可读性较好的 TileLang 编程接口，降低 Ascend 自定义算子开发门槛
- 保留足够低层的控制能力，使开发者仍然可以表达 tile 切分、片上存储、流水、向量化、Cube/Vector 协同等高性能算子所需的关键细节

因此它本质上是“面向 Ascend 的 TileLang 编译栈”，不是只放一些 `examples` 的应用层项目。

## 2. 整体架构理解

从源码静态阅读看，整个仓库大致可以分成四层。

### 第一层：用户编程层

开发者主要通过下面这种模式写算子：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(target="npuir")
def kernel_factory(...):
    @T.prim_func
    def main(...):
        with T.Kernel(..., is_npu=True) as (cid, _):
            ...
    return main
```

这一层的重点是：

- `@T.prim_func`：把 Python 函数变成 TVM `PrimFunc`
- `@tilelang.jit` 或 `tilelang.compile(...)`：触发编译，返回可调用 kernel
- `T.Kernel`、`T.alloc_*`、`T.copy`、`T.v*`、`T.gemm` 等 DSL：描述 kernel 的计算和访存逻辑

### 第二层：IR Lower / Pass 层

用户写出的 TileLang DSL 最终会被解析为 TVM TIR / TileLang 自定义 intrinsic，然后进入一系列 Lower 和优化 Pass。

这一层的作用是：

- 把前端 DSL 语义合法化
- 进行 NPU 相关的向量化、buffer 分配位置规划、block 降级等处理
- 为后续 NPUIR codegen 准备合适的中间表示

### 第三层：NPUIR / MLIR 编译层

当 target 是 `npuir` 时，项目不会像 CUDA 后端那样直接输出普通 device module，而是先生成 Ascend 对应的 NPUIR/MLIR，然后继续做一轮 MLIR pass 适配，最后再调用 Ascend 编译器产出目标文件。

这说明 `tilelang-ascend` 的后端链路不是单一 codegen，而是：

- TileLang/TVM IR
- NPUIR/MLIR
- TileLangIR pass pipeline
- Ascend 编译器

### 第四层：运行时与 JIT 层

编译完成后，项目会生成 Python launcher / runtime stub，把目标内核包装成 Python 可直接调用的对象。

最终用户使用体验是：

- 写一个 TileLang kernel 工厂
- JIT 编译
- 像普通 Python 函数一样调用它

## 3. 目录结构梳理

### 3.1 根目录主要内容

- `tilelang/`
  - Python 主包，是理解项目的第一入口。包含 DSL、JIT、Lower 封装、transform 包装、NPU 编译适配等核心逻辑。
- `src/`
  - C++ 核心实现。包含自定义 IR、算子定义、transform Pass、target codegen 和 runtime。
- `tilelangir/`
  - TileLangIR / MLIR 子项目，主要提供 native pass pipeline 和相关工具。
- `examples/`
  - 面向用户的示例，适合用来理解基本编程方式，包含 elementwise、gemm、norm、flash attention 等。
- `testing/`
  - 主测试目录，覆盖 Python 层、NPUIR 层、MLIR pass、autotune 等。
- `unittest/`
  - 一批偏底层的 NPUIR 用例与 `.mlir` 产物，适合对照生成结果。
- `docs/`
  - 用户文档、开发文档和 TileLang language 语法文档。
- `benchmark/`
  - 性能评测脚本。
- `3rdparty/`
  - 第三方依赖源码，重点包括 `tvm`。

### 3.2 `tilelang/` 目录职责

- `language/`
  - TileLang DSL 的主体。这里定义了 `T.prim_func`、`T.Kernel`、`T.Parallel`、`T.Pipelined`、buffer 分配、copy、reduce 和 NPU intrinsic。
- `jit/`
  - JIT 入口，负责 `tilelang.jit`、`tilelang.compile` 和 NPU 专用编译流程。
- `engine/`
  - Lower 主流程，负责组织 pass 和 codegen。
- `transform/`
  - Python 侧 pass 包装器，对应底层 C++ Pass。
- `tladapter/`
  - TileLangIR/MLIR 适配层，用于对生成的 MLIR 字符串继续跑 pass pipeline。
- `cache/`
  - 编译缓存。
- `autotuner/`
  - 自动调优能力。
- `layout/`、`primitives/`、`profiler/`、`utils/`
  - 分别承载布局、低层原语、性能分析和通用工具能力。

### 3.3 `src/` 目录职责

- `op/`
  - TileLang 自定义算子和 intrinsic 定义。
- `transform/`
  - 项目最关键的后端处理层，包含大量 Lower/Optimize Pass。
- `target/`
  - 多后端 codegen，其中 Ascend/NPUIR 对应 `codegen_npuir*.cc`、`rt_mod_npuir.cc` 等。
- `runtime/`
  - 运行时支持。
- `layout/`
  - 布局和 swizzle 相关逻辑。
- `tl_templates/`
  - 各 target 的模板代码。

### 3.4 `examples/`、`testing/`、`docs/` 的价值

- `examples/`
  - 适合看“怎么写”
- `testing/npuir/`
  - 适合看“支持哪些能力”
- `docs/Tilelang.language/`
  - 适合查“某个语法点怎么用”

如果是新人接手，这三个目录的价值甚至不低于源码本身。

## 4. 运行与编译流程梳理

## 4.1 前端阶段

用户写的 TileLang kernel 一般先通过 `@T.prim_func` 解析成 TVM `PrimFunc`。

这里的本质是：

- TileLang 并没有完全自建 parser
- 它建立在 TVM Script / TIR parser 之上
- 再通过 `tilelang.language` 提供额外 DSL 结构和 intrinsic

也就是说，TileLang 前端可以看作“TVM TIR 的一层面向 kernel 编程的扩展 DSL”。

## 4.2 JIT 入口分流

编译入口主要有两个：

- `tilelang.compile(func, target="npuir")`
- `@tilelang.jit(target="npuir")`

两者最终都会进入 NPU 专用逻辑。

从代码上看：

- 普通 target 走通用 `JITKernel`
- `target="npuir"` 时走 `compiler_npu().compile(...)`

`@tilelang.jit` 的本质不是直接装饰一个现成 kernel，而是装饰“返回 PrimFunc 的工厂函数”，首次按参数调用时生成并编译 kernel，后续复用缓存。

## 4.3 Lower 阶段

Lower 主入口在 `tilelang/engine/lower.py`。

对于 NPU 路径，主要流程是：

1. 把 `PrimFunc` 整理成 `IRModule`
2. 绑定 target
3. 做基础简化
4. 跑 NPU 相关 Pass
5. 进入 NPUIR codegen

在 `tilelang/engine/phase.py` 中，NPU 相关的重点 Pass 包括：

- `NpuLoopVectorize`
- `LegalizeNpuirBF16`
- `PlanAndUpdateBufferAllocationLocation`
- `LowerNpuirBlock`

这些 Pass 的职责大致可以理解为：

- 把高层并行/向量语义降成更接近 NPU 后端的 IR
- 处理 BF16 等目标限制
- 调整 buffer 分配和复用位置
- 清理高层 block 结构，方便后端 codegen

## 4.4 NPUIR / MLIR 生成阶段

当 target 为 `npuir` 时，Lower 不会直接得到普通意义上的 device module，而是进入 NPU codegen，生成 NPUIR / MLIR 文本。

这里根据环境变量 `TILELANG_ASCEND_MODE` 还会区分：

- Expert 模式
- Developer 模式

从仓库已有文档和源码逻辑看：

- Expert 更偏直接的 NPU API / codegen 路径
- Developer 更强调编程一致性和自动化优化能力

## 4.5 TileLangIR pass pipeline

生成初始 MLIR 后，项目不会立即停止，而是还会用 `tilelang/tladapter` 再跑一轮 MLIR pass pipeline，例如：

- canonicalize
- adapt_triton_kernel

这一步说明项目内部其实已经把“TVM Lower”和“MLIR 适配”串成了一条复合编译链。

## 4.6 Ascend 编译器与 launcher 生成

NPU 专用编译逻辑集中在 `tilelang/jit/jit_npu.py`。

这一层会做几件关键事情：

- 提取参数类型、shape、输出索引
- 处理动态 shape 符号变量
- 解析 grid、kernel name、signature、mix mode 等 metadata
- 调用外部 Ascend 编译器生成目标对象
- 再生成 Python launcher / stub

最终返回 `JitKernel_NPU`，用户调用它就像调用 Python 函数一样。

## 4.7 一条完整链路的简化表达

可以把整个运行链简化为下面这条主线：

**用户 DSL**
-> `@T.prim_func`
-> `PrimFunc`
-> `tilelang.compile / @tilelang.jit`
-> `lower`
-> TileLang/TVM NPU Pass
-> NPUIR/MLIR codegen
-> TileLangIR pass pipeline
-> Ascend 编译器
-> Python launcher
-> 可调用 JIT kernel

## 5. 代码结构重点理解

## 5.1 最重要的入口文件

如果只挑最关键的一批文件，建议优先理解：

- `tilelang/__init__.py`
- `tilelang/language/__init__.py`
- `tilelang/language/kernel.py`
- `tilelang/language/allocate.py`
- `tilelang/language/copy.py`
- `tilelang/language/customize_npuir.py`
- `tilelang/jit/__init__.py`
- `tilelang/engine/lower.py`
- `tilelang/engine/phase.py`
- `tilelang/jit/jit_npu.py`

## 5.2 推荐阅读路径

建议的阅读顺序是：

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
13. `testing/npuir/`

这条路径能比较快建立“前端 DSL -> Lower -> NPUIR -> JIT” 的完整认知。

## 6. TileLang 装饰器与基础语法总结

## 6.1 最重要的三个入口

### `@T.prim_func`

作用：

- 把 Python 函数解析成 TVM `PrimFunc`
- 是 TileLang DSL 的前端入口

### `@tilelang.jit(...)`

作用：

- 把“返回 PrimFunc 的工厂函数”包装成 JIT kernel 工厂
- 首次按参数调用时编译，后续缓存复用

### `tilelang.compile(func, target="npuir")`

作用：

- 直接把一个 `PrimFunc` 编译成可调用 kernel

## 6.2 Kernel 启动与并行语义

### `T.Kernel(...)`

作用：

- 定义 kernel 启动域

在 NPU 模式下，源码限制 `is_npu=True` 时 block 维度必须为 1，因此常见写法是：

```python
with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
    by = cid // T.ceildiv(N, block_N)
    bx = cid % T.ceildiv(N, block_N)
```

也就是说，NPU 编程里很常见的模式是：

- 逻辑上二维/三维任务
- 实际上先摊平成一维 `cid`
- 再在 kernel 内手动恢复坐标

### `T.Parallel(...)`

作用：

- 描述 tile 内的元素级并行计算

### `T.Pipelined(...)`

作用：

- 描述带 pipeline 语义的循环
- 常用于 K 维分块计算，把搬运与计算重叠

## 6.3 数据对象与内存分配

### 参数声明

- `T.Tensor(shape, dtype)`
  - 用作张量参数注解
- `T.int32`、`T.float32`
  - 用作标量参数类型

### 常见内存分配

- `T.alloc_shared`
- `T.alloc_local`
- `T.alloc_fragment`
- `T.alloc_var`

### Ascend/NPU 专用分配

- `T.alloc_L1`
- `T.alloc_L0A`
- `T.alloc_L0B`
- `T.alloc_L0C`
- `T.alloc_ub`

这些接口是写 NPU 高性能算子的核心，因为它们对应的是片上存储层级，而不是普通高层张量抽象。

## 6.4 数据搬运语法

### `T.copy`

作用：

- 在 global、shared、ub、fragment 等不同 buffer 之间搬运数据
- 支持整块搬运，也支持 slice 搬运

### NPU 特色搬运

- `T.load_nd2nz`
- `T.store_nz2nd`
- `T.store_fixpipe`

它们更多体现 Ascend 特有的数据布局和访存模式。

## 6.5 计算语法

### 向量/逐元素运算

最常见的一类是 `T.v*` 系列，例如：

- `T.vadd`
- `T.vsub`
- `T.vmul`
- `T.vdiv`
- `T.vmax`
- `T.vmin`
- `T.vrelu`
- `T.vexp`
- `T.vsqrt`
- `T.vcmp`
- `T.vselect`

这类接口构成了 elementwise / vector kernel 的主体。

### 线性代数

最重要的是 `T.gemm`。

在 NPU 上通常会配合：

- `L1/L0C` buffer
- `load_nd2nz`
- `store_fixpipe`

形成 Cube 风格的矩阵乘实现。

### 归约与扫描

常见包括：

- `T.reduce`
- `T.reduce_sum`
- `T.reduce_max`
- `T.reduce_min`
- `T.cumsum`

### 形状、索引、逻辑与同步

还支持大量辅助能力：

- 形状/索引：`reshape`、`view`、`arange`、`concat`、`pad`、`flip`、`gather`、`transpose`
- 条件/逻辑：`vcmp`、`vselect`、`vand`、`vor`、`vnot`
- 原子：`atomic_add`、`atomic_addx4`
- 同步：`set_flag`、`wait_flag`、`pipe_barrier`、`sync_block_set`、`sync_block_wait`
- 调试：`T.print`

## 6.6 两种最典型的编程风格

### Elementwise 风格

特征是：

- 把数据切到 UB / shared
- 用 `T.vadd`、`T.vmul` 等完成向量运算
- 再写回全局内存

典型代表是 `examples/elementwise/vec_add_1d.py`。

### GEMM / Cube 风格

特征是：

- 进入 `T.Scope("Cube")`
- 使用 `L1/L0C` 等 NPU 专用 buffer
- 通过 `load_nd2nz + gemm + store_fixpipe` 完成矩阵乘

典型代表是 `examples/gemm/matmul.py`。

## 7. 我的整体判断

从仓库结构和实现方式看，`tilelang-ascend` 的核心价值主要体现在三点：

### 第一，编程模型统一

它试图让开发者用尽量一致的 TileLang 方式写 kernel，而不是直接写过于底层的 Ascend 专用代码。

### 第二，后端链路完整

它不是“前端 DSL + 几个模板”这么简单，而是包含：

- DSL
- TVM/TIR IR
- 自定义 Pass
- NPUIR/MLIR
- 外部编译器
- Python JIT launcher

是一条比较完整的编译执行链。

### 第三，对 Ascend 特性做了明显适配

从源码中可以清楚看到，项目并不是简单复用通用 TileLang，而是针对 Ascend 增加了：

- NPU 专用 codegen
- NPU 向量化 Pass
- NPU block lowering
- BF16 legalize
- 专用内存层次分配接口
- `load_nd2nz` / `store_fixpipe` 等特色操作

所以它已经不是“TileLang 的一个配置项”，而是一个实质上带 Ascend 后端特征的分支实现。

## 8. 当前分析结论的边界

这次工作是静态分析，没有做真实运行验证，原因有两个：

- 当前机器没有 NPU 卡
- 任务本身明确要求“不必要试着跑”

因此本汇报中的结论可以认为对以下内容是可靠的：

- 项目定位
- 目录职责
- 编译主链路
- DSL 入口与基础能力
- 关键模块之间的关系

但不应把它视为已经动态验证过的内容：

- 当前分支在本机环境的可编译性
- 不同 Ascend 工具链版本的兼容情况
- 某些示例在当前环境下的真实运行状态

## 9. 已落地的配套材料

为了方便后续继续使用，我已经把相关内容落在工作区里：

- `docs/analysis/project_overview.md`
- `docs/analysis/runtime_flow.md`
- `docs/analysis/tilelang_dsl_guide.md`
- `docs/analysis/tilelang_ascend_report.md`
- `workspace_memory.md`

其中：

- 前三份是分主题展开版
- `tilelang_ascend_report.md` 是本次汇报版总文档
- `workspace_memory.md` 是给当前工作区后续线程恢复上下文用的记忆文件
