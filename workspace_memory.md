# Workspace Memory: tilelang-ascend

这份文件用于给当前工作区的后续线程快速恢复上下文。

## 仓库定位

- 仓库名：`tilelang-ascend`
- 定位：TileLang 在华为 Ascend NPU 上的专用变体
- 核心路线：Python DSL -> TVM PrimFunc / IRModule -> TileLang/TVM Pass -> NPUIR/MLIR -> Ascend 编译器 -> Python launcher / JIT kernel

## 重要入口

- Python 包入口：`tilelang/__init__.py`
- DSL 入口：`tilelang/language/__init__.py`
- `@T.prim_func`：`tilelang/language/tir/entry.py`
- JIT/compile 入口：`tilelang/jit/__init__.py`
- 通用 Lower 入口：`tilelang/engine/lower.py`
- Pass 顺序：`tilelang/engine/phase.py`
- NPU JIT 主逻辑：`tilelang/jit/jit_npu.py`

## 关键目录速记

- `tilelang/`
  - Python 主包；最重要。
- `src/`
  - C++ 核心实现，包括 op、transform、target codegen。
- `tilelangir/`
  - MLIR/TileLangIR 子项目。
- `examples/`
  - 推荐入门读例子。
- `testing/npuir/`
  - NPUIR 能力测试全集，按 op 分类。
- `docs/Tilelang.language/`
  - 已有 DSL 分项文档。

## NPU 路径关键认知

- `target="npuir"` 时，`tilelang.compile` 会特判进入 `compiler_npu().compile(...)`
- NPU 路径不会像 CUDA 一样直接产出普通 device module，而是先产出 NPUIR/MLIR
- `tilelang/engine/lower.py` 中：
  - 先做 NPU 相关 pass
  - 再走 `device_codegen(...)`
  - 然后通过 `tilelang.tladapter.utils.Pipeline` 跑 MLIR pass
- `jit_npu.py` 负责：
  - 动态 shape 符号提升
  - grid/signature/kernel metadata 解析
  - 调 Ascend 编译器
  - 生成 launcher stub
  - 返回 `JitKernel_NPU`

## TileLang DSL 速记

- `@T.prim_func`
  - 把 Python 函数解析成 TVM `PrimFunc`
- `@tilelang.jit`
  - 把“返回 PrimFunc 的工厂函数”变成 JIT kernel 工厂
- `tilelang.compile`
  - 直接编译 PrimFunc
- `T.Kernel(..., is_npu=True)`
  - NPU kernel 启动域；常见做法是单维 `cid` 映射成逻辑二维 block
- `T.Parallel`
  - 元素级并行循环
- `T.Pipelined`
  - K 维流水循环
- `T.alloc_*`
  - 片上 buffer 分配
- `T.copy`
  - 数据搬运
- `T.v*`
  - NPU 向量算子
- `T.gemm`
  - GEMM 核心

## 推荐阅读顺序

1. `README.md`
2. `examples/elementwise/vec_add_1d.py`
3. `examples/gemm/matmul.py`
4. `tilelang/language/__init__.py`
5. `tilelang/jit/__init__.py`
6. `tilelang/engine/lower.py`
7. `tilelang/engine/phase.py`
8. `tilelang/jit/jit_npu.py`
9. `testing/npuir/`

## 本工作区已新增的分析文档

- `docs/analysis/project_overview.md`
- `docs/analysis/runtime_flow.md`
- `docs/analysis/tilelang_dsl_guide.md`

## 注意事项

- 这次分析是静态阅读，没有实际运行编译或测试
- 原因是当前机器无 NPU，且任务明确要求“不必要试着跑”
- 工作区里当前有未提交改动：
  - `3rdparty/composable_kernel`
  - `3rdparty/tvm`
  - `examples/flash_attn_npuir_dev.py` 被删除
- 后续线程如果要继续改代码，避免误动这些已有变更
