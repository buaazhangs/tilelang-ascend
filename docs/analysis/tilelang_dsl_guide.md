# TileLang 装饰器与基础语法速查

## 1. 最重要的几个入口

### `@T.prim_func`

作用：

- 把 Python 函数解析成 TVM `tir.PrimFunc`
- 是 TileLang DSL 的前端入口

例子：

```python
import tilelang.language as T

@T.prim_func
def main(A: T.Tensor((1024,), "float32"), B: T.Tensor((1024,), "float32")):
    ...
```

### `@tilelang.jit(...)`

作用：

- 把“返回 `PrimFunc` 的 Python 工厂函数”包装成 JIT kernel 工厂
- 首次按参数调用时编译，后续复用缓存

例子：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(target="npuir")
def vec_add(N, block_N):
    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
        shape: T.int32,
    ):
        ...
    return main
```

### `tilelang.compile(func, target="npuir")`

作用：

- 直接把一个 `PrimFunc` 编译成可调用 kernel

例子：

```python
func = vec_add(...)
kernel = tilelang.compile(func, target="npuir")
kernel(a, b, c, shape)
```

## 2. `tilelang.language as T` 里最常用的基础语法

## 2.1 Kernel 启动

### `T.Kernel(...)`

作用：

- 定义 kernel 启动域
- 在 NPU 模式下，常把 1D `cid` 再手工映射成逻辑上的 2D/3D 块坐标

NPU 常见写法：

```python
with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
    by = cid // T.ceildiv(N, block_N)
    bx = cid % T.ceildiv(N, block_N)
```

理解要点：

- `is_npu=True` 时源码里要求 block 维度只能有 1 个
- 所以 NPU 上常见模式是“把多维任务摊平成单个 `cid`”

## 2.2 并行与流水

### `T.Parallel(...)`

作用：

- 构造并行循环
- 适合 tile 内元素级并行表达

例子：

```python
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = A_shared[i, j] + B_shared[i, j]
```

### `T.Pipelined(...)`

作用：

- 构造带 pipeline 语义的循环
- 典型用于 K 维分块迭代，把搬运和计算重叠起来

例子：

```python
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[k * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local, initC=(k == 0))
```

## 2.3 Buffer / Tensor 声明

### `T.Tensor(shape, dtype)`

作用：

- 作为 `PrimFunc` 参数类型注解，表示外部输入/输出张量

例子：

```python
A: T.Tensor((M, K), "float16")
```

### 标量参数

可以直接写 `T.int32`、`T.float32` 等。

例子：

```python
shape: T.int32
```

## 2.4 内存分配

源码在 `tilelang/language/allocate.py`。

### 通用分配

- `T.alloc_shared(shape, dtype)`
  - 分配共享/片上 buffer
- `T.alloc_local(shape, dtype)`
  - 分配局部 buffer
- `T.alloc_fragment(shape, dtype)`
  - 分配 fragment buffer
- `T.alloc_var(dtype)`
  - 分配单元素变量 buffer

例子：

```python
A_shared = T.alloc_shared((block_M, block_K), "float16")
C_local = T.alloc_fragment((block_M, block_N), "float32")
```

### Ascend/NPU 专用分配

- `T.alloc_L1`
- `T.alloc_L0A`
- `T.alloc_L0B`
- `T.alloc_L0C`
- `T.alloc_ub`

对应含义大致是：

- `L1`
- `L0A/L0B/L0C`
- `UB`

例子：

```python
A_BUF = T.alloc_L1([block_M, K_L1], dtype)
B_BUF = T.alloc_L1([K_L1, block_N], dtype)
C_BUF = T.alloc_L0C([block_M, block_N], accum_dtype)
```

## 2.5 数据搬运

### `T.copy(src, dst, size=None)`

作用：

- 在 global / shared / ub / fragment 等 buffer 之间搬运数据
- 支持整块 copy，也支持 slice copy

例子 1：整块 copy

```python
T.copy(A[by * block_M, bx * block_N], A_shared)
```

例子 2：带切片

```python
T.copy(A[start : start + tail], A_VEC[0:tail])
```

例子 3：显式指定 size

```python
T.copy(A[bx, i * K_L1], A_BUF, size=[block_M, K_L1])
```

### NPU 特定内存搬运

- `T.load_nd2nz`
- `T.store_nz2nd`
- `T.store_fixpipe`

这些更偏 Ascend 特定数据布局与搬运流程。

例子：

```python
T.load_nd2nz(A[bx, i * K_L1], A_BUF, [block_M, K_L1])
T.store_fixpipe(C_BUF, C[bx, by], size=[block_M, block_N], enable_nz2nd=True)
```

## 2.6 算术与向量操作

源码里大量 NPU 向量/标量操作集中在 `tilelang/language/customize_npuir.py`，并在 `tilelang/language/__init__.py` 里暴露别名。

最常见的一批：

- `T.vadd` / `T.npuir_add`
- `T.vsub`
- `T.vmul`
- `T.vdiv`
- `T.vmax`
- `T.vmin`
- `T.vexp`
- `T.vlog2`
- `T.vexp2`
- `T.vrelu`
- `T.vsqrt`
- `T.vrsqrt`
- `T.vabs`
- `T.vcos`
- `T.vsin`
- `T.vtanh`
- `T.verf`

例子：

```python
T.vadd(A_VEC, B_VEC, C_VEC)
T.vmul(X_VEC, 0.5, Y_VEC)
T.vrelu(IN_VEC, OUT_VEC)
```

## 2.7 线性代数

### `T.gemm(...)`

作用：

- 表达矩阵乘核心计算
- 在 NPU 上通常配合 `L1/L0C` buffer、`load_nd2nz`、`store_fixpipe` 使用

例子：

```python
T.gemm(A_BUF, B_BUF, C_BUF, initC=(k == 0), b_transpose=False, size=[block_M, K_L1, block_N])
```

## 2.8 归约和扫描

### 归约

- `T.reduce`
- `T.reduce_sum`
- `T.reduce_max`
- `T.reduce_min`
- `T.reduce_abssum`
- `T.reduce_absmax`

例子：

```python
T.reduce_sum(inp, out, dim=-1)
```

### 扫描

- `T.cumsum`

例子：

```python
T.cumsum(src, dst, dim=0, reverse=False)
```

## 2.9 形状与索引操作

常见能力：

- `T.reshape`
- `T.view`
- `T.arange`
- `T.concat`
- `T.pad`
- `T.flip`
- `T.gather`
- `T.interleave`
- `T.deinterleave`
- `T.transpose`
- `T.vbrc`

例子：

```python
T.arange(idx_buf)
T.concat(a, b, out)
T.gather(src, indices, out)
```

## 2.10 条件、比较、逻辑

常见能力：

- `T.vcmp`
- `T.vselect`
- `T.vclamp`
- `T.vand`
- `T.vor`
- `T.vxor`
- `T.vnot`
- `T.vshl`
- `T.vshr`
- `T.any_of`
- `T.all_of`

例子：

```python
mask = T.vcmp(A_VEC, B_VEC, CMP_VEC, "gt")
T.vselect(CMP_VEC, A_VEC, B_VEC, OUT_VEC)
```

## 2.11 原子、同步与调试

### 原子

- `T.atomic_add`
- `T.atomic_addx4`

### 同步/管线

- `T.set_flag`
- `T.wait_flag`
- `T.pipe_barrier`
- `T.block_barrier`
- `T.subblock_barrier`
- `T.sync_block_set`
- `T.sync_block_wait`
- `T.rs`

### 调试

- `T.print`

## 3. 最典型的两种编程风格

## 3.1 Elementwise 风格

特点：

- 用 `T.alloc_ub` / `T.alloc_shared` 搬运 tile
- 用 `T.vadd` / `T.vmul` 等完成向量计算
- 最后 copy 回全局内存

例子：

```python
@T.prim_func
def main(
    A: T.Tensor((N,), "float32"),
    B: T.Tensor((N,), "float32"),
    C: T.Tensor((N,), "float32"),
    shape: T.int32,
):
    with T.Kernel(n_num, is_npu=True) as (cid, _):
        A_VEC = T.alloc_ub((block_N,), "float32")
        B_VEC = T.alloc_ub((block_N,), "float32")
        C_VEC = T.alloc_ub((block_N,), "float32")
        start = cid * block_N
        tail = T.min(block_N, shape - start)
        T.copy(A[start:start + tail], A_VEC[0:tail])
        T.copy(B[start:start + tail], B_VEC[0:tail])
        T.vadd(A_VEC, B_VEC, C_VEC)
        T.copy(C_VEC[0:tail], C[start:start + tail])
```

## 3.2 GEMM / Cube 风格

特点：

- 常进入 `with T.Scope("Cube")`
- 用 `L1/L0C` 等专用 buffer
- 使用 `load_nd2nz + gemm + store_fixpipe`

例子：

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    with T.Scope("Cube"):
        bx = cid // n_num * block_M
        by = cid % n_num * block_N
        A_BUF = T.alloc_L1([block_M, K_L1], dtype)
        B_BUF = T.alloc_L1([K_L1, block_N], dtype)
        C_BUF = T.alloc_L0C([block_M, block_N], accum_dtype)

        for i in T.serial(T.ceildiv(K, K_L1)):
            T.load_nd2nz(A[bx, i * K_L1], A_BUF, [block_M, K_L1])
            T.load_nd2nz(B[i * K_L1, by], B_BUF, [K_L1, block_N])
            T.gemm(A_BUF, B_BUF, C_BUF, initC=(i == 0), b_transpose=False,
                   size=[block_M, K_L1, block_N])
            T.store_fixpipe(C_BUF, C[bx, by], size=[block_M, block_N], enable_nz2nd=True)
```

## 4. 快速心智模型

如果要快速记住这个 DSL，可以把它理解成：

- `@T.prim_func`
  - 定义一个可 Lower 的 kernel IR
- `T.Kernel`
  - 定义 NPU/GPU kernel 启动域
- `T.alloc_*`
  - 定义片上 buffer
- `T.copy` / `T.load_nd2nz` / `T.store_fixpipe`
  - 搬运数据
- `T.Parallel` / `T.Pipelined`
  - 描述 tile 内并行和流水
- `T.v*` / `T.gemm` / `T.reduce*`
  - 真正的计算
- `@tilelang.jit` / `tilelang.compile`
  - 把上面的 IR 变成可执行 kernel

## 5. 进一步查阅入口

如果想继续扩展这份速查，优先看：

- `tilelang/language/__init__.py`
- `tilelang/language/kernel.py`
- `tilelang/language/allocate.py`
- `tilelang/language/copy.py`
- `tilelang/language/reduce.py`
- `tilelang/language/customize_npuir.py`
- `docs/Tilelang.language/`
- `testing/npuir/`

后两者尤其适合查“某个 DSL op 怎么写实例”。
