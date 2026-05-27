# Sparse Mix Stride Align 问题简要总结

## 1. 基本原理

MLIR memref 的 `shape` 表示逻辑张量形状，`stride` 表示逻辑索引变化时物理地址跨过多少个元素。对 `f32` 来说：

```text
addr(i, j) = base + (i * stride[0] + j * stride[1]) * 4B
```

例如 `shape=[32,32], stride=[32,1]` 表示标准连续 row-major 布局：

```text
input[i,31]  -> base + (i * 32 + 31) * 4B
input[i+1,0] -> base + (i * 32 + 32) * 4B
```

二者物理地址只差 4B，因此整块输入在物理上连续。

Stride align 处理的是 UB/L1 等硬件 buffer 的对齐问题。UB 通常需要 32B 对齐。若编译器不能证明某个 memref 的访问布局天然满足连续和对齐要求，就可能保守地扩轴加 padding。例如：

```text
logical output:  shape=[32,1], stride=[1,1]
aligned output:  shape=[32,1], stride=[8,1]  // f32 下 8 个元素 = 32B
```

这时逻辑上仍是 32 个输出值，但物理上相邻两行间隔 32B：

```text
out0 pad pad pad pad pad pad pad
out1 pad pad pad pad pad pad pad
out2 pad pad pad pad pad pad pad
```

这种 strided 输出会让后端更难生成连续向量 reduce/store 路径，容易退化成多段或多条指令。当前 sparse mix 中 `vreduce max/sum` 在流水图上拆成多次 `MOVEMASK + VCMAX`，核心现象就是 reduce 输出从连续 layout 变成了 `stride=[8,1]`。

## 2. Sparse Dev 与 Sparse Mix 的区别

Sparse dev 的 reduce 形态类似：

```text
input  shape=[16,32], stride=[32,1]
output shape=[16,1],  stride=[1,1]
```

输出是连续布局，后端可以按连续向量结果处理。

Sparse mix 的 reduce 形态变成：

```text
input  shape=[32,32], stride=[32,1]
output shape=[32,1],  stride=[8,1]
```

输入本身仍然连续，问题主要在输出。`stride=[8,1]` 说明每个 reduce 结果按 32B pitch 存放，逻辑相邻输出不再物理连续，因此后续 lowering 更容易拆成多条 vector 指令。

目前看到的根因不是 sparse 算法本身，也不是单纯因为动态 shape。Dev 路径里也有动态 `reinterpret_cast`，但 NPUIR 的 `canonicalize-ext` 会把 `%c2048/%c32/%c1` 这类常量 stride 折回 memref 类型中，例如从：

```text
strided<[?, ?, ?, ?]>
```

折成：

```text
strided<[?, 2048, 32, 1]>
```

这样后续 `MarkStrideAlign` 能看出尾轴连续，不会错误插入 `hivm.stride_align`。

Sparse mix 路径中，在进入 `MarkStrideAlign/EnableStrideAlign` 前，部分 `reinterpret_cast/subview` 的静态 stride 信息仍停留在 SSA 常量参数里，但 memref 类型里还是 `?`。因此 stride-align pass 无法证明尾轴连续，保守加了 stride align，最终把 UB 临时 buffer 改成带 padding 的 layout。

## 3. 与 FA Mix 的区别

FA mix 同样走 mix 分支，也有 workspace、scope、multi-buffer 等机制，但它没有出现这个 reduce 输出被扩成 `stride=[8,1]` 的问题。

主要区别是：FA mix 在进入 stride-align 相关 pass 时，关键 UB reduce buffer 的布局信息仍然是静态可见或可被已有 canonicalize 规则整理出来的；而 sparse mix 由于 top-k/gather、动态 KV index、workspace/subview/reinterpret_cast 链路更复杂，部分 stride 信息在类型层面丢成了 `?`。

因此 FA mix 没有触发错误的 `hivm.stride_align` 扩轴；sparse mix 触发了，导致 reduce 输出 layout 改变，并进一步影响 vector 指令生成。

## 4. 当前判断与后续方向

当前问题更像是 lowering pipeline 的类型规范化时机问题：mix 路径在 stride-align 前缺少一次足够强的 canonicalize，使 `reinterpret_cast/subview` 中的常量 stride 回写到 memref type。

后续可优先验证两类修复：

1. 在 mix lowering 中，于 `hivm-mark-stride-align` 前补充或提前运行 `canonicalize-ext`，确认 `hivm.stride_align` 数量是否消失。
2. 增强 `MarkStrideAlign`/相关 layout 分析，让它不要只依赖 memref type 中的静态 stride，也能识别 `reinterpret_cast` mixed stride 参数里的 SSA 常量。

短期判断标准：如果 sparse mix 在 stride-align 后不再出现 `memref<32x1xf32, strided<[8,1]>, ub>` 这类 reduce 输出，流水里的 reduce 指令拆分现象应该会明显缓解。
