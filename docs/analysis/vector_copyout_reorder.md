# Vector Copy-Out 指令重排方法

本文记录当前在 `npuir-mix` / `v4-mix` sparse attention 调优中使用的 Vector 指令重排原则。目标是给后续 agent 或 pass 开发提供固定上下文：当 Vector scope 里既要把数据写到 workspace 给 Cube 使用，又要更新只在 Vector 内部使用的 online 状态时，应优先生成并搬出跨核数据，把其余状态更新尽量排到 copy 之后，从而让 MTE3 搬出与后续 Vector 指令重叠。

## 核心原则

在一个 Vector scope 内，先识别“必须 copy 到 workspace 的数据”。对这个 copy-out 源值做 backward slice，只保留生成该源值真正需要的依赖在 copy 之前；与 copy-out 源值无关、只服务于后续 Vector 状态更新的计算，尽量排到 copy 之后。

简化规则如下：

```text
copy-out source 的依赖链
    -> vcast / local layout convert
    -> T.copy(local -> workspace)
其余只更新 Vector 内部状态的操作
    -> 放到 copy 之后
```

这里的 copy-out 通常是 Vector 产出的概率矩阵 `P`，后续 Cube 用它做 `P @ V`。因此 `P` 一旦可用，就应尽早通过 MTE3 写 workspace。

## 为什么有效

在混编流水里，`T.copy(local, workspace)` 对应的是 Vector 侧向 GM/workspace 的搬出。若 copy 之后还有足够多的 Vector 指令，硬件流水中 MTE3 搬出有机会和这些 Vector 指令重叠。

如果先做完所有状态更新再 copy，MTE3 会更晚启动，Vector 后面也没有可重叠的计算，流水更容易出现空泡。

## FA Mix 中的模式

`FA_mix.py` 的 softmax 阶段可以看成这种模式的参考。它先生成要给 Cube 的概率 `P`，并把 `scores_cast` 写到 `Workspace2`：

```python
T.vexp(scores_ub, scores_ub)
T.vcast(scores_ub, scores_cast, round_mode="rint")
T.copy(scores_cast, Workspace2[0, vid * block_m // 2, 0], size=[block_m // 2, block_n])
```

copy 之后再更新 online softmax 状态：

```python
T.vsub(acc_m, new_max, tmp1)
T.vexp(tmp1, correction)
T.vmul(acc_l, correction, acc_l)
T.vadd(acc_l, local_sum, acc_l)
T.vbrc(value_zero, tmp1)
T.vadd(tmp1, new_max, acc_m)
```

这些状态更新不影响已经写出的 `P`，但会被后续输出累加和下一轮 online softmax 使用。因此它们是适合放在 copy 后面的 Vector 尾部计算。

注意：现有 FA 写法中 `reduce_sum(scores_ub, local_sum)` 仍在 copy 前。按更激进的依赖切分，它也不是 `scores_cast` copy-out 的必要依赖，理论上可以继续评估是否后移。但 pass 化时应先选择保守规则，确保别跨过会改写 `scores_ub` 或影响 `local_sum` 语义的操作。

## Sparse Mix 中的模式

Sparse attention 的 V2 阶段同样会生成概率 `P`，写到 `workspace_prob` 给后续 Cube 阶段做 `P @ V`。

`scores_cast` 的必要依赖是：

```python
T.copy(workspace_score[0, vid * block_heads_half, 0], scores_ub, ...)
T.copy(scores_max, scores_max_prev)
T.vmul(scores_ub, scale, scores_ub)
T.reduce_max(scores_ub, scores_max, dim=1)
for i, j in T.Parallel(block_heads_half, block_top_k):
    scores_ub[i, j] = T.exp(scores_ub[i, j] - scores_max[i, 0])
for i, j in T.Parallel(block_heads_half, block_top_k):
    scores_ub[i, j] *= mask_ub[0, j]
T.vcast(scores_ub, scores_cast, round_mode="rint")
```

这里有两个容易漏掉的依赖：

- `scores_cast` 依赖 `scores_max`，所以 `reduce_max` 必须在 `exp(scores_ub - scores_max)` 前。
- `scores_max_prev` 必须在 `reduce_max` 覆盖 `scores_max` 前保存，否则后面算 `scores_scale = exp(old_max - new_max)` 会丢旧值。

copy-out 之后可以做：

```python
T.copy(scores_cast, workspace_prob[0, vid * block_heads_half, 0], ...)
T.reduce_sum(scores_ub, scores_sum, dim=1)
for i in T.Parallel(block_heads_half):
    scores_scale[i, 0] = T.exp(scores_max_prev[i, 0] - scores_max[i, 0])
for i in T.Parallel(block_heads_half):
    sum_exp[i, 0] = sum_exp[i, 0] * scores_scale[i, 0] + scores_sum[i, 0]
```

`scores_scale` 类似 FA 里的 `correction`，只用于修正历史分母和历史输出累加，不参与生成当前要搬出的 `P`，因此可以放到 `workspace_prob` copy 之后。

## Pass 化时的建议

未来如果做成编译器 pass，建议先限定在单个 Vector `scope.scope` 内，不跨 `scope`、不跨 `scf.for` 边界、不跨同步原语做全局调度。

推荐流程：

1. 找到 Vector scope 内的 local-to-workspace copy。
2. 对 copy 的 source 做 backward slice，收集生成 copy-out 源值所需的 op。
3. backward slice 需要透过 view-like op，例如 `memref.subview`、`memref.reinterpret_cast`、`memref.collapse_shape`、`memref.expand_shape`。
4. 如果某个值会被 pre-copy 依赖覆盖，而 post-copy 还需要旧值，则保留或提前插入旧值保存操作，例如 `scores_max -> scores_max_prev`。
5. 只移动可证明安全的 Vector 计算，例如 elementwise、reduce、broadcast、本地 memref 状态更新。
6. 不移动 workspace/GM copy、Cube op、同步 op、带未知副作用的 op。
7. 移动时必须保持 memref alias 安全：不能把读操作移到写它的 op 之后，也不能把写操作移过可能读同一 alias 的 op。

保守地说，第一版 pass 可以只识别固定模式：

```text
vcast(P) -> copy(P, workspace) -> reduce_sum(P) / correction / denominator update
```

先覆盖 FA/sparse attention 的 online softmax 场景，再逐步泛化。

## 验证标准

每次重排后至少检查三件事：

- TileLang/AscendNPU IR 能编译通过。
- 精度与原 kernel 对齐。
- 流水图中 copy-out 对应的 MTE3 搬出是否提前，且 copy 后是否还有 Vector 指令可以与其重叠。

如果流水图中 MTE3 仍然排在 Vector 状态更新之后，说明重排没有真正触发目标效果；如果 copy 后没有足够 Vector 尾部指令，收益也会很有限。
