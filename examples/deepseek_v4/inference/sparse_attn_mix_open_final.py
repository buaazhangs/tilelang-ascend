# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
from typing import Optional

import torch
import tilelang
import tilelang.language as T

FP16 = "float16"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


@tilelang.jit(target="npuir")
def sparse_attn_mix_kernel(
    block_top_k,
    block_heads,
    num_heads,
    dim,
    multibuffer=2,
    scale=None,
    dtype=BF16,
    accum_dtype=FP32,
    indices_dtype=INT32,
):
    if scale is None:
        scale = (1.0 / dim) ** 0.5

    assert block_heads % 2 == 0, "mix kernel maps one cube block to two vector sub-blocks"

    batch_size = T.symbolic("batchSize")
    seq_len = T.symbolic("seqLen")
    seq_len_kv = T.symbolic("seqLenKV")
    top_k = T.symbolic("topK")

    block_heads_half = block_heads // 2

    @T.prim_func
    def sparseAttnMix(
        Q: T.Tensor((batch_size, seq_len, num_heads, dim), dtype),
        KV: T.Tensor((batch_size, seq_len_kv, dim), dtype),
        Output: T.Tensor((batch_size, seq_len, num_heads, dim), dtype),
        AttnSink: T.Tensor((num_heads,), accum_dtype),
        TopKIndices: T.Tensor((batch_size, seq_len, top_k), indices_dtype),
    ):
        with T.Kernel(batch_size * seq_len, is_npu=True) as (cid, vid):
            by = cid // seq_len
            bx = cid % seq_len
            value_zero = 0
            value_min = -T.infinity(accum_dtype)

            q_shared = T.alloc_shared((block_heads, dim), dtype)
            kv_shared = T.alloc_shared((block_top_k, dim), dtype)
            prob_shared = T.alloc_shared((block_heads, block_top_k), dtype)
            scores = T.alloc_fragment((block_heads, block_top_k), accum_dtype)
            scores_cast = T.alloc_shared((block_heads_half, block_top_k), dtype)
            pv_acc = T.alloc_fragment((block_heads, dim), accum_dtype)

            kv_ub = T.alloc_shared((block_top_k, dim), dtype)
            idxs = T.alloc_fragment((block_top_k,), indices_dtype)
            mask_ub = T.alloc_shared((1, block_top_k), accum_dtype)
            scores_ub = T.alloc_shared((block_heads_half, block_top_k), accum_dtype)
            scores_max = T.alloc_shared((block_heads_half, 1), accum_dtype)
            scores_max_prev = T.alloc_shared((block_heads_half, 1), accum_dtype)
            scores_scale = T.alloc_shared((block_heads_half, 1), accum_dtype)
            scores_sum = T.alloc_shared((block_heads_half, 1), accum_dtype)
            sum_exp = T.alloc_shared((block_heads_half, 1), accum_dtype)
            acc_o = T.alloc_shared((block_heads_half, dim), accum_dtype)
            acc_o_new = T.alloc_shared((block_heads_half, dim), accum_dtype)
            o_cast = T.alloc_shared((block_heads_half, dim), dtype)

            workspace_kv = T.alloc_workspace(
                (block_top_k, dim), dtype, multi_buffer=multibuffer
            )
            workspace_mask = T.alloc_workspace(
                (1, block_top_k), accum_dtype, multi_buffer=multibuffer
            )
            workspace_score = T.alloc_workspace(
                (block_heads, block_top_k), accum_dtype, multi_buffer=multibuffer
            )
            workspace_prob = T.alloc_workspace(
                (block_heads, block_top_k), dtype, multi_buffer=multibuffer
            )
            workspace_out = T.alloc_workspace(
                (block_heads, dim), accum_dtype, multi_buffer=multibuffer
            )

            # The mix lowering treats a pipelined loop as the C/V pipeline region.
            for n in T.serial(T.ceildiv(num_heads, block_heads)):
                T.vbrc(value_zero, acc_o)
                T.vbrc(value_zero, sum_exp)
                T.vbrc(value_min, scores_max)
                T.copy(Q[by, bx, n * block_heads, 0], q_shared, size=[block_heads, dim])

                for k in T.Pipelined(T.ceildiv(top_k, block_top_k), num_stages=multibuffer):
                    real_block_top_k = T.min(top_k - k * block_top_k, block_top_k)

                    T.vbrc(value_zero, kv_ub)
                    T.vbrc(value_zero, mask_ub)
                    T.copy(
                        TopKIndices[by, bx, k * block_top_k],
                        idxs,
                        size=[real_block_top_k],
                    )
                    for i in T.serial(real_block_top_k):
                        cur_idx = idxs[i]
                        if cur_idx != -1:
                            mask_ub[0, i] = 1.0
                            T.copy(KV[by, cur_idx, 0], kv_ub[i, 0], size=[1, dim])

                    T.copy(kv_ub, workspace_kv, size=[block_top_k, dim])
                    T.copy(mask_ub, workspace_mask, size=[1, block_top_k])

                    T.copy(workspace_kv, kv_shared, size=[block_top_k, dim])
                    T.gemm(
                        q_shared,
                        kv_shared,
                        scores,
                        initC=True,
                        b_transpose=True,
                        size=[block_heads, dim, block_top_k],
                    )
                    T.copy(scores, workspace_score, size=[block_heads, block_top_k])

                    T.copy(
                        workspace_score[vid * block_heads_half, 0],
                        scores_ub,
                        size=[block_heads_half, block_top_k],
                    )
                    T.copy(workspace_mask, mask_ub, size=[1, block_top_k])

                    T.copy(scores_max, scores_max_prev)
                    T.vmul(scores_ub, scale, scores_ub)
                    T.reduce_max(scores_ub, scores_max, dim=1)
                    for i in T.Parallel(block_heads_half):
                        scores_scale[i, 0] = T.exp(scores_max_prev[i, 0] - scores_max[i, 0])
                    for i, j in T.Parallel(block_heads_half, block_top_k):
                        scores_ub[i, j] = T.exp(scores_ub[i, j] - scores_max[i, 0])
                    for i, j in T.Parallel(block_heads_half, block_top_k):
                        scores_ub[i, j] *= mask_ub[0, j]
                    T.reduce_sum(scores_ub, scores_sum, dim=1)
                    for i in T.Parallel(block_heads_half):
                        sum_exp[i, 0] = sum_exp[i, 0] * scores_scale[i, 0] + scores_sum[i, 0]
                    T.vcast(scores_ub, scores_cast, round_mode="rint")
                    T.copy(
                        scores_cast,
                        workspace_prob[vid * block_heads_half, 0],
                        size=[block_heads_half, block_top_k],
                    )

                    T.copy(workspace_prob, prob_shared, size=[block_heads, block_top_k])
                    T.copy(workspace_kv, kv_shared, size=[block_top_k, dim])
                    T.gemm(
                        prob_shared,
                        kv_shared,
                        pv_acc,
                        initC=True,
                        size=[block_heads, block_top_k, dim],
                    )
                    T.copy(pv_acc, workspace_out, size=[block_heads, dim])

                    T.copy(
                        workspace_out[vid * block_heads_half, 0],
                        acc_o_new,
                        size=[block_heads_half, dim],
                    )
                    T.vmul(acc_o, scores_scale, acc_o)
                    T.vadd(acc_o, acc_o_new, acc_o)

                T.copy(
                    AttnSink[n * block_heads + vid * block_heads_half],
                    scores_max_prev[:, 0],
                    size=[block_heads_half],
                )
                for i in T.Parallel(block_heads_half):
                    sum_exp[i, 0] += T.exp(scores_max_prev[i, 0] - scores_max[i, 0])
                T.vdiv(acc_o, sum_exp, acc_o)
                T.vcast(acc_o, o_cast, round_mode="rint")
                real_heads = T.min(
                    block_heads_half,
                    num_heads - n * block_heads - vid * block_heads_half,
                )
                T.copy(
                    o_cast,
                    Output[by, bx, n * block_heads + vid * block_heads_half, 0],
                    size=[real_heads, dim],
                )

    return sparseAttnMix


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
):
    block = 32
    block_heads = 16
    multibuffer = 2
    batch_size, seq_len, num_heads, dim = q.size()
    if (
        not hasattr(sparse_attn, "kernel")
        or sparse_attn.num_heads != num_heads
        or sparse_attn.dim != dim
        or sparse_attn.top_k != topk_idxs.shape[-1]
    ):
        os.environ["TILELANG_ASCEND_MODE"] = "Expert"
        sparse_attn.kernel = sparse_attn_mix_kernel(
            block,
            block_heads,
            num_heads,
            dim,
            multibuffer,
            softmax_scale,
        )
        sparse_attn.num_heads = num_heads
        sparse_attn.dim = dim
        sparse_attn.top_k = topk_idxs.shape[-1]

    output = torch.empty((batch_size, seq_len, num_heads, dim), dtype=q.dtype, device=q.device)
    sparse_attn.kernel(q, kv.contiguous(), output, attn_sink, topk_idxs)
    return output


def gather_from_kv(KV, indices):
    b, s1, k = indices.shape
    batch_idx = torch.arange(b, device=KV.device).view(b, 1, 1).expand(-1, s1, k)
    indices_flat = indices.long()
    out = KV[batch_idx, indices_flat, :].squeeze(dim=2)

    mask = (indices != -1).float().unsqueeze(-1)
    out = out * mask

    return out


def softmax_with_sink(x: torch.Tensor, attn_sink: torch.Tensor, head_dim, dim=-1):
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_vals)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    sink_view_shape = [1] * x.dim()
    sink_view_shape[head_dim if head_dim > 0 else head_dim % x.dim()] = x.shape[head_dim]

    sink_term = torch.exp(attn_sink.view(sink_view_shape) - max_vals)
    adjusted_sum = sum_exp + sink_term

    return exp_x / adjusted_sum


def sparse_attn_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
):
    base_dtype = torch.bfloat16
    kv_sparse = gather_from_kv(kv, topk_idxs)
    mask_acc_s = torch.where((topk_idxs == -1).unsqueeze(-2), -torch.inf, 0.0)
    mask_acc_s = mask_acc_s.to(device=q.device, dtype=torch.float32)
    ref_output = (
        softmax_with_sink(
            ((q @ kv_sparse.transpose(-2, -1)).to(torch.float32) + mask_acc_s)
            * softmax_scale,
            attn_sink,
            head_dim=-2,
            dim=-1,
        ).to(base_dtype)
        @ kv_sparse
    )

    return ref_output


def rand_sparse_attn_input(
    batch_size, num_heads, seq_len, seq_len_kv, top_k, dim, seed=88888888
):
    base_dtype = torch.bfloat16
    torch.manual_seed(seed)

    q = torch.randn((batch_size, seq_len, num_heads, dim), dtype=base_dtype).npu()
    kv = torch.randn((batch_size, seq_len_kv, dim), dtype=base_dtype).npu()
    attn_sink = torch.randn((num_heads,), dtype=torch.float32).npu()
    top_k_indices = torch.randint(
        low=0,
        high=seq_len_kv,
        size=(batch_size, seq_len, top_k),
        dtype=torch.int32,
    ).npu()

    max_len = max(seq_len, top_k)
    causal_mask = torch.tril(torch.ones(max_len, max_len)).to(top_k_indices.device)
    causal_mask = causal_mask[:seq_len, :top_k]
    causal_mask = causal_mask.unsqueeze(dim=0).bool()
    top_k_indices = torch.where(causal_mask, top_k_indices, -1)

    scale = (1.0 / dim) ** 0.5

    return {
        "q": q,
        "kv": kv,
        "attn_sink": attn_sink,
        "topk_idxs": top_k_indices,
        "softmax_scale": scale,
    }


def generate_and_save_data(case_id, **kwargs):
    inputs = rand_sparse_attn_input(**kwargs)
    outputs = sparse_attn_torch(**inputs)
    torch.save({"inputs": inputs, "outputs": outputs}, f"case_{case_id}.pt")


def generate_data():
    generate_and_save_data(
        case_id=0,
        batch_size=1,
        num_heads=64,
        seq_len=256,
        seq_len_kv=256,
        top_k=128,
        dim=512,
    )


def run_test():
    data = torch.load("case_0.pt", map_location=torch.device("npu"))
    output = sparse_attn(**data["inputs"])

    torch.testing.assert_close(data["outputs"], output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed.\033[0m")


if __name__ == "__main__":
    generate_data()
    run_test()
