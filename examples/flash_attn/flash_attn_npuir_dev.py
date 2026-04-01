# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T


seq_len = 512
dim = 128

torch.npu.set_device(0)


@tilelang.jit(out_idx=[-2,-1], target="npuir")
def online_flash_attention(
    block_M, block_N, block_K, dtype="float16", accum_dtype="float32"
):
    shape_q = [seq_len, dim]
    shape_k = [seq_len, dim]
    shape_v = [seq_len, dim]
    shape_o = [seq_len, dim]
    shape_lse = [seq_len]
    shape_work = [seq_len, seq_len]
    block_m = block_M
    block_n = block_N

    @T.prim_func
    def flash_attention(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        lse: T.Tensor(shape_lse, accum_dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_m), is_npu=True) as (cid, _):
            offset = cid * block_m
            Q_shared = T.alloc_shared([block_m, dim], dtype)
            T.copy(Q[offset : offset + block_m, 0:dim], Q_shared)

            K_shared = T.alloc_shared([block_n, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)
            scores = T.alloc_fragment([block_m, block_n], accum_dtype)
            scores_cast = T.alloc_fragment([block_m, block_n], dtype)
            correction = T.alloc_fragment([block_m, 1], accum_dtype)  # 前块校正值
            local_max = T.alloc_fragment([block_m, 1], accum_dtype)  # 当前块最大值
            local_sum = T.alloc_fragment([block_m, 1], accum_dtype)
            acc_m = T.alloc_fragment(
                [block_m, 1], accum_dtype
            )  # 行最大值，每个BN块更新
            acc_l = T.alloc_fragment(
                [block_m, 1], accum_dtype
            )  # sum（行-行最大值），每个BN块更新
            acc_o = T.alloc_fragment(
                [block_m, dim], accum_dtype
            )  # 当前bm块 结果最终值，多个bn块结果累加
            tmp = T.alloc_fragment([block_m, block_n], accum_dtype)
            tmp1 = T.alloc_fragment(
                [block_m, 1], accum_dtype
            )  # 当前bn块最大时，前面块的差值（每行）（当前块不是最大就为0）
            new_max = T.alloc_fragment([block_m, 1], accum_dtype)
            scales = T.alloc_fragment([block_m, block_n], accum_dtype)

            value_zero = 0
            scale = (1.0 / dim) ** 0.5
            value_min = -T.infinity(accum_dtype)
            T.vbrc(value_zero, acc_o)  # 初始为0
            T.vbrc(value_zero, acc_l)
            T.vbrc(value_min, acc_m)
            T.vbrc(scale, scales)

            for k in T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=2):

                # cube
                T.copy(K[k * block_n : (k + 1) * block_n, 0:dim], K_shared)
                T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True)

                # vec
                T.vmul(scores, scales, scores)
                T.reduce_max(scores, local_max, dim=1)  # local_max：当前BN块的行最大值
                T.vmax(acc_m, local_max, new_max)  # new_max：最新行最大值
                T.vsub(acc_m, new_max, tmp1)  # tmp1：校正值，负的最大值差
                T.vexp(tmp1, correction)  # exp校正值
                # scores for current loop
                T.vsub(scores, new_max, tmp)
                T.vexp(tmp, scores)  # 减去行最大值的exp(score)
                T.reduce_sum(scores, local_sum, dim=1)
                # acc_l更新：
                # [j=1,BN](exp(xj-max[k=1,BN](xk)))*correct+[j=BN,2BN](exp(xj-max[k=1,2BN](xk)))
                T.vmul(
                    acc_l, correction, acc_l
                )  # 修正值更新，后续用于修正前块计算结果中，未考虑后块最大值
                T.vadd(acc_l, local_sum, acc_l)  # 加上当前BN块的sum行
                T.vmul(acc_o, correction, acc_o)  # 更新acc_o,旧的累积结果需考虑新最大值
                T.vcast(scores, scores_cast, round_mode="rint")
                # copy new_max to acc_m
                T.vbrc(value_zero, tmp1)
                T.vadd(tmp1, new_max, acc_m)

                # cube
                T.copy(V[k * block_n : (k + 1) * block_n, 0:dim], V_shared)
                T.gemm(scores_cast, V_shared, acc_o, initC=False)

            T.vdiv(acc_o, acc_l, acc_o)  # 除以sum（行-行最大值）
            O_cast = T.alloc_shared([block_m, dim], dtype)
            T.vcast(acc_o, O_cast, round_mode="rint")
            real_m = T.min(block_m, seq_len - cid * block_m)  # 尾块
            T.copy(O_cast, Output[cid * block_m : cid * block_m + real_m, 0:dim])
            # lse
            lse_cast = T.alloc_shared([block_m, 1],  accum_dtype)
            lse_reshape = T.alloc_shared([block_m],  accum_dtype)
            T.vln(acc_l, acc_l)
            T.vadd(acc_l, acc_m, lse_cast)
            T.reshape(lse_cast, lse_reshape)
            T.copy(lse_reshape, lse[cid * block_m : cid * block_m + real_m])

    return flash_attention


def main():
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    kernel = online_flash_attention(64, 64, 32)

    q = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    k = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    v = torch.randn((seq_len, dim), dtype=torch.float16).npu()

    lse, output = kernel(q, k, v)

    scale = (1.0 / dim) ** 0.5
    ref_output = (
        torch.nn.functional.softmax((q @ k.T).to(torch.float32) * scale, dim=-1).to(
            torch.float16
        )
        @ v
    )
    ref_lse = torch.logsumexp((q @ k.T).to(torch.float32) * scale, dim=-1)
    print("output:")
    print(output)
    print("ref_output:")
    print(ref_output)
    print("lse:",lse)
    print("ref_lse:",ref_lse)
    torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ref_lse, lse, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    print("bwd begin:")
    # torch

    # tilelang


if __name__ == "__main__":
    main()
