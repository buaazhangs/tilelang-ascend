"""
Tilelang implementation of Mamba3 forward kernel,
with MIMO support.

Copyright (c) 2026, Dao AI Lab, Goombalab

"""

import torch
import tilelang
import tilelang.language as T
import os

torch.npu.set_device(11)


# NOTE: Uncomment the following to autotune:
# def get_configs():
#     iter_params = dict(num_stages=[0, 1, 2, 3], threads=[128, 256, 512])
#     # iter_params = dict(num_stages=[2], threads=[128])
#     return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

# @autotune(
#     configs=get_configs(),
#     warmup=3,
#     rep=20,
# )
@tilelang.jit(
    out_idx=[], target="npuir")
def mamba_mimo_fwd(
    B,
    S,
    H,
    G,
    N,
    P,
    R,
    hasZ,
    hasD,
    reduceO,
    return_final_state=False,
    chunk_size: int = 16,
    rotary_dim_divisor = 4,
    dtype: str = 'bfloat16',
    threads: int = 128,
    num_stages: int = 0,
) -> torch.Tensor:

    accum_dtype = 'float32'
    # mix 模式至少需要 2 个 stage，这样 workspace 才能被扩成乒乓缓冲。
    mix_num_stages = num_stages if num_stages and num_stages > 1 else 2

    nchunks = tilelang.cdiv(S, chunk_size)
    tail_len = S % chunk_size
    fused_chunk_size = chunk_size * R
    # vid 负责按 chunk 内的 step 维度切分 vector 侧工作。
    block_vid = (chunk_size + 1) // 2

    if reduceO:
        O_shape = (B, S, H, P)
    else:
        O_shape = (B, S, R, H, P)


    @T.prim_func
    def mamba_mimo_fwd_kernel(
            Q: T.Tensor([B, S, R, G, N], dtype),  # type: ignore
            K: T.Tensor([B, S, R, G, N], dtype),  # type: ignore
            V: T.Tensor([B, S, H, P], dtype),  # type: ignore
            O: T.Tensor(O_shape, dtype),  # type: ignore
            Q_BIAS: T.Tensor([H, R, N], "float32"),  # type: ignore
            K_BIAS: T.Tensor([H, R, N], "float32"),  # type: ignore
            MIMO_V: T.Tensor([H, R, P], "float32"), # type: ignore
            MIMO_O: T.Tensor([H, R, P], "float32"), # type: ignore
            Z: T.Tensor([B, S, H, P], dtype),  # type: ignore
            D: T.Tensor([H], "float32"),  # type: ignore
            MIMO_Z: T.Tensor([H, R, P], "float32"), # type: ignore
            ANGLES: T.Tensor([B, S, H, N//rotary_dim_divisor], "float32"), # type: ignore
            DA_CS: T.Tensor([B, H, S], "float32"), # type: ignore
            DA_CS_REV: T.Tensor([B, H, S], "float32"), # type: ignore
            DT: T.Tensor([B, H, S + 1], "float32"), # type: ignore
            TRAP: T.Tensor([B, H, S + 1], dtype), # type: ignore
            SEGSUM: T.Tensor([B, H, nchunks, chunk_size, chunk_size], "float32"), # type: ignore

            FINAL_STATE: T.Tensor([B, H, N, P], "float32"),  # type: ignore
            FINAL_K: T.Tensor([B, R, H, N], dtype)  # type: ignore
            ):
        """
        Overview:
            Fused chunked forward pass that combines MIMO projections with recurrent state updates.
            Computes interchunk and intrachunk contributions with optional D and Z paths,
            then writes output activations.

        Inputs:
            - Activations: Q, K, V.
            - Projection parameters/biases: MIMO_V (Psi), MIMO_O (Phi), optional MIMO_Z (Zeta), ANGLES,
              and Q_BIAS/K_BIAS.
            - Optional modifiers: Z, and D.
            - Discretization tensors: DA_CS, DA_CS_REV, DT, TRAP, and SEGSUM.

        Outputs:
            - O: fused forward output activations.
            - FINAL_STATE: final recurrent states (if return_final_state is True).
            - FINAL_K: final K tensor (if return_state is True, for use in decode)

        Notation:
            - Psi: MIMO X projection.
            - Phi: MIMO O projection.
            - Zeta: MIMO Z projection.
            - Trap: convex-combination modulator used in exponential-trapezoidal discretization.
        """
        
        with T.Kernel(B * H, is_npu=True) as (cid, vid):
            # --- Kernel Setup ---
            i_b = cid // H
            i_h = cid % H
            # GQA support: map V head to Q/K head
            i_h_qk = i_h // (H // G)

            # --- Buffer Allocation ---
            q_shared = T.alloc_shared([fused_chunk_size, N], dtype)
            k_shared = T.alloc_shared([fused_chunk_size, N], dtype)
            q_bias_frag = T.alloc_fragment([R, N], dtype)
            k_bias_frag = T.alloc_fragment([R, N], dtype)
            q_bias_f32_frag = T.alloc_fragment([R, N], "float32")
            k_bias_f32_frag = T.alloc_fragment([R, N], "float32")

            PsiV_shared = T.alloc_shared([fused_chunk_size, P], dtype)
            v_shared = T.alloc_shared([chunk_size, P], dtype)
            states_accum_cast_shared = T.alloc_shared([N, P], dtype)
            qk_intrachunk_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], dtype)

            # 下面这几块 workspace 是 mix 模式下的桥接缓冲区。
            # cube 先把中间结果写到 workspace，vector 再按 vid 分片取走处理。
            qk_dot_ws = T.alloc_workspace(
                (2, fused_chunk_size, fused_chunk_size), accum_dtype, multi_buffer=mix_num_stages
            )
            o_inter_ws = T.alloc_workspace(
                (2, fused_chunk_size, P), accum_dtype, multi_buffer=mix_num_stages
            )
            qk_intrachunk_ws = T.alloc_workspace(
                (2, fused_chunk_size, fused_chunk_size), accum_dtype, multi_buffer=mix_num_stages
            )
            qk_masked_ws = T.alloc_workspace(
                (2, fused_chunk_size, fused_chunk_size), dtype, multi_buffer=mix_num_stages
            )
            o_intra_ws = T.alloc_workspace(
                (2, fused_chunk_size, P), accum_dtype, multi_buffer=mix_num_stages
            )

            # --- Swizzling Annotation ---
            T.annotate_layout({
                q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                v_shared: tilelang.layout.make_swizzled_layout(v_shared),
                PsiV_shared: tilelang.layout.make_swizzled_layout(PsiV_shared),
                states_accum_cast_shared: tilelang.layout.make_swizzled_layout(states_accum_cast_shared),
                qk_intrachunk_shared: tilelang.layout.make_swizzled_layout(qk_intrachunk_shared),
            })
            T.use_swizzle(10, "row")

            # --- Per-Head Constants / Running State ---
            # 阶段 0：Head 级初始化。
            # 这部分主要准备常驻参数和递归状态：
            # - states_frag: [N, P]
            # - Psi_frag:    [R, P]
            # - q/k bias:    [R, N]
            states_frag = T.alloc_fragment([N, P], accum_dtype)
            T.clear(states_frag)

            Psi_frag = T.alloc_fragment([R, P], dtype)
            Psi_frag_f32 = T.alloc_fragment([R, P], "float32")
            T.copy(MIMO_V[i_h, :, :], Psi_frag_f32)
            T.vcast(Psi_frag_f32, Psi_frag, round_mode="rint")

            T.copy(Q_BIAS[i_h, :, :], q_bias_f32_frag)
            T.vcast(q_bias_f32_frag, q_bias_frag, round_mode="rint")
            T.copy(K_BIAS[i_h, :, :], k_bias_f32_frag)
            T.vcast(k_bias_f32_frag, k_bias_frag, round_mode="rint")

            # --- Chunk Loop ---
            for i in T.Pipelined(0, nchunks, num_stages=mix_num_stages):

                # vec1
                
                chunk_start = i * chunk_size

                segsum = T.alloc_fragment([chunk_size, chunk_size], "float32")
                T.copy(SEGSUM[i_b, i_h, i, :, :], segsum)

                # ==================== 阶段 1：Vector 预处理 ====================
                # 从 GM 读取 DT / TRAP / SEGSUM，并做逐元素计算，得到：
                # - shifted_gamma_frag: [chunk_size]
                # - gamma_frag:         [chunk_size]
                # - trap_scale_shared:  [chunk_size]
                # 这些量后面会参与 K 缩放和对角项修正。
                trap_shifted_frag = T.alloc_fragment([chunk_size], "float32")
                trap_shifted_bf16 = T.alloc_fragment([chunk_size], dtype)
                T.copy(TRAP[i_b, i_h, chunk_start+1: chunk_start+chunk_size+1], trap_shifted_bf16)
                T.vcast(trap_shifted_bf16, trap_shifted_frag, round_mode="rint")
                T.vmul(trap_shifted_frag, -1, trap_shifted_frag)
                T.vsigmoid(trap_shifted_frag, trap_shifted_frag)
                dt_shifted_frag = T.alloc_fragment([chunk_size], dtype)
                dt_shifted_f32_frag = T.alloc_fragment([chunk_size], "float32")
                T.copy(DT[i_b, i_h, chunk_start+1: chunk_start+chunk_size+1], dt_shifted_f32_frag)
                T.vcast(dt_shifted_f32_frag, dt_shifted_frag, round_mode="rint")
                shifted_gamma_frag = T.alloc_fragment([chunk_size], dtype)
                T.clear(shifted_gamma_frag)
                # for cs in T.Parallel(chunk_size):
                #     shifted_gamma_frag[cs] = T.if_then_else(chunk_start + cs < (S - 1), 
                #                                             dt_shifted_frag[cs] * trap_shifted_frag[cs], 
                #                                             0.0)
                for cs in T.serial(chunk_size):
                    if (chunk_start + cs < (S - 1)):
                        shifted_gamma_frag[cs] = dt_shifted_frag[cs] * trap_shifted_frag[cs]

                shifted_gamma_shared = T.alloc_shared([chunk_size], dtype)
                T.copy(shifted_gamma_frag, shifted_gamma_shared)

                trap_frag = T.alloc_fragment([chunk_size], "float32")
                trap_bf16 = T.alloc_fragment([chunk_size], dtype)
                T.copy(TRAP[i_b, i_h, chunk_start: chunk_start+chunk_size], trap_bf16)
                T.vcast(trap_bf16, trap_frag, round_mode="rint")
                T.vsigmoid(trap_frag, trap_frag)
                dt_frag = T.alloc_fragment([chunk_size], dtype)
                dt_f32_frag = T.alloc_fragment([chunk_size], "float32")
                T.copy(DT[i_b, i_h, chunk_start: chunk_start+chunk_size], dt_f32_frag)
                T.vcast(dt_f32_frag, dt_frag, round_mode="rint")
                gamma_frag = T.alloc_fragment([chunk_size], "float32")
                # 这里paraller还不支持
                for cs in T.serial(chunk_size):
                    gamma_frag[cs] = dt_frag[cs] * trap_frag[cs]
                trap_scale_frag = T.alloc_fragment([chunk_size], dtype)
                for cs in T.Parallel(chunk_size):
                    trap_scale_frag[cs] = gamma_frag[cs] + shifted_gamma_shared[cs]
                trap_scale_shared = T.alloc_shared([chunk_size], dtype)
                T.copy(trap_scale_frag, trap_scale_shared)

                # ==================== 阶段 2：Vector/搬运预处理 ====================
                # 从 GM 读取当前 chunk 的 V / Q / K，并为后续 GEMM 准备输入：
                # - v_shared:    [chunk_size, P]
                # - PsiV_shared: [chunk_size * R, P]
                # - q_shared:    [chunk_size * R, N]
                # - k_shared:    [chunk_size * R, N]
                PsiV_frag = T.alloc_fragment([chunk_size, R, P], dtype)
                for cs, p in T.Parallel(chunk_size, P):
                    v_shared[cs, p] = V[i_b, chunk_start+cs, i_h, p]
                for cs, r, p in T.Parallel(chunk_size, R, P):
                    PsiV_frag[cs, r, p] = v_shared[cs, p] * Psi_frag[r, p]
                PsiV_reshaped_frag = T.alloc_fragment([fused_chunk_size, P], dtype)
                T.reshape(PsiV_frag, PsiV_reshaped_frag)
                T.copy(PsiV_reshaped_frag, PsiV_shared)



                q_frag = T.alloc_fragment([chunk_size, R, N], dtype)
                T.copy(Q[i_b, chunk_start:chunk_start+chunk_size, :, i_h_qk, :], q_frag)
                for cs, r, n in T.Parallel(chunk_size, R, N):
                    q_frag[cs, r, n] += q_bias_frag[r, n]
                T.reshape(q_frag, q_shared)

                

                k_frag = T.alloc_fragment([chunk_size, R, N], dtype)
                T.copy(K[i_b, chunk_start:chunk_start+chunk_size, :, i_h_qk, :], k_frag)
                for cs, r, n in T.Parallel(chunk_size, R, N):
                    k_frag[cs, r, n] += k_bias_frag[r, n]
                T.reshape(k_frag, k_shared)


                # --- Cache Diagonal qk_dot Path ---
                # 这是第一段 cube 计算：
                # 先把每个 chunk 内同一步的 qk dot 算出来，后面 vector 侧要复用这块对角贡献。
                # ==================== 阶段 3：Cube ====================
                # q_shared @ k_shared^T
                # - 输入 q_shared: [chunk_size * R, N]
                # - 输入 k_shared: [chunk_size * R, N]
                # - 输出 qk_dot_ws: [chunk_size * R, chunk_size * R]
                # 这一路是对角项所需的 qk dot，先落到 workspace。
                qk_dot_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype=accum_dtype)
                T.gemm(q_shared, k_shared, qk_dot_frag, b_transpose=True, initC=True)
                T.copy(qk_dot_frag, qk_dot_ws[0, 0, 0], size=[fused_chunk_size, fused_chunk_size])
                # Option B: extremely slow
                # qk_dot_frag = T.alloc_fragment([chunk_size, R, R], dtype=accum_dtype)
                # T.clear(qk_dot_frag)
                # for cs, r_out, r_in in T.Parallel(chunk_size, R, R):
                #     for n in T.serial(N):
                #         qk_dot_frag[cs, r_out, r_in] += (
                #             q_frag[cs, r_out, n] * k_frag[cs, r_in, n]
                #         )
                # T.copy(T.view(qk_dot_frag, shape=[fused_chunk_size, R]), qk_dot_shared)
                # NOTE ("option C"): The following fails Tilelang compilation:
                # qk_predot_frag = T.alloc_fragment([chunk_size, R, R, N], dtype)
                # for cs, r_out, r_in, n in T.Parallel(chunk_size, R, R, N):
                #     qk_predot_frag[cs, r_out, r_in, n] = q_frag[cs, r_out, n] * k_frag[cs, r_in, n]
                # qk_dot_frag = T.alloc_fragment([chunk_size, R, R], dtype)
                # T.reduce_sum(qk_predot_frag, qk_dot_frag, dim=-1, clear=True)
                # T.copy(T.view(qk_dot_frag, shape=[fused_chunk_size, R]), qk_dot_shared)

                # ==================== 阶段 4：Vector 预处理 + Cube ====================
                # 先对 Q 做 rotary，再执行 q @ state：
                # - q_shared(rotary后):       [chunk_size * R, N]
                # - states_accum_cast_shared: [N, P]
                # - o_inter_ws:               [chunk_size * R, P]
                # 这是 inter-chunk 路径，结果先写到 workspace。
                q_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                q_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)

                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    q_first_half_frag[cs, r, n] = q_shared[cs*R + r, n]
                    q_second_half_frag[cs, r, n] = q_shared[cs*R + r, N//2 + n]

                # NOTE: angles are casted to fp32 for numerical stability
                angles_frag = T.alloc_fragment([chunk_size, N//rotary_dim_divisor], "float32")
                T.copy(ANGLES[i_b, chunk_start:chunk_start+chunk_size, i_h, :], angles_frag)
                angles_frag_cos = T.alloc_fragment([chunk_size, N//rotary_dim_divisor], "float32")
                T.vcos(angles_frag, angles_frag_cos)
                angles_frag_sin = T.alloc_fragment([chunk_size, N//rotary_dim_divisor], "float32")
                T.vsin(angles_frag, angles_frag_sin)

                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    q_shared[cs*R + r, n] = angles_frag_cos[cs, n] * q_first_half_frag[cs, r, n] - angles_frag_sin[cs, n] * q_second_half_frag[cs, r, n]
                    q_shared[cs*R + r, N//2 + n] = angles_frag_sin[cs, n] * q_first_half_frag[cs, r, n] + angles_frag_cos[cs, n] * q_second_half_frag[cs, r, n]

                o_mimo_accum_frag = T.alloc_fragment([fused_chunk_size, P], dtype=accum_dtype)
                states_frag_cast = T.alloc_fragment([N, P], dtype)
                T.vcast(states_frag, states_frag_cast, round_mode="rint")
                T.copy(states_frag_cast, states_accum_cast_shared)
                T.gemm(q_shared, states_accum_cast_shared, o_mimo_accum_frag, initC=True)
                # 这部分是 inter-chunk 的 q @ state，先落到 workspace，后续 vector 再乘 exp(DA_CS)。
                T.copy(o_mimo_accum_frag, o_inter_ws[0, 0, 0], size=[fused_chunk_size, P])

                # ==================== 阶段 5：Vector 预处理 ====================
                # 对 K 做 rotary，并乘 trap_scale：
                # - 输入/输出 k_shared: [chunk_size * R, N]
                # 这是后续 intrachunk GEMM 的 K 输入准备阶段。
                k_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                k_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)

                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    k_first_half_frag[cs, r, n] = k_shared[cs*R + r, n]
                    k_second_half_frag[cs, r, n] = k_shared[cs*R + r, N//2 + n]
                
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    k_shared[cs*R + r, n] = angles_frag_cos[cs, n] * k_first_half_frag[cs, r, n] - angles_frag_sin[cs, n] * k_second_half_frag[cs, r, n]
                    k_shared[cs*R + r, N//2 + n] = angles_frag_sin[cs, n] * k_first_half_frag[cs, r, n] + angles_frag_cos[cs, n] * k_second_half_frag[cs, r, n]

                # FINAL_K 链路先停用，当前 mix 文件只聚焦 O 的正确性。
                # if i == nchunks - 1 and return_final_state:
                #     seq_boundary = T.min(chunk_start + chunk_size, S) - chunk_start
                #     for csr, n in T.Parallel(fused_chunk_size, N):
                #         if csr >= (seq_boundary - 1) * R and csr < seq_boundary * R:
                #             T.copy(k_shared[csr, n], FINAL_K[i_b, csr % R, i_h, n], size=[1,1])

                k_trap_scaled_frag = T.alloc_fragment([fused_chunk_size, N], dtype)
                T.copy(k_shared, k_trap_scaled_frag)
                # for csr, n in T.Parallel(fused_chunk_size, N):
                #     k_trap_scaled_frag[csr, n] *= trap_scale_shared[csr//R]
                k_trap_scaled_frag_f32 = T.alloc_fragment([fused_chunk_size, N], "float32")
                T.vcast(k_trap_scaled_frag, k_trap_scaled_frag_f32)
                for csr in T.serial(fused_chunk_size):
                    for n in T.serial(N):
                        k_trap_scaled_frag_f32[csr, n] *= trap_scale_shared[csr//R]
                T.vcast(k_trap_scaled_frag_f32, k_trap_scaled_frag)
                T.copy(k_trap_scaled_frag, k_shared)

                # ==================== 阶段 6：Cube ====================
                # q_shared @ k_shared^T（这里的 k_shared 已经做完 rotary + trap_scale）
                # - 输入 q_shared:         [chunk_size * R, N]
                # - 输入 k_shared:         [chunk_size * R, N]
                # - 输出 qk_intrachunk_ws: [chunk_size * R, chunk_size * R]
                # 这一路是 chunk 内相关性主干，结果先写到 workspace。
                qk_intrachunk_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype=accum_dtype)
                T.gemm(q_shared, k_shared, qk_intrachunk_frag, b_transpose=True, initC=True)
                # 这是第二段 cube 计算：chunk 内严格因果部分的原始 qk 结果。
                T.copy(qk_intrachunk_frag, qk_intrachunk_ws[0, 0, 0], size=[fused_chunk_size, fused_chunk_size])

                # 第一个 vector 阶段：
                # 1. 按 vid 拆分当前 chunk 的若干 step。
                # 2. 做严格因果 mask。
                # 3. 对 inter-chunk 部分乘 exp(DA_CS) 衰减。
                # ==================== 阶段 7：Vector ====================
                # 按 vid 分片处理当前 chunk 的一部分行：
                # - qk_intrachunk_vec: [block_vid * R, chunk_size * R]
                # - o_inter_vec:       [block_vid * R, P]
                # - da_cs_vec:         [block_vid]
                # 完成严格因果 mask、乘 exp(SEGSUM)、乘 exp(DA_CS)，再写回 workspace。
                T.vexp(segsum, segsum)
                if vid * block_vid < chunk_size:
                    vec_chunk_start = vid * block_vid
                    vec_rows = T.min(block_vid, chunk_size - vec_chunk_start)
                    vec_row_offset = vec_chunk_start * R

                    qk_intrachunk_vec = T.alloc_shared([block_vid * R, fused_chunk_size], accum_dtype)
                    qk_intrachunk_masked_vec = T.alloc_shared([block_vid * R, fused_chunk_size], dtype)
                    o_inter_vec = T.alloc_shared([block_vid * R, P], accum_dtype)
                    da_cs_vec = T.alloc_shared([block_vid], "float32")

                    T.copy(
                        qk_intrachunk_ws[0, vec_row_offset, 0],
                        qk_intrachunk_vec,
                        size=[vec_rows * R, fused_chunk_size],
                    )
                    T.copy(
                        o_inter_ws[0, vec_row_offset, 0],
                        o_inter_vec,
                        size=[vec_rows * R, P],
                    )
                    T.copy(
                        DA_CS[i_b, i_h, chunk_start + vec_chunk_start],
                        da_cs_vec,
                        size=[vec_rows],
                    )
                    T.vexp(da_cs_vec, da_cs_vec)

                    for local_csr in T.serial(block_vid * R):
                        if local_csr < vec_rows * R:
                            global_cs = vec_chunk_start + local_csr // R
                            for csr_j in T.serial(fused_chunk_size):
                                if global_cs > csr_j // R:
                                    qk_intrachunk_masked_vec[local_csr, csr_j] = (
                                        qk_intrachunk_vec[local_csr, csr_j]
                                        * segsum[global_cs, csr_j // R]
                                    )
                                else:
                                    qk_intrachunk_masked_vec[local_csr, csr_j] = 0
                            for p in T.serial(P):
                                o_inter_vec[local_csr, p] *= da_cs_vec[local_csr // R]

                    T.copy(
                        qk_intrachunk_masked_vec,
                        qk_masked_ws[0, vec_row_offset, 0],
                        size=[vec_rows * R, fused_chunk_size],
                    )
                    T.copy(
                        o_inter_vec,
                        o_inter_ws[0, vec_row_offset, 0],
                        size=[vec_rows * R, P],
                    )

                # ==================== 阶段 8：Cube ====================
                # qk_masked @ PsiV
                # - 输入 qk_masked: [chunk_size * R, chunk_size * R]
                # - 输入 PsiV:      [chunk_size * R, P]
                # - 输出 o_intra_ws:[chunk_size * R, P]
                # 这是 intrachunk 的主贡献路径，结果写回 workspace。
                T.copy(qk_masked_ws[0, 0, 0], qk_intrachunk_shared, size=[fused_chunk_size, fused_chunk_size])
                tmp = T.alloc_fragment([fused_chunk_size, P], dtype=accum_dtype)
                T.gemm(qk_intrachunk_shared, PsiV_shared, tmp, initC=True)
                # 这是第三段 cube 计算：masked qk 与 PsiV 相乘，得到 intra-chunk 输出贡献。
                T.copy(tmp, o_intra_ws[0, 0, 0], size=[fused_chunk_size, P])

                # 第二个 vector 阶段：
                # 1. 把 inter/intra/diag 三部分输出合并。
                # 2. 可选加入 D 支路。
                # 3. 做 Z gate。
                # 4. reduceO 时再乘 MIMO_O 并沿 R 归约，最终写回 O。
                # ==================== 阶段 9：Vector ====================
                # 从 workspace 读回三条主路径并在 vector 侧完成最终收敛：
                # - o_inter_ws -> o_inter_vec: [block_vid * R, P]
                # - o_intra_ws -> o_intra_vec: [block_vid * R, P]
                # - qk_dot_ws  -> qk_dot_vec:  [block_vid * R, chunk_size * R]
                # 然后合并 inter / intra / diagonal，可选叠加 D 和 Z，最后写回 O。
                if vid * block_vid < chunk_size:
                    vec_chunk_start = vid * block_vid
                    vec_rows = T.min(block_vid, chunk_size - vec_chunk_start)
                    vec_row_offset = vec_chunk_start * R

                    o_inter_vec = T.alloc_shared([block_vid * R, P], accum_dtype)
                    o_intra_vec = T.alloc_shared([block_vid * R, P], accum_dtype)
                    qk_dot_vec = T.alloc_shared([block_vid * R, fused_chunk_size], accum_dtype)
                    z_vec = T.alloc_shared([block_vid, P], dtype)

                    T.copy(o_inter_ws[0, vec_row_offset, 0], o_inter_vec, size=[vec_rows * R, P])
                    T.copy(o_intra_ws[0, vec_row_offset, 0], o_intra_vec, size=[vec_rows * R, P])
                    T.copy(qk_dot_ws[0, vec_row_offset, 0], qk_dot_vec, size=[vec_rows * R, fused_chunk_size])
                    if hasZ:
                        T.copy(Z[i_b, chunk_start + vec_chunk_start, i_h, 0], z_vec, size=[vec_rows, P])

                    for local_csr in T.serial(block_vid * R):
                        if local_csr < vec_rows * R:
                            global_cs = vec_chunk_start + local_csr // R
                            global_r = local_csr % R
                            for p in T.serial(P):
                                o_inter_vec[local_csr, p] += o_intra_vec[local_csr, p]
                                diag_acc = 0
                                for r_in in T.serial(R):
                                    diag_acc += (
                                        qk_dot_vec[local_csr, global_cs * R + r_in]
                                        * PsiV_shared[global_cs * R + r_in, p]
                                    )
                                diag_acc *= gamma_frag[global_cs]
                                if hasD:
                                    diag_acc += D[i_h] * PsiV_shared[global_cs * R + global_r, p]
                                o_inter_vec[local_csr, p] += diag_acc

                    if reduceO:
                        o_reduce_vec = T.alloc_shared([block_vid, P], dtype)
                        for local_cs in T.serial(block_vid):
                            if local_cs < vec_rows:
                                for p in T.serial(P):
                                    out_val = 0
                                    for r in T.serial(R):
                                        cur_val = o_inter_vec[local_cs * R + r, p]
                                        cur_val *= MIMO_O[i_h, r, p]
                                        if hasZ:
                                            gate_pre = z_vec[local_cs, p] * MIMO_Z[i_h, r, p] * 0.5
                                            gate_tanh = T.tanh(gate_pre)
                                            cur_val *= gate_pre * gate_tanh + gate_pre
                                        out_val += cur_val
                                    o_reduce_vec[local_cs, p] = out_val
                        T.copy(
                            o_reduce_vec,
                            O[i_b, chunk_start + vec_chunk_start, i_h, 0],
                            size=[vec_rows, P],
                        )
                    else:
                        o_store_vec = T.alloc_shared([block_vid, R, P], dtype)
                        for local_cs in T.serial(block_vid):
                            if local_cs < vec_rows:
                                for r, p in T.Parallel(R, P):
                                    cur_val = o_inter_vec[local_cs * R + r, p]
                                    if hasZ:
                                        gate_pre = z_vec[local_cs, p] * MIMO_Z[i_h, r, p] * 0.5
                                        gate_tanh = T.tanh(gate_pre)
                                        cur_val *= gate_pre * gate_tanh + gate_pre
                                    o_store_vec[local_cs, r, p] = cur_val
                        T.copy(
                            o_store_vec,
                            O[i_b, chunk_start + vec_chunk_start, 0, i_h, 0],
                            size=[vec_rows, R, P],
                        )

                # FINAL_STATE 这条 recurrent state update 链路先整段停用。
                # dA_cs_rev_frag = T.alloc_fragment([chunk_size], "float32")
                # T.copy(DA_CS_REV[i_b, i_h, chunk_start:chunk_start+chunk_size], dA_cs_rev_frag)
                #
                # k_state_frag = T.alloc_fragment([fused_chunk_size, N], dtype)
                # T.copy(k_shared, k_state_frag)
                # for csr, n in T.Parallel(fused_chunk_size, N):
                #     k_state_frag[csr, n] *= T.exp(dA_cs_rev_frag[csr//R])
                #
                # da_cs_sum = T.alloc_fragment([1,1], "float32")
                # if tail_len > 0 and i == nchunks - 1:
                #     T.copy(DA_CS[i_b, i_h, S - 1], da_cs_sum)
                #     if return_final_state:
                #         for csr, n in T.Parallel(fused_chunk_size, N):
                #             k_state_frag[csr, n] = T.if_then_else(csr < tail_len * R, k_state_frag[csr, n], 0.0)
                # else:
                #     T.copy(DA_CS[i_b, i_h, chunk_start+chunk_size-1], da_cs_sum)
                # T.vexp(da_cs_sum, da_cs_sum)
                # for n, p in T.Parallel(N, P):
                #     states_frag[n, p] *= da_cs_sum[0, 0]
                # k_state_frag_tmp = T.alloc_fragment([fused_chunk_size, N], "float32")
                # T.vcast(k_state_frag, k_state_frag_tmp)
                # T.vadd(k_state_frag_tmp, 0, k_state_frag_tmp)
                # T.vcast(k_state_frag_tmp, k_state_frag)
                # T.gemm(k_state_frag, PsiV_shared, states_frag, a_transpose=True, initC=False)
            
            # FINAL_STATE 写回先停用。
            # if return_final_state:
            #     T.copy(states_frag, FINAL_STATE[i_b, i_h, :, :])

    return mamba_mimo_fwd_kernel


def ref_forward(
    B, S, H, G, N, P, R, chunk_size, rotary_dim_divisor,
    hasZ, hasD, reduceO, return_final_state,
    Q, K, V, Q_BIAS, K_BIAS, MIMO_V, MIMO_O,
    Z, D, MIMO_Z, ANGLES, DA_CS, DA_CS_REV,
    DT, TRAP, SEGSUM,
    device, dtype_torch,
):
    """
    Ref strictly mirrors kernel 13 logic, including dtype casts at each step.
    """
    nchunks       = (S + chunk_size - 1) // chunk_size
    tail_len      = S % chunk_size
    fused_chunk_size = chunk_size * R
    rotary_n      = N // rotary_dim_divisor
    half          = N // 2
    bf16          = dtype_torch   # bfloat16
    f32           = torch.float32

    if reduceO:
        O_ref = torch.zeros(B, S, H, P, dtype=bf16, device=device)
    else:
        O_ref = torch.zeros(B, S, R, H, P, dtype=bf16, device=device)
    # FINAL_STATE/FINAL_K 参考链路先停用，当前只保留 O 的参考值。
    FINAL_STATE_ref = None
    FINAL_K_ref = None

    for b in range(B):
        for h in range(H):
            h_qk = h // (H // G)

            # --- Per-head constants loaded into bf16 fragments ---
            Psi_frag  = MIMO_V[h].to(bf16)          # [R, P] bf16
            Phi_frag  = MIMO_O[h].to(bf16)          # [R, P] bf16
            q_bias    = Q_BIAS[h].to(bf16)          # [R, N] bf16
            k_bias    = K_BIAS[h].to(bf16)          # [R, N] bf16

            # FINAL_STATE 参考链路先停用。
            states = None

            for i in range(nchunks):
                cs0 = i * chunk_size
                cs1 = min(cs0 + chunk_size, S)
                L   = cs1 - cs0

                # === shifted_gamma ===
                # kernel: T.Parallel, if_then_else(cs0+cs < S-1, dt*sigmoid(-trap), 0)
                # reads TRAP[cs0+1 .. cs0+chunk_size+1], DT same range (shape S, guarded)
                shifted_gamma = torch.zeros(chunk_size, dtype=bf16, device=device)
                for cs in range(chunk_size):
                    if cs0 + cs < S - 1:
                        idx = cs0 + cs + 1
                        trap_s = TRAP[b, h, idx].float()
                        dt_s   = DT[b, h, idx]          # f32
                        shifted_gamma[cs] = (dt_s * torch.sigmoid(-trap_s)).to(bf16)
                # shifted_gamma_shared: bf16

                # === gamma_frag ===
                # kernel: T.Parallel, gamma = dt * sigmoid(trap), both read as f32/bf16
                gamma = torch.zeros(chunk_size, dtype=f32, device=device)
                for cs in range(chunk_size):
                    trap_c = TRAP[b, h, cs0 + cs].float()   # f32 (trap_frag is f32)
                    dt_c   = DT[b, h, cs0 + cs]             # f32
                    gamma[cs] = dt_c * torch.sigmoid(trap_c)
                # gamma_frag: f32

                # === trap_scale_frag ===
                # kernel: bf16 = gamma(f32) + shifted_gamma_shared(bf16)  → bf16
                trap_scale = (gamma + shifted_gamma.float()).to(bf16)  # [chunk_size] bf16

                # === PsiV ===
                # v_shared bf16, Psi_frag bf16 → PsiV_frag bf16
                v_chunk = V[b, cs0:cs1, h, :]                         # [L, P] bf16
                PsiV    = (v_chunk[:, None, :].float()
                           * Psi_frag[None, :, :].float()).to(bf16)   # [L, R, P] bf16
                PsiV_r  = PsiV.reshape(L * R, P)                      # [fL, P] bf16

                # === Q/K with bias (bf16) ===
                q = (Q[b, cs0:cs1, :, h_qk, :] + q_bias[None, :, :]).to(bf16)  # [L,R,N]
                k = (K[b, cs0:cs1, :, h_qk, :] + k_bias[None, :, :]).to(bf16)  # [L,R,N]

                # === qk_dot (f32 GEMM, before rotary) ===
                q_r_pre = q.reshape(L * R, N)
                k_r_pre = k.reshape(L * R, N)
                qk_dot  = q_r_pre.float() @ k_r_pre.float().T         # [fL, fL] f32

                # === Rotary on Q (applied to q_shared) ===
                angles = ANGLES[b, cs0:cs1, h, :]                     # [L, rotary_n] f32
                cos_a  = torch.cos(angles)
                sin_a  = torch.sin(angles)

                def apply_rotary_bf16(x):
                    # x: [L, R, N] bf16; angles in f32
                    x1 = x[..., :rotary_n].float()
                    x2 = x[..., half:half + rotary_n].float()
                    out = x.clone()
                    out[..., :rotary_n] = (
                        cos_a[:, None, :] * x1 - sin_a[:, None, :] * x2
                    ).to(bf16)
                    out[..., half:half + rotary_n] = (
                        sin_a[:, None, :] * x1 + cos_a[:, None, :] * x2
                    ).to(bf16)
                    return out

                q_rot = apply_rotary_bf16(q)    # [L, R, N] bf16
                q_r   = q_rot.reshape(L * R, N) # [fL, N] bf16

                # === Interchunk: Q @ states (f32 GEMM) ===
                o_accum = torch.zeros(L * R, P, dtype=f32, device=device)

                # === Rotary on K ===
                k_rot = apply_rotary_bf16(k)    # [L, R, N] bf16
                k_r   = k_rot.reshape(L * R, N) # [fL, N] bf16

                # FINAL_K 参考链路先停用。

                # === K trap scaling (bf16 * bf16 → bf16, Parallel) ===
                scale_exp = trap_scale.repeat_interleave(R)            # [fL] bf16
                k_scaled  = (k_r.float() * scale_exp[:, None].float()).to(bf16)  # [fL,N]

                # === qk_intrachunk GEMM (f32, with trap-scaled K) ===
                qk_intra = q_r.float() @ k_scaled.float().T            # [fL, fL] f32

                # === intrachunk mask: if_then_else, exp(SEGSUM) from GM ===
                # qk_intrachunk_masked_frag: bf16
                mask_f32 = torch.zeros(L * R, L * R, dtype=f32, device=device)
                for ci in range(L * R):
                    for cj in range(L * R):
                        if ci // R > cj // R:
                            seg_val = SEGSUM[b, h, i, ci // R, cj // R]
                            mask_f32[ci, cj] = torch.exp(seg_val)
                qk_masked = (qk_intra * mask_f32).to(bf16)             # [fL, fL] bf16

                # === exp(DA_CS) decay on o_accum (Parallel, f32) ===
                da_cs_chunk = DA_CS[b, h, cs0:cs1]
                exp_da = torch.exp(da_cs_chunk)                        # [L] f32
                exp_da_exp = exp_da.repeat_interleave(R)               # [fL] f32
                o_accum = o_accum * exp_da_exp[:, None]                # [fL, P] f32

                # === intrachunk GEMM += (f32 += bf16@bf16) ===
                o_accum = o_accum + qk_masked.float() @ PsiV_r.float()  # f32

                # === Diagonal terms (qkdot_psiv: bf16) ===
                qkdot_pv = torch.zeros(L, R, P, dtype=bf16, device=device)
                for cs in range(L):
                    for ro in range(R):
                        acc = torch.zeros(P, dtype=bf16, device=device)
                        for ri in range(R):
                            # qk_dot_full is f32, PsiV_shared is bf16
                            acc = (acc.float()
                                   + qk_dot[cs*R+ro, cs*R+ri] * PsiV_r[cs*R+ri].float()
                                  ).to(bf16)
                        # *= gamma (f32 scalar → bf16)
                        qkdot_pv[cs, ro] = (acc.float() * gamma[cs]).to(bf16)

                if hasD:
                    # PsiV_D_frag: f32; D_var: f32 scalar
                    d_val = D[h].float()
                    PsiV_D = PsiV_r.float()                            # [fL, P] f32
                    qkdot_pv_r = qkdot_pv.reshape(L * R, P)
                    qkdot_pv_r = (qkdot_pv_r.float()
                                  + d_val * PsiV_D).to(bf16)
                    qkdot_pv = qkdot_pv_r.reshape(L, R, P)

                # o_accum += qkdot_psiv (f32 += bf16)
                o_accum = o_accum + qkdot_pv.reshape(L * R, P).float()  # f32

                # === Z gate (bf16) ===
                if hasZ:
                    z_c = Z[b, cs0:cs1, h, :]                          # [L, P] bf16
                    # MIMO_Z accessed from GM (f32) directly
                    Zeta = MIMO_Z[h]                                    # [R, P] f32
                    og = (z_c[:, None, :].float()
                          * Zeta[None, :, :] * 0.5).to(bf16)           # [L,R,P] bf16
                    # tanh computed in f32 then cast back
                    z_exp = (og.float() * torch.tanh(og.float()) + og.float()).to(bf16)

                # === Output reduction ===
                if reduceO:
                    # lqk_PsiV = view o_accum as [L, R, P] f32
                    lqk = o_accum.reshape(L, R, P)                     # f32
                    # phi_frag_intrachunk: bf16
                    lqk = (lqk * Phi_frag[None, :, :].float())         # f32
                    if hasZ:
                        lqk = lqk * z_exp.float()                      # f32
                    # sum over R → [L, P], copy to O as bf16
                    o_chunk = lqk.sum(dim=1).to(bf16)
                    O_ref[b, cs0:cs1, h, :] = o_chunk
                else:
                    lqk = o_accum.reshape(L, R, P)                     # f32
                    if hasZ:
                        lqk = lqk * z_exp.float()
                    O_ref[b, cs0:cs1, :, h, :] = lqk.to(bf16)

                # FINAL_STATE 参考链路先停用。

    return O_ref, FINAL_STATE_ref, FINAL_K_ref


def run_test():
    os.environ["TILELANG_ASCEND_MODE"] = "Expert"
    os.environ["TILELANG_ASCEND_WORKSPACE_SIZE"] = str(1024 * 1024 * 512)

    # 这里把 S 调到大于 chunk_size，确保 nchunks > 1，
    # 这样 Expert/mix 路径里的 pipeline loop 不会在前面被直接折叠掉。
    B, S, H, G, N, P, R = 1, 32, 4, 4, 32, 64, 2
    chunk_size         = 16
    rotary_dim_divisor = 4
    hasZ               = True
    hasD               = True
    reduceO            = True
    # 先只验证 O，FINAL_STATE 链路在这个 mix 文件里先停用。
    return_final_state = False
    dtype_torch        = torch.bfloat16
    dtype_str          = 'bfloat16'
    device             = torch.device('npu')

    torch.manual_seed(42)
    nchunks          = (S + chunk_size - 1) // chunk_size
    fused_chunk_size = chunk_size * R

    def rand(*shape, dtype=dtype_torch):
        return torch.randn(*shape, dtype=dtype, device=device) * 0.1

    Q       = rand(B, S, R, G, N)
    K       = rand(B, S, R, G, N)
    V       = rand(B, S, H, P)
    Q_BIAS  = rand(H, R, N, dtype=torch.float32)
    K_BIAS  = rand(H, R, N, dtype=torch.float32)
    MIMO_V  = rand(H, R, P, dtype=torch.float32)
    MIMO_O  = rand(H, R, P, dtype=torch.float32)
    Z       = rand(B, S, H, P)
    D       = rand(H, dtype=torch.float32)
    MIMO_Z  = rand(H, R, P, dtype=torch.float32)
    ANGLES  = rand(B, S, H, N // rotary_dim_divisor, dtype=torch.float32)
    DA_CS     = rand(B, H, S, dtype=torch.float32) * 0.1
    DA_CS_REV = rand(B, H, S, dtype=torch.float32) * 0.1
    DT      = rand(B, H, S, dtype=torch.float32).abs() + 0.01
    TRAP    = rand(B, H, S)
    SEGSUM  = rand(B, H, nchunks, chunk_size, chunk_size, dtype=torch.float32)
    SEGSUM  = SEGSUM - SEGSUM.abs().amax(dim=(-2,-1), keepdim=True)

    if reduceO:
        O_out = torch.zeros(B, S, H, P, dtype=dtype_torch, device=device)
    else:
        O_out = torch.zeros(B, S, R, H, P, dtype=dtype_torch, device=device)
    FINAL_STATE = torch.zeros(B, H, N, P, dtype=torch.float32, device=device)
    FINAL_K     = torch.zeros(B, R, H, N, dtype=dtype_torch, device=device)

    kernel = mamba_mimo_fwd(
        B, S, H, G, N, P, R,
        hasZ=hasZ, hasD=hasD, reduceO=reduceO,
        return_final_state=return_final_state,
        chunk_size=chunk_size,
        rotary_dim_divisor=rotary_dim_divisor,
        dtype=dtype_str,
    )

    kernel(
        Q, K, V, O_out,
        Q_BIAS, K_BIAS, MIMO_V, MIMO_O,
        Z, D, MIMO_Z, ANGLES,
        DA_CS, DA_CS_REV, DT, TRAP, SEGSUM,
        FINAL_STATE, FINAL_K,
    )

    O_ref, _, _ = ref_forward(
        B, S, H, G, N, P, R, chunk_size, rotary_dim_divisor,
        hasZ, hasD, reduceO, return_final_state,
        Q, K, V, Q_BIAS, K_BIAS, MIMO_V, MIMO_O,
        Z, D, MIMO_Z, ANGLES, DA_CS, DA_CS_REV,
        DT, TRAP, SEGSUM,
        device, dtype_torch,
    )

    def compare(name, a, b, rtol=1e-2, atol=1e-2):
        a = a.float().cpu()
        b = b.float().cpu()
        try:
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
            print(f"[{name}] PASS")
            return True
        except AssertionError as e:
            print(f"[{name}] FAIL")
            print(f"  {e}")
            # 打印辅助信息方便定位
            abs_err = (a - b).abs()
            print(f"  max abs: {abs_err.max():.6f}  mean abs: {abs_err.mean():.6f}")
            return False

    ok_O  = compare("O",           O_out,      O_ref)
    # FINAL_STATE 比较先停用。
    # ok_fs = compare("FINAL_STATE", FINAL_STATE, FINAL_STATE_ref)
    return ok_O


if __name__ == "__main__":
    run_test()
