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

    nchunks = tilelang.cdiv(S, chunk_size)
    tail_len = S % chunk_size
    fused_chunk_size = chunk_size * R

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
        
        with T.Kernel(H, B) as (i_h, i_b):
            # --- Kernel Setup ---
            # GQA support: map V head to Q/K head
            i_h_qk = i_h // (H // G)

            # --- Buffer Allocation ---
            q_shared = T.alloc_shared([fused_chunk_size, N], dtype)
            k_shared = T.alloc_shared([fused_chunk_size, N], dtype)
            q_bias_frag = T.alloc_fragment([R, N], dtype)
            k_bias_frag = T.alloc_fragment([R, N], dtype)

            angles_shared = T.alloc_shared([chunk_size, N], dtype)

            PsiV_shared = T.alloc_shared([fused_chunk_size, P], dtype)
            qs_shared = T.alloc_shared([fused_chunk_size, P], dtype)
            o_shared = T.alloc_shared([chunk_size, P], dtype)
            v_shared = T.alloc_shared([chunk_size, P], dtype)
            states_accum_cast_shared = T.alloc_shared([N, P], dtype)
            qk_intrachunk_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], dtype)
            qk_dot_full_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], dtype)

            # --- Swizzling Annotation ---
            T.annotate_layout({
                q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                v_shared: tilelang.layout.make_swizzled_layout(v_shared),

                angles_shared: tilelang.layout.make_swizzled_layout(angles_shared),

                PsiV_shared: tilelang.layout.make_swizzled_layout(PsiV_shared),
                qs_shared: tilelang.layout.make_swizzled_layout(qs_shared),
                o_shared: tilelang.layout.make_swizzled_layout(o_shared),
                states_accum_cast_shared: tilelang.layout.make_swizzled_layout(states_accum_cast_shared),
                qk_dot_full_shared: tilelang.layout.make_swizzled_layout(qk_dot_full_shared),
                qk_intrachunk_shared: tilelang.layout.make_swizzled_layout(qk_intrachunk_shared),
            })
            T.use_swizzle(10, "row")

            # --- Per-Head Constants / Running State ---
            states_frag = T.alloc_fragment([N, P], accum_dtype)
            T.clear(states_frag)

            phi_frag_intrachunk = T.alloc_fragment([R, P], dtype=dtype)
            if reduceO:
                T.copy(MIMO_O[i_h, :, :], phi_frag_intrachunk)
            Psi_frag = T.alloc_fragment([R, P], dtype)
            T.copy(MIMO_V[i_h, :, :], Psi_frag)

            T.copy(Q_BIAS[i_h, :, :], q_bias_frag)
            T.copy(K_BIAS[i_h, :, :], k_bias_frag)

            # --- Chunk Loop ---
            for i in T.Pipelined(0, nchunks, num_stages=num_stages):
                chunk_start = i * chunk_size

                segsum = T.alloc_fragment([chunk_size, chunk_size], "float32")
                T.copy(SEGSUM[i_b, i_h, i, :, :], segsum)

                # --- Discretization Factors (Shifted Gamma + Trap Scale) ---
                trap_shifted_frag = T.alloc_fragment([chunk_size], "float32")
                T.copy(TRAP[i_b, i_h, chunk_start+1: chunk_start+chunk_size+1], trap_shifted_frag)
                T.vmul(trap_shifted_frag, -1, trap_shifted_frag)
                T.vsigmoid(trap_shifted_frag, trap_shifted_frag)
                dt_shifted_frag = T.alloc_fragment([chunk_size], dtype)
                T.copy(DT[i_b, i_h, chunk_start+1: chunk_start+chunk_size+1], dt_shifted_frag)
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
                T.copy(TRAP[i_b, i_h, chunk_start: chunk_start+chunk_size], trap_frag)
                T.vsigmoid(trap_frag, trap_frag)
                dt_frag = T.alloc_fragment([chunk_size], dtype)
                T.copy(DT[i_b, i_h, chunk_start: chunk_start+chunk_size], dt_frag)
                gamma_frag = T.alloc_fragment([chunk_size], "float32")
                # 这里paraller还不支持
                for cs in T.serial(chunk_size):
                    gamma_frag[cs] = dt_frag[cs] * trap_frag[cs]
                trap_scale_frag = T.alloc_fragment([chunk_size], dtype)
                for cs in T.Parallel(chunk_size):
                    trap_scale_frag[cs] = gamma_frag[cs] + shifted_gamma_shared[cs]
                trap_scale_shared = T.alloc_shared([chunk_size], dtype)
                T.copy(trap_scale_frag, trap_scale_shared)

                # --- Up-Project V and Prepare Biased Q/K ---
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
                # Keep full qk_dot in shared memory because we reuse same-step R x R blocks later.
                qk_dot_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype=accum_dtype)
                T.gemm(q_shared, k_shared, qk_dot_frag, b_transpose=True, initC=True)
                T.copy(qk_dot_frag, qk_dot_full_shared)
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

                # --- Rotary Q + Interchunk Contribution ---
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
                T.copy(states_frag, states_accum_cast_shared)
                T.gemm(q_shared, states_accum_cast_shared, o_mimo_accum_frag, initC=True)

                # --- Rotary K + Trap Scaling + Intrachunk Contribution ---
                k_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                k_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)

                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    k_first_half_frag[cs, r, n] = k_shared[cs*R + r, n]
                    k_second_half_frag[cs, r, n] = k_shared[cs*R + r, N//2 + n]
                
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    k_shared[cs*R + r, n] = angles_frag_cos[cs, n] * k_first_half_frag[cs, r, n] - angles_frag_sin[cs, n] * k_second_half_frag[cs, r, n]
                    k_shared[cs*R + r, N//2 + n] = angles_frag_sin[cs, n] * k_first_half_frag[cs, r, n] + angles_frag_cos[cs, n] * k_second_half_frag[cs, r, n]

                if i == nchunks - 1 and return_final_state:
                    seq_boundary = T.min(chunk_start + chunk_size, S) - chunk_start
                    for csr, n in T.Parallel(fused_chunk_size, N):
                        if csr >= (seq_boundary - 1) * R and csr < seq_boundary * R:  # Only copy the last chunk's R rows to FINAL_K
                            # FINAL_K[i_b, csr % R, i_h, n] = k_shared[csr, n]
                            T.copy(k_shared[csr, n], FINAL_K[i_b, csr % R, i_h, n], size=[1,1])

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

                qk_intrachunk_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype=accum_dtype)
                T.gemm(q_shared, k_shared, qk_intrachunk_frag, b_transpose=True, initC=True)

                # Strictly causal mask over chunk steps (exclude same-step diagonal).
                da_cs__or__exp_da_cs_shared = T.alloc_shared([chunk_size], "float32")
                T.copy(DA_CS[i_b, i_h, chunk_start:chunk_start+chunk_size], da_cs__or__exp_da_cs_shared)
                qk_intrachunk_masked_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype=dtype)
                T.clear(qk_intrachunk_masked_frag)
                T.vexp(segsum, segsum)
                # for csr_i, csr_j in T.Parallel(fused_chunk_size, fused_chunk_size):
                #     qk_intrachunk_masked_frag[csr_i, csr_j] = T.if_then_else(
                #                                 csr_i//R > csr_j//R, # NOTE: we do indeed want to exclude the diagonal
                #                                 qk_intrachunk_frag[csr_i, csr_j] 
                #                                 * T.exp(SEGSUM[i_b, i_h, i, csr_i//R, csr_j//R]),
                #                                 0.0
                #                             )
                for csr_i in T.serial(fused_chunk_size):
                    for csr_j in T.serial(fused_chunk_size):
                        if csr_i//R > csr_j//R:
                            qk_intrachunk_masked_frag[csr_i, csr_j] = qk_intrachunk_frag[csr_i, csr_j]  * segsum[csr_i//R, csr_j//R]

                # Exponentiate da_cs__or__exp_da_cs_shared so that later usage does not have to:
                # for cs in T.Parallel(chunk_size):
                #     da_cs__or__exp_da_cs_shared[cs] = T.exp(da_cs__or__exp_da_cs_shared[cs])
                T.vexp(da_cs__or__exp_da_cs_shared, da_cs__or__exp_da_cs_shared)

                exp_da_cs_frag = T.alloc_fragment([chunk_size], dtype="float32")
                T.copy(da_cs__or__exp_da_cs_shared, exp_da_cs_frag)
                # for csr, p in T.Parallel(fused_chunk_size, P):
                #     o_mimo_accum_frag[csr, p] *= exp_da_cs_frag[csr//R]
                for csr in T.serial(fused_chunk_size):
                    for p in T.serial(P):
                        o_mimo_accum_frag[csr, p] *= exp_da_cs_frag[csr//R]


                T.copy(qk_intrachunk_masked_frag, qk_intrachunk_shared)
                # T.gemm(qk_intrachunk_shared, PsiV_shared, o_mimo_accum_frag, initC=False)

                tmp = T.alloc_fragment([fused_chunk_size, P], dtype=accum_dtype)
                T.gemm(qk_intrachunk_shared, PsiV_shared, tmp, initC=True)
                T.vadd(o_mimo_accum_frag, tmp, o_mimo_accum_frag)

                # --- Add Diagonal Terms (qk_dot and optional D) ---
                qkdot_psiv_frag = T.alloc_fragment([chunk_size, R, P], dtype=dtype)
                T.clear(qkdot_psiv_frag)
                for cs, r_out, p in T.Parallel(chunk_size, R, P):
                    for r_in in T.serial(R):
                        qkdot_psiv_frag[cs, r_out, p] += qk_dot_full_shared[cs * R + r_out, cs * R + r_in] * PsiV_shared[cs * R + r_in, p]                    
                    qkdot_psiv_frag[cs, r_out, p] *= gamma_frag[cs] # Apply shifted gamma

                if hasD:
                    PsiV_D_frag = T.alloc_fragment([chunk_size, R, P], "float32")
                    for cs, r, p in T.Parallel(chunk_size, R, P):
                        PsiV_D_frag[cs, r, p] = PsiV_shared[cs * R + r, p]
                    for cs, r_out, p in T.Parallel(chunk_size, R, P):
                        qkdot_psiv_frag[cs, r_out, p] += D[i_h] * PsiV_D_frag[cs, r_out, p]
                qkdot_psiv_reshaped_frag = T.alloc_fragment([fused_chunk_size, P], dtype=dtype)
                T.reshape(qkdot_psiv_frag, qkdot_psiv_reshaped_frag)
                # qkdot_psiv_reshaped_frag = T.view(qkdot_psiv_frag, shape=[fused_chunk_size, P])
                for csr, p in T.Parallel(fused_chunk_size, P):
                    o_mimo_accum_frag[csr, p] += qkdot_psiv_reshaped_frag[csr, p]

                # --- Optional Z Gating + Down-Projection ---
                if reduceO:
                    lqk_PsiV_reshaped_frag = T.alloc_fragment([chunk_size, R, P], dtype=accum_dtype)
                    T.reshape(o_mimo_accum_frag, lqk_PsiV_reshaped_frag)
                    if hasZ:
                        z_frag = T.alloc_fragment([chunk_size, P], dtype)
                        T.copy(Z[i_b, chunk_start:chunk_start+chunk_size, i_h, :], z_frag)
                        z_expanded_frag = T.alloc_fragment([chunk_size, R, P], dtype)
                        o_gated = T.alloc_fragment([chunk_size, R, P], dtype)
                        o_gated_tanh_cast = T.alloc_fragment([chunk_size, R, P], accum_dtype)
                        o_gated_tanh = T.alloc_fragment([chunk_size, R, P], accum_dtype)
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            # Apply SiLU to z_expanded_frag[cs, r, p]:
                            o_gated[cs, r, p] = z_frag[cs, p] * MIMO_Z[i_h, r, p] * 0.5
                        T.vcast(o_gated, o_gated_tanh_cast)
                        T.vtanh(o_gated_tanh_cast, o_gated_tanh_cast)
                        T.vcast(o_gated_tanh_cast, o_gated_tanh)
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            o_gated_val = o_gated[cs, r, p]
                            z_expanded_frag[cs, r, p] = o_gated_val * o_gated_tanh[cs, r, p] + o_gated_val

                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            lqk_PsiV_reshaped_frag[cs, r, p] *= phi_frag_intrachunk[r, p] * z_expanded_frag[cs, r, p]
                    else:
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            lqk_PsiV_reshaped_frag[cs, r, p] *= phi_frag_intrachunk[r, p]
                    lqk_PsiV_reshaped_shared = T.alloc_shared([chunk_size, R, P], dtype)
                    T.copy(lqk_PsiV_reshaped_frag, lqk_PsiV_reshaped_shared)
                    o_frag = T.alloc_fragment([1, chunk_size, P], dtype)
                    T.clear(o_frag)
                    for r in T.serial(R):
                        for cs, p in T.Parallel(chunk_size, P):
                            o_frag[0, cs, p] += lqk_PsiV_reshaped_shared[cs, r, p]
                    T.copy(o_frag, O[i_b, chunk_start:chunk_start+chunk_size, i_h, :])
                else:
                    if hasZ:
                        z_frag = T.alloc_fragment([chunk_size, P], dtype)
                        T.copy(Z[i_b, chunk_start:chunk_start+chunk_size, i_h, :], z_frag)
                        z_expanded_frag = T.alloc_fragment([chunk_size, R, P], dtype)
                        o_gated = T.alloc_fragment([chunk_size, R, P], dtype)
                        o_gated_tanh_cast = T.alloc_fragment([chunk_size, R, P], accum_dtype)
                        o_gated_tanh = T.alloc_fragment([chunk_size, R, P], accum_dtype)
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            # Apply SiLU to z_expanded_frag[cs, r, p]:
                            o_gated[cs, r, p] = z_frag[cs, p] * MIMO_Z[i_h, r, p] * 0.5
                        T.vcast(o_gated, o_gated_tanh_cast)
                        T.vtanh(o_gated_tanh_cast, o_gated_tanh_cast)
                        T.vcast(o_gated_tanh_cast, o_gated_tanh)
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            o_gated_val = o_gated[cs, r, p]
                            z_expanded_frag[cs, r, p,] = o_gated_val * o_gated_tanh[cs, r, p] + o_gated_val
                        lqk_PsiV_reshaped_shared = T.alloc_shared([chunk_size, R, P], dtype)
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            lqk_PsiV_reshaped_shared[cs, r, p] = o_mimo_accum_frag[cs* R + r, p] * z_expanded_frag[cs, r, p]
                        # T.copy(lqk_PsiV_frag, lqk_PsiV_reshaped_shared)
                        # for cs, r, p in T.Parallel(chunk_size, R, P):
                        #     lqk_PsiV_reshaped_shared[cs, r, p] *= z_expanded_frag[cs, r, p]
                        T.copy(lqk_PsiV_reshaped_shared, O[i_b, chunk_start:chunk_start+chunk_size, :, i_h, :])
                    else:
                        lqk_PsiV_reshaped_shared = T.alloc_shared([chunk_size, R, P], dtype)
                        # T.copy(lqk_PsiV_reshaped_frag, lqk_PsiV_reshaped_shared)
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            lqk_PsiV_reshaped_shared[cs, r, p] = o_mimo_accum_frag[cs* R + r, p]
                        T.copy(lqk_PsiV_reshaped_shared, O[i_b, chunk_start:chunk_start+chunk_size, :, i_h, :])

                # --- Recurrent State Update ---
                # DA_CS_REV scales per-step K contributions for state accumulation.
                dA_cs_rev_frag = T.alloc_fragment([chunk_size], "float32")
                T.copy(DA_CS_REV[i_b, i_h, chunk_start:chunk_start+chunk_size], dA_cs_rev_frag)

                k_state_frag = T.alloc_fragment([fused_chunk_size, N], dtype)
                T.copy(k_shared, k_state_frag)
                for csr, n in T.Parallel(fused_chunk_size, N):
                    k_state_frag[csr, n] *= T.exp(dA_cs_rev_frag[csr//R])

                # DA_CS(last) applies the chunk-level decay to the carried state.
                da_cs_sum = T.alloc_fragment([1,1], "float32")
                if tail_len > 0 and i == nchunks - 1:
                    T.copy(DA_CS[i_b, i_h, S - 1], da_cs_sum)
                    if return_final_state:
                        for csr, n in T.Parallel(fused_chunk_size, N):
                            k_state_frag[csr, n] = T.if_then_else(csr < tail_len * R, k_state_frag[csr, n], 0.0)
                else:
                    T.copy(DA_CS[i_b, i_h, chunk_start+chunk_size-1], da_cs_sum)
                T.vexp(da_cs_sum, da_cs_sum)
                for n, p in T.Parallel(N, P):
                    states_frag[n, p] *= da_cs_sum[0, 0]
                k_state_frag_tmp = T.alloc_fragment([fused_chunk_size, N], "float32")
                T.vcast(k_state_frag, k_state_frag_tmp)
                T.vadd(k_state_frag_tmp, 0, k_state_frag_tmp)
                T.vcast(k_state_frag_tmp, k_state_frag)
                T.gemm(k_state_frag, PsiV_shared, states_frag, a_transpose=True, initC=False)
            
            # --- Save Last State (if applicable) ---
            if return_final_state:
                T.copy(states_frag, FINAL_STATE[i_b, i_h, :, :])

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
    FINAL_STATE_ref = torch.zeros(B, H, N, P, dtype=f32, device=device)
    FINAL_K_ref     = torch.zeros(B, R, H, N, dtype=bf16, device=device)

    for b in range(B):
        for h in range(H):
            h_qk = h // (H // G)

            # --- Per-head constants loaded into bf16 fragments ---
            Psi_frag  = MIMO_V[h].to(bf16)          # [R, P] bf16
            Phi_frag  = MIMO_O[h].to(bf16)          # [R, P] bf16
            q_bias    = Q_BIAS[h].to(bf16)          # [R, N] bf16
            k_bias    = K_BIAS[h].to(bf16)          # [R, N] bf16

            # states_frag: f32, shape [N, P]
            states = torch.zeros(N, P, dtype=f32, device=device)

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
                # states_accum_cast_shared: bf16 copy of states
                states_cast = states.to(bf16)   # [N, P] bf16
                o_accum = q_r.float() @ states_cast.float()  # [fL, P] f32

                # === Rotary on K ===
                k_rot = apply_rotary_bf16(k)    # [L, R, N] bf16
                k_r   = k_rot.reshape(L * R, N) # [fL, N] bf16

                # === FINAL_K (last chunk only) ===
                if i == nchunks - 1 and return_final_state:
                    seq_boundary = min(cs0 + chunk_size, S) - cs0
                    for csr in range(fused_chunk_size):
                        if (seq_boundary - 1) * R <= csr < seq_boundary * R:
                            r_idx = csr % R
                            FINAL_K_ref[b, r_idx, h, :] = k_r[csr]

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

                # === State update ===
                # k_state_frag: bf16; *= exp(dA_cs_rev) (f32)
                da_rev    = DA_CS_REV[b, h, cs0:cs1]                  # [L] f32
                exp_rev   = torch.exp(da_rev).repeat_interleave(R)    # [fL] f32
                k_state   = (k_scaled.float() * exp_rev[:, None]).to(bf16)  # [fL,N] bf16

                # tail masking
                if tail_len > 0 and i == nchunks - 1:
                    da_last = DA_CS[b, h, S - 1]
                    if return_final_state:
                        for csr in range(fused_chunk_size):
                            if csr >= tail_len * R:
                                k_state[csr] = 0.0
                else:
                    da_last = DA_CS[b, h, cs0 + chunk_size - 1]

                # states *= exp(da_cs_sum) (f32 scalar)
                states = states * torch.exp(da_last)

                # GEMM: k_state^T @ PsiV (bf16@bf16 → f32 accumulate)
                states = states + k_state.float().T @ PsiV_r.float()

            if return_final_state:
                FINAL_STATE_ref[b, h] = states

    return O_ref, FINAL_STATE_ref, FINAL_K_ref


def run_test():
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"

    B, S, H, G, N, P, R = 1, 16, 4, 4, 32, 64, 2
    chunk_size         = 16
    rotary_dim_divisor = 4
    hasZ               = True
    hasD               = True
    reduceO            = True
    return_final_state = True
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

    O_ref, FINAL_STATE_ref, _ = ref_forward(
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
    ok_fs = compare("FINAL_STATE", FINAL_STATE, FINAL_STATE_ref)
    return ok_O and ok_fs


if __name__ == "__main__":
    run_test()