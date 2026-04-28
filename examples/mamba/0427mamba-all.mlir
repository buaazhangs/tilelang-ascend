2026-04-27 19:50:14  [TileLang:tilelang.env:WARNING]: Loading tilelang libs from dev root: /home/z00910011/tilelang-ascend/build
====== TVM IR ======
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def mamba_mimo_fwd_kernel(Q: T.Buffer((1, 32, 2, 4, 32), "bfloat16"), K: T.Buffer((1, 32, 2, 4, 32), "bfloat16"), V: T.Buffer((1, 32, 4, 64), "bfloat16"), O: T.Buffer((1, 32, 4, 64), "bfloat16"), Q_BIAS: T.Buffer((4, 2, 32), "float32"), K_BIAS: T.Buffer((4, 2, 32), "float32"), MIMO_V: T.Buffer((4, 2, 64), "float32"), MIMO_O: T.Buffer((4, 2, 64), "float32"), Z: T.Buffer((1, 32, 4, 64), "bfloat16"), D: T.Buffer((4,), "float32"), MIMO_Z: T.Buffer((4, 2, 64), "float32"), ANGLES: T.Buffer((1, 32, 4, 8), "float32"), DA_CS: T.Buffer((1, 4, 32), "float32"), DA_CS_REV: T.Buffer((1, 4, 32), "float32"), DT: T.Buffer((1, 4, 33), "float32"), TRAP: T.Buffer((1, 4, 33), "bfloat16"), SEGSUM: T.Buffer((1, 4, 2, 16, 16), "float32"), FINAL_STATE: T.Buffer((1, 4, 32, 64), "float32"), FINAL_K: T.Buffer((1, 2, 4, 32), "bfloat16")):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "stackvm", "tag": ""}, "keys": [], "kind": "npuir", "tag": ""})})
        cid = T.launch_thread("blockIdx.x", 4)
        vid = T.launch_thread("blockIdx.y", 8)
        qk_dot_ws = T.decl_buffer((2, 32, 32), scope="global.workspace;multi_buffer=2")
        o_inter_ws = T.decl_buffer((2, 32, 64), scope="global.workspace;multi_buffer=2")
        qk_intrachunk_ws = T.decl_buffer((2, 32, 32), scope="global.workspace;multi_buffer=2")
        qk_masked_ws = T.decl_buffer((2, 32, 32), "bfloat16", scope="global.workspace;multi_buffer=2")
        o_intra_ws = T.decl_buffer((2, 32, 64), scope="global.workspace;multi_buffer=2")
        q_biased_ws = T.decl_buffer((2, 32, 32), "bfloat16", scope="global.workspace;multi_buffer=2")
        k_biased_ws = T.decl_buffer((2, 32, 32), "bfloat16", scope="global.workspace;multi_buffer=2")
        q_rot_ws = T.decl_buffer((2, 32, 32), "bfloat16", scope="global.workspace;multi_buffer=2")
        k_scaled_ws = T.decl_buffer((2, 32, 32), "bfloat16", scope="global.workspace;multi_buffer=2")
        states_cast_ws = T.decl_buffer((2, 32, 64), "bfloat16", scope="global.workspace;multi_buffer=2")
        psiv_ws = T.decl_buffer((2, 32, 64), "bfloat16", scope="global.workspace;multi_buffer=2")
        q_bias_frag = T.decl_buffer((2, 32), "bfloat16", scope="local.fragment")
        k_bias_frag = T.decl_buffer((2, 32), "bfloat16", scope="local.fragment")
        q_bias_f32_frag = T.decl_buffer((2, 32), scope="local.fragment")
        k_bias_f32_frag = T.decl_buffer((2, 32), scope="local.fragment")
        states_frag = T.decl_buffer((32, 64), scope="local.fragment")
        Psi_frag = T.decl_buffer((2, 64), "bfloat16", scope="local.fragment")
        Psi_frag_f32 = T.decl_buffer((2, 64), scope="local.fragment")
        T.attr(None, "threadblock_swizzle_pattern", "tl::rasterization2DRow<10>")
        T.npuir_brc(0, T.region(states_frag[0, 0], 2, 32, 64))
        T.copy(T.region(MIMO_V[cid, 0, 0], 1, 1, 2, 64), T.region(Psi_frag_f32[0, 0], 2, 2, 64))
        T.npuir_cast(T.region(Psi_frag_f32[0, 0], 1, 2, 64), T.region(Psi_frag[0, 0], 2, 2, 64), "rint")
        T.copy(T.region(Q_BIAS[cid, 0, 0], 1, 1, 2, 32), T.region(q_bias_f32_frag[0, 0], 2, 2, 32))
        T.npuir_cast(T.region(q_bias_f32_frag[0, 0], 1, 2, 32), T.region(q_bias_frag[0, 0], 2, 2, 32), "rint")
        T.copy(T.region(K_BIAS[cid, 0, 0], 1, 1, 2, 32), T.region(k_bias_f32_frag[0, 0], 2, 2, 32))
        T.npuir_cast(T.region(k_bias_f32_frag[0, 0], 1, 2, 32), T.region(k_bias_frag[0, 0], 2, 2, 32), "rint")
        for i in T.serial(2, annotations={"num_stages": 2}):
            q_shared = T.decl_buffer((32, 32), "bfloat16", scope="shared.flat")
            k_shared = T.decl_buffer((32, 32), "bfloat16", scope="shared.flat")
            PsiV_shared = T.decl_buffer((32, 64), "bfloat16", scope="shared.flat")
            v_shared = T.decl_buffer((16, 64), "bfloat16", scope="shared.flat")
            states_accum_cast_shared = T.decl_buffer((32, 64), "bfloat16", scope="shared.flat")
            qk_intrachunk_shared = T.decl_buffer((32, 32), "bfloat16", scope="shared.flat")
            segsum = T.decl_buffer((16, 16), scope="local.fragment")
            trap_shifted_frag = T.decl_buffer((16,), scope="local.fragment")
            trap_shifted_bf16 = T.decl_buffer((16,), "bfloat16", scope="local.fragment")
            dt_shifted_frag = T.decl_buffer((16,), "bfloat16", scope="local.fragment")
            dt_shifted_f32_frag = T.decl_buffer((16,), scope="local.fragment")
            shifted_gamma_frag = T.decl_buffer((16,), "bfloat16", scope="local.fragment")
            shifted_gamma_shared = T.decl_buffer((16,), "bfloat16", scope="shared.flat")
            trap_frag = T.decl_buffer((16,), scope="local.fragment")
            trap_bf16 = T.decl_buffer((16,), "bfloat16", scope="local.fragment")
            dt_frag = T.decl_buffer((16,), "bfloat16", scope="local.fragment")
            dt_f32_frag = T.decl_buffer((16,), scope="local.fragment")
            gamma_frag = T.decl_buffer((16,), scope="local.fragment")
            trap_scale_frag = T.decl_buffer((16,), "bfloat16", scope="local.fragment")
            trap_scale_shared = T.decl_buffer((16,), "bfloat16", scope="shared.flat")
            PsiV_frag = T.decl_buffer((16, 2, 64), "bfloat16", scope="local.fragment")
            PsiV_reshaped_frag = T.decl_buffer((32, 64), "bfloat16", scope="local.fragment")
            q_frag = T.decl_buffer((16, 2, 32), "bfloat16", scope="local.fragment")
            q_biased_flat = T.decl_buffer((32, 32), "bfloat16", scope="local.fragment")
            k_frag = T.decl_buffer((16, 2, 32), "bfloat16", scope="local.fragment")
            k_biased_flat = T.decl_buffer((32, 32), "bfloat16", scope="local.fragment")
            qk_dot_frag = T.decl_buffer((32, 32), scope="local.fragment")
            q_rot_frag = T.decl_buffer((32, 32), "bfloat16", scope="local.fragment")
            q_first_half_frag = T.decl_buffer((16, 2, 8), "bfloat16", scope="local.fragment")
            q_second_half_frag = T.decl_buffer((16, 2, 8), "bfloat16", scope="local.fragment")
            angles_frag = T.decl_buffer((16, 8), scope="local.fragment")
            angles_frag_cos = T.decl_buffer((16, 8), scope="local.fragment")
            angles_frag_sin = T.decl_buffer((16, 8), scope="local.fragment")
            o_mimo_accum_frag = T.decl_buffer((32, 64), scope="local.fragment")
            states_frag_cast = T.decl_buffer((32, 64), "bfloat16", scope="local.fragment")
            k_rot_frag = T.decl_buffer((32, 32), "bfloat16", scope="local.fragment")
            k_first_half_frag = T.decl_buffer((16, 2, 8), "bfloat16", scope="local.fragment")
            k_second_half_frag = T.decl_buffer((16, 2, 8), "bfloat16", scope="local.fragment")
            k_trap_scaled_frag = T.decl_buffer((32, 32), "bfloat16", scope="local.fragment")
            k_trap_scaled_frag_f32 = T.decl_buffer((32, 32), scope="local.fragment")
            qk_intrachunk_frag = T.decl_buffer((32, 32), scope="local.fragment")
            qk_intrachunk_vec = T.decl_buffer((16, 32), scope="shared.flat")
            qk_intrachunk_masked_vec = T.decl_buffer((16, 32), "bfloat16", scope="shared.flat")
            o_inter_vec = T.decl_buffer((16, 64), scope="shared.flat")
            da_cs_vec = T.decl_buffer((8,), scope="shared.flat")
            tmp = T.decl_buffer((32, 64), scope="local.fragment")
            o_inter_vec_1 = T.decl_buffer((16, 64), scope="shared.flat")
            o_intra_vec = T.decl_buffer((16, 64), scope="shared.flat")
            qk_dot_vec = T.decl_buffer((16, 32), scope="shared.flat")
            psiv_vec = T.decl_buffer((32, 64), "bfloat16", scope="shared.flat")
            z_vec = T.decl_buffer((8, 64), "bfloat16", scope="shared.flat")
            o_reduce_vec = T.decl_buffer((8, 64), "bfloat16", scope="shared.flat")
            local_src_buf = T.decl_buffer((16, 64), "bfloat16", scope="local.fragment")
            reshape_view_buf = T.decl_buffer((16, 1, 64), "bfloat16", scope="local.fragment")
            brc_buf = T.decl_buffer((16, 2, 64), "bfloat16", scope="local.fragment")
            local_src_buf_1 = T.decl_buffer((2, 64), "bfloat16", scope="local.fragment")
            reshape_view_buf_1 = T.decl_buffer((1, 2, 64), "bfloat16", scope="local.fragment")
            brc_buf_1 = T.decl_buffer((16, 2, 64), "bfloat16", scope="local.fragment")
            local_src_buf_2 = T.decl_buffer((2, 32), "bfloat16", scope="local.fragment")
            reshape_view_buf_2 = T.decl_buffer((1, 2, 32), "bfloat16", scope="local.fragment")
            brc_buf_2 = T.decl_buffer((16, 2, 32), "bfloat16", scope="local.fragment")
            local_src_buf_3 = T.decl_buffer((2, 32), "bfloat16", scope="local.fragment")
            reshape_view_buf_3 = T.decl_buffer((1, 2, 32), "bfloat16", scope="local.fragment")
            brc_buf_3 = T.decl_buffer((16, 2, 32), "bfloat16", scope="local.fragment")
            bf16_src_f32_0 = T.decl_buffer((16, 2, 64), scope="local.fragment")
            bf16_src_f32_1 = T.decl_buffer((16, 2, 64), scope="local.fragment")
            bf16_dst_f32_2 = T.decl_buffer((16, 2, 64), scope="local.fragment")
            bf16_src_f32_3 = T.decl_buffer((16, 2, 32), scope="local.fragment")
            bf16_src_f32_4 = T.decl_buffer((16, 2, 32), scope="local.fragment")
            bf16_dst_f32_5 = T.decl_buffer((16, 2, 32), scope="local.fragment")
            bf16_src_f32_6 = T.decl_buffer((16, 2, 32), scope="local.fragment")
            bf16_src_f32_7 = T.decl_buffer((16, 2, 32), scope="local.fragment")
            bf16_dst_f32_8 = T.decl_buffer((16, 2, 32), scope="local.fragment")
            T.copy(T.region(SEGSUM[0, cid, i, 0, 0], 1, 1, 1, 1, 16, 16), T.region(segsum[0, 0], 2, 16, 16))
            T.copy(T.region(TRAP[0, cid, i * 16 + 1], 1, 1, 1, 16), T.region(trap_shifted_bf16[0], 2, 16))
            T.npuir_cast(T.region(trap_shifted_bf16[0], 1, 16), T.region(trap_shifted_frag[0], 2, 16), "rint")
            T.npuir_mul(T.region(trap_shifted_frag[0], 1, 16), T.float32(-1), T.region(trap_shifted_frag[0], 2, 16))
            T.npuir_sigmoid(T.region(trap_shifted_frag[0], 1, 16), T.region(trap_shifted_frag[0], 2, 16))
            T.copy(T.region(DT[0, cid, i * 16 + 1], 1, 1, 1, 16), T.region(dt_shifted_f32_frag[0], 2, 16))
            T.npuir_cast(T.region(dt_shifted_f32_frag[0], 1, 16), T.region(dt_shifted_frag[0], 2, 16), "rint")
            T.npuir_brc(0, T.region(shifted_gamma_frag[0], 2, 16))
            for cs in range(16):
                if i * 16 + cs < 31:
                    shifted_gamma_frag[cs] = T.Cast("bfloat16", T.Cast("float32", dt_shifted_frag[cs]) * trap_shifted_frag[cs])
            T.copy(T.region(shifted_gamma_frag[0], 1, 16), T.region(shifted_gamma_shared[0], 2, 16))
            T.copy(T.region(TRAP[0, cid, i * 16], 1, 1, 1, 16), T.region(trap_bf16[0], 2, 16))
            T.npuir_cast(T.region(trap_bf16[0], 1, 16), T.region(trap_frag[0], 2, 16), "rint")
            T.npuir_sigmoid(T.region(trap_frag[0], 1, 16), T.region(trap_frag[0], 2, 16))
            T.copy(T.region(DT[0, cid, i * 16], 1, 1, 1, 16), T.region(dt_f32_frag[0], 2, 16))
            T.npuir_cast(T.region(dt_f32_frag[0], 1, 16), T.region(dt_frag[0], 2, 16), "rint")
            for cs in range(16):
                gamma_frag[cs] = T.Cast("float32", dt_frag[cs]) * trap_frag[cs]
            for cs in T.parallel(16):
                trap_scale_frag[cs] = T.Cast("bfloat16", gamma_frag[cs] + T.Cast("float32", shifted_gamma_shared[cs]))
            T.copy(T.region(trap_scale_frag[0], 1, 16), T.region(trap_scale_shared[0], 2, 16))
            for cs_split in T.parallel(16):
                T.copy(T.region(V[0, i * 16 + cs_split, cid, 0], 1, 1, 1, 1, 64), T.region(v_shared[cs_split, 0], 2, 1, 64))
            T.copy(T.region(v_shared[0, 0], 1, 16, 64), T.region(local_src_buf[0, 0], 2, 16, 64))
            T.npuir_reshape(T.region(local_src_buf[0, 0], 1, 16, 64), T.region(reshape_view_buf[0, 0, 0], 2, 16, 1, 64))
            T.npuir_brc(T.region(reshape_view_buf[0, 0, 0], 1, 16, 1, 64), T.region(brc_buf[0, 0, 0], 2, 16, 2, 64))
            T.copy(T.region(Psi_frag[0, 0], 1, 2, 64), T.region(local_src_buf_1[0, 0], 2, 2, 64))
            T.npuir_reshape(T.region(local_src_buf_1[0, 0], 1, 2, 64), T.region(reshape_view_buf_1[0, 0, 0], 2, 1, 2, 64))
            T.npuir_brc(T.region(reshape_view_buf_1[0, 0, 0], 1, 1, 2, 64), T.region(brc_buf_1[0, 0, 0], 2, 16, 2, 64))
            T.npuir_cast(T.region(brc_buf[0, 0, 0], 1, 16, 2, 64), T.region(bf16_src_f32_0[0, 0, 0], 2, 16, 2, 64), "rint")
            T.npuir_cast(T.region(brc_buf_1[0, 0, 0], 1, 16, 2, 64), T.region(bf16_src_f32_1[0, 0, 0], 2, 16, 2, 64), "rint")
            T.npuir_mul(T.region(bf16_src_f32_0[0, 0, 0], 1, 16, 2, 64), T.region(bf16_src_f32_1[0, 0, 0], 1, 16, 2, 64), T.region(bf16_dst_f32_2[0, 0, 0], 2, 16, 2, 64))
            T.npuir_cast(T.region(bf16_dst_f32_2[0, 0, 0], 1, 16, 2, 64), T.region(PsiV_frag[0, 0, 0], 2, 16, 2, 64), "rint")
            T.npuir_reshape(T.region(PsiV_frag[0, 0, 0], 2, 16, 2, 64), T.region(PsiV_reshaped_frag[0, 0], 1, 32, 64))
            T.copy(T.region(PsiV_reshaped_frag[0, 0], 1, 32, 64), T.region(psiv_ws[0, 0, 0], 2, 1, 32, 64))
            T.copy(T.region(Q[0, i * 16, 0, cid, 0], 1, 1, 16, 2, 1, 32), T.region(q_frag[0, 0, 0], 2, 16, 2, 32))
            T.copy(T.region(q_bias_frag[0, 0], 1, 2, 32), T.region(local_src_buf_2[0, 0], 2, 2, 32))
            T.npuir_reshape(T.region(local_src_buf_2[0, 0], 1, 2, 32), T.region(reshape_view_buf_2[0, 0, 0], 2, 1, 2, 32))
            T.npuir_brc(T.region(reshape_view_buf_2[0, 0, 0], 1, 1, 2, 32), T.region(brc_buf_2[0, 0, 0], 2, 16, 2, 32))
            T.npuir_cast(T.region(q_frag[0, 0, 0], 1, 16, 2, 32), T.region(bf16_src_f32_3[0, 0, 0], 2, 16, 2, 32), "rint")
            T.npuir_cast(T.region(brc_buf_2[0, 0, 0], 1, 16, 2, 32), T.region(bf16_src_f32_4[0, 0, 0], 2, 16, 2, 32), "rint")
            T.npuir_add(T.region(bf16_src_f32_3[0, 0, 0], 1, 16, 2, 32), T.region(bf16_src_f32_4[0, 0, 0], 1, 16, 2, 32), T.region(bf16_dst_f32_5[0, 0, 0], 2, 16, 2, 32))
            T.npuir_cast(T.region(bf16_dst_f32_5[0, 0, 0], 1, 16, 2, 32), T.region(q_frag[0, 0, 0], 2, 16, 2, 32), "rint")
            T.npuir_reshape(T.region(q_frag[0, 0, 0], 2, 16, 2, 32), T.region(q_biased_flat[0, 0], 1, 32, 32))
            T.copy(T.region(q_biased_flat[0, 0], 1, 32, 32), T.region(q_biased_ws[0, 0, 0], 2, 1, 32, 32))
            T.copy(T.region(K[0, i * 16, 0, cid, 0], 1, 1, 16, 2, 1, 32), T.region(k_frag[0, 0, 0], 2, 16, 2, 32))
            T.copy(T.region(k_bias_frag[0, 0], 1, 2, 32), T.region(local_src_buf_3[0, 0], 2, 2, 32))
            T.npuir_reshape(T.region(local_src_buf_3[0, 0], 1, 2, 32), T.region(reshape_view_buf_3[0, 0, 0], 2, 1, 2, 32))
            T.npuir_brc(T.region(reshape_view_buf_3[0, 0, 0], 1, 1, 2, 32), T.region(brc_buf_3[0, 0, 0], 2, 16, 2, 32))
            T.npuir_cast(T.region(k_frag[0, 0, 0], 1, 16, 2, 32), T.region(bf16_src_f32_6[0, 0, 0], 2, 16, 2, 32), "rint")
            T.npuir_cast(T.region(brc_buf_3[0, 0, 0], 1, 16, 2, 32), T.region(bf16_src_f32_7[0, 0, 0], 2, 16, 2, 32), "rint")
            T.npuir_add(T.region(bf16_src_f32_6[0, 0, 0], 1, 16, 2, 32), T.region(bf16_src_f32_7[0, 0, 0], 1, 16, 2, 32), T.region(bf16_dst_f32_8[0, 0, 0], 2, 16, 2, 32))
            T.npuir_cast(T.region(bf16_dst_f32_8[0, 0, 0], 1, 16, 2, 32), T.region(k_frag[0, 0, 0], 2, 16, 2, 32), "rint")
            T.npuir_reshape(T.region(k_frag[0, 0, 0], 2, 16, 2, 32), T.region(k_biased_flat[0, 0], 1, 32, 32))
            T.copy(T.region(k_biased_flat[0, 0], 1, 32, 32), T.region(k_biased_ws[0, 0, 0], 2, 1, 32, 32))
            T.copy(T.region(q_biased_ws[0, 0, 0], 1, 1, 32, 32), T.region(q_shared[0, 0], 2, 32, 32))
            T.copy(T.region(k_biased_ws[0, 0, 0], 1, 1, 32, 32), T.region(k_shared[0, 0], 2, 32, 32))
            T.npuir_dot(T.region(q_shared[0, 0], 1, 32, 32), T.region(k_shared[0, 0], 1, 32, 32), T.region(qk_dot_frag[0, 0], 3, 32, 32), T.bool(True), T.bool(False), T.bool(True))
            T.copy(T.region(qk_dot_frag[0, 0], 1, 32, 32), T.region(qk_dot_ws[0, 0, 0], 2, 1, 32, 32))
            T.copy(T.region(q_biased_ws[0, 0, 0], 1, 1, 32, 32), T.region(q_rot_frag[0, 0], 2, 32, 32))
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        q_first_half_frag[cs_split, r_split, n_split] = q_rot_frag[cs_split * 2 + r_split, n_split]
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        q_second_half_frag[cs_split, r_split, n_split] = q_rot_frag[cs_split * 2 + r_split, n_split + 16]
            T.copy(T.region(ANGLES[0, i * 16, cid, 0], 1, 1, 16, 1, 8), T.region(angles_frag[0, 0], 2, 16, 8))
            T.npuir_vcos(T.region(angles_frag_cos[0, 0], 2, 16, 8), T.region(angles_frag[0, 0], 1, 16, 8))
            T.npuir_vsin(T.region(angles_frag_sin[0, 0], 2, 16, 8), T.region(angles_frag[0, 0], 1, 16, 8))
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        q_rot_frag[cs_split * 2 + r_split, n_split] = T.Cast("bfloat16", angles_frag_cos[cs_split, n_split] * T.Cast("float32", q_first_half_frag[cs_split, r_split, n_split]) - angles_frag_sin[cs_split, n_split] * T.Cast("float32", q_second_half_frag[cs_split, r_split, n_split]))
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        q_rot_frag[cs_split * 2 + r_split, n_split + 16] = T.Cast("bfloat16", angles_frag_sin[cs_split, n_split] * T.Cast("float32", q_first_half_frag[cs_split, r_split, n_split]) + angles_frag_cos[cs_split, n_split] * T.Cast("float32", q_second_half_frag[cs_split, r_split, n_split]))
            T.copy(T.region(q_rot_frag[0, 0], 1, 32, 32), T.region(q_rot_ws[0, 0, 0], 2, 1, 32, 32))
            T.npuir_cast(T.region(states_frag[0, 0], 1, 32, 64), T.region(states_frag_cast[0, 0], 2, 32, 64), "rint")
            T.copy(T.region(states_frag_cast[0, 0], 1, 32, 64), T.region(states_cast_ws[0, 0, 0], 2, 1, 32, 64))
            T.copy(T.region(q_rot_ws[0, 0, 0], 1, 1, 32, 32), T.region(q_shared[0, 0], 2, 32, 32))
            T.copy(T.region(states_cast_ws[0, 0, 0], 1, 1, 32, 64), T.region(states_accum_cast_shared[0, 0], 2, 32, 64))
            T.npuir_dot(T.region(q_shared[0, 0], 1, 32, 32), T.region(states_accum_cast_shared[0, 0], 1, 32, 64), T.region(o_mimo_accum_frag[0, 0], 3, 32, 64), T.bool(True), T.bool(False), T.bool(False))
            T.copy(T.region(o_mimo_accum_frag[0, 0], 1, 32, 64), T.region(o_inter_ws[0, 0, 0], 2, 1, 32, 64))
            T.copy(T.region(k_biased_ws[0, 0, 0], 1, 1, 32, 32), T.region(k_rot_frag[0, 0], 2, 32, 32))
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        k_first_half_frag[cs_split, r_split, n_split] = k_rot_frag[cs_split * 2 + r_split, n_split]
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        k_second_half_frag[cs_split, r_split, n_split] = k_rot_frag[cs_split * 2 + r_split, n_split + 16]
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        k_rot_frag[cs_split * 2 + r_split, n_split] = T.Cast("bfloat16", angles_frag_cos[cs_split, n_split] * T.Cast("float32", k_first_half_frag[cs_split, r_split, n_split]) - angles_frag_sin[cs_split, n_split] * T.Cast("float32", k_second_half_frag[cs_split, r_split, n_split]))
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        k_rot_frag[cs_split * 2 + r_split, n_split + 16] = T.Cast("bfloat16", angles_frag_sin[cs_split, n_split] * T.Cast("float32", k_first_half_frag[cs_split, r_split, n_split]) + angles_frag_cos[cs_split, n_split] * T.Cast("float32", k_second_half_frag[cs_split, r_split, n_split]))
            T.copy(T.region(k_rot_frag[0, 0], 1, 32, 32), T.region(k_trap_scaled_frag[0, 0], 2, 32, 32))
            T.npuir_cast(T.region(k_trap_scaled_frag[0, 0], 1, 32, 32), T.region(k_trap_scaled_frag_f32[0, 0], 2, 32, 32), "rint")
            for csr, n in T.grid(32, 32):
                k_trap_scaled_frag_f32[csr, n] = k_trap_scaled_frag_f32[csr, n] * T.Cast("float32", trap_scale_shared[csr // 2])
            T.npuir_cast(T.region(k_trap_scaled_frag_f32[0, 0], 1, 32, 32), T.region(k_trap_scaled_frag[0, 0], 2, 32, 32), "rint")
            T.copy(T.region(k_trap_scaled_frag[0, 0], 1, 32, 32), T.region(k_scaled_ws[0, 0, 0], 2, 1, 32, 32))
            T.copy(T.region(q_rot_ws[0, 0, 0], 1, 1, 32, 32), T.region(q_shared[0, 0], 2, 32, 32))
            T.copy(T.region(k_scaled_ws[0, 0, 0], 1, 1, 32, 32), T.region(k_shared[0, 0], 2, 32, 32))
            T.npuir_dot(T.region(q_shared[0, 0], 1, 32, 32), T.region(k_shared[0, 0], 1, 32, 32), T.region(qk_intrachunk_frag[0, 0], 3, 32, 32), T.bool(True), T.bool(False), T.bool(True))
            T.copy(T.region(qk_intrachunk_frag[0, 0], 1, 32, 32), T.region(qk_intrachunk_ws[0, 0, 0], 2, 1, 32, 32))
            T.npuir_exp(T.region(segsum[0, 0], 1, 16, 16), T.region(segsum[0, 0], 2, 16, 16))
            if vid < 2:
                T.copy(T.region(qk_intrachunk_ws[0, vid * 16, 0], 1, 1, 16, 32), T.region(qk_intrachunk_vec[0, 0], 2, 16, 32))
                T.copy(T.region(o_inter_ws[0, vid * 16, 0], 1, 1, 16, 64), T.region(o_inter_vec[0, 0], 2, 16, 64))
                T.copy(T.region(DA_CS[0, cid, i * 16 + vid * 8], 1, 1, 1, 8), T.region(da_cs_vec[0], 2, 8))
                T.npuir_exp(T.region(da_cs_vec[0], 1, 8), T.region(da_cs_vec[0], 2, 8))
                for local_csr in range(16):
                    for csr_j in range(32):
                        if csr_j // 2 < vid * 8 + local_csr // 2:
                            qk_intrachunk_masked_vec[local_csr, csr_j] = T.Cast("bfloat16", qk_intrachunk_vec[local_csr, csr_j] * segsum[vid * 8 + local_csr // 2, csr_j // 2])
                        else:
                            qk_intrachunk_masked_vec[local_csr, csr_j] = T.bfloat16(0)
                    for p in range(64):
                        o_inter_vec[local_csr, p] = o_inter_vec[local_csr, p] * da_cs_vec[local_csr // 2]
                T.copy(T.region(qk_intrachunk_masked_vec[0, 0], 1, 16, 32), T.region(qk_masked_ws[0, vid * 16, 0], 2, 1, 16, 32))
                T.copy(T.region(o_inter_vec[0, 0], 1, 16, 64), T.region(o_inter_ws[0, vid * 16, 0], 2, 1, 16, 64))
            T.copy(T.region(qk_masked_ws[0, 0, 0], 1, 1, 32, 32), T.region(qk_intrachunk_shared[0, 0], 2, 32, 32))
            T.copy(T.region(psiv_ws[0, 0, 0], 1, 1, 32, 64), T.region(PsiV_shared[0, 0], 2, 32, 64))
            T.npuir_dot(T.region(qk_intrachunk_shared[0, 0], 1, 32, 32), T.region(PsiV_shared[0, 0], 1, 32, 64), T.region(tmp[0, 0], 3, 32, 64), T.bool(True), T.bool(False), T.bool(False))
            T.copy(T.region(tmp[0, 0], 1, 32, 64), T.region(o_intra_ws[0, 0, 0], 2, 1, 32, 64))
            if vid < 2:
                T.copy(T.region(o_inter_ws[0, vid * 16, 0], 1, 1, 16, 64), T.region(o_inter_vec_1[0, 0], 2, 16, 64))
                T.copy(T.region(o_intra_ws[0, vid * 16, 0], 1, 1, 16, 64), T.region(o_intra_vec[0, 0], 2, 16, 64))
                T.copy(T.region(qk_dot_ws[0, vid * 16, 0], 1, 1, 16, 32), T.region(qk_dot_vec[0, 0], 2, 16, 32))
                T.copy(T.region(psiv_ws[0, 0, 0], 1, 1, 32, 64), T.region(psiv_vec[0, 0], 2, 32, 64))
                T.copy(T.region(Z[0, i * 16 + vid * 8, cid, 0], 1, 1, 1, 8, 64), T.region(z_vec[0, 0], 2, 8, 64))
                for local_csr, p in T.grid(16, 64):
                    o_inter_vec_1[local_csr, p] = o_inter_vec_1[local_csr, p] + o_intra_vec[local_csr, p]
                for local_cs, p in T.grid(8, 64):
                    o_reduce_vec[local_cs, p] = T.bfloat16(0)
                T.copy(T.region(o_reduce_vec[0, 0], 1, 8, 64), T.region(O[0, i * 16 + vid * 8, cid, 0], 2, 1, 1, 8, 64))

====== npuir ======
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @mamba_mimo_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xbf16, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: memref<?xf32, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: memref<?xbf16, #hivm.address_space<gm>>, %arg12: memref<?xf32, #hivm.address_space<gm>>, %arg13: memref<?xf32, #hivm.address_space<gm>>, %arg14: memref<?xf32, #hivm.address_space<gm>>, %arg15: memref<?xf32, #hivm.address_space<gm>>, %arg16: memref<?xf32, #hivm.address_space<gm>>, %arg17: memref<?xf32, #hivm.address_space<gm>>, %arg18: memref<?xbf16, #hivm.address_space<gm>>, %arg19: memref<?xf32, #hivm.address_space<gm>>, %arg20: memref<?xf32, #hivm.address_space<gm>>, %arg21: memref<?xbf16, #hivm.address_space<gm>>, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c32_i32 = arith.constant 32 : i32
    %1 = arith.muli %c32_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %c4_i32 = arith.constant 4 : i32
    %3 = arith.muli %c4_i32, %1 : i32
    %4 = arith.index_cast %3 : i32 to index
    %c2_i32 = arith.constant 2 : i32
    %5 = arith.muli %c2_i32, %3 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.muli %c32_i32, %5 : i32
    %8 = arith.index_cast %7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%8, %6, %4, %2, %0] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %c64_i32 = arith.constant 64 : i32
    %9 = arith.muli %c64_i32, %c1_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %11 = arith.muli %c4_i32, %9 : i32
    %12 = arith.index_cast %11 : i32 to index
    %13 = arith.muli %c32_i32, %11 : i32
    %14 = arith.index_cast %13 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 32, 4, 64], strides: [%14, %12, %10, %0] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %15 = arith.muli %c2_i32, %9 : i32
    %16 = arith.index_cast %15 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [4, 2, 64], strides: [%16, %10, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>
    %17 = arith.muli %c2_i32, %1 : i32
    %18 = arith.index_cast %17 : i32 to index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [4, 2, 32], strides: [%18, %2, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %c8_i32 = arith.constant 8 : i32
    %19 = arith.muli %c8_i32, %c1_i32 : i32
    %20 = arith.index_cast %19 : i32 to index
    %21 = arith.muli %c4_i32, %19 : i32
    %22 = arith.index_cast %21 : i32 to index
    %23 = arith.muli %c32_i32, %21 : i32
    %24 = arith.index_cast %23 : i32 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg14 to offset: [0], sizes: [1, 32, 4, 8], strides: [%24, %22, %20, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg15 to offset: [0], sizes: [1, 4, 32], strides: [%4, %2, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg11 to offset: [0], sizes: [1, 32, 4, 64], strides: [%14, %12, %10, %0] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg21 to offset: [0], sizes: [1, 2, 4, 32], strides: [%6, %4, %2, %0] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x2x4x32xbf16, strided<[256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [4, 2, 32], strides: [%18, %2, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %c33_i32 = arith.constant 33 : i32
    %25 = arith.muli %c33_i32, %c1_i32 : i32
    %26 = arith.index_cast %25 : i32 to index
    %27 = arith.muli %c4_i32, %25 : i32
    %28 = arith.index_cast %27 : i32 to index
    %reinterpret_cast_8 = memref.reinterpret_cast %arg18 to offset: [0], sizes: [1, 4, 33], strides: [%28, %26, %0] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%8, %6, %4, %2, %0] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1, 32, 4, 64], strides: [%14, %12, %10, %0] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_11 = memref.reinterpret_cast %arg12 to offset: [0], sizes: [4], strides: [%0] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg16 to offset: [0], sizes: [1, 4, 32], strides: [%4, %2, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_13 = memref.reinterpret_cast %arg17 to offset: [0], sizes: [1, 4, 33], strides: [%28, %26, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %c16_i32 = arith.constant 16 : i32
    %29 = arith.muli %c16_i32, %c1_i32 : i32
    %30 = arith.index_cast %29 : i32 to index
    %31 = arith.muli %c16_i32, %29 : i32
    %32 = arith.index_cast %31 : i32 to index
    %33 = arith.muli %c2_i32, %31 : i32
    %34 = arith.index_cast %33 : i32 to index
    %35 = arith.muli %c4_i32, %33 : i32
    %36 = arith.index_cast %35 : i32 to index
    %reinterpret_cast_14 = memref.reinterpret_cast %arg19 to offset: [0], sizes: [1, 4, 2, 16, 16], strides: [%36, %34, %32, %30, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>>
    %37 = arith.muli %c32_i32, %9 : i32
    %38 = arith.index_cast %37 : i32 to index
    %39 = arith.muli %c4_i32, %37 : i32
    %40 = arith.index_cast %39 : i32 to index
    %reinterpret_cast_15 = memref.reinterpret_cast %arg20 to offset: [0], sizes: [1, 4, 32, 64], strides: [%40, %38, %10, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x32x64xf32, strided<[8192, 2048, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_16 = memref.reinterpret_cast %arg10 to offset: [0], sizes: [4, 2, 64], strides: [%16, %10, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_17 = memref.reinterpret_cast %arg13 to offset: [0], sizes: [4, 2, 64], strides: [%16, %10, %0] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>
    %41 = hivm.hir.get_block_idx -> i64
    %42 = arith.trunci %41 : i64 to i32
    %43 = hivm.hir.get_sub_block_idx -> i64
    %44 = arith.trunci %43 : i64 to i32
    %45 = memref_ext.alloc_workspace() : memref<2x32x32xf32>
    annotation.mark %45 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32>
    %46 = memref_ext.alloc_workspace() : memref<2x32x64xf32>
    annotation.mark %46 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32>
    %47 = memref_ext.alloc_workspace() : memref<2x32x32xf32>
    annotation.mark %47 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32>
    %48 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %48 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %49 = memref_ext.alloc_workspace() : memref<2x32x64xf32>
    annotation.mark %49 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32>
    %50 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %50 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %51 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %51 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %52 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %52 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %53 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %53 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %54 = memref_ext.alloc_workspace() : memref<2x32x64xbf16>
    annotation.mark %54 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16>
    %55 = memref_ext.alloc_workspace() : memref<2x32x64xbf16>
    annotation.mark %55 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16>
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_18 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_19 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_20 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_21 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
    %alloc_22 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
    %alloc_23 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>>
    %c0_i32 = arith.constant 0 : i32
    %56 = arith.sitofp %c0_i32 : i32 to f32
    hivm.hir.vbrc ins(%56 : f32) outs(%alloc_21 : memref<32x64xf32, strided<[64, 1]>>)
    %57 = arith.index_cast %42 : i32 to index
    %subview = memref.subview %reinterpret_cast_1[%57, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_23 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x64xf32, strided<[64, 1]>>) outs(%alloc_22 : memref<2x64xbf16, strided<[64, 1]>>)
    %subview_24 = memref.subview %reinterpret_cast_2[%57, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_24, %alloc_19 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_19 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>>)
    %subview_25 = memref.subview %reinterpret_cast_7[%57, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_25, %alloc_20 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_20 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc_18 : memref<2x32xbf16, strided<[32, 1]>>)
    %c1_i32_26 = arith.constant 1 : i32
    scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32_26  : i32 {
      %alloc_27 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_28 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_29 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_30 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_31 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_32 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_33 = memref.alloc() : memref<16x16xf32, strided<[16, 1]>>
      %alloc_34 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_35 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_36 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_37 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_38 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_39 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_40 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_41 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_42 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_43 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_44 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_45 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_46 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_47 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_48 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_49 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_50 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_51 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_52 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_53 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_54 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_55 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_56 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_57 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_58 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_59 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_60 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_61 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_62 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_63 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_64 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_65 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_66 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_67 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_68 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_69 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
      %alloc_70 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_71 = memref.alloc() : memref<8xf32, strided<[1]>>
      %alloc_72 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_73 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_74 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_75 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_76 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_77 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_78 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_79 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_80 = memref.alloc() : memref<16x1x64xbf16, strided<[64, 64, 1]>>
      %alloc_81 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_82 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
      %alloc_83 = memref.alloc() : memref<1x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_84 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_85 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_86 = memref.alloc() : memref<1x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_87 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_88 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_89 = memref.alloc() : memref<1x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_90 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_91 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_92 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_93 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_94 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_95 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_96 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_97 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_98 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_99 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %58 = arith.index_cast %42 : i32 to index
      %59 = arith.index_cast %arg28 : i32 to index
      %subview_100 = memref.subview %reinterpret_cast_14[0, %58, %59, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_100, %alloc_33 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>>
      %c16_i32_101 = arith.constant 16 : i32
      %60 = arith.muli %arg28, %c16_i32_101 : i32
      %c1_i32_102 = arith.constant 1 : i32
      %61 = arith.addi %60, %c1_i32_102 : i32
      %62 = arith.index_cast %61 : i32 to index
      %subview_103 = memref.subview %reinterpret_cast_8[0, %58, %62] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_103, %alloc_35 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_35 : memref<16xbf16, strided<[1]>>) outs(%alloc_34 : memref<16xf32, strided<[1]>>)
      %cst = arith.constant -1.000000e+00 : f32
      hivm.hir.vmul ins(%alloc_34, %cst : memref<16xf32, strided<[1]>>, f32) outs(%alloc_34 : memref<16xf32, strided<[1]>>)
      %alloc_104 = memref.alloc() : memref<16xf32>
      %cst_105 = arith.constant 0.000000e+00 : f32
      %cst_106 = arith.constant 1.000000e+00 : f32
      hivm.hir.vsub ins(%cst_105, %alloc_34 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_104 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_104 : memref<16xf32>) outs(%alloc_104 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_104, %cst_106 : memref<16xf32>, f32) outs(%alloc_104 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_106, %alloc_104 : f32, memref<16xf32>) outs(%alloc_34 : memref<16xf32, strided<[1]>>)
      %subview_107 = memref.subview %reinterpret_cast_13[0, %58, %62] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_107, %alloc_37 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xf32, strided<[1]>>) outs(%alloc_36 : memref<16xbf16, strided<[1]>>)
      %c0_i32_108 = arith.constant 0 : i32
      %63 = arith.sitofp %c0_i32_108 : i32 to bf16
      hivm.hir.vbrc ins(%63 : bf16) outs(%alloc_38 : memref<16xbf16, strided<[1]>>)
      %c1_i32_109 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c16_i32_101 step %c1_i32_109  : i32 {
        %c16_i32_179 = arith.constant 16 : i32
        %73 = arith.muli %arg28, %c16_i32_179 : i32
        %74 = arith.addi %73, %arg29 : i32
        %c31_i32 = arith.constant 31 : i32
        %75 = arith.cmpi slt, %74, %c31_i32 : i32
        scf.if %75 {
          %76 = arith.index_cast %arg29 : i32 to index
          %77 = memref.load %alloc_36[%76] : memref<16xbf16, strided<[1]>>
          %78 = arith.extf %77 : bf16 to f32
          %79 = memref.load %alloc_34[%76] : memref<16xf32, strided<[1]>>
          %80 = arith.mulf %78, %79 : f32
          %81 = arith.truncf %80 : f32 to bf16
          memref.store %81, %alloc_38[%76] : memref<16xbf16, strided<[1]>>
        }
      }
      memref.copy %alloc_38, %alloc_39 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %64 = arith.index_cast %60 : i32 to index
      %subview_110 = memref.subview %reinterpret_cast_8[0, %58, %64] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_110, %alloc_41 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_41 : memref<16xbf16, strided<[1]>>) outs(%alloc_40 : memref<16xf32, strided<[1]>>)
      %alloc_111 = memref.alloc() : memref<16xf32>
      %cst_112 = arith.constant 0.000000e+00 : f32
      %cst_113 = arith.constant 1.000000e+00 : f32
      hivm.hir.vsub ins(%cst_112, %alloc_40 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_111 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_111 : memref<16xf32>) outs(%alloc_111 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_111, %cst_113 : memref<16xf32>, f32) outs(%alloc_111 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_113, %alloc_111 : f32, memref<16xf32>) outs(%alloc_40 : memref<16xf32, strided<[1]>>)
      %subview_114 = memref.subview %reinterpret_cast_13[0, %58, %64] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_114, %alloc_43 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_43 : memref<16xf32, strided<[1]>>) outs(%alloc_42 : memref<16xbf16, strided<[1]>>)
      %c1_i32_115 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c16_i32_101 step %c1_i32_115  : i32 {
        %73 = arith.index_cast %arg29 : i32 to index
        %74 = memref.load %alloc_42[%73] : memref<16xbf16, strided<[1]>>
        %75 = arith.extf %74 : bf16 to f32
        %76 = memref.load %alloc_40[%73] : memref<16xf32, strided<[1]>>
        %77 = arith.mulf %75, %76 : f32
        memref.store %77, %alloc_44[%73] : memref<16xf32, strided<[1]>>
      }
      %c1_i32_116 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c16_i32_101 step %c1_i32_116  : i32 {
        %73 = arith.index_cast %arg29 : i32 to index
        %74 = memref.load %alloc_44[%73] : memref<16xf32, strided<[1]>>
        %75 = memref.load %alloc_39[%73] : memref<16xbf16, strided<[1]>>
        %76 = arith.extf %75 : bf16 to f32
        %77 = arith.addf %74, %76 : f32
        %78 = arith.truncf %77 : f32 to bf16
        memref.store %78, %alloc_45[%73] : memref<16xbf16, strided<[1]>>
      }
      memref.copy %alloc_45, %alloc_46 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %c1_i32_117 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c16_i32_101 step %c1_i32_117  : i32 {
        %c16_i32_179 = arith.constant 16 : i32
        %73 = arith.muli %arg28, %c16_i32_179 : i32
        %74 = arith.addi %73, %arg29 : i32
        %75 = arith.index_cast %74 : i32 to index
        %76 = arith.index_cast %42 : i32 to index
        %subview_180 = memref.subview %reinterpret_cast_0[0, %75, %76, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %77 = arith.index_cast %arg29 : i32 to index
        %subview_181 = memref.subview %alloc_30[%77, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<64xbf16, strided<[1], offset: ?>>
        memref.copy %subview_180, %subview_181 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>>
      }
      memref.copy %alloc_30, %alloc_79 : memref<16x64xbf16, strided<[64, 1]>> to memref<16x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_118 = memref.reinterpret_cast %alloc_79 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<16x1x64xbf16, strided<[64, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_118 : memref<16x1x64xbf16, strided<[64, 64, 1]>>) outs(%alloc_81 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [1]
      memref.copy %alloc_22, %alloc_82 : memref<2x64xbf16, strided<[64, 1]>> to memref<2x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_119 = memref.reinterpret_cast %alloc_82 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>> to memref<1x2x64xbf16, strided<[128, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_119 : memref<1x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_84 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_81 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_91 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_84 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_92 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vmul ins(%alloc_91, %alloc_92 : memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_93 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_93 : memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_47 : memref<16x2x64xbf16, strided<[128, 64, 1]>>)
      %reinterpret_cast_120 = memref.reinterpret_cast %alloc_47 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_121 = memref.subview %55[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %reinterpret_cast_120, %subview_121 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_122 = memref.subview %reinterpret_cast[0, %64, 0, %58, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_122, %alloc_49 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc, %alloc_85 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_123 = memref.reinterpret_cast %alloc_85 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_123 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_87 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_49 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_94 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_87 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_95 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_94, %alloc_95 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_96 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_96 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_49 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_124 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_125 = memref.subview %50[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %reinterpret_cast_124, %subview_125 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_126 = memref.subview %reinterpret_cast_9[0, %64, 0, %58, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_126, %alloc_51 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc_18, %alloc_88 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_127 = memref.reinterpret_cast %alloc_88 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_127 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_90 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_97 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_90 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_98 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_97, %alloc_98 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_99 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_99 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_128 = memref.reinterpret_cast %alloc_51 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_129 = memref.subview %51[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %reinterpret_cast_128, %subview_129 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_130 = memref.subview %50[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_130, %alloc_27 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_131 = memref.subview %51[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_131, %alloc_28 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %true = arith.constant true
      %c32_i32_132 = arith.constant 32 : i32
      %65 = arith.index_cast %c32_i32_132 : i32 to index
      hivm.hir.mmadL1 {b_transpose} ins(%alloc_27, %alloc_28, %true, %65, %65, %65 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_53 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_133 = memref.subview %45[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_53, %subview_133 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      %subview_134 = memref.subview %50[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_134, %alloc_54 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %c1_i32_135 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c16_i32_101 step %c1_i32_135  : i32 {
        %c0_i32_179 = arith.constant 0 : i32
        %c2_i32_180 = arith.constant 2 : i32
        %c1_i32_181 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_179 to %c2_i32_180 step %c1_i32_181  : i32 {
          %c0_i32_182 = arith.constant 0 : i32
          %c8_i32_183 = arith.constant 8 : i32
          %c1_i32_184 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_182 to %c8_i32_183 step %c1_i32_184  : i32 {
            %c2_i32_185 = arith.constant 2 : i32
            %73 = arith.muli %arg29, %c2_i32_185 : i32
            %74 = arith.addi %73, %arg30 : i32
            %75 = arith.index_cast %74 : i32 to index
            %76 = arith.index_cast %arg31 : i32 to index
            %77 = memref.load %alloc_54[%75, %76] : memref<32x32xbf16, strided<[32, 1]>>
            %78 = arith.index_cast %arg29 : i32 to index
            %79 = arith.index_cast %arg30 : i32 to index
            memref.store %77, %alloc_55[%78, %79, %76] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %c1_i32_136 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c16_i32_101 step %c1_i32_136  : i32 {
        %c0_i32_179 = arith.constant 0 : i32
        %c2_i32_180 = arith.constant 2 : i32
        %c1_i32_181 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_179 to %c2_i32_180 step %c1_i32_181  : i32 {
          %c0_i32_182 = arith.constant 0 : i32
          %c8_i32_183 = arith.constant 8 : i32
          %c1_i32_184 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_182 to %c8_i32_183 step %c1_i32_184  : i32 {
            %c2_i32_185 = arith.constant 2 : i32
            %73 = arith.muli %arg29, %c2_i32_185 : i32
            %74 = arith.addi %73, %arg30 : i32
            %75 = arith.index_cast %74 : i32 to index
            %c16_i32_186 = arith.constant 16 : i32
            %76 = arith.addi %arg31, %c16_i32_186 : i32
            %77 = arith.index_cast %76 : i32 to index
            %78 = memref.load %alloc_54[%75, %77] : memref<32x32xbf16, strided<[32, 1]>>
            %79 = arith.index_cast %arg29 : i32 to index
            %80 = arith.index_cast %arg30 : i32 to index
            %81 = arith.index_cast %arg31 : i32 to index
            memref.store %78, %alloc_56[%79, %80, %81] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %subview_137 = memref.subview %reinterpret_cast_3[0, %64, %58, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_137, %alloc_57 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>>
      %cst_138 = arith.constant 1.000000e+00 : f32
      %cst_139 = arith.constant -5.000000e-01 : f32
      %cst_140 = arith.constant 2.400000e+01 : f32
      %cst_141 = arith.constant 7.200000e+02 : f32
      %cst_142 = arith.constant -1.000000e+00 : f32
      %66 = arith.divf %cst_138, %cst_140 : f32
      %67 = arith.divf %cst_142, %cst_141 : f32
      %alloc_143 = memref.alloc() : memref<16x8xf32>
      %alloc_144 = memref.alloc() : memref<16x8xf32>
      %alloc_145 = memref.alloc() : memref<16x8xf32>
      %alloc_146 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_57, %alloc_57 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_143 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_143, %alloc_143 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_144 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_143, %alloc_144 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_145 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_143, %cst_139 : memref<16x8xf32>, f32) outs(%alloc_143 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_144, %66 : memref<16x8xf32>, f32) outs(%alloc_144 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_145, %67 : memref<16x8xf32>, f32) outs(%alloc_145 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_143, %cst_138 : memref<16x8xf32>, f32) outs(%alloc_146 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_144, %alloc_146 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_146 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_145, %alloc_146 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_58 : memref<16x8xf32, strided<[8, 1]>>)
      %cst_147 = arith.constant 1.000000e+00 : f32
      %cst_148 = arith.constant -1.000000e+00 : f32
      %cst_149 = arith.constant 6.000000e+00 : f32
      %cst_150 = arith.constant 1.200000e+02 : f32
      %cst_151 = arith.constant 5.040000e+03 : f32
      %68 = arith.divf %cst_148, %cst_149 : f32
      %69 = arith.divf %cst_147, %cst_150 : f32
      %70 = arith.divf %cst_148, %cst_151 : f32
      %alloc_152 = memref.alloc() : memref<16x8xf32>
      %alloc_153 = memref.alloc() : memref<16x8xf32>
      %alloc_154 = memref.alloc() : memref<16x8xf32>
      %alloc_155 = memref.alloc() : memref<16x8xf32>
      %alloc_156 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_57, %alloc_57 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_152 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_152, %alloc_57 : memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_153 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_153, %alloc_152 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_154 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_154, %alloc_152 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_155 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_153, %68 : memref<16x8xf32>, f32) outs(%alloc_153 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_154, %69 : memref<16x8xf32>, f32) outs(%alloc_154 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_155, %70 : memref<16x8xf32>, f32) outs(%alloc_155 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_57, %alloc_153 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) outs(%alloc_156 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_154, %alloc_156 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_156 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_155, %alloc_156 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_59 : memref<16x8xf32, strided<[8, 1]>>)
      %c1_i32_157 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c16_i32_101 step %c1_i32_157  : i32 {
        %c0_i32_179 = arith.constant 0 : i32
        %c2_i32_180 = arith.constant 2 : i32
        %c1_i32_181 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_179 to %c2_i32_180 step %c1_i32_181  : i32 {
          %c0_i32_182 = arith.constant 0 : i32
          %c8_i32_183 = arith.constant 8 : i32
          %c1_i32_184 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_182 to %c8_i32_183 step %c1_i32_184  : i32 {
            %73 = arith.index_cast %arg29 : i32 to index
            %74 = arith.index_cast %arg31 : i32 to index
            %75 = memref.load %alloc_58[%73, %74] : memref<16x8xf32, strided<[8, 1]>>
            %76 = arith.index_cast %arg30 : i32 to index
            %77 = memref.load %alloc_55[%73, %76, %74] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %78 = arith.extf %77 : bf16 to f32
            %79 = arith.mulf %75, %78 : f32
            %80 = memref.load %alloc_59[%73, %74] : memref<16x8xf32, strided<[8, 1]>>
            %81 = memref.load %alloc_56[%73, %76, %74] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %82 = arith.extf %81 : bf16 to f32
            %83 = arith.subf %79, %79 : f32
            %84 = arith.truncf %83 : f32 to bf16
            %c2_i32_185 = arith.constant 2 : i32
            %85 = arith.muli %arg29, %c2_i32_185 : i32
            %86 = arith.addi %85, %arg30 : i32
            %87 = arith.index_cast %86 : i32 to index
            memref.store %84, %alloc_54[%87, %74] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      %c1_i32_158 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c16_i32_101 step %c1_i32_158  : i32 {
        %c0_i32_179 = arith.constant 0 : i32
        %c2_i32_180 = arith.constant 2 : i32
        %c1_i32_181 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_179 to %c2_i32_180 step %c1_i32_181  : i32 {
          %c0_i32_182 = arith.constant 0 : i32
          %c8_i32_183 = arith.constant 8 : i32
          %c1_i32_184 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_182 to %c8_i32_183 step %c1_i32_184  : i32 {
            %73 = arith.index_cast %arg29 : i32 to index
            %74 = arith.index_cast %arg31 : i32 to index
            %75 = memref.load %alloc_59[%73, %74] : memref<16x8xf32, strided<[8, 1]>>
            %76 = arith.index_cast %arg30 : i32 to index
            %77 = memref.load %alloc_55[%73, %76, %74] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %78 = arith.extf %77 : bf16 to f32
            %79 = arith.mulf %75, %78 : f32
            %80 = memref.load %alloc_58[%73, %74] : memref<16x8xf32, strided<[8, 1]>>
            %81 = memref.load %alloc_56[%73, %76, %74] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %82 = arith.extf %81 : bf16 to f32
            %83 = arith.addf %79, %79 : f32
            %84 = arith.truncf %83 : f32 to bf16
            %c2_i32_185 = arith.constant 2 : i32
            %85 = arith.muli %arg29, %c2_i32_185 : i32
            %86 = arith.addi %85, %arg30 : i32
            %87 = arith.index_cast %86 : i32 to index
            %c16_i32_186 = arith.constant 16 : i32
            %88 = arith.addi %arg31, %c16_i32_186 : i32
            %89 = arith.index_cast %88 : i32 to index
            memref.store %84, %alloc_54[%87, %89] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      %subview_159 = memref.subview %52[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_54, %subview_159 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_21 : memref<32x64xf32, strided<[64, 1]>>) outs(%alloc_61 : memref<32x64xbf16, strided<[64, 1]>>)
      %subview_160 = memref.subview %54[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %alloc_61, %subview_160 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_161 = memref.subview %52[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_161, %alloc_27 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_162 = memref.subview %54[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %subview_162, %alloc_31 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %c64_i32_163 = arith.constant 64 : i32
      %71 = arith.index_cast %c64_i32_163 : i32 to index
      hivm.hir.mmadL1 ins(%alloc_27, %alloc_31, %true, %65, %65, %71 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_60 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_164 = memref.subview %46[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_60, %subview_164 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      %subview_165 = memref.subview %51[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_165, %alloc_62 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %c1_i32_166 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c16_i32_101 step %c1_i32_166  : i32 {
        %c0_i32_179 = arith.constant 0 : i32
        %c2_i32_180 = arith.constant 2 : i32
        %c1_i32_181 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_179 to %c2_i32_180 step %c1_i32_181  : i32 {
          %c0_i32_182 = arith.constant 0 : i32
          %c8_i32_183 = arith.constant 8 : i32
          %c1_i32_184 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_182 to %c8_i32_183 step %c1_i32_184  : i32 {
            %c2_i32_185 = arith.constant 2 : i32
            %73 = arith.muli %arg29, %c2_i32_185 : i32
            %74 = arith.addi %73, %arg30 : i32
            %75 = arith.index_cast %74 : i32 to index
            %76 = arith.index_cast %arg31 : i32 to index
            %77 = memref.load %alloc_62[%75, %76] : memref<32x32xbf16, strided<[32, 1]>>
            %78 = arith.index_cast %arg29 : i32 to index
            %79 = arith.index_cast %arg30 : i32 to index
            memref.store %77, %alloc_63[%78, %79, %76] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %c1_i32_167 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c16_i32_101 step %c1_i32_167  : i32 {
        %c0_i32_179 = arith.constant 0 : i32
        %c2_i32_180 = arith.constant 2 : i32
        %c1_i32_181 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_179 to %c2_i32_180 step %c1_i32_181  : i32 {
          %c0_i32_182 = arith.constant 0 : i32
          %c8_i32_183 = arith.constant 8 : i32
          %c1_i32_184 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_182 to %c8_i32_183 step %c1_i32_184  : i32 {
            %c2_i32_185 = arith.constant 2 : i32
            %73 = arith.muli %arg29, %c2_i32_185 : i32
            %74 = arith.addi %73, %arg30 : i32
            %75 = arith.index_cast %74 : i32 to index
            %c16_i32_186 = arith.constant 16 : i32
            %76 = arith.addi %arg31, %c16_i32_186 : i32
            %77 = arith.index_cast %76 : i32 to index
            %78 = memref.load %alloc_62[%75, %77] : memref<32x32xbf16, strided<[32, 1]>>
            %79 = arith.index_cast %arg29 : i32 to index
            %80 = arith.index_cast %arg30 : i32 to index
            %81 = arith.index_cast %arg31 : i32 to index
            memref.store %78, %alloc_64[%79, %80, %81] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %c1_i32_168 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c16_i32_101 step %c1_i32_168  : i32 {
        %c0_i32_179 = arith.constant 0 : i32
        %c2_i32_180 = arith.constant 2 : i32
        %c1_i32_181 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_179 to %c2_i32_180 step %c1_i32_181  : i32 {
          %c0_i32_182 = arith.constant 0 : i32
          %c8_i32_183 = arith.constant 8 : i32
          %c1_i32_184 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_182 to %c8_i32_183 step %c1_i32_184  : i32 {
            %73 = arith.index_cast %arg29 : i32 to index
            %74 = arith.index_cast %arg31 : i32 to index
            %75 = memref.load %alloc_58[%73, %74] : memref<16x8xf32, strided<[8, 1]>>
            %76 = arith.index_cast %arg30 : i32 to index
            %77 = memref.load %alloc_63[%73, %76, %74] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %78 = arith.extf %77 : bf16 to f32
            %79 = arith.mulf %75, %78 : f32
            %80 = memref.load %alloc_59[%73, %74] : memref<16x8xf32, strided<[8, 1]>>
            %81 = memref.load %alloc_64[%73, %76, %74] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %82 = arith.extf %81 : bf16 to f32
            %83 = arith.subf %79, %79 : f32
            %84 = arith.truncf %83 : f32 to bf16
            %c2_i32_185 = arith.constant 2 : i32
            %85 = arith.muli %arg29, %c2_i32_185 : i32
            %86 = arith.addi %85, %arg30 : i32
            %87 = arith.index_cast %86 : i32 to index
            memref.store %84, %alloc_62[%87, %74] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      %c1_i32_169 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c16_i32_101 step %c1_i32_169  : i32 {
        %c0_i32_179 = arith.constant 0 : i32
        %c2_i32_180 = arith.constant 2 : i32
        %c1_i32_181 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_179 to %c2_i32_180 step %c1_i32_181  : i32 {
          %c0_i32_182 = arith.constant 0 : i32
          %c8_i32_183 = arith.constant 8 : i32
          %c1_i32_184 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_182 to %c8_i32_183 step %c1_i32_184  : i32 {
            %73 = arith.index_cast %arg29 : i32 to index
            %74 = arith.index_cast %arg31 : i32 to index
            %75 = memref.load %alloc_59[%73, %74] : memref<16x8xf32, strided<[8, 1]>>
            %76 = arith.index_cast %arg30 : i32 to index
            %77 = memref.load %alloc_63[%73, %76, %74] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %78 = arith.extf %77 : bf16 to f32
            %79 = arith.mulf %75, %78 : f32
            %80 = memref.load %alloc_58[%73, %74] : memref<16x8xf32, strided<[8, 1]>>
            %81 = memref.load %alloc_64[%73, %76, %74] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %82 = arith.extf %81 : bf16 to f32
            %83 = arith.addf %79, %79 : f32
            %84 = arith.truncf %83 : f32 to bf16
            %c2_i32_185 = arith.constant 2 : i32
            %85 = arith.muli %arg29, %c2_i32_185 : i32
            %86 = arith.addi %85, %arg30 : i32
            %87 = arith.index_cast %86 : i32 to index
            %c16_i32_186 = arith.constant 16 : i32
            %88 = arith.addi %arg31, %c16_i32_186 : i32
            %89 = arith.index_cast %88 : i32 to index
            memref.store %84, %alloc_62[%87, %89] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      memref.copy %alloc_62, %alloc_65 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_65 : memref<32x32xbf16, strided<[32, 1]>>) outs(%alloc_66 : memref<32x32xf32, strided<[32, 1]>>)
      %c1_i32_170 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_108 to %c32_i32_132 step %c1_i32_170  : i32 {
        %c0_i32_179 = arith.constant 0 : i32
        %c32_i32_180 = arith.constant 32 : i32
        %c1_i32_181 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_179 to %c32_i32_180 step %c1_i32_181  : i32 {
          %73 = arith.index_cast %arg29 : i32 to index
          %74 = arith.index_cast %arg30 : i32 to index
          %75 = memref.load %alloc_66[%73, %74] : memref<32x32xf32, strided<[32, 1]>>
          %c2_i32_182 = arith.constant 2 : i32
          %76 = arith.divsi %arg29, %c2_i32_182 : i32
          %77 = arith.index_cast %76 : i32 to index
          %78 = memref.load %alloc_46[%77] : memref<16xbf16, strided<[1]>>
          %79 = arith.extf %78 : bf16 to f32
          %80 = arith.mulf %75, %79 : f32
          memref.store %80, %alloc_66[%73, %74] : memref<32x32xf32, strided<[32, 1]>>
        }
      }
      hivm.hir.vcast ins(%alloc_66 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc_65 : memref<32x32xbf16, strided<[32, 1]>>)
      %subview_171 = memref.subview %53[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_65, %subview_171 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_172 = memref.subview %52[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_172, %alloc_27 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_173 = memref.subview %53[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_173, %alloc_28 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%alloc_27, %alloc_28, %true, %65, %65, %65 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_67 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_174 = memref.subview %47[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_67, %subview_174 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      hivm.hir.vexp ins(%alloc_33 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_33 : memref<16x16xf32, strided<[16, 1]>>)
      %c2_i32_175 = arith.constant 2 : i32
      %72 = arith.cmpi slt, %44, %c2_i32_175 : i32
      scf.if %72 {
        %c16_i32_179 = arith.constant 16 : i32
        %73 = arith.muli %44, %c16_i32_179 : i32
        %74 = arith.index_cast %73 : i32 to index
        %subview_180 = memref.subview %47[0, %74, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_180, %alloc_68 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_181 = memref.subview %46[0, %74, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_181, %alloc_70 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %75 = arith.index_cast %42 : i32 to index
        %76 = arith.muli %arg28, %c16_i32_179 : i32
        %c8_i32_182 = arith.constant 8 : i32
        %77 = arith.muli %44, %c8_i32_182 : i32
        %78 = arith.addi %76, %77 : i32
        %79 = arith.index_cast %78 : i32 to index
        %subview_183 = memref.subview %reinterpret_cast_4[0, %75, %79] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_183, %alloc_71 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>>
        hivm.hir.vexp ins(%alloc_71 : memref<8xf32, strided<[1]>>) outs(%alloc_71 : memref<8xf32, strided<[1]>>)
        %c0_i32_184 = arith.constant 0 : i32
        %c1_i32_185 = arith.constant 1 : i32
        scf.for %arg29 = %c0_i32_184 to %c16_i32_179 step %c1_i32_185  : i32 {
          %c0_i32_188 = arith.constant 0 : i32
          %c32_i32_189 = arith.constant 32 : i32
          %c1_i32_190 = arith.constant 1 : i32
          scf.for %arg30 = %c0_i32_188 to %c32_i32_189 step %c1_i32_190  : i32 {
            %c2_i32_193 = arith.constant 2 : i32
            %80 = arith.divsi %arg30, %c2_i32_193 : i32
            %c8_i32_194 = arith.constant 8 : i32
            %81 = arith.muli %44, %c8_i32_194 : i32
            %82 = arith.divsi %arg29, %c2_i32_193 : i32
            %83 = arith.addi %81, %82 : i32
            %84 = arith.cmpi slt, %80, %83 : i32
            scf.if %84 {
              %85 = arith.index_cast %arg29 : i32 to index
              %86 = arith.index_cast %arg30 : i32 to index
              %87 = memref.load %alloc_68[%85, %86] : memref<16x32xf32, strided<[32, 1]>>
              %c8_i32_195 = arith.constant 8 : i32
              %88 = arith.muli %44, %c8_i32_195 : i32
              %c2_i32_196 = arith.constant 2 : i32
              %89 = arith.divsi %arg29, %c2_i32_196 : i32
              %90 = arith.addi %88, %89 : i32
              %91 = arith.index_cast %90 : i32 to index
              %92 = arith.divsi %arg30, %c2_i32_196 : i32
              %93 = arith.index_cast %92 : i32 to index
              %94 = memref.load %alloc_33[%91, %93] : memref<16x16xf32, strided<[16, 1]>>
              %95 = arith.mulf %87, %94 : f32
              %96 = arith.truncf %95 : f32 to bf16
              memref.store %96, %alloc_69[%85, %86] : memref<16x32xbf16, strided<[32, 1]>>
            } else {
              %cst_195 = arith.constant 0.000000e+00 : bf16
              %85 = arith.index_cast %arg29 : i32 to index
              %86 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_195, %alloc_69[%85, %86] : memref<16x32xbf16, strided<[32, 1]>>
            }
          }
          %c64_i32_191 = arith.constant 64 : i32
          %c1_i32_192 = arith.constant 1 : i32
          scf.for %arg30 = %c0_i32_188 to %c64_i32_191 step %c1_i32_192  : i32 {
            %80 = arith.index_cast %arg29 : i32 to index
            %81 = arith.index_cast %arg30 : i32 to index
            %82 = memref.load %alloc_70[%80, %81] : memref<16x64xf32, strided<[64, 1]>>
            %c2_i32_193 = arith.constant 2 : i32
            %83 = arith.divsi %arg29, %c2_i32_193 : i32
            %84 = arith.index_cast %83 : i32 to index
            %85 = memref.load %alloc_71[%84] : memref<8xf32, strided<[1]>>
            %86 = arith.mulf %82, %85 : f32
            memref.store %86, %alloc_70[%80, %81] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %subview_186 = memref.subview %48[0, %74, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_69, %subview_186 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        %subview_187 = memref.subview %46[0, %74, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %alloc_70, %subview_187 : memref<16x64xf32, strided<[64, 1]>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      }
      %subview_176 = memref.subview %48[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_176, %alloc_32 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_177 = memref.subview %55[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %subview_177, %alloc_29 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%alloc_32, %alloc_29, %true, %65, %65, %71 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_72 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_178 = memref.subview %49[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_72, %subview_178 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.if %72 {
        %c16_i32_179 = arith.constant 16 : i32
        %73 = arith.muli %44, %c16_i32_179 : i32
        %74 = arith.index_cast %73 : i32 to index
        %subview_180 = memref.subview %46[0, %74, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_180, %alloc_73 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_181 = memref.subview %49[0, %74, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_181, %alloc_74 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_182 = memref.subview %45[0, %74, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_182, %alloc_75 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_183 = memref.subview %55[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
        memref.copy %subview_183, %alloc_76 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
        %75 = arith.muli %arg28, %c16_i32_179 : i32
        %c8_i32_184 = arith.constant 8 : i32
        %76 = arith.muli %44, %c8_i32_184 : i32
        %77 = arith.addi %75, %76 : i32
        %78 = arith.index_cast %77 : i32 to index
        %79 = arith.index_cast %42 : i32 to index
        %subview_185 = memref.subview %reinterpret_cast_5[0, %78, %79, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_185, %alloc_77 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>>
        %c0_i32_186 = arith.constant 0 : i32
        %c1_i32_187 = arith.constant 1 : i32
        scf.for %arg29 = %c0_i32_186 to %c16_i32_179 step %c1_i32_187  : i32 {
          %c0_i32_190 = arith.constant 0 : i32
          %c64_i32_191 = arith.constant 64 : i32
          %c1_i32_192 = arith.constant 1 : i32
          scf.for %arg30 = %c0_i32_190 to %c64_i32_191 step %c1_i32_192  : i32 {
            %80 = arith.index_cast %arg29 : i32 to index
            %81 = arith.index_cast %arg30 : i32 to index
            %82 = memref.load %alloc_73[%80, %81] : memref<16x64xf32, strided<[64, 1]>>
            %83 = memref.load %alloc_74[%80, %81] : memref<16x64xf32, strided<[64, 1]>>
            %84 = arith.addf %82, %83 : f32
            memref.store %84, %alloc_73[%80, %81] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %c1_i32_188 = arith.constant 1 : i32
        scf.for %arg29 = %c0_i32_186 to %c8_i32_184 step %c1_i32_188  : i32 {
          %c0_i32_190 = arith.constant 0 : i32
          %c64_i32_191 = arith.constant 64 : i32
          %c1_i32_192 = arith.constant 1 : i32
          scf.for %arg30 = %c0_i32_190 to %c64_i32_191 step %c1_i32_192  : i32 {
            %cst_193 = arith.constant 0.000000e+00 : bf16
            %80 = arith.index_cast %arg29 : i32 to index
            %81 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_193, %alloc_78[%80, %81] : memref<8x64xbf16, strided<[64, 1]>>
          }
        }
        %subview_189 = memref.subview %reinterpret_cast_10[0, %78, %79, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_78, %subview_189 : memref<8x64xbf16, strided<[64, 1]>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      }
    } {tilelangir.num_stages = 2 : i32}
    return
  }
}
// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @mamba_mimo_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xbf16, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: memref<?xf32, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: memref<?xbf16, #hivm.address_space<gm>>, %arg12: memref<?xf32, #hivm.address_space<gm>>, %arg13: memref<?xf32, #hivm.address_space<gm>>, %arg14: memref<?xf32, #hivm.address_space<gm>>, %arg15: memref<?xf32, #hivm.address_space<gm>>, %arg16: memref<?xf32, #hivm.address_space<gm>>, %arg17: memref<?xf32, #hivm.address_space<gm>>, %arg18: memref<?xbf16, #hivm.address_space<gm>>, %arg19: memref<?xf32, #hivm.address_space<gm>>, %arg20: memref<?xf32, #hivm.address_space<gm>>, %arg21: memref<?xbf16, #hivm.address_space<gm>>, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %cst = arith.constant -1.98412701E-4 : f32
    %cst_0 = arith.constant 0.00833333377 : f32
    %cst_1 = arith.constant -0.166666672 : f32
    %cst_2 = arith.constant -0.00138888892 : f32
    %cst_3 = arith.constant 0.0416666679 : f32
    %c2048 = arith.constant 2048 : index
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %c132 = arith.constant 132 : index
    %c33 = arith.constant 33 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c8192 = arith.constant 8192 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %cst_4 = arith.constant 0.000000e+00 : bf16
    %cst_5 = arith.constant -5.000000e-01 : f32
    %true = arith.constant true
    %c31_i32 = arith.constant 31 : i32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant -1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_10 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [4, 2, 64], strides: [%c128, %c64, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg14 to offset: [0], sizes: [1, 32, 4, 8], strides: [%c1024, %c32, %c8, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_13 = memref.reinterpret_cast %arg15 to offset: [0], sizes: [1, 4, 32], strides: [%c128, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_14 = memref.reinterpret_cast %arg11 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_15 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_16 = memref.reinterpret_cast %arg18 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_17 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_18 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_19 = memref.reinterpret_cast %arg17 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_20 = memref.reinterpret_cast %arg19 to offset: [0], sizes: [1, 4, 2, 16, 16], strides: [%c2048, %c512, %c256, %c16, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x32x32xf32>
    annotation.mark %4 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32>
    %5 = memref_ext.alloc_workspace() : memref<2x32x64xf32>
    annotation.mark %5 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32>
    %6 = memref_ext.alloc_workspace() : memref<2x32x32xf32>
    annotation.mark %6 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32>
    %7 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %7 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %8 = memref_ext.alloc_workspace() : memref<2x32x64xf32>
    annotation.mark %8 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32>
    %9 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %9 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %10 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %11 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %12 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %13 = memref_ext.alloc_workspace() : memref<2x32x64xbf16>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16>
    %14 = memref_ext.alloc_workspace() : memref<2x32x64xbf16>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16>
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_21 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_22 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_23 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_24 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
    %alloc_25 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
    %alloc_26 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vbrc ins(%cst_7 : f32) outs(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>)
    %15 = arith.index_cast %1 : i32 to index
    %subview = memref.subview %reinterpret_cast_10[%15, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_26 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vcast ins(%alloc_26 : memref<2x64xf32, strided<[64, 1]>>) outs(%alloc_25 : memref<2x64xbf16, strided<[64, 1]>>)
    %subview_27 = memref.subview %reinterpret_cast_11[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_27, %alloc_22 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_22 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>>)
    %subview_28 = memref.subview %reinterpret_cast_15[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_28, %alloc_23 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc_21 : memref<2x32xbf16, strided<[32, 1]>>)
    scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %alloc_29 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_30 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_31 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_32 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_33 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_34 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_35 = memref.alloc() : memref<16x16xf32, strided<[16, 1]>>
      %alloc_36 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_37 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_38 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_39 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_40 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_41 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_42 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_43 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_44 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_45 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_46 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_47 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_48 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_49 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_50 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_51 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_52 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_53 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_54 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_55 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_56 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_57 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_58 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_59 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_60 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_61 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_62 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_63 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_64 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_65 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_66 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
      %alloc_67 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_68 = memref.alloc() : memref<8xf32, strided<[1]>>
      %alloc_69 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_70 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_71 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_72 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_73 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_74 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_75 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_76 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_77 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_78 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
      %alloc_79 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_80 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_81 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_82 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_83 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_84 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_85 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_86 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_87 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_88 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_89 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_90 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_91 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_92 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %16 = arith.index_cast %1 : i32 to index
      %17 = arith.index_cast %arg28 : i32 to index
      %subview_93 = memref.subview %reinterpret_cast_20[0, %16, %17, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_93, %alloc_35 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>>
      %18 = arith.muli %arg28, %c16_i32 : i32
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_94 = memref.subview %reinterpret_cast_16[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_94, %alloc_37 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xbf16, strided<[1]>>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      hivm.hir.vmul ins(%alloc_36, %cst_8 : memref<16xf32, strided<[1]>>, f32) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %alloc_95 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_36 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_95 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_95 : memref<16xf32>) outs(%alloc_95 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_95, %cst_6 : memref<16xf32>, f32) outs(%alloc_95 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_95 : f32, memref<16xf32>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %subview_96 = memref.subview %reinterpret_cast_19[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_96, %alloc_39 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_39 : memref<16xf32, strided<[1]>>) outs(%alloc_38 : memref<16xbf16, strided<[1]>>)
      hivm.hir.vbrc ins(%cst_4 : bf16) outs(%alloc_40 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.muli %arg28, %c16_i32 : i32
        %24 = arith.addi %23, %arg29 : i32
        %25 = arith.cmpi slt, %24, %c31_i32 : i32
        scf.if %25 {
          %26 = arith.index_cast %arg29 : i32 to index
          %27 = memref.load %alloc_38[%26] : memref<16xbf16, strided<[1]>>
          %28 = arith.extf %27 : bf16 to f32
          %29 = memref.load %alloc_36[%26] : memref<16xf32, strided<[1]>>
          %30 = arith.mulf %28, %29 : f32
          %31 = arith.truncf %30 : f32 to bf16
          memref.store %31, %alloc_40[%26] : memref<16xbf16, strided<[1]>>
        }
      }
      memref.copy %alloc_40, %alloc_41 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %21 = arith.index_cast %18 : i32 to index
      %subview_97 = memref.subview %reinterpret_cast_16[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_97, %alloc_43 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_43 : memref<16xbf16, strided<[1]>>) outs(%alloc_42 : memref<16xf32, strided<[1]>>)
      %alloc_98 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_42 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_98 : memref<16xf32>) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_98, %cst_6 : memref<16xf32>, f32) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_98 : f32, memref<16xf32>) outs(%alloc_42 : memref<16xf32, strided<[1]>>)
      %subview_99 = memref.subview %reinterpret_cast_19[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_99, %alloc_45 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_45 : memref<16xf32, strided<[1]>>) outs(%alloc_44 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_44[%23] : memref<16xbf16, strided<[1]>>
        %25 = arith.extf %24 : bf16 to f32
        %26 = memref.load %alloc_42[%23] : memref<16xf32, strided<[1]>>
        %27 = arith.mulf %25, %26 : f32
        memref.store %27, %alloc_46[%23] : memref<16xf32, strided<[1]>>
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_46[%23] : memref<16xf32, strided<[1]>>
        %25 = memref.load %alloc_41[%23] : memref<16xbf16, strided<[1]>>
        %26 = arith.extf %25 : bf16 to f32
        %27 = arith.addf %24, %26 : f32
        %28 = arith.truncf %27 : f32 to bf16
        memref.store %28, %alloc_47[%23] : memref<16xbf16, strided<[1]>>
      }
      memref.copy %alloc_47, %alloc_48 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.muli %arg28, %c16_i32 : i32
        %24 = arith.addi %23, %arg29 : i32
        %25 = arith.index_cast %24 : i32 to index
        %26 = arith.index_cast %1 : i32 to index
        %subview_139 = memref.subview %reinterpret_cast_9[0, %25, %26, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %27 = arith.index_cast %arg29 : i32 to index
        %subview_140 = memref.subview %alloc_32[%27, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<64xbf16, strided<[1], offset: ?>>
        memref.copy %subview_139, %subview_140 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>>
      }
      memref.copy %alloc_32, %alloc_76 : memref<16x64xbf16, strided<[64, 1]>> to memref<16x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_100 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<16x1x64xbf16, strided<[64, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_100 : memref<16x1x64xbf16, strided<[64, 64, 1]>>) outs(%alloc_77 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [1]
      memref.copy %alloc_25, %alloc_78 : memref<2x64xbf16, strided<[64, 1]>> to memref<2x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_101 = memref.reinterpret_cast %alloc_78 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>> to memref<1x2x64xbf16, strided<[128, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_101 : memref<1x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_79 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_77 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_84 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_79 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_85 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vmul ins(%alloc_84, %alloc_85 : memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_86 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_86 : memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_49 : memref<16x2x64xbf16, strided<[128, 64, 1]>>)
      %reinterpret_cast_102 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_103 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %reinterpret_cast_102, %subview_103 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_104 = memref.subview %reinterpret_cast[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_104, %alloc_50 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc, %alloc_80 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_105 = memref.reinterpret_cast %alloc_80 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_105 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_81 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_50 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_87 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_81 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_88 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_87, %alloc_88 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_89 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_89 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_50 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_106 = memref.reinterpret_cast %alloc_50 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_107 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %reinterpret_cast_106, %subview_107 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_108 = memref.subview %reinterpret_cast_17[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_108, %alloc_51 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc_21, %alloc_82 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_109 = memref.reinterpret_cast %alloc_82 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_109 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_83 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_90 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_83 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_90, %alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_110 = memref.reinterpret_cast %alloc_51 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_111 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %reinterpret_cast_110, %subview_111 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_112 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_112, %alloc_29 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_113 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_113, %alloc_30 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%alloc_29, %alloc_30, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_52 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_114 = memref.subview %4[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_52, %subview_114 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      %subview_115 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_115, %alloc_53 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.muli %arg29, %c2_i32 : i32
            %24 = arith.addi %23, %arg30 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = arith.index_cast %arg31 : i32 to index
            %27 = memref.load %alloc_53[%25, %26] : memref<32x32xbf16, strided<[32, 1]>>
            %28 = arith.index_cast %arg29 : i32 to index
            %29 = arith.index_cast %arg30 : i32 to index
            memref.store %27, %alloc_54[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %subview_116 = memref.subview %reinterpret_cast_12[0, %21, %16, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_116, %alloc_55 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>>
      %alloc_117 = memref.alloc() : memref<16x8xf32>
      %alloc_118 = memref.alloc() : memref<16x8xf32>
      %alloc_119 = memref.alloc() : memref<16x8xf32>
      %alloc_120 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_55, %alloc_55 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_117 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_117, %alloc_117 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_118 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_117, %alloc_118 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_119 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_117, %cst_5 : memref<16x8xf32>, f32) outs(%alloc_117 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_118, %cst_3 : memref<16x8xf32>, f32) outs(%alloc_118 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_119, %cst_2 : memref<16x8xf32>, f32) outs(%alloc_119 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_117, %cst_6 : memref<16x8xf32>, f32) outs(%alloc_120 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_118, %alloc_120 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_120 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_119, %alloc_120 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_56 : memref<16x8xf32, strided<[8, 1]>>)
      %alloc_121 = memref.alloc() : memref<16x8xf32>
      %alloc_122 = memref.alloc() : memref<16x8xf32>
      %alloc_123 = memref.alloc() : memref<16x8xf32>
      %alloc_124 = memref.alloc() : memref<16x8xf32>
      %alloc_125 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_55, %alloc_55 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_121 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_121, %alloc_55 : memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_122 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_122, %alloc_121 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_123 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_123, %alloc_121 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_124 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_122, %cst_1 : memref<16x8xf32>, f32) outs(%alloc_122 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_123, %cst_0 : memref<16x8xf32>, f32) outs(%alloc_123 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_124, %cst : memref<16x8xf32>, f32) outs(%alloc_124 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_55, %alloc_122 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) outs(%alloc_125 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_123, %alloc_125 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_125 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_124, %alloc_125 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_57 : memref<16x8xf32, strided<[8, 1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_56[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_54[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.subf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            memref.store %31, %alloc_53[%34, %24] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_57[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_54[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.addf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = arith.addi %arg31, %c16_i32 : i32
            %36 = arith.index_cast %35 : i32 to index
            memref.store %31, %alloc_53[%34, %36] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      %subview_126 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_53, %subview_126 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>) outs(%alloc_59 : memref<32x64xbf16, strided<[64, 1]>>)
      %subview_127 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %alloc_59, %subview_127 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_128 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_128, %alloc_29 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_129 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %subview_129, %alloc_33 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%alloc_29, %alloc_33, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_58 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_130 = memref.subview %5[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_58, %subview_130 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      %subview_131 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_131, %alloc_60 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.muli %arg29, %c2_i32 : i32
            %24 = arith.addi %23, %arg30 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = arith.index_cast %arg31 : i32 to index
            %27 = memref.load %alloc_60[%25, %26] : memref<32x32xbf16, strided<[32, 1]>>
            %28 = arith.index_cast %arg29 : i32 to index
            %29 = arith.index_cast %arg30 : i32 to index
            memref.store %27, %alloc_61[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_56[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_61[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.subf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            memref.store %31, %alloc_60[%34, %24] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_57[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_61[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.addf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = arith.addi %arg31, %c16_i32 : i32
            %36 = arith.index_cast %35 : i32 to index
            memref.store %31, %alloc_60[%34, %36] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      memref.copy %alloc_60, %alloc_62 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_62 : memref<32x32xbf16, strided<[32, 1]>>) outs(%alloc_63 : memref<32x32xf32, strided<[32, 1]>>)
      scf.for %arg29 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
          %23 = arith.index_cast %arg29 : i32 to index
          %24 = arith.index_cast %arg30 : i32 to index
          %25 = memref.load %alloc_63[%23, %24] : memref<32x32xf32, strided<[32, 1]>>
          %26 = arith.divsi %arg29, %c2_i32 : i32
          %27 = arith.index_cast %26 : i32 to index
          %28 = memref.load %alloc_48[%27] : memref<16xbf16, strided<[1]>>
          %29 = arith.extf %28 : bf16 to f32
          %30 = arith.mulf %25, %29 : f32
          memref.store %30, %alloc_63[%23, %24] : memref<32x32xf32, strided<[32, 1]>>
        }
      }
      hivm.hir.vcast ins(%alloc_63 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc_62 : memref<32x32xbf16, strided<[32, 1]>>)
      %subview_132 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_62, %subview_132 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_133 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_133, %alloc_29 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_134 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_134, %alloc_30 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%alloc_29, %alloc_30, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_64 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_135 = memref.subview %6[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_64, %subview_135 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      hivm.hir.vexp ins(%alloc_35 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_35 : memref<16x16xf32, strided<[16, 1]>>)
      %22 = arith.cmpi slt, %3, %c2_i32 : i32
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_139 = memref.subview %6[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_139, %alloc_65 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_140 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_140, %alloc_67 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %25 = arith.index_cast %1 : i32 to index
        %26 = arith.muli %arg28, %c16_i32 : i32
        %27 = arith.muli %3, %c8_i32 : i32
        %28 = arith.addi %26, %27 : i32
        %29 = arith.index_cast %28 : i32 to index
        %subview_141 = memref.subview %reinterpret_cast_13[0, %25, %29] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_141, %alloc_68 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>>
        hivm.hir.vexp ins(%alloc_68 : memref<8xf32, strided<[1]>>) outs(%alloc_68 : memref<8xf32, strided<[1]>>)
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %30 = arith.divsi %arg30, %c2_i32 : i32
            %31 = arith.muli %3, %c8_i32 : i32
            %32 = arith.divsi %arg29, %c2_i32 : i32
            %33 = arith.addi %31, %32 : i32
            %34 = arith.cmpi slt, %30, %33 : i32
            scf.if %34 {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              %37 = memref.load %alloc_65[%35, %36] : memref<16x32xf32, strided<[32, 1]>>
              %38 = arith.muli %3, %c8_i32 : i32
              %39 = arith.divsi %arg29, %c2_i32 : i32
              %40 = arith.addi %38, %39 : i32
              %41 = arith.index_cast %40 : i32 to index
              %42 = arith.divsi %arg30, %c2_i32 : i32
              %43 = arith.index_cast %42 : i32 to index
              %44 = memref.load %alloc_35[%41, %43] : memref<16x16xf32, strided<[16, 1]>>
              %45 = arith.mulf %37, %44 : f32
              %46 = arith.truncf %45 : f32 to bf16
              memref.store %46, %alloc_66[%35, %36] : memref<16x32xbf16, strided<[32, 1]>>
            } else {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_4, %alloc_66[%35, %36] : memref<16x32xbf16, strided<[32, 1]>>
            }
          }
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_67[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %33 = arith.divsi %arg29, %c2_i32 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = memref.load %alloc_68[%34] : memref<8xf32, strided<[1]>>
            %36 = arith.mulf %32, %35 : f32
            memref.store %36, %alloc_67[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %subview_142 = memref.subview %7[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_66, %subview_142 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        %subview_143 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %alloc_67, %subview_143 : memref<16x64xf32, strided<[64, 1]>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      }
      %subview_136 = memref.subview %7[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_136, %alloc_34 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_137 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %subview_137, %alloc_31 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%alloc_34, %alloc_31, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_69 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_138 = memref.subview %8[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_69, %subview_138 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_139 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_139, %alloc_70 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_140 = memref.subview %8[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_140, %alloc_71 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_141 = memref.subview %4[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_141, %alloc_72 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_142 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
        memref.copy %subview_142, %alloc_73 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
        %25 = arith.muli %arg28, %c16_i32 : i32
        %26 = arith.muli %3, %c8_i32 : i32
        %27 = arith.addi %25, %26 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.index_cast %1 : i32 to index
        %subview_143 = memref.subview %reinterpret_cast_14[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_143, %alloc_74 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_70[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %33 = memref.load %alloc_71[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %34 = arith.addf %32, %33 : f32
            memref.store %34, %alloc_70[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        scf.for %arg29 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_4, %alloc_75[%30, %31] : memref<8x64xbf16, strided<[64, 1]>>
          }
        }
        %subview_144 = memref.subview %reinterpret_cast_18[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_75, %subview_144 : memref<8x64xbf16, strided<[64, 1]>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      }
    } {tilelangir.num_stages = 2 : i32}
    return
  }
}


// -----// IR Dump After AdaptTritonKernel (adapt-triton-kernel) ('builtin.module' operation) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @mamba_mimo_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xbf16, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: memref<?xf32, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: memref<?xbf16, #hivm.address_space<gm>>, %arg12: memref<?xf32, #hivm.address_space<gm>>, %arg13: memref<?xf32, #hivm.address_space<gm>>, %arg14: memref<?xf32, #hivm.address_space<gm>>, %arg15: memref<?xf32, #hivm.address_space<gm>>, %arg16: memref<?xf32, #hivm.address_space<gm>>, %arg17: memref<?xf32, #hivm.address_space<gm>>, %arg18: memref<?xbf16, #hivm.address_space<gm>>, %arg19: memref<?xf32, #hivm.address_space<gm>>, %arg20: memref<?xf32, #hivm.address_space<gm>>, %arg21: memref<?xbf16, #hivm.address_space<gm>>, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %cst = arith.constant -1.98412701E-4 : f32
    %cst_0 = arith.constant 0.00833333377 : f32
    %cst_1 = arith.constant -0.166666672 : f32
    %cst_2 = arith.constant -0.00138888892 : f32
    %cst_3 = arith.constant 0.0416666679 : f32
    %c2048 = arith.constant 2048 : index
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %c132 = arith.constant 132 : index
    %c33 = arith.constant 33 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c8192 = arith.constant 8192 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %cst_4 = arith.constant 0.000000e+00 : bf16
    %cst_5 = arith.constant -5.000000e-01 : f32
    %true = arith.constant true
    %c31_i32 = arith.constant 31 : i32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant -1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_10 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [4, 2, 64], strides: [%c128, %c64, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg14 to offset: [0], sizes: [1, 32, 4, 8], strides: [%c1024, %c32, %c8, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_13 = memref.reinterpret_cast %arg15 to offset: [0], sizes: [1, 4, 32], strides: [%c128, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_14 = memref.reinterpret_cast %arg11 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_15 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_16 = memref.reinterpret_cast %arg18 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_17 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_18 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_19 = memref.reinterpret_cast %arg17 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_20 = memref.reinterpret_cast %arg19 to offset: [0], sizes: [1, 4, 2, 16, 16], strides: [%c2048, %c512, %c256, %c16, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x32x32xf32>
    annotation.mark %4 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32>
    %5 = memref_ext.alloc_workspace() : memref<2x32x64xf32>
    annotation.mark %5 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32>
    %6 = memref_ext.alloc_workspace() : memref<2x32x32xf32>
    annotation.mark %6 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32>
    %7 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %7 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %8 = memref_ext.alloc_workspace() : memref<2x32x64xf32>
    annotation.mark %8 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32>
    %9 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %9 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %10 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %11 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %12 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %13 = memref_ext.alloc_workspace() : memref<2x32x64xbf16>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16>
    %14 = memref_ext.alloc_workspace() : memref<2x32x64xbf16>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16>
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_21 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_22 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_23 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_24 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
    %alloc_25 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
    %alloc_26 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vbrc ins(%cst_7 : f32) outs(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>)
    %15 = arith.index_cast %1 : i32 to index
    %subview = memref.subview %reinterpret_cast_10[%15, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_26 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vcast ins(%alloc_26 : memref<2x64xf32, strided<[64, 1]>>) outs(%alloc_25 : memref<2x64xbf16, strided<[64, 1]>>)
    %subview_27 = memref.subview %reinterpret_cast_11[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_27, %alloc_22 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_22 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>>)
    %subview_28 = memref.subview %reinterpret_cast_15[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_28, %alloc_23 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc_21 : memref<2x32xbf16, strided<[32, 1]>>)
    scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %alloc_29 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_30 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_31 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_32 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_33 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_34 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_35 = memref.alloc() : memref<16x16xf32, strided<[16, 1]>>
      %alloc_36 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_37 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_38 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_39 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_40 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_41 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_42 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_43 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_44 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_45 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_46 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_47 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_48 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_49 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_50 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_51 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_52 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_53 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_54 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_55 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_56 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_57 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_58 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_59 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_60 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_61 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_62 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_63 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_64 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_65 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_66 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
      %alloc_67 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_68 = memref.alloc() : memref<8xf32, strided<[1]>>
      %alloc_69 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_70 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_71 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_72 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_73 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_74 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_75 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_76 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_77 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_78 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
      %alloc_79 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_80 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_81 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_82 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_83 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_84 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_85 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_86 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_87 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_88 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_89 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_90 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_91 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_92 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %16 = arith.index_cast %1 : i32 to index
      %17 = arith.index_cast %arg28 : i32 to index
      %subview_93 = memref.subview %reinterpret_cast_20[0, %16, %17, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_93, %alloc_35 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>>
      %18 = arith.muli %arg28, %c16_i32 : i32
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_94 = memref.subview %reinterpret_cast_16[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_94, %alloc_37 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xbf16, strided<[1]>>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      hivm.hir.vmul ins(%alloc_36, %cst_8 : memref<16xf32, strided<[1]>>, f32) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %alloc_95 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_36 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_95 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_95 : memref<16xf32>) outs(%alloc_95 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_95, %cst_6 : memref<16xf32>, f32) outs(%alloc_95 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_95 : f32, memref<16xf32>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %subview_96 = memref.subview %reinterpret_cast_19[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_96, %alloc_39 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_39 : memref<16xf32, strided<[1]>>) outs(%alloc_38 : memref<16xbf16, strided<[1]>>)
      hivm.hir.vbrc ins(%cst_4 : bf16) outs(%alloc_40 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.muli %arg28, %c16_i32 : i32
        %24 = arith.addi %23, %arg29 : i32
        %25 = arith.cmpi slt, %24, %c31_i32 : i32
        scf.if %25 {
          %26 = arith.index_cast %arg29 : i32 to index
          %27 = memref.load %alloc_38[%26] : memref<16xbf16, strided<[1]>>
          %28 = arith.extf %27 : bf16 to f32
          %29 = memref.load %alloc_36[%26] : memref<16xf32, strided<[1]>>
          %30 = arith.mulf %28, %29 : f32
          %31 = arith.truncf %30 : f32 to bf16
          memref.store %31, %alloc_40[%26] : memref<16xbf16, strided<[1]>>
        }
      }
      memref.copy %alloc_40, %alloc_41 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %21 = arith.index_cast %18 : i32 to index
      %subview_97 = memref.subview %reinterpret_cast_16[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_97, %alloc_43 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_43 : memref<16xbf16, strided<[1]>>) outs(%alloc_42 : memref<16xf32, strided<[1]>>)
      %alloc_98 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_42 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_98 : memref<16xf32>) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_98, %cst_6 : memref<16xf32>, f32) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_98 : f32, memref<16xf32>) outs(%alloc_42 : memref<16xf32, strided<[1]>>)
      %subview_99 = memref.subview %reinterpret_cast_19[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_99, %alloc_45 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_45 : memref<16xf32, strided<[1]>>) outs(%alloc_44 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_44[%23] : memref<16xbf16, strided<[1]>>
        %25 = arith.extf %24 : bf16 to f32
        %26 = memref.load %alloc_42[%23] : memref<16xf32, strided<[1]>>
        %27 = arith.mulf %25, %26 : f32
        memref.store %27, %alloc_46[%23] : memref<16xf32, strided<[1]>>
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_46[%23] : memref<16xf32, strided<[1]>>
        %25 = memref.load %alloc_41[%23] : memref<16xbf16, strided<[1]>>
        %26 = arith.extf %25 : bf16 to f32
        %27 = arith.addf %24, %26 : f32
        %28 = arith.truncf %27 : f32 to bf16
        memref.store %28, %alloc_47[%23] : memref<16xbf16, strided<[1]>>
      }
      memref.copy %alloc_47, %alloc_48 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.muli %arg28, %c16_i32 : i32
        %24 = arith.addi %23, %arg29 : i32
        %25 = arith.index_cast %24 : i32 to index
        %26 = arith.index_cast %1 : i32 to index
        %subview_139 = memref.subview %reinterpret_cast_9[0, %25, %26, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %27 = arith.index_cast %arg29 : i32 to index
        %subview_140 = memref.subview %alloc_32[%27, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<64xbf16, strided<[1], offset: ?>>
        memref.copy %subview_139, %subview_140 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>>
      }
      memref.copy %alloc_32, %alloc_76 : memref<16x64xbf16, strided<[64, 1]>> to memref<16x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_100 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<16x1x64xbf16, strided<[64, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_100 : memref<16x1x64xbf16, strided<[64, 64, 1]>>) outs(%alloc_77 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [1]
      memref.copy %alloc_25, %alloc_78 : memref<2x64xbf16, strided<[64, 1]>> to memref<2x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_101 = memref.reinterpret_cast %alloc_78 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>> to memref<1x2x64xbf16, strided<[128, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_101 : memref<1x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_79 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_77 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_84 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_79 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_85 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vmul ins(%alloc_84, %alloc_85 : memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_86 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_86 : memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_49 : memref<16x2x64xbf16, strided<[128, 64, 1]>>)
      %reinterpret_cast_102 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_103 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %reinterpret_cast_102, %subview_103 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_104 = memref.subview %reinterpret_cast[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_104, %alloc_50 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc, %alloc_80 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_105 = memref.reinterpret_cast %alloc_80 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_105 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_81 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_50 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_87 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_81 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_88 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_87, %alloc_88 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_89 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_89 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_50 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_106 = memref.reinterpret_cast %alloc_50 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_107 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %reinterpret_cast_106, %subview_107 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_108 = memref.subview %reinterpret_cast_17[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_108, %alloc_51 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc_21, %alloc_82 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_109 = memref.reinterpret_cast %alloc_82 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_109 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_83 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_90 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_83 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_90, %alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_110 = memref.reinterpret_cast %alloc_51 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_111 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %reinterpret_cast_110, %subview_111 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_112 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_112, %alloc_29 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_113 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_113, %alloc_30 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%alloc_29, %alloc_30, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_52 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_114 = memref.subview %4[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_52, %subview_114 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      %subview_115 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_115, %alloc_53 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.muli %arg29, %c2_i32 : i32
            %24 = arith.addi %23, %arg30 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = arith.index_cast %arg31 : i32 to index
            %27 = memref.load %alloc_53[%25, %26] : memref<32x32xbf16, strided<[32, 1]>>
            %28 = arith.index_cast %arg29 : i32 to index
            %29 = arith.index_cast %arg30 : i32 to index
            memref.store %27, %alloc_54[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %subview_116 = memref.subview %reinterpret_cast_12[0, %21, %16, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_116, %alloc_55 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>>
      %alloc_117 = memref.alloc() : memref<16x8xf32>
      %alloc_118 = memref.alloc() : memref<16x8xf32>
      %alloc_119 = memref.alloc() : memref<16x8xf32>
      %alloc_120 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_55, %alloc_55 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_117 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_117, %alloc_117 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_118 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_117, %alloc_118 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_119 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_117, %cst_5 : memref<16x8xf32>, f32) outs(%alloc_117 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_118, %cst_3 : memref<16x8xf32>, f32) outs(%alloc_118 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_119, %cst_2 : memref<16x8xf32>, f32) outs(%alloc_119 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_117, %cst_6 : memref<16x8xf32>, f32) outs(%alloc_120 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_118, %alloc_120 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_120 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_119, %alloc_120 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_56 : memref<16x8xf32, strided<[8, 1]>>)
      %alloc_121 = memref.alloc() : memref<16x8xf32>
      %alloc_122 = memref.alloc() : memref<16x8xf32>
      %alloc_123 = memref.alloc() : memref<16x8xf32>
      %alloc_124 = memref.alloc() : memref<16x8xf32>
      %alloc_125 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_55, %alloc_55 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_121 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_121, %alloc_55 : memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_122 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_122, %alloc_121 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_123 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_123, %alloc_121 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_124 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_122, %cst_1 : memref<16x8xf32>, f32) outs(%alloc_122 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_123, %cst_0 : memref<16x8xf32>, f32) outs(%alloc_123 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_124, %cst : memref<16x8xf32>, f32) outs(%alloc_124 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_55, %alloc_122 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) outs(%alloc_125 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_123, %alloc_125 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_125 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_124, %alloc_125 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_57 : memref<16x8xf32, strided<[8, 1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_56[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_54[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.subf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            memref.store %31, %alloc_53[%34, %24] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_57[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_54[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.addf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = arith.addi %arg31, %c16_i32 : i32
            %36 = arith.index_cast %35 : i32 to index
            memref.store %31, %alloc_53[%34, %36] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      %subview_126 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_53, %subview_126 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>) outs(%alloc_59 : memref<32x64xbf16, strided<[64, 1]>>)
      %subview_127 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %alloc_59, %subview_127 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_128 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_128, %alloc_29 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_129 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %subview_129, %alloc_33 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%alloc_29, %alloc_33, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_58 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_130 = memref.subview %5[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_58, %subview_130 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      %subview_131 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_131, %alloc_60 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.muli %arg29, %c2_i32 : i32
            %24 = arith.addi %23, %arg30 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = arith.index_cast %arg31 : i32 to index
            %27 = memref.load %alloc_60[%25, %26] : memref<32x32xbf16, strided<[32, 1]>>
            %28 = arith.index_cast %arg29 : i32 to index
            %29 = arith.index_cast %arg30 : i32 to index
            memref.store %27, %alloc_61[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_56[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_61[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.subf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            memref.store %31, %alloc_60[%34, %24] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_57[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_61[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.addf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = arith.addi %arg31, %c16_i32 : i32
            %36 = arith.index_cast %35 : i32 to index
            memref.store %31, %alloc_60[%34, %36] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      memref.copy %alloc_60, %alloc_62 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_62 : memref<32x32xbf16, strided<[32, 1]>>) outs(%alloc_63 : memref<32x32xf32, strided<[32, 1]>>)
      scf.for %arg29 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
          %23 = arith.index_cast %arg29 : i32 to index
          %24 = arith.index_cast %arg30 : i32 to index
          %25 = memref.load %alloc_63[%23, %24] : memref<32x32xf32, strided<[32, 1]>>
          %26 = arith.divsi %arg29, %c2_i32 : i32
          %27 = arith.index_cast %26 : i32 to index
          %28 = memref.load %alloc_48[%27] : memref<16xbf16, strided<[1]>>
          %29 = arith.extf %28 : bf16 to f32
          %30 = arith.mulf %25, %29 : f32
          memref.store %30, %alloc_63[%23, %24] : memref<32x32xf32, strided<[32, 1]>>
        }
      }
      hivm.hir.vcast ins(%alloc_63 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc_62 : memref<32x32xbf16, strided<[32, 1]>>)
      %subview_132 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_62, %subview_132 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_133 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_133, %alloc_29 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_134 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_134, %alloc_30 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%alloc_29, %alloc_30, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_64 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_135 = memref.subview %6[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_64, %subview_135 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      hivm.hir.vexp ins(%alloc_35 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_35 : memref<16x16xf32, strided<[16, 1]>>)
      %22 = arith.cmpi slt, %3, %c2_i32 : i32
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_139 = memref.subview %6[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_139, %alloc_65 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_140 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_140, %alloc_67 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %25 = arith.index_cast %1 : i32 to index
        %26 = arith.muli %arg28, %c16_i32 : i32
        %27 = arith.muli %3, %c8_i32 : i32
        %28 = arith.addi %26, %27 : i32
        %29 = arith.index_cast %28 : i32 to index
        %subview_141 = memref.subview %reinterpret_cast_13[0, %25, %29] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_141, %alloc_68 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>>
        hivm.hir.vexp ins(%alloc_68 : memref<8xf32, strided<[1]>>) outs(%alloc_68 : memref<8xf32, strided<[1]>>)
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %30 = arith.divsi %arg30, %c2_i32 : i32
            %31 = arith.muli %3, %c8_i32 : i32
            %32 = arith.divsi %arg29, %c2_i32 : i32
            %33 = arith.addi %31, %32 : i32
            %34 = arith.cmpi slt, %30, %33 : i32
            scf.if %34 {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              %37 = memref.load %alloc_65[%35, %36] : memref<16x32xf32, strided<[32, 1]>>
              %38 = arith.muli %3, %c8_i32 : i32
              %39 = arith.divsi %arg29, %c2_i32 : i32
              %40 = arith.addi %38, %39 : i32
              %41 = arith.index_cast %40 : i32 to index
              %42 = arith.divsi %arg30, %c2_i32 : i32
              %43 = arith.index_cast %42 : i32 to index
              %44 = memref.load %alloc_35[%41, %43] : memref<16x16xf32, strided<[16, 1]>>
              %45 = arith.mulf %37, %44 : f32
              %46 = arith.truncf %45 : f32 to bf16
              memref.store %46, %alloc_66[%35, %36] : memref<16x32xbf16, strided<[32, 1]>>
            } else {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_4, %alloc_66[%35, %36] : memref<16x32xbf16, strided<[32, 1]>>
            }
          }
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_67[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %33 = arith.divsi %arg29, %c2_i32 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = memref.load %alloc_68[%34] : memref<8xf32, strided<[1]>>
            %36 = arith.mulf %32, %35 : f32
            memref.store %36, %alloc_67[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %subview_142 = memref.subview %7[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_66, %subview_142 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        %subview_143 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %alloc_67, %subview_143 : memref<16x64xf32, strided<[64, 1]>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      }
      %subview_136 = memref.subview %7[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_136, %alloc_34 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_137 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %subview_137, %alloc_31 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%alloc_34, %alloc_31, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_69 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_138 = memref.subview %8[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_69, %subview_138 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_139 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_139, %alloc_70 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_140 = memref.subview %8[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_140, %alloc_71 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_141 = memref.subview %4[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_141, %alloc_72 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_142 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
        memref.copy %subview_142, %alloc_73 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
        %25 = arith.muli %arg28, %c16_i32 : i32
        %26 = arith.muli %3, %c8_i32 : i32
        %27 = arith.addi %25, %26 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.index_cast %1 : i32 to index
        %subview_143 = memref.subview %reinterpret_cast_14[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_143, %alloc_74 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_70[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %33 = memref.load %alloc_71[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %34 = arith.addf %32, %33 : f32
            memref.store %34, %alloc_70[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        scf.for %arg29 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_4, %alloc_75[%30, %31] : memref<8x64xbf16, strided<[64, 1]>>
          }
        }
        %subview_144 = memref.subview %reinterpret_cast_18[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_75, %subview_144 : memref<8x64xbf16, strided<[64, 1]>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      }
    } {tilelangir.num_stages = 2 : i32}
    return
  }
}


// -----// IR Dump After TileLangIRInsertWorkspace (tilelangir-insert-workspace) ('func.func' operation: @mamba_mimo_fwd_kernel) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @mamba_mimo_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xbf16, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: memref<?xf32, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: memref<?xbf16, #hivm.address_space<gm>>, %arg12: memref<?xf32, #hivm.address_space<gm>>, %arg13: memref<?xf32, #hivm.address_space<gm>>, %arg14: memref<?xf32, #hivm.address_space<gm>>, %arg15: memref<?xf32, #hivm.address_space<gm>>, %arg16: memref<?xf32, #hivm.address_space<gm>>, %arg17: memref<?xf32, #hivm.address_space<gm>>, %arg18: memref<?xbf16, #hivm.address_space<gm>>, %arg19: memref<?xf32, #hivm.address_space<gm>>, %arg20: memref<?xf32, #hivm.address_space<gm>>, %arg21: memref<?xbf16, #hivm.address_space<gm>>, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %cst = arith.constant -1.98412701E-4 : f32
    %cst_0 = arith.constant 0.00833333377 : f32
    %cst_1 = arith.constant -0.166666672 : f32
    %cst_2 = arith.constant -0.00138888892 : f32
    %cst_3 = arith.constant 0.0416666679 : f32
    %c2048 = arith.constant 2048 : index
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %c132 = arith.constant 132 : index
    %c33 = arith.constant 33 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c8192 = arith.constant 8192 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %cst_4 = arith.constant 0.000000e+00 : bf16
    %cst_5 = arith.constant -5.000000e-01 : f32
    %true = arith.constant true
    %c31_i32 = arith.constant 31 : i32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant -1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_10 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [4, 2, 64], strides: [%c128, %c64, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg14 to offset: [0], sizes: [1, 32, 4, 8], strides: [%c1024, %c32, %c8, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_13 = memref.reinterpret_cast %arg15 to offset: [0], sizes: [1, 4, 32], strides: [%c128, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_14 = memref.reinterpret_cast %arg11 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_15 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_16 = memref.reinterpret_cast %arg18 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_17 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_18 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_19 = memref.reinterpret_cast %arg17 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_20 = memref.reinterpret_cast %arg19 to offset: [0], sizes: [1, 4, 2, 16, 16], strides: [%c2048, %c512, %c256, %c16, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x32x32xf32>
    annotation.mark %4 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32>
    %5 = memref_ext.alloc_workspace() : memref<2x32x64xf32>
    annotation.mark %5 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32>
    %6 = memref_ext.alloc_workspace() : memref<2x32x32xf32>
    annotation.mark %6 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32>
    %7 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %7 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %8 = memref_ext.alloc_workspace() : memref<2x32x64xf32>
    annotation.mark %8 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32>
    %9 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %9 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %10 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %11 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %12 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %13 = memref_ext.alloc_workspace() : memref<2x32x64xbf16>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16>
    %14 = memref_ext.alloc_workspace() : memref<2x32x64xbf16>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16>
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_21 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_22 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_23 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_24 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
    %alloc_25 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
    %alloc_26 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vbrc ins(%cst_7 : f32) outs(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>)
    %15 = arith.index_cast %1 : i32 to index
    %subview = memref.subview %reinterpret_cast_10[%15, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_26 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vcast ins(%alloc_26 : memref<2x64xf32, strided<[64, 1]>>) outs(%alloc_25 : memref<2x64xbf16, strided<[64, 1]>>)
    %subview_27 = memref.subview %reinterpret_cast_11[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_27, %alloc_22 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_22 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>>)
    %subview_28 = memref.subview %reinterpret_cast_15[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_28, %alloc_23 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc_21 : memref<2x32xbf16, strided<[32, 1]>>)
    scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %alloc_29 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_30 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_31 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_32 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_33 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_34 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_35 = memref.alloc() : memref<16x16xf32, strided<[16, 1]>>
      %alloc_36 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_37 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_38 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_39 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_40 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_41 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_42 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_43 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_44 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_45 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_46 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_47 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_48 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_49 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_50 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_51 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_52 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_53 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_54 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_55 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_56 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_57 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_58 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_59 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_60 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_61 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_62 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_63 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_64 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_65 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_66 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
      %alloc_67 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_68 = memref.alloc() : memref<8xf32, strided<[1]>>
      %alloc_69 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_70 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_71 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_72 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_73 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_74 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_75 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_76 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_77 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_78 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
      %alloc_79 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_80 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_81 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_82 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_83 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_84 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_85 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_86 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_87 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_88 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_89 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_90 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_91 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_92 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %16 = arith.index_cast %1 : i32 to index
      %17 = arith.index_cast %arg28 : i32 to index
      %subview_93 = memref.subview %reinterpret_cast_20[0, %16, %17, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_93, %alloc_35 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>>
      %18 = arith.muli %arg28, %c16_i32 : i32
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_94 = memref.subview %reinterpret_cast_16[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_94, %alloc_37 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xbf16, strided<[1]>>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      hivm.hir.vmul ins(%alloc_36, %cst_8 : memref<16xf32, strided<[1]>>, f32) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %alloc_95 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_36 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_95 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_95 : memref<16xf32>) outs(%alloc_95 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_95, %cst_6 : memref<16xf32>, f32) outs(%alloc_95 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_95 : f32, memref<16xf32>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %subview_96 = memref.subview %reinterpret_cast_19[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_96, %alloc_39 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_39 : memref<16xf32, strided<[1]>>) outs(%alloc_38 : memref<16xbf16, strided<[1]>>)
      hivm.hir.vbrc ins(%cst_4 : bf16) outs(%alloc_40 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.muli %arg28, %c16_i32 : i32
        %24 = arith.addi %23, %arg29 : i32
        %25 = arith.cmpi slt, %24, %c31_i32 : i32
        scf.if %25 {
          %26 = arith.index_cast %arg29 : i32 to index
          %27 = memref.load %alloc_38[%26] : memref<16xbf16, strided<[1]>>
          %28 = arith.extf %27 : bf16 to f32
          %29 = memref.load %alloc_36[%26] : memref<16xf32, strided<[1]>>
          %30 = arith.mulf %28, %29 : f32
          %31 = arith.truncf %30 : f32 to bf16
          memref.store %31, %alloc_40[%26] : memref<16xbf16, strided<[1]>>
        }
      }
      memref.copy %alloc_40, %alloc_41 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %21 = arith.index_cast %18 : i32 to index
      %subview_97 = memref.subview %reinterpret_cast_16[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_97, %alloc_43 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_43 : memref<16xbf16, strided<[1]>>) outs(%alloc_42 : memref<16xf32, strided<[1]>>)
      %alloc_98 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_42 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_98 : memref<16xf32>) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_98, %cst_6 : memref<16xf32>, f32) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_98 : f32, memref<16xf32>) outs(%alloc_42 : memref<16xf32, strided<[1]>>)
      %subview_99 = memref.subview %reinterpret_cast_19[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_99, %alloc_45 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_45 : memref<16xf32, strided<[1]>>) outs(%alloc_44 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_44[%23] : memref<16xbf16, strided<[1]>>
        %25 = arith.extf %24 : bf16 to f32
        %26 = memref.load %alloc_42[%23] : memref<16xf32, strided<[1]>>
        %27 = arith.mulf %25, %26 : f32
        memref.store %27, %alloc_46[%23] : memref<16xf32, strided<[1]>>
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_46[%23] : memref<16xf32, strided<[1]>>
        %25 = memref.load %alloc_41[%23] : memref<16xbf16, strided<[1]>>
        %26 = arith.extf %25 : bf16 to f32
        %27 = arith.addf %24, %26 : f32
        %28 = arith.truncf %27 : f32 to bf16
        memref.store %28, %alloc_47[%23] : memref<16xbf16, strided<[1]>>
      }
      memref.copy %alloc_47, %alloc_48 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.muli %arg28, %c16_i32 : i32
        %24 = arith.addi %23, %arg29 : i32
        %25 = arith.index_cast %24 : i32 to index
        %26 = arith.index_cast %1 : i32 to index
        %subview_139 = memref.subview %reinterpret_cast_9[0, %25, %26, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %27 = arith.index_cast %arg29 : i32 to index
        %subview_140 = memref.subview %alloc_32[%27, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<64xbf16, strided<[1], offset: ?>>
        memref.copy %subview_139, %subview_140 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>>
      }
      memref.copy %alloc_32, %alloc_76 : memref<16x64xbf16, strided<[64, 1]>> to memref<16x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_100 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<16x1x64xbf16, strided<[64, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_100 : memref<16x1x64xbf16, strided<[64, 64, 1]>>) outs(%alloc_77 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [1]
      memref.copy %alloc_25, %alloc_78 : memref<2x64xbf16, strided<[64, 1]>> to memref<2x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_101 = memref.reinterpret_cast %alloc_78 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>> to memref<1x2x64xbf16, strided<[128, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_101 : memref<1x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_79 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_77 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_84 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_79 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_85 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vmul ins(%alloc_84, %alloc_85 : memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_86 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_86 : memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_49 : memref<16x2x64xbf16, strided<[128, 64, 1]>>)
      %reinterpret_cast_102 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_103 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %reinterpret_cast_102, %subview_103 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_104 = memref.subview %reinterpret_cast[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_104, %alloc_50 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc, %alloc_80 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_105 = memref.reinterpret_cast %alloc_80 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_105 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_81 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_50 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_87 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_81 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_88 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_87, %alloc_88 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_89 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_89 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_50 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_106 = memref.reinterpret_cast %alloc_50 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_107 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %reinterpret_cast_106, %subview_107 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_108 = memref.subview %reinterpret_cast_17[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_108, %alloc_51 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc_21, %alloc_82 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_109 = memref.reinterpret_cast %alloc_82 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_109 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_83 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_90 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_83 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_90, %alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_110 = memref.reinterpret_cast %alloc_51 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_111 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %reinterpret_cast_110, %subview_111 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_112 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_112, %alloc_29 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_113 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_113, %alloc_30 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%alloc_29, %alloc_30, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_52 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_114 = memref.subview %4[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_52, %subview_114 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      %subview_115 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_115, %alloc_53 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.muli %arg29, %c2_i32 : i32
            %24 = arith.addi %23, %arg30 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = arith.index_cast %arg31 : i32 to index
            %27 = memref.load %alloc_53[%25, %26] : memref<32x32xbf16, strided<[32, 1]>>
            %28 = arith.index_cast %arg29 : i32 to index
            %29 = arith.index_cast %arg30 : i32 to index
            memref.store %27, %alloc_54[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %subview_116 = memref.subview %reinterpret_cast_12[0, %21, %16, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_116, %alloc_55 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>>
      %alloc_117 = memref.alloc() : memref<16x8xf32>
      %alloc_118 = memref.alloc() : memref<16x8xf32>
      %alloc_119 = memref.alloc() : memref<16x8xf32>
      %alloc_120 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_55, %alloc_55 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_117 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_117, %alloc_117 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_118 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_117, %alloc_118 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_119 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_117, %cst_5 : memref<16x8xf32>, f32) outs(%alloc_117 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_118, %cst_3 : memref<16x8xf32>, f32) outs(%alloc_118 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_119, %cst_2 : memref<16x8xf32>, f32) outs(%alloc_119 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_117, %cst_6 : memref<16x8xf32>, f32) outs(%alloc_120 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_118, %alloc_120 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_120 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_119, %alloc_120 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_56 : memref<16x8xf32, strided<[8, 1]>>)
      %alloc_121 = memref.alloc() : memref<16x8xf32>
      %alloc_122 = memref.alloc() : memref<16x8xf32>
      %alloc_123 = memref.alloc() : memref<16x8xf32>
      %alloc_124 = memref.alloc() : memref<16x8xf32>
      %alloc_125 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_55, %alloc_55 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_121 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_121, %alloc_55 : memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_122 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_122, %alloc_121 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_123 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_123, %alloc_121 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_124 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_122, %cst_1 : memref<16x8xf32>, f32) outs(%alloc_122 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_123, %cst_0 : memref<16x8xf32>, f32) outs(%alloc_123 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_124, %cst : memref<16x8xf32>, f32) outs(%alloc_124 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_55, %alloc_122 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) outs(%alloc_125 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_123, %alloc_125 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_125 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_124, %alloc_125 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_57 : memref<16x8xf32, strided<[8, 1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_56[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_54[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.subf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            memref.store %31, %alloc_53[%34, %24] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_57[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_54[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.addf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = arith.addi %arg31, %c16_i32 : i32
            %36 = arith.index_cast %35 : i32 to index
            memref.store %31, %alloc_53[%34, %36] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      %subview_126 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_53, %subview_126 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>) outs(%alloc_59 : memref<32x64xbf16, strided<[64, 1]>>)
      %subview_127 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %alloc_59, %subview_127 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_128 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_128, %alloc_29 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_129 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %subview_129, %alloc_33 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%alloc_29, %alloc_33, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_58 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_130 = memref.subview %5[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_58, %subview_130 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      %subview_131 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_131, %alloc_60 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.muli %arg29, %c2_i32 : i32
            %24 = arith.addi %23, %arg30 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = arith.index_cast %arg31 : i32 to index
            %27 = memref.load %alloc_60[%25, %26] : memref<32x32xbf16, strided<[32, 1]>>
            %28 = arith.index_cast %arg29 : i32 to index
            %29 = arith.index_cast %arg30 : i32 to index
            memref.store %27, %alloc_61[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_56[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_61[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.subf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            memref.store %31, %alloc_60[%34, %24] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_57[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_61[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.addf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = arith.addi %arg31, %c16_i32 : i32
            %36 = arith.index_cast %35 : i32 to index
            memref.store %31, %alloc_60[%34, %36] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      memref.copy %alloc_60, %alloc_62 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_62 : memref<32x32xbf16, strided<[32, 1]>>) outs(%alloc_63 : memref<32x32xf32, strided<[32, 1]>>)
      scf.for %arg29 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
          %23 = arith.index_cast %arg29 : i32 to index
          %24 = arith.index_cast %arg30 : i32 to index
          %25 = memref.load %alloc_63[%23, %24] : memref<32x32xf32, strided<[32, 1]>>
          %26 = arith.divsi %arg29, %c2_i32 : i32
          %27 = arith.index_cast %26 : i32 to index
          %28 = memref.load %alloc_48[%27] : memref<16xbf16, strided<[1]>>
          %29 = arith.extf %28 : bf16 to f32
          %30 = arith.mulf %25, %29 : f32
          memref.store %30, %alloc_63[%23, %24] : memref<32x32xf32, strided<[32, 1]>>
        }
      }
      hivm.hir.vcast ins(%alloc_63 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc_62 : memref<32x32xbf16, strided<[32, 1]>>)
      %subview_132 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_62, %subview_132 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_133 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_133, %alloc_29 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_134 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_134, %alloc_30 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%alloc_29, %alloc_30, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_64 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_135 = memref.subview %6[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_64, %subview_135 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      hivm.hir.vexp ins(%alloc_35 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_35 : memref<16x16xf32, strided<[16, 1]>>)
      %22 = arith.cmpi slt, %3, %c2_i32 : i32
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_139 = memref.subview %6[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_139, %alloc_65 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_140 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_140, %alloc_67 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %25 = arith.index_cast %1 : i32 to index
        %26 = arith.muli %arg28, %c16_i32 : i32
        %27 = arith.muli %3, %c8_i32 : i32
        %28 = arith.addi %26, %27 : i32
        %29 = arith.index_cast %28 : i32 to index
        %subview_141 = memref.subview %reinterpret_cast_13[0, %25, %29] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_141, %alloc_68 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>>
        hivm.hir.vexp ins(%alloc_68 : memref<8xf32, strided<[1]>>) outs(%alloc_68 : memref<8xf32, strided<[1]>>)
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %30 = arith.divsi %arg30, %c2_i32 : i32
            %31 = arith.muli %3, %c8_i32 : i32
            %32 = arith.divsi %arg29, %c2_i32 : i32
            %33 = arith.addi %31, %32 : i32
            %34 = arith.cmpi slt, %30, %33 : i32
            scf.if %34 {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              %37 = memref.load %alloc_65[%35, %36] : memref<16x32xf32, strided<[32, 1]>>
              %38 = arith.muli %3, %c8_i32 : i32
              %39 = arith.divsi %arg29, %c2_i32 : i32
              %40 = arith.addi %38, %39 : i32
              %41 = arith.index_cast %40 : i32 to index
              %42 = arith.divsi %arg30, %c2_i32 : i32
              %43 = arith.index_cast %42 : i32 to index
              %44 = memref.load %alloc_35[%41, %43] : memref<16x16xf32, strided<[16, 1]>>
              %45 = arith.mulf %37, %44 : f32
              %46 = arith.truncf %45 : f32 to bf16
              memref.store %46, %alloc_66[%35, %36] : memref<16x32xbf16, strided<[32, 1]>>
            } else {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_4, %alloc_66[%35, %36] : memref<16x32xbf16, strided<[32, 1]>>
            }
          }
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_67[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %33 = arith.divsi %arg29, %c2_i32 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = memref.load %alloc_68[%34] : memref<8xf32, strided<[1]>>
            %36 = arith.mulf %32, %35 : f32
            memref.store %36, %alloc_67[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %subview_142 = memref.subview %7[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_66, %subview_142 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        %subview_143 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %alloc_67, %subview_143 : memref<16x64xf32, strided<[64, 1]>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      }
      %subview_136 = memref.subview %7[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_136, %alloc_34 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_137 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %subview_137, %alloc_31 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%alloc_34, %alloc_31, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_69 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_138 = memref.subview %8[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_69, %subview_138 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_139 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_139, %alloc_70 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_140 = memref.subview %8[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_140, %alloc_71 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_141 = memref.subview %4[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_141, %alloc_72 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_142 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
        memref.copy %subview_142, %alloc_73 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
        %25 = arith.muli %arg28, %c16_i32 : i32
        %26 = arith.muli %3, %c8_i32 : i32
        %27 = arith.addi %25, %26 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.index_cast %1 : i32 to index
        %subview_143 = memref.subview %reinterpret_cast_14[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_143, %alloc_74 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_70[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %33 = memref.load %alloc_71[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %34 = arith.addf %32, %33 : f32
            memref.store %34, %alloc_70[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        scf.for %arg29 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_4, %alloc_75[%30, %31] : memref<8x64xbf16, strided<[64, 1]>>
          }
        }
        %subview_144 = memref.subview %reinterpret_cast_18[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_75, %subview_144 : memref<8x64xbf16, strided<[64, 1]>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      }
    } {tilelangir.num_stages = 2 : i32}
    return
  }
}


// -----// IR Dump After TileLangIRMarkMultiBuffer (tilelangir-mark-multibuffer) ('func.func' operation: @mamba_mimo_fwd_kernel) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @mamba_mimo_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xbf16, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: memref<?xf32, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: memref<?xbf16, #hivm.address_space<gm>>, %arg12: memref<?xf32, #hivm.address_space<gm>>, %arg13: memref<?xf32, #hivm.address_space<gm>>, %arg14: memref<?xf32, #hivm.address_space<gm>>, %arg15: memref<?xf32, #hivm.address_space<gm>>, %arg16: memref<?xf32, #hivm.address_space<gm>>, %arg17: memref<?xf32, #hivm.address_space<gm>>, %arg18: memref<?xbf16, #hivm.address_space<gm>>, %arg19: memref<?xf32, #hivm.address_space<gm>>, %arg20: memref<?xf32, #hivm.address_space<gm>>, %arg21: memref<?xbf16, #hivm.address_space<gm>>, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %cst = arith.constant -1.98412701E-4 : f32
    %cst_0 = arith.constant 0.00833333377 : f32
    %cst_1 = arith.constant -0.166666672 : f32
    %cst_2 = arith.constant -0.00138888892 : f32
    %cst_3 = arith.constant 0.0416666679 : f32
    %c2048 = arith.constant 2048 : index
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %c132 = arith.constant 132 : index
    %c33 = arith.constant 33 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c8192 = arith.constant 8192 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %cst_4 = arith.constant 0.000000e+00 : bf16
    %cst_5 = arith.constant -5.000000e-01 : f32
    %true = arith.constant true
    %c31_i32 = arith.constant 31 : i32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant -1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_10 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [4, 2, 64], strides: [%c128, %c64, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg14 to offset: [0], sizes: [1, 32, 4, 8], strides: [%c1024, %c32, %c8, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_13 = memref.reinterpret_cast %arg15 to offset: [0], sizes: [1, 4, 32], strides: [%c128, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_14 = memref.reinterpret_cast %arg11 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_15 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_16 = memref.reinterpret_cast %arg18 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_17 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_18 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_19 = memref.reinterpret_cast %arg17 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_20 = memref.reinterpret_cast %arg19 to offset: [0], sizes: [1, 4, 2, 16, 16], strides: [%c2048, %c512, %c256, %c16, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x32x32xf32>
    annotation.mark %4 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32>
    %5 = memref_ext.alloc_workspace() : memref<2x32x64xf32>
    annotation.mark %5 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32>
    %6 = memref_ext.alloc_workspace() : memref<2x32x32xf32>
    annotation.mark %6 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32>
    %7 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %7 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %8 = memref_ext.alloc_workspace() : memref<2x32x64xf32>
    annotation.mark %8 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32>
    %9 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %9 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %10 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %11 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %12 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %13 = memref_ext.alloc_workspace() : memref<2x32x64xbf16>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16>
    %14 = memref_ext.alloc_workspace() : memref<2x32x64xbf16>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16>
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_21 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_22 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_23 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_24 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
    %alloc_25 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
    %alloc_26 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vbrc ins(%cst_7 : f32) outs(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>)
    %15 = arith.index_cast %1 : i32 to index
    %subview = memref.subview %reinterpret_cast_10[%15, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_26 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vcast ins(%alloc_26 : memref<2x64xf32, strided<[64, 1]>>) outs(%alloc_25 : memref<2x64xbf16, strided<[64, 1]>>)
    %subview_27 = memref.subview %reinterpret_cast_11[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_27, %alloc_22 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_22 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>>)
    %subview_28 = memref.subview %reinterpret_cast_15[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_28, %alloc_23 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc_21 : memref<2x32xbf16, strided<[32, 1]>>)
    scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %alloc_29 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_30 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_31 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_32 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_33 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_34 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_35 = memref.alloc() : memref<16x16xf32, strided<[16, 1]>>
      %alloc_36 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_37 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_38 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_39 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_40 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_41 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_42 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_43 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_44 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_45 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_46 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_47 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_48 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_49 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_50 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_51 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_52 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_53 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_54 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_55 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_56 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_57 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_58 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_59 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_60 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_61 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_62 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_63 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_64 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_65 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_66 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
      %alloc_67 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_68 = memref.alloc() : memref<8xf32, strided<[1]>>
      %alloc_69 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_70 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_71 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_72 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_73 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_74 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_75 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_76 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_77 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_78 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
      %alloc_79 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_80 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_81 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_82 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_83 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_84 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_85 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_86 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_87 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_88 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_89 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_90 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_91 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_92 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %16 = arith.index_cast %1 : i32 to index
      %17 = arith.index_cast %arg28 : i32 to index
      %subview_93 = memref.subview %reinterpret_cast_20[0, %16, %17, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_93, %alloc_35 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>>
      %18 = arith.muli %arg28, %c16_i32 : i32
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_94 = memref.subview %reinterpret_cast_16[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_94, %alloc_37 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xbf16, strided<[1]>>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      hivm.hir.vmul ins(%alloc_36, %cst_8 : memref<16xf32, strided<[1]>>, f32) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %alloc_95 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_36 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_95 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_95 : memref<16xf32>) outs(%alloc_95 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_95, %cst_6 : memref<16xf32>, f32) outs(%alloc_95 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_95 : f32, memref<16xf32>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %subview_96 = memref.subview %reinterpret_cast_19[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_96, %alloc_39 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_39 : memref<16xf32, strided<[1]>>) outs(%alloc_38 : memref<16xbf16, strided<[1]>>)
      hivm.hir.vbrc ins(%cst_4 : bf16) outs(%alloc_40 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.muli %arg28, %c16_i32 : i32
        %24 = arith.addi %23, %arg29 : i32
        %25 = arith.cmpi slt, %24, %c31_i32 : i32
        scf.if %25 {
          %26 = arith.index_cast %arg29 : i32 to index
          %27 = memref.load %alloc_38[%26] : memref<16xbf16, strided<[1]>>
          %28 = arith.extf %27 : bf16 to f32
          %29 = memref.load %alloc_36[%26] : memref<16xf32, strided<[1]>>
          %30 = arith.mulf %28, %29 : f32
          %31 = arith.truncf %30 : f32 to bf16
          memref.store %31, %alloc_40[%26] : memref<16xbf16, strided<[1]>>
        }
      }
      memref.copy %alloc_40, %alloc_41 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %21 = arith.index_cast %18 : i32 to index
      %subview_97 = memref.subview %reinterpret_cast_16[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_97, %alloc_43 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_43 : memref<16xbf16, strided<[1]>>) outs(%alloc_42 : memref<16xf32, strided<[1]>>)
      %alloc_98 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_42 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_98 : memref<16xf32>) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_98, %cst_6 : memref<16xf32>, f32) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_98 : f32, memref<16xf32>) outs(%alloc_42 : memref<16xf32, strided<[1]>>)
      %subview_99 = memref.subview %reinterpret_cast_19[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_99, %alloc_45 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_45 : memref<16xf32, strided<[1]>>) outs(%alloc_44 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_44[%23] : memref<16xbf16, strided<[1]>>
        %25 = arith.extf %24 : bf16 to f32
        %26 = memref.load %alloc_42[%23] : memref<16xf32, strided<[1]>>
        %27 = arith.mulf %25, %26 : f32
        memref.store %27, %alloc_46[%23] : memref<16xf32, strided<[1]>>
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_46[%23] : memref<16xf32, strided<[1]>>
        %25 = memref.load %alloc_41[%23] : memref<16xbf16, strided<[1]>>
        %26 = arith.extf %25 : bf16 to f32
        %27 = arith.addf %24, %26 : f32
        %28 = arith.truncf %27 : f32 to bf16
        memref.store %28, %alloc_47[%23] : memref<16xbf16, strided<[1]>>
      }
      memref.copy %alloc_47, %alloc_48 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.muli %arg28, %c16_i32 : i32
        %24 = arith.addi %23, %arg29 : i32
        %25 = arith.index_cast %24 : i32 to index
        %26 = arith.index_cast %1 : i32 to index
        %subview_139 = memref.subview %reinterpret_cast_9[0, %25, %26, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %27 = arith.index_cast %arg29 : i32 to index
        %subview_140 = memref.subview %alloc_32[%27, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<64xbf16, strided<[1], offset: ?>>
        memref.copy %subview_139, %subview_140 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>>
      }
      memref.copy %alloc_32, %alloc_76 : memref<16x64xbf16, strided<[64, 1]>> to memref<16x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_100 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<16x1x64xbf16, strided<[64, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_100 : memref<16x1x64xbf16, strided<[64, 64, 1]>>) outs(%alloc_77 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [1]
      memref.copy %alloc_25, %alloc_78 : memref<2x64xbf16, strided<[64, 1]>> to memref<2x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_101 = memref.reinterpret_cast %alloc_78 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>> to memref<1x2x64xbf16, strided<[128, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_101 : memref<1x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_79 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_77 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_84 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_79 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_85 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vmul ins(%alloc_84, %alloc_85 : memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_86 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_86 : memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_49 : memref<16x2x64xbf16, strided<[128, 64, 1]>>)
      %reinterpret_cast_102 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_103 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %reinterpret_cast_102, %subview_103 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_104 = memref.subview %reinterpret_cast[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_104, %alloc_50 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc, %alloc_80 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_105 = memref.reinterpret_cast %alloc_80 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_105 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_81 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_50 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_87 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_81 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_88 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_87, %alloc_88 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_89 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_89 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_50 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_106 = memref.reinterpret_cast %alloc_50 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_107 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %reinterpret_cast_106, %subview_107 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_108 = memref.subview %reinterpret_cast_17[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_108, %alloc_51 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc_21, %alloc_82 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_109 = memref.reinterpret_cast %alloc_82 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_109 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_83 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_90 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_83 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_90, %alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_110 = memref.reinterpret_cast %alloc_51 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_111 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %reinterpret_cast_110, %subview_111 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_112 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_112, %alloc_29 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_113 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_113, %alloc_30 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%alloc_29, %alloc_30, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_52 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_114 = memref.subview %4[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_52, %subview_114 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      %subview_115 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_115, %alloc_53 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.muli %arg29, %c2_i32 : i32
            %24 = arith.addi %23, %arg30 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = arith.index_cast %arg31 : i32 to index
            %27 = memref.load %alloc_53[%25, %26] : memref<32x32xbf16, strided<[32, 1]>>
            %28 = arith.index_cast %arg29 : i32 to index
            %29 = arith.index_cast %arg30 : i32 to index
            memref.store %27, %alloc_54[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %subview_116 = memref.subview %reinterpret_cast_12[0, %21, %16, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_116, %alloc_55 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>>
      %alloc_117 = memref.alloc() : memref<16x8xf32>
      %alloc_118 = memref.alloc() : memref<16x8xf32>
      %alloc_119 = memref.alloc() : memref<16x8xf32>
      %alloc_120 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_55, %alloc_55 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_117 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_117, %alloc_117 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_118 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_117, %alloc_118 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_119 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_117, %cst_5 : memref<16x8xf32>, f32) outs(%alloc_117 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_118, %cst_3 : memref<16x8xf32>, f32) outs(%alloc_118 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_119, %cst_2 : memref<16x8xf32>, f32) outs(%alloc_119 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_117, %cst_6 : memref<16x8xf32>, f32) outs(%alloc_120 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_118, %alloc_120 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_120 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_119, %alloc_120 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_56 : memref<16x8xf32, strided<[8, 1]>>)
      %alloc_121 = memref.alloc() : memref<16x8xf32>
      %alloc_122 = memref.alloc() : memref<16x8xf32>
      %alloc_123 = memref.alloc() : memref<16x8xf32>
      %alloc_124 = memref.alloc() : memref<16x8xf32>
      %alloc_125 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_55, %alloc_55 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_121 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_121, %alloc_55 : memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_122 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_122, %alloc_121 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_123 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_123, %alloc_121 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_124 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_122, %cst_1 : memref<16x8xf32>, f32) outs(%alloc_122 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_123, %cst_0 : memref<16x8xf32>, f32) outs(%alloc_123 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_124, %cst : memref<16x8xf32>, f32) outs(%alloc_124 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_55, %alloc_122 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) outs(%alloc_125 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_123, %alloc_125 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_125 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_124, %alloc_125 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_57 : memref<16x8xf32, strided<[8, 1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_56[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_54[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.subf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            memref.store %31, %alloc_53[%34, %24] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_57[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_54[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.addf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = arith.addi %arg31, %c16_i32 : i32
            %36 = arith.index_cast %35 : i32 to index
            memref.store %31, %alloc_53[%34, %36] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      %subview_126 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_53, %subview_126 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>) outs(%alloc_59 : memref<32x64xbf16, strided<[64, 1]>>)
      %subview_127 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %alloc_59, %subview_127 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_128 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_128, %alloc_29 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_129 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %subview_129, %alloc_33 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%alloc_29, %alloc_33, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_58 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_130 = memref.subview %5[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_58, %subview_130 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      %subview_131 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_131, %alloc_60 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.muli %arg29, %c2_i32 : i32
            %24 = arith.addi %23, %arg30 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = arith.index_cast %arg31 : i32 to index
            %27 = memref.load %alloc_60[%25, %26] : memref<32x32xbf16, strided<[32, 1]>>
            %28 = arith.index_cast %arg29 : i32 to index
            %29 = arith.index_cast %arg30 : i32 to index
            memref.store %27, %alloc_61[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_56[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_61[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.subf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            memref.store %31, %alloc_60[%34, %24] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg31 : i32 to index
            %25 = memref.load %alloc_57[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
            %26 = arith.index_cast %arg30 : i32 to index
            %27 = memref.load %alloc_61[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %28 = arith.extf %27 : bf16 to f32
            %29 = arith.mulf %25, %28 : f32
            %30 = arith.addf %29, %29 : f32
            %31 = arith.truncf %30 : f32 to bf16
            %32 = arith.muli %arg29, %c2_i32 : i32
            %33 = arith.addi %32, %arg30 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = arith.addi %arg31, %c16_i32 : i32
            %36 = arith.index_cast %35 : i32 to index
            memref.store %31, %alloc_60[%34, %36] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      memref.copy %alloc_60, %alloc_62 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_62 : memref<32x32xbf16, strided<[32, 1]>>) outs(%alloc_63 : memref<32x32xf32, strided<[32, 1]>>)
      scf.for %arg29 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
          %23 = arith.index_cast %arg29 : i32 to index
          %24 = arith.index_cast %arg30 : i32 to index
          %25 = memref.load %alloc_63[%23, %24] : memref<32x32xf32, strided<[32, 1]>>
          %26 = arith.divsi %arg29, %c2_i32 : i32
          %27 = arith.index_cast %26 : i32 to index
          %28 = memref.load %alloc_48[%27] : memref<16xbf16, strided<[1]>>
          %29 = arith.extf %28 : bf16 to f32
          %30 = arith.mulf %25, %29 : f32
          memref.store %30, %alloc_63[%23, %24] : memref<32x32xf32, strided<[32, 1]>>
        }
      }
      hivm.hir.vcast ins(%alloc_63 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc_62 : memref<32x32xbf16, strided<[32, 1]>>)
      %subview_132 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_62, %subview_132 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_133 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_133, %alloc_29 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_134 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_134, %alloc_30 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%alloc_29, %alloc_30, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_64 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_135 = memref.subview %6[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_64, %subview_135 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      hivm.hir.vexp ins(%alloc_35 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_35 : memref<16x16xf32, strided<[16, 1]>>)
      %22 = arith.cmpi slt, %3, %c2_i32 : i32
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_139 = memref.subview %6[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_139, %alloc_65 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_140 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_140, %alloc_67 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %25 = arith.index_cast %1 : i32 to index
        %26 = arith.muli %arg28, %c16_i32 : i32
        %27 = arith.muli %3, %c8_i32 : i32
        %28 = arith.addi %26, %27 : i32
        %29 = arith.index_cast %28 : i32 to index
        %subview_141 = memref.subview %reinterpret_cast_13[0, %25, %29] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_141, %alloc_68 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>>
        hivm.hir.vexp ins(%alloc_68 : memref<8xf32, strided<[1]>>) outs(%alloc_68 : memref<8xf32, strided<[1]>>)
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %30 = arith.divsi %arg30, %c2_i32 : i32
            %31 = arith.muli %3, %c8_i32 : i32
            %32 = arith.divsi %arg29, %c2_i32 : i32
            %33 = arith.addi %31, %32 : i32
            %34 = arith.cmpi slt, %30, %33 : i32
            scf.if %34 {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              %37 = memref.load %alloc_65[%35, %36] : memref<16x32xf32, strided<[32, 1]>>
              %38 = arith.muli %3, %c8_i32 : i32
              %39 = arith.divsi %arg29, %c2_i32 : i32
              %40 = arith.addi %38, %39 : i32
              %41 = arith.index_cast %40 : i32 to index
              %42 = arith.divsi %arg30, %c2_i32 : i32
              %43 = arith.index_cast %42 : i32 to index
              %44 = memref.load %alloc_35[%41, %43] : memref<16x16xf32, strided<[16, 1]>>
              %45 = arith.mulf %37, %44 : f32
              %46 = arith.truncf %45 : f32 to bf16
              memref.store %46, %alloc_66[%35, %36] : memref<16x32xbf16, strided<[32, 1]>>
            } else {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_4, %alloc_66[%35, %36] : memref<16x32xbf16, strided<[32, 1]>>
            }
          }
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_67[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %33 = arith.divsi %arg29, %c2_i32 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = memref.load %alloc_68[%34] : memref<8xf32, strided<[1]>>
            %36 = arith.mulf %32, %35 : f32
            memref.store %36, %alloc_67[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %subview_142 = memref.subview %7[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_66, %subview_142 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        %subview_143 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %alloc_67, %subview_143 : memref<16x64xf32, strided<[64, 1]>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      }
      %subview_136 = memref.subview %7[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_136, %alloc_34 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_137 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %subview_137, %alloc_31 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%alloc_34, %alloc_31, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_69 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_138 = memref.subview %8[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_69, %subview_138 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_139 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_139, %alloc_70 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_140 = memref.subview %8[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_140, %alloc_71 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_141 = memref.subview %4[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_141, %alloc_72 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_142 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
        memref.copy %subview_142, %alloc_73 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
        %25 = arith.muli %arg28, %c16_i32 : i32
        %26 = arith.muli %3, %c8_i32 : i32
        %27 = arith.addi %25, %26 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.index_cast %1 : i32 to index
        %subview_143 = memref.subview %reinterpret_cast_14[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_143, %alloc_74 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_70[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %33 = memref.load %alloc_71[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %34 = arith.addf %32, %33 : f32
            memref.store %34, %alloc_70[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        scf.for %arg29 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_4, %alloc_75[%30, %31] : memref<8x64xbf16, strided<[64, 1]>>
          }
        }
        %subview_144 = memref.subview %reinterpret_cast_18[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_75, %subview_144 : memref<8x64xbf16, strided<[64, 1]>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      }
    } {tilelangir.num_stages = 2 : i32}
    return
  }
}


// -----// IR Dump After TileLangIRCVSplit (tilelangir-cv-split) ('func.func' operation: @mamba_mimo_fwd_kernel) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @mamba_mimo_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xbf16, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: memref<?xf32, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: memref<?xbf16, #hivm.address_space<gm>>, %arg12: memref<?xf32, #hivm.address_space<gm>>, %arg13: memref<?xf32, #hivm.address_space<gm>>, %arg14: memref<?xf32, #hivm.address_space<gm>>, %arg15: memref<?xf32, #hivm.address_space<gm>>, %arg16: memref<?xf32, #hivm.address_space<gm>>, %arg17: memref<?xf32, #hivm.address_space<gm>>, %arg18: memref<?xbf16, #hivm.address_space<gm>>, %arg19: memref<?xf32, #hivm.address_space<gm>>, %arg20: memref<?xf32, #hivm.address_space<gm>>, %arg21: memref<?xbf16, #hivm.address_space<gm>>, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %cst = arith.constant -1.98412701E-4 : f32
    %cst_0 = arith.constant 0.00833333377 : f32
    %cst_1 = arith.constant -0.166666672 : f32
    %cst_2 = arith.constant -0.00138888892 : f32
    %cst_3 = arith.constant 0.0416666679 : f32
    %c2048 = arith.constant 2048 : index
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %c132 = arith.constant 132 : index
    %c33 = arith.constant 33 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c8192 = arith.constant 8192 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %cst_4 = arith.constant 0.000000e+00 : bf16
    %cst_5 = arith.constant -5.000000e-01 : f32
    %true = arith.constant true
    %c31_i32 = arith.constant 31 : i32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant -1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_10 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [4, 2, 64], strides: [%c128, %c64, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg14 to offset: [0], sizes: [1, 32, 4, 8], strides: [%c1024, %c32, %c8, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_13 = memref.reinterpret_cast %arg15 to offset: [0], sizes: [1, 4, 32], strides: [%c128, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_14 = memref.reinterpret_cast %arg11 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_15 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_16 = memref.reinterpret_cast %arg18 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_17 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_18 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_19 = memref.reinterpret_cast %arg17 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_20 = memref.reinterpret_cast %arg19 to offset: [0], sizes: [1, 4, 2, 16, 16], strides: [%c2048, %c512, %c256, %c16, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x32x32xf32>
    annotation.mark %4 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32>
    %5 = memref_ext.alloc_workspace() : memref<2x32x64xf32>
    annotation.mark %5 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32>
    %6 = memref_ext.alloc_workspace() : memref<2x32x32xf32>
    annotation.mark %6 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32>
    %7 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %7 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %8 = memref_ext.alloc_workspace() : memref<2x32x64xf32>
    annotation.mark %8 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32>
    %9 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %9 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %10 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %11 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %12 = memref_ext.alloc_workspace() : memref<2x32x32xbf16>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16>
    %13 = memref_ext.alloc_workspace() : memref<2x32x64xbf16>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16>
    %14 = memref_ext.alloc_workspace() : memref<2x32x64xbf16>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16>
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_21 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_22 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_23 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_24 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
    %alloc_25 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
    %alloc_26 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vbrc ins(%cst_7 : f32) outs(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>)
    %15 = arith.index_cast %1 : i32 to index
    %subview = memref.subview %reinterpret_cast_10[%15, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_26 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vcast ins(%alloc_26 : memref<2x64xf32, strided<[64, 1]>>) outs(%alloc_25 : memref<2x64xbf16, strided<[64, 1]>>)
    %subview_27 = memref.subview %reinterpret_cast_11[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_27, %alloc_22 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_22 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>>)
    %subview_28 = memref.subview %reinterpret_cast_15[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_28, %alloc_23 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc_21 : memref<2x32xbf16, strided<[32, 1]>>)
    scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %alloc_29 = memref.alloc() : memref<16x16xf32, strided<[16, 1]>>
      %alloc_30 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_31 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_32 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_33 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_34 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_35 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_36 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_37 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_38 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_39 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_40 = memref.alloc() : memref<16xf32, strided<[1]>>
      %alloc_41 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_42 = memref.alloc() : memref<16xbf16, strided<[1]>>
      %alloc_43 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_44 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_45 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_46 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_47 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_48 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_49 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
      %alloc_50 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_51 = memref.alloc() : memref<8xf32, strided<[1]>>
      %alloc_52 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_53 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_54 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_55 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_56 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_57 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %16 = arith.index_cast %1 : i32 to index
      %17 = arith.index_cast %arg28 : i32 to index
      %subview_58 = memref.subview %reinterpret_cast_20[0, %16, %17, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_58, %alloc_29 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>>
      %18 = arith.muli %arg28, %c16_i32 : i32
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_59 = memref.subview %reinterpret_cast_16[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_59, %alloc_31 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_31 : memref<16xbf16, strided<[1]>>) outs(%alloc_30 : memref<16xf32, strided<[1]>>)
      hivm.hir.vmul ins(%alloc_30, %cst_8 : memref<16xf32, strided<[1]>>, f32) outs(%alloc_30 : memref<16xf32, strided<[1]>>)
      %alloc_60 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_30 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_60 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_60 : memref<16xf32>) outs(%alloc_60 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_60, %cst_6 : memref<16xf32>, f32) outs(%alloc_60 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_60 : f32, memref<16xf32>) outs(%alloc_30 : memref<16xf32, strided<[1]>>)
      %subview_61 = memref.subview %reinterpret_cast_19[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_61, %alloc_33 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_33 : memref<16xf32, strided<[1]>>) outs(%alloc_32 : memref<16xbf16, strided<[1]>>)
      hivm.hir.vbrc ins(%cst_4 : bf16) outs(%alloc_34 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.muli %arg28, %c16_i32 : i32
        %24 = arith.addi %23, %arg29 : i32
        %25 = arith.cmpi slt, %24, %c31_i32 : i32
        scf.if %25 {
          %26 = arith.index_cast %arg29 : i32 to index
          %27 = memref.load %alloc_32[%26] : memref<16xbf16, strided<[1]>>
          %28 = arith.extf %27 : bf16 to f32
          %29 = memref.load %alloc_30[%26] : memref<16xf32, strided<[1]>>
          %30 = arith.mulf %28, %29 : f32
          %31 = arith.truncf %30 : f32 to bf16
          memref.store %31, %alloc_34[%26] : memref<16xbf16, strided<[1]>>
        }
      }
      memref.copy %alloc_34, %alloc_35 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %21 = arith.index_cast %18 : i32 to index
      %subview_62 = memref.subview %reinterpret_cast_16[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_62, %alloc_37 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xbf16, strided<[1]>>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %alloc_63 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_36 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_63 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_63 : memref<16xf32>) outs(%alloc_63 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_63, %cst_6 : memref<16xf32>, f32) outs(%alloc_63 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_63 : f32, memref<16xf32>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %subview_64 = memref.subview %reinterpret_cast_19[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_64, %alloc_39 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_39 : memref<16xf32, strided<[1]>>) outs(%alloc_38 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_38[%23] : memref<16xbf16, strided<[1]>>
        %25 = arith.extf %24 : bf16 to f32
        %26 = memref.load %alloc_36[%23] : memref<16xf32, strided<[1]>>
        %27 = arith.mulf %25, %26 : f32
        memref.store %27, %alloc_40[%23] : memref<16xf32, strided<[1]>>
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_40[%23] : memref<16xf32, strided<[1]>>
        %25 = memref.load %alloc_35[%23] : memref<16xbf16, strided<[1]>>
        %26 = arith.extf %25 : bf16 to f32
        %27 = arith.addf %24, %26 : f32
        %28 = arith.truncf %27 : f32 to bf16
        memref.store %28, %alloc_41[%23] : memref<16xbf16, strided<[1]>>
      }
      memref.copy %alloc_41, %alloc_42 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %subview_65 = memref.subview %reinterpret_cast_12[0, %21, %16, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_65, %alloc_44 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>>
      %alloc_66 = memref.alloc() : memref<16x8xf32>
      %alloc_67 = memref.alloc() : memref<16x8xf32>
      %alloc_68 = memref.alloc() : memref<16x8xf32>
      %alloc_69 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_44, %alloc_44 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_66 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_66, %alloc_66 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_67 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_66, %alloc_67 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_68 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_66, %cst_5 : memref<16x8xf32>, f32) outs(%alloc_66 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_67, %cst_3 : memref<16x8xf32>, f32) outs(%alloc_67 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_68, %cst_2 : memref<16x8xf32>, f32) outs(%alloc_68 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_66, %cst_6 : memref<16x8xf32>, f32) outs(%alloc_69 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_67, %alloc_69 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_69 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_68, %alloc_69 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_45 : memref<16x8xf32, strided<[8, 1]>>)
      %alloc_70 = memref.alloc() : memref<16x8xf32>
      %alloc_71 = memref.alloc() : memref<16x8xf32>
      %alloc_72 = memref.alloc() : memref<16x8xf32>
      %alloc_73 = memref.alloc() : memref<16x8xf32>
      %alloc_74 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_44, %alloc_44 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_70 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_70, %alloc_44 : memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_71 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_71, %alloc_70 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_72 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_72, %alloc_70 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_73 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_71, %cst_1 : memref<16x8xf32>, f32) outs(%alloc_71 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_72, %cst_0 : memref<16x8xf32>, f32) outs(%alloc_72 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_73, %cst : memref<16x8xf32>, f32) outs(%alloc_73 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_44, %alloc_71 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) outs(%alloc_74 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_72, %alloc_74 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_74 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_73, %alloc_74 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_46 : memref<16x8xf32, strided<[8, 1]>>)
      hivm.hir.vexp ins(%alloc_29 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_29 : memref<16x16xf32, strided<[16, 1]>>)
      %22 = arith.cmpi slt, %3, %c2_i32 : i32
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_75 = memref.subview %6[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_75, %alloc_48 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_76 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_76, %alloc_50 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %25 = arith.index_cast %1 : i32 to index
        %26 = arith.muli %arg28, %c16_i32 : i32
        %27 = arith.muli %3, %c8_i32 : i32
        %28 = arith.addi %26, %27 : i32
        %29 = arith.index_cast %28 : i32 to index
        %subview_77 = memref.subview %reinterpret_cast_13[0, %25, %29] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_77, %alloc_51 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>>
        hivm.hir.vexp ins(%alloc_51 : memref<8xf32, strided<[1]>>) outs(%alloc_51 : memref<8xf32, strided<[1]>>)
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %30 = arith.divsi %arg30, %c2_i32 : i32
            %31 = arith.muli %3, %c8_i32 : i32
            %32 = arith.divsi %arg29, %c2_i32 : i32
            %33 = arith.addi %31, %32 : i32
            %34 = arith.cmpi slt, %30, %33 : i32
            scf.if %34 {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              %37 = memref.load %alloc_48[%35, %36] : memref<16x32xf32, strided<[32, 1]>>
              %38 = arith.muli %3, %c8_i32 : i32
              %39 = arith.divsi %arg29, %c2_i32 : i32
              %40 = arith.addi %38, %39 : i32
              %41 = arith.index_cast %40 : i32 to index
              %42 = arith.divsi %arg30, %c2_i32 : i32
              %43 = arith.index_cast %42 : i32 to index
              %44 = memref.load %alloc_29[%41, %43] : memref<16x16xf32, strided<[16, 1]>>
              %45 = arith.mulf %37, %44 : f32
              %46 = arith.truncf %45 : f32 to bf16
              memref.store %46, %alloc_49[%35, %36] : memref<16x32xbf16, strided<[32, 1]>>
            } else {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_4, %alloc_49[%35, %36] : memref<16x32xbf16, strided<[32, 1]>>
            }
          }
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_50[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %33 = arith.divsi %arg29, %c2_i32 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = memref.load %alloc_51[%34] : memref<8xf32, strided<[1]>>
            %36 = arith.mulf %32, %35 : f32
            memref.store %36, %alloc_50[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %subview_78 = memref.subview %7[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_49, %subview_78 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        %subview_79 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %alloc_50, %subview_79 : memref<16x64xf32, strided<[64, 1]>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      }
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_75 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_75, %alloc_52 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_76 = memref.subview %8[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_76, %alloc_53 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_77 = memref.subview %4[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_77, %alloc_54 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_78 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
        memref.copy %subview_78, %alloc_55 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
        %25 = arith.muli %arg28, %c16_i32 : i32
        %26 = arith.muli %3, %c8_i32 : i32
        %27 = arith.addi %25, %26 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.index_cast %1 : i32 to index
        %subview_79 = memref.subview %reinterpret_cast_14[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_79, %alloc_56 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_52[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %33 = memref.load %alloc_53[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
            %34 = arith.addf %32, %33 : f32
            memref.store %34, %alloc_52[%30, %31] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        scf.for %arg29 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_4, %alloc_57[%30, %31] : memref<8x64xbf16, strided<[64, 1]>>
          }
        }
        %subview_80 = memref.subview %reinterpret_cast_18[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_57, %subview_80 : memref<8x64xbf16, strided<[64, 1]>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      }
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
        %alloc_76 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
        %alloc_77 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
        %alloc_78 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
        %alloc_79 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
        %alloc_80 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
        %alloc_81 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
        %alloc_82 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
        %alloc_83 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          %23 = arith.muli %arg28, %c16_i32 : i32
          %24 = arith.addi %23, %arg29 : i32
          %25 = arith.index_cast %24 : i32 to index
          %26 = arith.index_cast %1 : i32 to index
          %subview_88 = memref.subview %reinterpret_cast_9[0, %25, %26, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %27 = arith.index_cast %arg29 : i32 to index
          %subview_89 = memref.subview %alloc_75[%27, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<64xbf16, strided<[1], offset: ?>>
          memref.copy %subview_88, %subview_89 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>>
        }
        memref.copy %alloc_75, %alloc_77 : memref<16x64xbf16, strided<[64, 1]>> to memref<16x64xbf16, strided<[64, 1]>>
        %reinterpret_cast_84 = memref.reinterpret_cast %alloc_77 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<16x1x64xbf16, strided<[64, 64, 1]>>
        hivm.hir.vbrc ins(%reinterpret_cast_84 : memref<16x1x64xbf16, strided<[64, 64, 1]>>) outs(%alloc_78 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [1]
        memref.copy %alloc_25, %alloc_79 : memref<2x64xbf16, strided<[64, 1]>> to memref<2x64xbf16, strided<[64, 1]>>
        %reinterpret_cast_85 = memref.reinterpret_cast %alloc_79 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>> to memref<1x2x64xbf16, strided<[128, 64, 1]>>
        hivm.hir.vbrc ins(%reinterpret_cast_85 : memref<1x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_80 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [0]
        hivm.hir.vcast ins(%alloc_78 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_81 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
        hivm.hir.vcast ins(%alloc_80 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_82 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
        hivm.hir.vmul ins(%alloc_81, %alloc_82 : memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_83 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
        hivm.hir.vcast ins(%alloc_83 : memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_76 : memref<16x2x64xbf16, strided<[128, 64, 1]>>)
        %reinterpret_cast_86 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
        %subview_87 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
        memref.copy %reinterpret_cast_86, %subview_87 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
        %alloc_76 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
        %alloc_77 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
        %alloc_78 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
        %alloc_79 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
        %alloc_80 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
        %subview_81 = memref.subview %reinterpret_cast[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_81, %alloc_75 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
        memref.copy %alloc, %alloc_76 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
        %reinterpret_cast_82 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
        hivm.hir.vbrc ins(%reinterpret_cast_82 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_77 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
        hivm.hir.vcast ins(%alloc_75 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_78 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
        hivm.hir.vcast ins(%alloc_77 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_79 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
        hivm.hir.vadd ins(%alloc_78, %alloc_79 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_80 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
        hivm.hir.vcast ins(%alloc_80 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_75 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
        %reinterpret_cast_83 = memref.reinterpret_cast %alloc_75 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        %subview_84 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
        memref.copy %reinterpret_cast_83, %subview_84 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
        %alloc_76 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
        %alloc_77 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
        %alloc_78 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
        %alloc_79 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
        %alloc_80 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
        %subview_81 = memref.subview %reinterpret_cast_17[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_81, %alloc_75 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
        memref.copy %alloc_21, %alloc_76 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
        %reinterpret_cast_82 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
        hivm.hir.vbrc ins(%reinterpret_cast_82 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_77 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
        hivm.hir.vcast ins(%alloc_75 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_78 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
        hivm.hir.vcast ins(%alloc_77 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_79 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
        hivm.hir.vadd ins(%alloc_78, %alloc_79 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_80 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
        hivm.hir.vcast ins(%alloc_80 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_75 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
        %reinterpret_cast_83 = memref.reinterpret_cast %alloc_75 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        %subview_84 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
        memref.copy %reinterpret_cast_83, %subview_84 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
        %alloc_76 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
        %alloc_77 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
        %alloc_78 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
        %alloc_79 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
        %alloc_80 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
        %subview_81 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
        memref.copy %subview_81, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        %subview_82 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
        memref.copy %subview_82, %alloc_76 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc_75, %alloc_76, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_78 : memref<32x32xf32, strided<[32, 1]>>)
        %subview_83 = memref.subview %4[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
        memref.copy %alloc_78, %subview_83 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
        %subview_84 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
        memref.copy %subview_84, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        %subview_85 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
        memref.copy %subview_85, %alloc_77 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
        hivm.hir.mmadL1 ins(%alloc_75, %alloc_77, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_79 : memref<32x64xf32, strided<[64, 1]>>)
        %subview_86 = memref.subview %5[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
        memref.copy %alloc_79, %subview_86 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
        %subview_87 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
        memref.copy %subview_87, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        %subview_88 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
        memref.copy %subview_88, %alloc_76 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc_75, %alloc_76, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_80 : memref<32x32xf32, strided<[32, 1]>>)
        %subview_89 = memref.subview %6[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
        memref.copy %alloc_80, %subview_89 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
        %subview_76 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
        memref.copy %subview_76, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.muli %arg29, %c2_i32 : i32
              %24 = arith.addi %23, %arg30 : i32
              %25 = arith.index_cast %24 : i32 to index
              %26 = arith.index_cast %arg31 : i32 to index
              %27 = memref.load %alloc_75[%25, %26] : memref<32x32xbf16, strided<[32, 1]>>
              %28 = arith.index_cast %arg29 : i32 to index
              %29 = arith.index_cast %arg30 : i32 to index
              memref.store %27, %alloc_43[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            }
          }
        }
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.index_cast %arg29 : i32 to index
              %24 = arith.index_cast %arg31 : i32 to index
              %25 = memref.load %alloc_45[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
              %26 = arith.index_cast %arg30 : i32 to index
              %27 = memref.load %alloc_43[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
              %28 = arith.extf %27 : bf16 to f32
              %29 = arith.mulf %25, %28 : f32
              %30 = arith.subf %29, %29 : f32
              %31 = arith.truncf %30 : f32 to bf16
              %32 = arith.muli %arg29, %c2_i32 : i32
              %33 = arith.addi %32, %arg30 : i32
              %34 = arith.index_cast %33 : i32 to index
              memref.store %31, %alloc_75[%34, %24] : memref<32x32xbf16, strided<[32, 1]>>
            }
          }
        }
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.index_cast %arg29 : i32 to index
              %24 = arith.index_cast %arg31 : i32 to index
              %25 = memref.load %alloc_46[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
              %26 = arith.index_cast %arg30 : i32 to index
              %27 = memref.load %alloc_43[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
              %28 = arith.extf %27 : bf16 to f32
              %29 = arith.mulf %25, %28 : f32
              %30 = arith.addf %29, %29 : f32
              %31 = arith.truncf %30 : f32 to bf16
              %32 = arith.muli %arg29, %c2_i32 : i32
              %33 = arith.addi %32, %arg30 : i32
              %34 = arith.index_cast %33 : i32 to index
              %35 = arith.addi %arg31, %c16_i32 : i32
              %36 = arith.index_cast %35 : i32 to index
              memref.store %31, %alloc_75[%34, %36] : memref<32x32xbf16, strided<[32, 1]>>
            }
          }
        }
        %subview_77 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
        memref.copy %alloc_75, %subview_77 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
        hivm.hir.vcast ins(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>) outs(%alloc_75 : memref<32x64xbf16, strided<[64, 1]>>)
        %subview_76 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
        memref.copy %alloc_75, %subview_76 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
        %alloc_76 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
        %alloc_77 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
        %subview_78 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
        memref.copy %subview_78, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.muli %arg29, %c2_i32 : i32
              %24 = arith.addi %23, %arg30 : i32
              %25 = arith.index_cast %24 : i32 to index
              %26 = arith.index_cast %arg31 : i32 to index
              %27 = memref.load %alloc_75[%25, %26] : memref<32x32xbf16, strided<[32, 1]>>
              %28 = arith.index_cast %arg29 : i32 to index
              %29 = arith.index_cast %arg30 : i32 to index
              memref.store %27, %alloc_47[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            }
          }
        }
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.index_cast %arg29 : i32 to index
              %24 = arith.index_cast %arg31 : i32 to index
              %25 = memref.load %alloc_45[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
              %26 = arith.index_cast %arg30 : i32 to index
              %27 = memref.load %alloc_47[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
              %28 = arith.extf %27 : bf16 to f32
              %29 = arith.mulf %25, %28 : f32
              %30 = arith.subf %29, %29 : f32
              %31 = arith.truncf %30 : f32 to bf16
              %32 = arith.muli %arg29, %c2_i32 : i32
              %33 = arith.addi %32, %arg30 : i32
              %34 = arith.index_cast %33 : i32 to index
              memref.store %31, %alloc_75[%34, %24] : memref<32x32xbf16, strided<[32, 1]>>
            }
          }
        }
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.index_cast %arg29 : i32 to index
              %24 = arith.index_cast %arg31 : i32 to index
              %25 = memref.load %alloc_46[%23, %24] : memref<16x8xf32, strided<[8, 1]>>
              %26 = arith.index_cast %arg30 : i32 to index
              %27 = memref.load %alloc_47[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
              %28 = arith.extf %27 : bf16 to f32
              %29 = arith.mulf %25, %28 : f32
              %30 = arith.addf %29, %29 : f32
              %31 = arith.truncf %30 : f32 to bf16
              %32 = arith.muli %arg29, %c2_i32 : i32
              %33 = arith.addi %32, %arg30 : i32
              %34 = arith.index_cast %33 : i32 to index
              %35 = arith.addi %arg31, %c16_i32 : i32
              %36 = arith.index_cast %35 : i32 to index
              memref.store %31, %alloc_75[%34, %36] : memref<32x32xbf16, strided<[32, 1]>>
            }
          }
        }
        memref.copy %alloc_75, %alloc_76 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        hivm.hir.vcast ins(%alloc_76 : memref<32x32xbf16, strided<[32, 1]>>) outs(%alloc_77 : memref<32x32xf32, strided<[32, 1]>>)
        scf.for %arg29 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg30 : i32 to index
            %25 = memref.load %alloc_77[%23, %24] : memref<32x32xf32, strided<[32, 1]>>
            %26 = arith.divsi %arg29, %c2_i32 : i32
            %27 = arith.index_cast %26 : i32 to index
            %28 = memref.load %alloc_42[%27] : memref<16xbf16, strided<[1]>>
            %29 = arith.extf %28 : bf16 to f32
            %30 = arith.mulf %25, %29 : f32
            memref.store %30, %alloc_77[%23, %24] : memref<32x32xf32, strided<[32, 1]>>
          }
        }
        hivm.hir.vcast ins(%alloc_77 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc_76 : memref<32x32xbf16, strided<[32, 1]>>)
        %subview_79 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
        memref.copy %alloc_76, %subview_79 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
        %alloc_76 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
        %alloc_77 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
        %subview_78 = memref.subview %7[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
        memref.copy %subview_78, %alloc_76 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
        %subview_79 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
        memref.copy %subview_79, %alloc_75 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
        hivm.hir.mmadL1 ins(%alloc_76, %alloc_75, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_77 : memref<32x64xf32, strided<[64, 1]>>)
        %subview_80 = memref.subview %8[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
        memref.copy %alloc_77, %subview_80 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    } {tilelangir.num_stages = 2 : i32}
    return
  }
}


// -----// IR Dump After TileLangIRInferMemScope (tilelangir-infer-mem-scope) ('func.func' operation: @mamba_mimo_fwd_kernel) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @mamba_mimo_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xbf16, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: memref<?xf32, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: memref<?xbf16, #hivm.address_space<gm>>, %arg12: memref<?xf32, #hivm.address_space<gm>>, %arg13: memref<?xf32, #hivm.address_space<gm>>, %arg14: memref<?xf32, #hivm.address_space<gm>>, %arg15: memref<?xf32, #hivm.address_space<gm>>, %arg16: memref<?xf32, #hivm.address_space<gm>>, %arg17: memref<?xf32, #hivm.address_space<gm>>, %arg18: memref<?xbf16, #hivm.address_space<gm>>, %arg19: memref<?xf32, #hivm.address_space<gm>>, %arg20: memref<?xf32, #hivm.address_space<gm>>, %arg21: memref<?xbf16, #hivm.address_space<gm>>, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %cst = arith.constant -1.98412701E-4 : f32
    %cst_0 = arith.constant 0.00833333377 : f32
    %cst_1 = arith.constant -0.166666672 : f32
    %cst_2 = arith.constant -0.00138888892 : f32
    %cst_3 = arith.constant 0.0416666679 : f32
    %c2048 = arith.constant 2048 : index
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %c132 = arith.constant 132 : index
    %c33 = arith.constant 33 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c8192 = arith.constant 8192 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %cst_4 = arith.constant 0.000000e+00 : bf16
    %cst_5 = arith.constant -5.000000e-01 : f32
    %true = arith.constant true
    %c31_i32 = arith.constant 31 : i32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant -1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_10 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [4, 2, 64], strides: [%c128, %c64, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg14 to offset: [0], sizes: [1, 32, 4, 8], strides: [%c1024, %c32, %c8, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_13 = memref.reinterpret_cast %arg15 to offset: [0], sizes: [1, 4, 32], strides: [%c128, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_14 = memref.reinterpret_cast %arg11 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_15 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_16 = memref.reinterpret_cast %arg18 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_17 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_18 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_19 = memref.reinterpret_cast %arg17 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_20 = memref.reinterpret_cast %arg19 to offset: [0], sizes: [1, 4, 2, 16, 16], strides: [%c2048, %c512, %c256, %c16, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x32x32xf32, #hivm.address_space<gm>>
    annotation.mark %4 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() : memref<2x32x64xf32, #hivm.address_space<gm>>
    annotation.mark %5 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() : memref<2x32x32xf32, #hivm.address_space<gm>>
    annotation.mark %6 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32, #hivm.address_space<gm>>
    %7 = memref_ext.alloc_workspace() : memref<2x32x32xbf16, #hivm.address_space<gm>>
    annotation.mark %7 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16, #hivm.address_space<gm>>
    %8 = memref_ext.alloc_workspace() : memref<2x32x64xf32, #hivm.address_space<gm>>
    annotation.mark %8 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32, #hivm.address_space<gm>>
    %9 = memref_ext.alloc_workspace() : memref<2x32x32xbf16, #hivm.address_space<gm>>
    annotation.mark %9 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16, #hivm.address_space<gm>>
    %10 = memref_ext.alloc_workspace() : memref<2x32x32xbf16, #hivm.address_space<gm>>
    annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16, #hivm.address_space<gm>>
    %11 = memref_ext.alloc_workspace() : memref<2x32x32xbf16, #hivm.address_space<gm>>
    annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16, #hivm.address_space<gm>>
    %12 = memref_ext.alloc_workspace() : memref<2x32x32xbf16, #hivm.address_space<gm>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16, #hivm.address_space<gm>>
    %13 = memref_ext.alloc_workspace() : memref<2x32x64xbf16, #hivm.address_space<gm>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16, #hivm.address_space<gm>>
    %14 = memref_ext.alloc_workspace() : memref<2x32x64xbf16, #hivm.address_space<gm>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_21 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_22 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_23 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_24 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    %alloc_25 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
    %alloc_26 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_7 : f32) outs(%alloc_24 : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>)
    %15 = arith.index_cast %1 : i32 to index
    %subview = memref.subview %reinterpret_cast_10[%15, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_26 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%alloc_26 : memref<2x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>)
    %subview_27 = memref.subview %reinterpret_cast_11[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_27, %alloc_22 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%alloc_22 : memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
    %subview_28 = memref.subview %reinterpret_cast_15[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_28, %alloc_23 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
    scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %alloc_29 = memref.alloc() : memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %alloc_30 = memref.alloc() : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      %alloc_31 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %alloc_32 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %alloc_33 = memref.alloc() : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      %alloc_34 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %alloc_35 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %alloc_36 = memref.alloc() : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      %alloc_37 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %alloc_38 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %alloc_39 = memref.alloc() : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      %alloc_40 = memref.alloc() : memref<16xf32, strided<[1]>, #hivm.address_space<cbuf>>
      %alloc_41 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %alloc_42 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %alloc_43 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
      %alloc_44 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %alloc_45 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %alloc_46 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %alloc_47 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
      %alloc_48 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
      %alloc_49 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
      %alloc_50 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %alloc_51 = memref.alloc() : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
      %alloc_52 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %alloc_53 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %alloc_54 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
      %alloc_55 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %alloc_56 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %alloc_57 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %16 = arith.index_cast %1 : i32 to index
      %17 = arith.index_cast %arg28 : i32 to index
      %subview_58 = memref.subview %reinterpret_cast_20[0, %16, %17, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_58, %alloc_29 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %18 = arith.muli %arg28, %c16_i32 : i32
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_59 = memref.subview %reinterpret_cast_16[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_59, %alloc_31 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_31 : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_30, %cst_8 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>, f32) outs(%alloc_30 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>)
      %alloc_60 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vsub ins(%cst_7, %alloc_30 : f32, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_60 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_60 : memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_60 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_60, %cst_6 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%alloc_60 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%cst_6, %alloc_60 : f32, memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_30 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>)
      %subview_61 = memref.subview %reinterpret_cast_19[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_61, %alloc_33 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_33 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_4 : bf16) outs(%alloc_34 : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.muli %arg28, %c16_i32 : i32
        %24 = arith.addi %23, %arg29 : i32
        %25 = arith.cmpi slt, %24, %c31_i32 : i32
        scf.if %25 {
          %26 = arith.index_cast %arg29 : i32 to index
          %27 = memref.load %alloc_32[%26] : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
          %28 = arith.extf %27 : bf16 to f32
          %29 = memref.load %alloc_30[%26] : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
          %30 = arith.mulf %28, %29 : f32
          %31 = arith.truncf %30 : f32 to bf16
          memref.store %31, %alloc_34[%26] : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
        }
      }
      memref.copy %alloc_34, %alloc_35 : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %21 = arith.index_cast %18 : i32 to index
      %subview_62 = memref.subview %reinterpret_cast_16[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_62, %alloc_37 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_36 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>)
      %alloc_63 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vsub ins(%cst_7, %alloc_36 : f32, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_63 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_63 : memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_63 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_63, %cst_6 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%alloc_63 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%cst_6, %alloc_63 : f32, memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_36 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>)
      %subview_64 = memref.subview %reinterpret_cast_19[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_64, %alloc_39 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_39 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_38 : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_38[%23] : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
        %25 = arith.extf %24 : bf16 to f32
        %26 = memref.load %alloc_36[%23] : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
        %27 = arith.mulf %25, %26 : f32
        memref.store %27, %alloc_40[%23] : memref<16xf32, strided<[1]>, #hivm.address_space<cbuf>>
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_40[%23] : memref<16xf32, strided<[1]>, #hivm.address_space<cbuf>>
        %25 = memref.load %alloc_35[%23] : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
        %26 = arith.extf %25 : bf16 to f32
        %27 = arith.addf %24, %26 : f32
        %28 = arith.truncf %27 : f32 to bf16
        memref.store %28, %alloc_41[%23] : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      }
      memref.copy %alloc_41, %alloc_42 : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>> to memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %subview_65 = memref.subview %reinterpret_cast_12[0, %21, %16, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_65, %alloc_44 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %alloc_66 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_67 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_68 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_69 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%alloc_44, %alloc_44 : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%alloc_66 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_66, %alloc_66 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_67 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_66, %alloc_67 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_68 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_66, %cst_5 : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_66 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_67, %cst_3 : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_67 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_68, %cst_2 : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_68 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_66, %cst_6 : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_69 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_67, %alloc_69 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_69 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_68, %alloc_69 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_45 : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      %alloc_70 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_71 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_72 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_73 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_74 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%alloc_44, %alloc_44 : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%alloc_70 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_70, %alloc_44 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%alloc_71 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_71, %alloc_70 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_72 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_72, %alloc_70 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_73 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_71, %cst_1 : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_71 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_72, %cst_0 : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_72 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_73, %cst : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_73 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_44, %alloc_71 : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_74 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_72, %alloc_74 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_74 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_73, %alloc_74 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_46 : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_29 : memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>)
      %22 = arith.cmpi slt, %3, %c2_i32 : i32
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_75 = memref.subview %6[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_75, %alloc_48 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_76 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_76, %alloc_50 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %25 = arith.index_cast %1 : i32 to index
        %26 = arith.muli %arg28, %c16_i32 : i32
        %27 = arith.muli %3, %c8_i32 : i32
        %28 = arith.addi %26, %27 : i32
        %29 = arith.index_cast %28 : i32 to index
        %subview_77 = memref.subview %reinterpret_cast_13[0, %25, %29] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_77, %alloc_51 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.vexp ins(%alloc_51 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_51 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>)
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %30 = arith.divsi %arg30, %c2_i32 : i32
            %31 = arith.muli %3, %c8_i32 : i32
            %32 = arith.divsi %arg29, %c2_i32 : i32
            %33 = arith.addi %31, %32 : i32
            %34 = arith.cmpi slt, %30, %33 : i32
            scf.if %34 {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              %37 = memref.load %alloc_48[%35, %36] : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
              %38 = arith.muli %3, %c8_i32 : i32
              %39 = arith.divsi %arg29, %c2_i32 : i32
              %40 = arith.addi %38, %39 : i32
              %41 = arith.index_cast %40 : i32 to index
              %42 = arith.divsi %arg30, %c2_i32 : i32
              %43 = arith.index_cast %42 : i32 to index
              %44 = memref.load %alloc_29[%41, %43] : memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
              %45 = arith.mulf %37, %44 : f32
              %46 = arith.truncf %45 : f32 to bf16
              memref.store %46, %alloc_49[%35, %36] : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
            } else {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_4, %alloc_49[%35, %36] : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
            }
          }
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_50[%30, %31] : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
            %33 = arith.divsi %arg29, %c2_i32 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = memref.load %alloc_51[%34] : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
            %36 = arith.mulf %32, %35 : f32
            memref.store %36, %alloc_50[%30, %31] : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
          }
        }
        %subview_78 = memref.subview %7[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_49, %subview_78 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>> to memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_79 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_50, %subview_79 : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      }
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_75 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_75, %alloc_52 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %subview_76 = memref.subview %8[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_76, %alloc_53 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %subview_77 = memref.subview %4[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_77, %alloc_54 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_78 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_78, %alloc_55 : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %25 = arith.muli %arg28, %c16_i32 : i32
        %26 = arith.muli %3, %c8_i32 : i32
        %27 = arith.addi %25, %26 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.index_cast %1 : i32 to index
        %subview_79 = memref.subview %reinterpret_cast_14[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_79, %alloc_56 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_52[%30, %31] : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
            %33 = memref.load %alloc_53[%30, %31] : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
            %34 = arith.addf %32, %33 : f32
            memref.store %34, %alloc_52[%30, %31] : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
          }
        }
        scf.for %arg29 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_4, %alloc_57[%30, %31] : memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
          }
        }
        %subview_80 = memref.subview %reinterpret_cast_18[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_57, %subview_80 : memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      }
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %alloc_76 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %alloc_77 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %alloc_78 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %alloc_79 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %alloc_80 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %alloc_81 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %alloc_82 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %alloc_83 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          %23 = arith.muli %arg28, %c16_i32 : i32
          %24 = arith.addi %23, %arg29 : i32
          %25 = arith.index_cast %24 : i32 to index
          %26 = arith.index_cast %1 : i32 to index
          %subview_88 = memref.subview %reinterpret_cast_9[0, %25, %26, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %27 = arith.index_cast %arg29 : i32 to index
          %subview_89 = memref.subview %alloc_75[%27, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
          memref.copy %subview_88, %subview_89 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
        }
        memref.copy %alloc_75, %alloc_77 : memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %reinterpret_cast_84 = memref.reinterpret_cast %alloc_77 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<16x1x64xbf16, strided<[64, 64, 1]>, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%reinterpret_cast_84 : memref<16x1x64xbf16, strided<[64, 64, 1]>, #hivm.address_space<ub>>) outs(%alloc_78 : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
        memref.copy %alloc_25, %alloc_79 : memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %reinterpret_cast_85 = memref.reinterpret_cast %alloc_79 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<1x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%reinterpret_cast_85 : memref<1x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) outs(%alloc_80 : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
        hivm.hir.vcast ins(%alloc_78 : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) outs(%alloc_81 : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_80 : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) outs(%alloc_82 : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vmul ins(%alloc_81, %alloc_82 : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>, memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>) outs(%alloc_83 : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_83 : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>) outs(%alloc_76 : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>)
        %reinterpret_cast_86 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %subview_87 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %reinterpret_cast_86, %subview_87 : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_76 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_77 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_78 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_79 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_80 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %subview_81 = memref.subview %reinterpret_cast[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_81, %alloc_75 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        memref.copy %alloc, %alloc_76 : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %reinterpret_cast_82 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<1x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%reinterpret_cast_82 : memref<1x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_77 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
        hivm.hir.vcast ins(%alloc_75 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_78 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_77 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_79 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vadd ins(%alloc_78, %alloc_79 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_80 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_80 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_75 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        %reinterpret_cast_83 = memref.reinterpret_cast %alloc_75 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %subview_84 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %reinterpret_cast_83, %subview_84 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_76 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_77 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_78 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_79 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_80 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %subview_81 = memref.subview %reinterpret_cast_17[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_81, %alloc_75 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        memref.copy %alloc_21, %alloc_76 : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %reinterpret_cast_82 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<1x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%reinterpret_cast_82 : memref<1x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_77 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
        hivm.hir.vcast ins(%alloc_75 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_78 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_77 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_79 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vadd ins(%alloc_78, %alloc_79 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_80 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_80 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_75 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        %reinterpret_cast_83 = memref.reinterpret_cast %alloc_75 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %subview_84 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %reinterpret_cast_83, %subview_84 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_76 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_77 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %alloc_78 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_79 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>
        %alloc_80 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %subview_81 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_81, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_82 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_82, %alloc_76 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc_75, %alloc_76, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_78 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
        %subview_83 = memref.subview %4[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_78, %subview_83 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>> to memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<gm>>
        %subview_84 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_84, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_85 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_85, %alloc_77 : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        hivm.hir.mmadL1 ins(%alloc_75, %alloc_77, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_79 : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>)
        %subview_86 = memref.subview %5[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_79, %subview_86 : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>> to memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
        %subview_87 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_87, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_88 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_88, %alloc_76 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc_75, %alloc_76, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_80 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
        %subview_89 = memref.subview %6[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_80, %subview_89 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>> to memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_76 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_76, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.muli %arg29, %c2_i32 : i32
              %24 = arith.addi %23, %arg30 : i32
              %25 = arith.index_cast %24 : i32 to index
              %26 = arith.index_cast %arg31 : i32 to index
              %27 = memref.load %alloc_75[%25, %26] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
              %28 = arith.index_cast %arg29 : i32 to index
              %29 = arith.index_cast %arg30 : i32 to index
              memref.store %27, %alloc_43[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
            }
          }
        }
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.index_cast %arg29 : i32 to index
              %24 = arith.index_cast %arg31 : i32 to index
              %25 = memref.load %alloc_45[%23, %24] : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
              %26 = arith.index_cast %arg30 : i32 to index
              %27 = memref.load %alloc_43[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
              %28 = arith.extf %27 : bf16 to f32
              %29 = arith.mulf %25, %28 : f32
              %30 = arith.subf %29, %29 : f32
              %31 = arith.truncf %30 : f32 to bf16
              %32 = arith.muli %arg29, %c2_i32 : i32
              %33 = arith.addi %32, %arg30 : i32
              %34 = arith.index_cast %33 : i32 to index
              memref.store %31, %alloc_75[%34, %24] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
            }
          }
        }
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.index_cast %arg29 : i32 to index
              %24 = arith.index_cast %arg31 : i32 to index
              %25 = memref.load %alloc_46[%23, %24] : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
              %26 = arith.index_cast %arg30 : i32 to index
              %27 = memref.load %alloc_43[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
              %28 = arith.extf %27 : bf16 to f32
              %29 = arith.mulf %25, %28 : f32
              %30 = arith.addf %29, %29 : f32
              %31 = arith.truncf %30 : f32 to bf16
              %32 = arith.muli %arg29, %c2_i32 : i32
              %33 = arith.addi %32, %arg30 : i32
              %34 = arith.index_cast %33 : i32 to index
              %35 = arith.addi %arg31, %c16_i32 : i32
              %36 = arith.index_cast %35 : i32 to index
              memref.store %31, %alloc_75[%34, %36] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
            }
          }
        }
        %subview_77 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_75, %subview_77 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        hivm.hir.vcast ins(%alloc_24 : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%alloc_75 : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>)
        %subview_76 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_75, %subview_76 : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_76 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_77 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %subview_78 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_78, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.muli %arg29, %c2_i32 : i32
              %24 = arith.addi %23, %arg30 : i32
              %25 = arith.index_cast %24 : i32 to index
              %26 = arith.index_cast %arg31 : i32 to index
              %27 = memref.load %alloc_75[%25, %26] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
              %28 = arith.index_cast %arg29 : i32 to index
              %29 = arith.index_cast %arg30 : i32 to index
              memref.store %27, %alloc_47[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
            }
          }
        }
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.index_cast %arg29 : i32 to index
              %24 = arith.index_cast %arg31 : i32 to index
              %25 = memref.load %alloc_45[%23, %24] : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
              %26 = arith.index_cast %arg30 : i32 to index
              %27 = memref.load %alloc_47[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
              %28 = arith.extf %27 : bf16 to f32
              %29 = arith.mulf %25, %28 : f32
              %30 = arith.subf %29, %29 : f32
              %31 = arith.truncf %30 : f32 to bf16
              %32 = arith.muli %arg29, %c2_i32 : i32
              %33 = arith.addi %32, %arg30 : i32
              %34 = arith.index_cast %33 : i32 to index
              memref.store %31, %alloc_75[%34, %24] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
            }
          }
        }
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.index_cast %arg29 : i32 to index
              %24 = arith.index_cast %arg31 : i32 to index
              %25 = memref.load %alloc_46[%23, %24] : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
              %26 = arith.index_cast %arg30 : i32 to index
              %27 = memref.load %alloc_47[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
              %28 = arith.extf %27 : bf16 to f32
              %29 = arith.mulf %25, %28 : f32
              %30 = arith.addf %29, %29 : f32
              %31 = arith.truncf %30 : f32 to bf16
              %32 = arith.muli %arg29, %c2_i32 : i32
              %33 = arith.addi %32, %arg30 : i32
              %34 = arith.index_cast %33 : i32 to index
              %35 = arith.addi %arg31, %c16_i32 : i32
              %36 = arith.index_cast %35 : i32 to index
              memref.store %31, %alloc_75[%34, %36] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
            }
          }
        }
        memref.copy %alloc_75, %alloc_76 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        hivm.hir.vcast ins(%alloc_76 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_77 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
        scf.for %arg29 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg30 : i32 to index
            %25 = memref.load %alloc_77[%23, %24] : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
            %26 = arith.divsi %arg29, %c2_i32 : i32
            %27 = arith.index_cast %26 : i32 to index
            %28 = memref.load %alloc_42[%27] : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
            %29 = arith.extf %28 : bf16 to f32
            %30 = arith.mulf %25, %29 : f32
            memref.store %30, %alloc_77[%23, %24] : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          }
        }
        hivm.hir.vcast ins(%alloc_77 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_76 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
        %subview_79 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_76, %subview_79 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %alloc_76 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_77 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>
        %subview_78 = memref.subview %7[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_78, %alloc_76 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_79 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_79, %alloc_75 : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        hivm.hir.mmadL1 ins(%alloc_76, %alloc_75, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_77 : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>)
        %subview_80 = memref.subview %8[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_77, %subview_80 : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>> to memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    } {tilelangir.num_stages = 2 : i32}
    return
  }
}


// -----// IR Dump After TileLangIRMergeCopyChains (tilelangir-merge-copy-chains) ('func.func' operation: @mamba_mimo_fwd_kernel) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @mamba_mimo_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xbf16, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: memref<?xf32, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: memref<?xbf16, #hivm.address_space<gm>>, %arg12: memref<?xf32, #hivm.address_space<gm>>, %arg13: memref<?xf32, #hivm.address_space<gm>>, %arg14: memref<?xf32, #hivm.address_space<gm>>, %arg15: memref<?xf32, #hivm.address_space<gm>>, %arg16: memref<?xf32, #hivm.address_space<gm>>, %arg17: memref<?xf32, #hivm.address_space<gm>>, %arg18: memref<?xbf16, #hivm.address_space<gm>>, %arg19: memref<?xf32, #hivm.address_space<gm>>, %arg20: memref<?xf32, #hivm.address_space<gm>>, %arg21: memref<?xbf16, #hivm.address_space<gm>>, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %cst = arith.constant -1.98412701E-4 : f32
    %cst_0 = arith.constant 0.00833333377 : f32
    %cst_1 = arith.constant -0.166666672 : f32
    %cst_2 = arith.constant -0.00138888892 : f32
    %cst_3 = arith.constant 0.0416666679 : f32
    %c2048 = arith.constant 2048 : index
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %c132 = arith.constant 132 : index
    %c33 = arith.constant 33 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c8192 = arith.constant 8192 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %cst_4 = arith.constant 0.000000e+00 : bf16
    %cst_5 = arith.constant -5.000000e-01 : f32
    %true = arith.constant true
    %c31_i32 = arith.constant 31 : i32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant -1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_10 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [4, 2, 64], strides: [%c128, %c64, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg14 to offset: [0], sizes: [1, 32, 4, 8], strides: [%c1024, %c32, %c8, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_13 = memref.reinterpret_cast %arg15 to offset: [0], sizes: [1, 4, 32], strides: [%c128, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_14 = memref.reinterpret_cast %arg11 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_15 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [4, 2, 32], strides: [%c64, %c32, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_16 = memref.reinterpret_cast %arg18 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_17 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 32, 2, 4, 32], strides: [%c8192, %c256, %c128, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_18 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1, 32, 4, 64], strides: [%c8192, %c256, %c64, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_19 = memref.reinterpret_cast %arg17 to offset: [0], sizes: [1, 4, 33], strides: [%c132, %c33, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_20 = memref.reinterpret_cast %arg19 to offset: [0], sizes: [1, 4, 2, 16, 16], strides: [%c2048, %c512, %c256, %c16, %c1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x32x32xf32, #hivm.address_space<gm>>
    annotation.mark %4 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() : memref<2x32x64xf32, #hivm.address_space<gm>>
    annotation.mark %5 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() : memref<2x32x32xf32, #hivm.address_space<gm>>
    annotation.mark %6 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xf32, #hivm.address_space<gm>>
    %7 = memref_ext.alloc_workspace() : memref<2x32x32xbf16, #hivm.address_space<gm>>
    annotation.mark %7 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16, #hivm.address_space<gm>>
    %8 = memref_ext.alloc_workspace() : memref<2x32x64xf32, #hivm.address_space<gm>>
    annotation.mark %8 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xf32, #hivm.address_space<gm>>
    %9 = memref_ext.alloc_workspace() : memref<2x32x32xbf16, #hivm.address_space<gm>>
    annotation.mark %9 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16, #hivm.address_space<gm>>
    %10 = memref_ext.alloc_workspace() : memref<2x32x32xbf16, #hivm.address_space<gm>>
    annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16, #hivm.address_space<gm>>
    %11 = memref_ext.alloc_workspace() : memref<2x32x32xbf16, #hivm.address_space<gm>>
    annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16, #hivm.address_space<gm>>
    %12 = memref_ext.alloc_workspace() : memref<2x32x32xbf16, #hivm.address_space<gm>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<2x32x32xbf16, #hivm.address_space<gm>>
    %13 = memref_ext.alloc_workspace() : memref<2x32x64xbf16, #hivm.address_space<gm>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16, #hivm.address_space<gm>>
    %14 = memref_ext.alloc_workspace() : memref<2x32x64xbf16, #hivm.address_space<gm>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<2x32x64xbf16, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_21 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_22 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_23 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_24 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    %alloc_25 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
    %alloc_26 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_7 : f32) outs(%alloc_24 : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>)
    %15 = arith.index_cast %1 : i32 to index
    %subview = memref.subview %reinterpret_cast_10[%15, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_26 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%alloc_26 : memref<2x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>)
    %subview_27 = memref.subview %reinterpret_cast_11[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_27, %alloc_22 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%alloc_22 : memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
    %subview_28 = memref.subview %reinterpret_cast_15[%15, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_28, %alloc_23 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
    scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %alloc_29 = memref.alloc() : memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %alloc_30 = memref.alloc() : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      %alloc_31 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %alloc_32 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %alloc_33 = memref.alloc() : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      %alloc_34 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %alloc_35 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %alloc_36 = memref.alloc() : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      %alloc_37 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %alloc_38 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %alloc_39 = memref.alloc() : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      %alloc_40 = memref.alloc() : memref<16xf32, strided<[1]>, #hivm.address_space<cbuf>>
      %alloc_41 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %alloc_42 = memref.alloc() : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %alloc_43 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
      %alloc_44 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %alloc_45 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %alloc_46 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %alloc_47 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
      %alloc_48 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
      %alloc_49 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
      %alloc_50 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %alloc_51 = memref.alloc() : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
      %alloc_52 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %alloc_53 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %alloc_54 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
      %alloc_55 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %alloc_56 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %alloc_57 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %16 = arith.index_cast %1 : i32 to index
      %17 = arith.index_cast %arg28 : i32 to index
      %subview_58 = memref.subview %reinterpret_cast_20[0, %16, %17, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_58, %alloc_29 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %18 = arith.muli %arg28, %c16_i32 : i32
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_59 = memref.subview %reinterpret_cast_16[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_59, %alloc_31 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_31 : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_30, %cst_8 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>, f32) outs(%alloc_30 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>)
      %alloc_60 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vsub ins(%cst_7, %alloc_30 : f32, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_60 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_60 : memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_60 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_60, %cst_6 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%alloc_60 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%cst_6, %alloc_60 : f32, memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_30 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>)
      %subview_61 = memref.subview %reinterpret_cast_19[0, %16, %20] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_61, %alloc_33 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_33 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_4 : bf16) outs(%alloc_34 : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.muli %arg28, %c16_i32 : i32
        %24 = arith.addi %23, %arg29 : i32
        %25 = arith.cmpi slt, %24, %c31_i32 : i32
        scf.if %25 {
          %26 = arith.index_cast %arg29 : i32 to index
          %27 = memref.load %alloc_32[%26] : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
          %28 = arith.extf %27 : bf16 to f32
          %29 = memref.load %alloc_30[%26] : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
          %30 = arith.mulf %28, %29 : f32
          %31 = arith.truncf %30 : f32 to bf16
          memref.store %31, %alloc_34[%26] : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
        }
      }
      memref.copy %alloc_34, %alloc_35 : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %21 = arith.index_cast %18 : i32 to index
      %subview_62 = memref.subview %reinterpret_cast_16[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_62, %alloc_37 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_36 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>)
      %alloc_63 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vsub ins(%cst_7, %alloc_36 : f32, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_63 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_63 : memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_63 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_63, %cst_6 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%alloc_63 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%cst_6, %alloc_63 : f32, memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_36 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>)
      %subview_64 = memref.subview %reinterpret_cast_19[0, %16, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_64, %alloc_39 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_39 : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_38 : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_38[%23] : memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
        %25 = arith.extf %24 : bf16 to f32
        %26 = memref.load %alloc_36[%23] : memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
        %27 = arith.mulf %25, %26 : f32
        memref.store %27, %alloc_40[%23] : memref<16xf32, strided<[1]>, #hivm.address_space<cbuf>>
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %23 = arith.index_cast %arg29 : i32 to index
        %24 = memref.load %alloc_40[%23] : memref<16xf32, strided<[1]>, #hivm.address_space<cbuf>>
        %25 = memref.load %alloc_35[%23] : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
        %26 = arith.extf %25 : bf16 to f32
        %27 = arith.addf %24, %26 : f32
        %28 = arith.truncf %27 : f32 to bf16
        memref.store %28, %alloc_41[%23] : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      }
      memref.copy %alloc_41, %alloc_42 : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>> to memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %subview_65 = memref.subview %reinterpret_cast_12[0, %21, %16, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_65, %alloc_44 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %alloc_66 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_67 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_68 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_69 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%alloc_44, %alloc_44 : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%alloc_66 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_66, %alloc_66 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_67 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_66, %alloc_67 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_68 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_66, %cst_5 : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_66 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_67, %cst_3 : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_67 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_68, %cst_2 : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_68 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_66, %cst_6 : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_69 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_67, %alloc_69 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_69 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_68, %alloc_69 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_45 : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      %alloc_70 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_71 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_72 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_73 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      %alloc_74 = memref.alloc() : memref<16x8xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%alloc_44, %alloc_44 : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%alloc_70 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_70, %alloc_44 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%alloc_71 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_71, %alloc_70 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_72 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_72, %alloc_70 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_73 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_71, %cst_1 : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_71 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_72, %cst_0 : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_72 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_73, %cst : memref<16x8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_73 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_44, %alloc_71 : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_74 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_72, %alloc_74 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_74 : memref<16x8xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_73, %alloc_74 : memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) outs(%alloc_46 : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_29 : memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>)
      %22 = arith.cmpi slt, %3, %c2_i32 : i32
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_75 = memref.subview %6[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_75, %alloc_48 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_76 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_76, %alloc_50 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %25 = arith.index_cast %1 : i32 to index
        %26 = arith.muli %arg28, %c16_i32 : i32
        %27 = arith.muli %3, %c8_i32 : i32
        %28 = arith.addi %26, %27 : i32
        %29 = arith.index_cast %28 : i32 to index
        %subview_77 = memref.subview %reinterpret_cast_13[0, %25, %29] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_77, %alloc_51 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.vexp ins(%alloc_51 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_51 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>)
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %30 = arith.divsi %arg30, %c2_i32 : i32
            %31 = arith.muli %3, %c8_i32 : i32
            %32 = arith.divsi %arg29, %c2_i32 : i32
            %33 = arith.addi %31, %32 : i32
            %34 = arith.cmpi slt, %30, %33 : i32
            scf.if %34 {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              %37 = memref.load %alloc_48[%35, %36] : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
              %38 = arith.muli %3, %c8_i32 : i32
              %39 = arith.divsi %arg29, %c2_i32 : i32
              %40 = arith.addi %38, %39 : i32
              %41 = arith.index_cast %40 : i32 to index
              %42 = arith.divsi %arg30, %c2_i32 : i32
              %43 = arith.index_cast %42 : i32 to index
              %44 = memref.load %alloc_29[%41, %43] : memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
              %45 = arith.mulf %37, %44 : f32
              %46 = arith.truncf %45 : f32 to bf16
              memref.store %46, %alloc_49[%35, %36] : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
            } else {
              %35 = arith.index_cast %arg29 : i32 to index
              %36 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_4, %alloc_49[%35, %36] : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
            }
          }
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_50[%30, %31] : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
            %33 = arith.divsi %arg29, %c2_i32 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = memref.load %alloc_51[%34] : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
            %36 = arith.mulf %32, %35 : f32
            memref.store %36, %alloc_50[%30, %31] : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
          }
        }
        %subview_78 = memref.subview %7[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_49, %subview_78 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>> to memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_79 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_50, %subview_79 : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      }
      scf.if %22 {
        %23 = arith.muli %3, %c16_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_75 = memref.subview %5[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_75, %alloc_52 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %subview_76 = memref.subview %8[0, %24, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_76, %alloc_53 : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %subview_77 = memref.subview %4[0, %24, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_77, %alloc_54 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_78 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_78, %alloc_55 : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %25 = arith.muli %arg28, %c16_i32 : i32
        %26 = arith.muli %3, %c8_i32 : i32
        %27 = arith.addi %25, %26 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.index_cast %1 : i32 to index
        %subview_79 = memref.subview %reinterpret_cast_14[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_79, %alloc_56 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            %32 = memref.load %alloc_52[%30, %31] : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
            %33 = memref.load %alloc_53[%30, %31] : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
            %34 = arith.addf %32, %33 : f32
            memref.store %34, %alloc_52[%30, %31] : memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
          }
        }
        scf.for %arg29 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %30 = arith.index_cast %arg29 : i32 to index
            %31 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_4, %alloc_57[%30, %31] : memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
          }
        }
        %subview_80 = memref.subview %reinterpret_cast_18[0, %28, %29, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_57, %subview_80 : memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      }
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %alloc_76 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %alloc_77 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %alloc_78 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %alloc_79 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %alloc_80 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %alloc_81 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %alloc_82 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %alloc_83 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          %23 = arith.muli %arg28, %c16_i32 : i32
          %24 = arith.addi %23, %arg29 : i32
          %25 = arith.index_cast %24 : i32 to index
          %26 = arith.index_cast %1 : i32 to index
          %subview_88 = memref.subview %reinterpret_cast_9[0, %25, %26, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %27 = arith.index_cast %arg29 : i32 to index
          %subview_89 = memref.subview %alloc_75[%27, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
          memref.copy %subview_88, %subview_89 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
        }
        memref.copy %alloc_75, %alloc_77 : memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %reinterpret_cast_84 = memref.reinterpret_cast %alloc_77 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<16x1x64xbf16, strided<[64, 64, 1]>, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%reinterpret_cast_84 : memref<16x1x64xbf16, strided<[64, 64, 1]>, #hivm.address_space<ub>>) outs(%alloc_78 : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
        memref.copy %alloc_25, %alloc_79 : memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %reinterpret_cast_85 = memref.reinterpret_cast %alloc_79 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<1x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%reinterpret_cast_85 : memref<1x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) outs(%alloc_80 : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
        hivm.hir.vcast ins(%alloc_78 : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) outs(%alloc_81 : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_80 : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) outs(%alloc_82 : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vmul ins(%alloc_81, %alloc_82 : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>, memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>) outs(%alloc_83 : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_83 : memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>) outs(%alloc_76 : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>)
        %reinterpret_cast_86 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %subview_87 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %reinterpret_cast_86, %subview_87 : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_76 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_77 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_78 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_79 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_80 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %subview_81 = memref.subview %reinterpret_cast[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_81, %alloc_75 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        memref.copy %alloc, %alloc_76 : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %reinterpret_cast_82 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<1x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%reinterpret_cast_82 : memref<1x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_77 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
        hivm.hir.vcast ins(%alloc_75 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_78 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_77 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_79 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vadd ins(%alloc_78, %alloc_79 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_80 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_80 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_75 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        %reinterpret_cast_83 = memref.reinterpret_cast %alloc_75 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %subview_84 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %reinterpret_cast_83, %subview_84 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_76 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_77 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_78 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_79 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %alloc_80 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %subview_81 = memref.subview %reinterpret_cast_17[0, %21, 0, %16, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_81, %alloc_75 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        memref.copy %alloc_21, %alloc_76 : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %reinterpret_cast_82 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<1x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%reinterpret_cast_82 : memref<1x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_77 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
        hivm.hir.vcast ins(%alloc_75 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_78 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_77 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_79 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vadd ins(%alloc_78, %alloc_79 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_80 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_80 : memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) outs(%alloc_75 : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>)
        %reinterpret_cast_83 = memref.reinterpret_cast %alloc_75 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %subview_84 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %reinterpret_cast_83, %subview_84 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_76 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_77 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %alloc_78 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_79 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>
        %alloc_80 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %subview_81 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_81, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_82 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_82, %alloc_76 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc_75, %alloc_76, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_78 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
        %subview_83 = memref.subview %4[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_78, %subview_83 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>> to memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<gm>>
        %subview_84 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_84, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_85 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_85, %alloc_77 : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        hivm.hir.mmadL1 ins(%alloc_75, %alloc_77, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_79 : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>)
        %subview_86 = memref.subview %5[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_79, %subview_86 : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>> to memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
        %subview_87 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_87, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_88 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_88, %alloc_76 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc_75, %alloc_76, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_80 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
        %subview_89 = memref.subview %6[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_80, %subview_89 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>> to memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_76 = memref.subview %9[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_76, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.muli %arg29, %c2_i32 : i32
              %24 = arith.addi %23, %arg30 : i32
              %25 = arith.index_cast %24 : i32 to index
              %26 = arith.index_cast %arg31 : i32 to index
              %27 = memref.load %alloc_75[%25, %26] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
              %28 = arith.index_cast %arg29 : i32 to index
              %29 = arith.index_cast %arg30 : i32 to index
              memref.store %27, %alloc_43[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
            }
          }
        }
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.index_cast %arg29 : i32 to index
              %24 = arith.index_cast %arg31 : i32 to index
              %25 = memref.load %alloc_45[%23, %24] : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
              %26 = arith.index_cast %arg30 : i32 to index
              %27 = memref.load %alloc_43[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
              %28 = arith.extf %27 : bf16 to f32
              %29 = arith.mulf %25, %28 : f32
              %30 = arith.subf %29, %29 : f32
              %31 = arith.truncf %30 : f32 to bf16
              %32 = arith.muli %arg29, %c2_i32 : i32
              %33 = arith.addi %32, %arg30 : i32
              %34 = arith.index_cast %33 : i32 to index
              memref.store %31, %alloc_75[%34, %24] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
            }
          }
        }
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.index_cast %arg29 : i32 to index
              %24 = arith.index_cast %arg31 : i32 to index
              %25 = memref.load %alloc_46[%23, %24] : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
              %26 = arith.index_cast %arg30 : i32 to index
              %27 = memref.load %alloc_43[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
              %28 = arith.extf %27 : bf16 to f32
              %29 = arith.mulf %25, %28 : f32
              %30 = arith.addf %29, %29 : f32
              %31 = arith.truncf %30 : f32 to bf16
              %32 = arith.muli %arg29, %c2_i32 : i32
              %33 = arith.addi %32, %arg30 : i32
              %34 = arith.index_cast %33 : i32 to index
              %35 = arith.addi %arg31, %c16_i32 : i32
              %36 = arith.index_cast %35 : i32 to index
              memref.store %31, %alloc_75[%34, %36] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
            }
          }
        }
        %subview_77 = memref.subview %11[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_75, %subview_77 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        hivm.hir.vcast ins(%alloc_24 : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%alloc_75 : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>)
        %subview_76 = memref.subview %13[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_75, %subview_76 : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_76 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_77 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %subview_78 = memref.subview %10[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_78, %alloc_75 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.muli %arg29, %c2_i32 : i32
              %24 = arith.addi %23, %arg30 : i32
              %25 = arith.index_cast %24 : i32 to index
              %26 = arith.index_cast %arg31 : i32 to index
              %27 = memref.load %alloc_75[%25, %26] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
              %28 = arith.index_cast %arg29 : i32 to index
              %29 = arith.index_cast %arg30 : i32 to index
              memref.store %27, %alloc_47[%28, %29, %26] : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
            }
          }
        }
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.index_cast %arg29 : i32 to index
              %24 = arith.index_cast %arg31 : i32 to index
              %25 = memref.load %alloc_45[%23, %24] : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
              %26 = arith.index_cast %arg30 : i32 to index
              %27 = memref.load %alloc_47[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
              %28 = arith.extf %27 : bf16 to f32
              %29 = arith.mulf %25, %28 : f32
              %30 = arith.subf %29, %29 : f32
              %31 = arith.truncf %30 : f32 to bf16
              %32 = arith.muli %arg29, %c2_i32 : i32
              %33 = arith.addi %32, %arg30 : i32
              %34 = arith.index_cast %33 : i32 to index
              memref.store %31, %alloc_75[%34, %24] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
            }
          }
        }
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
            scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
              %23 = arith.index_cast %arg29 : i32 to index
              %24 = arith.index_cast %arg31 : i32 to index
              %25 = memref.load %alloc_46[%23, %24] : memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
              %26 = arith.index_cast %arg30 : i32 to index
              %27 = memref.load %alloc_47[%23, %26, %24] : memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
              %28 = arith.extf %27 : bf16 to f32
              %29 = arith.mulf %25, %28 : f32
              %30 = arith.addf %29, %29 : f32
              %31 = arith.truncf %30 : f32 to bf16
              %32 = arith.muli %arg29, %c2_i32 : i32
              %33 = arith.addi %32, %arg30 : i32
              %34 = arith.index_cast %33 : i32 to index
              %35 = arith.addi %arg31, %c16_i32 : i32
              %36 = arith.index_cast %35 : i32 to index
              memref.store %31, %alloc_75[%34, %36] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
            }
          }
        }
        memref.copy %alloc_75, %alloc_76 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        hivm.hir.vcast ins(%alloc_76 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_77 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
        scf.for %arg29 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %23 = arith.index_cast %arg29 : i32 to index
            %24 = arith.index_cast %arg30 : i32 to index
            %25 = memref.load %alloc_77[%23, %24] : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
            %26 = arith.divsi %arg29, %c2_i32 : i32
            %27 = arith.index_cast %26 : i32 to index
            %28 = memref.load %alloc_42[%27] : memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
            %29 = arith.extf %28 : bf16 to f32
            %30 = arith.mulf %25, %29 : f32
            memref.store %30, %alloc_77[%23, %24] : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          }
        }
        hivm.hir.vcast ins(%alloc_77 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_76 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
        %subview_79 = memref.subview %12[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_76, %subview_79 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scope.scope : () -> () {
        %alloc_75 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %alloc_76 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_77 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>
        %subview_78 = memref.subview %7[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_78, %alloc_76 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %subview_79 = memref.subview %14[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xbf16, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %subview_79, %alloc_75 : memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>> to memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        hivm.hir.mmadL1 ins(%alloc_76, %alloc_75, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_77 : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>)
        %subview_80 = memref.subview %8[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
        memref.copy %alloc_77, %subview_80 : memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>> to memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<gm>>
        scope.return
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    } {tilelangir.num_stages = 2 : i32}
    return
  }
}


  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [getIndexFactor] Checking value: %48 = "arith.trunci"(%47) : (i64) -> i32
  [getIndexFactor] No match
  [getIndexFactor] Checking value: %29 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  [getIndexFactor] No match
  [getIndexFactor] Checking value: %111 = "arith.index_cast"(%105) : (i32) -> index
  [getIndexFactor] Following IndexCastOp
  [getIndexFactor] Checking value: %105 = "arith.muli"(%arg28, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  [getIndexFactor] Checking value: <block argument> of type 'i32' at index: 0
  [getIndexFactor] Found direct match (factor=1)
  [getIndexFactor] Found mul: 16 * 1 = 16
  [getIndexFactor] Checking value: %102 = "arith.index_cast"(%46) : (i32) -> index
  [getIndexFactor] Following IndexCastOp
  [getIndexFactor] Checking value: %46 = "arith.trunci"(%45) : (i64) -> i32
  [getIndexFactor] No match
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [getIndexFactor] Checking value: %48 = "arith.trunci"(%47) : (i64) -> i32
  [getIndexFactor] No match
  [getIndexFactor] Checking value: %29 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  [getIndexFactor] No match
  [getIndexFactor] Checking value: %111 = "arith.index_cast"(%105) : (i32) -> index
  [getIndexFactor] Following IndexCastOp
  [getIndexFactor] Checking value: %105 = "arith.muli"(%arg28, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  [getIndexFactor] Checking value: <block argument> of type 'i32' at index: 0
  [getIndexFactor] Found direct match (factor=1)
  [getIndexFactor] Found mul: 16 * 1 = 16
  [getIndexFactor] Checking value: %102 = "arith.index_cast"(%46) : (i32) -> index
  [getIndexFactor] Following IndexCastOp
  [getIndexFactor] Checking value: %46 = "arith.trunci"(%45) : (i64) -> i32
  [getIndexFactor] No match
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [getIndexFactor] Checking value: %48 = "arith.trunci"(%47) : (i64) -> i32
  [getIndexFactor] No match
  [getIndexFactor] Checking value: %29 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  [getIndexFactor] No match
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [getIndexFactor] Checking value: %48 = "arith.trunci"(%47) : (i64) -> i32
  [getIndexFactor] No match
  [getIndexFactor] Checking value: %29 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  [getIndexFactor] No match
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [getIndexFactor] Checking value: %48 = "arith.trunci"(%47) : (i64) -> i32
  [getIndexFactor] No match
  [getIndexFactor] Checking value: %29 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  [getIndexFactor] No match
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [getIndexFactor] Checking value: %48 = "arith.trunci"(%47) : (i64) -> i32
  [getIndexFactor] No match
  [getIndexFactor] Checking value: %29 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  [getIndexFactor] No match
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [getIndexFactor] Checking value: %48 = "arith.trunci"(%47) : (i64) -> i32
  [getIndexFactor] No match
  [getIndexFactor] Checking value: %29 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  [getIndexFactor] No match
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [getIndexFactor] Checking value: %48 = "arith.trunci"(%47) : (i64) -> i32
  [getIndexFactor] No match
  [getIndexFactor] Checking value: %29 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  [getIndexFactor] No match
WARNING: Expected 6 subviews to be processed, but got 0.
This implies some subviews were skipped or the pass crashed early.
loc("input.mlir":651:24): error: expected 4 offset values, got 3
// -----// IR Dump After TileLangIREnableMultiBuffer Failed (tilelangir-enable-multi-buffer) ('builtin.module' operation) //----- //
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{hacc.arg_type = #hacc.arg_type<ffts_base_address>}, {}, {hacc.arg_type = #hacc.arg_type<workspace>}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}], function_type = (i64, memref<?xi8, #hivm.address_space<gm>>, memref<?xi8, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, i32, i32, i32, i32, i32, i32) -> (), sym_name = "mamba_mimo_fwd_kernel"}> ({
  ^bb0(%arg0: i64, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>>, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xbf16, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: memref<?xf32, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: memref<?xbf16, #hivm.address_space<gm>>, %arg12: memref<?xf32, #hivm.address_space<gm>>, %arg13: memref<?xf32, #hivm.address_space<gm>>, %arg14: memref<?xf32, #hivm.address_space<gm>>, %arg15: memref<?xf32, #hivm.address_space<gm>>, %arg16: memref<?xf32, #hivm.address_space<gm>>, %arg17: memref<?xf32, #hivm.address_space<gm>>, %arg18: memref<?xbf16, #hivm.address_space<gm>>, %arg19: memref<?xf32, #hivm.address_space<gm>>, %arg20: memref<?xf32, #hivm.address_space<gm>>, %arg21: memref<?xbf16, #hivm.address_space<gm>>, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32):
    %0 = "arith.constant"() <{value = -1.98412701E-4 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 0.00833333377 : f32}> : () -> f32
    %2 = "arith.constant"() <{value = -0.166666672 : f32}> : () -> f32
    %3 = "arith.constant"() <{value = -0.00138888892 : f32}> : () -> f32
    %4 = "arith.constant"() <{value = 0.0416666679 : f32}> : () -> f32
    %5 = "arith.constant"() <{value = 2048 : index}> : () -> index
    %6 = "arith.constant"() <{value = 512 : index}> : () -> index
    %7 = "arith.constant"() <{value = 16 : index}> : () -> index
    %8 = "arith.constant"() <{value = 132 : index}> : () -> index
    %9 = "arith.constant"() <{value = 33 : index}> : () -> index
    %10 = "arith.constant"() <{value = 1024 : index}> : () -> index
    %11 = "arith.constant"() <{value = 8 : index}> : () -> index
    %12 = "arith.constant"() <{value = 64 : index}> : () -> index
    %13 = "arith.constant"() <{value = 8192 : index}> : () -> index
    %14 = "arith.constant"() <{value = 256 : index}> : () -> index
    %15 = "arith.constant"() <{value = 128 : index}> : () -> index
    %16 = "arith.constant"() <{value = 32 : index}> : () -> index
    %17 = "arith.constant"() <{value = 1 : index}> : () -> index
    %18 = "arith.constant"() <{value = 0.000000e+00 : bf16}> : () -> bf16
    %19 = "arith.constant"() <{value = -5.000000e-01 : f32}> : () -> f32
    %20 = "arith.constant"() <{value = true}> : () -> i1
    %21 = "arith.constant"() <{value = 31 : i32}> : () -> i32
    %22 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %23 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %24 = "arith.constant"() <{value = -1.000000e+00 : f32}> : () -> f32
    %25 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %26 = "arith.constant"() <{value = 16 : i32}> : () -> i32
    %27 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %28 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %29 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %30 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    %31 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "hivm.hir.set_ffts_base_addr"(%arg0) : (i64) -> ()
    %32 = "memref.reinterpret_cast"(%arg3, %13, %14, %15, %16, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 5>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 32, 2, 4, 32>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xbf16, #hivm.address_space<gm>>, index, index, index, index, index) -> memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %33 = "memref.reinterpret_cast"(%arg5, %13, %14, %12, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 4>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 32, 4, 64>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xbf16, #hivm.address_space<gm>>, index, index, index, index) -> memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %34 = "memref.reinterpret_cast"(%arg9, %15, %12, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 3>, static_offsets = array<i64: 0>, static_sizes = array<i64: 4, 2, 64>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xf32, #hivm.address_space<gm>>, index, index, index) -> memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>
    %35 = "memref.reinterpret_cast"(%arg7, %12, %16, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 3>, static_offsets = array<i64: 0>, static_sizes = array<i64: 4, 2, 32>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xf32, #hivm.address_space<gm>>, index, index, index) -> memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %36 = "memref.reinterpret_cast"(%arg14, %10, %16, %11, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 4>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 32, 4, 8>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xf32, #hivm.address_space<gm>>, index, index, index, index) -> memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>>
    %37 = "memref.reinterpret_cast"(%arg15, %15, %16, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 3>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 4, 32>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xf32, #hivm.address_space<gm>>, index, index, index) -> memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>>
    %38 = "memref.reinterpret_cast"(%arg11, %13, %14, %12, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 4>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 32, 4, 64>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xbf16, #hivm.address_space<gm>>, index, index, index, index) -> memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %39 = "memref.reinterpret_cast"(%arg8, %12, %16, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 3>, static_offsets = array<i64: 0>, static_sizes = array<i64: 4, 2, 32>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xf32, #hivm.address_space<gm>>, index, index, index) -> memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>
    %40 = "memref.reinterpret_cast"(%arg18, %8, %9, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 3>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 4, 33>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xbf16, #hivm.address_space<gm>>, index, index, index) -> memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %41 = "memref.reinterpret_cast"(%arg4, %13, %14, %15, %16, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 5>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 32, 2, 4, 32>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xbf16, #hivm.address_space<gm>>, index, index, index, index, index) -> memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>
    %42 = "memref.reinterpret_cast"(%arg6, %13, %14, %12, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 4>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 32, 4, 64>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xbf16, #hivm.address_space<gm>>, index, index, index, index) -> memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>
    %43 = "memref.reinterpret_cast"(%arg17, %8, %9, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 3>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 4, 33>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xf32, #hivm.address_space<gm>>, index, index, index) -> memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>
    %44 = "memref.reinterpret_cast"(%arg19, %5, %6, %14, %7, %17) <{operandSegmentSizes = array<i32: 1, 0, 0, 5>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 4, 2, 16, 16>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xf32, #hivm.address_space<gm>>, index, index, index, index, index) -> memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>>
    %45 = "hivm.hir.get_block_idx"() : () -> i64
    %46 = "arith.trunci"(%45) : (i64) -> i32
    %47 = "hivm.hir.get_sub_block_idx"() : () -> i64
    %48 = "arith.trunci"(%47) : (i64) -> i32
    %49 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x32x32xf32, #hivm.address_space<gm>>
    %50 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x32x64xf32, #hivm.address_space<gm>>
    %51 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x32x32xf32, #hivm.address_space<gm>>
    %52 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x32x32xbf16, #hivm.address_space<gm>>
    %53 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x32x64xf32, #hivm.address_space<gm>>
    %54 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x32x32xbf16, #hivm.address_space<gm>>
    %55 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x32x32xbf16, #hivm.address_space<gm>>
    %56 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x32x32xbf16, #hivm.address_space<gm>>
    %57 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x32x32xbf16, #hivm.address_space<gm>>
    %58 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x32x64xbf16, #hivm.address_space<gm>>
    %59 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x32x64xbf16, #hivm.address_space<gm>>
    %60 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
    %61 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
    %62 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %63 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %64 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    %65 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
    %66 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    "hivm.hir.vbrc"(%23, %64) <{broadcast_dims = array<i64>}> : (f32, memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>) -> ()
    %67 = "arith.index_cast"(%46) : (i32) -> index
    %68 = "memref.subview"(%34, %67) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 2, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>, index) -> memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    "memref.copy"(%68, %66) : (memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>, memref<2x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>) -> ()
    "hivm.hir.vcast"(%66, %65) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<2x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>, memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>) -> ()
    %69 = "memref.subview"(%35, %67) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 2, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>, index) -> memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    "memref.copy"(%69, %62) : (memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
    "hivm.hir.vcast"(%62, %60) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
    %70 = "memref.subview"(%39, %67) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 2, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>, index) -> memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    "memref.copy"(%70, %63) : (memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
    "hivm.hir.vcast"(%63, %61) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<2x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
    %71 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %72 = "arith.divsi"(%29, %71) : (i32, i32) -> i32
    "scf.for"(%25, %72, %31) ({
    ^bb0(%arg28: i32):
      %73 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %74 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      %75 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %76 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %77 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      %78 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %79 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %80 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      %81 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %82 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>
      %83 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32, strided<[1]>, #hivm.address_space<ub>>
      %84 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32, strided<[1]>, #hivm.address_space<cbuf>>
      %85 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %86 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>
      %87 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
      %88 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %89 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %90 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %91 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>
      %92 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
      %93 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
      %94 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %95 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
      %96 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %97 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %98 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>
      %99 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %100 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %101 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
      %102 = "arith.index_cast"(%46) : (i32) -> index
      %103 = "arith.index_cast"(%arg28) : (i32) -> index
      %104 = "memref.subview"(%44, %102, %103) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 1, 1, 16, 16>, static_strides = array<i64: 1, 1, 1, 1, 1>}> : (memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%104, %73) : (memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>) -> ()
      %105 = "arith.muli"(%arg28, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %106 = "arith.addi"(%105, %31) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %107 = "arith.index_cast"(%106) : (i32) -> index
      %108 = "memref.subview"(%40, %102, %107) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 1, 1, 16>, static_strides = array<i64: 1, 1, 1>}> : (memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%108, %75) : (memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vcast"(%75, %74) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vmul"(%74, %24, %74) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16xf32, strided<[1]>, #hivm.address_space<ub>>, f32, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) -> ()
      %109 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32, #hivm.address_space<ub>>
      "hivm.hir.vsub"(%23, %74, %109) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (f32, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vexp"(%109, %109) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vadd"(%109, %22, %109) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16xf32, #hivm.address_space<ub>>, f32, memref<16xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vdiv"(%22, %109, %74) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (f32, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) -> ()
      %110 = "memref.subview"(%43, %102, %107) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 1, 1, 16>, static_strides = array<i64: 1, 1, 1>}> : (memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%110, %77) : (memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vcast"(%77, %76) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16xf32, strided<[1]>, #hivm.address_space<ub>>, memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vbrc"(%18, %78) <{broadcast_dims = array<i64>}> : (bf16, memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>) -> ()
      "scf.for"(%25, %26, %31) ({
      ^bb0(%arg67: i32):
        %430 = "arith.muli"(%arg28, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %431 = "arith.addi"(%430, %arg67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %432 = "arith.cmpi"(%431, %21) <{predicate = 2 : i64}> : (i32, i32) -> i1
        "scf.if"(%432) ({
          %433 = "arith.index_cast"(%arg67) : (i32) -> index
          %434 = "memref.load"(%76, %433) : (memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>, index) -> bf16
          %435 = "arith.extf"(%434) : (bf16) -> f32
          %436 = "memref.load"(%74, %433) : (memref<16xf32, strided<[1]>, #hivm.address_space<ub>>, index) -> f32
          %437 = "arith.mulf"(%435, %436) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %438 = "arith.truncf"(%437) : (f32) -> bf16
          "memref.store"(%438, %78, %433) : (bf16, memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>, index) -> ()
          "scf.yield"() : () -> ()
        }, {
        }) : (i1) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "memref.copy"(%78, %79) : (memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>, memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>) -> ()
      %111 = "arith.index_cast"(%105) : (i32) -> index
      %112 = "memref.subview"(%40, %102, %111) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 1, 1, 16>, static_strides = array<i64: 1, 1, 1>}> : (memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%112, %81) : (memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vcast"(%81, %80) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) -> ()
      %113 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32, #hivm.address_space<ub>>
      "hivm.hir.vsub"(%23, %80, %113) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (f32, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vexp"(%113, %113) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vadd"(%113, %22, %113) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16xf32, #hivm.address_space<ub>>, f32, memref<16xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vdiv"(%22, %113, %80) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (f32, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) -> ()
      %114 = "memref.subview"(%43, %102, %111) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 1, 1, 16>, static_strides = array<i64: 1, 1, 1>}> : (memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%114, %83) : (memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<16xf32, strided<[1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vcast"(%83, %82) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16xf32, strided<[1]>, #hivm.address_space<ub>>, memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>) -> ()
      "scf.for"(%25, %26, %31) ({
      ^bb0(%arg66: i32):
        %425 = "arith.index_cast"(%arg66) : (i32) -> index
        %426 = "memref.load"(%82, %425) : (memref<16xbf16, strided<[1]>, #hivm.address_space<ub>>, index) -> bf16
        %427 = "arith.extf"(%426) : (bf16) -> f32
        %428 = "memref.load"(%80, %425) : (memref<16xf32, strided<[1]>, #hivm.address_space<ub>>, index) -> f32
        %429 = "arith.mulf"(%427, %428) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%429, %84, %425) : (f32, memref<16xf32, strided<[1]>, #hivm.address_space<cbuf>>, index) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "scf.for"(%25, %26, %31) ({
      ^bb0(%arg65: i32):
        %419 = "arith.index_cast"(%arg65) : (i32) -> index
        %420 = "memref.load"(%84, %419) : (memref<16xf32, strided<[1]>, #hivm.address_space<cbuf>>, index) -> f32
        %421 = "memref.load"(%79, %419) : (memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>, index) -> bf16
        %422 = "arith.extf"(%421) : (bf16) -> f32
        %423 = "arith.addf"(%420, %422) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %424 = "arith.truncf"(%423) : (f32) -> bf16
        "memref.store"(%424, %85, %419) : (bf16, memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>, index) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "memref.copy"(%85, %86) : (memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>, memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>) -> ()
      %115 = "memref.subview"(%36, %111, %102) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 1, 8>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%115, %88) : (memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>) -> ()
      %116 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, #hivm.address_space<ub>>
      %117 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, #hivm.address_space<ub>>
      %118 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, #hivm.address_space<ub>>
      %119 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, #hivm.address_space<ub>>
      "hivm.hir.vmul"(%88, %88, %116) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vmul"(%116, %116, %117) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vmul"(%116, %117, %118) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vmul"(%116, %19, %116) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, f32, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vmul"(%117, %4, %117) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, f32, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vmul"(%118, %3, %118) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, f32, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vadd"(%116, %22, %119) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, f32, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vadd"(%117, %119, %119) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vadd"(%118, %119, %89) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>) -> ()
      %120 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, #hivm.address_space<ub>>
      %121 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, #hivm.address_space<ub>>
      %122 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, #hivm.address_space<ub>>
      %123 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, #hivm.address_space<ub>>
      %124 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, #hivm.address_space<ub>>
      "hivm.hir.vmul"(%88, %88, %120) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vmul"(%120, %88, %121) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vmul"(%121, %120, %122) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vmul"(%122, %120, %123) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vmul"(%121, %2, %121) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, f32, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vmul"(%122, %1, %122) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, f32, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vmul"(%123, %0, %123) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, f32, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vadd"(%88, %121, %124) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vadd"(%122, %124, %124) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vadd"(%123, %124, %90) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, #hivm.address_space<ub>>, memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vexp"(%73, %73) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>, memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>) -> ()
      %125 = "arith.cmpi"(%48, %29) <{predicate = 2 : i64}> : (i32, i32) -> i1
      "scf.if"(%125) ({
        %381 = "arith.muli"(%48, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %382 = "arith.index_cast"(%381) : (i32) -> index
        %383 = "memref.subview"(%51, %382) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x2x32x32xf32, #hivm.address_space<gm>>, index) -> memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%383, %92) : (memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>) -> ()
        %384 = "memref.subview"(%50, %382) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x2x32x64xf32, #hivm.address_space<gm>>, index) -> memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%384, %94) : (memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>) -> ()
        %385 = "arith.index_cast"(%46) : (i32) -> index
        %386 = "arith.muli"(%arg28, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %387 = "arith.muli"(%48, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %388 = "arith.addi"(%386, %387) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %389 = "arith.index_cast"(%388) : (i32) -> index
        %390 = "memref.subview"(%37, %385, %389) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 1, 1, 8>, static_strides = array<i64: 1, 1, 1>}> : (memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>>, index, index) -> memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%390, %95) : (memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<8xf32, strided<[1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vexp"(%95, %95) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<8xf32, strided<[1]>, #hivm.address_space<ub>>, memref<8xf32, strided<[1]>, #hivm.address_space<ub>>) -> ()
        "scf.for"(%25, %26, %31) ({
        ^bb0(%arg62: i32):
          "scf.for"(%25, %30, %31) ({
          ^bb0(%arg64: i32):
            %400 = "arith.divsi"(%arg64, %29) : (i32, i32) -> i32
            %401 = "arith.muli"(%48, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %402 = "arith.divsi"(%arg62, %29) : (i32, i32) -> i32
            %403 = "arith.addi"(%401, %402) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %404 = "arith.cmpi"(%400, %403) <{predicate = 2 : i64}> : (i32, i32) -> i1
            "scf.if"(%404) ({
              %407 = "arith.index_cast"(%arg62) : (i32) -> index
              %408 = "arith.index_cast"(%arg64) : (i32) -> index
              %409 = "memref.load"(%92, %407, %408) : (memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>, index, index) -> f32
              %410 = "arith.muli"(%48, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %411 = "arith.divsi"(%arg62, %29) : (i32, i32) -> i32
              %412 = "arith.addi"(%410, %411) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %413 = "arith.index_cast"(%412) : (i32) -> index
              %414 = "arith.divsi"(%arg64, %29) : (i32, i32) -> i32
              %415 = "arith.index_cast"(%414) : (i32) -> index
              %416 = "memref.load"(%73, %413, %415) : (memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>, index, index) -> f32
              %417 = "arith.mulf"(%409, %416) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
              %418 = "arith.truncf"(%417) : (f32) -> bf16
              "memref.store"(%418, %93, %407, %408) : (bf16, memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, index, index) -> ()
              "scf.yield"() : () -> ()
            }, {
              %405 = "arith.index_cast"(%arg62) : (i32) -> index
              %406 = "arith.index_cast"(%arg64) : (i32) -> index
              "memref.store"(%18, %93, %405, %406) : (bf16, memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, index, index) -> ()
              "scf.yield"() : () -> ()
            }) : (i1) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.for"(%25, %28, %31) ({
          ^bb0(%arg63: i32):
            %393 = "arith.index_cast"(%arg62) : (i32) -> index
            %394 = "arith.index_cast"(%arg63) : (i32) -> index
            %395 = "memref.load"(%94, %393, %394) : (memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>, index, index) -> f32
            %396 = "arith.divsi"(%arg62, %29) : (i32, i32) -> i32
            %397 = "arith.index_cast"(%396) : (i32) -> index
            %398 = "memref.load"(%95, %397) : (memref<8xf32, strided<[1]>, #hivm.address_space<ub>>, index) -> f32
            %399 = "arith.mulf"(%395, %398) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            "memref.store"(%399, %94, %393, %394) : (f32, memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        %391 = "memref.subview"(%52, %382) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%93, %391) : (memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        %392 = "memref.subview"(%50, %382) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x2x32x64xf32, #hivm.address_space<gm>>, index) -> memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%94, %392) : (memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>, memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "scf.yield"() : () -> ()
      }, {
      }) : (i1) -> ()
      "scf.if"(%125) ({
        %361 = "arith.muli"(%48, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %362 = "arith.index_cast"(%361) : (i32) -> index
        %363 = "memref.subview"(%50, %362) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x2x32x64xf32, #hivm.address_space<gm>>, index) -> memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%363, %96) : (memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>) -> ()
        %364 = "memref.subview"(%53, %362) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x2x32x64xf32, #hivm.address_space<gm>>, index) -> memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%364, %97) : (memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>) -> ()
        %365 = "memref.subview"(%49, %362) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x2x32x32xf32, #hivm.address_space<gm>>, index) -> memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%365, %98) : (memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cbuf>>) -> ()
        %366 = "memref.subview"(%59) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x2x32x64xbf16, #hivm.address_space<gm>>) -> memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>
        "memref.copy"(%366, %99) : (memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<gm>>, memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>) -> ()
        %367 = "arith.muli"(%arg28, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %368 = "arith.muli"(%48, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %369 = "arith.addi"(%367, %368) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %370 = "arith.index_cast"(%369) : (i32) -> index
        %371 = "arith.index_cast"(%46) : (i32) -> index
        %372 = "memref.subview"(%38, %370, %371) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>, index, index) -> memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%372, %100) : (memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>, memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>) -> ()
        "scf.for"(%25, %26, %31) ({
        ^bb0(%arg60: i32):
          "scf.for"(%25, %28, %31) ({
          ^bb0(%arg61: i32):
            %376 = "arith.index_cast"(%arg60) : (i32) -> index
            %377 = "arith.index_cast"(%arg61) : (i32) -> index
            %378 = "memref.load"(%96, %376, %377) : (memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>, index, index) -> f32
            %379 = "memref.load"(%97, %376, %377) : (memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>, index, index) -> f32
            %380 = "arith.addf"(%378, %379) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            "memref.store"(%380, %96, %376, %377) : (f32, memref<16x64xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "scf.for"(%25, %27, %31) ({
        ^bb0(%arg58: i32):
          "scf.for"(%25, %28, %31) ({
          ^bb0(%arg59: i32):
            %374 = "arith.index_cast"(%arg58) : (i32) -> index
            %375 = "arith.index_cast"(%arg59) : (i32) -> index
            "memref.store"(%18, %101, %374, %375) : (bf16, memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        %373 = "memref.subview"(%42, %370, %371) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>, index, index) -> memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%101, %373) : (memref<8x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>, memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "scf.yield"() : () -> ()
      }, {
      }) : (i1) -> ()
      %126 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %127 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %128 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      "scf.for"(%126, %127, %128) ({
      ^bb0(%arg56: i32):
        %335 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %336 = "arith.muli"(%arg28, %335) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %337 = "arith.addi"(%336, %arg56) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %338 = "arith.index_cast"(%337) : (i32) -> index
        %339 = "arith.index_cast"(%arg56) : (i32) -> index
        %340 = "memref.subview"(%59, %339) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x64xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x64xbf16, strided<[4096, 2048, 64, 1], offset: ?>, #hivm.address_space<gm>>
        %341 = "memref.collapse_shape"(%340) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x64xbf16, strided<[4096, 2048, 64, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %342 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %343 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %344 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %345 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %346 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        %347 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %348 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %349 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        %350 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        "scf.for"(%25, %26, %31) ({
        ^bb0(%arg57: i32):
          %354 = "arith.muli"(%arg28, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %355 = "arith.addi"(%354, %arg57) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %356 = "arith.index_cast"(%355) : (i32) -> index
          %357 = "arith.index_cast"(%46) : (i32) -> index
          %358 = "memref.subview"(%33, %356, %357) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>, index, index) -> memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %359 = "arith.index_cast"(%arg57) : (i32) -> index
          %360 = "memref.subview"(%342, %359) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 1, 64>, static_strides = array<i64: 1, 1>}> : (memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>, index) -> memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
          "memref.copy"(%358, %360) : (memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "memref.copy"(%342, %344) : (memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>, memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>) -> ()
        %351 = "memref.reinterpret_cast"(%344) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 16, 1, 64>, static_strides = array<i64: 64, 64, 1>}> : (memref<16x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>) -> memref<16x1x64xbf16, strided<[64, 64, 1]>, #hivm.address_space<ub>>
        "hivm.hir.vbrc"(%351, %345) <{broadcast_dims = array<i64: 1>}> : (memref<16x1x64xbf16, strided<[64, 64, 1]>, #hivm.address_space<ub>>, memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) -> ()
        "memref.copy"(%65, %346) : (memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>, memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>) -> ()
        %352 = "memref.reinterpret_cast"(%346) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 2, 64>, static_strides = array<i64: 128, 64, 1>}> : (memref<2x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>) -> memref<1x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>
        "hivm.hir.vbrc"(%352, %347) <{broadcast_dims = array<i64: 0>}> : (memref<1x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>, memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vcast"(%345, %348) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>, memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vcast"(%347, %349) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>, memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vmul"(%348, %349, %350) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>, memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>, memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vcast"(%350, %343) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<ub>>, memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) -> ()
        %353 = "memref.reinterpret_cast"(%343) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 64, 1>}> : (memref<16x2x64xbf16, strided<[128, 64, 1]>, #hivm.address_space<ub>>) -> memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        "memref.copy"(%353, %341) : (memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>, memref<32x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "scf.yield"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : (i32, i32, i32) -> ()
      %129 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %130 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %131 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      "scf.for"(%129, %130, %131) ({
      ^bb0(%arg55: i32):
        %316 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %317 = "arith.muli"(%arg28, %316) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %318 = "arith.addi"(%317, %arg55) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %319 = "arith.index_cast"(%318) : (i32) -> index
        %320 = "arith.index_cast"(%arg55) : (i32) -> index
        %321 = "arith.constant"() <{value = 16 : i32}> : () -> i32
        %322 = "arith.muli"(%318, %321) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %323 = "arith.index_cast"(%322) : (i32) -> index
        %324 = "memref.subview"(%32, %323, %102) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 2, 1, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>}> : (memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>, index, index) -> memref<1x16x2x1x32xbf16, strided<[8192, 256, 128, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %325 = "memref.subview"(%54, %320) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %326 = "memref.collapse_shape"(%325) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %327 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %328 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %329 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %330 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %331 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %332 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        "memref.copy"(%324, %327) : (memref<1x16x2x1x32xbf16, strided<[8192, 256, 128, 32, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> ()
        "memref.copy"(%60, %328) : (memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>, memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
        %333 = "memref.reinterpret_cast"(%328) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 2, 32>, static_strides = array<i64: 64, 32, 1>}> : (memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) -> memref<1x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        "hivm.hir.vbrc"(%333, %329) <{broadcast_dims = array<i64: 0>}> : (memref<1x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vcast"(%327, %330) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vcast"(%329, %331) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vadd"(%330, %331, %332) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vcast"(%332, %327) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> ()
        %334 = "memref.reinterpret_cast"(%327) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 32, 1>}> : (memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        "memref.copy"(%334, %326) : (memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "scf.yield"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : (i32, i32, i32) -> ()
      %132 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %133 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %134 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      "scf.for"(%132, %133, %134) ({
      ^bb0(%arg54: i32):
        %297 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %298 = "arith.muli"(%arg28, %297) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %299 = "arith.addi"(%298, %arg54) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %300 = "arith.index_cast"(%299) : (i32) -> index
        %301 = "arith.index_cast"(%arg54) : (i32) -> index
        %302 = "arith.constant"() <{value = 16 : i32}> : () -> i32
        %303 = "arith.muli"(%299, %302) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %304 = "arith.index_cast"(%303) : (i32) -> index
        %305 = "memref.subview"(%41, %304, %102) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 2, 1, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>}> : (memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>, index, index) -> memref<1x16x2x1x32xbf16, strided<[8192, 256, 128, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %306 = "memref.subview"(%55, %301) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %307 = "memref.collapse_shape"(%306) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %308 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %309 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %310 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %311 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %312 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        %313 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        "memref.copy"(%305, %308) : (memref<1x16x2x1x32xbf16, strided<[8192, 256, 128, 32, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> ()
        "memref.copy"(%61, %309) : (memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>, memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
        %314 = "memref.reinterpret_cast"(%309) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 2, 32>, static_strides = array<i64: 64, 32, 1>}> : (memref<2x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) -> memref<1x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>
        "hivm.hir.vbrc"(%314, %310) <{broadcast_dims = array<i64: 0>}> : (memref<1x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vcast"(%308, %311) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vcast"(%310, %312) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vadd"(%311, %312, %313) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vcast"(%313, %308) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<ub>>, memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> ()
        %315 = "memref.reinterpret_cast"(%308) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 32, 1>}> : (memref<16x2x32xbf16, strided<[64, 32, 1]>, #hivm.address_space<ub>>) -> memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        "memref.copy"(%315, %307) : (memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "scf.yield"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : (i32, i32, i32) -> ()
      %135 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %136 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %137 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      "scf.for"(%135, %136, %137) ({
      ^bb0(%arg53: i32):
        %268 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %269 = "arith.muli"(%arg28, %268) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %270 = "arith.addi"(%269, %arg53) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %271 = "arith.index_cast"(%270) : (i32) -> index
        %272 = "arith.index_cast"(%arg53) : (i32) -> index
        %273 = "memref.subview"(%54, %272) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %274 = "memref.collapse_shape"(%273) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %275 = "memref.subview"(%55, %272) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %276 = "memref.collapse_shape"(%275) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %277 = "memref.subview"(%49, %272) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xf32, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xf32, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %278 = "memref.collapse_shape"(%277) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xf32, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %279 = "memref.subview"(%56, %272) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %280 = "memref.collapse_shape"(%279) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %281 = "memref.subview"(%58, %272) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x64xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x64xbf16, strided<[4096, 2048, 64, 1], offset: ?>, #hivm.address_space<gm>>
        %282 = "memref.collapse_shape"(%281) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x64xbf16, strided<[4096, 2048, 64, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %283 = "memref.subview"(%50, %272) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x64xf32, #hivm.address_space<gm>>, index) -> memref<1x1x32x64xf32, strided<[4096, 2048, 64, 1], offset: ?>, #hivm.address_space<gm>>
        %284 = "memref.collapse_shape"(%283) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x64xf32, strided<[4096, 2048, 64, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %285 = "memref.subview"(%56, %272) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %286 = "memref.collapse_shape"(%285) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %287 = "memref.subview"(%57, %272) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %288 = "memref.collapse_shape"(%287) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %289 = "memref.subview"(%51, %272) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xf32, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xf32, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %290 = "memref.collapse_shape"(%289) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xf32, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %291 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %292 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %293 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %294 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %295 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>
        %296 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        "memref.copy"(%274, %291) : (memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) -> ()
        "memref.copy"(%276, %292) : (memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) -> ()
        "hivm.hir.mmadL1"(%291, %292, %20, %16, %16, %16, %294) <{b_transpose, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> : (memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) -> ()
        "memref.copy"(%294, %278) : (memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>, memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "memref.copy"(%280, %291) : (memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) -> ()
        "memref.copy"(%282, %293) : (memref<32x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>, memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>) -> ()
        "hivm.hir.mmadL1"(%291, %293, %20, %16, %16, %12, %295) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> : (memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>) -> ()
        "memref.copy"(%295, %284) : (memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>, memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "memref.copy"(%286, %291) : (memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) -> ()
        "memref.copy"(%288, %292) : (memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) -> ()
        "hivm.hir.mmadL1"(%291, %292, %20, %16, %16, %16, %296) <{b_transpose, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> : (memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) -> ()
        "memref.copy"(%296, %290) : (memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>, memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "scf.yield"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<CUBE>} : (i32, i32, i32) -> ()
      %138 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %139 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %140 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      "scf.for"(%138, %139, %140) ({
      ^bb0(%arg43: i32):
        %225 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %226 = "arith.muli"(%arg28, %225) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %227 = "arith.addi"(%226, %arg43) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %228 = "arith.index_cast"(%227) : (i32) -> index
        %229 = "arith.index_cast"(%arg43) : (i32) -> index
        %230 = "memref.subview"(%54, %229) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %231 = "memref.collapse_shape"(%230) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %232 = "memref.subview"(%56, %229) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %233 = "memref.collapse_shape"(%232) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %234 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        "memref.copy"(%231, %234) : (memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) -> ()
        "scf.for"(%25, %26, %31) ({
        ^bb0(%arg50: i32):
          "scf.for"(%25, %29, %31) ({
          ^bb0(%arg51: i32):
            "scf.for"(%25, %27, %31) ({
            ^bb0(%arg52: i32):
              %261 = "arith.muli"(%arg50, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %262 = "arith.addi"(%261, %arg51) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %263 = "arith.index_cast"(%262) : (i32) -> index
              %264 = "arith.index_cast"(%arg52) : (i32) -> index
              %265 = "memref.load"(%234, %263, %264) : (memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, index, index) -> bf16
              %266 = "arith.index_cast"(%arg50) : (i32) -> index
              %267 = "arith.index_cast"(%arg51) : (i32) -> index
              "memref.store"(%265, %87, %266, %267, %264) : (bf16, memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>, index, index, index) -> ()
              "scf.yield"() : () -> ()
            }) : (i32, i32, i32) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "scf.for"(%25, %26, %31) ({
        ^bb0(%arg47: i32):
          "scf.for"(%25, %29, %31) ({
          ^bb0(%arg48: i32):
            "scf.for"(%25, %27, %31) ({
            ^bb0(%arg49: i32):
              %249 = "arith.index_cast"(%arg47) : (i32) -> index
              %250 = "arith.index_cast"(%arg49) : (i32) -> index
              %251 = "memref.load"(%89, %249, %250) : (memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, index, index) -> f32
              %252 = "arith.index_cast"(%arg48) : (i32) -> index
              %253 = "memref.load"(%87, %249, %252, %250) : (memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>, index, index, index) -> bf16
              %254 = "arith.extf"(%253) : (bf16) -> f32
              %255 = "arith.mulf"(%251, %254) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
              %256 = "arith.subf"(%255, %255) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
              %257 = "arith.truncf"(%256) : (f32) -> bf16
              %258 = "arith.muli"(%arg47, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %259 = "arith.addi"(%258, %arg48) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %260 = "arith.index_cast"(%259) : (i32) -> index
              "memref.store"(%257, %234, %260, %250) : (bf16, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, index, index) -> ()
              "scf.yield"() : () -> ()
            }) : (i32, i32, i32) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "scf.for"(%25, %26, %31) ({
        ^bb0(%arg44: i32):
          "scf.for"(%25, %29, %31) ({
          ^bb0(%arg45: i32):
            "scf.for"(%25, %27, %31) ({
            ^bb0(%arg46: i32):
              %235 = "arith.index_cast"(%arg44) : (i32) -> index
              %236 = "arith.index_cast"(%arg46) : (i32) -> index
              %237 = "memref.load"(%90, %235, %236) : (memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, index, index) -> f32
              %238 = "arith.index_cast"(%arg45) : (i32) -> index
              %239 = "memref.load"(%87, %235, %238, %236) : (memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>, index, index, index) -> bf16
              %240 = "arith.extf"(%239) : (bf16) -> f32
              %241 = "arith.mulf"(%237, %240) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
              %242 = "arith.addf"(%241, %241) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
              %243 = "arith.truncf"(%242) : (f32) -> bf16
              %244 = "arith.muli"(%arg44, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %245 = "arith.addi"(%244, %arg45) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %246 = "arith.index_cast"(%245) : (i32) -> index
              %247 = "arith.addi"(%arg46, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %248 = "arith.index_cast"(%247) : (i32) -> index
              "memref.store"(%243, %234, %246, %248) : (bf16, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, index, index) -> ()
              "scf.yield"() : () -> ()
            }) : (i32, i32, i32) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "memref.copy"(%234, %233) : (memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "scf.yield"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>} : (i32, i32, i32) -> ()
      %141 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %142 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %143 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      "scf.for"(%141, %142, %143) ({
      ^bb0(%arg42: i32):
        %217 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %218 = "arith.muli"(%arg28, %217) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %219 = "arith.addi"(%218, %arg42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %220 = "arith.index_cast"(%219) : (i32) -> index
        %221 = "arith.index_cast"(%arg42) : (i32) -> index
        %222 = "memref.subview"(%58, %221) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x64xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x64xbf16, strided<[4096, 2048, 64, 1], offset: ?>, #hivm.address_space<gm>>
        %223 = "memref.collapse_shape"(%222) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x64xbf16, strided<[4096, 2048, 64, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %224 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>
        "hivm.hir.vcast"(%64, %224) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>, memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>) -> ()
        "memref.copy"(%224, %223) : (memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<ub>>, memref<32x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "scf.yield"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : (i32, i32, i32) -> ()
      %144 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %145 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %146 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      "scf.for"(%144, %145, %146) ({
      ^bb0(%arg30: i32):
        %164 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %165 = "arith.muli"(%arg28, %164) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %166 = "arith.addi"(%165, %arg30) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %167 = "arith.index_cast"(%166) : (i32) -> index
        %168 = "arith.index_cast"(%arg30) : (i32) -> index
        %169 = "memref.subview"(%55, %168) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %170 = "memref.collapse_shape"(%169) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %171 = "memref.subview"(%57, %168) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %172 = "memref.collapse_shape"(%171) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %173 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %174 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %175 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        "memref.copy"(%170, %173) : (memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
        "scf.for"(%25, %26, %31) ({
        ^bb0(%arg39: i32):
          "scf.for"(%25, %29, %31) ({
          ^bb0(%arg40: i32):
            "scf.for"(%25, %27, %31) ({
            ^bb0(%arg41: i32):
              %210 = "arith.muli"(%arg39, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %211 = "arith.addi"(%210, %arg40) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %212 = "arith.index_cast"(%211) : (i32) -> index
              %213 = "arith.index_cast"(%arg41) : (i32) -> index
              %214 = "memref.load"(%173, %212, %213) : (memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>, index, index) -> bf16
              %215 = "arith.index_cast"(%arg39) : (i32) -> index
              %216 = "arith.index_cast"(%arg40) : (i32) -> index
              "memref.store"(%214, %91, %215, %216, %213) : (bf16, memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>, index, index, index) -> ()
              "scf.yield"() : () -> ()
            }) : (i32, i32, i32) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "scf.for"(%25, %26, %31) ({
        ^bb0(%arg36: i32):
          "scf.for"(%25, %29, %31) ({
          ^bb0(%arg37: i32):
            "scf.for"(%25, %27, %31) ({
            ^bb0(%arg38: i32):
              %198 = "arith.index_cast"(%arg36) : (i32) -> index
              %199 = "arith.index_cast"(%arg38) : (i32) -> index
              %200 = "memref.load"(%89, %198, %199) : (memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, index, index) -> f32
              %201 = "arith.index_cast"(%arg37) : (i32) -> index
              %202 = "memref.load"(%91, %198, %201, %199) : (memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>, index, index, index) -> bf16
              %203 = "arith.extf"(%202) : (bf16) -> f32
              %204 = "arith.mulf"(%200, %203) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
              %205 = "arith.subf"(%204, %204) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
              %206 = "arith.truncf"(%205) : (f32) -> bf16
              %207 = "arith.muli"(%arg36, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %208 = "arith.addi"(%207, %arg37) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %209 = "arith.index_cast"(%208) : (i32) -> index
              "memref.store"(%206, %173, %209, %199) : (bf16, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>, index, index) -> ()
              "scf.yield"() : () -> ()
            }) : (i32, i32, i32) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "scf.for"(%25, %26, %31) ({
        ^bb0(%arg33: i32):
          "scf.for"(%25, %29, %31) ({
          ^bb0(%arg34: i32):
            "scf.for"(%25, %27, %31) ({
            ^bb0(%arg35: i32):
              %184 = "arith.index_cast"(%arg33) : (i32) -> index
              %185 = "arith.index_cast"(%arg35) : (i32) -> index
              %186 = "memref.load"(%90, %184, %185) : (memref<16x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>, index, index) -> f32
              %187 = "arith.index_cast"(%arg34) : (i32) -> index
              %188 = "memref.load"(%91, %184, %187, %185) : (memref<16x2x8xbf16, strided<[16, 8, 1]>, #hivm.address_space<cbuf>>, index, index, index) -> bf16
              %189 = "arith.extf"(%188) : (bf16) -> f32
              %190 = "arith.mulf"(%186, %189) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
              %191 = "arith.addf"(%190, %190) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
              %192 = "arith.truncf"(%191) : (f32) -> bf16
              %193 = "arith.muli"(%arg33, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %194 = "arith.addi"(%193, %arg34) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %195 = "arith.index_cast"(%194) : (i32) -> index
              %196 = "arith.addi"(%arg35, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %197 = "arith.index_cast"(%196) : (i32) -> index
              "memref.store"(%192, %173, %195, %197) : (bf16, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>, index, index) -> ()
              "scf.yield"() : () -> ()
            }) : (i32, i32, i32) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "memref.copy"(%173, %174) : (memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
        "hivm.hir.vcast"(%174, %175) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
        "scf.for"(%25, %30, %31) ({
        ^bb0(%arg31: i32):
          "scf.for"(%25, %30, %31) ({
          ^bb0(%arg32: i32):
            %176 = "arith.index_cast"(%arg31) : (i32) -> index
            %177 = "arith.index_cast"(%arg32) : (i32) -> index
            %178 = "memref.load"(%175, %176, %177) : (memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, index, index) -> f32
            %179 = "arith.divsi"(%arg31, %29) : (i32, i32) -> i32
            %180 = "arith.index_cast"(%179) : (i32) -> index
            %181 = "memref.load"(%86, %180) : (memref<16xbf16, strided<[1]>, #hivm.address_space<cbuf>>, index) -> bf16
            %182 = "arith.extf"(%181) : (bf16) -> f32
            %183 = "arith.mulf"(%178, %182) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            "memref.store"(%183, %175, %176, %177) : (f32, memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "hivm.hir.vcast"(%175, %174) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
        "memref.copy"(%174, %172) : (memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "scf.yield"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : (i32, i32, i32) -> ()
      %147 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %148 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %149 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      "scf.for"(%147, %148, %149) ({
      ^bb0(%arg29: i32):
        %150 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %151 = "arith.muli"(%arg28, %150) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %152 = "arith.addi"(%151, %arg29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %153 = "arith.index_cast"(%152) : (i32) -> index
        %154 = "arith.index_cast"(%arg29) : (i32) -> index
        %155 = "memref.subview"(%52, %154) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %156 = "memref.collapse_shape"(%155) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        %157 = "memref.subview"(%59, %154) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x64xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x64xbf16, strided<[4096, 2048, 64, 1], offset: ?>, #hivm.address_space<gm>>
        %158 = "memref.collapse_shape"(%157) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x64xbf16, strided<[4096, 2048, 64, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %159 = "memref.subview"(%53, %154) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x64xf32, #hivm.address_space<gm>>, index) -> memref<1x1x32x64xf32, strided<[4096, 2048, 64, 1], offset: ?>, #hivm.address_space<gm>>
        %160 = "memref.collapse_shape"(%159) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x64xf32, strided<[4096, 2048, 64, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %161 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>
        %162 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %163 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>
        "memref.copy"(%156, %162) : (memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) -> ()
        "memref.copy"(%158, %161) : (memref<32x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>, memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>) -> ()
        "hivm.hir.mmadL1"(%162, %161, %20, %16, %16, %12, %163) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> : (memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x64xbf16, strided<[64, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>) -> ()
        "memref.copy"(%163, %160) : (memref<32x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>, memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "scf.yield"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<CUBE>} : (i32, i32, i32) -> ()
      "scf.yield"() : () -> ()
    }) {tilelangir.num_stages = 2 : i32} : (i32, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} : () -> ()
}) {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} : () -> ()



Traceback (most recent call last):
  File "/home/z00910011/tilelang-ascend/examples/mamba/mamba3_mimo_fwd_npu_mix.py", line 963, in <module>
    run_test()
  File "/home/z00910011/tilelang-ascend/examples/mamba/mamba3_mimo_fwd_npu_mix.py", line 915, in run_test
    kernel = mamba_mimo_fwd(
             ^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend/tilelang/jit/__init__.py", line 197, in wrapper
    kernel_result = compile(
                    ^^^^^^^^
  File "/home/z00910011/tilelang-ascend/tilelang/jit/__init__.py", line 74, in compile
    return compile_npuir.compile(func, out_idx)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend/tilelang/jit/jit_npu.py", line 1125, in compile
    mlir_path = lower(self.mod)
                ^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend/tilelang/engine/lower.py", line 305, in lower
    mlir_str = pipeline.run(mlir_str)
               ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend/tilelang/tladapter/utils.py", line 103, in run
    return self._pp.run(mlir_str)
           ^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Pass pipeline run failed
[ERROR] 2026-04-27-19:50:18 (PID:1879048, Device:11, RankID:-1) ERR99999 UNKNOWN applicaiton exception
