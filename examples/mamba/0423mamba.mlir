2026-04-23 20:18:10  [TileLang:tilelang.env:WARNING]: Loading tilelang libs from dev root: /home/z00910011/tilelang-ascend/build
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
            k_frag = T.decl_buffer((16, 2, 32), "bfloat16", scope="local.fragment")
            qk_dot_frag = T.decl_buffer((32, 32), scope="local.fragment")
            q_first_half_frag = T.decl_buffer((16, 2, 8), "bfloat16", scope="local.fragment")
            q_second_half_frag = T.decl_buffer((16, 2, 8), "bfloat16", scope="local.fragment")
            angles_frag = T.decl_buffer((16, 8), scope="local.fragment")
            angles_frag_cos = T.decl_buffer((16, 8), scope="local.fragment")
            angles_frag_sin = T.decl_buffer((16, 8), scope="local.fragment")
            o_mimo_accum_frag = T.decl_buffer((32, 64), scope="local.fragment")
            states_frag_cast = T.decl_buffer((32, 64), "bfloat16", scope="local.fragment")
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
            T.copy(T.region(PsiV_reshaped_frag[0, 0], 1, 32, 64), T.region(PsiV_shared[0, 0], 2, 32, 64))
            T.copy(T.region(Q[0, i * 16, 0, cid, 0], 1, 1, 16, 2, 1, 32), T.region(q_frag[0, 0, 0], 2, 16, 2, 32))
            T.copy(T.region(q_bias_frag[0, 0], 1, 2, 32), T.region(local_src_buf_2[0, 0], 2, 2, 32))
            T.npuir_reshape(T.region(local_src_buf_2[0, 0], 1, 2, 32), T.region(reshape_view_buf_2[0, 0, 0], 2, 1, 2, 32))
            T.npuir_brc(T.region(reshape_view_buf_2[0, 0, 0], 1, 1, 2, 32), T.region(brc_buf_2[0, 0, 0], 2, 16, 2, 32))
            T.npuir_cast(T.region(q_frag[0, 0, 0], 1, 16, 2, 32), T.region(bf16_src_f32_3[0, 0, 0], 2, 16, 2, 32), "rint")
            T.npuir_cast(T.region(brc_buf_2[0, 0, 0], 1, 16, 2, 32), T.region(bf16_src_f32_4[0, 0, 0], 2, 16, 2, 32), "rint")
            T.npuir_add(T.region(bf16_src_f32_3[0, 0, 0], 1, 16, 2, 32), T.region(bf16_src_f32_4[0, 0, 0], 1, 16, 2, 32), T.region(bf16_dst_f32_5[0, 0, 0], 2, 16, 2, 32))
            T.npuir_cast(T.region(bf16_dst_f32_5[0, 0, 0], 1, 16, 2, 32), T.region(q_frag[0, 0, 0], 2, 16, 2, 32), "rint")
            T.npuir_reshape(T.region(q_frag[0, 0, 0], 2, 16, 2, 32), T.region(q_shared[0, 0], 1, 32, 32))
            T.copy(T.region(K[0, i * 16, 0, cid, 0], 1, 1, 16, 2, 1, 32), T.region(k_frag[0, 0, 0], 2, 16, 2, 32))
            T.copy(T.region(k_bias_frag[0, 0], 1, 2, 32), T.region(local_src_buf_3[0, 0], 2, 2, 32))
            T.npuir_reshape(T.region(local_src_buf_3[0, 0], 1, 2, 32), T.region(reshape_view_buf_3[0, 0, 0], 2, 1, 2, 32))
            T.npuir_brc(T.region(reshape_view_buf_3[0, 0, 0], 1, 1, 2, 32), T.region(brc_buf_3[0, 0, 0], 2, 16, 2, 32))
            T.npuir_cast(T.region(k_frag[0, 0, 0], 1, 16, 2, 32), T.region(bf16_src_f32_6[0, 0, 0], 2, 16, 2, 32), "rint")
            T.npuir_cast(T.region(brc_buf_3[0, 0, 0], 1, 16, 2, 32), T.region(bf16_src_f32_7[0, 0, 0], 2, 16, 2, 32), "rint")
            T.npuir_add(T.region(bf16_src_f32_6[0, 0, 0], 1, 16, 2, 32), T.region(bf16_src_f32_7[0, 0, 0], 1, 16, 2, 32), T.region(bf16_dst_f32_8[0, 0, 0], 2, 16, 2, 32))
            T.npuir_cast(T.region(bf16_dst_f32_8[0, 0, 0], 1, 16, 2, 32), T.region(k_frag[0, 0, 0], 2, 16, 2, 32), "rint")
            T.npuir_reshape(T.region(k_frag[0, 0, 0], 2, 16, 2, 32), T.region(k_shared[0, 0], 1, 32, 32))
            T.npuir_dot(T.region(q_shared[0, 0], 1, 32, 32), T.region(k_shared[0, 0], 1, 32, 32), T.region(qk_dot_frag[0, 0], 3, 32, 32), T.bool(True), T.bool(False), T.bool(True))
            T.copy(T.region(qk_dot_frag[0, 0], 1, 32, 32), T.region(qk_dot_ws[0, 0, 0], 2, 1, 32, 32))
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        q_first_half_frag[cs_split, r_split, n_split] = q_shared[cs_split * 2 + r_split, n_split]
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        q_second_half_frag[cs_split, r_split, n_split] = q_shared[cs_split * 2 + r_split, n_split + 16]
            T.copy(T.region(ANGLES[0, i * 16, cid, 0], 1, 1, 16, 1, 8), T.region(angles_frag[0, 0], 2, 16, 8))
            T.npuir_vcos(T.region(angles_frag_cos[0, 0], 2, 16, 8), T.region(angles_frag[0, 0], 1, 16, 8))
            T.npuir_vsin(T.region(angles_frag_sin[0, 0], 2, 16, 8), T.region(angles_frag[0, 0], 1, 16, 8))
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        q_shared[cs_split * 2 + r_split, n_split] = T.Cast("bfloat16", angles_frag_cos[cs_split, n_split] * T.Cast("float32", q_first_half_frag[cs_split, r_split, n_split]) - angles_frag_sin[cs_split, n_split] * T.Cast("float32", q_second_half_frag[cs_split, r_split, n_split]))
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        q_shared[cs_split * 2 + r_split, n_split + 16] = T.Cast("bfloat16", angles_frag_sin[cs_split, n_split] * T.Cast("float32", q_first_half_frag[cs_split, r_split, n_split]) + angles_frag_cos[cs_split, n_split] * T.Cast("float32", q_second_half_frag[cs_split, r_split, n_split]))
            T.npuir_cast(T.region(states_frag[0, 0], 1, 32, 64), T.region(states_frag_cast[0, 0], 2, 32, 64), "rint")
            T.copy(T.region(states_frag_cast[0, 0], 1, 32, 64), T.region(states_accum_cast_shared[0, 0], 2, 32, 64))
            T.npuir_dot(T.region(q_shared[0, 0], 1, 32, 32), T.region(states_accum_cast_shared[0, 0], 1, 32, 64), T.region(o_mimo_accum_frag[0, 0], 3, 32, 64), T.bool(True), T.bool(False), T.bool(False))
            T.copy(T.region(o_mimo_accum_frag[0, 0], 1, 32, 64), T.region(o_inter_ws[0, 0, 0], 2, 1, 32, 64))
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        k_first_half_frag[cs_split, r_split, n_split] = k_shared[cs_split * 2 + r_split, n_split]
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        k_second_half_frag[cs_split, r_split, n_split] = k_shared[cs_split * 2 + r_split, n_split + 16]
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        k_shared[cs_split * 2 + r_split, n_split] = T.Cast("bfloat16", angles_frag_cos[cs_split, n_split] * T.Cast("float32", k_first_half_frag[cs_split, r_split, n_split]) - angles_frag_sin[cs_split, n_split] * T.Cast("float32", k_second_half_frag[cs_split, r_split, n_split]))
            for cs_split in T.parallel(16):
                for r_split in T.parallel(2):
                    for n_split in T.parallel(8):
                        k_shared[cs_split * 2 + r_split, n_split + 16] = T.Cast("bfloat16", angles_frag_sin[cs_split, n_split] * T.Cast("float32", k_first_half_frag[cs_split, r_split, n_split]) + angles_frag_cos[cs_split, n_split] * T.Cast("float32", k_second_half_frag[cs_split, r_split, n_split]))
            T.copy(T.region(k_shared[0, 0], 1, 32, 32), T.region(k_trap_scaled_frag[0, 0], 2, 32, 32))
            T.npuir_cast(T.region(k_trap_scaled_frag[0, 0], 1, 32, 32), T.region(k_trap_scaled_frag_f32[0, 0], 2, 32, 32), "rint")
            for csr, n in T.grid(32, 32):
                k_trap_scaled_frag_f32[csr, n] = k_trap_scaled_frag_f32[csr, n] * T.Cast("float32", trap_scale_shared[csr // 2])
            T.npuir_cast(T.region(k_trap_scaled_frag_f32[0, 0], 1, 32, 32), T.region(k_trap_scaled_frag[0, 0], 2, 32, 32), "rint")
            T.copy(T.region(k_trap_scaled_frag[0, 0], 1, 32, 32), T.region(k_shared[0, 0], 2, 32, 32))
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
            T.npuir_dot(T.region(qk_intrachunk_shared[0, 0], 1, 32, 32), T.region(PsiV_shared[0, 0], 1, 32, 64), T.region(tmp[0, 0], 3, 32, 64), T.bool(True), T.bool(False), T.bool(False))
            T.copy(T.region(tmp[0, 0], 1, 32, 64), T.region(o_intra_ws[0, 0, 0], 2, 1, 32, 64))
            if vid < 2:
                T.copy(T.region(o_inter_ws[0, vid * 16, 0], 1, 1, 16, 64), T.region(o_inter_vec_1[0, 0], 2, 16, 64))
                T.copy(T.region(o_intra_ws[0, vid * 16, 0], 1, 1, 16, 64), T.region(o_intra_vec[0, 0], 2, 16, 64))
                T.copy(T.region(qk_dot_ws[0, vid * 16, 0], 1, 1, 16, 32), T.region(qk_dot_vec[0, 0], 2, 16, 32))
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
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_18 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_19 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_20 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_21 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
    %alloc_22 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
    %alloc_23 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>>
    %c0_i32 = arith.constant 0 : i32
    %50 = arith.sitofp %c0_i32 : i32 to f32
    hivm.hir.vbrc ins(%50 : f32) outs(%alloc_21 : memref<32x64xf32, strided<[64, 1]>>)
    %51 = arith.index_cast %42 : i32 to index
    %subview = memref.subview %reinterpret_cast_1[%51, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_23 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x64xf32, strided<[64, 1]>>) outs(%alloc_22 : memref<2x64xbf16, strided<[64, 1]>>)
    %subview_24 = memref.subview %reinterpret_cast_2[%51, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_24, %alloc_19 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_19 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>>)
    %subview_25 = memref.subview %reinterpret_cast_7[%51, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
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
      %alloc_50 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_51 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_52 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_53 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_54 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_55 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_56 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_57 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_58 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_59 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_60 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_61 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_62 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_63 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_64 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_65 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
      %alloc_66 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_67 = memref.alloc() : memref<8xf32, strided<[1]>>
      %alloc_68 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_69 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_70 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_71 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_72 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_73 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_74 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_75 = memref.alloc() : memref<16x1x64xbf16, strided<[64, 64, 1]>>
      %alloc_76 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_77 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
      %alloc_78 = memref.alloc() : memref<1x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_79 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_80 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_81 = memref.alloc() : memref<1x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_82 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_83 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_84 = memref.alloc() : memref<1x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_85 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_86 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_87 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_88 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_89 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_90 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_91 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_92 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_93 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_94 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %52 = arith.index_cast %42 : i32 to index
      %53 = arith.index_cast %arg28 : i32 to index
      %subview_95 = memref.subview %reinterpret_cast_14[0, %52, %53, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_95, %alloc_33 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>>
      %c16_i32_96 = arith.constant 16 : i32
      %54 = arith.muli %arg28, %c16_i32_96 : i32
      %c1_i32_97 = arith.constant 1 : i32
      %55 = arith.addi %54, %c1_i32_97 : i32
      %56 = arith.index_cast %55 : i32 to index
      %subview_98 = memref.subview %reinterpret_cast_8[0, %52, %56] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_98, %alloc_35 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_35 : memref<16xbf16, strided<[1]>>) outs(%alloc_34 : memref<16xf32, strided<[1]>>)
      %cst = arith.constant -1.000000e+00 : f32
      hivm.hir.vmul ins(%alloc_34, %cst : memref<16xf32, strided<[1]>>, f32) outs(%alloc_34 : memref<16xf32, strided<[1]>>)
      %alloc_99 = memref.alloc() : memref<16xf32>
      %cst_100 = arith.constant 0.000000e+00 : f32
      %cst_101 = arith.constant 1.000000e+00 : f32
      hivm.hir.vsub ins(%cst_100, %alloc_34 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_99 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_99 : memref<16xf32>) outs(%alloc_99 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_99, %cst_101 : memref<16xf32>, f32) outs(%alloc_99 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_101, %alloc_99 : f32, memref<16xf32>) outs(%alloc_34 : memref<16xf32, strided<[1]>>)
      %subview_102 = memref.subview %reinterpret_cast_13[0, %52, %56] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_102, %alloc_37 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xf32, strided<[1]>>) outs(%alloc_36 : memref<16xbf16, strided<[1]>>)
      %c0_i32_103 = arith.constant 0 : i32
      %57 = arith.sitofp %c0_i32_103 : i32 to bf16
      hivm.hir.vbrc ins(%57 : bf16) outs(%alloc_38 : memref<16xbf16, strided<[1]>>)
      %c1_i32_104 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c16_i32_96 step %c1_i32_104  : i32 {
        %c16_i32_159 = arith.constant 16 : i32
        %67 = arith.muli %arg28, %c16_i32_159 : i32
        %68 = arith.addi %67, %arg29 : i32
        %c31_i32 = arith.constant 31 : i32
        %69 = arith.cmpi slt, %68, %c31_i32 : i32
        scf.if %69 {
          %70 = arith.index_cast %arg29 : i32 to index
          %71 = memref.load %alloc_36[%70] : memref<16xbf16, strided<[1]>>
          %72 = arith.extf %71 : bf16 to f32
          %73 = memref.load %alloc_34[%70] : memref<16xf32, strided<[1]>>
          %74 = arith.mulf %72, %73 : f32
          %75 = arith.truncf %74 : f32 to bf16
          memref.store %75, %alloc_38[%70] : memref<16xbf16, strided<[1]>>
        }
      }
      memref.copy %alloc_38, %alloc_39 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %58 = arith.index_cast %54 : i32 to index
      %subview_105 = memref.subview %reinterpret_cast_8[0, %52, %58] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_105, %alloc_41 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_41 : memref<16xbf16, strided<[1]>>) outs(%alloc_40 : memref<16xf32, strided<[1]>>)
      %alloc_106 = memref.alloc() : memref<16xf32>
      %cst_107 = arith.constant 0.000000e+00 : f32
      %cst_108 = arith.constant 1.000000e+00 : f32
      hivm.hir.vsub ins(%cst_107, %alloc_40 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_106 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_106 : memref<16xf32>) outs(%alloc_106 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_106, %cst_108 : memref<16xf32>, f32) outs(%alloc_106 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_108, %alloc_106 : f32, memref<16xf32>) outs(%alloc_40 : memref<16xf32, strided<[1]>>)
      %subview_109 = memref.subview %reinterpret_cast_13[0, %52, %58] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_109, %alloc_43 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_43 : memref<16xf32, strided<[1]>>) outs(%alloc_42 : memref<16xbf16, strided<[1]>>)
      %c1_i32_110 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c16_i32_96 step %c1_i32_110  : i32 {
        %67 = arith.index_cast %arg29 : i32 to index
        %68 = memref.load %alloc_42[%67] : memref<16xbf16, strided<[1]>>
        %69 = arith.extf %68 : bf16 to f32
        %70 = memref.load %alloc_40[%67] : memref<16xf32, strided<[1]>>
        %71 = arith.mulf %69, %70 : f32
        memref.store %71, %alloc_44[%67] : memref<16xf32, strided<[1]>>
      }
      %c1_i32_111 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c16_i32_96 step %c1_i32_111  : i32 {
        %67 = arith.index_cast %arg29 : i32 to index
        %68 = memref.load %alloc_44[%67] : memref<16xf32, strided<[1]>>
        %69 = memref.load %alloc_39[%67] : memref<16xbf16, strided<[1]>>
        %70 = arith.extf %69 : bf16 to f32
        %71 = arith.addf %68, %70 : f32
        %72 = arith.truncf %71 : f32 to bf16
        memref.store %72, %alloc_45[%67] : memref<16xbf16, strided<[1]>>
      }
      memref.copy %alloc_45, %alloc_46 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %c1_i32_112 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c16_i32_96 step %c1_i32_112  : i32 {
        %c16_i32_159 = arith.constant 16 : i32
        %67 = arith.muli %arg28, %c16_i32_159 : i32
        %68 = arith.addi %67, %arg29 : i32
        %69 = arith.index_cast %68 : i32 to index
        %70 = arith.index_cast %42 : i32 to index
        %subview_160 = memref.subview %reinterpret_cast_0[0, %69, %70, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %71 = arith.index_cast %arg29 : i32 to index
        %subview_161 = memref.subview %alloc_30[%71, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<64xbf16, strided<[1], offset: ?>>
        memref.copy %subview_160, %subview_161 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>>
      }
      memref.copy %alloc_30, %alloc_74 : memref<16x64xbf16, strided<[64, 1]>> to memref<16x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_113 = memref.reinterpret_cast %alloc_74 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<16x1x64xbf16, strided<[64, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_113 : memref<16x1x64xbf16, strided<[64, 64, 1]>>) outs(%alloc_76 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [1]
      memref.copy %alloc_22, %alloc_77 : memref<2x64xbf16, strided<[64, 1]>> to memref<2x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_114 = memref.reinterpret_cast %alloc_77 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>> to memref<1x2x64xbf16, strided<[128, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_114 : memref<1x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_79 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_76 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_86 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_79 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_87 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vmul ins(%alloc_86, %alloc_87 : memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_88 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_88 : memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_47 : memref<16x2x64xbf16, strided<[128, 64, 1]>>)
      %reinterpret_cast_115 = memref.reinterpret_cast %alloc_47 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %reinterpret_cast_115, %alloc_29 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_116 = memref.subview %reinterpret_cast[0, %58, 0, %52, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_116, %alloc_49 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc, %alloc_80 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_117 = memref.reinterpret_cast %alloc_80 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_117 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_82 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_49 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_89 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_82 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_90 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_89, %alloc_90 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_49 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_118 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_119 = memref.subview %reinterpret_cast_9[0, %58, 0, %52, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_119, %alloc_50 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc_18, %alloc_83 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_120 = memref.reinterpret_cast %alloc_83 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_120 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_85 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_50 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_85 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_93 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_92, %alloc_93 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_94 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_94 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_50 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_121 = memref.reinterpret_cast %alloc_50 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %true = arith.constant true
      %c32_i32_122 = arith.constant 32 : i32
      %59 = arith.index_cast %c32_i32_122 : i32 to index
      hivm.hir.mmadL1 {b_transpose} ins(%reinterpret_cast_118, %reinterpret_cast_121, %true, %59, %59, %59 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_51 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_123 = memref.subview %45[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_51, %subview_123 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      %c1_i32_124 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c16_i32_96 step %c1_i32_124  : i32 {
        %c0_i32_159 = arith.constant 0 : i32
        %c2_i32_160 = arith.constant 2 : i32
        %c1_i32_161 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_159 to %c2_i32_160 step %c1_i32_161  : i32 {
          %c0_i32_162 = arith.constant 0 : i32
          %c8_i32_163 = arith.constant 8 : i32
          %c1_i32_164 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_162 to %c8_i32_163 step %c1_i32_164  : i32 {
            %c2_i32_165 = arith.constant 2 : i32
            %67 = arith.muli %arg29, %c2_i32_165 : i32
            %68 = arith.addi %67, %arg30 : i32
            %69 = arith.index_cast %68 : i32 to index
            %70 = arith.index_cast %arg31 : i32 to index
            %71 = memref.load %reinterpret_cast_118[%69, %70] : memref<32x32xbf16, strided<[32, 1]>>
            %72 = arith.index_cast %arg29 : i32 to index
            %73 = arith.index_cast %arg30 : i32 to index
            memref.store %71, %alloc_52[%72, %73, %70] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %c1_i32_125 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c16_i32_96 step %c1_i32_125  : i32 {
        %c0_i32_159 = arith.constant 0 : i32
        %c2_i32_160 = arith.constant 2 : i32
        %c1_i32_161 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_159 to %c2_i32_160 step %c1_i32_161  : i32 {
          %c0_i32_162 = arith.constant 0 : i32
          %c8_i32_163 = arith.constant 8 : i32
          %c1_i32_164 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_162 to %c8_i32_163 step %c1_i32_164  : i32 {
            %c2_i32_165 = arith.constant 2 : i32
            %67 = arith.muli %arg29, %c2_i32_165 : i32
            %68 = arith.addi %67, %arg30 : i32
            %69 = arith.index_cast %68 : i32 to index
            %c16_i32_166 = arith.constant 16 : i32
            %70 = arith.addi %arg31, %c16_i32_166 : i32
            %71 = arith.index_cast %70 : i32 to index
            %72 = memref.load %reinterpret_cast_118[%69, %71] : memref<32x32xbf16, strided<[32, 1]>>
            %73 = arith.index_cast %arg29 : i32 to index
            %74 = arith.index_cast %arg30 : i32 to index
            %75 = arith.index_cast %arg31 : i32 to index
            memref.store %72, %alloc_53[%73, %74, %75] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %subview_126 = memref.subview %reinterpret_cast_3[0, %58, %52, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_126, %alloc_54 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>>
      %cst_127 = arith.constant 1.000000e+00 : f32
      %cst_128 = arith.constant -5.000000e-01 : f32
      %cst_129 = arith.constant 2.400000e+01 : f32
      %cst_130 = arith.constant 7.200000e+02 : f32
      %cst_131 = arith.constant -1.000000e+00 : f32
      %60 = arith.divf %cst_127, %cst_129 : f32
      %61 = arith.divf %cst_131, %cst_130 : f32
      %alloc_132 = memref.alloc() : memref<16x8xf32>
      %alloc_133 = memref.alloc() : memref<16x8xf32>
      %alloc_134 = memref.alloc() : memref<16x8xf32>
      %alloc_135 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_54, %alloc_54 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_132 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_132, %alloc_132 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_133 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_132, %alloc_133 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_134 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_132, %cst_128 : memref<16x8xf32>, f32) outs(%alloc_132 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_133, %60 : memref<16x8xf32>, f32) outs(%alloc_133 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_134, %61 : memref<16x8xf32>, f32) outs(%alloc_134 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_132, %cst_127 : memref<16x8xf32>, f32) outs(%alloc_135 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_133, %alloc_135 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_135 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_134, %alloc_135 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_55 : memref<16x8xf32, strided<[8, 1]>>)
      %cst_136 = arith.constant 1.000000e+00 : f32
      %cst_137 = arith.constant -1.000000e+00 : f32
      %cst_138 = arith.constant 6.000000e+00 : f32
      %cst_139 = arith.constant 1.200000e+02 : f32
      %cst_140 = arith.constant 5.040000e+03 : f32
      %62 = arith.divf %cst_137, %cst_138 : f32
      %63 = arith.divf %cst_136, %cst_139 : f32
      %64 = arith.divf %cst_137, %cst_140 : f32
      %alloc_141 = memref.alloc() : memref<16x8xf32>
      %alloc_142 = memref.alloc() : memref<16x8xf32>
      %alloc_143 = memref.alloc() : memref<16x8xf32>
      %alloc_144 = memref.alloc() : memref<16x8xf32>
      %alloc_145 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_54, %alloc_54 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_141 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_141, %alloc_54 : memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_142 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_142, %alloc_141 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_143 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_143, %alloc_141 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_144 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_142, %62 : memref<16x8xf32>, f32) outs(%alloc_142 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_143, %63 : memref<16x8xf32>, f32) outs(%alloc_143 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_144, %64 : memref<16x8xf32>, f32) outs(%alloc_144 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_54, %alloc_142 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) outs(%alloc_145 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_143, %alloc_145 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_145 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_144, %alloc_145 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_56 : memref<16x8xf32, strided<[8, 1]>>)
      %c1_i32_146 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c16_i32_96 step %c1_i32_146  : i32 {
        %c0_i32_159 = arith.constant 0 : i32
        %c2_i32_160 = arith.constant 2 : i32
        %c1_i32_161 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_159 to %c2_i32_160 step %c1_i32_161  : i32 {
          %c0_i32_162 = arith.constant 0 : i32
          %c8_i32_163 = arith.constant 8 : i32
          %c1_i32_164 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_162 to %c8_i32_163 step %c1_i32_164  : i32 {
            %67 = arith.index_cast %arg29 : i32 to index
            %68 = arith.index_cast %arg31 : i32 to index
            %69 = memref.load %alloc_55[%67, %68] : memref<16x8xf32, strided<[8, 1]>>
            %70 = arith.index_cast %arg30 : i32 to index
            %71 = memref.load %alloc_52[%67, %70, %68] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %72 = arith.extf %71 : bf16 to f32
            %73 = arith.mulf %69, %72 : f32
            %74 = memref.load %alloc_56[%67, %68] : memref<16x8xf32, strided<[8, 1]>>
            %75 = memref.load %alloc_53[%67, %70, %68] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %76 = arith.extf %75 : bf16 to f32
            %77 = arith.subf %73, %73 : f32
            %78 = arith.truncf %77 : f32 to bf16
            %c2_i32_165 = arith.constant 2 : i32
            %79 = arith.muli %arg29, %c2_i32_165 : i32
            %80 = arith.addi %79, %arg30 : i32
            %81 = arith.index_cast %80 : i32 to index
            memref.store %78, %reinterpret_cast_118[%81, %68] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      %c1_i32_147 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c16_i32_96 step %c1_i32_147  : i32 {
        %c0_i32_159 = arith.constant 0 : i32
        %c2_i32_160 = arith.constant 2 : i32
        %c1_i32_161 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_159 to %c2_i32_160 step %c1_i32_161  : i32 {
          %c0_i32_162 = arith.constant 0 : i32
          %c8_i32_163 = arith.constant 8 : i32
          %c1_i32_164 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_162 to %c8_i32_163 step %c1_i32_164  : i32 {
            %67 = arith.index_cast %arg29 : i32 to index
            %68 = arith.index_cast %arg31 : i32 to index
            %69 = memref.load %alloc_56[%67, %68] : memref<16x8xf32, strided<[8, 1]>>
            %70 = arith.index_cast %arg30 : i32 to index
            %71 = memref.load %alloc_52[%67, %70, %68] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %72 = arith.extf %71 : bf16 to f32
            %73 = arith.mulf %69, %72 : f32
            %74 = memref.load %alloc_55[%67, %68] : memref<16x8xf32, strided<[8, 1]>>
            %75 = memref.load %alloc_53[%67, %70, %68] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %76 = arith.extf %75 : bf16 to f32
            %77 = arith.addf %73, %73 : f32
            %78 = arith.truncf %77 : f32 to bf16
            %c2_i32_165 = arith.constant 2 : i32
            %79 = arith.muli %arg29, %c2_i32_165 : i32
            %80 = arith.addi %79, %arg30 : i32
            %81 = arith.index_cast %80 : i32 to index
            %c16_i32_166 = arith.constant 16 : i32
            %82 = arith.addi %arg31, %c16_i32_166 : i32
            %83 = arith.index_cast %82 : i32 to index
            memref.store %78, %reinterpret_cast_118[%81, %83] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      hivm.hir.vcast ins(%alloc_21 : memref<32x64xf32, strided<[64, 1]>>) outs(%alloc_58 : memref<32x64xbf16, strided<[64, 1]>>)
      memref.copy %alloc_58, %alloc_31 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %c64_i32_148 = arith.constant 64 : i32
      %65 = arith.index_cast %c64_i32_148 : i32 to index
      hivm.hir.mmadL1 ins(%reinterpret_cast_118, %alloc_31, %true, %59, %59, %65 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_57 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_149 = memref.subview %46[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_57, %subview_149 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      %c1_i32_150 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c16_i32_96 step %c1_i32_150  : i32 {
        %c0_i32_159 = arith.constant 0 : i32
        %c2_i32_160 = arith.constant 2 : i32
        %c1_i32_161 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_159 to %c2_i32_160 step %c1_i32_161  : i32 {
          %c0_i32_162 = arith.constant 0 : i32
          %c8_i32_163 = arith.constant 8 : i32
          %c1_i32_164 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_162 to %c8_i32_163 step %c1_i32_164  : i32 {
            %c2_i32_165 = arith.constant 2 : i32
            %67 = arith.muli %arg29, %c2_i32_165 : i32
            %68 = arith.addi %67, %arg30 : i32
            %69 = arith.index_cast %68 : i32 to index
            %70 = arith.index_cast %arg31 : i32 to index
            %71 = memref.load %reinterpret_cast_121[%69, %70] : memref<32x32xbf16, strided<[32, 1]>>
            %72 = arith.index_cast %arg29 : i32 to index
            %73 = arith.index_cast %arg30 : i32 to index
            memref.store %71, %alloc_59[%72, %73, %70] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %c1_i32_151 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c16_i32_96 step %c1_i32_151  : i32 {
        %c0_i32_159 = arith.constant 0 : i32
        %c2_i32_160 = arith.constant 2 : i32
        %c1_i32_161 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_159 to %c2_i32_160 step %c1_i32_161  : i32 {
          %c0_i32_162 = arith.constant 0 : i32
          %c8_i32_163 = arith.constant 8 : i32
          %c1_i32_164 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_162 to %c8_i32_163 step %c1_i32_164  : i32 {
            %c2_i32_165 = arith.constant 2 : i32
            %67 = arith.muli %arg29, %c2_i32_165 : i32
            %68 = arith.addi %67, %arg30 : i32
            %69 = arith.index_cast %68 : i32 to index
            %c16_i32_166 = arith.constant 16 : i32
            %70 = arith.addi %arg31, %c16_i32_166 : i32
            %71 = arith.index_cast %70 : i32 to index
            %72 = memref.load %reinterpret_cast_121[%69, %71] : memref<32x32xbf16, strided<[32, 1]>>
            %73 = arith.index_cast %arg29 : i32 to index
            %74 = arith.index_cast %arg30 : i32 to index
            %75 = arith.index_cast %arg31 : i32 to index
            memref.store %72, %alloc_60[%73, %74, %75] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %c1_i32_152 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c16_i32_96 step %c1_i32_152  : i32 {
        %c0_i32_159 = arith.constant 0 : i32
        %c2_i32_160 = arith.constant 2 : i32
        %c1_i32_161 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_159 to %c2_i32_160 step %c1_i32_161  : i32 {
          %c0_i32_162 = arith.constant 0 : i32
          %c8_i32_163 = arith.constant 8 : i32
          %c1_i32_164 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_162 to %c8_i32_163 step %c1_i32_164  : i32 {
            %67 = arith.index_cast %arg29 : i32 to index
            %68 = arith.index_cast %arg31 : i32 to index
            %69 = memref.load %alloc_55[%67, %68] : memref<16x8xf32, strided<[8, 1]>>
            %70 = arith.index_cast %arg30 : i32 to index
            %71 = memref.load %alloc_59[%67, %70, %68] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %72 = arith.extf %71 : bf16 to f32
            %73 = arith.mulf %69, %72 : f32
            %74 = memref.load %alloc_56[%67, %68] : memref<16x8xf32, strided<[8, 1]>>
            %75 = memref.load %alloc_60[%67, %70, %68] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %76 = arith.extf %75 : bf16 to f32
            %77 = arith.subf %73, %73 : f32
            %78 = arith.truncf %77 : f32 to bf16
            %c2_i32_165 = arith.constant 2 : i32
            %79 = arith.muli %arg29, %c2_i32_165 : i32
            %80 = arith.addi %79, %arg30 : i32
            %81 = arith.index_cast %80 : i32 to index
            memref.store %78, %reinterpret_cast_121[%81, %68] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      %c1_i32_153 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c16_i32_96 step %c1_i32_153  : i32 {
        %c0_i32_159 = arith.constant 0 : i32
        %c2_i32_160 = arith.constant 2 : i32
        %c1_i32_161 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_159 to %c2_i32_160 step %c1_i32_161  : i32 {
          %c0_i32_162 = arith.constant 0 : i32
          %c8_i32_163 = arith.constant 8 : i32
          %c1_i32_164 = arith.constant 1 : i32
          scf.for %arg31 = %c0_i32_162 to %c8_i32_163 step %c1_i32_164  : i32 {
            %67 = arith.index_cast %arg29 : i32 to index
            %68 = arith.index_cast %arg31 : i32 to index
            %69 = memref.load %alloc_56[%67, %68] : memref<16x8xf32, strided<[8, 1]>>
            %70 = arith.index_cast %arg30 : i32 to index
            %71 = memref.load %alloc_59[%67, %70, %68] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %72 = arith.extf %71 : bf16 to f32
            %73 = arith.mulf %69, %72 : f32
            %74 = memref.load %alloc_55[%67, %68] : memref<16x8xf32, strided<[8, 1]>>
            %75 = memref.load %alloc_60[%67, %70, %68] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %76 = arith.extf %75 : bf16 to f32
            %77 = arith.addf %73, %73 : f32
            %78 = arith.truncf %77 : f32 to bf16
            %c2_i32_165 = arith.constant 2 : i32
            %79 = arith.muli %arg29, %c2_i32_165 : i32
            %80 = arith.addi %79, %arg30 : i32
            %81 = arith.index_cast %80 : i32 to index
            %c16_i32_166 = arith.constant 16 : i32
            %82 = arith.addi %arg31, %c16_i32_166 : i32
            %83 = arith.index_cast %82 : i32 to index
            memref.store %78, %reinterpret_cast_121[%81, %83] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      memref.copy %reinterpret_cast_121, %alloc_61 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_61 : memref<32x32xbf16, strided<[32, 1]>>) outs(%alloc_62 : memref<32x32xf32, strided<[32, 1]>>)
      %c1_i32_154 = arith.constant 1 : i32
      scf.for %arg29 = %c0_i32_103 to %c32_i32_122 step %c1_i32_154  : i32 {
        %c0_i32_159 = arith.constant 0 : i32
        %c32_i32_160 = arith.constant 32 : i32
        %c1_i32_161 = arith.constant 1 : i32
        scf.for %arg30 = %c0_i32_159 to %c32_i32_160 step %c1_i32_161  : i32 {
          %67 = arith.index_cast %arg29 : i32 to index
          %68 = arith.index_cast %arg30 : i32 to index
          %69 = memref.load %alloc_62[%67, %68] : memref<32x32xf32, strided<[32, 1]>>
          %c2_i32_162 = arith.constant 2 : i32
          %70 = arith.divsi %arg29, %c2_i32_162 : i32
          %71 = arith.index_cast %70 : i32 to index
          %72 = memref.load %alloc_46[%71] : memref<16xbf16, strided<[1]>>
          %73 = arith.extf %72 : bf16 to f32
          %74 = arith.mulf %69, %73 : f32
          memref.store %74, %alloc_62[%67, %68] : memref<32x32xf32, strided<[32, 1]>>
        }
      }
      hivm.hir.vcast ins(%alloc_62 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc_61 : memref<32x32xbf16, strided<[32, 1]>>)
      memref.copy %alloc_61, %reinterpret_cast_121 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%reinterpret_cast_118, %reinterpret_cast_121, %true, %59, %59, %59 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_63 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_155 = memref.subview %47[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_63, %subview_155 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      hivm.hir.vexp ins(%alloc_33 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_33 : memref<16x16xf32, strided<[16, 1]>>)
      %c2_i32_156 = arith.constant 2 : i32
      %66 = arith.cmpi slt, %44, %c2_i32_156 : i32
      scf.if %66 {
        %c16_i32_159 = arith.constant 16 : i32
        %67 = arith.muli %44, %c16_i32_159 : i32
        %68 = arith.index_cast %67 : i32 to index
        %subview_160 = memref.subview %47[0, %68, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_160, %alloc_64 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_161 = memref.subview %46[0, %68, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_161, %alloc_66 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %69 = arith.index_cast %42 : i32 to index
        %70 = arith.muli %arg28, %c16_i32_159 : i32
        %c8_i32_162 = arith.constant 8 : i32
        %71 = arith.muli %44, %c8_i32_162 : i32
        %72 = arith.addi %70, %71 : i32
        %73 = arith.index_cast %72 : i32 to index
        %subview_163 = memref.subview %reinterpret_cast_4[0, %69, %73] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_163, %alloc_67 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>>
        hivm.hir.vexp ins(%alloc_67 : memref<8xf32, strided<[1]>>) outs(%alloc_67 : memref<8xf32, strided<[1]>>)
        %c0_i32_164 = arith.constant 0 : i32
        %c1_i32_165 = arith.constant 1 : i32
        scf.for %arg29 = %c0_i32_164 to %c16_i32_159 step %c1_i32_165  : i32 {
          %c0_i32_168 = arith.constant 0 : i32
          %c32_i32_169 = arith.constant 32 : i32
          %c1_i32_170 = arith.constant 1 : i32
          scf.for %arg30 = %c0_i32_168 to %c32_i32_169 step %c1_i32_170  : i32 {
            %c2_i32_173 = arith.constant 2 : i32
            %74 = arith.divsi %arg30, %c2_i32_173 : i32
            %c8_i32_174 = arith.constant 8 : i32
            %75 = arith.muli %44, %c8_i32_174 : i32
            %76 = arith.divsi %arg29, %c2_i32_173 : i32
            %77 = arith.addi %75, %76 : i32
            %78 = arith.cmpi slt, %74, %77 : i32
            scf.if %78 {
              %79 = arith.index_cast %arg29 : i32 to index
              %80 = arith.index_cast %arg30 : i32 to index
              %81 = memref.load %alloc_64[%79, %80] : memref<16x32xf32, strided<[32, 1]>>
              %c8_i32_175 = arith.constant 8 : i32
              %82 = arith.muli %44, %c8_i32_175 : i32
              %c2_i32_176 = arith.constant 2 : i32
              %83 = arith.divsi %arg29, %c2_i32_176 : i32
              %84 = arith.addi %82, %83 : i32
              %85 = arith.index_cast %84 : i32 to index
              %86 = arith.divsi %arg30, %c2_i32_176 : i32
              %87 = arith.index_cast %86 : i32 to index
              %88 = memref.load %alloc_33[%85, %87] : memref<16x16xf32, strided<[16, 1]>>
              %89 = arith.mulf %81, %88 : f32
              %90 = arith.truncf %89 : f32 to bf16
              memref.store %90, %alloc_65[%79, %80] : memref<16x32xbf16, strided<[32, 1]>>
            } else {
              %cst_175 = arith.constant 0.000000e+00 : bf16
              %79 = arith.index_cast %arg29 : i32 to index
              %80 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_175, %alloc_65[%79, %80] : memref<16x32xbf16, strided<[32, 1]>>
            }
          }
          %c64_i32_171 = arith.constant 64 : i32
          %c1_i32_172 = arith.constant 1 : i32
          scf.for %arg30 = %c0_i32_168 to %c64_i32_171 step %c1_i32_172  : i32 {
            %74 = arith.index_cast %arg29 : i32 to index
            %75 = arith.index_cast %arg30 : i32 to index
            %76 = memref.load %alloc_66[%74, %75] : memref<16x64xf32, strided<[64, 1]>>
            %c2_i32_173 = arith.constant 2 : i32
            %77 = arith.divsi %arg29, %c2_i32_173 : i32
            %78 = arith.index_cast %77 : i32 to index
            %79 = memref.load %alloc_67[%78] : memref<8xf32, strided<[1]>>
            %80 = arith.mulf %76, %79 : f32
            memref.store %80, %alloc_66[%74, %75] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %subview_166 = memref.subview %48[0, %68, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_65, %subview_166 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        %subview_167 = memref.subview %46[0, %68, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %alloc_66, %subview_167 : memref<16x64xf32, strided<[64, 1]>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      }
      %subview_157 = memref.subview %48[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_157, %alloc_32 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 ins(%alloc_32, %alloc_29, %true, %59, %59, %65 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_68 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_158 = memref.subview %49[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_68, %subview_158 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.if %66 {
        %c16_i32_159 = arith.constant 16 : i32
        %67 = arith.muli %44, %c16_i32_159 : i32
        %68 = arith.index_cast %67 : i32 to index
        %subview_160 = memref.subview %46[0, %68, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_160, %alloc_69 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_161 = memref.subview %49[0, %68, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_161, %alloc_70 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_162 = memref.subview %45[0, %68, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_162, %alloc_71 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %69 = arith.muli %arg28, %c16_i32_159 : i32
        %c8_i32_163 = arith.constant 8 : i32
        %70 = arith.muli %44, %c8_i32_163 : i32
        %71 = arith.addi %69, %70 : i32
        %72 = arith.index_cast %71 : i32 to index
        %73 = arith.index_cast %42 : i32 to index
        %subview_164 = memref.subview %reinterpret_cast_5[0, %72, %73, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_164, %alloc_72 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>>
        %c0_i32_165 = arith.constant 0 : i32
        %c1_i32_166 = arith.constant 1 : i32
        scf.for %arg29 = %c0_i32_165 to %c16_i32_159 step %c1_i32_166  : i32 {
          %c0_i32_169 = arith.constant 0 : i32
          %c64_i32_170 = arith.constant 64 : i32
          %c1_i32_171 = arith.constant 1 : i32
          scf.for %arg30 = %c0_i32_169 to %c64_i32_170 step %c1_i32_171  : i32 {
            %74 = arith.index_cast %arg29 : i32 to index
            %75 = arith.index_cast %arg30 : i32 to index
            %76 = memref.load %alloc_69[%74, %75] : memref<16x64xf32, strided<[64, 1]>>
            %77 = memref.load %alloc_70[%74, %75] : memref<16x64xf32, strided<[64, 1]>>
            %78 = arith.addf %76, %77 : f32
            memref.store %78, %alloc_69[%74, %75] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %c1_i32_167 = arith.constant 1 : i32
        scf.for %arg29 = %c0_i32_165 to %c8_i32_163 step %c1_i32_167  : i32 {
          %c0_i32_169 = arith.constant 0 : i32
          %c64_i32_170 = arith.constant 64 : i32
          %c1_i32_171 = arith.constant 1 : i32
          scf.for %arg30 = %c0_i32_169 to %c64_i32_170 step %c1_i32_171  : i32 {
            %cst_172 = arith.constant 0.000000e+00 : bf16
            %74 = arith.index_cast %arg29 : i32 to index
            %75 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_172, %alloc_73[%74, %75] : memref<8x64xbf16, strided<[64, 1]>>
          }
        }
        %subview_168 = memref.subview %reinterpret_cast_10[0, %72, %73, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_73, %subview_168 : memref<8x64xbf16, strided<[64, 1]>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
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
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_21 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_22 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_23 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_24 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
    %alloc_25 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
    %alloc_26 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vbrc ins(%cst_7 : f32) outs(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>)
    %9 = arith.index_cast %1 : i32 to index
    %subview = memref.subview %reinterpret_cast_10[%9, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_26 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vcast ins(%alloc_26 : memref<2x64xf32, strided<[64, 1]>>) outs(%alloc_25 : memref<2x64xbf16, strided<[64, 1]>>)
    %subview_27 = memref.subview %reinterpret_cast_11[%9, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_27, %alloc_22 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_22 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>>)
    %subview_28 = memref.subview %reinterpret_cast_15[%9, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_28, %alloc_23 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc_21 : memref<2x32xbf16, strided<[32, 1]>>)
    scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
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
      %alloc_48 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_49 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_50 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_51 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_52 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_53 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_54 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_55 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_56 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_57 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_58 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_59 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_60 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_61 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_62 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
      %alloc_63 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_64 = memref.alloc() : memref<8xf32, strided<[1]>>
      %alloc_65 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_66 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_67 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_68 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_69 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_70 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_71 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_72 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_73 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
      %alloc_74 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_75 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_76 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_77 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_78 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_79 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_80 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_81 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_82 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_83 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_84 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_85 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_86 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_87 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %10 = arith.index_cast %1 : i32 to index
      %11 = arith.index_cast %arg28 : i32 to index
      %subview_88 = memref.subview %reinterpret_cast_20[0, %10, %11, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_88, %alloc_33 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>>
      %12 = arith.muli %arg28, %c16_i32 : i32
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.index_cast %13 : i32 to index
      %subview_89 = memref.subview %reinterpret_cast_16[0, %10, %14] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_89, %alloc_35 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_35 : memref<16xbf16, strided<[1]>>) outs(%alloc_34 : memref<16xf32, strided<[1]>>)
      hivm.hir.vmul ins(%alloc_34, %cst_8 : memref<16xf32, strided<[1]>>, f32) outs(%alloc_34 : memref<16xf32, strided<[1]>>)
      %alloc_90 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_34 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_90 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_90 : memref<16xf32>) outs(%alloc_90 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_90, %cst_6 : memref<16xf32>, f32) outs(%alloc_90 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_90 : f32, memref<16xf32>) outs(%alloc_34 : memref<16xf32, strided<[1]>>)
      %subview_91 = memref.subview %reinterpret_cast_19[0, %10, %14] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_91, %alloc_37 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xf32, strided<[1]>>) outs(%alloc_36 : memref<16xbf16, strided<[1]>>)
      hivm.hir.vbrc ins(%cst_4 : bf16) outs(%alloc_38 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %17 = arith.muli %arg28, %c16_i32 : i32
        %18 = arith.addi %17, %arg29 : i32
        %19 = arith.cmpi slt, %18, %c31_i32 : i32
        scf.if %19 {
          %20 = arith.index_cast %arg29 : i32 to index
          %21 = memref.load %alloc_36[%20] : memref<16xbf16, strided<[1]>>
          %22 = arith.extf %21 : bf16 to f32
          %23 = memref.load %alloc_34[%20] : memref<16xf32, strided<[1]>>
          %24 = arith.mulf %22, %23 : f32
          %25 = arith.truncf %24 : f32 to bf16
          memref.store %25, %alloc_38[%20] : memref<16xbf16, strided<[1]>>
        }
      }
      memref.copy %alloc_38, %alloc_39 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %15 = arith.index_cast %12 : i32 to index
      %subview_92 = memref.subview %reinterpret_cast_16[0, %10, %15] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_92, %alloc_41 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_41 : memref<16xbf16, strided<[1]>>) outs(%alloc_40 : memref<16xf32, strided<[1]>>)
      %alloc_93 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_40 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_93 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_93 : memref<16xf32>) outs(%alloc_93 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_93, %cst_6 : memref<16xf32>, f32) outs(%alloc_93 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_93 : f32, memref<16xf32>) outs(%alloc_40 : memref<16xf32, strided<[1]>>)
      %subview_94 = memref.subview %reinterpret_cast_19[0, %10, %15] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_94, %alloc_43 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_43 : memref<16xf32, strided<[1]>>) outs(%alloc_42 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %17 = arith.index_cast %arg29 : i32 to index
        %18 = memref.load %alloc_42[%17] : memref<16xbf16, strided<[1]>>
        %19 = arith.extf %18 : bf16 to f32
        %20 = memref.load %alloc_40[%17] : memref<16xf32, strided<[1]>>
        %21 = arith.mulf %19, %20 : f32
        memref.store %21, %alloc_44[%17] : memref<16xf32, strided<[1]>>
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %17 = arith.index_cast %arg29 : i32 to index
        %18 = memref.load %alloc_44[%17] : memref<16xf32, strided<[1]>>
        %19 = memref.load %alloc_39[%17] : memref<16xbf16, strided<[1]>>
        %20 = arith.extf %19 : bf16 to f32
        %21 = arith.addf %18, %20 : f32
        %22 = arith.truncf %21 : f32 to bf16
        memref.store %22, %alloc_45[%17] : memref<16xbf16, strided<[1]>>
      }
      memref.copy %alloc_45, %alloc_46 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %17 = arith.muli %arg28, %c16_i32 : i32
        %18 = arith.addi %17, %arg29 : i32
        %19 = arith.index_cast %18 : i32 to index
        %20 = arith.index_cast %1 : i32 to index
        %subview_119 = memref.subview %reinterpret_cast_9[0, %19, %20, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %21 = arith.index_cast %arg29 : i32 to index
        %subview_120 = memref.subview %alloc_30[%21, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<64xbf16, strided<[1], offset: ?>>
        memref.copy %subview_119, %subview_120 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>>
      }
      memref.copy %alloc_30, %alloc_71 : memref<16x64xbf16, strided<[64, 1]>> to memref<16x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_95 = memref.reinterpret_cast %alloc_71 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<16x1x64xbf16, strided<[64, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_95 : memref<16x1x64xbf16, strided<[64, 64, 1]>>) outs(%alloc_72 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [1]
      memref.copy %alloc_25, %alloc_73 : memref<2x64xbf16, strided<[64, 1]>> to memref<2x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_96 = memref.reinterpret_cast %alloc_73 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>> to memref<1x2x64xbf16, strided<[128, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_96 : memref<1x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_74 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_72 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_79 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_74 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_80 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vmul ins(%alloc_79, %alloc_80 : memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_81 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_81 : memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_47 : memref<16x2x64xbf16, strided<[128, 64, 1]>>)
      %reinterpret_cast_97 = memref.reinterpret_cast %alloc_47 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %reinterpret_cast_97, %alloc_29 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_98 = memref.subview %reinterpret_cast[0, %15, 0, %10, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_98, %alloc_48 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc, %alloc_75 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_99 = memref.reinterpret_cast %alloc_75 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_99 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_76 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_48 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_82 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_76 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_83 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_82, %alloc_83 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_84 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_84 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_48 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_100 = memref.reinterpret_cast %alloc_48 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_101 = memref.subview %reinterpret_cast_17[0, %15, 0, %10, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_101, %alloc_49 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc_21, %alloc_77 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_102 = memref.reinterpret_cast %alloc_77 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_102 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_78 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_49 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_85 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_78 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_86 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_85, %alloc_86 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_87 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_87 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_49 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_103 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%reinterpret_cast_100, %reinterpret_cast_103, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_50 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_104 = memref.subview %4[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_50, %subview_104 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.muli %arg29, %c2_i32 : i32
            %18 = arith.addi %17, %arg30 : i32
            %19 = arith.index_cast %18 : i32 to index
            %20 = arith.index_cast %arg31 : i32 to index
            %21 = memref.load %reinterpret_cast_100[%19, %20] : memref<32x32xbf16, strided<[32, 1]>>
            %22 = arith.index_cast %arg29 : i32 to index
            %23 = arith.index_cast %arg30 : i32 to index
            memref.store %21, %alloc_51[%22, %23, %20] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %subview_105 = memref.subview %reinterpret_cast_12[0, %15, %10, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_105, %alloc_52 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>>
      %alloc_106 = memref.alloc() : memref<16x8xf32>
      %alloc_107 = memref.alloc() : memref<16x8xf32>
      %alloc_108 = memref.alloc() : memref<16x8xf32>
      %alloc_109 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_52, %alloc_52 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_106 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_106, %alloc_106 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_107 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_106, %alloc_107 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_108 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_106, %cst_5 : memref<16x8xf32>, f32) outs(%alloc_106 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_107, %cst_3 : memref<16x8xf32>, f32) outs(%alloc_107 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_108, %cst_2 : memref<16x8xf32>, f32) outs(%alloc_108 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_106, %cst_6 : memref<16x8xf32>, f32) outs(%alloc_109 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_107, %alloc_109 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_109 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_108, %alloc_109 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_53 : memref<16x8xf32, strided<[8, 1]>>)
      %alloc_110 = memref.alloc() : memref<16x8xf32>
      %alloc_111 = memref.alloc() : memref<16x8xf32>
      %alloc_112 = memref.alloc() : memref<16x8xf32>
      %alloc_113 = memref.alloc() : memref<16x8xf32>
      %alloc_114 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_52, %alloc_52 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_110 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_110, %alloc_52 : memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_111 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_111, %alloc_110 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_112 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_112, %alloc_110 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_113 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_111, %cst_1 : memref<16x8xf32>, f32) outs(%alloc_111 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_112, %cst_0 : memref<16x8xf32>, f32) outs(%alloc_112 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_113, %cst : memref<16x8xf32>, f32) outs(%alloc_113 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_52, %alloc_111 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) outs(%alloc_114 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_112, %alloc_114 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_114 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_113, %alloc_114 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_54 : memref<16x8xf32, strided<[8, 1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.index_cast %arg29 : i32 to index
            %18 = arith.index_cast %arg31 : i32 to index
            %19 = memref.load %alloc_53[%17, %18] : memref<16x8xf32, strided<[8, 1]>>
            %20 = arith.index_cast %arg30 : i32 to index
            %21 = memref.load %alloc_51[%17, %20, %18] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %22 = arith.extf %21 : bf16 to f32
            %23 = arith.mulf %19, %22 : f32
            %24 = arith.subf %23, %23 : f32
            %25 = arith.truncf %24 : f32 to bf16
            %26 = arith.muli %arg29, %c2_i32 : i32
            %27 = arith.addi %26, %arg30 : i32
            %28 = arith.index_cast %27 : i32 to index
            memref.store %25, %reinterpret_cast_100[%28, %18] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.index_cast %arg29 : i32 to index
            %18 = arith.index_cast %arg31 : i32 to index
            %19 = memref.load %alloc_54[%17, %18] : memref<16x8xf32, strided<[8, 1]>>
            %20 = arith.index_cast %arg30 : i32 to index
            %21 = memref.load %alloc_51[%17, %20, %18] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %22 = arith.extf %21 : bf16 to f32
            %23 = arith.mulf %19, %22 : f32
            %24 = arith.addf %23, %23 : f32
            %25 = arith.truncf %24 : f32 to bf16
            %26 = arith.muli %arg29, %c2_i32 : i32
            %27 = arith.addi %26, %arg30 : i32
            %28 = arith.index_cast %27 : i32 to index
            %29 = arith.addi %arg31, %c16_i32 : i32
            %30 = arith.index_cast %29 : i32 to index
            memref.store %25, %reinterpret_cast_100[%28, %30] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      hivm.hir.vcast ins(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>) outs(%alloc_56 : memref<32x64xbf16, strided<[64, 1]>>)
      memref.copy %alloc_56, %alloc_31 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%reinterpret_cast_100, %alloc_31, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_55 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_115 = memref.subview %5[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_55, %subview_115 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.muli %arg29, %c2_i32 : i32
            %18 = arith.addi %17, %arg30 : i32
            %19 = arith.index_cast %18 : i32 to index
            %20 = arith.index_cast %arg31 : i32 to index
            %21 = memref.load %reinterpret_cast_103[%19, %20] : memref<32x32xbf16, strided<[32, 1]>>
            %22 = arith.index_cast %arg29 : i32 to index
            %23 = arith.index_cast %arg30 : i32 to index
            memref.store %21, %alloc_57[%22, %23, %20] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.index_cast %arg29 : i32 to index
            %18 = arith.index_cast %arg31 : i32 to index
            %19 = memref.load %alloc_53[%17, %18] : memref<16x8xf32, strided<[8, 1]>>
            %20 = arith.index_cast %arg30 : i32 to index
            %21 = memref.load %alloc_57[%17, %20, %18] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %22 = arith.extf %21 : bf16 to f32
            %23 = arith.mulf %19, %22 : f32
            %24 = arith.subf %23, %23 : f32
            %25 = arith.truncf %24 : f32 to bf16
            %26 = arith.muli %arg29, %c2_i32 : i32
            %27 = arith.addi %26, %arg30 : i32
            %28 = arith.index_cast %27 : i32 to index
            memref.store %25, %reinterpret_cast_103[%28, %18] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.index_cast %arg29 : i32 to index
            %18 = arith.index_cast %arg31 : i32 to index
            %19 = memref.load %alloc_54[%17, %18] : memref<16x8xf32, strided<[8, 1]>>
            %20 = arith.index_cast %arg30 : i32 to index
            %21 = memref.load %alloc_57[%17, %20, %18] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %22 = arith.extf %21 : bf16 to f32
            %23 = arith.mulf %19, %22 : f32
            %24 = arith.addf %23, %23 : f32
            %25 = arith.truncf %24 : f32 to bf16
            %26 = arith.muli %arg29, %c2_i32 : i32
            %27 = arith.addi %26, %arg30 : i32
            %28 = arith.index_cast %27 : i32 to index
            %29 = arith.addi %arg31, %c16_i32 : i32
            %30 = arith.index_cast %29 : i32 to index
            memref.store %25, %reinterpret_cast_103[%28, %30] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      memref.copy %reinterpret_cast_103, %alloc_58 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_58 : memref<32x32xbf16, strided<[32, 1]>>) outs(%alloc_59 : memref<32x32xf32, strided<[32, 1]>>)
      scf.for %arg29 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
          %17 = arith.index_cast %arg29 : i32 to index
          %18 = arith.index_cast %arg30 : i32 to index
          %19 = memref.load %alloc_59[%17, %18] : memref<32x32xf32, strided<[32, 1]>>
          %20 = arith.divsi %arg29, %c2_i32 : i32
          %21 = arith.index_cast %20 : i32 to index
          %22 = memref.load %alloc_46[%21] : memref<16xbf16, strided<[1]>>
          %23 = arith.extf %22 : bf16 to f32
          %24 = arith.mulf %19, %23 : f32
          memref.store %24, %alloc_59[%17, %18] : memref<32x32xf32, strided<[32, 1]>>
        }
      }
      hivm.hir.vcast ins(%alloc_59 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc_58 : memref<32x32xbf16, strided<[32, 1]>>)
      memref.copy %alloc_58, %reinterpret_cast_103 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%reinterpret_cast_100, %reinterpret_cast_103, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_60 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_116 = memref.subview %6[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_60, %subview_116 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      hivm.hir.vexp ins(%alloc_33 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_33 : memref<16x16xf32, strided<[16, 1]>>)
      %16 = arith.cmpi slt, %3, %c2_i32 : i32
      scf.if %16 {
        %17 = arith.muli %3, %c16_i32 : i32
        %18 = arith.index_cast %17 : i32 to index
        %subview_119 = memref.subview %6[0, %18, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_119, %alloc_61 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_120 = memref.subview %5[0, %18, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_120, %alloc_63 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %19 = arith.index_cast %1 : i32 to index
        %20 = arith.muli %arg28, %c16_i32 : i32
        %21 = arith.muli %3, %c8_i32 : i32
        %22 = arith.addi %20, %21 : i32
        %23 = arith.index_cast %22 : i32 to index
        %subview_121 = memref.subview %reinterpret_cast_13[0, %19, %23] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_121, %alloc_64 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>>
        hivm.hir.vexp ins(%alloc_64 : memref<8xf32, strided<[1]>>) outs(%alloc_64 : memref<8xf32, strided<[1]>>)
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %24 = arith.divsi %arg30, %c2_i32 : i32
            %25 = arith.muli %3, %c8_i32 : i32
            %26 = arith.divsi %arg29, %c2_i32 : i32
            %27 = arith.addi %25, %26 : i32
            %28 = arith.cmpi slt, %24, %27 : i32
            scf.if %28 {
              %29 = arith.index_cast %arg29 : i32 to index
              %30 = arith.index_cast %arg30 : i32 to index
              %31 = memref.load %alloc_61[%29, %30] : memref<16x32xf32, strided<[32, 1]>>
              %32 = arith.muli %3, %c8_i32 : i32
              %33 = arith.divsi %arg29, %c2_i32 : i32
              %34 = arith.addi %32, %33 : i32
              %35 = arith.index_cast %34 : i32 to index
              %36 = arith.divsi %arg30, %c2_i32 : i32
              %37 = arith.index_cast %36 : i32 to index
              %38 = memref.load %alloc_33[%35, %37] : memref<16x16xf32, strided<[16, 1]>>
              %39 = arith.mulf %31, %38 : f32
              %40 = arith.truncf %39 : f32 to bf16
              memref.store %40, %alloc_62[%29, %30] : memref<16x32xbf16, strided<[32, 1]>>
            } else {
              %29 = arith.index_cast %arg29 : i32 to index
              %30 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_4, %alloc_62[%29, %30] : memref<16x32xbf16, strided<[32, 1]>>
            }
          }
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg30 : i32 to index
            %26 = memref.load %alloc_63[%24, %25] : memref<16x64xf32, strided<[64, 1]>>
            %27 = arith.divsi %arg29, %c2_i32 : i32
            %28 = arith.index_cast %27 : i32 to index
            %29 = memref.load %alloc_64[%28] : memref<8xf32, strided<[1]>>
            %30 = arith.mulf %26, %29 : f32
            memref.store %30, %alloc_63[%24, %25] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %subview_122 = memref.subview %7[0, %18, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_62, %subview_122 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        %subview_123 = memref.subview %5[0, %18, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %alloc_63, %subview_123 : memref<16x64xf32, strided<[64, 1]>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      }
      %subview_117 = memref.subview %7[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_117, %alloc_32 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 ins(%alloc_32, %alloc_29, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_65 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_118 = memref.subview %8[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_65, %subview_118 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.if %16 {
        %17 = arith.muli %3, %c16_i32 : i32
        %18 = arith.index_cast %17 : i32 to index
        %subview_119 = memref.subview %5[0, %18, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_119, %alloc_66 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_120 = memref.subview %8[0, %18, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_120, %alloc_67 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_121 = memref.subview %4[0, %18, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_121, %alloc_68 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %19 = arith.muli %arg28, %c16_i32 : i32
        %20 = arith.muli %3, %c8_i32 : i32
        %21 = arith.addi %19, %20 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = arith.index_cast %1 : i32 to index
        %subview_122 = memref.subview %reinterpret_cast_14[0, %22, %23, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_122, %alloc_69 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg30 : i32 to index
            %26 = memref.load %alloc_66[%24, %25] : memref<16x64xf32, strided<[64, 1]>>
            %27 = memref.load %alloc_67[%24, %25] : memref<16x64xf32, strided<[64, 1]>>
            %28 = arith.addf %26, %27 : f32
            memref.store %28, %alloc_66[%24, %25] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        scf.for %arg29 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_4, %alloc_70[%24, %25] : memref<8x64xbf16, strided<[64, 1]>>
          }
        }
        %subview_123 = memref.subview %reinterpret_cast_18[0, %22, %23, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_70, %subview_123 : memref<8x64xbf16, strided<[64, 1]>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
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
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_21 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_22 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_23 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_24 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
    %alloc_25 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
    %alloc_26 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vbrc ins(%cst_7 : f32) outs(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>)
    %9 = arith.index_cast %1 : i32 to index
    %subview = memref.subview %reinterpret_cast_10[%9, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_26 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vcast ins(%alloc_26 : memref<2x64xf32, strided<[64, 1]>>) outs(%alloc_25 : memref<2x64xbf16, strided<[64, 1]>>)
    %subview_27 = memref.subview %reinterpret_cast_11[%9, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_27, %alloc_22 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_22 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>>)
    %subview_28 = memref.subview %reinterpret_cast_15[%9, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_28, %alloc_23 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc_21 : memref<2x32xbf16, strided<[32, 1]>>)
    scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
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
      %alloc_48 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_49 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_50 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_51 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_52 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_53 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_54 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_55 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_56 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_57 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_58 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_59 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_60 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_61 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_62 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
      %alloc_63 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_64 = memref.alloc() : memref<8xf32, strided<[1]>>
      %alloc_65 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_66 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_67 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_68 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_69 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_70 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_71 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_72 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_73 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
      %alloc_74 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_75 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_76 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_77 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_78 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_79 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_80 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_81 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_82 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_83 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_84 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_85 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_86 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_87 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %10 = arith.index_cast %1 : i32 to index
      %11 = arith.index_cast %arg28 : i32 to index
      %subview_88 = memref.subview %reinterpret_cast_20[0, %10, %11, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_88, %alloc_33 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>>
      %12 = arith.muli %arg28, %c16_i32 : i32
      %13 = arith.addi %12, %c1_i32 : i32
      %14 = arith.index_cast %13 : i32 to index
      %subview_89 = memref.subview %reinterpret_cast_16[0, %10, %14] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_89, %alloc_35 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_35 : memref<16xbf16, strided<[1]>>) outs(%alloc_34 : memref<16xf32, strided<[1]>>)
      hivm.hir.vmul ins(%alloc_34, %cst_8 : memref<16xf32, strided<[1]>>, f32) outs(%alloc_34 : memref<16xf32, strided<[1]>>)
      %alloc_90 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_34 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_90 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_90 : memref<16xf32>) outs(%alloc_90 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_90, %cst_6 : memref<16xf32>, f32) outs(%alloc_90 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_90 : f32, memref<16xf32>) outs(%alloc_34 : memref<16xf32, strided<[1]>>)
      %subview_91 = memref.subview %reinterpret_cast_19[0, %10, %14] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_91, %alloc_37 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xf32, strided<[1]>>) outs(%alloc_36 : memref<16xbf16, strided<[1]>>)
      hivm.hir.vbrc ins(%cst_4 : bf16) outs(%alloc_38 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %17 = arith.muli %arg28, %c16_i32 : i32
        %18 = arith.addi %17, %arg29 : i32
        %19 = arith.cmpi slt, %18, %c31_i32 : i32
        scf.if %19 {
          %20 = arith.index_cast %arg29 : i32 to index
          %21 = memref.load %alloc_36[%20] : memref<16xbf16, strided<[1]>>
          %22 = arith.extf %21 : bf16 to f32
          %23 = memref.load %alloc_34[%20] : memref<16xf32, strided<[1]>>
          %24 = arith.mulf %22, %23 : f32
          %25 = arith.truncf %24 : f32 to bf16
          memref.store %25, %alloc_38[%20] : memref<16xbf16, strided<[1]>>
        }
      }
      memref.copy %alloc_38, %alloc_39 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %15 = arith.index_cast %12 : i32 to index
      %subview_92 = memref.subview %reinterpret_cast_16[0, %10, %15] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_92, %alloc_41 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_41 : memref<16xbf16, strided<[1]>>) outs(%alloc_40 : memref<16xf32, strided<[1]>>)
      %alloc_93 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_40 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_93 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_93 : memref<16xf32>) outs(%alloc_93 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_93, %cst_6 : memref<16xf32>, f32) outs(%alloc_93 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_93 : f32, memref<16xf32>) outs(%alloc_40 : memref<16xf32, strided<[1]>>)
      %subview_94 = memref.subview %reinterpret_cast_19[0, %10, %15] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_94, %alloc_43 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_43 : memref<16xf32, strided<[1]>>) outs(%alloc_42 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %17 = arith.index_cast %arg29 : i32 to index
        %18 = memref.load %alloc_42[%17] : memref<16xbf16, strided<[1]>>
        %19 = arith.extf %18 : bf16 to f32
        %20 = memref.load %alloc_40[%17] : memref<16xf32, strided<[1]>>
        %21 = arith.mulf %19, %20 : f32
        memref.store %21, %alloc_44[%17] : memref<16xf32, strided<[1]>>
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %17 = arith.index_cast %arg29 : i32 to index
        %18 = memref.load %alloc_44[%17] : memref<16xf32, strided<[1]>>
        %19 = memref.load %alloc_39[%17] : memref<16xbf16, strided<[1]>>
        %20 = arith.extf %19 : bf16 to f32
        %21 = arith.addf %18, %20 : f32
        %22 = arith.truncf %21 : f32 to bf16
        memref.store %22, %alloc_45[%17] : memref<16xbf16, strided<[1]>>
      }
      memref.copy %alloc_45, %alloc_46 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %17 = arith.muli %arg28, %c16_i32 : i32
        %18 = arith.addi %17, %arg29 : i32
        %19 = arith.index_cast %18 : i32 to index
        %20 = arith.index_cast %1 : i32 to index
        %subview_119 = memref.subview %reinterpret_cast_9[0, %19, %20, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %21 = arith.index_cast %arg29 : i32 to index
        %subview_120 = memref.subview %alloc_30[%21, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<64xbf16, strided<[1], offset: ?>>
        memref.copy %subview_119, %subview_120 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>>
      }
      memref.copy %alloc_30, %alloc_71 : memref<16x64xbf16, strided<[64, 1]>> to memref<16x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_95 = memref.reinterpret_cast %alloc_71 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<16x1x64xbf16, strided<[64, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_95 : memref<16x1x64xbf16, strided<[64, 64, 1]>>) outs(%alloc_72 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [1]
      memref.copy %alloc_25, %alloc_73 : memref<2x64xbf16, strided<[64, 1]>> to memref<2x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_96 = memref.reinterpret_cast %alloc_73 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>> to memref<1x2x64xbf16, strided<[128, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_96 : memref<1x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_74 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_72 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_79 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_74 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_80 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vmul ins(%alloc_79, %alloc_80 : memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_81 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_81 : memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_47 : memref<16x2x64xbf16, strided<[128, 64, 1]>>)
      %reinterpret_cast_97 = memref.reinterpret_cast %alloc_47 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %reinterpret_cast_97, %alloc_29 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_98 = memref.subview %reinterpret_cast[0, %15, 0, %10, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_98, %alloc_48 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc, %alloc_75 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_99 = memref.reinterpret_cast %alloc_75 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_99 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_76 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_48 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_82 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_76 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_83 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_82, %alloc_83 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_84 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_84 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_48 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_100 = memref.reinterpret_cast %alloc_48 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_101 = memref.subview %reinterpret_cast_17[0, %15, 0, %10, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_101, %alloc_49 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc_21, %alloc_77 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_102 = memref.reinterpret_cast %alloc_77 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_102 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_78 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_49 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_85 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_78 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_86 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_85, %alloc_86 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_87 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_87 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_49 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      %reinterpret_cast_103 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%reinterpret_cast_100, %reinterpret_cast_103, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_50 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_104 = memref.subview %4[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_50, %subview_104 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.muli %arg29, %c2_i32 : i32
            %18 = arith.addi %17, %arg30 : i32
            %19 = arith.index_cast %18 : i32 to index
            %20 = arith.index_cast %arg31 : i32 to index
            %21 = memref.load %reinterpret_cast_100[%19, %20] : memref<32x32xbf16, strided<[32, 1]>>
            %22 = arith.index_cast %arg29 : i32 to index
            %23 = arith.index_cast %arg30 : i32 to index
            memref.store %21, %alloc_51[%22, %23, %20] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %subview_105 = memref.subview %reinterpret_cast_12[0, %15, %10, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_105, %alloc_52 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>>
      %alloc_106 = memref.alloc() : memref<16x8xf32>
      %alloc_107 = memref.alloc() : memref<16x8xf32>
      %alloc_108 = memref.alloc() : memref<16x8xf32>
      %alloc_109 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_52, %alloc_52 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_106 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_106, %alloc_106 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_107 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_106, %alloc_107 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_108 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_106, %cst_5 : memref<16x8xf32>, f32) outs(%alloc_106 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_107, %cst_3 : memref<16x8xf32>, f32) outs(%alloc_107 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_108, %cst_2 : memref<16x8xf32>, f32) outs(%alloc_108 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_106, %cst_6 : memref<16x8xf32>, f32) outs(%alloc_109 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_107, %alloc_109 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_109 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_108, %alloc_109 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_53 : memref<16x8xf32, strided<[8, 1]>>)
      %alloc_110 = memref.alloc() : memref<16x8xf32>
      %alloc_111 = memref.alloc() : memref<16x8xf32>
      %alloc_112 = memref.alloc() : memref<16x8xf32>
      %alloc_113 = memref.alloc() : memref<16x8xf32>
      %alloc_114 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_52, %alloc_52 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_110 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_110, %alloc_52 : memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_111 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_111, %alloc_110 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_112 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_112, %alloc_110 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_113 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_111, %cst_1 : memref<16x8xf32>, f32) outs(%alloc_111 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_112, %cst_0 : memref<16x8xf32>, f32) outs(%alloc_112 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_113, %cst : memref<16x8xf32>, f32) outs(%alloc_113 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_52, %alloc_111 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) outs(%alloc_114 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_112, %alloc_114 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_114 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_113, %alloc_114 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_54 : memref<16x8xf32, strided<[8, 1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.index_cast %arg29 : i32 to index
            %18 = arith.index_cast %arg31 : i32 to index
            %19 = memref.load %alloc_53[%17, %18] : memref<16x8xf32, strided<[8, 1]>>
            %20 = arith.index_cast %arg30 : i32 to index
            %21 = memref.load %alloc_51[%17, %20, %18] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %22 = arith.extf %21 : bf16 to f32
            %23 = arith.mulf %19, %22 : f32
            %24 = arith.subf %23, %23 : f32
            %25 = arith.truncf %24 : f32 to bf16
            %26 = arith.muli %arg29, %c2_i32 : i32
            %27 = arith.addi %26, %arg30 : i32
            %28 = arith.index_cast %27 : i32 to index
            memref.store %25, %reinterpret_cast_100[%28, %18] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.index_cast %arg29 : i32 to index
            %18 = arith.index_cast %arg31 : i32 to index
            %19 = memref.load %alloc_54[%17, %18] : memref<16x8xf32, strided<[8, 1]>>
            %20 = arith.index_cast %arg30 : i32 to index
            %21 = memref.load %alloc_51[%17, %20, %18] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %22 = arith.extf %21 : bf16 to f32
            %23 = arith.mulf %19, %22 : f32
            %24 = arith.addf %23, %23 : f32
            %25 = arith.truncf %24 : f32 to bf16
            %26 = arith.muli %arg29, %c2_i32 : i32
            %27 = arith.addi %26, %arg30 : i32
            %28 = arith.index_cast %27 : i32 to index
            %29 = arith.addi %arg31, %c16_i32 : i32
            %30 = arith.index_cast %29 : i32 to index
            memref.store %25, %reinterpret_cast_100[%28, %30] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      hivm.hir.vcast ins(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>) outs(%alloc_56 : memref<32x64xbf16, strided<[64, 1]>>)
      memref.copy %alloc_56, %alloc_31 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%reinterpret_cast_100, %alloc_31, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_55 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_115 = memref.subview %5[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_55, %subview_115 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.muli %arg29, %c2_i32 : i32
            %18 = arith.addi %17, %arg30 : i32
            %19 = arith.index_cast %18 : i32 to index
            %20 = arith.index_cast %arg31 : i32 to index
            %21 = memref.load %reinterpret_cast_103[%19, %20] : memref<32x32xbf16, strided<[32, 1]>>
            %22 = arith.index_cast %arg29 : i32 to index
            %23 = arith.index_cast %arg30 : i32 to index
            memref.store %21, %alloc_57[%22, %23, %20] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.index_cast %arg29 : i32 to index
            %18 = arith.index_cast %arg31 : i32 to index
            %19 = memref.load %alloc_53[%17, %18] : memref<16x8xf32, strided<[8, 1]>>
            %20 = arith.index_cast %arg30 : i32 to index
            %21 = memref.load %alloc_57[%17, %20, %18] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %22 = arith.extf %21 : bf16 to f32
            %23 = arith.mulf %19, %22 : f32
            %24 = arith.subf %23, %23 : f32
            %25 = arith.truncf %24 : f32 to bf16
            %26 = arith.muli %arg29, %c2_i32 : i32
            %27 = arith.addi %26, %arg30 : i32
            %28 = arith.index_cast %27 : i32 to index
            memref.store %25, %reinterpret_cast_103[%28, %18] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %17 = arith.index_cast %arg29 : i32 to index
            %18 = arith.index_cast %arg31 : i32 to index
            %19 = memref.load %alloc_54[%17, %18] : memref<16x8xf32, strided<[8, 1]>>
            %20 = arith.index_cast %arg30 : i32 to index
            %21 = memref.load %alloc_57[%17, %20, %18] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %22 = arith.extf %21 : bf16 to f32
            %23 = arith.mulf %19, %22 : f32
            %24 = arith.addf %23, %23 : f32
            %25 = arith.truncf %24 : f32 to bf16
            %26 = arith.muli %arg29, %c2_i32 : i32
            %27 = arith.addi %26, %arg30 : i32
            %28 = arith.index_cast %27 : i32 to index
            %29 = arith.addi %arg31, %c16_i32 : i32
            %30 = arith.index_cast %29 : i32 to index
            memref.store %25, %reinterpret_cast_103[%28, %30] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      memref.copy %reinterpret_cast_103, %alloc_58 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_58 : memref<32x32xbf16, strided<[32, 1]>>) outs(%alloc_59 : memref<32x32xf32, strided<[32, 1]>>)
      scf.for %arg29 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
          %17 = arith.index_cast %arg29 : i32 to index
          %18 = arith.index_cast %arg30 : i32 to index
          %19 = memref.load %alloc_59[%17, %18] : memref<32x32xf32, strided<[32, 1]>>
          %20 = arith.divsi %arg29, %c2_i32 : i32
          %21 = arith.index_cast %20 : i32 to index
          %22 = memref.load %alloc_46[%21] : memref<16xbf16, strided<[1]>>
          %23 = arith.extf %22 : bf16 to f32
          %24 = arith.mulf %19, %23 : f32
          memref.store %24, %alloc_59[%17, %18] : memref<32x32xf32, strided<[32, 1]>>
        }
      }
      hivm.hir.vcast ins(%alloc_59 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc_58 : memref<32x32xbf16, strided<[32, 1]>>)
      memref.copy %alloc_58, %reinterpret_cast_103 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%reinterpret_cast_100, %reinterpret_cast_103, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_60 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_116 = memref.subview %6[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_60, %subview_116 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      hivm.hir.vexp ins(%alloc_33 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_33 : memref<16x16xf32, strided<[16, 1]>>)
      %16 = arith.cmpi slt, %3, %c2_i32 : i32
      scf.if %16 {
        %17 = arith.muli %3, %c16_i32 : i32
        %18 = arith.index_cast %17 : i32 to index
        %subview_119 = memref.subview %6[0, %18, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_119, %alloc_61 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_120 = memref.subview %5[0, %18, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_120, %alloc_63 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %19 = arith.index_cast %1 : i32 to index
        %20 = arith.muli %arg28, %c16_i32 : i32
        %21 = arith.muli %3, %c8_i32 : i32
        %22 = arith.addi %20, %21 : i32
        %23 = arith.index_cast %22 : i32 to index
        %subview_121 = memref.subview %reinterpret_cast_13[0, %19, %23] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_121, %alloc_64 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>>
        hivm.hir.vexp ins(%alloc_64 : memref<8xf32, strided<[1]>>) outs(%alloc_64 : memref<8xf32, strided<[1]>>)
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %24 = arith.divsi %arg30, %c2_i32 : i32
            %25 = arith.muli %3, %c8_i32 : i32
            %26 = arith.divsi %arg29, %c2_i32 : i32
            %27 = arith.addi %25, %26 : i32
            %28 = arith.cmpi slt, %24, %27 : i32
            scf.if %28 {
              %29 = arith.index_cast %arg29 : i32 to index
              %30 = arith.index_cast %arg30 : i32 to index
              %31 = memref.load %alloc_61[%29, %30] : memref<16x32xf32, strided<[32, 1]>>
              %32 = arith.muli %3, %c8_i32 : i32
              %33 = arith.divsi %arg29, %c2_i32 : i32
              %34 = arith.addi %32, %33 : i32
              %35 = arith.index_cast %34 : i32 to index
              %36 = arith.divsi %arg30, %c2_i32 : i32
              %37 = arith.index_cast %36 : i32 to index
              %38 = memref.load %alloc_33[%35, %37] : memref<16x16xf32, strided<[16, 1]>>
              %39 = arith.mulf %31, %38 : f32
              %40 = arith.truncf %39 : f32 to bf16
              memref.store %40, %alloc_62[%29, %30] : memref<16x32xbf16, strided<[32, 1]>>
            } else {
              %29 = arith.index_cast %arg29 : i32 to index
              %30 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_4, %alloc_62[%29, %30] : memref<16x32xbf16, strided<[32, 1]>>
            }
          }
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg30 : i32 to index
            %26 = memref.load %alloc_63[%24, %25] : memref<16x64xf32, strided<[64, 1]>>
            %27 = arith.divsi %arg29, %c2_i32 : i32
            %28 = arith.index_cast %27 : i32 to index
            %29 = memref.load %alloc_64[%28] : memref<8xf32, strided<[1]>>
            %30 = arith.mulf %26, %29 : f32
            memref.store %30, %alloc_63[%24, %25] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %subview_122 = memref.subview %7[0, %18, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_62, %subview_122 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        %subview_123 = memref.subview %5[0, %18, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %alloc_63, %subview_123 : memref<16x64xf32, strided<[64, 1]>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      }
      %subview_117 = memref.subview %7[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_117, %alloc_32 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 ins(%alloc_32, %alloc_29, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_65 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_118 = memref.subview %8[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_65, %subview_118 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.if %16 {
        %17 = arith.muli %3, %c16_i32 : i32
        %18 = arith.index_cast %17 : i32 to index
        %subview_119 = memref.subview %5[0, %18, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_119, %alloc_66 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_120 = memref.subview %8[0, %18, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_120, %alloc_67 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_121 = memref.subview %4[0, %18, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_121, %alloc_68 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %19 = arith.muli %arg28, %c16_i32 : i32
        %20 = arith.muli %3, %c8_i32 : i32
        %21 = arith.addi %19, %20 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = arith.index_cast %1 : i32 to index
        %subview_122 = memref.subview %reinterpret_cast_14[0, %22, %23, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_122, %alloc_69 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg30 : i32 to index
            %26 = memref.load %alloc_66[%24, %25] : memref<16x64xf32, strided<[64, 1]>>
            %27 = memref.load %alloc_67[%24, %25] : memref<16x64xf32, strided<[64, 1]>>
            %28 = arith.addf %26, %27 : f32
            memref.store %28, %alloc_66[%24, %25] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        scf.for %arg29 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_4, %alloc_70[%24, %25] : memref<8x64xbf16, strided<[64, 1]>>
          }
        }
        %subview_123 = memref.subview %reinterpret_cast_18[0, %22, %23, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_70, %subview_123 : memref<8x64xbf16, strided<[64, 1]>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
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
    %9 = memref_ext.alloc_workspace() : memref<32x64xbf16>
    %10 = memref_ext.alloc_workspace() : memref<32x64xbf16>
    %11 = memref_ext.alloc_workspace() : memref<16x2x64xbf16>
    %12 = memref_ext.alloc_workspace() : memref<16x2x32xbf16>
    %13 = memref_ext.alloc_workspace() : memref<16x2x32xbf16>
    %14 = memref_ext.alloc_workspace() : memref<32x64xbf16>
    %15 = memref_ext.alloc_workspace() : memref<32x32xbf16>
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_21 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_22 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_23 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_24 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
    %alloc_25 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
    %alloc_26 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vbrc ins(%cst_7 : f32) outs(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>)
    %16 = arith.index_cast %1 : i32 to index
    %subview = memref.subview %reinterpret_cast_10[%16, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_26 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vcast ins(%alloc_26 : memref<2x64xf32, strided<[64, 1]>>) outs(%alloc_25 : memref<2x64xbf16, strided<[64, 1]>>)
    %subview_27 = memref.subview %reinterpret_cast_11[%16, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_27, %alloc_22 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_22 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>>)
    %subview_28 = memref.subview %reinterpret_cast_15[%16, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_28, %alloc_23 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc_21 : memref<2x32xbf16, strided<[32, 1]>>)
    scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %alloc_29 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_30 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_31 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_32 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
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
      %alloc_50 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_51 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_52 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_53 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_54 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_55 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_56 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_57 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_58 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_59 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_60 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_61 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_62 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_63 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_64 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_65 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_66 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_67 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_68 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_69 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_70 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
      %alloc_71 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_72 = memref.alloc() : memref<8xf32, strided<[1]>>
      %alloc_73 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_74 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_75 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_76 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_77 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_78 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_79 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_80 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_81 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
      %alloc_82 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_83 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_84 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_85 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_86 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_87 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_88 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_89 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_90 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_91 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_92 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_93 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_94 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_95 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %17 = arith.index_cast %1 : i32 to index
      %18 = arith.index_cast %arg28 : i32 to index
      %subview_96 = memref.subview %reinterpret_cast_20[0, %17, %18, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_96, %alloc_35 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>>
      %19 = arith.muli %arg28, %c16_i32 : i32
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.index_cast %20 : i32 to index
      %subview_97 = memref.subview %reinterpret_cast_16[0, %17, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_97, %alloc_37 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xbf16, strided<[1]>>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      hivm.hir.vmul ins(%alloc_36, %cst_8 : memref<16xf32, strided<[1]>>, f32) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %alloc_98 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_36 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_98 : memref<16xf32>) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_98, %cst_6 : memref<16xf32>, f32) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_98 : f32, memref<16xf32>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %subview_99 = memref.subview %reinterpret_cast_19[0, %17, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_99, %alloc_39 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_39 : memref<16xf32, strided<[1]>>) outs(%alloc_38 : memref<16xbf16, strided<[1]>>)
      hivm.hir.vbrc ins(%cst_4 : bf16) outs(%alloc_40 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %24 = arith.muli %arg28, %c16_i32 : i32
        %25 = arith.addi %24, %arg29 : i32
        %26 = arith.cmpi slt, %25, %c31_i32 : i32
        scf.if %26 {
          %27 = arith.index_cast %arg29 : i32 to index
          %28 = memref.load %alloc_38[%27] : memref<16xbf16, strided<[1]>>
          %29 = arith.extf %28 : bf16 to f32
          %30 = memref.load %alloc_36[%27] : memref<16xf32, strided<[1]>>
          %31 = arith.mulf %29, %30 : f32
          %32 = arith.truncf %31 : f32 to bf16
          memref.store %32, %alloc_40[%27] : memref<16xbf16, strided<[1]>>
        }
      }
      memref.copy %alloc_40, %alloc_41 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %22 = arith.index_cast %19 : i32 to index
      %subview_100 = memref.subview %reinterpret_cast_16[0, %17, %22] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_100, %alloc_43 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_43 : memref<16xbf16, strided<[1]>>) outs(%alloc_42 : memref<16xf32, strided<[1]>>)
      %alloc_101 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_42 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_101 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_101 : memref<16xf32>) outs(%alloc_101 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_101, %cst_6 : memref<16xf32>, f32) outs(%alloc_101 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_101 : f32, memref<16xf32>) outs(%alloc_42 : memref<16xf32, strided<[1]>>)
      %subview_102 = memref.subview %reinterpret_cast_19[0, %17, %22] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_102, %alloc_45 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_45 : memref<16xf32, strided<[1]>>) outs(%alloc_44 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %24 = arith.index_cast %arg29 : i32 to index
        %25 = memref.load %alloc_44[%24] : memref<16xbf16, strided<[1]>>
        %26 = arith.extf %25 : bf16 to f32
        %27 = memref.load %alloc_42[%24] : memref<16xf32, strided<[1]>>
        %28 = arith.mulf %26, %27 : f32
        memref.store %28, %alloc_46[%24] : memref<16xf32, strided<[1]>>
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %24 = arith.index_cast %arg29 : i32 to index
        %25 = memref.load %alloc_46[%24] : memref<16xf32, strided<[1]>>
        %26 = memref.load %alloc_41[%24] : memref<16xbf16, strided<[1]>>
        %27 = arith.extf %26 : bf16 to f32
        %28 = arith.addf %25, %27 : f32
        %29 = arith.truncf %28 : f32 to bf16
        memref.store %29, %alloc_47[%24] : memref<16xbf16, strided<[1]>>
      }
      memref.copy %alloc_47, %alloc_48 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %24 = arith.muli %arg28, %c16_i32 : i32
        %25 = arith.addi %24, %arg29 : i32
        %26 = arith.index_cast %25 : i32 to index
        %27 = arith.index_cast %1 : i32 to index
        %subview_127 = memref.subview %reinterpret_cast_9[0, %26, %27, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %28 = arith.index_cast %arg29 : i32 to index
        %subview_128 = memref.subview %alloc_31[%28, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<64xbf16, strided<[1], offset: ?>>
        memref.copy %subview_127, %subview_128 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>>
      }
      memref.copy %alloc_31, %alloc_79 : memref<16x64xbf16, strided<[64, 1]>> to memref<16x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_103 = memref.reinterpret_cast %alloc_79 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<16x1x64xbf16, strided<[64, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_103 : memref<16x1x64xbf16, strided<[64, 64, 1]>>) outs(%alloc_80 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [1]
      memref.copy %alloc_25, %alloc_81 : memref<2x64xbf16, strided<[64, 1]>> to memref<2x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_104 = memref.reinterpret_cast %alloc_81 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>> to memref<1x2x64xbf16, strided<[128, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_104 : memref<1x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_82 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_80 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_87 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_82 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_88 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vmul ins(%alloc_87, %alloc_88 : memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_89 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_89 : memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_49 : memref<16x2x64xbf16, strided<[128, 64, 1]>>)
      memref.copy %alloc_49, %11 : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<16x2x64xbf16>
      memref.copy %11, %alloc_50 : memref<16x2x64xbf16> to memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %reinterpret_cast_105 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %reinterpret_cast_105, %alloc_29 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %alloc_29, %9 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16>
      memref.copy %9, %alloc_30 : memref<32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_106 = memref.subview %reinterpret_cast[0, %22, 0, %17, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_106, %alloc_51 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc, %alloc_83 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_107 = memref.reinterpret_cast %alloc_83 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_107 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_84 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_90 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_84 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_90, %alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      memref.copy %alloc_51, %12 : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<16x2x32xbf16>
      memref.copy %12, %alloc_52 : memref<16x2x32xbf16> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %reinterpret_cast_108 = memref.reinterpret_cast %alloc_51 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_109 = memref.subview %reinterpret_cast_17[0, %22, 0, %17, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_109, %alloc_53 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc_21, %alloc_85 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_110 = memref.reinterpret_cast %alloc_85 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_110 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_86 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_53 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_93 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_86 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_94 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_93, %alloc_94 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_95 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_95 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_53 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      memref.copy %alloc_53, %13 : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<16x2x32xbf16>
      memref.copy %13, %alloc_55 : memref<16x2x32xbf16> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %reinterpret_cast_111 = memref.reinterpret_cast %alloc_53 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%reinterpret_cast_108, %reinterpret_cast_111, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_56 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_112 = memref.subview %4[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_56, %subview_112 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %24 = arith.muli %arg29, %c2_i32 : i32
            %25 = arith.addi %24, %arg30 : i32
            %26 = arith.index_cast %25 : i32 to index
            %27 = arith.index_cast %arg31 : i32 to index
            %28 = memref.load %reinterpret_cast_108[%26, %27] : memref<32x32xbf16, strided<[32, 1]>>
            %29 = arith.index_cast %arg29 : i32 to index
            %30 = arith.index_cast %arg30 : i32 to index
            memref.store %28, %alloc_57[%29, %30, %27] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %subview_113 = memref.subview %reinterpret_cast_12[0, %22, %17, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_113, %alloc_58 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>>
      %alloc_114 = memref.alloc() : memref<16x8xf32>
      %alloc_115 = memref.alloc() : memref<16x8xf32>
      %alloc_116 = memref.alloc() : memref<16x8xf32>
      %alloc_117 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_58, %alloc_58 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_114 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_114, %alloc_114 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_115 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_114, %alloc_115 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_116 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_114, %cst_5 : memref<16x8xf32>, f32) outs(%alloc_114 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_115, %cst_3 : memref<16x8xf32>, f32) outs(%alloc_115 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_116, %cst_2 : memref<16x8xf32>, f32) outs(%alloc_116 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_114, %cst_6 : memref<16x8xf32>, f32) outs(%alloc_117 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_115, %alloc_117 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_117 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_116, %alloc_117 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_59 : memref<16x8xf32, strided<[8, 1]>>)
      %alloc_118 = memref.alloc() : memref<16x8xf32>
      %alloc_119 = memref.alloc() : memref<16x8xf32>
      %alloc_120 = memref.alloc() : memref<16x8xf32>
      %alloc_121 = memref.alloc() : memref<16x8xf32>
      %alloc_122 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_58, %alloc_58 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_118 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_118, %alloc_58 : memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_119 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_119, %alloc_118 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_120 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_120, %alloc_118 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_121 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_119, %cst_1 : memref<16x8xf32>, f32) outs(%alloc_119 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_120, %cst_0 : memref<16x8xf32>, f32) outs(%alloc_120 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_121, %cst : memref<16x8xf32>, f32) outs(%alloc_121 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_58, %alloc_119 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) outs(%alloc_122 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_120, %alloc_122 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_122 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_121, %alloc_122 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_60 : memref<16x8xf32, strided<[8, 1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg31 : i32 to index
            %26 = memref.load %alloc_59[%24, %25] : memref<16x8xf32, strided<[8, 1]>>
            %27 = arith.index_cast %arg30 : i32 to index
            %28 = memref.load %alloc_57[%24, %27, %25] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %29 = arith.extf %28 : bf16 to f32
            %30 = arith.mulf %26, %29 : f32
            %31 = arith.subf %30, %30 : f32
            %32 = arith.truncf %31 : f32 to bf16
            %33 = arith.muli %arg29, %c2_i32 : i32
            %34 = arith.addi %33, %arg30 : i32
            %35 = arith.index_cast %34 : i32 to index
            memref.store %32, %reinterpret_cast_108[%35, %25] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg31 : i32 to index
            %26 = memref.load %alloc_60[%24, %25] : memref<16x8xf32, strided<[8, 1]>>
            %27 = arith.index_cast %arg30 : i32 to index
            %28 = memref.load %alloc_57[%24, %27, %25] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %29 = arith.extf %28 : bf16 to f32
            %30 = arith.mulf %26, %29 : f32
            %31 = arith.addf %30, %30 : f32
            %32 = arith.truncf %31 : f32 to bf16
            %33 = arith.muli %arg29, %c2_i32 : i32
            %34 = arith.addi %33, %arg30 : i32
            %35 = arith.index_cast %34 : i32 to index
            %36 = arith.addi %arg31, %c16_i32 : i32
            %37 = arith.index_cast %36 : i32 to index
            memref.store %32, %reinterpret_cast_108[%35, %37] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      hivm.hir.vcast ins(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>) outs(%alloc_62 : memref<32x64xbf16, strided<[64, 1]>>)
      memref.copy %alloc_62, %14 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16>
      memref.copy %14, %alloc_63 : memref<32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %alloc_63, %alloc_32 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %alloc_32, %10 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16>
      memref.copy %10, %alloc_33 : memref<32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%reinterpret_cast_108, %alloc_33, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_61 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_123 = memref.subview %5[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_61, %subview_123 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %24 = arith.muli %arg29, %c2_i32 : i32
            %25 = arith.addi %24, %arg30 : i32
            %26 = arith.index_cast %25 : i32 to index
            %27 = arith.index_cast %arg31 : i32 to index
            %28 = memref.load %reinterpret_cast_111[%26, %27] : memref<32x32xbf16, strided<[32, 1]>>
            %29 = arith.index_cast %arg29 : i32 to index
            %30 = arith.index_cast %arg30 : i32 to index
            memref.store %28, %alloc_64[%29, %30, %27] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg31 : i32 to index
            %26 = memref.load %alloc_59[%24, %25] : memref<16x8xf32, strided<[8, 1]>>
            %27 = arith.index_cast %arg30 : i32 to index
            %28 = memref.load %alloc_64[%24, %27, %25] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %29 = arith.extf %28 : bf16 to f32
            %30 = arith.mulf %26, %29 : f32
            %31 = arith.subf %30, %30 : f32
            %32 = arith.truncf %31 : f32 to bf16
            %33 = arith.muli %arg29, %c2_i32 : i32
            %34 = arith.addi %33, %arg30 : i32
            %35 = arith.index_cast %34 : i32 to index
            memref.store %32, %reinterpret_cast_111[%35, %25] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg31 : i32 to index
            %26 = memref.load %alloc_60[%24, %25] : memref<16x8xf32, strided<[8, 1]>>
            %27 = arith.index_cast %arg30 : i32 to index
            %28 = memref.load %alloc_64[%24, %27, %25] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %29 = arith.extf %28 : bf16 to f32
            %30 = arith.mulf %26, %29 : f32
            %31 = arith.addf %30, %30 : f32
            %32 = arith.truncf %31 : f32 to bf16
            %33 = arith.muli %arg29, %c2_i32 : i32
            %34 = arith.addi %33, %arg30 : i32
            %35 = arith.index_cast %34 : i32 to index
            %36 = arith.addi %arg31, %c16_i32 : i32
            %37 = arith.index_cast %36 : i32 to index
            memref.store %32, %reinterpret_cast_111[%35, %37] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      memref.copy %reinterpret_cast_111, %alloc_65 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_65 : memref<32x32xbf16, strided<[32, 1]>>) outs(%alloc_67 : memref<32x32xf32, strided<[32, 1]>>)
      scf.for %arg29 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
          %24 = arith.index_cast %arg29 : i32 to index
          %25 = arith.index_cast %arg30 : i32 to index
          %26 = memref.load %alloc_67[%24, %25] : memref<32x32xf32, strided<[32, 1]>>
          %27 = arith.divsi %arg29, %c2_i32 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = memref.load %alloc_48[%28] : memref<16xbf16, strided<[1]>>
          %30 = arith.extf %29 : bf16 to f32
          %31 = arith.mulf %26, %30 : f32
          memref.store %31, %alloc_67[%24, %25] : memref<32x32xf32, strided<[32, 1]>>
        }
      }
      hivm.hir.vcast ins(%alloc_67 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc_65 : memref<32x32xbf16, strided<[32, 1]>>)
      memref.copy %alloc_65, %15 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16>
      memref.copy %15, %alloc_66 : memref<32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_66, %reinterpret_cast_111 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_55, %13 : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<16x2x32xbf16>
      memref.copy %13, %alloc_54 : memref<16x2x32xbf16> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%reinterpret_cast_108, %reinterpret_cast_111, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_68 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_124 = memref.subview %6[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_68, %subview_124 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      hivm.hir.vexp ins(%alloc_35 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_35 : memref<16x16xf32, strided<[16, 1]>>)
      %23 = arith.cmpi slt, %3, %c2_i32 : i32
      scf.if %23 {
        %24 = arith.muli %3, %c16_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %subview_127 = memref.subview %6[0, %25, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_127, %alloc_69 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_128 = memref.subview %5[0, %25, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_128, %alloc_71 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %26 = arith.index_cast %1 : i32 to index
        %27 = arith.muli %arg28, %c16_i32 : i32
        %28 = arith.muli %3, %c8_i32 : i32
        %29 = arith.addi %27, %28 : i32
        %30 = arith.index_cast %29 : i32 to index
        %subview_129 = memref.subview %reinterpret_cast_13[0, %26, %30] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_129, %alloc_72 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>>
        hivm.hir.vexp ins(%alloc_72 : memref<8xf32, strided<[1]>>) outs(%alloc_72 : memref<8xf32, strided<[1]>>)
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %31 = arith.divsi %arg30, %c2_i32 : i32
            %32 = arith.muli %3, %c8_i32 : i32
            %33 = arith.divsi %arg29, %c2_i32 : i32
            %34 = arith.addi %32, %33 : i32
            %35 = arith.cmpi slt, %31, %34 : i32
            scf.if %35 {
              %36 = arith.index_cast %arg29 : i32 to index
              %37 = arith.index_cast %arg30 : i32 to index
              %38 = memref.load %alloc_69[%36, %37] : memref<16x32xf32, strided<[32, 1]>>
              %39 = arith.muli %3, %c8_i32 : i32
              %40 = arith.divsi %arg29, %c2_i32 : i32
              %41 = arith.addi %39, %40 : i32
              %42 = arith.index_cast %41 : i32 to index
              %43 = arith.divsi %arg30, %c2_i32 : i32
              %44 = arith.index_cast %43 : i32 to index
              %45 = memref.load %alloc_35[%42, %44] : memref<16x16xf32, strided<[16, 1]>>
              %46 = arith.mulf %38, %45 : f32
              %47 = arith.truncf %46 : f32 to bf16
              memref.store %47, %alloc_70[%36, %37] : memref<16x32xbf16, strided<[32, 1]>>
            } else {
              %36 = arith.index_cast %arg29 : i32 to index
              %37 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_4, %alloc_70[%36, %37] : memref<16x32xbf16, strided<[32, 1]>>
            }
          }
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %31 = arith.index_cast %arg29 : i32 to index
            %32 = arith.index_cast %arg30 : i32 to index
            %33 = memref.load %alloc_71[%31, %32] : memref<16x64xf32, strided<[64, 1]>>
            %34 = arith.divsi %arg29, %c2_i32 : i32
            %35 = arith.index_cast %34 : i32 to index
            %36 = memref.load %alloc_72[%35] : memref<8xf32, strided<[1]>>
            %37 = arith.mulf %33, %36 : f32
            memref.store %37, %alloc_71[%31, %32] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %subview_130 = memref.subview %7[0, %25, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_70, %subview_130 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        %subview_131 = memref.subview %5[0, %25, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %alloc_71, %subview_131 : memref<16x64xf32, strided<[64, 1]>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      }
      %subview_125 = memref.subview %7[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_125, %alloc_34 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 ins(%alloc_34, %alloc_30, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_73 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_126 = memref.subview %8[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_73, %subview_126 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.if %23 {
        %24 = arith.muli %3, %c16_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %subview_127 = memref.subview %5[0, %25, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_127, %alloc_74 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_128 = memref.subview %8[0, %25, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_128, %alloc_75 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_129 = memref.subview %4[0, %25, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_129, %alloc_76 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %26 = arith.muli %arg28, %c16_i32 : i32
        %27 = arith.muli %3, %c8_i32 : i32
        %28 = arith.addi %26, %27 : i32
        %29 = arith.index_cast %28 : i32 to index
        %30 = arith.index_cast %1 : i32 to index
        %subview_130 = memref.subview %reinterpret_cast_14[0, %29, %30, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_130, %alloc_77 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %31 = arith.index_cast %arg29 : i32 to index
            %32 = arith.index_cast %arg30 : i32 to index
            %33 = memref.load %alloc_74[%31, %32] : memref<16x64xf32, strided<[64, 1]>>
            %34 = memref.load %alloc_75[%31, %32] : memref<16x64xf32, strided<[64, 1]>>
            %35 = arith.addf %33, %34 : f32
            memref.store %35, %alloc_74[%31, %32] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        scf.for %arg29 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %31 = arith.index_cast %arg29 : i32 to index
            %32 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_4, %alloc_78[%31, %32] : memref<8x64xbf16, strided<[64, 1]>>
          }
        }
        %subview_131 = memref.subview %reinterpret_cast_18[0, %29, %30, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_78, %subview_131 : memref<8x64xbf16, strided<[64, 1]>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
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
    %9 = memref_ext.alloc_workspace() : memref<32x64xbf16>
    annotation.mark %9 {hivm.multi_buffer = 2 : i32} : memref<32x64xbf16>
    %10 = memref_ext.alloc_workspace() : memref<32x64xbf16>
    annotation.mark %10 {hivm.multi_buffer = 2 : i32} : memref<32x64xbf16>
    %11 = memref_ext.alloc_workspace() : memref<16x2x64xbf16>
    annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<16x2x64xbf16>
    %12 = memref_ext.alloc_workspace() : memref<16x2x32xbf16>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<16x2x32xbf16>
    %13 = memref_ext.alloc_workspace() : memref<16x2x32xbf16>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<16x2x32xbf16>
    %14 = memref_ext.alloc_workspace() : memref<32x64xbf16>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<32x64xbf16>
    %15 = memref_ext.alloc_workspace() : memref<32x32xbf16>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<32x32xbf16>
    %alloc = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_21 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
    %alloc_22 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_23 = memref.alloc() : memref<2x32xf32, strided<[32, 1]>>
    %alloc_24 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
    %alloc_25 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
    %alloc_26 = memref.alloc() : memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vbrc ins(%cst_7 : f32) outs(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>)
    %16 = arith.index_cast %1 : i32 to index
    %subview = memref.subview %reinterpret_cast_10[%16, 0, 0] [1, 2, 64] [1, 1, 1] : memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_26 : memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x64xf32, strided<[64, 1]>>
    hivm.hir.vcast ins(%alloc_26 : memref<2x64xf32, strided<[64, 1]>>) outs(%alloc_25 : memref<2x64xbf16, strided<[64, 1]>>)
    %subview_27 = memref.subview %reinterpret_cast_11[%16, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_27, %alloc_22 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_22 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc : memref<2x32xbf16, strided<[32, 1]>>)
    %subview_28 = memref.subview %reinterpret_cast_15[%16, 0, 0] [1, 2, 32] [1, 1, 1] : memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_28, %alloc_23 : memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<2x32xf32, strided<[32, 1]>>
    hivm.hir.vcast ins(%alloc_23 : memref<2x32xf32, strided<[32, 1]>>) outs(%alloc_21 : memref<2x32xbf16, strided<[32, 1]>>)
    scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %alloc_29 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_30 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_31 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_32 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
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
      %alloc_50 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_51 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_52 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_53 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_54 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_55 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_56 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_57 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_58 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_59 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_60 = memref.alloc() : memref<16x8xf32, strided<[8, 1]>>
      %alloc_61 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_62 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_63 = memref.alloc() : memref<32x64xbf16, strided<[64, 1]>>
      %alloc_64 = memref.alloc() : memref<16x2x8xbf16, strided<[16, 8, 1]>>
      %alloc_65 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_66 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>>
      %alloc_67 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_68 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>>
      %alloc_69 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_70 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
      %alloc_71 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_72 = memref.alloc() : memref<8xf32, strided<[1]>>
      %alloc_73 = memref.alloc() : memref<32x64xf32, strided<[64, 1]>>
      %alloc_74 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_75 = memref.alloc() : memref<16x64xf32, strided<[64, 1]>>
      %alloc_76 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
      %alloc_77 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_78 = memref.alloc() : memref<8x64xbf16, strided<[64, 1]>>
      %alloc_79 = memref.alloc() : memref<16x64xbf16, strided<[64, 1]>>
      %alloc_80 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_81 = memref.alloc() : memref<2x64xbf16, strided<[64, 1]>>
      %alloc_82 = memref.alloc() : memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %alloc_83 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_84 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_85 = memref.alloc() : memref<2x32xbf16, strided<[32, 1]>>
      %alloc_86 = memref.alloc() : memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %alloc_87 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_88 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_89 = memref.alloc() : memref<16x2x64xf32, strided<[128, 64, 1]>>
      %alloc_90 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_91 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_92 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_93 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_94 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %alloc_95 = memref.alloc() : memref<16x2x32xf32, strided<[64, 32, 1]>>
      %17 = arith.index_cast %1 : i32 to index
      %18 = arith.index_cast %arg28 : i32 to index
      %subview_96 = memref.subview %reinterpret_cast_20[0, %17, %18, 0, 0] [1, 1, 1, 16, 16] [1, 1, 1, 1, 1] : memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_96, %alloc_35 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1]>>
      %19 = arith.muli %arg28, %c16_i32 : i32
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = arith.index_cast %20 : i32 to index
      %subview_97 = memref.subview %reinterpret_cast_16[0, %17, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_97, %alloc_37 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_37 : memref<16xbf16, strided<[1]>>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      hivm.hir.vmul ins(%alloc_36, %cst_8 : memref<16xf32, strided<[1]>>, f32) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %alloc_98 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_36 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_98 : memref<16xf32>) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_98, %cst_6 : memref<16xf32>, f32) outs(%alloc_98 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_98 : f32, memref<16xf32>) outs(%alloc_36 : memref<16xf32, strided<[1]>>)
      %subview_99 = memref.subview %reinterpret_cast_19[0, %17, %21] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_99, %alloc_39 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_39 : memref<16xf32, strided<[1]>>) outs(%alloc_38 : memref<16xbf16, strided<[1]>>)
      hivm.hir.vbrc ins(%cst_4 : bf16) outs(%alloc_40 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %24 = arith.muli %arg28, %c16_i32 : i32
        %25 = arith.addi %24, %arg29 : i32
        %26 = arith.cmpi slt, %25, %c31_i32 : i32
        scf.if %26 {
          %27 = arith.index_cast %arg29 : i32 to index
          %28 = memref.load %alloc_38[%27] : memref<16xbf16, strided<[1]>>
          %29 = arith.extf %28 : bf16 to f32
          %30 = memref.load %alloc_36[%27] : memref<16xf32, strided<[1]>>
          %31 = arith.mulf %29, %30 : f32
          %32 = arith.truncf %31 : f32 to bf16
          memref.store %32, %alloc_40[%27] : memref<16xbf16, strided<[1]>>
        }
      }
      memref.copy %alloc_40, %alloc_41 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      %22 = arith.index_cast %19 : i32 to index
      %subview_100 = memref.subview %reinterpret_cast_16[0, %17, %22] [1, 1, 16] [1, 1, 1] : memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_100, %alloc_43 : memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1]>>
      hivm.hir.vcast ins(%alloc_43 : memref<16xbf16, strided<[1]>>) outs(%alloc_42 : memref<16xf32, strided<[1]>>)
      %alloc_101 = memref.alloc() : memref<16xf32>
      hivm.hir.vsub ins(%cst_7, %alloc_42 : f32, memref<16xf32, strided<[1]>>) outs(%alloc_101 : memref<16xf32>)
      hivm.hir.vexp ins(%alloc_101 : memref<16xf32>) outs(%alloc_101 : memref<16xf32>)
      hivm.hir.vadd ins(%alloc_101, %cst_6 : memref<16xf32>, f32) outs(%alloc_101 : memref<16xf32>)
      hivm.hir.vdiv ins(%cst_6, %alloc_101 : f32, memref<16xf32>) outs(%alloc_42 : memref<16xf32, strided<[1]>>)
      %subview_102 = memref.subview %reinterpret_cast_19[0, %17, %22] [1, 1, 16] [1, 1, 1] : memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_102, %alloc_45 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<16xf32, strided<[1]>>
      hivm.hir.vcast ins(%alloc_45 : memref<16xf32, strided<[1]>>) outs(%alloc_44 : memref<16xbf16, strided<[1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %24 = arith.index_cast %arg29 : i32 to index
        %25 = memref.load %alloc_44[%24] : memref<16xbf16, strided<[1]>>
        %26 = arith.extf %25 : bf16 to f32
        %27 = memref.load %alloc_42[%24] : memref<16xf32, strided<[1]>>
        %28 = arith.mulf %26, %27 : f32
        memref.store %28, %alloc_46[%24] : memref<16xf32, strided<[1]>>
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %24 = arith.index_cast %arg29 : i32 to index
        %25 = memref.load %alloc_46[%24] : memref<16xf32, strided<[1]>>
        %26 = memref.load %alloc_41[%24] : memref<16xbf16, strided<[1]>>
        %27 = arith.extf %26 : bf16 to f32
        %28 = arith.addf %25, %27 : f32
        %29 = arith.truncf %28 : f32 to bf16
        memref.store %29, %alloc_47[%24] : memref<16xbf16, strided<[1]>>
      }
      memref.copy %alloc_47, %alloc_48 : memref<16xbf16, strided<[1]>> to memref<16xbf16, strided<[1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %24 = arith.muli %arg28, %c16_i32 : i32
        %25 = arith.addi %24, %arg29 : i32
        %26 = arith.index_cast %25 : i32 to index
        %27 = arith.index_cast %1 : i32 to index
        %subview_127 = memref.subview %reinterpret_cast_9[0, %26, %27, 0] [1, 1, 1, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %28 = arith.index_cast %arg29 : i32 to index
        %subview_128 = memref.subview %alloc_31[%28, 0] [1, 64] [1, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<64xbf16, strided<[1], offset: ?>>
        memref.copy %subview_127, %subview_128 : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xbf16, strided<[1], offset: ?>>
      }
      memref.copy %alloc_31, %alloc_79 : memref<16x64xbf16, strided<[64, 1]>> to memref<16x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_103 = memref.reinterpret_cast %alloc_79 to offset: [0], sizes: [16, 1, 64], strides: [64, 64, 1] : memref<16x64xbf16, strided<[64, 1]>> to memref<16x1x64xbf16, strided<[64, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_103 : memref<16x1x64xbf16, strided<[64, 64, 1]>>) outs(%alloc_80 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [1]
      memref.copy %alloc_25, %alloc_81 : memref<2x64xbf16, strided<[64, 1]>> to memref<2x64xbf16, strided<[64, 1]>>
      %reinterpret_cast_104 = memref.reinterpret_cast %alloc_81 to offset: [0], sizes: [1, 2, 64], strides: [128, 64, 1] : memref<2x64xbf16, strided<[64, 1]>> to memref<1x2x64xbf16, strided<[128, 64, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_104 : memref<1x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_82 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_80 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_87 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_82 : memref<16x2x64xbf16, strided<[128, 64, 1]>>) outs(%alloc_88 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vmul ins(%alloc_87, %alloc_88 : memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_89 : memref<16x2x64xf32, strided<[128, 64, 1]>>)
      hivm.hir.vcast ins(%alloc_89 : memref<16x2x64xf32, strided<[128, 64, 1]>>) outs(%alloc_49 : memref<16x2x64xbf16, strided<[128, 64, 1]>>)
      memref.copy %alloc_49, %11 : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<16x2x64xbf16>
      memref.copy %11, %alloc_50 : memref<16x2x64xbf16> to memref<16x2x64xbf16, strided<[128, 64, 1]>>
      %reinterpret_cast_105 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<16x2x64xbf16, strided<[128, 64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %reinterpret_cast_105, %alloc_29 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %alloc_29, %9 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16>
      memref.copy %9, %alloc_30 : memref<32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      %subview_106 = memref.subview %reinterpret_cast[0, %22, 0, %17, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_106, %alloc_51 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc, %alloc_83 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_107 = memref.reinterpret_cast %alloc_83 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_107 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_84 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_90 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_84 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_90, %alloc_91 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_92 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_51 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      memref.copy %alloc_51, %12 : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<16x2x32xbf16>
      memref.copy %12, %alloc_52 : memref<16x2x32xbf16> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %reinterpret_cast_108 = memref.reinterpret_cast %alloc_51 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      %subview_109 = memref.subview %reinterpret_cast_17[0, %22, 0, %17, 0] [1, 16, 2, 1, 32] [1, 1, 1, 1, 1] : memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_109, %alloc_53 : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      memref.copy %alloc_21, %alloc_85 : memref<2x32xbf16, strided<[32, 1]>> to memref<2x32xbf16, strided<[32, 1]>>
      %reinterpret_cast_110 = memref.reinterpret_cast %alloc_85 to offset: [0], sizes: [1, 2, 32], strides: [64, 32, 1] : memref<2x32xbf16, strided<[32, 1]>> to memref<1x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.vbrc ins(%reinterpret_cast_110 : memref<1x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_86 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) broadcast_dims = [0]
      hivm.hir.vcast ins(%alloc_53 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_93 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_86 : memref<16x2x32xbf16, strided<[64, 32, 1]>>) outs(%alloc_94 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vadd ins(%alloc_93, %alloc_94 : memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_95 : memref<16x2x32xf32, strided<[64, 32, 1]>>)
      hivm.hir.vcast ins(%alloc_95 : memref<16x2x32xf32, strided<[64, 32, 1]>>) outs(%alloc_53 : memref<16x2x32xbf16, strided<[64, 32, 1]>>)
      memref.copy %alloc_53, %13 : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<16x2x32xbf16>
      memref.copy %13, %alloc_55 : memref<16x2x32xbf16> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      %reinterpret_cast_111 = memref.reinterpret_cast %alloc_53 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%reinterpret_cast_108, %reinterpret_cast_111, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_56 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_112 = memref.subview %4[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_56, %subview_112 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %24 = arith.muli %arg29, %c2_i32 : i32
            %25 = arith.addi %24, %arg30 : i32
            %26 = arith.index_cast %25 : i32 to index
            %27 = arith.index_cast %arg31 : i32 to index
            %28 = memref.load %reinterpret_cast_108[%26, %27] : memref<32x32xbf16, strided<[32, 1]>>
            %29 = arith.index_cast %arg29 : i32 to index
            %30 = arith.index_cast %arg30 : i32 to index
            memref.store %28, %alloc_57[%29, %30, %27] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      %subview_113 = memref.subview %reinterpret_cast_12[0, %22, %17, 0] [1, 16, 1, 8] [1, 1, 1, 1] : memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_113, %alloc_58 : memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[8, 1]>>
      %alloc_114 = memref.alloc() : memref<16x8xf32>
      %alloc_115 = memref.alloc() : memref<16x8xf32>
      %alloc_116 = memref.alloc() : memref<16x8xf32>
      %alloc_117 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_58, %alloc_58 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_114 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_114, %alloc_114 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_115 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_114, %alloc_115 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_116 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_114, %cst_5 : memref<16x8xf32>, f32) outs(%alloc_114 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_115, %cst_3 : memref<16x8xf32>, f32) outs(%alloc_115 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_116, %cst_2 : memref<16x8xf32>, f32) outs(%alloc_116 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_114, %cst_6 : memref<16x8xf32>, f32) outs(%alloc_117 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_115, %alloc_117 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_117 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_116, %alloc_117 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_59 : memref<16x8xf32, strided<[8, 1]>>)
      %alloc_118 = memref.alloc() : memref<16x8xf32>
      %alloc_119 = memref.alloc() : memref<16x8xf32>
      %alloc_120 = memref.alloc() : memref<16x8xf32>
      %alloc_121 = memref.alloc() : memref<16x8xf32>
      %alloc_122 = memref.alloc() : memref<16x8xf32>
      hivm.hir.vmul ins(%alloc_58, %alloc_58 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_118 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_118, %alloc_58 : memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) outs(%alloc_119 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_119, %alloc_118 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_120 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_120, %alloc_118 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_121 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_119, %cst_1 : memref<16x8xf32>, f32) outs(%alloc_119 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_120, %cst_0 : memref<16x8xf32>, f32) outs(%alloc_120 : memref<16x8xf32>)
      hivm.hir.vmul ins(%alloc_121, %cst : memref<16x8xf32>, f32) outs(%alloc_121 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_58, %alloc_119 : memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) outs(%alloc_122 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_120, %alloc_122 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_122 : memref<16x8xf32>)
      hivm.hir.vadd ins(%alloc_121, %alloc_122 : memref<16x8xf32>, memref<16x8xf32>) outs(%alloc_60 : memref<16x8xf32, strided<[8, 1]>>)
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg31 : i32 to index
            %26 = memref.load %alloc_59[%24, %25] : memref<16x8xf32, strided<[8, 1]>>
            %27 = arith.index_cast %arg30 : i32 to index
            %28 = memref.load %alloc_57[%24, %27, %25] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %29 = arith.extf %28 : bf16 to f32
            %30 = arith.mulf %26, %29 : f32
            %31 = arith.subf %30, %30 : f32
            %32 = arith.truncf %31 : f32 to bf16
            %33 = arith.muli %arg29, %c2_i32 : i32
            %34 = arith.addi %33, %arg30 : i32
            %35 = arith.index_cast %34 : i32 to index
            memref.store %32, %reinterpret_cast_108[%35, %25] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg31 : i32 to index
            %26 = memref.load %alloc_60[%24, %25] : memref<16x8xf32, strided<[8, 1]>>
            %27 = arith.index_cast %arg30 : i32 to index
            %28 = memref.load %alloc_57[%24, %27, %25] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %29 = arith.extf %28 : bf16 to f32
            %30 = arith.mulf %26, %29 : f32
            %31 = arith.addf %30, %30 : f32
            %32 = arith.truncf %31 : f32 to bf16
            %33 = arith.muli %arg29, %c2_i32 : i32
            %34 = arith.addi %33, %arg30 : i32
            %35 = arith.index_cast %34 : i32 to index
            %36 = arith.addi %arg31, %c16_i32 : i32
            %37 = arith.index_cast %36 : i32 to index
            memref.store %32, %reinterpret_cast_108[%35, %37] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      hivm.hir.vcast ins(%alloc_24 : memref<32x64xf32, strided<[64, 1]>>) outs(%alloc_62 : memref<32x64xbf16, strided<[64, 1]>>)
      memref.copy %alloc_62, %14 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16>
      memref.copy %14, %alloc_63 : memref<32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %alloc_63, %alloc_32 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16, strided<[64, 1]>>
      memref.copy %alloc_32, %10 : memref<32x64xbf16, strided<[64, 1]>> to memref<32x64xbf16>
      memref.copy %10, %alloc_33 : memref<32x64xbf16> to memref<32x64xbf16, strided<[64, 1]>>
      hivm.hir.mmadL1 ins(%reinterpret_cast_108, %alloc_33, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_61 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_123 = memref.subview %5[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_61, %subview_123 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %24 = arith.muli %arg29, %c2_i32 : i32
            %25 = arith.addi %24, %arg30 : i32
            %26 = arith.index_cast %25 : i32 to index
            %27 = arith.index_cast %arg31 : i32 to index
            %28 = memref.load %reinterpret_cast_111[%26, %27] : memref<32x32xbf16, strided<[32, 1]>>
            %29 = arith.index_cast %arg29 : i32 to index
            %30 = arith.index_cast %arg30 : i32 to index
            memref.store %28, %alloc_64[%29, %30, %27] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg31 : i32 to index
            %26 = memref.load %alloc_59[%24, %25] : memref<16x8xf32, strided<[8, 1]>>
            %27 = arith.index_cast %arg30 : i32 to index
            %28 = memref.load %alloc_64[%24, %27, %25] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %29 = arith.extf %28 : bf16 to f32
            %30 = arith.mulf %26, %29 : f32
            %31 = arith.subf %30, %30 : f32
            %32 = arith.truncf %31 : f32 to bf16
            %33 = arith.muli %arg29, %c2_i32 : i32
            %34 = arith.addi %33, %arg30 : i32
            %35 = arith.index_cast %34 : i32 to index
            memref.store %32, %reinterpret_cast_111[%35, %25] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          scf.for %arg31 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %24 = arith.index_cast %arg29 : i32 to index
            %25 = arith.index_cast %arg31 : i32 to index
            %26 = memref.load %alloc_60[%24, %25] : memref<16x8xf32, strided<[8, 1]>>
            %27 = arith.index_cast %arg30 : i32 to index
            %28 = memref.load %alloc_64[%24, %27, %25] : memref<16x2x8xbf16, strided<[16, 8, 1]>>
            %29 = arith.extf %28 : bf16 to f32
            %30 = arith.mulf %26, %29 : f32
            %31 = arith.addf %30, %30 : f32
            %32 = arith.truncf %31 : f32 to bf16
            %33 = arith.muli %arg29, %c2_i32 : i32
            %34 = arith.addi %33, %arg30 : i32
            %35 = arith.index_cast %34 : i32 to index
            %36 = arith.addi %arg31, %c16_i32 : i32
            %37 = arith.index_cast %36 : i32 to index
            memref.store %32, %reinterpret_cast_111[%35, %37] : memref<32x32xbf16, strided<[32, 1]>>
          }
        }
      }
      memref.copy %reinterpret_cast_111, %alloc_65 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.vcast ins(%alloc_65 : memref<32x32xbf16, strided<[32, 1]>>) outs(%alloc_67 : memref<32x32xf32, strided<[32, 1]>>)
      scf.for %arg29 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
          %24 = arith.index_cast %arg29 : i32 to index
          %25 = arith.index_cast %arg30 : i32 to index
          %26 = memref.load %alloc_67[%24, %25] : memref<32x32xf32, strided<[32, 1]>>
          %27 = arith.divsi %arg29, %c2_i32 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = memref.load %alloc_48[%28] : memref<16xbf16, strided<[1]>>
          %30 = arith.extf %29 : bf16 to f32
          %31 = arith.mulf %26, %30 : f32
          memref.store %31, %alloc_67[%24, %25] : memref<32x32xf32, strided<[32, 1]>>
        }
      }
      hivm.hir.vcast ins(%alloc_67 : memref<32x32xf32, strided<[32, 1]>>) outs(%alloc_65 : memref<32x32xbf16, strided<[32, 1]>>)
      memref.copy %alloc_65, %15 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16>
      memref.copy %15, %alloc_66 : memref<32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_66, %reinterpret_cast_111 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %alloc_55, %13 : memref<16x2x32xbf16, strided<[64, 32, 1]>> to memref<16x2x32xbf16>
      memref.copy %13, %alloc_54 : memref<16x2x32xbf16> to memref<16x2x32xbf16, strided<[64, 32, 1]>>
      hivm.hir.mmadL1 {b_transpose} ins(%reinterpret_cast_108, %reinterpret_cast_111, %true, %c32, %c32, %c32 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index) outs(%alloc_68 : memref<32x32xf32, strided<[32, 1]>>)
      %subview_124 = memref.subview %6[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
      memref.copy %alloc_68, %subview_124 : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32, strided<[32, 1]>>
      hivm.hir.vexp ins(%alloc_35 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_35 : memref<16x16xf32, strided<[16, 1]>>)
      %23 = arith.cmpi slt, %3, %c2_i32 : i32
      scf.if %23 {
        %24 = arith.muli %3, %c16_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %subview_127 = memref.subview %6[0, %25, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_127, %alloc_69 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %subview_128 = memref.subview %5[0, %25, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_128, %alloc_71 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %26 = arith.index_cast %1 : i32 to index
        %27 = arith.muli %arg28, %c16_i32 : i32
        %28 = arith.muli %3, %c8_i32 : i32
        %29 = arith.addi %27, %28 : i32
        %30 = arith.index_cast %29 : i32 to index
        %subview_129 = memref.subview %reinterpret_cast_13[0, %26, %30] [1, 1, 8] [1, 1, 1] : memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_129, %alloc_72 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>>
        hivm.hir.vexp ins(%alloc_72 : memref<8xf32, strided<[1]>>) outs(%alloc_72 : memref<8xf32, strided<[1]>>)
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
            %31 = arith.divsi %arg30, %c2_i32 : i32
            %32 = arith.muli %3, %c8_i32 : i32
            %33 = arith.divsi %arg29, %c2_i32 : i32
            %34 = arith.addi %32, %33 : i32
            %35 = arith.cmpi slt, %31, %34 : i32
            scf.if %35 {
              %36 = arith.index_cast %arg29 : i32 to index
              %37 = arith.index_cast %arg30 : i32 to index
              %38 = memref.load %alloc_69[%36, %37] : memref<16x32xf32, strided<[32, 1]>>
              %39 = arith.muli %3, %c8_i32 : i32
              %40 = arith.divsi %arg29, %c2_i32 : i32
              %41 = arith.addi %39, %40 : i32
              %42 = arith.index_cast %41 : i32 to index
              %43 = arith.divsi %arg30, %c2_i32 : i32
              %44 = arith.index_cast %43 : i32 to index
              %45 = memref.load %alloc_35[%42, %44] : memref<16x16xf32, strided<[16, 1]>>
              %46 = arith.mulf %38, %45 : f32
              %47 = arith.truncf %46 : f32 to bf16
              memref.store %47, %alloc_70[%36, %37] : memref<16x32xbf16, strided<[32, 1]>>
            } else {
              %36 = arith.index_cast %arg29 : i32 to index
              %37 = arith.index_cast %arg30 : i32 to index
              memref.store %cst_4, %alloc_70[%36, %37] : memref<16x32xbf16, strided<[32, 1]>>
            }
          }
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %31 = arith.index_cast %arg29 : i32 to index
            %32 = arith.index_cast %arg30 : i32 to index
            %33 = memref.load %alloc_71[%31, %32] : memref<16x64xf32, strided<[64, 1]>>
            %34 = arith.divsi %arg29, %c2_i32 : i32
            %35 = arith.index_cast %34 : i32 to index
            %36 = memref.load %alloc_72[%35] : memref<8xf32, strided<[1]>>
            %37 = arith.mulf %33, %36 : f32
            memref.store %37, %alloc_71[%31, %32] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        %subview_130 = memref.subview %7[0, %25, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_70, %subview_130 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1], offset: ?>>
        %subview_131 = memref.subview %5[0, %25, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %alloc_71, %subview_131 : memref<16x64xf32, strided<[64, 1]>> to memref<16x64xf32, strided<[64, 1], offset: ?>>
      }
      %subview_125 = memref.subview %7[0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
      memref.copy %subview_125, %alloc_34 : memref<32x32xbf16, strided<[32, 1]>> to memref<32x32xbf16, strided<[32, 1]>>
      hivm.hir.mmadL1 ins(%alloc_34, %alloc_30, %true, %c32, %c32, %c64 : memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index) outs(%alloc_73 : memref<32x64xf32, strided<[64, 1]>>)
      %subview_126 = memref.subview %8[0, 0, 0] [1, 32, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<32x64xf32, strided<[64, 1]>>
      memref.copy %alloc_73, %subview_126 : memref<32x64xf32, strided<[64, 1]>> to memref<32x64xf32, strided<[64, 1]>>
      scf.if %23 {
        %24 = arith.muli %3, %c16_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %subview_127 = memref.subview %5[0, %25, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_127, %alloc_74 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_128 = memref.subview %8[0, %25, 0] [1, 16, 64] [1, 1, 1] : memref<2x32x64xf32> to memref<16x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %subview_128, %alloc_75 : memref<16x64xf32, strided<[64, 1], offset: ?>> to memref<16x64xf32, strided<[64, 1]>>
        %subview_129 = memref.subview %4[0, %25, 0] [1, 16, 32] [1, 1, 1] : memref<2x32x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_129, %alloc_76 : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<16x32xf32, strided<[32, 1]>>
        %26 = arith.muli %arg28, %c16_i32 : i32
        %27 = arith.muli %3, %c8_i32 : i32
        %28 = arith.addi %26, %27 : i32
        %29 = arith.index_cast %28 : i32 to index
        %30 = arith.index_cast %1 : i32 to index
        %subview_130 = memref.subview %reinterpret_cast_14[0, %29, %30, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_130, %alloc_77 : memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1]>>
        scf.for %arg29 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %31 = arith.index_cast %arg29 : i32 to index
            %32 = arith.index_cast %arg30 : i32 to index
            %33 = memref.load %alloc_74[%31, %32] : memref<16x64xf32, strided<[64, 1]>>
            %34 = memref.load %alloc_75[%31, %32] : memref<16x64xf32, strided<[64, 1]>>
            %35 = arith.addf %33, %34 : f32
            memref.store %35, %alloc_74[%31, %32] : memref<16x64xf32, strided<[64, 1]>>
          }
        }
        scf.for %arg29 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          scf.for %arg30 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
            %31 = arith.index_cast %arg29 : i32 to index
            %32 = arith.index_cast %arg30 : i32 to index
            memref.store %cst_4, %alloc_78[%31, %32] : memref<8x64xbf16, strided<[64, 1]>>
          }
        }
        %subview_131 = memref.subview %reinterpret_cast_18[0, %29, %30, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %alloc_78, %subview_131 : memref<8x64xbf16, strided<[64, 1]>> to memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
      }
    } {tilelangir.num_stages = 2 : i32}
    return
  }
}


loc("input.mlir":587:17): error: operand #0 does not dominate this use
// -----// IR Dump After TileLangIRCVSplit Failed (tilelangir-cv-split) ('func.func' operation: @mamba_mimo_fwd_kernel) //----- //
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{hacc.arg_type = #hacc.arg_type<ffts_base_address>}, {}, {hacc.arg_type = #hacc.arg_type<workspace>}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}], function_type = (i64, memref<?xi8>, memref<?xi8>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, i32, i32, i32, i32, i32, i32) -> (), sym_name = "mamba_mimo_fwd_kernel"}> ({
  ^bb0(%arg0: i64, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xbf16, #hivm.address_space<gm>>, %arg7: memref<?xf32, #hivm.address_space<gm>>, %arg8: memref<?xf32, #hivm.address_space<gm>>, %arg9: memref<?xf32, #hivm.address_space<gm>>, %arg10: memref<?xf32, #hivm.address_space<gm>>, %arg11: memref<?xbf16, #hivm.address_space<gm>>, %arg12: memref<?xf32, #hivm.address_space<gm>>, %arg13: memref<?xf32, #hivm.address_space<gm>>, %arg14: memref<?xf32, #hivm.address_space<gm>>, %arg15: memref<?xf32, #hivm.address_space<gm>>, %arg16: memref<?xf32, #hivm.address_space<gm>>, %arg17: memref<?xf32, #hivm.address_space<gm>>, %arg18: memref<?xbf16, #hivm.address_space<gm>>, %arg19: memref<?xf32, #hivm.address_space<gm>>, %arg20: memref<?xf32, #hivm.address_space<gm>>, %arg21: memref<?xbf16, #hivm.address_space<gm>>, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32):
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
    %49 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x32x32xf32>
    "annotation.mark"(%49) {hivm.multi_buffer = 2 : i32} : (memref<2x32x32xf32>) -> ()
    %50 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x32x64xf32>
    "annotation.mark"(%50) {hivm.multi_buffer = 2 : i32} : (memref<2x32x64xf32>) -> ()
    %51 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x32x32xf32>
    "annotation.mark"(%51) {hivm.multi_buffer = 2 : i32} : (memref<2x32x32xf32>) -> ()
    %52 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x32x32xbf16>
    "annotation.mark"(%52) {hivm.multi_buffer = 2 : i32} : (memref<2x32x32xbf16>) -> ()
    %53 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x32x64xf32>
    "annotation.mark"(%53) {hivm.multi_buffer = 2 : i32} : (memref<2x32x64xf32>) -> ()
    %54 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<32x64xbf16>
    "annotation.mark"(%54) {hivm.multi_buffer = 2 : i32} : (memref<32x64xbf16>) -> ()
    %55 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<32x64xbf16>
    "annotation.mark"(%55) {hivm.multi_buffer = 2 : i32} : (memref<32x64xbf16>) -> ()
    %56 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<16x2x64xbf16>
    "annotation.mark"(%56) {hivm.multi_buffer = 2 : i32} : (memref<16x2x64xbf16>) -> ()
    %57 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<16x2x32xbf16>
    "annotation.mark"(%57) {hivm.multi_buffer = 2 : i32} : (memref<16x2x32xbf16>) -> ()
    %58 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<16x2x32xbf16>
    "annotation.mark"(%58) {hivm.multi_buffer = 2 : i32} : (memref<16x2x32xbf16>) -> ()
    %59 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<32x64xbf16>
    "annotation.mark"(%59) {hivm.multi_buffer = 2 : i32} : (memref<32x64xbf16>) -> ()
    %60 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<32x32xbf16>
    "annotation.mark"(%60) {hivm.multi_buffer = 2 : i32} : (memref<32x32xbf16>) -> ()
    %61 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32xbf16, strided<[32, 1]>>
    %62 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32xbf16, strided<[32, 1]>>
    %63 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32xf32, strided<[32, 1]>>
    %64 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32xf32, strided<[32, 1]>>
    %65 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32, strided<[64, 1]>>
    %66 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x64xbf16, strided<[64, 1]>>
    %67 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x64xf32, strided<[64, 1]>>
    "hivm.hir.vbrc"(%23, %65) <{broadcast_dims = array<i64>}> : (f32, memref<32x64xf32, strided<[64, 1]>>) -> ()
    %68 = "arith.index_cast"(%46) : (i32) -> index
    %69 = "memref.subview"(%34, %68) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 2, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x2x64xf32, strided<[128, 64, 1]>, #hivm.address_space<gm>>, index) -> memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    "memref.copy"(%69, %67) : (memref<2x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>, memref<2x64xf32, strided<[64, 1]>>) -> ()
    "hivm.hir.vcast"(%67, %66) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<2x64xf32, strided<[64, 1]>>, memref<2x64xbf16, strided<[64, 1]>>) -> ()
    %70 = "memref.subview"(%35, %68) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 2, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>, index) -> memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    "memref.copy"(%70, %63) : (memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<2x32xf32, strided<[32, 1]>>) -> ()
    "hivm.hir.vcast"(%63, %61) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<2x32xf32, strided<[32, 1]>>, memref<2x32xbf16, strided<[32, 1]>>) -> ()
    %71 = "memref.subview"(%39, %68) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 2, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x2x32xf32, strided<[64, 32, 1]>, #hivm.address_space<gm>>, index) -> memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
    "memref.copy"(%71, %64) : (memref<2x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<2x32xf32, strided<[32, 1]>>) -> ()
    "hivm.hir.vcast"(%64, %62) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<2x32xf32, strided<[32, 1]>>, memref<2x32xbf16, strided<[32, 1]>>) -> ()
    "scf.for"(%25, %29, %31) ({
    ^bb0(%arg28: i32):
      %72 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x16xf32, strided<[16, 1]>>
      %73 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32, strided<[1]>>
      %74 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>>
      %75 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>>
      %76 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32, strided<[1]>>
      %77 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>>
      %78 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>>
      %79 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32, strided<[1]>>
      %80 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>>
      %81 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>>
      %82 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32, strided<[1]>>
      %83 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32, strided<[1]>>
      %84 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>>
      %85 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xbf16, strided<[1]>>
      %86 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x32xf32, strided<[32, 1]>>
      %87 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x32xbf16, strided<[32, 1]>>
      %88 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x64xf32, strided<[64, 1]>>
      %89 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8xf32, strided<[1]>>
      %90 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x64xf32, strided<[64, 1]>>
      %91 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x64xf32, strided<[64, 1]>>
      %92 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x32xf32, strided<[32, 1]>>
      %93 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x64xbf16, strided<[64, 1]>>
      %94 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x64xbf16, strided<[64, 1]>>
      %95 = "arith.index_cast"(%46) : (i32) -> index
      %96 = "arith.index_cast"(%arg28) : (i32) -> index
      %97 = "memref.subview"(%44, %95, %96) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 1, 1, 16, 16>, static_strides = array<i64: 1, 1, 1, 1, 1>}> : (memref<1x4x2x16x16xf32, strided<[2048, 512, 256, 16, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%97, %72) : (memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x16xf32, strided<[16, 1]>>) -> ()
      %98 = "arith.muli"(%arg28, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %99 = "arith.addi"(%98, %31) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %100 = "arith.index_cast"(%99) : (i32) -> index
      %101 = "memref.subview"(%40, %95, %100) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 1, 1, 16>, static_strides = array<i64: 1, 1, 1>}> : (memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%101, %74) : (memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<16xbf16, strided<[1]>>) -> ()
      "hivm.hir.vcast"(%74, %73) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16xbf16, strided<[1]>>, memref<16xf32, strided<[1]>>) -> ()
      "hivm.hir.vmul"(%73, %24, %73) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16xf32, strided<[1]>>, f32, memref<16xf32, strided<[1]>>) -> ()
      %102 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32>
      "hivm.hir.vsub"(%23, %73, %102) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (f32, memref<16xf32, strided<[1]>>, memref<16xf32>) -> ()
      "hivm.hir.vexp"(%102, %102) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<16xf32>, memref<16xf32>) -> ()
      "hivm.hir.vadd"(%102, %22, %102) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16xf32>, f32, memref<16xf32>) -> ()
      "hivm.hir.vdiv"(%22, %102, %73) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (f32, memref<16xf32>, memref<16xf32, strided<[1]>>) -> ()
      %103 = "memref.subview"(%43, %95, %100) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 1, 1, 16>, static_strides = array<i64: 1, 1, 1>}> : (memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%103, %76) : (memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<16xf32, strided<[1]>>) -> ()
      "hivm.hir.vcast"(%76, %75) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16xf32, strided<[1]>>, memref<16xbf16, strided<[1]>>) -> ()
      "hivm.hir.vbrc"(%18, %77) <{broadcast_dims = array<i64>}> : (bf16, memref<16xbf16, strided<[1]>>) -> ()
      "scf.for"(%25, %26, %31) ({
      ^bb0(%arg59: i32):
        %326 = "arith.muli"(%arg28, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %327 = "arith.addi"(%326, %arg59) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %328 = "arith.cmpi"(%327, %21) <{predicate = 2 : i64}> : (i32, i32) -> i1
        "scf.if"(%328) ({
          %329 = "arith.index_cast"(%arg59) : (i32) -> index
          %330 = "memref.load"(%75, %329) : (memref<16xbf16, strided<[1]>>, index) -> bf16
          %331 = "arith.extf"(%330) : (bf16) -> f32
          %332 = "memref.load"(%73, %329) : (memref<16xf32, strided<[1]>>, index) -> f32
          %333 = "arith.mulf"(%331, %332) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %334 = "arith.truncf"(%333) : (f32) -> bf16
          "memref.store"(%334, %77, %329) : (bf16, memref<16xbf16, strided<[1]>>, index) -> ()
          "scf.yield"() : () -> ()
        }, {
        }) : (i1) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "memref.copy"(%77, %78) : (memref<16xbf16, strided<[1]>>, memref<16xbf16, strided<[1]>>) -> ()
      %104 = "arith.index_cast"(%98) : (i32) -> index
      %105 = "memref.subview"(%40, %95, %104) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 1, 1, 16>, static_strides = array<i64: 1, 1, 1>}> : (memref<1x4x33xbf16, strided<[132, 33, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%105, %80) : (memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<16xbf16, strided<[1]>>) -> ()
      "hivm.hir.vcast"(%80, %79) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16xbf16, strided<[1]>>, memref<16xf32, strided<[1]>>) -> ()
      %106 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16xf32>
      "hivm.hir.vsub"(%23, %79, %106) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (f32, memref<16xf32, strided<[1]>>, memref<16xf32>) -> ()
      "hivm.hir.vexp"(%106, %106) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<16xf32>, memref<16xf32>) -> ()
      "hivm.hir.vadd"(%106, %22, %106) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16xf32>, f32, memref<16xf32>) -> ()
      "hivm.hir.vdiv"(%22, %106, %79) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (f32, memref<16xf32>, memref<16xf32, strided<[1]>>) -> ()
      %107 = "memref.subview"(%43, %95, %104) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 1, 1, 16>, static_strides = array<i64: 1, 1, 1>}> : (memref<1x4x33xf32, strided<[132, 33, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%107, %82) : (memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<16xf32, strided<[1]>>) -> ()
      "hivm.hir.vcast"(%82, %81) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16xf32, strided<[1]>>, memref<16xbf16, strided<[1]>>) -> ()
      "scf.for"(%25, %26, %31) ({
      ^bb0(%arg58: i32):
        %321 = "arith.index_cast"(%arg58) : (i32) -> index
        %322 = "memref.load"(%81, %321) : (memref<16xbf16, strided<[1]>>, index) -> bf16
        %323 = "arith.extf"(%322) : (bf16) -> f32
        %324 = "memref.load"(%79, %321) : (memref<16xf32, strided<[1]>>, index) -> f32
        %325 = "arith.mulf"(%323, %324) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%325, %83, %321) : (f32, memref<16xf32, strided<[1]>>, index) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "scf.for"(%25, %26, %31) ({
      ^bb0(%arg57: i32):
        %315 = "arith.index_cast"(%arg57) : (i32) -> index
        %316 = "memref.load"(%83, %315) : (memref<16xf32, strided<[1]>>, index) -> f32
        %317 = "memref.load"(%78, %315) : (memref<16xbf16, strided<[1]>>, index) -> bf16
        %318 = "arith.extf"(%317) : (bf16) -> f32
        %319 = "arith.addf"(%316, %318) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %320 = "arith.truncf"(%319) : (f32) -> bf16
        "memref.store"(%320, %84, %315) : (bf16, memref<16xbf16, strided<[1]>>, index) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "memref.copy"(%84, %85) : (memref<16xbf16, strided<[1]>>, memref<16xbf16, strided<[1]>>) -> ()
      "scf.for"(%25, %26, %31) ({
      ^bb0(%arg56: i32):
        %308 = "arith.muli"(%arg28, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %309 = "arith.addi"(%308, %arg56) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %310 = "arith.index_cast"(%309) : (i32) -> index
        %311 = "arith.index_cast"(%46) : (i32) -> index
        %312 = "memref.subview"(%33, %310, %311) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>, index, index) -> memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %313 = "arith.index_cast"(%arg56) : (i32) -> index
        %314 = "memref.subview"(%165, %313) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 1, 64>, static_strides = array<i64: 1, 1>}> : (memref<16x64xbf16, strided<[64, 1]>>, index) -> memref<64xbf16, strided<[1], offset: ?>>
        "memref.copy"(%312, %314) : (memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<64xbf16, strided<[1], offset: ?>>) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "scf.for"(%25, %26, %31) ({
      ^bb0(%arg53: i32):
        "scf.for"(%25, %29, %31) ({
        ^bb0(%arg54: i32):
          "scf.for"(%25, %27, %31) ({
          ^bb0(%arg55: i32):
            %301 = "arith.muli"(%arg53, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %302 = "arith.addi"(%301, %arg54) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %303 = "arith.index_cast"(%302) : (i32) -> index
            %304 = "arith.index_cast"(%arg55) : (i32) -> index
            %305 = "memref.load"(%141, %303, %304) : (memref<32x32xbf16, strided<[32, 1]>>, index, index) -> bf16
            %306 = "arith.index_cast"(%arg53) : (i32) -> index
            %307 = "arith.index_cast"(%arg54) : (i32) -> index
            "memref.store"(%305, %119, %306, %307, %304) : (bf16, memref<16x2x8xbf16, strided<[16, 8, 1]>>, index, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "scf.for"(%25, %26, %31) ({
      ^bb0(%arg50: i32):
        "scf.for"(%25, %29, %31) ({
        ^bb0(%arg51: i32):
          "scf.for"(%25, %27, %31) ({
          ^bb0(%arg52: i32):
            %289 = "arith.index_cast"(%arg50) : (i32) -> index
            %290 = "arith.index_cast"(%arg52) : (i32) -> index
            %291 = "memref.load"(%121, %289, %290) : (memref<16x8xf32, strided<[8, 1]>>, index, index) -> f32
            %292 = "arith.index_cast"(%arg51) : (i32) -> index
            %293 = "memref.load"(%119, %289, %292, %290) : (memref<16x2x8xbf16, strided<[16, 8, 1]>>, index, index, index) -> bf16
            %294 = "arith.extf"(%293) : (bf16) -> f32
            %295 = "arith.mulf"(%291, %294) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            %296 = "arith.subf"(%295, %295) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            %297 = "arith.truncf"(%296) : (f32) -> bf16
            %298 = "arith.muli"(%arg50, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %299 = "arith.addi"(%298, %arg51) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %300 = "arith.index_cast"(%299) : (i32) -> index
            "memref.store"(%297, %141, %300, %290) : (bf16, memref<32x32xbf16, strided<[32, 1]>>, index, index) -> ()
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
            %275 = "arith.index_cast"(%arg47) : (i32) -> index
            %276 = "arith.index_cast"(%arg49) : (i32) -> index
            %277 = "memref.load"(%122, %275, %276) : (memref<16x8xf32, strided<[8, 1]>>, index, index) -> f32
            %278 = "arith.index_cast"(%arg48) : (i32) -> index
            %279 = "memref.load"(%119, %275, %278, %276) : (memref<16x2x8xbf16, strided<[16, 8, 1]>>, index, index, index) -> bf16
            %280 = "arith.extf"(%279) : (bf16) -> f32
            %281 = "arith.mulf"(%277, %280) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            %282 = "arith.addf"(%281, %281) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            %283 = "arith.truncf"(%282) : (f32) -> bf16
            %284 = "arith.muli"(%arg47, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %285 = "arith.addi"(%284, %arg48) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %286 = "arith.index_cast"(%285) : (i32) -> index
            %287 = "arith.addi"(%arg49, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %288 = "arith.index_cast"(%287) : (i32) -> index
            "memref.store"(%283, %141, %286, %288) : (bf16, memref<32x32xbf16, strided<[32, 1]>>, index, index) -> ()
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
            %268 = "arith.muli"(%arg44, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %269 = "arith.addi"(%268, %arg45) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %270 = "arith.index_cast"(%269) : (i32) -> index
            %271 = "arith.index_cast"(%arg46) : (i32) -> index
            %272 = "memref.load"(%144, %270, %271) : (memref<32x32xbf16, strided<[32, 1]>>, index, index) -> bf16
            %273 = "arith.index_cast"(%arg44) : (i32) -> index
            %274 = "arith.index_cast"(%arg45) : (i32) -> index
            "memref.store"(%272, %124, %273, %274, %271) : (bf16, memref<16x2x8xbf16, strided<[16, 8, 1]>>, index, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "scf.for"(%25, %26, %31) ({
      ^bb0(%arg41: i32):
        "scf.for"(%25, %29, %31) ({
        ^bb0(%arg42: i32):
          "scf.for"(%25, %27, %31) ({
          ^bb0(%arg43: i32):
            %256 = "arith.index_cast"(%arg41) : (i32) -> index
            %257 = "arith.index_cast"(%arg43) : (i32) -> index
            %258 = "memref.load"(%121, %256, %257) : (memref<16x8xf32, strided<[8, 1]>>, index, index) -> f32
            %259 = "arith.index_cast"(%arg42) : (i32) -> index
            %260 = "memref.load"(%124, %256, %259, %257) : (memref<16x2x8xbf16, strided<[16, 8, 1]>>, index, index, index) -> bf16
            %261 = "arith.extf"(%260) : (bf16) -> f32
            %262 = "arith.mulf"(%258, %261) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            %263 = "arith.subf"(%262, %262) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            %264 = "arith.truncf"(%263) : (f32) -> bf16
            %265 = "arith.muli"(%arg41, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %266 = "arith.addi"(%265, %arg42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %267 = "arith.index_cast"(%266) : (i32) -> index
            "memref.store"(%264, %144, %267, %257) : (bf16, memref<32x32xbf16, strided<[32, 1]>>, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "scf.for"(%25, %26, %31) ({
      ^bb0(%arg38: i32):
        "scf.for"(%25, %29, %31) ({
        ^bb0(%arg39: i32):
          "scf.for"(%25, %27, %31) ({
          ^bb0(%arg40: i32):
            %242 = "arith.index_cast"(%arg38) : (i32) -> index
            %243 = "arith.index_cast"(%arg40) : (i32) -> index
            %244 = "memref.load"(%122, %242, %243) : (memref<16x8xf32, strided<[8, 1]>>, index, index) -> f32
            %245 = "arith.index_cast"(%arg39) : (i32) -> index
            %246 = "memref.load"(%124, %242, %245, %243) : (memref<16x2x8xbf16, strided<[16, 8, 1]>>, index, index, index) -> bf16
            %247 = "arith.extf"(%246) : (bf16) -> f32
            %248 = "arith.mulf"(%244, %247) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            %249 = "arith.addf"(%248, %248) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            %250 = "arith.truncf"(%249) : (f32) -> bf16
            %251 = "arith.muli"(%arg38, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %252 = "arith.addi"(%251, %arg39) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %253 = "arith.index_cast"(%252) : (i32) -> index
            %254 = "arith.addi"(%arg40, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %255 = "arith.index_cast"(%254) : (i32) -> index
            "memref.store"(%250, %144, %253, %255) : (bf16, memref<32x32xbf16, strided<[32, 1]>>, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "scf.for"(%25, %30, %31) ({
      ^bb0(%arg36: i32):
        "scf.for"(%25, %30, %31) ({
        ^bb0(%arg37: i32):
          %234 = "arith.index_cast"(%arg36) : (i32) -> index
          %235 = "arith.index_cast"(%arg37) : (i32) -> index
          %236 = "memref.load"(%127, %234, %235) : (memref<32x32xf32, strided<[32, 1]>>, index, index) -> f32
          %237 = "arith.divsi"(%arg36, %29) : (i32, i32) -> i32
          %238 = "arith.index_cast"(%237) : (i32) -> index
          %239 = "memref.load"(%85, %238) : (memref<16xbf16, strided<[1]>>, index) -> bf16
          %240 = "arith.extf"(%239) : (bf16) -> f32
          %241 = "arith.mulf"(%236, %240) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%241, %127, %234, %235) : (f32, memref<32x32xf32, strided<[32, 1]>>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "hivm.hir.vexp"(%72, %72) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<16x16xf32, strided<[16, 1]>>, memref<16x16xf32, strided<[16, 1]>>) -> ()
      %108 = "arith.cmpi"(%48, %29) <{predicate = 2 : i64}> : (i32, i32) -> i1
      "scf.if"(%108) ({
        %196 = "arith.muli"(%48, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %197 = "arith.index_cast"(%196) : (i32) -> index
        %198 = "memref.subview"(%51, %197) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x32xf32>, index) -> memref<16x32xf32, strided<[32, 1], offset: ?>>
        "memref.copy"(%198, %86) : (memref<16x32xf32, strided<[32, 1], offset: ?>>, memref<16x32xf32, strided<[32, 1]>>) -> ()
        %199 = "memref.subview"(%50, %197) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x64xf32>, index) -> memref<16x64xf32, strided<[64, 1], offset: ?>>
        "memref.copy"(%199, %88) : (memref<16x64xf32, strided<[64, 1], offset: ?>>, memref<16x64xf32, strided<[64, 1]>>) -> ()
        %200 = "arith.index_cast"(%46) : (i32) -> index
        %201 = "arith.muli"(%arg28, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %202 = "arith.muli"(%48, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %203 = "arith.addi"(%201, %202) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %204 = "arith.index_cast"(%203) : (i32) -> index
        %205 = "memref.subview"(%37, %200, %204) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 1, 1, 8>, static_strides = array<i64: 1, 1, 1>}> : (memref<1x4x32xf32, strided<[128, 32, 1]>, #hivm.address_space<gm>>, index, index) -> memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%205, %89) : (memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>, memref<8xf32, strided<[1]>>) -> ()
        "hivm.hir.vexp"(%89, %89) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<8xf32, strided<[1]>>, memref<8xf32, strided<[1]>>) -> ()
        "scf.for"(%25, %26, %31) ({
        ^bb0(%arg33: i32):
          "scf.for"(%25, %30, %31) ({
          ^bb0(%arg35: i32):
            %215 = "arith.divsi"(%arg35, %29) : (i32, i32) -> i32
            %216 = "arith.muli"(%48, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %217 = "arith.divsi"(%arg33, %29) : (i32, i32) -> i32
            %218 = "arith.addi"(%216, %217) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %219 = "arith.cmpi"(%215, %218) <{predicate = 2 : i64}> : (i32, i32) -> i1
            "scf.if"(%219) ({
              %222 = "arith.index_cast"(%arg33) : (i32) -> index
              %223 = "arith.index_cast"(%arg35) : (i32) -> index
              %224 = "memref.load"(%86, %222, %223) : (memref<16x32xf32, strided<[32, 1]>>, index, index) -> f32
              %225 = "arith.muli"(%48, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %226 = "arith.divsi"(%arg33, %29) : (i32, i32) -> i32
              %227 = "arith.addi"(%225, %226) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %228 = "arith.index_cast"(%227) : (i32) -> index
              %229 = "arith.divsi"(%arg35, %29) : (i32, i32) -> i32
              %230 = "arith.index_cast"(%229) : (i32) -> index
              %231 = "memref.load"(%72, %228, %230) : (memref<16x16xf32, strided<[16, 1]>>, index, index) -> f32
              %232 = "arith.mulf"(%224, %231) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
              %233 = "arith.truncf"(%232) : (f32) -> bf16
              "memref.store"(%233, %87, %222, %223) : (bf16, memref<16x32xbf16, strided<[32, 1]>>, index, index) -> ()
              "scf.yield"() : () -> ()
            }, {
              %220 = "arith.index_cast"(%arg33) : (i32) -> index
              %221 = "arith.index_cast"(%arg35) : (i32) -> index
              "memref.store"(%18, %87, %220, %221) : (bf16, memref<16x32xbf16, strided<[32, 1]>>, index, index) -> ()
              "scf.yield"() : () -> ()
            }) : (i1) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.for"(%25, %28, %31) ({
          ^bb0(%arg34: i32):
            %208 = "arith.index_cast"(%arg33) : (i32) -> index
            %209 = "arith.index_cast"(%arg34) : (i32) -> index
            %210 = "memref.load"(%88, %208, %209) : (memref<16x64xf32, strided<[64, 1]>>, index, index) -> f32
            %211 = "arith.divsi"(%arg33, %29) : (i32, i32) -> i32
            %212 = "arith.index_cast"(%211) : (i32) -> index
            %213 = "memref.load"(%89, %212) : (memref<8xf32, strided<[1]>>, index) -> f32
            %214 = "arith.mulf"(%210, %213) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            "memref.store"(%214, %88, %208, %209) : (f32, memref<16x64xf32, strided<[64, 1]>>, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        %206 = "memref.subview"(%52, %197) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x32xbf16>, index) -> memref<16x32xbf16, strided<[32, 1], offset: ?>>
        "memref.copy"(%87, %206) : (memref<16x32xbf16, strided<[32, 1]>>, memref<16x32xbf16, strided<[32, 1], offset: ?>>) -> ()
        %207 = "memref.subview"(%50, %197) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x64xf32>, index) -> memref<16x64xf32, strided<[64, 1], offset: ?>>
        "memref.copy"(%88, %207) : (memref<16x64xf32, strided<[64, 1]>>, memref<16x64xf32, strided<[64, 1], offset: ?>>) -> ()
        "scf.yield"() : () -> ()
      }, {
      }) : (i1) -> ()
      "scf.if"(%108) ({
        %177 = "arith.muli"(%48, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %178 = "arith.index_cast"(%177) : (i32) -> index
        %179 = "memref.subview"(%50, %178) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x64xf32>, index) -> memref<16x64xf32, strided<[64, 1], offset: ?>>
        "memref.copy"(%179, %90) : (memref<16x64xf32, strided<[64, 1], offset: ?>>, memref<16x64xf32, strided<[64, 1]>>) -> ()
        %180 = "memref.subview"(%53, %178) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x64xf32>, index) -> memref<16x64xf32, strided<[64, 1], offset: ?>>
        "memref.copy"(%180, %91) : (memref<16x64xf32, strided<[64, 1], offset: ?>>, memref<16x64xf32, strided<[64, 1]>>) -> ()
        %181 = "memref.subview"(%49, %178) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x32xf32>, index) -> memref<16x32xf32, strided<[32, 1], offset: ?>>
        "memref.copy"(%181, %92) : (memref<16x32xf32, strided<[32, 1], offset: ?>>, memref<16x32xf32, strided<[32, 1]>>) -> ()
        %182 = "arith.muli"(%arg28, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %183 = "arith.muli"(%48, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %184 = "arith.addi"(%182, %183) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %185 = "arith.index_cast"(%184) : (i32) -> index
        %186 = "arith.index_cast"(%46) : (i32) -> index
        %187 = "memref.subview"(%38, %185, %186) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>, index, index) -> memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%187, %93) : (memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>, memref<8x64xbf16, strided<[64, 1]>>) -> ()
        "scf.for"(%25, %26, %31) ({
        ^bb0(%arg31: i32):
          "scf.for"(%25, %28, %31) ({
          ^bb0(%arg32: i32):
            %191 = "arith.index_cast"(%arg31) : (i32) -> index
            %192 = "arith.index_cast"(%arg32) : (i32) -> index
            %193 = "memref.load"(%90, %191, %192) : (memref<16x64xf32, strided<[64, 1]>>, index, index) -> f32
            %194 = "memref.load"(%91, %191, %192) : (memref<16x64xf32, strided<[64, 1]>>, index, index) -> f32
            %195 = "arith.addf"(%193, %194) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
            "memref.store"(%195, %90, %191, %192) : (f32, memref<16x64xf32, strided<[64, 1]>>, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        "scf.for"(%25, %27, %31) ({
        ^bb0(%arg29: i32):
          "scf.for"(%25, %28, %31) ({
          ^bb0(%arg30: i32):
            %189 = "arith.index_cast"(%arg29) : (i32) -> index
            %190 = "arith.index_cast"(%arg30) : (i32) -> index
            "memref.store"(%18, %94, %189, %190) : (bf16, memref<8x64xbf16, strided<[64, 1]>>, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "scf.yield"() : () -> ()
        }) : (i32, i32, i32) -> ()
        %188 = "memref.subview"(%42, %185, %186) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<1x32x4x64xbf16, strided<[8192, 256, 64, 1]>, #hivm.address_space<gm>>, index, index) -> memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%94, %188) : (memref<8x64xbf16, strided<[64, 1]>>, memref<8x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
        "scf.yield"() : () -> ()
      }, {
      }) : (i1) -> ()
      "scope.scope"() ({
        %164 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xbf16, strided<[64, 1]>>
        %165 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x64xbf16, strided<[64, 1]>>
        %166 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xbf16, strided<[128, 64, 1]>>
        %167 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x64xbf16, strided<[64, 1]>>
        %168 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xbf16, strided<[128, 64, 1]>>
        %169 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x64xbf16, strided<[64, 1]>>
        %170 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xbf16, strided<[128, 64, 1]>>
        %171 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xf32, strided<[128, 64, 1]>>
        %172 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xf32, strided<[128, 64, 1]>>
        %173 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xf32, strided<[128, 64, 1]>>
        "memref.copy"(%165, %167) : (memref<16x64xbf16, strided<[64, 1]>>, memref<16x64xbf16, strided<[64, 1]>>) -> ()
        %174 = "memref.reinterpret_cast"(%167) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 16, 1, 64>, static_strides = array<i64: 64, 64, 1>}> : (memref<16x64xbf16, strided<[64, 1]>>) -> memref<16x1x64xbf16, strided<[64, 64, 1]>>
        "hivm.hir.vbrc"(%174, %168) <{broadcast_dims = array<i64: 1>}> : (memref<16x1x64xbf16, strided<[64, 64, 1]>>, memref<16x2x64xbf16, strided<[128, 64, 1]>>) -> ()
        "memref.copy"(%66, %169) : (memref<2x64xbf16, strided<[64, 1]>>, memref<2x64xbf16, strided<[64, 1]>>) -> ()
        %175 = "memref.reinterpret_cast"(%169) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 2, 64>, static_strides = array<i64: 128, 64, 1>}> : (memref<2x64xbf16, strided<[64, 1]>>) -> memref<1x2x64xbf16, strided<[128, 64, 1]>>
        "hivm.hir.vbrc"(%175, %170) <{broadcast_dims = array<i64: 0>}> : (memref<1x2x64xbf16, strided<[128, 64, 1]>>, memref<16x2x64xbf16, strided<[128, 64, 1]>>) -> ()
        "hivm.hir.vcast"(%168, %171) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x64xbf16, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) -> ()
        "hivm.hir.vcast"(%170, %172) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x64xbf16, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) -> ()
        "hivm.hir.vmul"(%171, %172, %173) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xf32, strided<[128, 64, 1]>>) -> ()
        "hivm.hir.vcast"(%173, %166) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x64xf32, strided<[128, 64, 1]>>, memref<16x2x64xbf16, strided<[128, 64, 1]>>) -> ()
        "memref.copy"(%166, %56) : (memref<16x2x64xbf16, strided<[128, 64, 1]>>, memref<16x2x64xbf16>) -> ()
        %176 = "memref.reinterpret_cast"(%166) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 32, 64>, static_strides = array<i64: 64, 1>}> : (memref<16x2x64xbf16, strided<[128, 64, 1]>>) -> memref<32x64xbf16, strided<[64, 1]>>
        "memref.copy"(%176, %164) : (memref<32x64xbf16, strided<[64, 1]>>, memref<32x64xbf16, strided<[64, 1]>>) -> ()
        "memref.copy"(%164, %54) : (memref<32x64xbf16, strided<[64, 1]>>, memref<32x64xbf16>) -> ()
        "scope.return"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : () -> ()
      "scope.scope"() ({
        %163 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x64xbf16, strided<[128, 64, 1]>>
        "memref.copy"(%56, %163) : (memref<16x2x64xbf16>, memref<16x2x64xbf16, strided<[128, 64, 1]>>) -> ()
        "scope.return"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>} : () -> ()
      "scope.scope"() ({
        %158 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xbf16, strided<[64, 1]>>
        %159 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xbf16, strided<[32, 1]>>
        %160 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32, strided<[64, 1]>>
        "memref.copy"(%54, %158) : (memref<32x64xbf16>, memref<32x64xbf16, strided<[64, 1]>>) -> ()
        %161 = "memref.subview"(%52) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x32xbf16>) -> memref<32x32xbf16, strided<[32, 1]>>
        "memref.copy"(%161, %159) : (memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>) -> ()
        "hivm.hir.mmadL1"(%159, %158, %20, %16, %16, %12, %160) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> : (memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index, memref<32x64xf32, strided<[64, 1]>>) -> ()
        %162 = "memref.subview"(%53) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x64xf32>) -> memref<32x64xf32, strided<[64, 1]>>
        "memref.copy"(%160, %162) : (memref<32x64xf32, strided<[64, 1]>>, memref<32x64xf32, strided<[64, 1]>>) -> ()
        "scope.return"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<CUBE>} : () -> ()
      "scope.scope"() ({
        %115 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xbf16, strided<[64, 1]>>
        %116 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xbf16, strided<[64, 32, 1]>>
        %117 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xbf16, strided<[64, 32, 1]>>
        %118 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32, strided<[32, 1]>>
        %119 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x8xbf16, strided<[16, 8, 1]>>
        %120 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, strided<[8, 1]>>
        %121 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, strided<[8, 1]>>
        %122 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32, strided<[8, 1]>>
        %123 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xf32, strided<[64, 1]>>
        %124 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x8xbf16, strided<[16, 8, 1]>>
        %125 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xbf16, strided<[32, 1]>>
        %126 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xbf16, strided<[32, 1]>>
        %127 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32, strided<[32, 1]>>
        %128 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32, strided<[32, 1]>>
        %129 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32xbf16, strided<[32, 1]>>
        %130 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xbf16, strided<[64, 32, 1]>>
        %131 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32xbf16, strided<[32, 1]>>
        %132 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xbf16, strided<[64, 32, 1]>>
        %133 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xf32, strided<[64, 32, 1]>>
        %134 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xf32, strided<[64, 32, 1]>>
        %135 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xf32, strided<[64, 32, 1]>>
        %136 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xf32, strided<[64, 32, 1]>>
        %137 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xf32, strided<[64, 32, 1]>>
        %138 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xf32, strided<[64, 32, 1]>>
        %139 = "memref.subview"(%32, %104, %95) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 2, 1, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>}> : (memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%139, %116) : (memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x2x32xbf16, strided<[64, 32, 1]>>) -> ()
        "memref.copy"(%61, %129) : (memref<2x32xbf16, strided<[32, 1]>>, memref<2x32xbf16, strided<[32, 1]>>) -> ()
        %140 = "memref.reinterpret_cast"(%129) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 2, 32>, static_strides = array<i64: 64, 32, 1>}> : (memref<2x32xbf16, strided<[32, 1]>>) -> memref<1x2x32xbf16, strided<[64, 32, 1]>>
        "hivm.hir.vbrc"(%140, %130) <{broadcast_dims = array<i64: 0>}> : (memref<1x2x32xbf16, strided<[64, 32, 1]>>, memref<16x2x32xbf16, strided<[64, 32, 1]>>) -> ()
        "hivm.hir.vcast"(%116, %133) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x32xbf16, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) -> ()
        "hivm.hir.vcast"(%130, %134) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x32xbf16, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) -> ()
        "hivm.hir.vadd"(%133, %134, %135) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) -> ()
        "hivm.hir.vcast"(%135, %116) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xbf16, strided<[64, 32, 1]>>) -> ()
        "memref.copy"(%116, %57) : (memref<16x2x32xbf16, strided<[64, 32, 1]>>, memref<16x2x32xbf16>) -> ()
        %141 = "memref.reinterpret_cast"(%116) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 32, 1>}> : (memref<16x2x32xbf16, strided<[64, 32, 1]>>) -> memref<32x32xbf16, strided<[32, 1]>>
        %142 = "memref.subview"(%41, %104, %95) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 2, 1, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>}> : (memref<1x32x2x4x32xbf16, strided<[8192, 256, 128, 32, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%142, %117) : (memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x2x32xbf16, strided<[64, 32, 1]>>) -> ()
        "memref.copy"(%62, %131) : (memref<2x32xbf16, strided<[32, 1]>>, memref<2x32xbf16, strided<[32, 1]>>) -> ()
        %143 = "memref.reinterpret_cast"(%131) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 2, 32>, static_strides = array<i64: 64, 32, 1>}> : (memref<2x32xbf16, strided<[32, 1]>>) -> memref<1x2x32xbf16, strided<[64, 32, 1]>>
        "hivm.hir.vbrc"(%143, %132) <{broadcast_dims = array<i64: 0>}> : (memref<1x2x32xbf16, strided<[64, 32, 1]>>, memref<16x2x32xbf16, strided<[64, 32, 1]>>) -> ()
        "hivm.hir.vcast"(%117, %136) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x32xbf16, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) -> ()
        "hivm.hir.vcast"(%132, %137) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x32xbf16, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) -> ()
        "hivm.hir.vadd"(%136, %137, %138) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xf32, strided<[64, 32, 1]>>) -> ()
        "hivm.hir.vcast"(%138, %117) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<16x2x32xf32, strided<[64, 32, 1]>>, memref<16x2x32xbf16, strided<[64, 32, 1]>>) -> ()
        "memref.copy"(%117, %58) : (memref<16x2x32xbf16, strided<[64, 32, 1]>>, memref<16x2x32xbf16>) -> ()
        %144 = "memref.reinterpret_cast"(%117) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 32, 1>}> : (memref<16x2x32xbf16, strided<[64, 32, 1]>>) -> memref<32x32xbf16, strided<[32, 1]>>
        "hivm.hir.mmadL1"(%141, %144, %20, %16, %16, %16, %118) <{b_transpose, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> : (memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index, memref<32x32xf32, strided<[32, 1]>>) -> ()
        %145 = "memref.subview"(%49) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x32xf32>) -> memref<32x32xf32, strided<[32, 1]>>
        "memref.copy"(%118, %145) : (memref<32x32xf32, strided<[32, 1]>>, memref<32x32xf32, strided<[32, 1]>>) -> ()
        %146 = "memref.subview"(%36, %104, %95) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: 0, -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 16, 1, 8>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<1x32x4x8xf32, strided<[1024, 32, 8, 1]>, #hivm.address_space<gm>>, index, index) -> memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        "memref.copy"(%146, %120) : (memref<16x8xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x8xf32, strided<[8, 1]>>) -> ()
        %147 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32>
        %148 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32>
        %149 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32>
        %150 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32>
        "hivm.hir.vmul"(%120, %120, %147) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) -> ()
        "hivm.hir.vmul"(%147, %147, %148) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, memref<16x8xf32>, memref<16x8xf32>) -> ()
        "hivm.hir.vmul"(%147, %148, %149) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, memref<16x8xf32>, memref<16x8xf32>) -> ()
        "hivm.hir.vmul"(%147, %19, %147) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, f32, memref<16x8xf32>) -> ()
        "hivm.hir.vmul"(%148, %4, %148) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, f32, memref<16x8xf32>) -> ()
        "hivm.hir.vmul"(%149, %3, %149) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, f32, memref<16x8xf32>) -> ()
        "hivm.hir.vadd"(%147, %22, %150) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, f32, memref<16x8xf32>) -> ()
        "hivm.hir.vadd"(%148, %150, %150) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, memref<16x8xf32>, memref<16x8xf32>) -> ()
        "hivm.hir.vadd"(%149, %150, %121) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) -> ()
        %151 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32>
        %152 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32>
        %153 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32>
        %154 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32>
        %155 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x8xf32>
        "hivm.hir.vmul"(%120, %120, %151) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) -> ()
        "hivm.hir.vmul"(%151, %120, %152) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>) -> ()
        "hivm.hir.vmul"(%152, %151, %153) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, memref<16x8xf32>, memref<16x8xf32>) -> ()
        "hivm.hir.vmul"(%153, %151, %154) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, memref<16x8xf32>, memref<16x8xf32>) -> ()
        "hivm.hir.vmul"(%152, %2, %152) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, f32, memref<16x8xf32>) -> ()
        "hivm.hir.vmul"(%153, %1, %153) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, f32, memref<16x8xf32>) -> ()
        "hivm.hir.vmul"(%154, %0, %154) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, f32, memref<16x8xf32>) -> ()
        "hivm.hir.vadd"(%120, %152, %155) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32, strided<[8, 1]>>, memref<16x8xf32>, memref<16x8xf32>) -> ()
        "hivm.hir.vadd"(%153, %155, %155) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, memref<16x8xf32>, memref<16x8xf32>) -> ()
        "hivm.hir.vadd"(%154, %155, %122) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<16x8xf32>, memref<16x8xf32>, memref<16x8xf32, strided<[8, 1]>>) -> ()
        "memref.copy"(%55, %115) : (memref<32x64xbf16>, memref<32x64xbf16, strided<[64, 1]>>) -> ()
        "hivm.hir.mmadL1"(%141, %115, %20, %16, %16, %12, %123) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> : (memref<32x32xbf16, strided<[32, 1]>>, memref<32x64xbf16, strided<[64, 1]>>, i1, index, index, index, memref<32x64xf32, strided<[64, 1]>>) -> ()
        %156 = "memref.subview"(%50) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x64xf32>) -> memref<32x64xf32, strided<[64, 1]>>
        "memref.copy"(%123, %156) : (memref<32x64xf32, strided<[64, 1]>>, memref<32x64xf32, strided<[64, 1]>>) -> ()
        "memref.copy"(%144, %125) : (memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>) -> ()
        "hivm.hir.vcast"(%125, %127) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xf32, strided<[32, 1]>>) -> ()
        "hivm.hir.vcast"(%127, %125) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<32x32xf32, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>) -> ()
        "memref.copy"(%125, %60) : (memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16>) -> ()
        "memref.copy"(%60, %126) : (memref<32x32xbf16>, memref<32x32xbf16, strided<[32, 1]>>) -> ()
        "memref.copy"(%126, %144) : (memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>) -> ()
        "hivm.hir.mmadL1"(%141, %144, %20, %16, %16, %16, %128) <{b_transpose, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> : (memref<32x32xbf16, strided<[32, 1]>>, memref<32x32xbf16, strided<[32, 1]>>, i1, index, index, index, memref<32x32xf32, strided<[32, 1]>>) -> ()
        %157 = "memref.subview"(%51) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x32xf32>) -> memref<32x32xf32, strided<[32, 1]>>
        "memref.copy"(%128, %157) : (memref<32x32xf32, strided<[32, 1]>>, memref<32x32xf32, strided<[32, 1]>>) -> ()
        "scope.return"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>} : () -> ()
      "scope.scope"() ({
        %114 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xbf16, strided<[64, 32, 1]>>
        "memref.copy"(%57, %114) : (memref<16x2x32xbf16>, memref<16x2x32xbf16, strided<[64, 32, 1]>>) -> ()
        "scope.return"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>} : () -> ()
      "scope.scope"() ({
        %113 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xbf16, strided<[64, 32, 1]>>
        "memref.copy"(%58, %113) : (memref<16x2x32xbf16>, memref<16x2x32xbf16, strided<[64, 32, 1]>>) -> ()
        "memref.copy"(%113, %58) : (memref<16x2x32xbf16, strided<[64, 32, 1]>>, memref<16x2x32xbf16>) -> ()
        "scope.return"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>} : () -> ()
      "scope.scope"() ({
        %112 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xbf16, strided<[64, 1]>>
        "hivm.hir.vcast"(%65, %112) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<32x64xf32, strided<[64, 1]>>, memref<32x64xbf16, strided<[64, 1]>>) -> ()
        "memref.copy"(%112, %59) : (memref<32x64xbf16, strided<[64, 1]>>, memref<32x64xbf16>) -> ()
        "scope.return"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : () -> ()
      "scope.scope"() ({
        %110 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xbf16, strided<[64, 1]>>
        %111 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x64xbf16, strided<[64, 1]>>
        "memref.copy"(%59, %111) : (memref<32x64xbf16>, memref<32x64xbf16, strided<[64, 1]>>) -> ()
        "memref.copy"(%111, %110) : (memref<32x64xbf16, strided<[64, 1]>>, memref<32x64xbf16, strided<[64, 1]>>) -> ()
        "memref.copy"(%110, %55) : (memref<32x64xbf16, strided<[64, 1]>>, memref<32x64xbf16>) -> ()
        "scope.return"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>} : () -> ()
      "scope.scope"() ({
        %109 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x2x32xbf16, strided<[64, 32, 1]>>
        "memref.copy"(%58, %109) : (memref<16x2x32xbf16>, memref<16x2x32xbf16, strided<[64, 32, 1]>>) -> ()
        "scope.return"() : () -> ()
      }) {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>} : () -> ()
      "scf.yield"() : () -> ()
    }) {tilelangir.num_stages = 2 : i32} : (i32, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} : () -> ()
}) {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} : () -> ()



Traceback (most recent call last):
  File "/home/z00910011/tilelang-ascend/examples/mamba/mamba3_mimo_fwd_npu_mix.py", line 808, in <module>
    run_test()
  File "/home/z00910011/tilelang-ascend/examples/mamba/mamba3_mimo_fwd_npu_mix.py", line 760, in run_test
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
[ERROR] 2026-04-23-20:18:13 (PID:852519, Device:11, RankID:-1) ERR99999 UNKNOWN applicaiton exception
