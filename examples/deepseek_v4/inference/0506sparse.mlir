2026-05-06 16:12:44  [TileLang:tilelang.env:WARNING]: Loading tilelang libs from dev root: /home/z00910011/tilelang-ascend-test/tilelang-ascend/build
Warning: The current version of the file storing weights is old, and it is relanded due to internal bug of torch and compatibility issue. We will deprecate the loading support for this type of file in the future, please use newer torch to re-store the weight file.
====== TVM IR ======
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def sparseAttnMix(Q_handle: T.handle, KV_handle: T.handle, Output_handle: T.handle, AttnSink: T.Buffer((64,), "float32"), TopKIndices_handle: T.handle, batchSize: T.int32, seqLen: T.int32, seqLenKV: T.int32, topK: T.int32):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "stackvm", "tag": ""}, "keys": [], "kind": "npuir", "tag": ""})})
        Q = T.match_buffer(Q_handle, (batchSize, seqLen, 64, 512), "bfloat16")
        KV = T.match_buffer(KV_handle, (batchSize, seqLenKV, 512), "bfloat16")
        Output = T.match_buffer(Output_handle, (batchSize, seqLen, 64, 512), "bfloat16")
        TopKIndices = T.match_buffer(TopKIndices_handle, (batchSize, seqLen, topK), "int32")
        cid = T.launch_thread("blockIdx.x", batchSize * seqLen)
        vid = T.launch_thread("blockIdx.y", batchSize * seqLen * 2)
        workspace_kv = T.decl_buffer((2, 32, 512), "bfloat16", scope="global.workspace;multi_buffer=2")
        workspace_score = T.decl_buffer((2, 16, 32), scope="global.workspace;multi_buffer=2")
        workspace_prob = T.decl_buffer((2, 16, 32), "bfloat16", scope="global.workspace;multi_buffer=2")
        workspace_out = T.decl_buffer((2, 16, 512), scope="global.workspace;multi_buffer=2")
        for n in range(4):
            q_shared = T.decl_buffer((16, 512), "bfloat16", scope="shared.flat")
            scores_max = T.decl_buffer((8, 1), scope="shared.flat")
            scores_max_prev = T.decl_buffer((8, 1), scope="shared.flat")
            sum_exp = T.decl_buffer((8, 1), scope="shared.flat")
            acc_o = T.decl_buffer((8, 512), scope="shared.flat")
            o_cast = T.decl_buffer((8, 512), "bfloat16", scope="shared.flat")
            tmp_3_buf = T.decl_buffer((8, 1), scope="shared.flat")
            tmp_4_buf = T.decl_buffer((8, 1), scope="shared.flat")
            T.npuir_brc(0, T.region(acc_o[0, 0], 2, 8, 512))
            T.npuir_brc(0, T.region(sum_exp[0, 0], 2, 8, 1))
            T.npuir_brc(T.float32("-inf"), T.region(scores_max[0, 0], 2, 8, 1))
            T.copy(T.region(Q[cid // seqLen, cid % seqLen, n * 16, 0], 1, 1, 1, 16, 512), T.region(q_shared[0, 0], 2, 16, 512))
            for k in T.serial((topK + 31) // 32, annotations={"num_stages": 2}):
                kv_shared = T.decl_buffer((32, 512), "bfloat16", scope="shared.flat")
                prob_shared = T.decl_buffer((16, 32), "bfloat16", scope="shared.flat")
                scores = T.decl_buffer((16, 32), scope="local.fragment")
                scores_cast = T.decl_buffer((8, 32), "bfloat16", scope="shared.flat")
                pv_acc = T.decl_buffer((16, 512), scope="local.fragment")
                kv_ub = T.decl_buffer((32, 512), "bfloat16", scope="shared.flat")
                idxs = T.decl_buffer((32,), "int32", scope="local.fragment")
                mask_ub = T.decl_buffer((1, 32), scope="shared.flat")
                scores_ub = T.decl_buffer((8, 32), scope="shared.flat")
                scores_scale = T.decl_buffer((8, 1), scope="shared.flat")
                scores_sum = T.decl_buffer((8, 1), scope="shared.flat")
                acc_o_new = T.decl_buffer((8, 512), scope="shared.flat")
                tmp_0_buf = T.decl_buffer((8, 1), scope="shared.flat")
                local_src_buf = T.decl_buffer((8, 1), scope="shared.flat")
                reshape_view_buf = T.decl_buffer((8, 1), scope="shared.flat")
                brc_buf = T.decl_buffer((8, 32), scope="shared.flat")
                tmp_1_buf = T.decl_buffer((8, 32), scope="shared.flat")
                local_src_buf_1 = T.decl_buffer((1, 32), scope="shared.flat")
                reshape_view_buf_1 = T.decl_buffer((1, 32), scope="shared.flat")
                brc_buf_1 = T.decl_buffer((8, 32), scope="shared.flat")
                tmp_2_buf = T.decl_buffer((8, 1), scope="shared.flat")
                T.npuir_brc(0, T.region(kv_ub[0, 0], 2, 32, 512))
                T.npuir_brc(0, T.region(mask_ub[0, 0], 2, 1, 32))
                T.copy(T.region(TopKIndices[cid // seqLen, cid % seqLen, k * 32], 1, 1, 1, T.min(topK - k * 32, 32)), T.region(idxs[0], 2, T.min(topK - k * 32, 32)))
                for i in range(T.min(topK - k * 32, 32)):
                    cur_idx: T.int32 = idxs[i]
                    if cur_idx != -1:
                        mask_ub[0, i] = T.float32(1)
                        T.copy(T.region(KV[cid // seqLen, cur_idx, 0], 1, 1, 1, 512), T.region(kv_ub[i, 0], 2, 1, 512))
                T.copy(T.region(kv_ub[0, 0], 1, 32, 512), T.region(workspace_kv[0, 0, 0], 2, 1, 32, 512))
                T.copy(T.region(workspace_kv[0, 0, 0], 1, 1, 32, 512), T.region(kv_shared[0, 0], 2, 32, 512))
                T.npuir_dot(T.region(q_shared[0, 0], 1, 16, 512), T.region(kv_shared[0, 0], 1, 512, 32), T.region(scores[0, 0], 3, 16, 32), T.bool(True), T.bool(False), T.bool(True))
                T.copy(T.region(scores[0, 0], 1, 16, 32), T.region(workspace_score[0, 0, 0], 2, 1, 16, 32))
                T.copy(T.region(workspace_score[0, vid * 8, 0], 1, 1, 8, 32), T.region(scores_ub[0, 0], 2, 8, 32))
                T.copy(T.region(scores_max[0, 0], 1, 8, 1), T.region(scores_max_prev[0, 0], 2, 8, 1))
                T.npuir_mul(T.region(scores_ub[0, 0], 1, 8, 32), T.float32(0.044194173824159223), T.region(scores_ub[0, 0], 2, 8, 32))
                T.npuir_reduce(T.region(scores_ub[0, 0], 1, 8, 32), T.region(scores_max[0, 0], 2, 8, 1), "1", "max")
                T.npuir_sub(T.region(scores_max_prev[0, 0], 1, 8, 1), T.region(scores_max[0, 0], 1, 8, 1), T.region(tmp_0_buf[0, 0], 2, 8, 1))
                T.npuir_exp(T.region(tmp_0_buf[0, 0], 1, 8, 1), T.region(scores_scale[0, 0], 2, 8, 1))
                T.copy(T.region(scores_max[0, 0], 1, 8, 1), T.region(local_src_buf[0, 0], 2, 8, 1))
                T.npuir_reshape(T.region(local_src_buf[0, 0], 1, 8, 1), T.region(reshape_view_buf[0, 0], 2, 8, 1))
                T.npuir_brc(T.region(reshape_view_buf[0, 0], 1, 8, 1), T.region(brc_buf[0, 0], 2, 8, 32))
                T.npuir_sub(T.region(scores_ub[0, 0], 1, 8, 32), T.region(brc_buf[0, 0], 1, 8, 32), T.region(tmp_1_buf[0, 0], 2, 8, 32))
                T.npuir_exp(T.region(tmp_1_buf[0, 0], 1, 8, 32), T.region(scores_ub[0, 0], 2, 8, 32))
                T.copy(T.region(mask_ub[0, 0], 1, 1, 32), T.region(local_src_buf_1[0, 0], 2, 1, 32))
                T.npuir_reshape(T.region(local_src_buf_1[0, 0], 1, 1, 32), T.region(reshape_view_buf_1[0, 0], 2, 1, 32))
                T.npuir_brc(T.region(reshape_view_buf_1[0, 0], 1, 1, 32), T.region(brc_buf_1[0, 0], 2, 8, 32))
                T.npuir_mul(T.region(scores_ub[0, 0], 1, 8, 32), T.region(brc_buf_1[0, 0], 1, 8, 32), T.region(scores_ub[0, 0], 2, 8, 32))
                T.npuir_reduce(T.region(scores_ub[0, 0], 1, 8, 32), T.region(scores_sum[0, 0], 2, 8, 1), "1", "sum")
                T.npuir_mul(T.region(sum_exp[0, 0], 1, 8, 1), T.region(scores_scale[0, 0], 1, 8, 1), T.region(tmp_2_buf[0, 0], 2, 8, 1))
                T.npuir_add(T.region(tmp_2_buf[0, 0], 1, 8, 1), T.region(scores_sum[0, 0], 1, 8, 1), T.region(sum_exp[0, 0], 2, 8, 1))
                T.npuir_cast(T.region(scores_ub[0, 0], 1, 8, 32), T.region(scores_cast[0, 0], 2, 8, 32), "rint")
                T.copy(T.region(scores_cast[0, 0], 1, 8, 32), T.region(workspace_prob[0, vid * 8, 0], 2, 1, 8, 32))
                T.copy(T.region(workspace_prob[0, 0, 0], 1, 1, 16, 32), T.region(prob_shared[0, 0], 2, 16, 32))
                T.npuir_dot(T.region(prob_shared[0, 0], 1, 16, 32), T.region(kv_shared[0, 0], 1, 32, 512), T.region(pv_acc[0, 0], 3, 16, 512), T.bool(True), T.bool(False), T.bool(False))
                T.copy(T.region(pv_acc[0, 0], 1, 16, 512), T.region(workspace_out[0, 0, 0], 2, 1, 16, 512))
                T.copy(T.region(workspace_out[0, vid * 8, 0], 1, 1, 8, 512), T.region(acc_o_new[0, 0], 2, 8, 512))
                T.npuir_mul(T.region(acc_o[0, 0], 1, 8, 512), T.region(scores_scale[0, 0], 1, 8, 1), T.region(acc_o[0, 0], 2, 8, 512))
                T.npuir_add(T.region(acc_o[0, 0], 1, 8, 512), T.region(acc_o_new[0, 0], 1, 8, 512), T.region(acc_o[0, 0], 2, 8, 512))
            for i in T.parallel(8):
                if n * 2 + vid < 8:
                    scores_max_prev[i, 0] = AttnSink[n * 16 + vid * 8 + i]
                else:
                    scores_max_prev[i, 0] = T.float32("-inf")
            T.npuir_sub(T.region(scores_max_prev[0, 0], 1, 8, 1), T.region(scores_max[0, 0], 1, 8, 1), T.region(tmp_3_buf[0, 0], 2, 8, 1))
            T.npuir_exp(T.region(tmp_3_buf[0, 0], 1, 8, 1), T.region(tmp_4_buf[0, 0], 2, 8, 1))
            T.npuir_add(T.region(sum_exp[0, 0], 1, 8, 1), T.region(tmp_4_buf[0, 0], 1, 8, 1), T.region(sum_exp[0, 0], 2, 8, 1))
            T.npuir_div(T.region(acc_o[0, 0], 1, 8, 512), T.region(sum_exp[0, 0], 1, 8, 1), T.region(acc_o[0, 0], 2, 8, 512))
            T.npuir_cast(T.region(acc_o[0, 0], 1, 8, 512), T.region(o_cast[0, 0], 2, 8, 512), "rint")
            T.copy(T.region(o_cast[0, 0], 1, T.min(8, 64 - vid * 8 - n * 16), 512), T.region(Output[cid // seqLen, cid % seqLen, n * 16 + vid * 8, 0], 2, 1, 1, T.min(8, 64 - vid * 8 - n * 16), 512))

====== npuir ======
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %2 = arith.index_cast %c1_i32 : i32 to index
    %c512_i32 = arith.constant 512 : i32
    %3 = arith.muli %c512_i32, %c1_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %c64_i32 = arith.constant 64 : i32
    %5 = arith.muli %c64_i32, %3 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.muli %arg9, %5 : i32
    %8 = arith.index_cast %7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%8, %6, %4, %2] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%8, %6, %4, %2] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %9 = arith.index_cast %arg11 : i32 to index
    %10 = arith.muli %arg11, %c1_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = arith.muli %arg9, %10 : i32
    %13 = arith.index_cast %12 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %9], strides: [%13, %11, %2] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %14 = arith.index_cast %arg10 : i32 to index
    %15 = arith.muli %arg10, %3 : i32
    %16 = arith.index_cast %15 : i32 to index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %14, 512], strides: [%16, %4, %2] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%2] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %17 = hivm.hir.get_block_idx -> i64
    %18 = arith.trunci %17 : i64 to i32
    %19 = hivm.hir.get_sub_block_idx -> i64
    %20 = arith.trunci %19 : i64 to i32
    %21 = memref_ext.alloc_workspace() : memref<2x32x512xbf16>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<2x32x512xbf16>
    %22 = memref_ext.alloc_workspace() : memref<2x16x32xf32>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xf32>
    %23 = memref_ext.alloc_workspace() : memref<2x16x32xbf16>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xbf16>
    %24 = memref_ext.alloc_workspace() : memref<2x16x512xf32>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<2x16x512xf32>
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32_4 = arith.constant 1 : i32
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32_4  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>>
      %alloc_5 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_6 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_7 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_8 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>>
      %alloc_9 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_11 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %c0_i32_12 = arith.constant 0 : i32
      %25 = arith.sitofp %c0_i32_12 : i32 to f32
      hivm.hir.vbrc ins(%25 : f32) outs(%alloc_8 : memref<8x512xf32, strided<[512, 1]>>)
      %26 = arith.sitofp %c0_i32_12 : i32 to f32
      hivm.hir.vbrc ins(%26 : f32) outs(%alloc_7 : memref<8x1xf32, strided<[1, 1]>>)
      %cst = arith.constant 0xFF800000 : f32
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<8x1xf32, strided<[1, 1]>>)
      %27 = arith.divsi %18, %arg9 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.remsi %18, %arg9 : i32
      %30 = arith.index_cast %29 : i32 to index
      %c16_i32 = arith.constant 16 : i32
      %31 = arith.muli %arg18, %c16_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview = memref.subview %reinterpret_cast[%28, %30, %32, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>>
      %c31_i32 = arith.constant 31 : i32
      %33 = arith.addi %arg11, %c31_i32 : i32
      %c32_i32 = arith.constant 32 : i32
      %34 = arith.divsi %33, %c32_i32 : i32
      %c1_i32_13 = arith.constant 1 : i32
      scf.for %arg19 = %c0_i32_12 to %34 step %c1_i32_13  : i32 {
        %alloc_18 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_19 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
        %alloc_20 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
        %alloc_21 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>>
        %alloc_22 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>>
        %alloc_23 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_24 = memref.alloc() : memref<32xi32, strided<[1]>>
        %alloc_25 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_26 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_28 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_29 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_31 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_32 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_33 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_34 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_35 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_36 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_37 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_38 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %c0_i32_39 = arith.constant 0 : i32
        %42 = arith.sitofp %c0_i32_39 : i32 to bf16
        hivm.hir.vbrc ins(%42 : bf16) outs(%alloc_23 : memref<32x512xbf16, strided<[512, 1]>>)
        %43 = arith.sitofp %c0_i32_39 : i32 to f32
        hivm.hir.vbrc ins(%43 : f32) outs(%alloc_25 : memref<1x32xf32, strided<[32, 1]>>)
        %44 = arith.divsi %18, %arg9 : i32
        %45 = arith.index_cast %44 : i32 to index
        %46 = arith.remsi %18, %arg9 : i32
        %47 = arith.index_cast %46 : i32 to index
        %c32_i32_40 = arith.constant 32 : i32
        %48 = arith.muli %arg19, %c32_i32_40 : i32
        %49 = arith.index_cast %48 : i32 to index
        %50 = arith.subi %arg11, %48 : i32
        %51 = arith.minsi %50, %c32_i32_40 : i32
        %52 = arith.index_cast %51 : i32 to index
        %subview_41 = memref.subview %reinterpret_cast_1[%45, %47, %49] [1, 1, %52] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_42 = memref.subview %alloc_24[0] [%52] [1] : memref<32xi32, strided<[1]>> to memref<?xi32, strided<[1]>>
        memref.copy %subview_41, %subview_42 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>>
        %c1_i32_43 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_39 to %51 step %c1_i32_43  : i32 {
          %58 = arith.index_cast %arg20 : i32 to index
          %59 = memref.load %alloc_24[%58] : memref<32xi32, strided<[1]>>
          %c-1_i32 = arith.constant -1 : i32
          %60 = arith.cmpi ne, %59, %c-1_i32 : i32
          scf.if %60 {
            %cst_64 = arith.constant 1.000000e+00 : f32
            %c0_i32_65 = arith.constant 0 : i32
            %61 = arith.index_cast %c0_i32_65 : i32 to index
            %62 = arith.index_cast %arg20 : i32 to index
            memref.store %cst_64, %alloc_25[%61, %62] : memref<1x32xf32, strided<[32, 1]>>
            %63 = arith.divsi %18, %arg9 : i32
            %64 = arith.index_cast %63 : i32 to index
            %65 = arith.index_cast %59 : i32 to index
            %subview_66 = memref.subview %reinterpret_cast_2[%64, %65, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %subview_67 = memref.subview %alloc_23[%62, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>> to memref<512xbf16, strided<[1], offset: ?>>
            memref.copy %subview_66, %subview_67 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>>
          }
        }
        %subview_44 = memref.subview %21[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        memref.copy %alloc_23, %subview_44 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16, strided<[512, 1]>>
        %subview_45 = memref.subview %21[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        memref.copy %subview_45, %alloc_18 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16, strided<[512, 1]>>
        %true = arith.constant true
        %c16_i32_46 = arith.constant 16 : i32
        %53 = arith.index_cast %c16_i32_46 : i32 to index
        %c512_i32_47 = arith.constant 512 : i32
        %54 = arith.index_cast %c512_i32_47 : i32 to index
        %55 = arith.index_cast %c32_i32_40 : i32 to index
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_18, %true, %53, %54, %55 : memref<16x512xbf16, strided<[512, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_20 : memref<16x32xf32, strided<[32, 1]>>)
        %subview_48 = memref.subview %22[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xf32> to memref<16x32xf32, strided<[32, 1]>>
        memref.copy %alloc_20, %subview_48 : memref<16x32xf32, strided<[32, 1]>> to memref<16x32xf32, strided<[32, 1]>>
        %c8_i32_49 = arith.constant 8 : i32
        %56 = arith.muli %20, %c8_i32_49 : i32
        %57 = arith.index_cast %56 : i32 to index
        %subview_50 = memref.subview %22[0, %57, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_50, %alloc_26 : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1]>>
        %subview_51 = memref.subview %alloc_5[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_52 = memref.subview %alloc_6[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_51, %subview_52 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        %cst_53 = arith.constant 0.0441941731 : f32
        hivm.hir.vmul ins(%alloc_26, %cst_53 : memref<8x32xf32, strided<[32, 1]>>, f32) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_5 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vsub ins(%alloc_6, %alloc_5 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>>)
        %subview_54 = memref.subview %alloc_5[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_55 = memref.subview %alloc_31[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_54, %subview_55 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        %reinterpret_cast_56 = memref.reinterpret_cast %alloc_31 to offset: [0], sizes: [8, 1], strides: [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8x1xf32, strided<[1, 1]>>
        hivm.hir.vbrc ins(%reinterpret_cast_56 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_33 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [1]
        hivm.hir.vsub ins(%alloc_26, %alloc_33 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vexp ins(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>>)
        %subview_57 = memref.subview %alloc_25[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_58 = memref.subview %alloc_35[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_57, %subview_58 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        %reinterpret_cast_59 = memref.reinterpret_cast %alloc_35 to offset: [0], sizes: [1, 32], strides: [32, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<1x32xf32, strided<[32, 1]>>
        hivm.hir.vbrc ins(%reinterpret_cast_59 : memref<1x32xf32, strided<[32, 1]>>) outs(%alloc_37 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [0]
        hivm.hir.vmul ins(%alloc_26, %alloc_37 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_28 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmul ins(%alloc_7, %alloc_27 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_38 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_38, %alloc_28 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vcast ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_21 : memref<8x32xbf16, strided<[32, 1]>>)
        %subview_60 = memref.subview %23[0, %57, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xbf16> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_21, %subview_60 : memref<8x32xbf16, strided<[32, 1]>> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        %subview_61 = memref.subview %23[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xbf16> to memref<16x32xbf16, strided<[32, 1]>>
        memref.copy %subview_61, %alloc_19 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1]>>
        hivm.hir.mmadL1 ins(%alloc_19, %alloc_18, %true, %53, %55, %54 : memref<16x32xbf16, strided<[32, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_22 : memref<16x512xf32, strided<[512, 1]>>)
        %subview_62 = memref.subview %24[0, 0, 0] [1, 16, 512] [1, 1, 1] : memref<2x16x512xf32> to memref<16x512xf32, strided<[512, 1]>>
        memref.copy %alloc_22, %subview_62 : memref<16x512xf32, strided<[512, 1]>> to memref<16x512xf32, strided<[512, 1]>>
        %subview_63 = memref.subview %24[0, %57, 0] [1, 8, 512] [1, 1, 1] : memref<2x16x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
        memref.copy %subview_63, %alloc_29 : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<8x512xf32, strided<[512, 1]>>
        hivm.hir.vmul ins(%alloc_8, %alloc_27 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_8 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_8, %alloc_29 : memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_8 : memref<8x512xf32, strided<[512, 1]>>)
      } {tilelangir.num_stages = 2 : i32}
      %c8_i32 = arith.constant 8 : i32
      %c1_i32_14 = arith.constant 1 : i32
      scf.for %arg19 = %c0_i32_12 to %c8_i32 step %c1_i32_14  : i32 {
        %c2_i32 = arith.constant 2 : i32
        %42 = arith.muli %arg18, %c2_i32 : i32
        %43 = arith.addi %42, %20 : i32
        %c8_i32_18 = arith.constant 8 : i32
        %44 = arith.cmpi slt, %43, %c8_i32_18 : i32
        scf.if %44 {
          %c16_i32_19 = arith.constant 16 : i32
          %45 = arith.muli %arg18, %c16_i32_19 : i32
          %c8_i32_20 = arith.constant 8 : i32
          %46 = arith.muli %20, %c8_i32_20 : i32
          %47 = arith.addi %45, %46 : i32
          %48 = arith.addi %47, %arg19 : i32
          %49 = arith.index_cast %48 : i32 to index
          %50 = memref.load %reinterpret_cast_3[%49] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %51 = arith.index_cast %arg19 : i32 to index
          %c0_i32_21 = arith.constant 0 : i32
          %52 = arith.index_cast %c0_i32_21 : i32 to index
          memref.store %50, %alloc_6[%51, %52] : memref<8x1xf32, strided<[1, 1]>>
        } else {
          %cst_19 = arith.constant 0xFF800000 : f32
          %45 = arith.index_cast %arg19 : i32 to index
          %c0_i32_20 = arith.constant 0 : i32
          %46 = arith.index_cast %c0_i32_20 : i32 to index
          memref.store %cst_19, %alloc_6[%45, %46] : memref<8x1xf32, strided<[1, 1]>>
        }
      }
      hivm.hir.vsub ins(%alloc_6, %alloc_5 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_7, %alloc_11 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vdiv ins(%alloc_8, %alloc_7 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_8 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_8 : memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_9 : memref<8x512xbf16, strided<[512, 1]>>)
      %c64_i32_15 = arith.constant 64 : i32
      %35 = arith.muli %20, %c8_i32 : i32
      %36 = arith.subi %c64_i32_15, %35 : i32
      %37 = arith.subi %36, %31 : i32
      %38 = arith.minsi %c8_i32, %37 : i32
      %39 = arith.index_cast %38 : i32 to index
      %subview_16 = memref.subview %alloc_9[0, 0] [%39, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[512, 1]>>
      %40 = arith.addi %31, %35 : i32
      %41 = arith.index_cast %40 : i32 to index
      %subview_17 = memref.subview %reinterpret_cast_0[%28, %30, %41, 0] [1, 1, %39, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_16, %subview_17 : memref<?x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}
// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = memref_ext.alloc_workspace() : memref<2x32x512xbf16>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<2x32x512xbf16>
    %16 = memref_ext.alloc_workspace() : memref<2x16x32xf32>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xf32>
    %17 = memref_ext.alloc_workspace() : memref<2x16x32xbf16>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xbf16>
    %18 = memref_ext.alloc_workspace() : memref<2x16x512xf32>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<2x16x512xf32>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>>
      %alloc_8 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>>
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      scf.for %arg19 = %c0_i32 to %26 step %c1_i32  : i32 {
        %alloc_17 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_28 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>>
        %alloc_29 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_31 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_32 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_33 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_34 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_35 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_22 : memref<32x512xbf16, strided<[512, 1]>>)
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_24 : memref<1x32xf32, strided<[32, 1]>>)
        %34 = arith.divsi %12, %arg9 : i32
        %35 = arith.index_cast %34 : i32 to index
        %36 = arith.remsi %12, %arg9 : i32
        %37 = arith.index_cast %36 : i32 to index
        %38 = arith.muli %arg19, %c32_i32 : i32
        %39 = arith.index_cast %38 : i32 to index
        %40 = arith.subi %arg11, %38 : i32
        %41 = arith.minsi %40, %c32_i32 : i32
        %42 = arith.index_cast %41 : i32 to index
        %subview_36 = memref.subview %reinterpret_cast_5[%35, %37, %39] [1, 1, %42] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_37 = memref.subview %alloc_23[0] [%42] [1] : memref<32xi32, strided<[1]>> to memref<?xi32, strided<[1]>>
        memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>>
        scf.for %arg20 = %c0_i32 to %41 step %c1_i32  : i32 {
          %45 = arith.index_cast %arg20 : i32 to index
          %46 = memref.load %alloc_23[%45] : memref<32xi32, strided<[1]>>
          %47 = arith.cmpi ne, %46, %c-1_i32 : i32
          scf.if %47 {
            %48 = arith.index_cast %arg20 : i32 to index
            memref.store %cst_2, %alloc_24[%c0, %48] : memref<1x32xf32, strided<[32, 1]>>
            %49 = arith.divsi %12, %arg9 : i32
            %50 = arith.index_cast %49 : i32 to index
            %51 = arith.index_cast %46 : i32 to index
            %subview_52 = memref.subview %reinterpret_cast_6[%50, %51, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %subview_53 = memref.subview %alloc_22[%48, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>> to memref<512xbf16, strided<[1], offset: ?>>
            memref.copy %subview_52, %subview_53 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>>
          }
        }
        %subview_38 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        memref.copy %alloc_22, %subview_38 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16, strided<[512, 1]>>
        %subview_39 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        memref.copy %subview_39, %alloc_17 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_17, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_19 : memref<16x32xf32, strided<[32, 1]>>)
        %subview_40 = memref.subview %16[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xf32> to memref<16x32xf32, strided<[32, 1]>>
        memref.copy %alloc_19, %subview_40 : memref<16x32xf32, strided<[32, 1]>> to memref<16x32xf32, strided<[32, 1]>>
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        %subview_41 = memref.subview %16[0, %44, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_41, %alloc_25 : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1]>>
        %subview_42 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_43 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_42, %subview_43 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vmul ins(%alloc_25, %cst_1 : memref<8x32xf32, strided<[32, 1]>>, f32) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_26 : memref<8x1xf32, strided<[1, 1]>>)
        %subview_44 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_45 = memref.subview %alloc_30[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_44, %subview_45 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_31 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [1]
        hivm.hir.vsub ins(%alloc_25, %alloc_31 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vexp ins(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        %subview_46 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_47 = memref.subview %alloc_33[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_46, %subview_47 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_33 : memref<1x32xf32, strided<[32, 1]>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [0]
        hivm.hir.vmul ins(%alloc_25, %alloc_34 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmul ins(%alloc_10, %alloc_26 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_35 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_35, %alloc_27 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vcast ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_20 : memref<8x32xbf16, strided<[32, 1]>>)
        %subview_48 = memref.subview %17[0, %44, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xbf16> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_20, %subview_48 : memref<8x32xbf16, strided<[32, 1]>> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        %subview_49 = memref.subview %17[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xbf16> to memref<16x32xbf16, strided<[32, 1]>>
        memref.copy %subview_49, %alloc_18 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1]>>
        hivm.hir.mmadL1 ins(%alloc_18, %alloc_17, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_21 : memref<16x512xf32, strided<[512, 1]>>)
        %subview_50 = memref.subview %18[0, 0, 0] [1, 16, 512] [1, 1, 1] : memref<2x16x512xf32> to memref<16x512xf32, strided<[512, 1]>>
        memref.copy %alloc_21, %subview_50 : memref<16x512xf32, strided<[512, 1]>> to memref<16x512xf32, strided<[512, 1]>>
        %subview_51 = memref.subview %18[0, %44, 0] [1, 8, 512] [1, 1, 1] : memref<2x16x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
        memref.copy %subview_51, %alloc_28 : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<8x512xf32, strided<[512, 1]>>
        hivm.hir.vmul ins(%alloc_11, %alloc_26 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_11, %alloc_28 : memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %34 = arith.muli %arg18, %c2_i32 : i32
        %35 = arith.addi %34, %14 : i32
        %36 = arith.cmpi slt, %35, %c8_i32 : i32
        scf.if %36 {
          %37 = arith.muli %arg18, %c16_i32 : i32
          %38 = arith.muli %14, %c8_i32 : i32
          %39 = arith.addi %37, %38 : i32
          %40 = arith.addi %39, %arg19 : i32
          %41 = arith.index_cast %40 : i32 to index
          %42 = memref.load %reinterpret_cast_7[%41] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %43 = arith.index_cast %arg19 : i32 to index
          memref.store %42, %alloc_9[%43, %c0] : memref<8x1xf32, strided<[1, 1]>>
        } else {
          %37 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%37, %c0] : memref<8x1xf32, strided<[1, 1]>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>>)
      %27 = arith.muli %14, %c8_i32 : i32
      %28 = arith.subi %c64_i32, %27 : i32
      %29 = arith.subi %28, %23 : i32
      %30 = arith.minsi %29, %c8_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %subview_15 = memref.subview %alloc_12[0, 0] [%31, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[512, 1]>>
      %32 = arith.addi %23, %27 : i32
      %33 = arith.index_cast %32 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast_4[%20, %22, %33, 0] [1, 1, %31, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_15, %subview_16 : memref<?x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After AdaptTritonKernel (adapt-triton-kernel) ('builtin.module' operation) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = memref_ext.alloc_workspace() : memref<2x32x512xbf16>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<2x32x512xbf16>
    %16 = memref_ext.alloc_workspace() : memref<2x16x32xf32>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xf32>
    %17 = memref_ext.alloc_workspace() : memref<2x16x32xbf16>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xbf16>
    %18 = memref_ext.alloc_workspace() : memref<2x16x512xf32>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<2x16x512xf32>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>>
      %alloc_8 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>>
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      scf.for %arg19 = %c0_i32 to %26 step %c1_i32  : i32 {
        %alloc_17 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_28 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>>
        %alloc_29 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_31 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_32 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_33 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_34 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_35 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_22 : memref<32x512xbf16, strided<[512, 1]>>)
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_24 : memref<1x32xf32, strided<[32, 1]>>)
        %34 = arith.divsi %12, %arg9 : i32
        %35 = arith.index_cast %34 : i32 to index
        %36 = arith.remsi %12, %arg9 : i32
        %37 = arith.index_cast %36 : i32 to index
        %38 = arith.muli %arg19, %c32_i32 : i32
        %39 = arith.index_cast %38 : i32 to index
        %40 = arith.subi %arg11, %38 : i32
        %41 = arith.minsi %40, %c32_i32 : i32
        %42 = arith.index_cast %41 : i32 to index
        %subview_36 = memref.subview %reinterpret_cast_5[%35, %37, %39] [1, 1, %42] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_37 = memref.subview %alloc_23[0] [%42] [1] : memref<32xi32, strided<[1]>> to memref<?xi32, strided<[1]>>
        memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>>
        scf.for %arg20 = %c0_i32 to %41 step %c1_i32  : i32 {
          %45 = arith.index_cast %arg20 : i32 to index
          %46 = memref.load %alloc_23[%45] : memref<32xi32, strided<[1]>>
          %47 = arith.cmpi ne, %46, %c-1_i32 : i32
          scf.if %47 {
            %48 = arith.index_cast %arg20 : i32 to index
            memref.store %cst_2, %alloc_24[%c0, %48] : memref<1x32xf32, strided<[32, 1]>>
            %49 = arith.divsi %12, %arg9 : i32
            %50 = arith.index_cast %49 : i32 to index
            %51 = arith.index_cast %46 : i32 to index
            %subview_52 = memref.subview %reinterpret_cast_6[%50, %51, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %subview_53 = memref.subview %alloc_22[%48, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>> to memref<512xbf16, strided<[1], offset: ?>>
            memref.copy %subview_52, %subview_53 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>>
          }
        }
        %subview_38 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        memref.copy %alloc_22, %subview_38 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16, strided<[512, 1]>>
        %subview_39 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        memref.copy %subview_39, %alloc_17 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_17, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_19 : memref<16x32xf32, strided<[32, 1]>>)
        %subview_40 = memref.subview %16[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xf32> to memref<16x32xf32, strided<[32, 1]>>
        memref.copy %alloc_19, %subview_40 : memref<16x32xf32, strided<[32, 1]>> to memref<16x32xf32, strided<[32, 1]>>
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        %subview_41 = memref.subview %16[0, %44, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_41, %alloc_25 : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1]>>
        %subview_42 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_43 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_42, %subview_43 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vmul ins(%alloc_25, %cst_1 : memref<8x32xf32, strided<[32, 1]>>, f32) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_26 : memref<8x1xf32, strided<[1, 1]>>)
        %subview_44 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_45 = memref.subview %alloc_30[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_44, %subview_45 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_31 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [1]
        hivm.hir.vsub ins(%alloc_25, %alloc_31 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vexp ins(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        %subview_46 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_47 = memref.subview %alloc_33[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_46, %subview_47 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_33 : memref<1x32xf32, strided<[32, 1]>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [0]
        hivm.hir.vmul ins(%alloc_25, %alloc_34 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmul ins(%alloc_10, %alloc_26 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_35 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_35, %alloc_27 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vcast ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_20 : memref<8x32xbf16, strided<[32, 1]>>)
        %subview_48 = memref.subview %17[0, %44, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xbf16> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_20, %subview_48 : memref<8x32xbf16, strided<[32, 1]>> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        %subview_49 = memref.subview %17[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xbf16> to memref<16x32xbf16, strided<[32, 1]>>
        memref.copy %subview_49, %alloc_18 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1]>>
        hivm.hir.mmadL1 ins(%alloc_18, %alloc_17, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_21 : memref<16x512xf32, strided<[512, 1]>>)
        %subview_50 = memref.subview %18[0, 0, 0] [1, 16, 512] [1, 1, 1] : memref<2x16x512xf32> to memref<16x512xf32, strided<[512, 1]>>
        memref.copy %alloc_21, %subview_50 : memref<16x512xf32, strided<[512, 1]>> to memref<16x512xf32, strided<[512, 1]>>
        %subview_51 = memref.subview %18[0, %44, 0] [1, 8, 512] [1, 1, 1] : memref<2x16x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
        memref.copy %subview_51, %alloc_28 : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<8x512xf32, strided<[512, 1]>>
        hivm.hir.vmul ins(%alloc_11, %alloc_26 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_11, %alloc_28 : memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %34 = arith.muli %arg18, %c2_i32 : i32
        %35 = arith.addi %34, %14 : i32
        %36 = arith.cmpi slt, %35, %c8_i32 : i32
        scf.if %36 {
          %37 = arith.muli %arg18, %c16_i32 : i32
          %38 = arith.muli %14, %c8_i32 : i32
          %39 = arith.addi %37, %38 : i32
          %40 = arith.addi %39, %arg19 : i32
          %41 = arith.index_cast %40 : i32 to index
          %42 = memref.load %reinterpret_cast_7[%41] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %43 = arith.index_cast %arg19 : i32 to index
          memref.store %42, %alloc_9[%43, %c0] : memref<8x1xf32, strided<[1, 1]>>
        } else {
          %37 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%37, %c0] : memref<8x1xf32, strided<[1, 1]>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>>)
      %27 = arith.muli %14, %c8_i32 : i32
      %28 = arith.subi %c64_i32, %27 : i32
      %29 = arith.subi %28, %23 : i32
      %30 = arith.minsi %29, %c8_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %subview_15 = memref.subview %alloc_12[0, 0] [%31, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[512, 1]>>
      %32 = arith.addi %23, %27 : i32
      %33 = arith.index_cast %32 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast_4[%20, %22, %33, 0] [1, 1, %31, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_15, %subview_16 : memref<?x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRInsertWorkspace (tilelangir-insert-workspace) ('func.func' operation: @sparseAttnMix) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = memref_ext.alloc_workspace() : memref<2x32x512xbf16>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<2x32x512xbf16>
    %16 = memref_ext.alloc_workspace() : memref<2x16x32xf32>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xf32>
    %17 = memref_ext.alloc_workspace() : memref<2x16x32xbf16>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xbf16>
    %18 = memref_ext.alloc_workspace() : memref<2x16x512xf32>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<2x16x512xf32>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>>
      %alloc_8 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>>
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      scf.for %arg19 = %c0_i32 to %26 step %c1_i32  : i32 {
        %alloc_17 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_28 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>>
        %alloc_29 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_31 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_32 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_33 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_34 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_35 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_22 : memref<32x512xbf16, strided<[512, 1]>>)
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_24 : memref<1x32xf32, strided<[32, 1]>>)
        %34 = arith.divsi %12, %arg9 : i32
        %35 = arith.index_cast %34 : i32 to index
        %36 = arith.remsi %12, %arg9 : i32
        %37 = arith.index_cast %36 : i32 to index
        %38 = arith.muli %arg19, %c32_i32 : i32
        %39 = arith.index_cast %38 : i32 to index
        %40 = arith.subi %arg11, %38 : i32
        %41 = arith.minsi %40, %c32_i32 : i32
        %42 = arith.index_cast %41 : i32 to index
        %subview_36 = memref.subview %reinterpret_cast_5[%35, %37, %39] [1, 1, %42] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_37 = memref.subview %alloc_23[0] [%42] [1] : memref<32xi32, strided<[1]>> to memref<?xi32, strided<[1]>>
        memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>>
        scf.for %arg20 = %c0_i32 to %41 step %c1_i32  : i32 {
          %45 = arith.index_cast %arg20 : i32 to index
          %46 = memref.load %alloc_23[%45] : memref<32xi32, strided<[1]>>
          %47 = arith.cmpi ne, %46, %c-1_i32 : i32
          scf.if %47 {
            %48 = arith.index_cast %arg20 : i32 to index
            memref.store %cst_2, %alloc_24[%c0, %48] : memref<1x32xf32, strided<[32, 1]>>
            %49 = arith.divsi %12, %arg9 : i32
            %50 = arith.index_cast %49 : i32 to index
            %51 = arith.index_cast %46 : i32 to index
            %subview_52 = memref.subview %reinterpret_cast_6[%50, %51, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %subview_53 = memref.subview %alloc_22[%48, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>> to memref<512xbf16, strided<[1], offset: ?>>
            memref.copy %subview_52, %subview_53 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>>
          }
        }
        %subview_38 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        memref.copy %alloc_22, %subview_38 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16, strided<[512, 1]>>
        %subview_39 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        memref.copy %subview_39, %alloc_17 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_17, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_19 : memref<16x32xf32, strided<[32, 1]>>)
        %subview_40 = memref.subview %16[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xf32> to memref<16x32xf32, strided<[32, 1]>>
        memref.copy %alloc_19, %subview_40 : memref<16x32xf32, strided<[32, 1]>> to memref<16x32xf32, strided<[32, 1]>>
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        %subview_41 = memref.subview %16[0, %44, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_41, %alloc_25 : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1]>>
        %subview_42 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_43 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_42, %subview_43 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vmul ins(%alloc_25, %cst_1 : memref<8x32xf32, strided<[32, 1]>>, f32) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_26 : memref<8x1xf32, strided<[1, 1]>>)
        %subview_44 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_45 = memref.subview %alloc_30[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_44, %subview_45 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_31 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [1]
        hivm.hir.vsub ins(%alloc_25, %alloc_31 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vexp ins(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        %subview_46 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_47 = memref.subview %alloc_33[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_46, %subview_47 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_33 : memref<1x32xf32, strided<[32, 1]>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [0]
        hivm.hir.vmul ins(%alloc_25, %alloc_34 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmul ins(%alloc_10, %alloc_26 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_35 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_35, %alloc_27 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vcast ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_20 : memref<8x32xbf16, strided<[32, 1]>>)
        %subview_48 = memref.subview %17[0, %44, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xbf16> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_20, %subview_48 : memref<8x32xbf16, strided<[32, 1]>> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        %subview_49 = memref.subview %17[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xbf16> to memref<16x32xbf16, strided<[32, 1]>>
        memref.copy %subview_49, %alloc_18 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1]>>
        hivm.hir.mmadL1 ins(%alloc_18, %alloc_17, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_21 : memref<16x512xf32, strided<[512, 1]>>)
        %subview_50 = memref.subview %18[0, 0, 0] [1, 16, 512] [1, 1, 1] : memref<2x16x512xf32> to memref<16x512xf32, strided<[512, 1]>>
        memref.copy %alloc_21, %subview_50 : memref<16x512xf32, strided<[512, 1]>> to memref<16x512xf32, strided<[512, 1]>>
        %subview_51 = memref.subview %18[0, %44, 0] [1, 8, 512] [1, 1, 1] : memref<2x16x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
        memref.copy %subview_51, %alloc_28 : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<8x512xf32, strided<[512, 1]>>
        hivm.hir.vmul ins(%alloc_11, %alloc_26 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_11, %alloc_28 : memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %34 = arith.muli %arg18, %c2_i32 : i32
        %35 = arith.addi %34, %14 : i32
        %36 = arith.cmpi slt, %35, %c8_i32 : i32
        scf.if %36 {
          %37 = arith.muli %arg18, %c16_i32 : i32
          %38 = arith.muli %14, %c8_i32 : i32
          %39 = arith.addi %37, %38 : i32
          %40 = arith.addi %39, %arg19 : i32
          %41 = arith.index_cast %40 : i32 to index
          %42 = memref.load %reinterpret_cast_7[%41] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %43 = arith.index_cast %arg19 : i32 to index
          memref.store %42, %alloc_9[%43, %c0] : memref<8x1xf32, strided<[1, 1]>>
        } else {
          %37 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%37, %c0] : memref<8x1xf32, strided<[1, 1]>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>>)
      %27 = arith.muli %14, %c8_i32 : i32
      %28 = arith.subi %c64_i32, %27 : i32
      %29 = arith.subi %28, %23 : i32
      %30 = arith.minsi %29, %c8_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %subview_15 = memref.subview %alloc_12[0, 0] [%31, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[512, 1]>>
      %32 = arith.addi %23, %27 : i32
      %33 = arith.index_cast %32 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast_4[%20, %22, %33, 0] [1, 1, %31, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_15, %subview_16 : memref<?x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRMarkMultiBuffer (tilelangir-mark-multibuffer) ('func.func' operation: @sparseAttnMix) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = memref_ext.alloc_workspace() : memref<2x32x512xbf16>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<2x32x512xbf16>
    %16 = memref_ext.alloc_workspace() : memref<2x16x32xf32>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xf32>
    %17 = memref_ext.alloc_workspace() : memref<2x16x32xbf16>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xbf16>
    %18 = memref_ext.alloc_workspace() : memref<2x16x512xf32>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<2x16x512xf32>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>>
      %alloc_8 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>>
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      scf.for %arg19 = %c0_i32 to %26 step %c1_i32  : i32 {
        %alloc_17 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_28 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>>
        %alloc_29 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_31 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_32 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_33 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_34 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_35 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_22 : memref<32x512xbf16, strided<[512, 1]>>)
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_24 : memref<1x32xf32, strided<[32, 1]>>)
        %34 = arith.divsi %12, %arg9 : i32
        %35 = arith.index_cast %34 : i32 to index
        %36 = arith.remsi %12, %arg9 : i32
        %37 = arith.index_cast %36 : i32 to index
        %38 = arith.muli %arg19, %c32_i32 : i32
        %39 = arith.index_cast %38 : i32 to index
        %40 = arith.subi %arg11, %38 : i32
        %41 = arith.minsi %40, %c32_i32 : i32
        %42 = arith.index_cast %41 : i32 to index
        %subview_36 = memref.subview %reinterpret_cast_5[%35, %37, %39] [1, 1, %42] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_37 = memref.subview %alloc_23[0] [%42] [1] : memref<32xi32, strided<[1]>> to memref<?xi32, strided<[1]>>
        memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>>
        scf.for %arg20 = %c0_i32 to %41 step %c1_i32  : i32 {
          %45 = arith.index_cast %arg20 : i32 to index
          %46 = memref.load %alloc_23[%45] : memref<32xi32, strided<[1]>>
          %47 = arith.cmpi ne, %46, %c-1_i32 : i32
          scf.if %47 {
            %48 = arith.index_cast %arg20 : i32 to index
            memref.store %cst_2, %alloc_24[%c0, %48] : memref<1x32xf32, strided<[32, 1]>>
            %49 = arith.divsi %12, %arg9 : i32
            %50 = arith.index_cast %49 : i32 to index
            %51 = arith.index_cast %46 : i32 to index
            %subview_52 = memref.subview %reinterpret_cast_6[%50, %51, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %subview_53 = memref.subview %alloc_22[%48, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>> to memref<512xbf16, strided<[1], offset: ?>>
            memref.copy %subview_52, %subview_53 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>>
          }
        }
        %subview_38 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        memref.copy %alloc_22, %subview_38 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16, strided<[512, 1]>>
        %subview_39 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        memref.copy %subview_39, %alloc_17 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_17, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_19 : memref<16x32xf32, strided<[32, 1]>>)
        %subview_40 = memref.subview %16[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xf32> to memref<16x32xf32, strided<[32, 1]>>
        memref.copy %alloc_19, %subview_40 : memref<16x32xf32, strided<[32, 1]>> to memref<16x32xf32, strided<[32, 1]>>
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        %subview_41 = memref.subview %16[0, %44, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_41, %alloc_25 : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1]>>
        %subview_42 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_43 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_42, %subview_43 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vmul ins(%alloc_25, %cst_1 : memref<8x32xf32, strided<[32, 1]>>, f32) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_26 : memref<8x1xf32, strided<[1, 1]>>)
        %subview_44 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_45 = memref.subview %alloc_30[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_44, %subview_45 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_31 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [1]
        hivm.hir.vsub ins(%alloc_25, %alloc_31 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vexp ins(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        %subview_46 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_47 = memref.subview %alloc_33[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_46, %subview_47 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_33 : memref<1x32xf32, strided<[32, 1]>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [0]
        hivm.hir.vmul ins(%alloc_25, %alloc_34 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmul ins(%alloc_10, %alloc_26 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_35 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_35, %alloc_27 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vcast ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_20 : memref<8x32xbf16, strided<[32, 1]>>)
        %subview_48 = memref.subview %17[0, %44, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xbf16> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_20, %subview_48 : memref<8x32xbf16, strided<[32, 1]>> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        %subview_49 = memref.subview %17[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xbf16> to memref<16x32xbf16, strided<[32, 1]>>
        memref.copy %subview_49, %alloc_18 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1]>>
        hivm.hir.mmadL1 ins(%alloc_18, %alloc_17, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_21 : memref<16x512xf32, strided<[512, 1]>>)
        %subview_50 = memref.subview %18[0, 0, 0] [1, 16, 512] [1, 1, 1] : memref<2x16x512xf32> to memref<16x512xf32, strided<[512, 1]>>
        memref.copy %alloc_21, %subview_50 : memref<16x512xf32, strided<[512, 1]>> to memref<16x512xf32, strided<[512, 1]>>
        %subview_51 = memref.subview %18[0, %44, 0] [1, 8, 512] [1, 1, 1] : memref<2x16x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
        memref.copy %subview_51, %alloc_28 : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<8x512xf32, strided<[512, 1]>>
        hivm.hir.vmul ins(%alloc_11, %alloc_26 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_11, %alloc_28 : memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %34 = arith.muli %arg18, %c2_i32 : i32
        %35 = arith.addi %34, %14 : i32
        %36 = arith.cmpi slt, %35, %c8_i32 : i32
        scf.if %36 {
          %37 = arith.muli %arg18, %c16_i32 : i32
          %38 = arith.muli %14, %c8_i32 : i32
          %39 = arith.addi %37, %38 : i32
          %40 = arith.addi %39, %arg19 : i32
          %41 = arith.index_cast %40 : i32 to index
          %42 = memref.load %reinterpret_cast_7[%41] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %43 = arith.index_cast %arg19 : i32 to index
          memref.store %42, %alloc_9[%43, %c0] : memref<8x1xf32, strided<[1, 1]>>
        } else {
          %37 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%37, %c0] : memref<8x1xf32, strided<[1, 1]>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>>)
      %27 = arith.muli %14, %c8_i32 : i32
      %28 = arith.subi %c64_i32, %27 : i32
      %29 = arith.subi %28, %23 : i32
      %30 = arith.minsi %29, %c8_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %subview_15 = memref.subview %alloc_12[0, 0] [%31, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[512, 1]>>
      %32 = arith.addi %23, %27 : i32
      %33 = arith.index_cast %32 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast_4[%20, %22, %33, 0] [1, 1, %31, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_15, %subview_16 : memref<?x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRCVSplit (tilelangir-cv-split) ('func.func' operation: @sparseAttnMix) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = memref_ext.alloc_workspace() : memref<2x32x512xbf16>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<2x32x512xbf16>
    %16 = memref_ext.alloc_workspace() : memref<2x16x32xf32>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xf32>
    %17 = memref_ext.alloc_workspace() : memref<2x16x32xbf16>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xbf16>
    %18 = memref_ext.alloc_workspace() : memref<2x16x512xf32>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<2x16x512xf32>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>>
      %alloc_8 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>>
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      scf.for %arg19 = %c0_i32 to %26 step %c1_i32  : i32 {
        %alloc_17 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_28 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>>
        %alloc_29 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        %alloc_31 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_32 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_33 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        %alloc_34 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_35 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        scope.scope : () -> () {
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_22 : memref<32x512xbf16, strided<[512, 1]>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_24 : memref<1x32xf32, strided<[32, 1]>>)
          %36 = arith.divsi %12, %arg9 : i32
          %37 = arith.index_cast %36 : i32 to index
          %38 = arith.remsi %12, %arg9 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.muli %arg19, %c32_i32 : i32
          %41 = arith.index_cast %40 : i32 to index
          %42 = arith.subi %arg11, %40 : i32
          %43 = arith.minsi %42, %c32_i32 : i32
          %44 = arith.index_cast %43 : i32 to index
          %subview_36 = memref.subview %reinterpret_cast_5[%37, %39, %41] [1, 1, %44] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_37 = memref.subview %alloc_23[0] [%44] [1] : memref<32xi32, strided<[1]>> to memref<?xi32, strided<[1]>>
          memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>>
          scf.for %arg20 = %c0_i32 to %43 step %c1_i32  : i32 {
            %45 = arith.index_cast %arg20 : i32 to index
            %46 = memref.load %alloc_23[%45] : memref<32xi32, strided<[1]>>
            %47 = arith.cmpi ne, %46, %c-1_i32 : i32
            scf.if %47 {
              %48 = arith.index_cast %arg20 : i32 to index
              memref.store %cst_2, %alloc_24[%c0, %48] : memref<1x32xf32, strided<[32, 1]>>
              %49 = arith.divsi %12, %arg9 : i32
              %50 = arith.index_cast %49 : i32 to index
              %51 = arith.index_cast %46 : i32 to index
              %subview_39 = memref.subview %reinterpret_cast_6[%50, %51, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_40 = memref.subview %alloc_22[%48, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>> to memref<512xbf16, strided<[1], offset: ?>>
              memref.copy %subview_39, %subview_40 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>>
            }
          }
          %subview_38 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
          memref.copy %alloc_22, %subview_38 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16, strided<[512, 1]>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %subview_36 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
          memref.copy %subview_36, %alloc_17 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16, strided<[512, 1]>>
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_17, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_19 : memref<16x32xf32, strided<[32, 1]>>)
          %subview_37 = memref.subview %16[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xf32> to memref<16x32xf32, strided<[32, 1]>>
          memref.copy %alloc_19, %subview_37 : memref<16x32xf32, strided<[32, 1]>> to memref<16x32xf32, strided<[32, 1]>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %34 = arith.muli %14, %c8_i32 : i32
        %35 = arith.index_cast %34 : i32 to index
        scope.scope : () -> () {
          %subview_36 = memref.subview %16[0, %35, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
          memref.copy %subview_36, %alloc_25 : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1]>>
          %subview_37 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
          %subview_38 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
          memref.copy %subview_37, %subview_38 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
          hivm.hir.vmul ins(%alloc_25, %cst_1 : memref<8x32xf32, strided<[32, 1]>>, f32) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
          hivm.hir.vreduce <max> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>)
          hivm.hir.vexp ins(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_26 : memref<8x1xf32, strided<[1, 1]>>)
          %subview_39 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
          %subview_40 = memref.subview %alloc_30[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
          memref.copy %subview_39, %subview_40 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
          hivm.hir.vbrc ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_31 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_25, %alloc_31 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>)
          hivm.hir.vexp ins(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
          %subview_41 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
          %subview_42 = memref.subview %alloc_33[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
          memref.copy %subview_41, %subview_42 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
          hivm.hir.vbrc ins(%alloc_33 : memref<1x32xf32, strided<[32, 1]>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_25, %alloc_34 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
          hivm.hir.vreduce <sum> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_10, %alloc_26 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_35 : memref<8x1xf32, strided<[1, 1]>>)
          hivm.hir.vadd ins(%alloc_35, %alloc_27 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
          hivm.hir.vcast ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_20 : memref<8x32xbf16, strided<[32, 1]>>)
          %subview_43 = memref.subview %17[0, %35, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xbf16> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_20, %subview_43 : memref<8x32xbf16, strided<[32, 1]>> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %subview_36 = memref.subview %17[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xbf16> to memref<16x32xbf16, strided<[32, 1]>>
          memref.copy %subview_36, %alloc_18 : memref<16x32xbf16, strided<[32, 1]>> to memref<16x32xbf16, strided<[32, 1]>>
          hivm.hir.mmadL1 ins(%alloc_18, %alloc_17, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_21 : memref<16x512xf32, strided<[512, 1]>>)
          %subview_37 = memref.subview %18[0, 0, 0] [1, 16, 512] [1, 1, 1] : memref<2x16x512xf32> to memref<16x512xf32, strided<[512, 1]>>
          memref.copy %alloc_21, %subview_37 : memref<16x512xf32, strided<[512, 1]>> to memref<16x512xf32, strided<[512, 1]>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %subview_36 = memref.subview %18[0, %35, 0] [1, 8, 512] [1, 1, 1] : memref<2x16x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
          memref.copy %subview_36, %alloc_28 : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<8x512xf32, strided<[512, 1]>>
          hivm.hir.vmul ins(%alloc_11, %alloc_26 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_11, %alloc_28 : memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %34 = arith.muli %arg18, %c2_i32 : i32
        %35 = arith.addi %34, %14 : i32
        %36 = arith.cmpi slt, %35, %c8_i32 : i32
        scf.if %36 {
          %37 = arith.muli %arg18, %c16_i32 : i32
          %38 = arith.muli %14, %c8_i32 : i32
          %39 = arith.addi %37, %38 : i32
          %40 = arith.addi %39, %arg19 : i32
          %41 = arith.index_cast %40 : i32 to index
          %42 = memref.load %reinterpret_cast_7[%41] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %43 = arith.index_cast %arg19 : i32 to index
          memref.store %42, %alloc_9[%43, %c0] : memref<8x1xf32, strided<[1, 1]>>
        } else {
          %37 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%37, %c0] : memref<8x1xf32, strided<[1, 1]>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>>)
      %27 = arith.muli %14, %c8_i32 : i32
      %28 = arith.subi %c64_i32, %27 : i32
      %29 = arith.subi %28, %23 : i32
      %30 = arith.minsi %29, %c8_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %subview_15 = memref.subview %alloc_12[0, 0] [%31, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[512, 1]>>
      %32 = arith.addi %23, %27 : i32
      %33 = arith.index_cast %32 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast_4[%20, %22, %33, 0] [1, 1, %31, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_15, %subview_16 : memref<?x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRInferMemScope (tilelangir-infer-mem-scope) ('func.func' operation: @sparseAttnMix) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = memref_ext.alloc_workspace() : memref<2x32x512xbf16, #hivm.address_space<gm>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<2x32x512xbf16, #hivm.address_space<gm>>
    %16 = memref_ext.alloc_workspace() : memref<2x16x32xf32, #hivm.address_space<gm>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xf32, #hivm.address_space<gm>>
    %17 = memref_ext.alloc_workspace() : memref<2x16x32xbf16, #hivm.address_space<gm>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xbf16, #hivm.address_space<gm>>
    %18 = memref_ext.alloc_workspace() : memref<2x16x512xf32, #hivm.address_space<gm>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %alloc_8 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      scf.for %arg19 = %c0_i32 to %26 step %c1_i32  : i32 {
        %alloc_17 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_31 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_32 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_33 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_34 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_35 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        scope.scope : () -> () {
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_22 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_24 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %36 = arith.divsi %12, %arg9 : i32
          %37 = arith.index_cast %36 : i32 to index
          %38 = arith.remsi %12, %arg9 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.muli %arg19, %c32_i32 : i32
          %41 = arith.index_cast %40 : i32 to index
          %42 = arith.subi %arg11, %40 : i32
          %43 = arith.minsi %42, %c32_i32 : i32
          %44 = arith.index_cast %43 : i32 to index
          %subview_36 = memref.subview %reinterpret_cast_5[%37, %39, %41] [1, 1, %44] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_37 = memref.subview %alloc_23[0] [%44] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          scf.for %arg20 = %c0_i32 to %43 step %c1_i32  : i32 {
            %45 = arith.index_cast %arg20 : i32 to index
            %46 = memref.load %alloc_23[%45] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %47 = arith.cmpi ne, %46, %c-1_i32 : i32
            scf.if %47 {
              %48 = arith.index_cast %arg20 : i32 to index
              memref.store %cst_2, %alloc_24[%c0, %48] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %49 = arith.divsi %12, %arg9 : i32
              %50 = arith.index_cast %49 : i32 to index
              %51 = arith.index_cast %46 : i32 to index
              %subview_39 = memref.subview %reinterpret_cast_6[%50, %51, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_40 = memref.subview %alloc_22[%48, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_39, %subview_40 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_38 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16, #hivm.address_space<gm>> to memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<gm>>
          memref.copy %alloc_22, %subview_38 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<gm>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %subview_36 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16, #hivm.address_space<gm>> to memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<gm>>
          memref.copy %subview_36, %alloc_17 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_17, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_19 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_37 = memref.subview %16[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xf32, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<gm>>
          memref.copy %alloc_19, %subview_37 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>> to memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<gm>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %34 = arith.muli %14, %c8_i32 : i32
        %35 = arith.index_cast %34 : i32 to index
        scope.scope : () -> () {
          %subview_36 = memref.subview %16[0, %35, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xf32, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_36, %alloc_25 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_37 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_38 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_37, %subview_38 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_25, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_29 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_39 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_40 = memref.subview %alloc_30[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_39, %subview_40 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_25, %alloc_31 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_32 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_41 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_42 = memref.subview %alloc_33[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_41, %subview_42 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_33 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_25, %alloc_34 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_10, %alloc_26 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_35 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_35, %alloc_27 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_20 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_43 = memref.subview %17[0, %35, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xbf16, #hivm.address_space<gm>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_20, %subview_43 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %subview_36 = memref.subview %17[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xbf16, #hivm.address_space<gm>> to memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
          memref.copy %subview_36, %alloc_18 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 ins(%alloc_18, %alloc_17, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_21 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_37 = memref.subview %18[0, 0, 0] [1, 16, 512] [1, 1, 1] : memref<2x16x512xf32, #hivm.address_space<gm>> to memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<gm>>
          memref.copy %alloc_21, %subview_37 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>> to memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<gm>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %subview_36 = memref.subview %18[0, %35, 0] [1, 8, 512] [1, 1, 1] : memref<2x16x512xf32, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_36, %alloc_28 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_11, %alloc_26 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_11, %alloc_28 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %34 = arith.muli %arg18, %c2_i32 : i32
        %35 = arith.addi %34, %14 : i32
        %36 = arith.cmpi slt, %35, %c8_i32 : i32
        scf.if %36 {
          %37 = arith.muli %arg18, %c16_i32 : i32
          %38 = arith.muli %14, %c8_i32 : i32
          %39 = arith.addi %37, %38 : i32
          %40 = arith.addi %39, %arg19 : i32
          %41 = arith.index_cast %40 : i32 to index
          %42 = memref.load %reinterpret_cast_7[%41] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %43 = arith.index_cast %arg19 : i32 to index
          memref.store %42, %alloc_9[%43, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %37 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%37, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %27 = arith.muli %14, %c8_i32 : i32
      %28 = arith.subi %c64_i32, %27 : i32
      %29 = arith.subi %28, %23 : i32
      %30 = arith.minsi %29, %c8_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %subview_15 = memref.subview %alloc_12[0, 0] [%31, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %32 = arith.addi %23, %27 : i32
      %33 = arith.index_cast %32 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast_4[%20, %22, %33, 0] [1, 1, %31, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_15, %subview_16 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRMergeCopyChains (tilelangir-merge-copy-chains) ('func.func' operation: @sparseAttnMix) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = memref_ext.alloc_workspace() : memref<2x32x512xbf16, #hivm.address_space<gm>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<2x32x512xbf16, #hivm.address_space<gm>>
    %16 = memref_ext.alloc_workspace() : memref<2x16x32xf32, #hivm.address_space<gm>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xf32, #hivm.address_space<gm>>
    %17 = memref_ext.alloc_workspace() : memref<2x16x32xbf16, #hivm.address_space<gm>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<2x16x32xbf16, #hivm.address_space<gm>>
    %18 = memref_ext.alloc_workspace() : memref<2x16x512xf32, #hivm.address_space<gm>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %alloc_8 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      scf.for %arg19 = %c0_i32 to %26 step %c1_i32  : i32 {
        %alloc_17 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_31 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_32 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_33 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_34 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_35 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        scope.scope : () -> () {
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_22 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_24 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %36 = arith.divsi %12, %arg9 : i32
          %37 = arith.index_cast %36 : i32 to index
          %38 = arith.remsi %12, %arg9 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.muli %arg19, %c32_i32 : i32
          %41 = arith.index_cast %40 : i32 to index
          %42 = arith.subi %arg11, %40 : i32
          %43 = arith.minsi %42, %c32_i32 : i32
          %44 = arith.index_cast %43 : i32 to index
          %subview_36 = memref.subview %reinterpret_cast_5[%37, %39, %41] [1, 1, %44] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_37 = memref.subview %alloc_23[0] [%44] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          scf.for %arg20 = %c0_i32 to %43 step %c1_i32  : i32 {
            %45 = arith.index_cast %arg20 : i32 to index
            %46 = memref.load %alloc_23[%45] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %47 = arith.cmpi ne, %46, %c-1_i32 : i32
            scf.if %47 {
              %48 = arith.index_cast %arg20 : i32 to index
              memref.store %cst_2, %alloc_24[%c0, %48] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %49 = arith.divsi %12, %arg9 : i32
              %50 = arith.index_cast %49 : i32 to index
              %51 = arith.index_cast %46 : i32 to index
              %subview_39 = memref.subview %reinterpret_cast_6[%50, %51, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_40 = memref.subview %alloc_22[%48, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_39, %subview_40 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_38 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16, #hivm.address_space<gm>> to memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<gm>>
          memref.copy %alloc_22, %subview_38 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<gm>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %subview_36 = memref.subview %15[0, 0, 0] [1, 32, 512] [1, 1, 1] : memref<2x32x512xbf16, #hivm.address_space<gm>> to memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<gm>>
          memref.copy %subview_36, %alloc_17 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_17, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_19 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_37 = memref.subview %16[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xf32, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<gm>>
          memref.copy %alloc_19, %subview_37 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>> to memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<gm>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %34 = arith.muli %14, %c8_i32 : i32
        %35 = arith.index_cast %34 : i32 to index
        scope.scope : () -> () {
          %subview_36 = memref.subview %16[0, %35, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xf32, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_36, %alloc_25 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_37 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_38 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_37, %subview_38 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_25, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_29 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_39 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_40 = memref.subview %alloc_30[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_39, %subview_40 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_25, %alloc_31 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_32 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_41 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_42 = memref.subview %alloc_33[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_41, %subview_42 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_33 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_25, %alloc_34 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_10, %alloc_26 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_35 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_35, %alloc_27 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_20 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_43 = memref.subview %17[0, %35, 0] [1, 8, 32] [1, 1, 1] : memref<2x16x32xbf16, #hivm.address_space<gm>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_20, %subview_43 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %subview_36 = memref.subview %17[0, 0, 0] [1, 16, 32] [1, 1, 1] : memref<2x16x32xbf16, #hivm.address_space<gm>> to memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>>
          memref.copy %subview_36, %alloc_18 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<gm>> to memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 ins(%alloc_18, %alloc_17, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_21 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_37 = memref.subview %18[0, 0, 0] [1, 16, 512] [1, 1, 1] : memref<2x16x512xf32, #hivm.address_space<gm>> to memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<gm>>
          memref.copy %alloc_21, %subview_37 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>> to memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<gm>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %subview_36 = memref.subview %18[0, %35, 0] [1, 8, 512] [1, 1, 1] : memref<2x16x512xf32, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_36, %alloc_28 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_11, %alloc_26 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_11, %alloc_28 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %34 = arith.muli %arg18, %c2_i32 : i32
        %35 = arith.addi %34, %14 : i32
        %36 = arith.cmpi slt, %35, %c8_i32 : i32
        scf.if %36 {
          %37 = arith.muli %arg18, %c16_i32 : i32
          %38 = arith.muli %14, %c8_i32 : i32
          %39 = arith.addi %37, %38 : i32
          %40 = arith.addi %39, %arg19 : i32
          %41 = arith.index_cast %40 : i32 to index
          %42 = memref.load %reinterpret_cast_7[%41] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %43 = arith.index_cast %arg19 : i32 to index
          memref.store %42, %alloc_9[%43, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %37 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%37, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %27 = arith.muli %14, %c8_i32 : i32
      %28 = arith.subi %c64_i32, %27 : i32
      %29 = arith.subi %28, %23 : i32
      %30 = arith.minsi %29, %c8_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %subview_15 = memref.subview %alloc_12[0, 0] [%31, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %32 = arith.addi %23, %27 : i32
      %33 = arith.index_cast %32 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast_4[%20, %22, %33, 0] [1, 1, %31, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_15, %subview_16 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIREnableMultiBuffer (tilelangir-enable-multi-buffer) ('builtin.module' operation) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = memref_ext.alloc_workspace() : memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %16 = memref_ext.alloc_workspace() : memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %17 = memref_ext.alloc_workspace() : memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %18 = memref_ext.alloc_workspace() : memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %alloc_8 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      %c2_i32_15 = arith.constant 2 : i32
      %27 = arith.divsi %26, %c2_i32_15 : i32
      scf.for %arg19 = %c0_i32 to %27 step %c1_i32  : i32 {
        %alloc_18 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_20 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_21 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_22 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_23 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_25 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_31 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_32 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_33 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_34 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_35 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_36 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %c0_i32_37 = arith.constant 0 : i32
        %c2_i32_38 = arith.constant 2 : i32
        %c1_i32_39 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_37 to %c2_i32_38 step %c1_i32_39  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_23 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_25 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %41 = arith.divsi %12, %arg9 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = arith.remsi %12, %arg9 : i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = arith.muli %38, %c32_i32 : i32
          %46 = arith.index_cast %45 : i32 to index
          %47 = arith.subi %arg11, %45 : i32
          %48 = arith.minsi %47, %c32_i32 : i32
          %49 = arith.index_cast %48 : i32 to index
          %c32_i32_53 = arith.constant 32 : i32
          %50 = arith.muli %38, %c32_i32_53 : i32
          %51 = arith.index_cast %50 : i32 to index
          %subview_54 = memref.subview %reinterpret_cast_5[%42, %44, %51] [1, 1, %49] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_55 = memref.subview %alloc_24[0] [%49] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          memref.copy %subview_54, %subview_55 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          scf.for %arg21 = %c0_i32 to %48 step %c1_i32  : i32 {
            %52 = arith.index_cast %arg21 : i32 to index
            %53 = memref.load %alloc_24[%52] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %54 = arith.cmpi ne, %53, %c-1_i32 : i32
            scf.if %54 {
              %55 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_25[%c0, %55] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %56 = arith.divsi %12, %arg9 : i32
              %57 = arith.index_cast %56 : i32 to index
              %58 = arith.index_cast %53 : i32 to index
              %subview_57 = memref.subview %reinterpret_cast_6[%57, %58, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_58 = memref.subview %alloc_23[%55, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_57, %subview_58 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_56 = memref.subview %15[%40, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_56 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_23, %collapse_shape : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_40 = arith.constant 0 : i32
        %c2_i32_41 = arith.constant 2 : i32
        %c1_i32_42 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_40 to %c2_i32_41 step %c1_i32_42  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %15[%40, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_18 : memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_18, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_20 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_54 = memref.subview %16[%40, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_55 = memref.collapse_shape %subview_54 [[0, 1, 2], [3]] : memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_20, %collapse_shape_55 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>> to memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %35 = arith.muli %14, %c8_i32 : i32
        %36 = arith.index_cast %35 : i32 to index
        %c0_i32_43 = arith.constant 0 : i32
        %c2_i32_44 = arith.constant 2 : i32
        %c1_i32_45 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_43 to %c2_i32_44 step %c1_i32_45  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %16[%40, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_26 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_54 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_55 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_54, %subview_55 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_26, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_56 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_57 = memref.subview %alloc_31[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_56, %subview_57 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_31 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_26, %alloc_32 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_33 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_58 = memref.subview %alloc_25[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_59 = memref.subview %alloc_34[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_58, %subview_59 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_34 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_35 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_26, %alloc_35 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_10, %alloc_27 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_36 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_36, %alloc_28 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_60 = memref.subview %17[%40, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_61 = memref.collapse_shape %subview_60 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_21, %collapse_shape_61 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_46 = arith.constant 0 : i32
        %c2_i32_47 = arith.constant 2 : i32
        %c1_i32_48 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_46 to %c2_i32_47 step %c1_i32_48  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %17[%40, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_19 : memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 ins(%alloc_19, %alloc_18, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_22 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_54 = memref.subview %18[%40, 0, 0, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_55 = memref.collapse_shape %subview_54 [[0, 1, 2], [3]] : memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_22, %collapse_shape_55 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>> to memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_49 = arith.constant 0 : i32
        %c2_i32_50 = arith.constant 2 : i32
        %c1_i32_51 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_49 to %c2_i32_50 step %c1_i32_51  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %18[%40, 0, %36, 0] [1, 1, 8, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_29 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_11, %alloc_27 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_11, %alloc_29 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %35 = arith.muli %arg18, %c2_i32 : i32
        %36 = arith.addi %35, %14 : i32
        %37 = arith.cmpi slt, %36, %c8_i32 : i32
        scf.if %37 {
          %38 = arith.muli %arg18, %c16_i32 : i32
          %39 = arith.muli %14, %c8_i32 : i32
          %40 = arith.addi %38, %39 : i32
          %41 = arith.addi %40, %arg19 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = memref.load %reinterpret_cast_7[%42] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %44 = arith.index_cast %arg19 : i32 to index
          memref.store %43, %alloc_9[%44, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %38 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%38, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %28 = arith.muli %14, %c8_i32 : i32
      %29 = arith.subi %c64_i32, %28 : i32
      %30 = arith.subi %29, %23 : i32
      %31 = arith.minsi %30, %c8_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview_16 = memref.subview %alloc_12[0, 0] [%32, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %33 = arith.addi %23, %28 : i32
      %34 = arith.index_cast %33 : i32 to index
      %subview_17 = memref.subview %reinterpret_cast_4[%20, %22, %34, 0] [1, 1, %32, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_16, %subview_17 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIREnableLocalBuffer (tilelangir-enable-local-buffer) ('builtin.module' operation) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = memref_ext.alloc_workspace() : memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %16 = memref_ext.alloc_workspace() : memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %17 = memref_ext.alloc_workspace() : memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %18 = memref_ext.alloc_workspace() : memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %alloc_8 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      %c2_i32_15 = arith.constant 2 : i32
      %27 = arith.divsi %26, %c2_i32_15 : i32
      scf.for %arg19 = %c0_i32 to %27 step %c1_i32  : i32 {
        %alloc_18 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_20 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_21 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_22 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_23 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_25 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_31 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_32 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_33 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_34 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_35 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_36 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %c0_i32_37 = arith.constant 0 : i32
        %c2_i32_38 = arith.constant 2 : i32
        %c1_i32_39 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_37 to %c2_i32_38 step %c1_i32_39  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_23 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_25 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %41 = arith.divsi %12, %arg9 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = arith.remsi %12, %arg9 : i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = arith.muli %38, %c32_i32 : i32
          %46 = arith.index_cast %45 : i32 to index
          %47 = arith.subi %arg11, %45 : i32
          %48 = arith.minsi %47, %c32_i32 : i32
          %49 = arith.index_cast %48 : i32 to index
          %c32_i32_53 = arith.constant 32 : i32
          %50 = arith.muli %38, %c32_i32_53 : i32
          %51 = arith.index_cast %50 : i32 to index
          %subview_54 = memref.subview %reinterpret_cast_5[%42, %44, %51] [1, 1, %49] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_55 = memref.subview %alloc_24[0] [%49] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          memref.copy %subview_54, %subview_55 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          scf.for %arg21 = %c0_i32 to %48 step %c1_i32  : i32 {
            %52 = arith.index_cast %arg21 : i32 to index
            %53 = memref.load %alloc_24[%52] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %54 = arith.cmpi ne, %53, %c-1_i32 : i32
            scf.if %54 {
              %55 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_25[%c0, %55] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %56 = arith.divsi %12, %arg9 : i32
              %57 = arith.index_cast %56 : i32 to index
              %58 = arith.index_cast %53 : i32 to index
              %subview_57 = memref.subview %reinterpret_cast_6[%57, %58, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_58 = memref.subview %alloc_23[%55, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_57, %subview_58 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_56 = memref.subview %15[%40, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_56 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_23, %collapse_shape : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_40 = arith.constant 0 : i32
        %c2_i32_41 = arith.constant 2 : i32
        %c1_i32_42 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_40 to %c2_i32_41 step %c1_i32_42  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %15[%40, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_18 : memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_18, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_20 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_54 = memref.subview %16[%40, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_55 = memref.collapse_shape %subview_54 [[0, 1, 2], [3]] : memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_20, %collapse_shape_55 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>> to memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %35 = arith.muli %14, %c8_i32 : i32
        %36 = arith.index_cast %35 : i32 to index
        %c0_i32_43 = arith.constant 0 : i32
        %c2_i32_44 = arith.constant 2 : i32
        %c1_i32_45 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_43 to %c2_i32_44 step %c1_i32_45  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %16[%40, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_26 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_54 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_55 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_54, %subview_55 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_26, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_56 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_57 = memref.subview %alloc_31[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_56, %subview_57 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_31 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_26, %alloc_32 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_33 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_58 = memref.subview %alloc_25[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_59 = memref.subview %alloc_34[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_58, %subview_59 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_34 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_35 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_26, %alloc_35 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_10, %alloc_27 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_36 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_36, %alloc_28 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_60 = memref.subview %17[%40, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_61 = memref.collapse_shape %subview_60 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_21, %collapse_shape_61 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_46 = arith.constant 0 : i32
        %c2_i32_47 = arith.constant 2 : i32
        %c1_i32_48 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_46 to %c2_i32_47 step %c1_i32_48  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %17[%40, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_19 : memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 ins(%alloc_19, %alloc_18, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_22 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_54 = memref.subview %18[%40, 0, 0, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_55 = memref.collapse_shape %subview_54 [[0, 1, 2], [3]] : memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_22, %collapse_shape_55 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>> to memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_49 = arith.constant 0 : i32
        %c2_i32_50 = arith.constant 2 : i32
        %c1_i32_51 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_49 to %c2_i32_50 step %c1_i32_51  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %18[%40, 0, %36, 0] [1, 1, 8, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_29 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_11, %alloc_27 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_11, %alloc_29 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %35 = arith.muli %arg18, %c2_i32 : i32
        %36 = arith.addi %35, %14 : i32
        %37 = arith.cmpi slt, %36, %c8_i32 : i32
        scf.if %37 {
          %38 = arith.muli %arg18, %c16_i32 : i32
          %39 = arith.muli %14, %c8_i32 : i32
          %40 = arith.addi %38, %39 : i32
          %41 = arith.addi %40, %arg19 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = memref.load %reinterpret_cast_7[%42] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %44 = arith.index_cast %arg19 : i32 to index
          memref.store %43, %alloc_9[%44, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %38 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%38, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %28 = arith.muli %14, %c8_i32 : i32
      %29 = arith.subi %c64_i32, %28 : i32
      %30 = arith.subi %29, %23 : i32
      %31 = arith.minsi %30, %c8_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview_16 = memref.subview %alloc_12[0, 0] [%32, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %33 = arith.addi %23, %28 : i32
      %34 = arith.index_cast %33 : i32 to index
      %subview_17 = memref.subview %reinterpret_cast_4[%20, %22, %34, 0] [1, 1, %32, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_16, %subview_17 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRSpecializeCube (tilelangir-specialize-cube) ('func.func' operation: @sparseAttnMix) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = memref_ext.alloc_workspace() : memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %16 = memref_ext.alloc_workspace() : memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %17 = memref_ext.alloc_workspace() : memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %18 = memref_ext.alloc_workspace() : memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %alloc_8 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      %c2_i32_15 = arith.constant 2 : i32
      %27 = arith.divsi %26, %c2_i32_15 : i32
      scf.for %arg19 = %c0_i32 to %27 step %c1_i32  : i32 {
        %alloc_18 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_20 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_21 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_22 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_23 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_25 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_31 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_32 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_33 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_34 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_35 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_36 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %c0_i32_37 = arith.constant 0 : i32
        %c2_i32_38 = arith.constant 2 : i32
        %c1_i32_39 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_37 to %c2_i32_38 step %c1_i32_39  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_23 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_25 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %41 = arith.divsi %12, %arg9 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = arith.remsi %12, %arg9 : i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = arith.muli %38, %c32_i32 : i32
          %46 = arith.index_cast %45 : i32 to index
          %47 = arith.subi %arg11, %45 : i32
          %48 = arith.minsi %47, %c32_i32 : i32
          %49 = arith.index_cast %48 : i32 to index
          %c32_i32_53 = arith.constant 32 : i32
          %50 = arith.muli %38, %c32_i32_53 : i32
          %51 = arith.index_cast %50 : i32 to index
          %subview_54 = memref.subview %reinterpret_cast_5[%42, %44, %51] [1, 1, %49] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_55 = memref.subview %alloc_24[0] [%49] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_54 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_55 : memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          scf.for %arg21 = %c0_i32 to %48 step %c1_i32  : i32 {
            %52 = arith.index_cast %arg21 : i32 to index
            %53 = memref.load %alloc_24[%52] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %54 = arith.cmpi ne, %53, %c-1_i32 : i32
            scf.if %54 {
              %55 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_25[%c0, %55] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %56 = arith.divsi %12, %arg9 : i32
              %57 = arith.index_cast %56 : i32 to index
              %58 = arith.index_cast %53 : i32 to index
              %subview_57 = memref.subview %reinterpret_cast_6[%57, %58, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_58 = memref.subview %alloc_23[%55, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_57, %subview_58 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_56 = memref.subview %15[%40, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_56 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_23, %collapse_shape : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_40 = arith.constant 0 : i32
        %c2_i32_41 = arith.constant 2 : i32
        %c1_i32_42 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_40 to %c2_i32_41 step %c1_i32_42  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %15[%40, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_18 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_18, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_20 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_54 = memref.subview %16[%40, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_55 = memref.collapse_shape %subview_54 [[0, 1, 2], [3]] : memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_20 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_55 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %35 = arith.muli %14, %c8_i32 : i32
        %36 = arith.index_cast %35 : i32 to index
        %c0_i32_43 = arith.constant 0 : i32
        %c2_i32_44 = arith.constant 2 : i32
        %c1_i32_45 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_43 to %c2_i32_44 step %c1_i32_45  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %16[%40, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_26 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_54 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_55 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_54, %subview_55 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_26, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_56 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_57 = memref.subview %alloc_31[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_56, %subview_57 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_31 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_26, %alloc_32 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_33 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_58 = memref.subview %alloc_25[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_59 = memref.subview %alloc_34[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_58, %subview_59 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_34 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_35 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_26, %alloc_35 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_10, %alloc_27 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_36 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_36, %alloc_28 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_60 = memref.subview %17[%40, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_61 = memref.collapse_shape %subview_60 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_21, %collapse_shape_61 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_46 = arith.constant 0 : i32
        %c2_i32_47 = arith.constant 2 : i32
        %c1_i32_48 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_46 to %c2_i32_47 step %c1_i32_48  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %17[%40, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_19 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_19, %alloc_18, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_22 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_54 = memref.subview %18[%40, 0, 0, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_55 = memref.collapse_shape %subview_54 [[0, 1, 2], [3]] : memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_22 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_55 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_49 = arith.constant 0 : i32
        %c2_i32_50 = arith.constant 2 : i32
        %c1_i32_51 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_49 to %c2_i32_50 step %c1_i32_51  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %18[%40, 0, %36, 0] [1, 1, 8, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_29 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_11, %alloc_27 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_11, %alloc_29 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %35 = arith.muli %arg18, %c2_i32 : i32
        %36 = arith.addi %35, %14 : i32
        %37 = arith.cmpi slt, %36, %c8_i32 : i32
        scf.if %37 {
          %38 = arith.muli %arg18, %c16_i32 : i32
          %39 = arith.muli %14, %c8_i32 : i32
          %40 = arith.addi %38, %39 : i32
          %41 = arith.addi %40, %arg19 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = memref.load %reinterpret_cast_7[%42] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %44 = arith.index_cast %arg19 : i32 to index
          memref.store %43, %alloc_9[%44, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %38 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%38, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %28 = arith.muli %14, %c8_i32 : i32
      %29 = arith.subi %c64_i32, %28 : i32
      %30 = arith.subi %29, %23 : i32
      %31 = arith.minsi %30, %c8_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview_16 = memref.subview %alloc_12[0, 0] [%32, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %33 = arith.addi %23, %28 : i32
      %34 = arith.index_cast %33 : i32 to index
      %subview_17 = memref.subview %reinterpret_cast_4[%20, %22, %34, 0] [1, 1, %32, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_16, %subview_17 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After BindWorkSpaceArg (hivm-bind-workspace-arg) ('func.func' operation: @sparseAttnMix) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %16 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %17 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %18 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %alloc_8 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      %c2_i32_15 = arith.constant 2 : i32
      %27 = arith.divsi %26, %c2_i32_15 : i32
      scf.for %arg19 = %c0_i32 to %27 step %c1_i32  : i32 {
        %alloc_18 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_20 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_21 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_22 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_23 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_25 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_31 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_32 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_33 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_34 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_35 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_36 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %c0_i32_37 = arith.constant 0 : i32
        %c2_i32_38 = arith.constant 2 : i32
        %c1_i32_39 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_37 to %c2_i32_38 step %c1_i32_39  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_23 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_25 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %41 = arith.divsi %12, %arg9 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = arith.remsi %12, %arg9 : i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = arith.muli %38, %c32_i32 : i32
          %46 = arith.index_cast %45 : i32 to index
          %47 = arith.subi %arg11, %45 : i32
          %48 = arith.minsi %47, %c32_i32 : i32
          %49 = arith.index_cast %48 : i32 to index
          %c32_i32_53 = arith.constant 32 : i32
          %50 = arith.muli %38, %c32_i32_53 : i32
          %51 = arith.index_cast %50 : i32 to index
          %subview_54 = memref.subview %reinterpret_cast_5[%42, %44, %51] [1, 1, %49] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_55 = memref.subview %alloc_24[0] [%49] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_54 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_55 : memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          scf.for %arg21 = %c0_i32 to %48 step %c1_i32  : i32 {
            %52 = arith.index_cast %arg21 : i32 to index
            %53 = memref.load %alloc_24[%52] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %54 = arith.cmpi ne, %53, %c-1_i32 : i32
            scf.if %54 {
              %55 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_25[%c0, %55] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %56 = arith.divsi %12, %arg9 : i32
              %57 = arith.index_cast %56 : i32 to index
              %58 = arith.index_cast %53 : i32 to index
              %subview_57 = memref.subview %reinterpret_cast_6[%57, %58, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_58 = memref.subview %alloc_23[%55, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_57, %subview_58 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_56 = memref.subview %15[%40, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_56 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_23, %collapse_shape : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_40 = arith.constant 0 : i32
        %c2_i32_41 = arith.constant 2 : i32
        %c1_i32_42 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_40 to %c2_i32_41 step %c1_i32_42  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %15[%40, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_18 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_18, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_20 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_54 = memref.subview %16[%40, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_55 = memref.collapse_shape %subview_54 [[0, 1, 2], [3]] : memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_20 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_55 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %35 = arith.muli %14, %c8_i32 : i32
        %36 = arith.index_cast %35 : i32 to index
        %c0_i32_43 = arith.constant 0 : i32
        %c2_i32_44 = arith.constant 2 : i32
        %c1_i32_45 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_43 to %c2_i32_44 step %c1_i32_45  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %16[%40, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_26 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_54 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_55 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_54, %subview_55 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_26, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_56 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_57 = memref.subview %alloc_31[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_56, %subview_57 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_31 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_26, %alloc_32 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_33 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_58 = memref.subview %alloc_25[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_59 = memref.subview %alloc_34[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_58, %subview_59 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_34 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_35 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_26, %alloc_35 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_10, %alloc_27 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_36 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_36, %alloc_28 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_60 = memref.subview %17[%40, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_61 = memref.collapse_shape %subview_60 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_21, %collapse_shape_61 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_46 = arith.constant 0 : i32
        %c2_i32_47 = arith.constant 2 : i32
        %c1_i32_48 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_46 to %c2_i32_47 step %c1_i32_48  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %17[%40, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_19 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_19, %alloc_18, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_22 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_54 = memref.subview %18[%40, 0, 0, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_55 = memref.collapse_shape %subview_54 [[0, 1, 2], [3]] : memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_22 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_55 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_49 = arith.constant 0 : i32
        %c2_i32_50 = arith.constant 2 : i32
        %c1_i32_51 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_49 to %c2_i32_50 step %c1_i32_51  : i32 {
          %c2_i32_52 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_52 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_53 = memref.subview %18[%40, 0, %36, 0] [1, 1, 8, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_53 [[0, 1, 2], [3]] : memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_29 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_11, %alloc_27 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_11, %alloc_29 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %35 = arith.muli %arg18, %c2_i32 : i32
        %36 = arith.addi %35, %14 : i32
        %37 = arith.cmpi slt, %36, %c8_i32 : i32
        scf.if %37 {
          %38 = arith.muli %arg18, %c16_i32 : i32
          %39 = arith.muli %14, %c8_i32 : i32
          %40 = arith.addi %38, %39 : i32
          %41 = arith.addi %40, %arg19 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = memref.load %reinterpret_cast_7[%42] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %44 = arith.index_cast %arg19 : i32 to index
          memref.store %43, %alloc_9[%44, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %38 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%38, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %28 = arith.muli %14, %c8_i32 : i32
      %29 = arith.subi %c64_i32, %28 : i32
      %30 = arith.subi %29, %23 : i32
      %31 = arith.minsi %30, %c8_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview_16 = memref.subview %alloc_12[0, 0] [%32, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %33 = arith.addi %23, %28 : i32
      %34 = arith.index_cast %33 : i32 to index
      %subview_17 = memref.subview %reinterpret_cast_4[%20, %22, %34, 0] [1, 1, %32, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_16, %subview_17 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRPlanWorkspaceMemory (tilelangir-plan-workspace-memory) ('func.func' operation: @sparseAttnMix) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %c0_8 = arith.constant 0 : index
    %15 = memref_ext.alloc_workspace() from %arg2 offset = [%c0_8] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %c131072 = arith.constant 131072 : index
    %16 = memref_ext.alloc_workspace() from %arg2 offset = [%c131072] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %c139264 = arith.constant 139264 : index
    %17 = memref_ext.alloc_workspace() from %arg2 offset = [%c139264] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %c143360 = arith.constant 143360 : index
    %18 = memref_ext.alloc_workspace() from %arg2 offset = [%c143360] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_15 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_9 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      %c2_i32_16 = arith.constant 2 : i32
      %27 = arith.divsi %26, %c2_i32_16 : i32
      scf.for %arg19 = %c0_i32 to %27 step %c1_i32  : i32 {
        %alloc_19 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_20 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_21 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_22 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_23 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_24 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_25 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_26 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_31 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_32 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_33 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_34 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_35 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_36 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_37 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %c0_i32_38 = arith.constant 0 : i32
        %c2_i32_39 = arith.constant 2 : i32
        %c1_i32_40 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_38 to %c2_i32_39 step %c1_i32_40  : i32 {
          %c2_i32_53 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_53 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_24 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_26 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %41 = arith.divsi %12, %arg9 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = arith.remsi %12, %arg9 : i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = arith.muli %38, %c32_i32 : i32
          %46 = arith.index_cast %45 : i32 to index
          %47 = arith.subi %arg11, %45 : i32
          %48 = arith.minsi %47, %c32_i32 : i32
          %49 = arith.index_cast %48 : i32 to index
          %c32_i32_54 = arith.constant 32 : i32
          %50 = arith.muli %38, %c32_i32_54 : i32
          %51 = arith.index_cast %50 : i32 to index
          %subview_55 = memref.subview %reinterpret_cast_5[%42, %44, %51] [1, 1, %49] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_56 = memref.subview %alloc_25[0] [%49] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_55 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_56 : memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          scf.for %arg21 = %c0_i32 to %48 step %c1_i32  : i32 {
            %52 = arith.index_cast %arg21 : i32 to index
            %53 = memref.load %alloc_25[%52] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %54 = arith.cmpi ne, %53, %c-1_i32 : i32
            scf.if %54 {
              %55 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_26[%c0, %55] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %56 = arith.divsi %12, %arg9 : i32
              %57 = arith.index_cast %56 : i32 to index
              %58 = arith.index_cast %53 : i32 to index
              %subview_58 = memref.subview %reinterpret_cast_6[%57, %58, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_59 = memref.subview %alloc_24[%55, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_58, %subview_59 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_57 = memref.subview %15[%40, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_57 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_24, %collapse_shape : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_41 = arith.constant 0 : i32
        %c2_i32_42 = arith.constant 2 : i32
        %c1_i32_43 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_41 to %c2_i32_42 step %c1_i32_43  : i32 {
          %c2_i32_53 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_53 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_54 = memref.subview %15[%40, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_54 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_19 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_19, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_21 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_55 = memref.subview %16[%40, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_56 = memref.collapse_shape %subview_55 [[0, 1, 2], [3]] : memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_21 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_56 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %35 = arith.muli %14, %c8_i32 : i32
        %36 = arith.index_cast %35 : i32 to index
        %c0_i32_44 = arith.constant 0 : i32
        %c2_i32_45 = arith.constant 2 : i32
        %c1_i32_46 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_44 to %c2_i32_45 step %c1_i32_46  : i32 {
          %c2_i32_53 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_53 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_54 = memref.subview %16[%40, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_54 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_27 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_55 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_56 = memref.subview %alloc_10[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_55, %subview_56 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_27, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_10, %alloc_9 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_31 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_57 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_58 = memref.subview %alloc_32[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_57, %subview_58 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_32 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_27, %alloc_33 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_34 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_59 = memref.subview %alloc_26[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_60 = memref.subview %alloc_35[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_59, %subview_60 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_35 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_36 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_27, %alloc_36 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_11, %alloc_28 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_37 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_37, %alloc_29 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_61 = memref.subview %17[%40, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_62 = memref.collapse_shape %subview_61 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_22, %collapse_shape_62 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_47 = arith.constant 0 : i32
        %c2_i32_48 = arith.constant 2 : i32
        %c1_i32_49 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_47 to %c2_i32_48 step %c1_i32_49  : i32 {
          %c2_i32_53 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_53 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_54 = memref.subview %17[%40, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_54 [[0, 1, 2], [3]] : memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_20 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_20, %alloc_19, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_23 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_55 = memref.subview %18[%40, 0, 0, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_56 = memref.collapse_shape %subview_55 [[0, 1, 2], [3]] : memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_23 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_56 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_50 = arith.constant 0 : i32
        %c2_i32_51 = arith.constant 2 : i32
        %c1_i32_52 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_50 to %c2_i32_51 step %c1_i32_52  : i32 {
          %c2_i32_53 = arith.constant 2 : i32
          %37 = arith.muli %arg19, %c2_i32_53 : i32
          %38 = arith.addi %37, %arg20 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.index_cast %arg20 : i32 to index
          %subview_54 = memref.subview %18[%40, 0, %36, 0] [1, 1, 8, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_54 [[0, 1, 2], [3]] : memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_30 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_12, %alloc_28 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_12, %alloc_30 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %35 = arith.muli %arg18, %c2_i32 : i32
        %36 = arith.addi %35, %14 : i32
        %37 = arith.cmpi slt, %36, %c8_i32 : i32
        scf.if %37 {
          %38 = arith.muli %arg18, %c16_i32 : i32
          %39 = arith.muli %14, %c8_i32 : i32
          %40 = arith.addi %38, %39 : i32
          %41 = arith.addi %40, %arg19 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = memref.load %reinterpret_cast_7[%42] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %44 = arith.index_cast %arg19 : i32 to index
          memref.store %43, %alloc_10[%44, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %38 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_10[%38, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_10, %alloc_9 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_11, %alloc_15 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_12, %alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %28 = arith.muli %14, %c8_i32 : i32
      %29 = arith.subi %c64_i32, %28 : i32
      %30 = arith.subi %29, %23 : i32
      %31 = arith.minsi %30, %c8_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview_17 = memref.subview %alloc_13[0, 0] [%32, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %33 = arith.addi %23, %28 : i32
      %34 = arith.index_cast %33 : i32 to index
      %subview_18 = memref.subview %reinterpret_cast_4[%20, %22, %34, 0] [1, 1, %32, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_17, %subview_18 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRInsertCVSync (tilelangir-insert-cv-sync) ('builtin.module' operation) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %c0_8 = arith.constant 0 : index
    %15 = memref_ext.alloc_workspace() from %arg2 offset = [%c0_8] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %c131072 = arith.constant 131072 : index
    %16 = memref_ext.alloc_workspace() from %arg2 offset = [%c131072] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %c139264 = arith.constant 139264 : index
    %17 = memref_ext.alloc_workspace() from %arg2 offset = [%c139264] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %c143360 = arith.constant 143360 : index
    %18 = memref_ext.alloc_workspace() from %arg2 offset = [%c143360] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_15 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_9 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      %c2_i32_16 = arith.constant 2 : i32
      %27 = arith.divsi %26, %c2_i32_16 : i32
      %c0_i32_17 = arith.constant 0 : i32
      %c1_i32_18 = arith.constant 1 : i32
      %c2_i32_19 = arith.constant 2 : i32
      scf.for %arg19 = %c0_i32_17 to %c2_i32_19 step %c1_i32_18  : i32 {
        %35 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %35 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scf.for %arg19 = %c0_i32 to %27 step %c1_i32  : i32 {
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_23 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_24 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_25 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_27 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_29 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_31 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_32 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_33 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_34 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_35 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_36 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_37 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_38 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_39 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_40 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %c0_i32_41 = arith.constant 0 : i32
        %c2_i32_42 = arith.constant 2 : i32
        %c1_i32_43 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_41 to %c2_i32_42 step %c1_i32_43  : i32 {
          %37 = arith.extsi %arg20 : i32 to i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %37
          %c2_i32_56 = arith.constant 2 : i32
          %38 = arith.muli %arg19, %c2_i32_56 : i32
          %39 = arith.addi %38, %arg20 : i32
          %40 = arith.index_cast %39 : i32 to index
          %41 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_27 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_29 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %42 = arith.divsi %12, %arg9 : i32
          %43 = arith.index_cast %42 : i32 to index
          %44 = arith.remsi %12, %arg9 : i32
          %45 = arith.index_cast %44 : i32 to index
          %46 = arith.muli %39, %c32_i32 : i32
          %47 = arith.index_cast %46 : i32 to index
          %48 = arith.subi %arg11, %46 : i32
          %49 = arith.minsi %48, %c32_i32 : i32
          %50 = arith.index_cast %49 : i32 to index
          %c32_i32_57 = arith.constant 32 : i32
          %51 = arith.muli %39, %c32_i32_57 : i32
          %52 = arith.index_cast %51 : i32 to index
          %subview_58 = memref.subview %reinterpret_cast_5[%43, %45, %52] [1, 1, %50] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_59 = memref.subview %alloc_28[0] [%50] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_58 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_59 : memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          scf.for %arg21 = %c0_i32 to %49 step %c1_i32  : i32 {
            %53 = arith.index_cast %arg21 : i32 to index
            %54 = memref.load %alloc_28[%53] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %55 = arith.cmpi ne, %54, %c-1_i32 : i32
            scf.if %55 {
              %56 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_29[%c0, %56] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %57 = arith.divsi %12, %arg9 : i32
              %58 = arith.index_cast %57 : i32 to index
              %59 = arith.index_cast %54 : i32 to index
              %subview_61 = memref.subview %reinterpret_cast_6[%58, %59, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_62 = memref.subview %alloc_27[%56, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_61, %subview_62 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_60 = memref.subview %15[%41, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_60 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_27, %collapse_shape : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_44 = arith.constant 0 : i32
        %c2_i32_45 = arith.constant 2 : i32
        %c1_i32_46 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_44 to %c2_i32_45 step %c1_i32_46  : i32 {
          %37 = arith.extsi %arg20 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c2_i64 = arith.constant 2 : i64
          %38 = arith.addi %37, %c2_i64 : i64
          %39 = arith.remsi %38, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %37
          %c2_i32_56 = arith.constant 2 : i32
          %40 = arith.muli %arg19, %c2_i32_56 : i32
          %41 = arith.addi %40, %arg20 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = arith.index_cast %arg20 : i32 to index
          %subview_57 = memref.subview %15[%43, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_57 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_22 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_22, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_24 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_58 = memref.subview %16[%43, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_59 = memref.collapse_shape %subview_58 [[0, 1, 2], [3]] : memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_24 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_59 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %39 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %35 = arith.muli %14, %c8_i32 : i32
        %36 = arith.index_cast %35 : i32 to index
        %c0_i32_47 = arith.constant 0 : i32
        %c2_i32_48 = arith.constant 2 : i32
        %c1_i32_49 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_47 to %c2_i32_48 step %c1_i32_49  : i32 {
          %37 = arith.extsi %arg20 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c2_i64 = arith.constant 2 : i64
          %38 = arith.addi %37, %c2_i64 : i64
          %39 = arith.remsi %38, %c16_i64 : i64
          %c2_i64_56 = arith.constant 2 : i64
          %40 = arith.addi %37, %c2_i64_56 : i64
          %41 = arith.remsi %40, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %39
          %c2_i32_57 = arith.constant 2 : i32
          %42 = arith.muli %arg19, %c2_i32_57 : i32
          %43 = arith.addi %42, %arg20 : i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = arith.index_cast %arg20 : i32 to index
          %subview_58 = memref.subview %16[%45, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_58 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_30 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_59 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_60 = memref.subview %alloc_10[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_59, %subview_60 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_30, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_10, %alloc_9 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_34 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_34 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_61 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_62 = memref.subview %alloc_35[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_61, %subview_62 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_35 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_36 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_30, %alloc_36 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_37 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_37 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_63 = memref.subview %alloc_29[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_64 = memref.subview %alloc_38[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_63, %subview_64 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_38 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_39 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_30, %alloc_39 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_11, %alloc_31 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_40 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_40, %alloc_32 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_65 = memref.subview %17[%45, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_66 = memref.collapse_shape %subview_65 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_25, %collapse_shape_66 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %41 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_50 = arith.constant 0 : i32
        %c2_i32_51 = arith.constant 2 : i32
        %c1_i32_52 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_50 to %c2_i32_51 step %c1_i32_52  : i32 {
          %37 = arith.extsi %arg20 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c2_i64 = arith.constant 2 : i64
          %38 = arith.addi %37, %c2_i64 : i64
          %39 = arith.remsi %38, %c16_i64 : i64
          %c4_i64 = arith.constant 4 : i64
          %40 = arith.addi %37, %c4_i64 : i64
          %41 = arith.remsi %40, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %39
          %c2_i32_56 = arith.constant 2 : i32
          %42 = arith.muli %arg19, %c2_i32_56 : i32
          %43 = arith.addi %42, %arg20 : i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = arith.index_cast %arg20 : i32 to index
          %subview_57 = memref.subview %17[%45, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_57 [[0, 1, 2], [3]] : memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_23 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_23, %alloc_22, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_26 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_58 = memref.subview %18[%45, 0, 0, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_59 = memref.collapse_shape %subview_58 [[0, 1, 2], [3]] : memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_26 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_59 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %41 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_53 = arith.constant 0 : i32
        %c2_i32_54 = arith.constant 2 : i32
        %c1_i32_55 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_53 to %c2_i32_54 step %c1_i32_55  : i32 {
          %37 = arith.extsi %arg20 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c4_i64 = arith.constant 4 : i64
          %38 = arith.addi %37, %c4_i64 : i64
          %39 = arith.remsi %38, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %39
          %c2_i32_56 = arith.constant 2 : i32
          %40 = arith.muli %arg19, %c2_i32_56 : i32
          %41 = arith.addi %40, %arg20 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = arith.index_cast %arg20 : i32 to index
          %subview_57 = memref.subview %18[%43, 0, %36, 0] [1, 1, 8, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_57 [[0, 1, 2], [3]] : memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_33 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_12, %alloc_31 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_12, %alloc_33 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32_17 to %c2_i32_19 step %c1_i32_18  : i32 {
        %35 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %35
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %35 = arith.muli %arg18, %c2_i32 : i32
        %36 = arith.addi %35, %14 : i32
        %37 = arith.cmpi slt, %36, %c8_i32 : i32
        scf.if %37 {
          %38 = arith.muli %arg18, %c16_i32 : i32
          %39 = arith.muli %14, %c8_i32 : i32
          %40 = arith.addi %38, %39 : i32
          %41 = arith.addi %40, %arg19 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = memref.load %reinterpret_cast_7[%42] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %44 = arith.index_cast %arg19 : i32 to index
          memref.store %43, %alloc_10[%44, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %38 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_10[%38, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_10, %alloc_9 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_11, %alloc_15 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_12, %alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %28 = arith.muli %14, %c8_i32 : i32
      %29 = arith.subi %c64_i32, %28 : i32
      %30 = arith.subi %29, %23 : i32
      %31 = arith.minsi %30, %c8_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview_20 = memref.subview %alloc_13[0, 0] [%32, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %33 = arith.addi %23, %28 : i32
      %34 = arith.index_cast %33 : i32 to index
      %subview_21 = memref.subview %reinterpret_cast_4[%20, %22, %34, 0] [1, 1, %32, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_20, %subview_21 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After InsertInferWorkSpaceSizeFunc (hivm-insert-infer-workspace-size-func) ('func.func' operation: @sparseAttnMix) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c274432 = arith.constant 274432 : index
    return %c274432 : index
  }
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %c0_8 = arith.constant 0 : index
    %15 = memref_ext.alloc_workspace() from %arg2 offset = [%c0_8] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %c131072 = arith.constant 131072 : index
    %16 = memref_ext.alloc_workspace() from %arg2 offset = [%c131072] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %c139264 = arith.constant 139264 : index
    %17 = memref_ext.alloc_workspace() from %arg2 offset = [%c139264] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %c143360 = arith.constant 143360 : index
    %18 = memref_ext.alloc_workspace() from %arg2 offset = [%c143360] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_15 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_9 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %19 = arith.divsi %12, %arg9 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.remsi %12, %arg9 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = arith.muli %arg18, %c16_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %reinterpret_cast[%20, %22, %24, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %25 = arith.addi %arg11, %c31_i32 : i32
      %26 = arith.divsi %25, %c32_i32 : i32
      %c2_i32_16 = arith.constant 2 : i32
      %27 = arith.divsi %26, %c2_i32_16 : i32
      %c0_i32_17 = arith.constant 0 : i32
      %c1_i32_18 = arith.constant 1 : i32
      %c2_i32_19 = arith.constant 2 : i32
      scf.for %arg19 = %c0_i32_17 to %c2_i32_19 step %c1_i32_18  : i32 {
        %35 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %35 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scf.for %arg19 = %c0_i32 to %27 step %c1_i32  : i32 {
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_23 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_24 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_25 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_27 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_29 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_31 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_32 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_33 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_34 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_35 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_36 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_37 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_38 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_39 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_40 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %c0_i32_41 = arith.constant 0 : i32
        %c2_i32_42 = arith.constant 2 : i32
        %c1_i32_43 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_41 to %c2_i32_42 step %c1_i32_43  : i32 {
          %37 = arith.extsi %arg20 : i32 to i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %37
          %c2_i32_56 = arith.constant 2 : i32
          %38 = arith.muli %arg19, %c2_i32_56 : i32
          %39 = arith.addi %38, %arg20 : i32
          %40 = arith.index_cast %39 : i32 to index
          %41 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_27 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_29 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %42 = arith.divsi %12, %arg9 : i32
          %43 = arith.index_cast %42 : i32 to index
          %44 = arith.remsi %12, %arg9 : i32
          %45 = arith.index_cast %44 : i32 to index
          %46 = arith.muli %39, %c32_i32 : i32
          %47 = arith.index_cast %46 : i32 to index
          %48 = arith.subi %arg11, %46 : i32
          %49 = arith.minsi %48, %c32_i32 : i32
          %50 = arith.index_cast %49 : i32 to index
          %c32_i32_57 = arith.constant 32 : i32
          %51 = arith.muli %39, %c32_i32_57 : i32
          %52 = arith.index_cast %51 : i32 to index
          %subview_58 = memref.subview %reinterpret_cast_5[%43, %45, %52] [1, 1, %50] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_59 = memref.subview %alloc_28[0] [%50] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_58 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_59 : memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          scf.for %arg21 = %c0_i32 to %49 step %c1_i32  : i32 {
            %53 = arith.index_cast %arg21 : i32 to index
            %54 = memref.load %alloc_28[%53] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %55 = arith.cmpi ne, %54, %c-1_i32 : i32
            scf.if %55 {
              %56 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_29[%c0, %56] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %57 = arith.divsi %12, %arg9 : i32
              %58 = arith.index_cast %57 : i32 to index
              %59 = arith.index_cast %54 : i32 to index
              %subview_61 = memref.subview %reinterpret_cast_6[%58, %59, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_62 = memref.subview %alloc_27[%56, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_61, %subview_62 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_60 = memref.subview %15[%41, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_60 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_27, %collapse_shape : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_44 = arith.constant 0 : i32
        %c2_i32_45 = arith.constant 2 : i32
        %c1_i32_46 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_44 to %c2_i32_45 step %c1_i32_46  : i32 {
          %37 = arith.extsi %arg20 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c2_i64 = arith.constant 2 : i64
          %38 = arith.addi %37, %c2_i64 : i64
          %39 = arith.remsi %38, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %37
          %c2_i32_56 = arith.constant 2 : i32
          %40 = arith.muli %arg19, %c2_i32_56 : i32
          %41 = arith.addi %40, %arg20 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = arith.index_cast %arg20 : i32 to index
          %subview_57 = memref.subview %15[%43, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_57 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_22 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_22, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_24 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_58 = memref.subview %16[%43, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_59 = memref.collapse_shape %subview_58 [[0, 1, 2], [3]] : memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_24 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_59 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %39 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %35 = arith.muli %14, %c8_i32 : i32
        %36 = arith.index_cast %35 : i32 to index
        %c0_i32_47 = arith.constant 0 : i32
        %c2_i32_48 = arith.constant 2 : i32
        %c1_i32_49 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_47 to %c2_i32_48 step %c1_i32_49  : i32 {
          %37 = arith.extsi %arg20 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c2_i64 = arith.constant 2 : i64
          %38 = arith.addi %37, %c2_i64 : i64
          %39 = arith.remsi %38, %c16_i64 : i64
          %c2_i64_56 = arith.constant 2 : i64
          %40 = arith.addi %37, %c2_i64_56 : i64
          %41 = arith.remsi %40, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %39
          %c2_i32_57 = arith.constant 2 : i32
          %42 = arith.muli %arg19, %c2_i32_57 : i32
          %43 = arith.addi %42, %arg20 : i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = arith.index_cast %arg20 : i32 to index
          %subview_58 = memref.subview %16[%45, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_58 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_30 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_59 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_60 = memref.subview %alloc_10[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_59, %subview_60 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_30, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_10, %alloc_9 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_34 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_34 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_61 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_62 = memref.subview %alloc_35[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_61, %subview_62 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_35 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_36 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_30, %alloc_36 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_37 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_37 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_63 = memref.subview %alloc_29[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_64 = memref.subview %alloc_38[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_63, %subview_64 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_38 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_39 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_30, %alloc_39 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_11, %alloc_31 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_40 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_40, %alloc_32 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_65 = memref.subview %17[%45, 0, %36, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_66 = memref.collapse_shape %subview_65 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_25, %collapse_shape_66 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %41 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_50 = arith.constant 0 : i32
        %c2_i32_51 = arith.constant 2 : i32
        %c1_i32_52 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_50 to %c2_i32_51 step %c1_i32_52  : i32 {
          %37 = arith.extsi %arg20 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c2_i64 = arith.constant 2 : i64
          %38 = arith.addi %37, %c2_i64 : i64
          %39 = arith.remsi %38, %c16_i64 : i64
          %c4_i64 = arith.constant 4 : i64
          %40 = arith.addi %37, %c4_i64 : i64
          %41 = arith.remsi %40, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %39
          %c2_i32_56 = arith.constant 2 : i32
          %42 = arith.muli %arg19, %c2_i32_56 : i32
          %43 = arith.addi %42, %arg20 : i32
          %44 = arith.index_cast %43 : i32 to index
          %45 = arith.index_cast %arg20 : i32 to index
          %subview_57 = memref.subview %17[%45, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_57 [[0, 1, 2], [3]] : memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_23 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_23, %alloc_22, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_26 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_58 = memref.subview %18[%45, 0, 0, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_59 = memref.collapse_shape %subview_58 [[0, 1, 2], [3]] : memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_26 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_59 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %41 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_53 = arith.constant 0 : i32
        %c2_i32_54 = arith.constant 2 : i32
        %c1_i32_55 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_53 to %c2_i32_54 step %c1_i32_55  : i32 {
          %37 = arith.extsi %arg20 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c4_i64 = arith.constant 4 : i64
          %38 = arith.addi %37, %c4_i64 : i64
          %39 = arith.remsi %38, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %39
          %c2_i32_56 = arith.constant 2 : i32
          %40 = arith.muli %arg19, %c2_i32_56 : i32
          %41 = arith.addi %40, %arg20 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = arith.index_cast %arg20 : i32 to index
          %subview_57 = memref.subview %18[%43, 0, %36, 0] [1, 1, 8, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_57 [[0, 1, 2], [3]] : memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_33 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_12, %alloc_31 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_12, %alloc_33 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %37 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32_17 to %c2_i32_19 step %c1_i32_18  : i32 {
        %35 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %35
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %35 = arith.muli %arg18, %c2_i32 : i32
        %36 = arith.addi %35, %14 : i32
        %37 = arith.cmpi slt, %36, %c8_i32 : i32
        scf.if %37 {
          %38 = arith.muli %arg18, %c16_i32 : i32
          %39 = arith.muli %14, %c8_i32 : i32
          %40 = arith.addi %38, %39 : i32
          %41 = arith.addi %40, %arg19 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = memref.load %reinterpret_cast_7[%42] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %44 = arith.index_cast %arg19 : i32 to index
          memref.store %43, %alloc_10[%44, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %38 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_10[%38, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_10, %alloc_9 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_11, %alloc_15 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_12, %alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_12 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %28 = arith.muli %14, %c8_i32 : i32
      %29 = arith.subi %c64_i32, %28 : i32
      %30 = arith.subi %29, %23 : i32
      %31 = arith.minsi %30, %c8_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview_20 = memref.subview %alloc_13[0, 0] [%32, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %33 = arith.addi %23, %28 : i32
      %34 = arith.index_cast %33 : i32 to index
      %subview_21 = memref.subview %reinterpret_cast_4[%20, %22, %34, 0] [1, 1, %32, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_20, %subview_21 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After LowerMemRefExt (lower-memref-ext) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0] -> (d0 * 274432 + s0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @sparseAttnMix_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c274432 = arith.constant 274432 : index
    return %c274432 : index
  }
  func.func @sparseAttnMix(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %c143360 = arith.constant 143360 : index
    %c139264 = arith.constant 139264 : index
    %c131072 = arith.constant 131072 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = hivm.hir.get_block_idx -> i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = affine.apply #map(%16)[%c0]
    %view = memref.view %arg2[%17][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = affine.apply #map(%19)[%c131072]
    %view_8 = memref.view %arg2[%20][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = affine.apply #map(%22)[%c139264]
    %view_9 = memref.view %arg2[%23][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = affine.apply #map(%25)[%c143360]
    %view_10 = memref.view %arg2[%26][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %alloc_11 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_15 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_16 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_17 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_14 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %27 = arith.divsi %12, %arg9 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.remsi %12, %arg9 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.muli %arg18, %c16_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview = memref.subview %reinterpret_cast[%28, %30, %32, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %33 = arith.addi %arg11, %c31_i32 : i32
      %34 = arith.divsi %33, %c32_i32 : i32
      %35 = arith.divsi %34, %c2_i32 : i32
      scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %43 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %43 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scf.for %arg19 = %c0_i32 to %35 step %c1_i32  : i32 {
        %alloc_20 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_21 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_22 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_23 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_25 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_27 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_31 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_32 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_33 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_34 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_35 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_36 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_37 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_38 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %45
          %46 = arith.muli %arg19, %c2_i32 : i32
          %47 = arith.addi %46, %arg20 : i32
          %48 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_25 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_27 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %49 = arith.divsi %12, %arg9 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = arith.remsi %12, %arg9 : i32
          %52 = arith.index_cast %51 : i32 to index
          %53 = arith.muli %47, %c32_i32 : i32
          %54 = arith.subi %arg11, %53 : i32
          %55 = arith.minsi %54, %c32_i32 : i32
          %56 = arith.index_cast %55 : i32 to index
          %57 = arith.muli %47, %c32_i32 : i32
          %58 = arith.index_cast %57 : i32 to index
          %subview_39 = memref.subview %reinterpret_cast_5[%50, %52, %58] [1, 1, %56] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_40 = memref.subview %alloc_26[0] [%56] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>> to memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_39 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_40 : memref<?xi32, strided<[1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          scf.for %arg21 = %c0_i32 to %55 step %c1_i32  : i32 {
            %59 = arith.index_cast %arg21 : i32 to index
            %60 = memref.load %alloc_26[%59] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %61 = arith.cmpi ne, %60, %c-1_i32 : i32
            scf.if %61 {
              %62 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_27[%c0, %62] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %63 = arith.divsi %12, %arg9 : i32
              %64 = arith.index_cast %63 : i32 to index
              %65 = arith.index_cast %60 : i32 to index
              %subview_42 = memref.subview %reinterpret_cast_6[%64, %65, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_43 = memref.subview %alloc_25[%62, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_42, %subview_43 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_41 = memref.subview %view[%48, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_41 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_25, %collapse_shape : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %45 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %45
          %48 = arith.index_cast %arg20 : i32 to index
          %subview_39 = memref.subview %view[%48, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_39 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_20 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_20, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_22 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_40 = memref.subview %view_8[%48, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_41 = memref.collapse_shape %subview_40 [[0, 1, 2], [3]] : memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_22 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_41 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %47 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          %48 = arith.addi %45, %c2_i64 : i64
          %49 = arith.remsi %48, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %50 = arith.index_cast %arg20 : i32 to index
          %subview_39 = memref.subview %view_8[%50, 0, %44, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_39 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_28 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_40 = memref.subview %alloc_11[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_41 = memref.subview %alloc_12[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_40, %subview_41 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_28, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_28 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_28 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_12, %alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_32 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_42 = memref.subview %alloc_11[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_43 = memref.subview %alloc_33[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_42, %subview_43 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_33 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_28, %alloc_34 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_35 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_35 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_44 = memref.subview %alloc_27[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_45 = memref.subview %alloc_36[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_44, %subview_45 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_36 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_37 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_28, %alloc_37 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_28 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_13, %alloc_29 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_38 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_38, %alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_28 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_23 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_46 = memref.subview %view_9[%50, 0, %44, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_47 = memref.collapse_shape %subview_46 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_23, %collapse_shape_47 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %49 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          %48 = arith.addi %45, %c4_i64 : i64
          %49 = arith.remsi %48, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %50 = arith.index_cast %arg20 : i32 to index
          %subview_39 = memref.subview %view_9[%50, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_39 [[0, 1, 2], [3]] : memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_21 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_21, %alloc_20, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_24 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_40 = memref.subview %view_10[%50, 0, 0, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_41 = memref.collapse_shape %subview_40 [[0, 1, 2], [3]] : memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_24 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_41 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %49 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c4_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %48 = arith.index_cast %arg20 : i32 to index
          %subview_39 = memref.subview %view_10[%48, 0, %44, 0] [1, 1, 8, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_39 [[0, 1, 2], [3]] : memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_31 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_14, %alloc_29 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_14, %alloc_31 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %45 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %43 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %43
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %43 = arith.muli %arg18, %c2_i32 : i32
        %44 = arith.addi %43, %14 : i32
        %45 = arith.cmpi slt, %44, %c8_i32 : i32
        scf.if %45 {
          %46 = arith.muli %arg18, %c16_i32 : i32
          %47 = arith.muli %14, %c8_i32 : i32
          %48 = arith.addi %46, %47 : i32
          %49 = arith.addi %48, %arg19 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %reinterpret_cast_7[%50] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %52 = arith.index_cast %arg19 : i32 to index
          memref.store %51, %alloc_12[%52, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %46 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_12[%46, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_12, %alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_16 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_13, %alloc_17 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_14, %alloc_13 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_14 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %36 = arith.muli %14, %c8_i32 : i32
      %37 = arith.subi %c64_i32, %36 : i32
      %38 = arith.subi %37, %31 : i32
      %39 = arith.minsi %38, %c8_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %subview_18 = memref.subview %alloc_15[0, 0] [%40, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %41 = arith.addi %31, %36 : i32
      %42 = arith.index_cast %41 : i32 to index
      %subview_19 = memref.subview %reinterpret_cast_4[%28, %30, %42, 0] [1, 1, %40, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_18, %subview_19 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRSplitMixKernel (tilelangir-split-mix-kernel) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0] -> (d0 * 274432 + s0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @sparseAttnMix_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c274432 = arith.constant 274432 : index
    return %c274432 : index
  }
  func.func @sparseAttnMix_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %c143360 = arith.constant 143360 : index
    %c139264 = arith.constant 139264 : index
    %c131072 = arith.constant 131072 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = hivm.hir.get_block_idx -> i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = affine.apply #map(%16)[%c0]
    %view = memref.view %arg2[%17][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = affine.apply #map(%19)[%c131072]
    %view_5 = memref.view %arg2[%20][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = affine.apply #map(%22)[%c139264]
    %view_6 = memref.view %arg2[%23][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = affine.apply #map(%25)[%c143360]
    %view_7 = memref.view %arg2[%26][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %27 = arith.divsi %12, %arg9 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.remsi %12, %arg9 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.muli %arg18, %c16_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview = memref.subview %reinterpret_cast[%28, %30, %32, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %33 = arith.addi %arg11, %c31_i32 : i32
      %34 = arith.divsi %33, %c32_i32 : i32
      %35 = arith.divsi %34, %c2_i32 : i32
      scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %43 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %43 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg19 = %c0_i32 to %35 step %c1_i32  : i32 {
        %alloc_8 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_9 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_10 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_11 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %45
          %48 = arith.index_cast %arg20 : i32 to index
          %subview_12 = memref.subview %view[%48, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_12 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_8 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_8, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_10 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_13 = memref.subview %view_5[%48, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_14 = memref.collapse_shape %subview_13 [[0, 1, 2], [3]] : memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_10 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_14 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %47 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          %48 = arith.addi %45, %c4_i64 : i64
          %49 = arith.remsi %48, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %50 = arith.index_cast %arg20 : i32 to index
          %subview_12 = memref.subview %view_6[%50, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_12 [[0, 1, 2], [3]] : memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_9 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_9, %alloc_8, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_11 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_13 = memref.subview %view_7[%50, 0, 0, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_14 = memref.collapse_shape %subview_13 [[0, 1, 2], [3]] : memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_11 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_14 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %49 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %43 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %43
      }
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %43 = arith.muli %arg18, %c2_i32 : i32
        %44 = arith.addi %43, %14 : i32
        %45 = arith.cmpi slt, %44, %c8_i32 : i32
        scf.if %45 {
          %46 = arith.muli %arg18, %c16_i32 : i32
          %47 = arith.muli %14, %c8_i32 : i32
          %48 = arith.addi %46, %47 : i32
          %49 = arith.addi %48, %arg19 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %reinterpret_cast_4[%50] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %52 = arith.index_cast %arg19 : i32 to index
        } else {
          %46 = arith.index_cast %arg19 : i32 to index
        }
      }
      %36 = arith.muli %14, %c8_i32 : i32
      %37 = arith.subi %c64_i32, %36 : i32
      %38 = arith.subi %37, %31 : i32
      %39 = arith.minsi %38, %c8_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = arith.addi %31, %36 : i32
      %42 = arith.index_cast %41 : i32 to index
    }
    return
  }
  func.func @sparseAttnMix_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %c143360 = arith.constant 143360 : index
    %c139264 = arith.constant 139264 : index
    %c131072 = arith.constant 131072 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = hivm.hir.get_block_idx -> i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = affine.apply #map(%16)[%c0]
    %view = memref.view %arg2[%17][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = affine.apply #map(%19)[%c131072]
    %view_6 = memref.view %arg2[%20][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = affine.apply #map(%22)[%c139264]
    %view_7 = memref.view %arg2[%23][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = affine.apply #map(%25)[%c143360]
    %view_8 = memref.view %arg2[%26][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %27 = arith.divsi %12, %arg9 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.remsi %12, %arg9 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.muli %arg18, %c16_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = arith.addi %arg11, %c31_i32 : i32
      %34 = arith.divsi %33, %c32_i32 : i32
      %35 = arith.divsi %34, %c2_i32 : i32
      scf.for %arg19 = %c0_i32 to %35 step %c1_i32  : i32 {
        %alloc_16 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_17 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_18 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_20 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_21 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_22 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_23 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_25 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %45
          %46 = arith.muli %arg19, %c2_i32 : i32
          %47 = arith.addi %46, %arg20 : i32
          %48 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_17 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_19 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %49 = arith.divsi %12, %arg9 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = arith.remsi %12, %arg9 : i32
          %52 = arith.index_cast %51 : i32 to index
          %53 = arith.muli %47, %c32_i32 : i32
          %54 = arith.subi %arg11, %53 : i32
          %55 = arith.minsi %54, %c32_i32 : i32
          %56 = arith.index_cast %55 : i32 to index
          %57 = arith.muli %47, %c32_i32 : i32
          %58 = arith.index_cast %57 : i32 to index
          scf.for %arg21 = %c0_i32 to %55 step %c1_i32  : i32 {
            %59 = arith.index_cast %arg21 : i32 to index
            %60 = memref.load %alloc_18[%59] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %61 = arith.cmpi ne, %60, %c-1_i32 : i32
            scf.if %61 {
              %62 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_19[%c0, %62] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %63 = arith.divsi %12, %arg9 : i32
              %64 = arith.index_cast %63 : i32 to index
              %65 = arith.index_cast %60 : i32 to index
              %subview_32 = memref.subview %reinterpret_cast_4[%64, %65, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_33 = memref.subview %alloc_17[%62, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_32, %subview_33 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_31 = memref.subview %view[%48, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_17, %collapse_shape : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %45 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          %48 = arith.addi %45, %c2_i64 : i64
          %49 = arith.remsi %48, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %50 = arith.index_cast %arg20 : i32 to index
          %subview_31 = memref.subview %view_6[%50, 0, %44, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_20 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_32 = memref.subview %alloc[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_33 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_32, %subview_33 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_20, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_9, %alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_24 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_24 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_34 = memref.subview %alloc[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_35 = memref.subview %alloc_25[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_34, %subview_35 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_25 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_20, %alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_36 = memref.subview %alloc_19[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_37 = memref.subview %alloc_28[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_36, %subview_37 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_28 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_20, %alloc_29 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_10, %alloc_21 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_30, %alloc_22 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_38 = memref.subview %view_7[%50, 0, %44, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_39 = memref.collapse_shape %subview_38 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_16, %collapse_shape_39 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %49 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c4_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %48 = arith.index_cast %arg20 : i32 to index
          %subview_31 = memref.subview %view_8[%48, 0, %44, 0] [1, 1, 8, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_23 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_11, %alloc_21 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_11, %alloc_23 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %45 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %43 = arith.muli %arg18, %c2_i32 : i32
        %44 = arith.addi %43, %14 : i32
        %45 = arith.cmpi slt, %44, %c8_i32 : i32
        scf.if %45 {
          %46 = arith.muli %arg18, %c16_i32 : i32
          %47 = arith.muli %14, %c8_i32 : i32
          %48 = arith.addi %46, %47 : i32
          %49 = arith.addi %48, %arg19 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %reinterpret_cast_5[%50] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %52 = arith.index_cast %arg19 : i32 to index
          memref.store %51, %alloc_9[%52, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %46 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%46, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %36 = arith.muli %14, %c8_i32 : i32
      %37 = arith.subi %c64_i32, %36 : i32
      %38 = arith.subi %37, %31 : i32
      %39 = arith.minsi %38, %c8_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %subview = memref.subview %alloc_12[0, 0] [%40, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %41 = arith.addi %31, %36 : i32
      %42 = arith.index_cast %41 : i32 to index
      %subview_15 = memref.subview %reinterpret_cast[%28, %30, %42, 0] [1, 1, %40, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %subview_15 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRWrapHostFunction (tilelangir-wrap-host-function) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0] -> (d0 * 274432 + s0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @sparseAttnMix_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c274432 = arith.constant 274432 : index
    return %c274432 : index
  }
  func.func @sparseAttnMix_get_kernel_num_args() -> i32 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c9_i32 = arith.constant 9 : i32
    return %c9_i32 : i32
  }
  func.func @sparseAttnMix_get_kernel_arg_type(%arg0: i32) -> i32 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c-1_i32 = arith.constant -1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = arith.cmpi eq, %arg0, %c8_i32 : i32
    %1 = arith.select %0, %c5_i32, %c-1_i32 : i32
    %c7_i32 = arith.constant 7 : i32
    %c5_i32_0 = arith.constant 5 : i32
    %2 = arith.cmpi eq, %arg0, %c7_i32 : i32
    %3 = arith.select %2, %c5_i32_0, %1 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32_1 = arith.constant 5 : i32
    %4 = arith.cmpi eq, %arg0, %c6_i32 : i32
    %5 = arith.select %4, %c5_i32_1, %3 : i32
    %c5_i32_2 = arith.constant 5 : i32
    %c5_i32_3 = arith.constant 5 : i32
    %6 = arith.cmpi eq, %arg0, %c5_i32_2 : i32
    %7 = arith.select %6, %c5_i32_3, %5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c133_i32 = arith.constant 133 : i32
    %8 = arith.cmpi eq, %arg0, %c4_i32 : i32
    %9 = arith.select %8, %c133_i32, %7 : i32
    %c3_i32 = arith.constant 3 : i32
    %c128_i32 = arith.constant 128 : i32
    %10 = arith.cmpi eq, %arg0, %c3_i32 : i32
    %11 = arith.select %10, %c128_i32, %9 : i32
    %c2_i32 = arith.constant 2 : i32
    %c130_i32 = arith.constant 130 : i32
    %12 = arith.cmpi eq, %arg0, %c2_i32 : i32
    %13 = arith.select %12, %c130_i32, %11 : i32
    %c1_i32 = arith.constant 1 : i32
    %c130_i32_4 = arith.constant 130 : i32
    %14 = arith.cmpi eq, %arg0, %c1_i32 : i32
    %15 = arith.select %14, %c130_i32_4, %13 : i32
    %c0_i32 = arith.constant 0 : i32
    %c130_i32_5 = arith.constant 130 : i32
    %16 = arith.cmpi eq, %arg0, %c0_i32 : i32
    %17 = arith.select %16, %c130_i32_5, %15 : i32
    return %17 : i32
  }
  func.func @sparseAttnMix_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %c143360 = arith.constant 143360 : index
    %c139264 = arith.constant 139264 : index
    %c131072 = arith.constant 131072 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = hivm.hir.get_block_idx -> i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = affine.apply #map(%16)[%c0]
    %view = memref.view %arg2[%17][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = affine.apply #map(%19)[%c131072]
    %view_5 = memref.view %arg2[%20][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = affine.apply #map(%22)[%c139264]
    %view_6 = memref.view %arg2[%23][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = affine.apply #map(%25)[%c143360]
    %view_7 = memref.view %arg2[%26][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %27 = arith.divsi %12, %arg9 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.remsi %12, %arg9 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.muli %arg18, %c16_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview = memref.subview %reinterpret_cast[%28, %30, %32, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %33 = arith.addi %arg11, %c31_i32 : i32
      %34 = arith.divsi %33, %c32_i32 : i32
      %35 = arith.divsi %34, %c2_i32 : i32
      scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %43 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %43 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg19 = %c0_i32 to %35 step %c1_i32  : i32 {
        %alloc_8 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_9 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_10 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_11 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %45
          %48 = arith.index_cast %arg20 : i32 to index
          %subview_12 = memref.subview %view[%48, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_12 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_8 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_8, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_10 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_13 = memref.subview %view_5[%48, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_14 = memref.collapse_shape %subview_13 [[0, 1, 2], [3]] : memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_10 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_14 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %47 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          %48 = arith.addi %45, %c4_i64 : i64
          %49 = arith.remsi %48, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %50 = arith.index_cast %arg20 : i32 to index
          %subview_12 = memref.subview %view_6[%50, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_12 [[0, 1, 2], [3]] : memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_9 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_9, %alloc_8, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_11 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_13 = memref.subview %view_7[%50, 0, 0, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_14 = memref.collapse_shape %subview_13 [[0, 1, 2], [3]] : memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_11 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_14 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %49 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %43 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %43
      }
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %43 = arith.muli %arg18, %c2_i32 : i32
        %44 = arith.addi %43, %14 : i32
        %45 = arith.cmpi slt, %44, %c8_i32 : i32
        scf.if %45 {
          %46 = arith.muli %arg18, %c16_i32 : i32
          %47 = arith.muli %14, %c8_i32 : i32
          %48 = arith.addi %46, %47 : i32
          %49 = arith.addi %48, %arg19 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %reinterpret_cast_4[%50] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %52 = arith.index_cast %arg19 : i32 to index
        } else {
          %46 = arith.index_cast %arg19 : i32 to index
        }
      }
      %36 = arith.muli %14, %c8_i32 : i32
      %37 = arith.subi %c64_i32, %36 : i32
      %38 = arith.subi %37, %31 : i32
      %39 = arith.minsi %38, %c8_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = arith.addi %31, %36 : i32
      %42 = arith.index_cast %41 : i32 to index
    }
    return
  }
  func.func @sparseAttnMix_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %c143360 = arith.constant 143360 : index
    %c139264 = arith.constant 139264 : index
    %c131072 = arith.constant 131072 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = hivm.hir.get_block_idx -> i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = affine.apply #map(%16)[%c0]
    %view = memref.view %arg2[%17][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = affine.apply #map(%19)[%c131072]
    %view_6 = memref.view %arg2[%20][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = affine.apply #map(%22)[%c139264]
    %view_7 = memref.view %arg2[%23][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = affine.apply #map(%25)[%c143360]
    %view_8 = memref.view %arg2[%26][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %27 = arith.divsi %12, %arg9 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.remsi %12, %arg9 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.muli %arg18, %c16_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = arith.addi %arg11, %c31_i32 : i32
      %34 = arith.divsi %33, %c32_i32 : i32
      %35 = arith.divsi %34, %c2_i32 : i32
      scf.for %arg19 = %c0_i32 to %35 step %c1_i32  : i32 {
        %alloc_16 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_17 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_18 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_20 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_21 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_22 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_23 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_25 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %45
          %46 = arith.muli %arg19, %c2_i32 : i32
          %47 = arith.addi %46, %arg20 : i32
          %48 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_17 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_19 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %49 = arith.divsi %12, %arg9 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = arith.remsi %12, %arg9 : i32
          %52 = arith.index_cast %51 : i32 to index
          %53 = arith.muli %47, %c32_i32 : i32
          %54 = arith.subi %arg11, %53 : i32
          %55 = arith.minsi %54, %c32_i32 : i32
          %56 = arith.index_cast %55 : i32 to index
          %57 = arith.muli %47, %c32_i32 : i32
          %58 = arith.index_cast %57 : i32 to index
          scf.for %arg21 = %c0_i32 to %55 step %c1_i32  : i32 {
            %59 = arith.index_cast %arg21 : i32 to index
            %60 = memref.load %alloc_18[%59] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %61 = arith.cmpi ne, %60, %c-1_i32 : i32
            scf.if %61 {
              %62 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_19[%c0, %62] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %63 = arith.divsi %12, %arg9 : i32
              %64 = arith.index_cast %63 : i32 to index
              %65 = arith.index_cast %60 : i32 to index
              %subview_32 = memref.subview %reinterpret_cast_4[%64, %65, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_33 = memref.subview %alloc_17[%62, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_32, %subview_33 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_31 = memref.subview %view[%48, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_17, %collapse_shape : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %45 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          %48 = arith.addi %45, %c2_i64 : i64
          %49 = arith.remsi %48, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %50 = arith.index_cast %arg20 : i32 to index
          %subview_31 = memref.subview %view_6[%50, 0, %44, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_20 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_32 = memref.subview %alloc[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_33 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_32, %subview_33 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_20, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_9, %alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_24 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_24 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_34 = memref.subview %alloc[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_35 = memref.subview %alloc_25[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_34, %subview_35 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_25 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_20, %alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_36 = memref.subview %alloc_19[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_37 = memref.subview %alloc_28[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_36, %subview_37 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_28 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_20, %alloc_29 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_10, %alloc_21 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_30, %alloc_22 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_38 = memref.subview %view_7[%50, 0, %44, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_39 = memref.collapse_shape %subview_38 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_16, %collapse_shape_39 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %49 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c4_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %48 = arith.index_cast %arg20 : i32 to index
          %subview_31 = memref.subview %view_8[%48, 0, %44, 0] [1, 1, 8, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_23 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_11, %alloc_21 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_11, %alloc_23 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %45 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %43 = arith.muli %arg18, %c2_i32 : i32
        %44 = arith.addi %43, %14 : i32
        %45 = arith.cmpi slt, %44, %c8_i32 : i32
        scf.if %45 {
          %46 = arith.muli %arg18, %c16_i32 : i32
          %47 = arith.muli %14, %c8_i32 : i32
          %48 = arith.addi %46, %47 : i32
          %49 = arith.addi %48, %arg19 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %reinterpret_cast_5[%50] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %52 = arith.index_cast %arg19 : i32 to index
          memref.store %51, %alloc_9[%52, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %46 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%46, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %36 = arith.muli %14, %c8_i32 : i32
      %37 = arith.subi %c64_i32, %36 : i32
      %38 = arith.subi %37, %31 : i32
      %39 = arith.minsi %38, %c8_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %subview = memref.subview %alloc_12[0, 0] [%40, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %41 = arith.addi %31, %36 : i32
      %42 = arith.index_cast %41 : i32 to index
      %subview_15 = memref.subview %reinterpret_cast[%28, %30, %42, 0] [1, 1, %40, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %subview_15 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}



====== final npuir ======
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @sparseAttnMix_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c274432 = arith.constant 274432 : index
    return %c274432 : index
  }
  func.func @sparseAttnMix_get_kernel_num_args() -> i32 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c9_i32 = arith.constant 9 : i32
    return %c9_i32 : i32
  }
  func.func @sparseAttnMix_get_kernel_arg_type(%arg0: i32) -> i32 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c-1_i32 = arith.constant -1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = arith.cmpi eq, %arg0, %c8_i32 : i32
    %1 = arith.select %0, %c5_i32, %c-1_i32 : i32
    %c7_i32 = arith.constant 7 : i32
    %c5_i32_0 = arith.constant 5 : i32
    %2 = arith.cmpi eq, %arg0, %c7_i32 : i32
    %3 = arith.select %2, %c5_i32_0, %1 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32_1 = arith.constant 5 : i32
    %4 = arith.cmpi eq, %arg0, %c6_i32 : i32
    %5 = arith.select %4, %c5_i32_1, %3 : i32
    %c5_i32_2 = arith.constant 5 : i32
    %c5_i32_3 = arith.constant 5 : i32
    %6 = arith.cmpi eq, %arg0, %c5_i32_2 : i32
    %7 = arith.select %6, %c5_i32_3, %5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c133_i32 = arith.constant 133 : i32
    %8 = arith.cmpi eq, %arg0, %c4_i32 : i32
    %9 = arith.select %8, %c133_i32, %7 : i32
    %c3_i32 = arith.constant 3 : i32
    %c128_i32 = arith.constant 128 : i32
    %10 = arith.cmpi eq, %arg0, %c3_i32 : i32
    %11 = arith.select %10, %c128_i32, %9 : i32
    %c2_i32 = arith.constant 2 : i32
    %c130_i32 = arith.constant 130 : i32
    %12 = arith.cmpi eq, %arg0, %c2_i32 : i32
    %13 = arith.select %12, %c130_i32, %11 : i32
    %c1_i32 = arith.constant 1 : i32
    %c130_i32_4 = arith.constant 130 : i32
    %14 = arith.cmpi eq, %arg0, %c1_i32 : i32
    %15 = arith.select %14, %c130_i32_4, %13 : i32
    %c0_i32 = arith.constant 0 : i32
    %c130_i32_5 = arith.constant 130 : i32
    %16 = arith.cmpi eq, %arg0, %c0_i32 : i32
    %17 = arith.select %16, %c130_i32_5, %15 : i32
    return %17 : i32
  }
  func.func @sparseAttnMix_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %c143360 = arith.constant 143360 : index
    %c139264 = arith.constant 139264 : index
    %c131072 = arith.constant 131072 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = hivm.hir.get_block_idx -> i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%16)[%c0]
    %view = memref.view %arg2[%17][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%19)[%c131072]
    %view_5 = memref.view %arg2[%20][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%22)[%c139264]
    %view_6 = memref.view %arg2[%23][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%25)[%c143360]
    %view_7 = memref.view %arg2[%26][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %27 = arith.divsi %12, %arg9 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.remsi %12, %arg9 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.muli %arg18, %c16_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview = memref.subview %reinterpret_cast[%28, %30, %32, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %33 = arith.addi %arg11, %c31_i32 : i32
      %34 = arith.divsi %33, %c32_i32 : i32
      %35 = arith.divsi %34, %c2_i32 : i32
      scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %43 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %43 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg19 = %c0_i32 to %35 step %c1_i32  : i32 {
        %alloc_8 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_9 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_10 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_11 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %45
          %48 = arith.index_cast %arg20 : i32 to index
          %subview_12 = memref.subview %view[%48, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_12 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_8 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_8, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_10 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_13 = memref.subview %view_5[%48, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_14 = memref.collapse_shape %subview_13 [[0, 1, 2], [3]] : memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_10 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_14 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %47 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          %48 = arith.addi %45, %c4_i64 : i64
          %49 = arith.remsi %48, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %50 = arith.index_cast %arg20 : i32 to index
          %subview_12 = memref.subview %view_6[%50, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_12 [[0, 1, 2], [3]] : memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_9 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_9, %alloc_8, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_11 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_13 = memref.subview %view_7[%50, 0, 0, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_14 = memref.collapse_shape %subview_13 [[0, 1, 2], [3]] : memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_11 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_14 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %49 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %43 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %43
      }
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %43 = arith.muli %arg18, %c2_i32 : i32
        %44 = arith.addi %43, %14 : i32
        %45 = arith.cmpi slt, %44, %c8_i32 : i32
        scf.if %45 {
          %46 = arith.muli %arg18, %c16_i32 : i32
          %47 = arith.muli %14, %c8_i32 : i32
          %48 = arith.addi %46, %47 : i32
          %49 = arith.addi %48, %arg19 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %reinterpret_cast_4[%50] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %52 = arith.index_cast %arg19 : i32 to index
        } else {
          %46 = arith.index_cast %arg19 : i32 to index
        }
      }
      %36 = arith.muli %14, %c8_i32 : i32
      %37 = arith.subi %c64_i32, %36 : i32
      %38 = arith.subi %37, %31 : i32
      %39 = arith.minsi %38, %c8_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = arith.addi %31, %36 : i32
      %42 = arith.index_cast %41 : i32 to index
    }
    return
  }
  func.func @sparseAttnMix_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %c143360 = arith.constant 143360 : index
    %c139264 = arith.constant 139264 : index
    %c131072 = arith.constant 131072 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = hivm.hir.get_block_idx -> i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%16)[%c0]
    %view = memref.view %arg2[%17][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%19)[%c131072]
    %view_6 = memref.view %arg2[%20][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%22)[%c139264]
    %view_7 = memref.view %arg2[%23][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%25)[%c143360]
    %view_8 = memref.view %arg2[%26][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %27 = arith.divsi %12, %arg9 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.remsi %12, %arg9 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.muli %arg18, %c16_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = arith.addi %arg11, %c31_i32 : i32
      %34 = arith.divsi %33, %c32_i32 : i32
      %35 = arith.divsi %34, %c2_i32 : i32
      scf.for %arg19 = %c0_i32 to %35 step %c1_i32  : i32 {
        %alloc_16 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_17 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_18 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_20 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_21 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_22 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_23 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_25 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %45
          %46 = arith.muli %arg19, %c2_i32 : i32
          %47 = arith.addi %46, %arg20 : i32
          %48 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_17 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_19 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %49 = arith.divsi %12, %arg9 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = arith.remsi %12, %arg9 : i32
          %52 = arith.index_cast %51 : i32 to index
          %53 = arith.muli %47, %c32_i32 : i32
          %54 = arith.subi %arg11, %53 : i32
          %55 = arith.minsi %54, %c32_i32 : i32
          %56 = arith.index_cast %55 : i32 to index
          %57 = arith.muli %47, %c32_i32 : i32
          %58 = arith.index_cast %57 : i32 to index
          scf.for %arg21 = %c0_i32 to %55 step %c1_i32  : i32 {
            %59 = arith.index_cast %arg21 : i32 to index
            %60 = memref.load %alloc_18[%59] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %61 = arith.cmpi ne, %60, %c-1_i32 : i32
            scf.if %61 {
              %62 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_19[%c0, %62] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %63 = arith.divsi %12, %arg9 : i32
              %64 = arith.index_cast %63 : i32 to index
              %65 = arith.index_cast %60 : i32 to index
              %subview_32 = memref.subview %reinterpret_cast_4[%64, %65, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_33 = memref.subview %alloc_17[%62, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_32, %subview_33 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_31 = memref.subview %view[%48, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_17, %collapse_shape : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %45 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          %48 = arith.addi %45, %c2_i64 : i64
          %49 = arith.remsi %48, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %50 = arith.index_cast %arg20 : i32 to index
          %subview_31 = memref.subview %view_6[%50, 0, %44, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_20 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_32 = memref.subview %alloc[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_33 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_32, %subview_33 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_20, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_9, %alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_24 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_24 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_34 = memref.subview %alloc[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_35 = memref.subview %alloc_25[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_34, %subview_35 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_25 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_20, %alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_36 = memref.subview %alloc_19[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_37 = memref.subview %alloc_28[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_36, %subview_37 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_28 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_20, %alloc_29 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_10, %alloc_21 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_30, %alloc_22 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_38 = memref.subview %view_7[%50, 0, %44, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_39 = memref.collapse_shape %subview_38 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_16, %collapse_shape_39 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %49 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c4_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %48 = arith.index_cast %arg20 : i32 to index
          %subview_31 = memref.subview %view_8[%48, 0, %44, 0] [1, 1, 8, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_23 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_11, %alloc_21 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_11, %alloc_23 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %45 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %43 = arith.muli %arg18, %c2_i32 : i32
        %44 = arith.addi %43, %14 : i32
        %45 = arith.cmpi slt, %44, %c8_i32 : i32
        scf.if %45 {
          %46 = arith.muli %arg18, %c16_i32 : i32
          %47 = arith.muli %14, %c8_i32 : i32
          %48 = arith.addi %46, %47 : i32
          %49 = arith.addi %48, %arg19 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %reinterpret_cast_5[%50] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %52 = arith.index_cast %arg19 : i32 to index
          memref.store %51, %alloc_9[%52, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %46 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%46, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %36 = arith.muli %14, %c8_i32 : i32
      %37 = arith.subi %c64_i32, %36 : i32
      %38 = arith.subi %37, %31 : i32
      %39 = arith.minsi %38, %c8_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %subview = memref.subview %alloc_12[0, 0] [%40, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %41 = arith.addi %31, %36 : i32
      %42 = arith.index_cast %41 : i32 to index
      %subview_15 = memref.subview %reinterpret_cast[%28, %30, %42, 0] [1, 1, %40, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %subview_15 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}
cmd_list ['/home/CANN/CANN8.5.0/cann-8.5.0/bin/bishengir-compile', '/tmp/tmp8sg7__h2/kernel.npuir', '--enable-auto-multi-buffer=false', '--enable-triton-kernel-compile=true', '--enable-hivm-compile=true', '--disable-hivm-tensor-compile=true', '-o', '/tmp/tmp8sg7__h2/kernel']
AscendNPU IR:

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @sparseAttnMix_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c274432 = arith.constant 274432 : index
    return %c274432 : index
  }
  func.func @sparseAttnMix_get_kernel_num_args() -> i32 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c9_i32 = arith.constant 9 : i32
    return %c9_i32 : i32
  }
  func.func @sparseAttnMix_get_kernel_arg_type(%arg0: i32) -> i32 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c-1_i32 = arith.constant -1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = arith.cmpi eq, %arg0, %c8_i32 : i32
    %1 = arith.select %0, %c5_i32, %c-1_i32 : i32
    %c7_i32 = arith.constant 7 : i32
    %c5_i32_0 = arith.constant 5 : i32
    %2 = arith.cmpi eq, %arg0, %c7_i32 : i32
    %3 = arith.select %2, %c5_i32_0, %1 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32_1 = arith.constant 5 : i32
    %4 = arith.cmpi eq, %arg0, %c6_i32 : i32
    %5 = arith.select %4, %c5_i32_1, %3 : i32
    %c5_i32_2 = arith.constant 5 : i32
    %c5_i32_3 = arith.constant 5 : i32
    %6 = arith.cmpi eq, %arg0, %c5_i32_2 : i32
    %7 = arith.select %6, %c5_i32_3, %5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c133_i32 = arith.constant 133 : i32
    %8 = arith.cmpi eq, %arg0, %c4_i32 : i32
    %9 = arith.select %8, %c133_i32, %7 : i32
    %c3_i32 = arith.constant 3 : i32
    %c128_i32 = arith.constant 128 : i32
    %10 = arith.cmpi eq, %arg0, %c3_i32 : i32
    %11 = arith.select %10, %c128_i32, %9 : i32
    %c2_i32 = arith.constant 2 : i32
    %c130_i32 = arith.constant 130 : i32
    %12 = arith.cmpi eq, %arg0, %c2_i32 : i32
    %13 = arith.select %12, %c130_i32, %11 : i32
    %c1_i32 = arith.constant 1 : i32
    %c130_i32_4 = arith.constant 130 : i32
    %14 = arith.cmpi eq, %arg0, %c1_i32 : i32
    %15 = arith.select %14, %c130_i32_4, %13 : i32
    %c0_i32 = arith.constant 0 : i32
    %c130_i32_5 = arith.constant 130 : i32
    %16 = arith.cmpi eq, %arg0, %c0_i32 : i32
    %17 = arith.select %16, %c130_i32_5, %15 : i32
    return %17 : i32
  }
  func.func @sparseAttnMix_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %c143360 = arith.constant 143360 : index
    %c139264 = arith.constant 139264 : index
    %c131072 = arith.constant 131072 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = hivm.hir.get_block_idx -> i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%16)[%c0]
    %view = memref.view %arg2[%17][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%19)[%c131072]
    %view_5 = memref.view %arg2[%20][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%22)[%c139264]
    %view_6 = memref.view %arg2[%23][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%25)[%c143360]
    %view_7 = memref.view %arg2[%26][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %27 = arith.divsi %12, %arg9 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.remsi %12, %arg9 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.muli %arg18, %c16_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview = memref.subview %reinterpret_cast[%28, %30, %32, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %33 = arith.addi %arg11, %c31_i32 : i32
      %34 = arith.divsi %33, %c32_i32 : i32
      %35 = arith.divsi %34, %c2_i32 : i32
      scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %43 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %43 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg19 = %c0_i32 to %35 step %c1_i32  : i32 {
        %alloc_8 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_9 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_10 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_11 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %45
          %48 = arith.index_cast %arg20 : i32 to index
          %subview_12 = memref.subview %view[%48, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_12 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_8 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_8, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_10 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
          %subview_13 = memref.subview %view_5[%48, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_14 = memref.collapse_shape %subview_13 [[0, 1, 2], [3]] : memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_10 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_14 : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %47 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          %48 = arith.addi %45, %c4_i64 : i64
          %49 = arith.remsi %48, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %50 = arith.index_cast %arg20 : i32 to index
          %subview_12 = memref.subview %view_6[%50, 0, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_12 [[0, 1, 2], [3]] : memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_9 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_9, %alloc_8, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_11 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>)
          %subview_13 = memref.subview %view_7[%50, 0, 0, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_14 = memref.collapse_shape %subview_13 [[0, 1, 2], [3]] : memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_11 : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_14 : memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %49 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %43 = arith.extsi %arg19 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %43
      }
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %43 = arith.muli %arg18, %c2_i32 : i32
        %44 = arith.addi %43, %14 : i32
        %45 = arith.cmpi slt, %44, %c8_i32 : i32
        scf.if %45 {
          %46 = arith.muli %arg18, %c16_i32 : i32
          %47 = arith.muli %14, %c8_i32 : i32
          %48 = arith.addi %46, %47 : i32
          %49 = arith.addi %48, %arg19 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %reinterpret_cast_4[%50] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %52 = arith.index_cast %arg19 : i32 to index
        } else {
          %46 = arith.index_cast %arg19 : i32 to index
        }
      }
      %36 = arith.muli %14, %c8_i32 : i32
      %37 = arith.subi %c64_i32, %36 : i32
      %38 = arith.subi %37, %31 : i32
      %39 = arith.minsi %38, %c8_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = arith.addi %31, %36 : i32
      %42 = arith.index_cast %41 : i32 to index
    }
    return
  }
  func.func @sparseAttnMix_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %c143360 = arith.constant 143360 : index
    %c139264 = arith.constant 139264 : index
    %c131072 = arith.constant 131072 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32768 = arith.constant 32768 : index
    %c32768_i32 = arith.constant 32768 : i32
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 0.0441941731 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c32768_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 512], strides: [%3, %c32768, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c512_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 512], strides: [%10, %c512, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = hivm.hir.get_block_idx -> i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%16)[%c0]
    %view = memref.view %arg2[%17][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%19)[%c131072]
    %view_6 = memref.view %arg2[%20][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%22)[%c139264]
    %view_7 = memref.view %arg2[%23][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = affine.apply affine_map<(d0)[s0] -> (d0 * 274432 + s0)>(%25)[%c143360]
    %view_8 = memref.view %arg2[%26][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x512xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %27 = arith.divsi %12, %arg9 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.remsi %12, %arg9 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.muli %arg18, %c16_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = arith.addi %arg11, %c31_i32 : i32
      %34 = arith.divsi %33, %c32_i32 : i32
      %35 = arith.divsi %34, %c2_i32 : i32
      scf.for %arg19 = %c0_i32 to %35 step %c1_i32  : i32 {
        %alloc_16 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_17 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_18 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_20 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_21 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_22 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_23 = memref.alloc() : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_25 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %45
          %46 = arith.muli %arg19, %c2_i32 : i32
          %47 = arith.addi %46, %arg20 : i32
          %48 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_17 : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_19 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %49 = arith.divsi %12, %arg9 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = arith.remsi %12, %arg9 : i32
          %52 = arith.index_cast %51 : i32 to index
          %53 = arith.muli %47, %c32_i32 : i32
          %54 = arith.subi %arg11, %53 : i32
          %55 = arith.minsi %54, %c32_i32 : i32
          %56 = arith.index_cast %55 : i32 to index
          %57 = arith.muli %47, %c32_i32 : i32
          %58 = arith.index_cast %57 : i32 to index
          scf.for %arg21 = %c0_i32 to %55 step %c1_i32  : i32 {
            %59 = arith.index_cast %arg21 : i32 to index
            %60 = memref.load %alloc_18[%59] : memref<32xi32, strided<[1]>, #hivm.address_space<cbuf>>
            %61 = arith.cmpi ne, %60, %c-1_i32 : i32
            scf.if %61 {
              %62 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_19[%c0, %62] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %63 = arith.divsi %12, %arg9 : i32
              %64 = arith.index_cast %63 : i32 to index
              %65 = arith.index_cast %60 : i32 to index
              %subview_32 = memref.subview %reinterpret_cast_4[%64, %65, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_33 = memref.subview %alloc_17[%62, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_32, %subview_33 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_31 = memref.subview %view[%48, 0, 0, 0] [1, 1, 32, 512] [1, 1, 1, 1] : memref<2x2x32x512xbf16, #hivm.address_space<gm>> to memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_17, %collapse_shape : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %45 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        %43 = arith.muli %14, %c8_i32 : i32
        %44 = arith.index_cast %43 : i32 to index
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          %48 = arith.addi %45, %c2_i64 : i64
          %49 = arith.remsi %48, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %50 = arith.index_cast %arg20 : i32 to index
          %subview_31 = memref.subview %view_6[%50, 0, %44, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_20 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_32 = memref.subview %alloc[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_33 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_32, %subview_33 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_20, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_9, %alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_24 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_24 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_34 = memref.subview %alloc[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_35 = memref.subview %alloc_25[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_34, %subview_35 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_25 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_20, %alloc_26 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_36 = memref.subview %alloc_19[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_37 = memref.subview %alloc_28[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_36, %subview_37 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_28 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_20, %alloc_29 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_10, %alloc_21 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_30, %alloc_22 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_20 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_38 = memref.subview %view_7[%50, 0, %44, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_39 = memref.collapse_shape %subview_38 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_16, %collapse_shape_39 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %49 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c4_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %48 = arith.index_cast %arg20 : i32 to index
          %subview_31 = memref.subview %view_8[%48, 0, %44, 0] [1, 1, 8, 512] [1, 1, 1, 1] : memref<2x2x16x512xf32, #hivm.address_space<gm>> to memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_23 : memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_11, %alloc_21 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_11, %alloc_23 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %45 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 2 : i32}
      scf.for %arg19 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %43 = arith.muli %arg18, %c2_i32 : i32
        %44 = arith.addi %43, %14 : i32
        %45 = arith.cmpi slt, %44, %c8_i32 : i32
        scf.if %45 {
          %46 = arith.muli %arg18, %c16_i32 : i32
          %47 = arith.muli %14, %c8_i32 : i32
          %48 = arith.addi %46, %47 : i32
          %49 = arith.addi %48, %arg19 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %reinterpret_cast_5[%50] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %52 = arith.index_cast %arg19 : i32 to index
          memref.store %51, %alloc_9[%52, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %46 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%46, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>)
      %36 = arith.muli %14, %c8_i32 : i32
      %37 = arith.subi %c64_i32, %36 : i32
      %38 = arith.subi %37, %31 : i32
      %39 = arith.minsi %38, %c8_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %subview = memref.subview %alloc_12[0, 0] [%40, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %41 = arith.addi %31, %36 : i32
      %42 = arith.index_cast %41 : i32 to index
      %subview_15 = memref.subview %reinterpret_cast[%28, %30, %42, 0] [1, 1, %40, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %subview_15 : memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}
err cmd: /home/CANN/CANN8.5.0/cann-8.5.0/bin/bishengir-compile /tmp/tmp8sg7__h2/kernel.npuir --enable-auto-multi-buffer=false --enable-triton-kernel-compile=true --enable-hivm-compile=true --disable-hivm-tensor-compile=true -o /tmp/tmp8sg7__h2/kernel
err code: 1
Traceback (most recent call last):
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/tilelang/jit/jit_npu.py", line 1310, in _npuir_to_bin_enable_npu_compile
    ret = subprocess.run(
          ^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/shengming/lib/python3.11/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/home/CANN/CANN8.5.0/cann-8.5.0/bin/bishengir-compile', '/tmp/tmp8sg7__h2/kernel.npuir', '--enable-auto-multi-buffer=false', '--enable-triton-kernel-compile=true', '--enable-hivm-compile=true', '--disable-hivm-tensor-compile=true', '-o', '/tmp/tmp8sg7__h2/kernel']' returned non-zero exit status 1.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/examples/deepseek_v4/inference/sparse_attn_mix_open_final.py", line 405, in <module>
    run_test()
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/examples/deepseek_v4/inference/sparse_attn_mix_open_final.py", line 397, in run_test
    output = sparse_attn(**data["inputs"])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/examples/deepseek_v4/inference/sparse_attn_mix_open_final.py", line 276, in sparse_attn
    sparse_attn.kernel = sparse_attn_mix_kernel(
                         ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/tilelang/jit/__init__.py", line 197, in wrapper
    kernel_result = compile(
                    ^^^^^^^^
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/tilelang/jit/__init__.py", line 74, in compile
    return compile_npuir.compile(func, out_idx)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/tilelang/jit/jit_npu.py", line 1137, in compile
    self.metadata["kernel_src"] = self._npuir_to_bin_enable_npu_compile()
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/tilelang/jit/jit_npu.py", line 1321, in _npuir_to_bin_enable_npu_compile
    raise RuntimeError(f"NPU IR opt failed: {e.stderr}") from e
RuntimeError: NPU IR opt failed: loc("/tmp/tmp8sg7__h2/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline

loc("/tmp/tmp8sg7__h2/kernel.npuir":202:3): error: ub overflow, requires 7399424 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)
loc("/tmp/tmp8sg7__h2/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline

loc("/tmp/tmp8sg7__h2/kernel.npuir":202:3): error: ub overflow, requires 7399424 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)
loc("/tmp/tmp8sg7__h2/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline

loc("/tmp/tmp8sg7__h2/kernel.npuir":202:3): error: ub overflow, requires 7399424 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)
loc("/tmp/tmp8sg7__h2/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline

loc("/tmp/tmp8sg7__h2/kernel.npuir":202:3): error: ub overflow, requires 7399424 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)
loc("/tmp/tmp8sg7__h2/kernel.npuir":1:1): error: Failed to run BiShengHIR pipeline

loc("/tmp/tmp8sg7__h2/kernel.npuir":202:3): error: ub overflow, requires 7399424 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)
[ERROR] Failed to run BiShengIR pipeline

[ERROR] 2026-05-06-16:12:49 (PID:3640707, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
