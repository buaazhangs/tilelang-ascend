2026-05-08 10:55:39  [TileLang:tilelang.env:WARNING]: Loading tilelang libs from dev root: /home/z00910011/tilelang-ascend-test/tilelang-ascend/build
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
                kv_shared = T.decl_buffer((32, 512), "bfloat16", scope="shared.flat;multi_buffer=2")
                prob_shared = T.decl_buffer((16, 32), "bfloat16", scope="shared.flat")
                scores = T.decl_buffer((16, 32), scope="local.fragment")
                scores_cast = T.decl_buffer((8, 32), "bfloat16", scope="shared.flat")
                pv_acc = T.decl_buffer((16, 512), scope="local.fragment")
                kv_ub = T.decl_buffer((32, 512), "bfloat16", scope="shared.flat")
                idxs = T.decl_buffer((32,), "int32", scope="local.fragment")
                mask_ub = T.decl_buffer((1, 32), scope="shared.flat;multi_buffer=2")
                scores_ub = T.decl_buffer((8, 32), scope="shared.flat")
                scores_scale = T.decl_buffer((8, 1), scope="shared.flat;multi_buffer=2")
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
        annotation.mark %alloc_18 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_19 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
        %alloc_20 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
        %alloc_21 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>>
        %alloc_22 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>>
        %alloc_23 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_24 = memref.alloc() : memref<32xi32, strided<[1]>>
        %alloc_25 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        annotation.mark %alloc_25 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32, strided<[32, 1]>>
        %alloc_26 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_27 {hivm.multi_buffer = 2 : i32} : memref<8x1xf32, strided<[1, 1]>>
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
        annotation.mark %alloc_17 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        annotation.mark %alloc_24 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32, strided<[32, 1]>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_26 {hivm.multi_buffer = 2 : i32} : memref<8x1xf32, strided<[1, 1]>>
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
        annotation.mark %alloc_17 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        annotation.mark %alloc_24 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32, strided<[32, 1]>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_26 {hivm.multi_buffer = 2 : i32} : memref<8x1xf32, strided<[1, 1]>>
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
        annotation.mark %alloc_17 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        annotation.mark %alloc_24 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32, strided<[32, 1]>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_26 {hivm.multi_buffer = 2 : i32} : memref<8x1xf32, strided<[1, 1]>>
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
        annotation.mark %alloc_17 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        annotation.mark %alloc_24 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32, strided<[32, 1]>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_26 {hivm.multi_buffer = 2 : i32} : memref<8x1xf32, strided<[1, 1]>>
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
        annotation.mark %alloc_17 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>>
        annotation.mark %alloc_24 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32, strided<[32, 1]>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_26 {hivm.multi_buffer = 2 : i32} : memref<8x1xf32, strided<[1, 1]>>
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
        annotation.mark %alloc_17 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        annotation.mark %alloc_24 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        annotation.mark %alloc_26 {hivm.multi_buffer = 2 : i32} : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
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
          %subview_37 = memref.subview %alloc_23[0] [%44] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          scf.for %arg20 = %c0_i32 to %43 step %c1_i32  : i32 {
            %45 = arith.index_cast %arg20 : i32 to index
            %46 = memref.load %alloc_23[%45] : memref<32xi32, strided<[1]>, #hivm.address_space<ub>>
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
        annotation.mark %alloc_17 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_18 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_20 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_21 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_22 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_23 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        annotation.mark %alloc_24 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_25 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        annotation.mark %alloc_26 {hivm.multi_buffer = 2 : i32} : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
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
          %subview_37 = memref.subview %alloc_23[0] [%44] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          scf.for %arg20 = %c0_i32 to %43 step %c1_i32  : i32 {
            %45 = arith.index_cast %arg20 : i32 to index
            %46 = memref.load %alloc_23[%45] : memref<32xi32, strided<[1]>, #hivm.address_space<ub>>
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
        annotation.mark %alloc_18 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
        %alloc_19 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_20 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_21 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_22 = memref.alloc() : memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %alloc_23 = memref.alloc() : memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<ub>>
        %alloc_25 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        annotation.mark %alloc_25 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        annotation.mark %alloc_27 {hivm.multi_buffer = 2 : i32} : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
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
          %subview_55 = memref.subview %alloc_24[0] [%49] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_54, %subview_55 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          scf.for %arg21 = %c0_i32 to %48 step %c1_i32  : i32 {
            %52 = arith.index_cast %arg21 : i32 to index
            %53 = memref.load %alloc_24[%52] : memref<32xi32, strided<[1]>, #hivm.address_space<ub>>
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


loc("input.mlir":170:23): error: expected result type to be 'memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>' or a rank-reduced version. (mismatch of result layout) 
// -----// IR Dump After TileLangIREnableLocalBuffer Failed (tilelangir-enable-local-buffer) ('builtin.module' operation) //----- //
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{hacc.arg_type = #hacc.arg_type<ffts_base_address>}, {}, {hacc.arg_type = #hacc.arg_type<workspace>}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}], function_type = (i64, memref<?xi8, #hivm.address_space<gm>>, memref<?xi8, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xi32, #hivm.address_space<gm>>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (), sym_name = "sparseAttnMix"}> ({
  ^bb0(%arg0: i64, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>>, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32):
    %0 = "arith.constant"() <{value = 32 : index}> : () -> index
    %1 = "arith.constant"() <{value = 16 : index}> : () -> index
    %2 = "arith.constant"() <{value = 0 : index}> : () -> index
    %3 = "arith.constant"() <{value = 0.000000e+00 : bf16}> : () -> bf16
    %4 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %5 = "arith.constant"() <{value = 32768 : index}> : () -> index
    %6 = "arith.constant"() <{value = 32768 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 512 : index}> : () -> index
    %8 = "arith.constant"() <{value = 1 : index}> : () -> index
    %9 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %10 = "arith.constant"() <{value = 0.0441941731 : f32}> : () -> f32
    %11 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %12 = "arith.constant"() <{value = true}> : () -> i1
    %13 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %14 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    %16 = "arith.constant"() <{value = 31 : i32}> : () -> i32
    %17 = "arith.constant"() <{value = 16 : i32}> : () -> i32
    %18 = "arith.constant"() <{value = 0xFF800000 : f32}> : () -> f32
    %19 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %20 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %21 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %22 = "arith.constant"() <{value = 512 : i32}> : () -> i32
    %23 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "hivm.hir.set_ffts_base_addr"(%arg0) : (i64) -> ()
    %24 = "arith.index_cast"(%arg8) : (i32) -> index
    %25 = "arith.index_cast"(%arg9) : (i32) -> index
    %26 = "arith.muli"(%arg9, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %27 = "arith.index_cast"(%26) : (i32) -> index
    %28 = "memref.reinterpret_cast"(%arg3, %24, %25, %27, %5, %7, %8) <{operandSegmentSizes = array<i32: 1, 0, 2, 4>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808, 64, 512>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xbf16, #hivm.address_space<gm>>, index, index, index, index, index, index) -> memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %29 = "memref.reinterpret_cast"(%arg5, %24, %25, %27, %5, %7, %8) <{operandSegmentSizes = array<i32: 1, 0, 2, 4>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808, 64, 512>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xbf16, #hivm.address_space<gm>>, index, index, index, index, index, index) -> memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %30 = "arith.index_cast"(%arg11) : (i32) -> index
    %31 = "arith.index_cast"(%arg11) : (i32) -> index
    %32 = "arith.muli"(%arg9, %arg11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %33 = "arith.index_cast"(%32) : (i32) -> index
    %34 = "memref.reinterpret_cast"(%arg7, %24, %25, %30, %33, %31, %8) <{operandSegmentSizes = array<i32: 1, 0, 3, 3>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xi32, #hivm.address_space<gm>>, index, index, index, index, index, index) -> memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %35 = "arith.index_cast"(%arg10) : (i32) -> index
    %36 = "arith.muli"(%arg10, %22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %37 = "arith.index_cast"(%36) : (i32) -> index
    %38 = "memref.reinterpret_cast"(%arg4, %24, %35, %37, %7, %8) <{operandSegmentSizes = array<i32: 1, 0, 2, 3>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808, 512>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (memref<?xbf16, #hivm.address_space<gm>>, index, index, index, index, index) -> memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %39 = "memref.reinterpret_cast"(%arg6, %8) <{operandSegmentSizes = array<i32: 1, 0, 0, 1>, static_offsets = array<i64: 0>, static_sizes = array<i64: 64>, static_strides = array<i64: -9223372036854775808>}> : (memref<?xf32, #hivm.address_space<gm>>, index) -> memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %40 = "hivm.hir.get_block_idx"() : () -> i64
    %41 = "arith.trunci"(%40) : (i64) -> i32
    %42 = "hivm.hir.get_sub_block_idx"() : () -> i64
    %43 = "arith.trunci"(%42) : (i64) -> i32
    %44 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x32x512xbf16, #hivm.address_space<gm>>
    %45 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %46 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %47 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<2x2x16x512xf32, #hivm.address_space<gm>>
    "scf.for"(%20, %19, %23) ({
    ^bb0(%arg18: i32):
      %48 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>
      %49 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %50 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %51 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %52 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
      %53 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %54 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %55 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      "hivm.hir.vbrc"(%4, %52) <{broadcast_dims = array<i64>}> : (f32, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vbrc"(%4, %51) <{broadcast_dims = array<i64>}> : (f32, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vbrc"(%18, %49) <{broadcast_dims = array<i64>}> : (f32, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> ()
      %56 = "arith.divsi"(%41, %arg9) : (i32, i32) -> i32
      %57 = "arith.index_cast"(%56) : (i32) -> index
      %58 = "arith.remsi"(%41, %arg9) : (i32, i32) -> i32
      %59 = "arith.index_cast"(%58) : (i32) -> index
      %60 = "arith.muli"(%arg18, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %61 = "arith.index_cast"(%60) : (i32) -> index
      %62 = "memref.subview"(%28, %57, %59, %61) <{operandSegmentSizes = array<i32: 1, 3, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 16, 512>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>, index, index, index) -> memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%62, %48) : (memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>) -> ()
      %63 = "arith.addi"(%arg11, %16) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %64 = "arith.divsi"(%63, %15) : (i32, i32) -> i32
      %65 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %66 = "arith.divsi"(%64, %65) : (i32, i32) -> i32
      "scf.for"(%20, %66, %23) ({
      ^bb0(%arg20: i32):
        %87 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x32x512xbf16, strided<[16384, 512, 1]>, #hivm.address_space<cbuf>>
        %88 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %89 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %90 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %91 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>
        %92 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
        %93 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32xi32, strided<[1]>, #hivm.address_space<ub>>
        %94 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>>
        %95 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %96 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x8x1xf32, strided<[8, 1, 1]>, #hivm.address_space<ub>>
        %97 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %98 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>
        %99 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %100 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %101 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %102 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %103 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %104 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %105 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %106 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %107 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %108 = "arith.constant"() <{value = 1 : i32}> : () -> i32
        "scf.for"(%106, %107, %108) ({
        ^bb0(%arg25: i32):
          %184 = "arith.constant"() <{value = 2 : i32}> : () -> i32
          %185 = "arith.muli"(%arg20, %184) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %186 = "arith.addi"(%185, %arg25) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %187 = "arith.index_cast"(%186) : (i32) -> index
          %188 = "arith.index_cast"(%arg25) : (i32) -> index
          "hivm.hir.vbrc"(%3, %92) <{broadcast_dims = array<i64>}> : (bf16, memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>) -> ()
          %189 = "arith.index_cast"(%arg25) : (i32) -> index
          %190 = "memref.subview"(%94, %189) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 1, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>>, index) -> memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>>
          %191 = "memref.collapse_shape"(%190) <{reassociation = [[0, 1], [2]]}> : (memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>>) -> memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
          "hivm.hir.vbrc"(%4, %191) <{broadcast_dims = array<i64>}> : (f32, memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
          %192 = "arith.divsi"(%41, %arg9) : (i32, i32) -> i32
          %193 = "arith.index_cast"(%192) : (i32) -> index
          %194 = "arith.remsi"(%41, %arg9) : (i32, i32) -> i32
          %195 = "arith.index_cast"(%194) : (i32) -> index
          %196 = "arith.muli"(%186, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %197 = "arith.index_cast"(%196) : (i32) -> index
          %198 = "arith.subi"(%arg11, %196) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %199 = "arith.minsi"(%198, %15) : (i32, i32) -> i32
          %200 = "arith.index_cast"(%199) : (i32) -> index
          %201 = "arith.constant"() <{value = 32 : i32}> : () -> i32
          %202 = "arith.muli"(%186, %201) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %203 = "arith.index_cast"(%202) : (i32) -> index
          %204 = "memref.subview"(%34, %193, %195, %203, %200) <{operandSegmentSizes = array<i32: 1, 3, 1, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 1, 1, -9223372036854775808>, static_strides = array<i64: 1, 1, 1>}> : (memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>, index, index, index, index) -> memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %205 = "memref.subview"(%93, %200) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<32xi32, strided<[1]>, #hivm.address_space<ub>>, index) -> memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          "memref.copy"(%204, %205) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) -> ()
          "scf.for"(%20, %199, %23) ({
          ^bb0(%arg26: i32):
            %208 = "arith.index_cast"(%arg26) : (i32) -> index
            %209 = "memref.load"(%93, %208) : (memref<32xi32, strided<[1]>, #hivm.address_space<ub>>, index) -> i32
            %210 = "arith.cmpi"(%209, %14) <{predicate = 1 : i64}> : (i32, i32) -> i1
            "scf.if"(%210) ({
              %211 = "arith.index_cast"(%arg26) : (i32) -> index
              %212 = "arith.index_cast"(%arg25) : (i32) -> index
              %213 = "memref.subview"(%94, %212) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 1, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>>, index) -> memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>>
              %214 = "memref.collapse_shape"(%213) <{reassociation = [[0, 1], [2]]}> : (memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>>) -> memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
              "memref.store"(%13, %214, %2, %211) : (f32, memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
              %215 = "arith.divsi"(%41, %arg9) : (i32, i32) -> i32
              %216 = "arith.index_cast"(%215) : (i32) -> index
              %217 = "arith.index_cast"(%209) : (i32) -> index
              %218 = "memref.subview"(%38, %216, %217) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>}> : (memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>, index, index) -> memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %219 = "memref.subview"(%92, %211) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 1, 512>, static_strides = array<i64: 1, 1>}> : (memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>, index) -> memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              "memref.copy"(%218, %219) : (memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<512xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>) -> ()
              "scf.yield"() : () -> ()
            }, {
            }) : (i1) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          %206 = "memref.subview"(%44, %188) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 512>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x512xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %207 = "memref.collapse_shape"(%206) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          "memref.copy"(%92, %207) : (memref<32x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>, memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
          "scf.yield"() : () -> ()
        }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : (i32, i32, i32) -> ()
        %109 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %110 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %111 = "arith.constant"() <{value = 1 : i32}> : () -> i32
        "scf.for"(%109, %110, %111) ({
        ^bb0(%arg24: i32):
          %169 = "arith.constant"() <{value = 2 : i32}> : () -> i32
          %170 = "arith.muli"(%arg20, %169) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %171 = "arith.addi"(%170, %arg24) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %172 = "arith.index_cast"(%171) : (i32) -> index
          %173 = "arith.index_cast"(%arg24) : (i32) -> index
          %174 = "memref.subview"(%44, %173) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 512>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x32x512xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %175 = "memref.collapse_shape"(%174) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x32x512xbf16, strided<[32768, 16384, 512, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          %176 = "arith.index_cast"(%arg24) : (i32) -> index
          %177 = "memref.subview"(%87, %176) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 32, 512>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x512xbf16, strided<[16384, 512, 1]>, #hivm.address_space<cbuf>>, index) -> memref<1x32x512xbf16, strided<[16384, 512, 1], offset: ?>, #hivm.address_space<cbuf>>
          %178 = "memref.collapse_shape"(%177) <{reassociation = [[0, 1], [2]]}> : (memref<1x32x512xbf16, strided<[16384, 512, 1], offset: ?>, #hivm.address_space<cbuf>>) -> memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<cbuf>>
          "memref.copy"(%175, %178) : (memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>, memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          %179 = "arith.index_cast"(%arg24) : (i32) -> index
          %180 = "memref.subview"(%87, %179) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 32, 512>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x512xbf16, strided<[16384, 512, 1]>, #hivm.address_space<cbuf>>, index) -> memref<1x32x512xbf16, strided<[16384, 512, 1], offset: ?>, #hivm.address_space<cbuf>>
          %181 = "memref.collapse_shape"(%180) <{reassociation = [[0, 1], [2]]}> : (memref<1x32x512xbf16, strided<[16384, 512, 1], offset: ?>, #hivm.address_space<cbuf>>) -> memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<cbuf>>
          "hivm.hir.mmadL1"(%48, %181, %12, %1, %7, %0, %89) <{b_transpose, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> : (memref<16x512xbf16, strided<[512, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>) -> ()
          %182 = "memref.subview"(%45, %173) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 16, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x16x32xf32, #hivm.address_space<gm>>, index) -> memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %183 = "memref.collapse_shape"(%182) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x16x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          "memref.copy"(%89, %183) : (memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>, memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
          "scf.yield"() : () -> ()
        }) {hivm.tcore_type = #hivm.tcore_type<CUBE>} : (i32, i32, i32) -> ()
        %112 = "arith.muli"(%43, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %113 = "arith.index_cast"(%112) : (i32) -> index
        %114 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %115 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %116 = "arith.constant"() <{value = 1 : i32}> : () -> i32
        "scf.for"(%114, %115, %116) ({
        ^bb0(%arg23: i32):
          %145 = "arith.constant"() <{value = 2 : i32}> : () -> i32
          %146 = "arith.muli"(%arg20, %145) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %147 = "arith.addi"(%146, %arg23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %148 = "arith.index_cast"(%147) : (i32) -> index
          %149 = "arith.index_cast"(%arg23) : (i32) -> index
          %150 = "memref.subview"(%45, %149, %113) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 8, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x16x32xf32, #hivm.address_space<gm>>, index, index) -> memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %151 = "memref.collapse_shape"(%150) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          "memref.copy"(%151, %95) : (memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
          %152 = "memref.subview"(%49) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 8, 1>, static_strides = array<i64: 1, 1>}> : (memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %153 = "memref.subview"(%50) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 8, 1>, static_strides = array<i64: 1, 1>}> : (memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          "memref.copy"(%152, %153) : (memref<8xf32, strided<[1]>, #hivm.address_space<ub>>, memref<8xf32, strided<[1]>, #hivm.address_space<ub>>) -> ()
          "hivm.hir.vmul"(%95, %10, %95) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
          "hivm.hir.vreduce"(%95, %49) <{arith = #hivm.reduce_op<max>, operandSegmentSizes = array<i32: 1, 1, 0, 0>, reduce_dims = array<i64: 1>}> : (memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> ()
          "hivm.hir.vsub"(%50, %49, %99) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> ()
          %154 = "arith.index_cast"(%arg23) : (i32) -> index
          %155 = "memref.subview"(%96, %154) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 8, 1>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x8x1xf32, strided<[8, 1, 1]>, #hivm.address_space<ub>>, index) -> memref<1x8x1xf32, strided<[8, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %156 = "memref.collapse_shape"(%155) <{reassociation = [[0, 1], [2]]}> : (memref<1x8x1xf32, strided<[8, 1, 1], offset: ?>, #hivm.address_space<ub>>) -> memref<8x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          "hivm.hir.vexp"(%99, %156) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
          %157 = "memref.subview"(%49) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 8, 1>, static_strides = array<i64: 1, 1>}> : (memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %158 = "memref.subview"(%100) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 8, 1>, static_strides = array<i64: 1, 1>}> : (memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          "memref.copy"(%157, %158) : (memref<8xf32, strided<[1]>, #hivm.address_space<ub>>, memref<8xf32, strided<[1]>, #hivm.address_space<ub>>) -> ()
          "hivm.hir.vbrc"(%100, %101) <{broadcast_dims = array<i64: 1>}> : (memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
          "hivm.hir.vsub"(%95, %101, %102) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
          "hivm.hir.vexp"(%102, %95) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
          %159 = "arith.index_cast"(%arg23) : (i32) -> index
          %160 = "memref.subview"(%94, %159) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 1, 32>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>>, index) -> memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>>
          %161 = "memref.collapse_shape"(%160) <{reassociation = [[0, 1], [2]]}> : (memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>>) -> memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
          %162 = "memref.subview"(%161) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 32>, static_strides = array<i64: 1, 1>}> : (memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) -> memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %163 = "memref.subview"(%103) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 32>, static_strides = array<i64: 1, 1>}> : (memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) -> memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          "memref.copy"(%162, %163) : (memref<32xf32, strided<[1]>, #hivm.address_space<ub>>, memref<32xf32, strided<[1]>, #hivm.address_space<ub>>) -> ()
          "hivm.hir.vbrc"(%103, %104) <{broadcast_dims = array<i64: 0>}> : (memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
          "hivm.hir.vmul"(%95, %104, %95) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
          "hivm.hir.vreduce"(%95, %97) <{arith = #hivm.reduce_op<sum>, operandSegmentSizes = array<i32: 1, 1, 0, 0>, reduce_dims = array<i64: 1>}> : (memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> ()
          %164 = "arith.index_cast"(%arg23) : (i32) -> index
          %165 = "memref.subview"(%96, %164) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 8, 1>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x8x1xf32, strided<[8, 1, 1]>, #hivm.address_space<ub>>, index) -> memref<1x8x1xf32, strided<[8, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %166 = "memref.collapse_shape"(%165) <{reassociation = [[0, 1], [2]]}> : (memref<1x8x1xf32, strided<[8, 1, 1], offset: ?>, #hivm.address_space<ub>>) -> memref<8x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          "hivm.hir.vmul"(%51, %166, %105) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> ()
          "hivm.hir.vadd"(%105, %97, %51) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> ()
          "hivm.hir.vcast"(%95, %90) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) -> ()
          %167 = "memref.subview"(%46, %149, %113) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 8, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x16x32xbf16, #hivm.address_space<gm>>, index, index) -> memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %168 = "memref.collapse_shape"(%167) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          "memref.copy"(%90, %168) : (memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
          "scf.yield"() : () -> ()
        }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : (i32, i32, i32) -> ()
        %117 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %118 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %119 = "arith.constant"() <{value = 1 : i32}> : () -> i32
        "scf.for"(%117, %118, %119) ({
        ^bb0(%arg22: i32):
          %133 = "arith.constant"() <{value = 2 : i32}> : () -> i32
          %134 = "arith.muli"(%arg20, %133) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %135 = "arith.addi"(%134, %arg22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %136 = "arith.index_cast"(%135) : (i32) -> index
          %137 = "arith.index_cast"(%arg22) : (i32) -> index
          %138 = "memref.subview"(%46, %137) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 16, 32>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x16x32xbf16, #hivm.address_space<gm>>, index) -> memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %139 = "memref.collapse_shape"(%138) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x16x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          "memref.copy"(%139, %88) : (memref<16x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>, memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>) -> ()
          %140 = "arith.index_cast"(%arg22) : (i32) -> index
          %141 = "memref.subview"(%87, %140) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 32, 512>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x32x512xbf16, strided<[16384, 512, 1]>, #hivm.address_space<cbuf>>, index) -> memref<1x32x512xbf16, strided<[16384, 512, 1], offset: ?>, #hivm.address_space<cbuf>>
          %142 = "memref.collapse_shape"(%141) <{reassociation = [[0, 1], [2]]}> : (memref<1x32x512xbf16, strided<[16384, 512, 1], offset: ?>, #hivm.address_space<cbuf>>) -> memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<cbuf>>
          "hivm.hir.mmadL1"(%88, %142, %12, %1, %0, %7, %91) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> : (memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x512xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>) -> ()
          %143 = "memref.subview"(%47, %137) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0, 0>, static_sizes = array<i64: 1, 1, 16, 512>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x16x512xf32, #hivm.address_space<gm>>, index) -> memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %144 = "memref.collapse_shape"(%143) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x16x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          "memref.copy"(%91, %144) : (memref<16x512xf32, strided<[512, 1]>, #hivm.address_space<cc>>, memref<16x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) -> ()
          "scf.yield"() : () -> ()
        }) {hivm.tcore_type = #hivm.tcore_type<CUBE>} : (i32, i32, i32) -> ()
        %120 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %121 = "arith.constant"() <{value = 2 : i32}> : () -> i32
        %122 = "arith.constant"() <{value = 1 : i32}> : () -> i32
        "scf.for"(%120, %121, %122) ({
        ^bb0(%arg21: i32):
          %123 = "arith.constant"() <{value = 2 : i32}> : () -> i32
          %124 = "arith.muli"(%arg20, %123) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %125 = "arith.addi"(%124, %arg21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %126 = "arith.index_cast"(%125) : (i32) -> index
          %127 = "arith.index_cast"(%arg21) : (i32) -> index
          %128 = "memref.subview"(%47, %127, %113) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 8, 512>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<2x2x16x512xf32, #hivm.address_space<gm>>, index, index) -> memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>
          %129 = "memref.collapse_shape"(%128) <{reassociation = [[0, 1, 2], [3]]}> : (memref<1x1x8x512xf32, strided<[16384, 8192, 512, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
          "memref.copy"(%129, %98) : (memref<8x512xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) -> ()
          %130 = "arith.index_cast"(%arg21) : (i32) -> index
          %131 = "memref.subview"(%96, %130) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 8, 1>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x8x1xf32, strided<[8, 1, 1]>, #hivm.address_space<ub>>, index) -> memref<1x8x1xf32, strided<[8, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %132 = "memref.collapse_shape"(%131) <{reassociation = [[0, 1], [2]]}> : (memref<1x8x1xf32, strided<[8, 1, 1], offset: ?>, #hivm.address_space<ub>>) -> memref<8x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          "hivm.hir.vmul"(%52, %132, %52) <{broadcast = array<i64: 1>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) -> ()
          "hivm.hir.vadd"(%52, %98, %52) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) -> ()
          "scf.yield"() : () -> ()
        }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : (i32, i32, i32) -> ()
        "scf.yield"() : () -> ()
      }) {tilelangir.num_stages = 2 : i32} : (i32, i32, i32) -> ()
      "scf.for"(%20, %11, %23) ({
      ^bb0(%arg19: i32):
        %76 = "arith.muli"(%arg18, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %77 = "arith.addi"(%76, %43) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %78 = "arith.cmpi"(%77, %11) <{predicate = 2 : i64}> : (i32, i32) -> i1
        "scf.if"(%78) ({
          %80 = "arith.muli"(%arg18, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %81 = "arith.muli"(%43, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %82 = "arith.addi"(%80, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %83 = "arith.addi"(%82, %arg19) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %84 = "arith.index_cast"(%83) : (i32) -> index
          %85 = "memref.load"(%39, %84) : (memref<64xf32, strided<[1]>, #hivm.address_space<gm>>, index) -> f32
          %86 = "arith.index_cast"(%arg19) : (i32) -> index
          "memref.store"(%85, %50, %86, %2) : (f32, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, index, index) -> ()
          "scf.yield"() : () -> ()
        }, {
          %79 = "arith.index_cast"(%arg19) : (i32) -> index
          "memref.store"(%18, %50, %79, %2) : (f32, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (i1) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "hivm.hir.vsub"(%50, %49, %54) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vexp"(%54, %55) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vadd"(%51, %55, %51) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vdiv"(%52, %51, %52) <{broadcast = array<i64: 1>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>) -> ()
      "hivm.hir.vcast"(%52, %53) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<8x512xf32, strided<[512, 1]>, #hivm.address_space<ub>>, memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>) -> ()
      %67 = "arith.muli"(%43, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %68 = "arith.subi"(%21, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %69 = "arith.subi"(%68, %60) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %70 = "arith.minsi"(%69, %11) : (i32, i32) -> i32
      %71 = "arith.index_cast"(%70) : (i32) -> index
      %72 = "memref.subview"(%53, %71) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, 512>, static_strides = array<i64: 1, 1>}> : (memref<8x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>, index) -> memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>
      %73 = "arith.addi"(%60, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %74 = "arith.index_cast"(%73) : (i32) -> index
      %75 = "memref.subview"(%29, %57, %59, %74, %71) <{operandSegmentSizes = array<i32: 1, 3, 1, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, -9223372036854775808, 512>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>, index, index, index, index) -> memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%72, %75) : (memref<?x512xbf16, strided<[512, 1]>, #hivm.address_space<ub>>, memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) -> ()
      "scf.yield"() : () -> ()
    }) : (i32, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} : () -> ()
}) {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} : () -> ()



Traceback (most recent call last):
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/examples/deepseek_v4/inference/sparse_attn_mix_open_final.py", line 410, in <module>
    run_test()
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/examples/deepseek_v4/inference/sparse_attn_mix_open_final.py", line 402, in run_test
    output = sparse_attn(**data["inputs"])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/examples/deepseek_v4/inference/sparse_attn_mix_open_final.py", line 281, in sparse_attn
    sparse_attn.kernel = sparse_attn_mix_kernel(
                         ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/tilelang/jit/__init__.py", line 197, in wrapper
    kernel_result = compile(
                    ^^^^^^^^
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/tilelang/jit/__init__.py", line 74, in compile
    return compile_npuir.compile(func, out_idx)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/tilelang/jit/jit_npu.py", line 1125, in compile
    mlir_path = lower(self.mod)
                ^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/tilelang/engine/lower.py", line 305, in lower
    mlir_str = pipeline.run(mlir_str)
               ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/tilelang/tladapter/utils.py", line 103, in run
    return self._pp.run(mlir_str)
           ^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Pass pipeline run failed
[ERROR] 2026-05-08-10:55:43 (PID:2271438, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
