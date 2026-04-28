2026-04-28 19:29:28  [TileLang:tilelang.env:WARNING]: Loading tilelang libs from dev root: /home/z00910011/tilelang-ascend-test/tilelang-ascend/build
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
        workspace_kv = T.decl_buffer((32, 512), "bfloat16", scope="global.workspace;multi_buffer=2")
        workspace_mask = T.decl_buffer((1, 32), scope="global.workspace;multi_buffer=2")
        workspace_score = T.decl_buffer((16, 32), scope="global.workspace;multi_buffer=2")
        workspace_prob = T.decl_buffer((16, 32), "bfloat16", scope="global.workspace;multi_buffer=2")
        workspace_out = T.decl_buffer((16, 512), scope="global.workspace;multi_buffer=2")
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
                T.copy(T.region(kv_ub[0, 0], 1, 32, 512), T.region(workspace_kv[0, 0], 2, 32, 512))
                T.copy(T.region(mask_ub[0, 0], 1, 1, 32), T.region(workspace_mask[0, 0], 2, 1, 32))
                T.copy(T.region(workspace_kv[0, 0], 1, 32, 512), T.region(kv_shared[0, 0], 2, 32, 512))
                T.npuir_dot(T.region(q_shared[0, 0], 1, 16, 512), T.region(kv_shared[0, 0], 1, 512, 32), T.region(scores[0, 0], 3, 16, 32), T.bool(True), T.bool(False), T.bool(True))
                T.copy(T.region(scores[0, 0], 1, 16, 32), T.region(workspace_score[0, 0], 2, 16, 32))
                T.copy(T.region(workspace_score[vid * 8, 0], 1, 8, 32), T.region(scores_ub[0, 0], 2, 8, 32))
                T.copy(T.region(workspace_mask[0, 0], 1, 1, 32), T.region(mask_ub[0, 0], 2, 1, 32))
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
                T.copy(T.region(scores_cast[0, 0], 1, 8, 32), T.region(workspace_prob[vid * 8, 0], 2, 8, 32))
                T.copy(T.region(workspace_prob[0, 0], 1, 16, 32), T.region(prob_shared[0, 0], 2, 16, 32))
                T.copy(T.region(workspace_kv[0, 0], 1, 32, 512), T.region(kv_shared[0, 0], 2, 32, 512))
                T.npuir_dot(T.region(prob_shared[0, 0], 1, 16, 32), T.region(kv_shared[0, 0], 1, 32, 512), T.region(pv_acc[0, 0], 3, 16, 512), T.bool(True), T.bool(False), T.bool(False))
                T.copy(T.region(pv_acc[0, 0], 1, 16, 512), T.region(workspace_out[0, 0], 2, 16, 512))
                T.copy(T.region(workspace_out[vid * 8, 0], 1, 8, 512), T.region(acc_o_new[0, 0], 2, 8, 512))
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
    %21 = memref_ext.alloc_workspace() : memref<32x512xbf16>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16>
    %22 = memref_ext.alloc_workspace() : memref<1x32xf32>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32>
    %23 = memref_ext.alloc_workspace() : memref<16x32xf32>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<16x32xf32>
    %24 = memref_ext.alloc_workspace() : memref<16x32xbf16>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<16x32xbf16>
    %25 = memref_ext.alloc_workspace() : memref<16x512xf32>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<16x512xf32>
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
      %26 = arith.sitofp %c0_i32_12 : i32 to f32
      hivm.hir.vbrc ins(%26 : f32) outs(%alloc_8 : memref<8x512xf32, strided<[512, 1]>>)
      %27 = arith.sitofp %c0_i32_12 : i32 to f32
      hivm.hir.vbrc ins(%27 : f32) outs(%alloc_7 : memref<8x1xf32, strided<[1, 1]>>)
      %cst = arith.constant 0xFF800000 : f32
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<8x1xf32, strided<[1, 1]>>)
      %28 = arith.divsi %18, %arg9 : i32
      %29 = arith.index_cast %28 : i32 to index
      %30 = arith.remsi %18, %arg9 : i32
      %31 = arith.index_cast %30 : i32 to index
      %c16_i32 = arith.constant 16 : i32
      %32 = arith.muli %arg18, %c16_i32 : i32
      %33 = arith.index_cast %32 : i32 to index
      %subview = memref.subview %reinterpret_cast[%29, %31, %33, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>>
      %c31_i32 = arith.constant 31 : i32
      %34 = arith.addi %arg11, %c31_i32 : i32
      %c32_i32 = arith.constant 32 : i32
      %35 = arith.divsi %34, %c32_i32 : i32
      %c1_i32_13 = arith.constant 1 : i32
      scf.for %arg19 = %c0_i32_12 to %35 step %c1_i32_13  : i32 {
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
        %43 = arith.sitofp %c0_i32_39 : i32 to bf16
        hivm.hir.vbrc ins(%43 : bf16) outs(%alloc_23 : memref<32x512xbf16, strided<[512, 1]>>)
        %44 = arith.sitofp %c0_i32_39 : i32 to f32
        hivm.hir.vbrc ins(%44 : f32) outs(%alloc_25 : memref<1x32xf32, strided<[32, 1]>>)
        %45 = arith.divsi %18, %arg9 : i32
        %46 = arith.index_cast %45 : i32 to index
        %47 = arith.remsi %18, %arg9 : i32
        %48 = arith.index_cast %47 : i32 to index
        %c32_i32_40 = arith.constant 32 : i32
        %49 = arith.muli %arg19, %c32_i32_40 : i32
        %50 = arith.index_cast %49 : i32 to index
        %51 = arith.subi %arg11, %49 : i32
        %52 = arith.minsi %51, %c32_i32_40 : i32
        %53 = arith.index_cast %52 : i32 to index
        %subview_41 = memref.subview %reinterpret_cast_1[%46, %48, %50] [1, 1, %53] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_42 = memref.subview %alloc_24[0] [%53] [1] : memref<32xi32, strided<[1]>> to memref<?xi32, strided<[1]>>
        memref.copy %subview_41, %subview_42 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>>
        %c1_i32_43 = arith.constant 1 : i32
        scf.for %arg20 = %c0_i32_39 to %52 step %c1_i32_43  : i32 {
          %59 = arith.index_cast %arg20 : i32 to index
          %60 = memref.load %alloc_24[%59] : memref<32xi32, strided<[1]>>
          %c-1_i32 = arith.constant -1 : i32
          %61 = arith.cmpi ne, %60, %c-1_i32 : i32
          scf.if %61 {
            %cst_63 = arith.constant 1.000000e+00 : f32
            %c0_i32_64 = arith.constant 0 : i32
            %62 = arith.index_cast %c0_i32_64 : i32 to index
            %63 = arith.index_cast %arg20 : i32 to index
            memref.store %cst_63, %alloc_25[%62, %63] : memref<1x32xf32, strided<[32, 1]>>
            %64 = arith.divsi %18, %arg9 : i32
            %65 = arith.index_cast %64 : i32 to index
            %66 = arith.index_cast %60 : i32 to index
            %subview_65 = memref.subview %reinterpret_cast_2[%65, %66, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %subview_66 = memref.subview %alloc_23[%63, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>> to memref<512xbf16, strided<[1], offset: ?>>
            memref.copy %subview_65, %subview_66 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>>
          }
        }
        memref.copy %alloc_23, %21 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16>
        %subview_44 = memref.subview %alloc_25[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_45 = memref.subview %22[0, 0] [1, 32] [1, 1] : memref<1x32xf32> to memref<32xf32, strided<[1]>>
        memref.copy %subview_44, %subview_45 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        memref.copy %21, %alloc_18 : memref<32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        %true = arith.constant true
        %c16_i32_46 = arith.constant 16 : i32
        %54 = arith.index_cast %c16_i32_46 : i32 to index
        %c512_i32_47 = arith.constant 512 : i32
        %55 = arith.index_cast %c512_i32_47 : i32 to index
        %56 = arith.index_cast %c32_i32_40 : i32 to index
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_18, %true, %54, %55, %56 : memref<16x512xbf16, strided<[512, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_20 : memref<16x32xf32, strided<[32, 1]>>)
        memref.copy %alloc_20, %23 : memref<16x32xf32, strided<[32, 1]>> to memref<16x32xf32>
        %c8_i32_48 = arith.constant 8 : i32
        %57 = arith.muli %20, %c8_i32_48 : i32
        %58 = arith.index_cast %57 : i32 to index
        %subview_49 = memref.subview %23[%58, 0] [8, 32] [1, 1] : memref<16x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_49, %alloc_26 : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1]>>
        %subview_50 = memref.subview %22[0, 0] [1, 32] [1, 1] : memref<1x32xf32> to memref<32xf32, strided<[1]>>
        %subview_51 = memref.subview %alloc_25[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_50, %subview_51 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        %subview_52 = memref.subview %alloc_5[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_53 = memref.subview %alloc_6[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_52, %subview_53 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        %cst_54 = arith.constant 0.0441941731 : f32
        hivm.hir.vmul ins(%alloc_26, %cst_54 : memref<8x32xf32, strided<[32, 1]>>, f32) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_5 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vsub ins(%alloc_6, %alloc_5 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>>)
        %subview_55 = memref.subview %alloc_5[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_56 = memref.subview %alloc_31[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_55, %subview_56 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        %reinterpret_cast_57 = memref.reinterpret_cast %alloc_31 to offset: [0], sizes: [8, 1], strides: [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8x1xf32, strided<[1, 1]>>
        hivm.hir.vbrc ins(%reinterpret_cast_57 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_33 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [1]
        hivm.hir.vsub ins(%alloc_26, %alloc_33 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vexp ins(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>>)
        %subview_58 = memref.subview %alloc_25[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_59 = memref.subview %alloc_35[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_58, %subview_59 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        %reinterpret_cast_60 = memref.reinterpret_cast %alloc_35 to offset: [0], sizes: [1, 32], strides: [32, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<1x32xf32, strided<[32, 1]>>
        hivm.hir.vbrc ins(%reinterpret_cast_60 : memref<1x32xf32, strided<[32, 1]>>) outs(%alloc_37 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [0]
        hivm.hir.vmul ins(%alloc_26, %alloc_37 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_26 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_28 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmul ins(%alloc_7, %alloc_27 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_38 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_38, %alloc_28 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vcast ins(%alloc_26 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_21 : memref<8x32xbf16, strided<[32, 1]>>)
        %subview_61 = memref.subview %24[%58, 0] [8, 32] [1, 1] : memref<16x32xbf16> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_21, %subview_61 : memref<8x32xbf16, strided<[32, 1]>> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %24, %alloc_19 : memref<16x32xbf16> to memref<16x32xbf16, strided<[32, 1]>>
        memref.copy %21, %alloc_18 : memref<32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 ins(%alloc_19, %alloc_18, %true, %54, %56, %55 : memref<16x32xbf16, strided<[32, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_22 : memref<16x512xf32, strided<[512, 1]>>)
        memref.copy %alloc_22, %25 : memref<16x512xf32, strided<[512, 1]>> to memref<16x512xf32>
        %subview_62 = memref.subview %25[%58, 0] [8, 512] [1, 1] : memref<16x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
        memref.copy %subview_62, %alloc_29 : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<8x512xf32, strided<[512, 1]>>
        hivm.hir.vmul ins(%alloc_8, %alloc_27 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_8 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_8, %alloc_29 : memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_8 : memref<8x512xf32, strided<[512, 1]>>)
      } {tilelangir.num_stages = 2 : i32}
      %c8_i32 = arith.constant 8 : i32
      %c1_i32_14 = arith.constant 1 : i32
      scf.for %arg19 = %c0_i32_12 to %c8_i32 step %c1_i32_14  : i32 {
        %c2_i32 = arith.constant 2 : i32
        %43 = arith.muli %arg18, %c2_i32 : i32
        %44 = arith.addi %43, %20 : i32
        %c8_i32_18 = arith.constant 8 : i32
        %45 = arith.cmpi slt, %44, %c8_i32_18 : i32
        scf.if %45 {
          %c16_i32_19 = arith.constant 16 : i32
          %46 = arith.muli %arg18, %c16_i32_19 : i32
          %c8_i32_20 = arith.constant 8 : i32
          %47 = arith.muli %20, %c8_i32_20 : i32
          %48 = arith.addi %46, %47 : i32
          %49 = arith.addi %48, %arg19 : i32
          %50 = arith.index_cast %49 : i32 to index
          %51 = memref.load %reinterpret_cast_3[%50] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %52 = arith.index_cast %arg19 : i32 to index
          %c0_i32_21 = arith.constant 0 : i32
          %53 = arith.index_cast %c0_i32_21 : i32 to index
          memref.store %51, %alloc_6[%52, %53] : memref<8x1xf32, strided<[1, 1]>>
        } else {
          %cst_19 = arith.constant 0xFF800000 : f32
          %46 = arith.index_cast %arg19 : i32 to index
          %c0_i32_20 = arith.constant 0 : i32
          %47 = arith.index_cast %c0_i32_20 : i32 to index
          memref.store %cst_19, %alloc_6[%46, %47] : memref<8x1xf32, strided<[1, 1]>>
        }
      }
      hivm.hir.vsub ins(%alloc_6, %alloc_5 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_7, %alloc_11 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vdiv ins(%alloc_8, %alloc_7 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_8 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_8 : memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_9 : memref<8x512xbf16, strided<[512, 1]>>)
      %c64_i32_15 = arith.constant 64 : i32
      %36 = arith.muli %20, %c8_i32 : i32
      %37 = arith.subi %c64_i32_15, %36 : i32
      %38 = arith.subi %37, %32 : i32
      %39 = arith.minsi %c8_i32, %38 : i32
      %40 = arith.index_cast %39 : i32 to index
      %subview_16 = memref.subview %alloc_9[0, 0] [%40, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[512, 1]>>
      %41 = arith.addi %32, %36 : i32
      %42 = arith.index_cast %41 : i32 to index
      %subview_17 = memref.subview %reinterpret_cast_0[%29, %31, %42, 0] [1, 1, %40, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
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
    %15 = memref_ext.alloc_workspace() : memref<32x512xbf16>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16>
    %16 = memref_ext.alloc_workspace() : memref<1x32xf32>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32>
    %17 = memref_ext.alloc_workspace() : memref<16x32xf32>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<16x32xf32>
    %18 = memref_ext.alloc_workspace() : memref<16x32xbf16>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<16x32xbf16>
    %19 = memref_ext.alloc_workspace() : memref<16x512xf32>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<16x512xf32>
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
      %20 = arith.divsi %12, %arg9 : i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.remsi %12, %arg9 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = arith.muli %arg18, %c16_i32 : i32
      %25 = arith.index_cast %24 : i32 to index
      %subview = memref.subview %reinterpret_cast[%21, %23, %25, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>>
      %26 = arith.addi %arg11, %c31_i32 : i32
      %27 = arith.divsi %26, %c32_i32 : i32
      scf.for %arg19 = %c0_i32 to %27 step %c1_i32  : i32 {
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
        %35 = arith.divsi %12, %arg9 : i32
        %36 = arith.index_cast %35 : i32 to index
        %37 = arith.remsi %12, %arg9 : i32
        %38 = arith.index_cast %37 : i32 to index
        %39 = arith.muli %arg19, %c32_i32 : i32
        %40 = arith.index_cast %39 : i32 to index
        %41 = arith.subi %arg11, %39 : i32
        %42 = arith.minsi %41, %c32_i32 : i32
        %43 = arith.index_cast %42 : i32 to index
        %subview_36 = memref.subview %reinterpret_cast_5[%36, %38, %40] [1, 1, %43] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_37 = memref.subview %alloc_23[0] [%43] [1] : memref<32xi32, strided<[1]>> to memref<?xi32, strided<[1]>>
        memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>>
        scf.for %arg20 = %c0_i32 to %42 step %c1_i32  : i32 {
          %46 = arith.index_cast %arg20 : i32 to index
          %47 = memref.load %alloc_23[%46] : memref<32xi32, strided<[1]>>
          %48 = arith.cmpi ne, %47, %c-1_i32 : i32
          scf.if %48 {
            %49 = arith.index_cast %arg20 : i32 to index
            memref.store %cst_2, %alloc_24[%c0, %49] : memref<1x32xf32, strided<[32, 1]>>
            %50 = arith.divsi %12, %arg9 : i32
            %51 = arith.index_cast %50 : i32 to index
            %52 = arith.index_cast %47 : i32 to index
            %subview_51 = memref.subview %reinterpret_cast_6[%51, %52, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %subview_52 = memref.subview %alloc_22[%49, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>> to memref<512xbf16, strided<[1], offset: ?>>
            memref.copy %subview_51, %subview_52 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>>
          }
        }
        memref.copy %alloc_22, %15 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16>
        %subview_38 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_39 = memref.subview %16[0, 0] [1, 32] [1, 1] : memref<1x32xf32> to memref<32xf32, strided<[1]>>
        memref.copy %subview_38, %subview_39 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        memref.copy %15, %alloc_17 : memref<32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_17, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_19 : memref<16x32xf32, strided<[32, 1]>>)
        memref.copy %alloc_19, %17 : memref<16x32xf32, strided<[32, 1]>> to memref<16x32xf32>
        %44 = arith.muli %14, %c8_i32 : i32
        %45 = arith.index_cast %44 : i32 to index
        %subview_40 = memref.subview %17[%45, 0] [8, 32] [1, 1] : memref<16x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_40, %alloc_25 : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1]>>
        %subview_41 = memref.subview %16[0, 0] [1, 32] [1, 1] : memref<1x32xf32> to memref<32xf32, strided<[1]>>
        %subview_42 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_41, %subview_42 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        %subview_43 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_44 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_43, %subview_44 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vmul ins(%alloc_25, %cst_1 : memref<8x32xf32, strided<[32, 1]>>, f32) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_26 : memref<8x1xf32, strided<[1, 1]>>)
        %subview_45 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_46 = memref.subview %alloc_30[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_45, %subview_46 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_31 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [1]
        hivm.hir.vsub ins(%alloc_25, %alloc_31 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vexp ins(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        %subview_47 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_48 = memref.subview %alloc_33[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_47, %subview_48 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_33 : memref<1x32xf32, strided<[32, 1]>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [0]
        hivm.hir.vmul ins(%alloc_25, %alloc_34 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmul ins(%alloc_10, %alloc_26 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_35 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_35, %alloc_27 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vcast ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_20 : memref<8x32xbf16, strided<[32, 1]>>)
        %subview_49 = memref.subview %18[%45, 0] [8, 32] [1, 1] : memref<16x32xbf16> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_20, %subview_49 : memref<8x32xbf16, strided<[32, 1]>> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %18, %alloc_18 : memref<16x32xbf16> to memref<16x32xbf16, strided<[32, 1]>>
        memref.copy %15, %alloc_17 : memref<32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 ins(%alloc_18, %alloc_17, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_21 : memref<16x512xf32, strided<[512, 1]>>)
        memref.copy %alloc_21, %19 : memref<16x512xf32, strided<[512, 1]>> to memref<16x512xf32>
        %subview_50 = memref.subview %19[%45, 0] [8, 512] [1, 1] : memref<16x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
        memref.copy %subview_50, %alloc_28 : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<8x512xf32, strided<[512, 1]>>
        hivm.hir.vmul ins(%alloc_11, %alloc_26 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_11, %alloc_28 : memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
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
          memref.store %43, %alloc_9[%44, %c0] : memref<8x1xf32, strided<[1, 1]>>
        } else {
          %38 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%38, %c0] : memref<8x1xf32, strided<[1, 1]>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>>)
      %28 = arith.muli %14, %c8_i32 : i32
      %29 = arith.subi %c64_i32, %28 : i32
      %30 = arith.subi %29, %24 : i32
      %31 = arith.minsi %30, %c8_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview_15 = memref.subview %alloc_12[0, 0] [%32, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[512, 1]>>
      %33 = arith.addi %24, %28 : i32
      %34 = arith.index_cast %33 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast_4[%21, %23, %34, 0] [1, 1, %32, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
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
    %15 = memref_ext.alloc_workspace() : memref<32x512xbf16>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16>
    %16 = memref_ext.alloc_workspace() : memref<1x32xf32>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32>
    %17 = memref_ext.alloc_workspace() : memref<16x32xf32>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<16x32xf32>
    %18 = memref_ext.alloc_workspace() : memref<16x32xbf16>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<16x32xbf16>
    %19 = memref_ext.alloc_workspace() : memref<16x512xf32>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<16x512xf32>
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
      %20 = arith.divsi %12, %arg9 : i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.remsi %12, %arg9 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = arith.muli %arg18, %c16_i32 : i32
      %25 = arith.index_cast %24 : i32 to index
      %subview = memref.subview %reinterpret_cast[%21, %23, %25, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>>
      %26 = arith.addi %arg11, %c31_i32 : i32
      %27 = arith.divsi %26, %c32_i32 : i32
      scf.for %arg19 = %c0_i32 to %27 step %c1_i32  : i32 {
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
        %35 = arith.divsi %12, %arg9 : i32
        %36 = arith.index_cast %35 : i32 to index
        %37 = arith.remsi %12, %arg9 : i32
        %38 = arith.index_cast %37 : i32 to index
        %39 = arith.muli %arg19, %c32_i32 : i32
        %40 = arith.index_cast %39 : i32 to index
        %41 = arith.subi %arg11, %39 : i32
        %42 = arith.minsi %41, %c32_i32 : i32
        %43 = arith.index_cast %42 : i32 to index
        %subview_36 = memref.subview %reinterpret_cast_5[%36, %38, %40] [1, 1, %43] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_37 = memref.subview %alloc_23[0] [%43] [1] : memref<32xi32, strided<[1]>> to memref<?xi32, strided<[1]>>
        memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>>
        scf.for %arg20 = %c0_i32 to %42 step %c1_i32  : i32 {
          %46 = arith.index_cast %arg20 : i32 to index
          %47 = memref.load %alloc_23[%46] : memref<32xi32, strided<[1]>>
          %48 = arith.cmpi ne, %47, %c-1_i32 : i32
          scf.if %48 {
            %49 = arith.index_cast %arg20 : i32 to index
            memref.store %cst_2, %alloc_24[%c0, %49] : memref<1x32xf32, strided<[32, 1]>>
            %50 = arith.divsi %12, %arg9 : i32
            %51 = arith.index_cast %50 : i32 to index
            %52 = arith.index_cast %47 : i32 to index
            %subview_51 = memref.subview %reinterpret_cast_6[%51, %52, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %subview_52 = memref.subview %alloc_22[%49, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>> to memref<512xbf16, strided<[1], offset: ?>>
            memref.copy %subview_51, %subview_52 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>>
          }
        }
        memref.copy %alloc_22, %15 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16>
        %subview_38 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_39 = memref.subview %16[0, 0] [1, 32] [1, 1] : memref<1x32xf32> to memref<32xf32, strided<[1]>>
        memref.copy %subview_38, %subview_39 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        memref.copy %15, %alloc_17 : memref<32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_17, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_19 : memref<16x32xf32, strided<[32, 1]>>)
        memref.copy %alloc_19, %17 : memref<16x32xf32, strided<[32, 1]>> to memref<16x32xf32>
        %44 = arith.muli %14, %c8_i32 : i32
        %45 = arith.index_cast %44 : i32 to index
        %subview_40 = memref.subview %17[%45, 0] [8, 32] [1, 1] : memref<16x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_40, %alloc_25 : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1]>>
        %subview_41 = memref.subview %16[0, 0] [1, 32] [1, 1] : memref<1x32xf32> to memref<32xf32, strided<[1]>>
        %subview_42 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_41, %subview_42 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        %subview_43 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_44 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_43, %subview_44 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vmul ins(%alloc_25, %cst_1 : memref<8x32xf32, strided<[32, 1]>>, f32) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_26 : memref<8x1xf32, strided<[1, 1]>>)
        %subview_45 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_46 = memref.subview %alloc_30[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_45, %subview_46 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_31 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [1]
        hivm.hir.vsub ins(%alloc_25, %alloc_31 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vexp ins(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        %subview_47 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_48 = memref.subview %alloc_33[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_47, %subview_48 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_33 : memref<1x32xf32, strided<[32, 1]>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [0]
        hivm.hir.vmul ins(%alloc_25, %alloc_34 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmul ins(%alloc_10, %alloc_26 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_35 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_35, %alloc_27 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vcast ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_20 : memref<8x32xbf16, strided<[32, 1]>>)
        %subview_49 = memref.subview %18[%45, 0] [8, 32] [1, 1] : memref<16x32xbf16> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_20, %subview_49 : memref<8x32xbf16, strided<[32, 1]>> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %18, %alloc_18 : memref<16x32xbf16> to memref<16x32xbf16, strided<[32, 1]>>
        memref.copy %15, %alloc_17 : memref<32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 ins(%alloc_18, %alloc_17, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_21 : memref<16x512xf32, strided<[512, 1]>>)
        memref.copy %alloc_21, %19 : memref<16x512xf32, strided<[512, 1]>> to memref<16x512xf32>
        %subview_50 = memref.subview %19[%45, 0] [8, 512] [1, 1] : memref<16x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
        memref.copy %subview_50, %alloc_28 : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<8x512xf32, strided<[512, 1]>>
        hivm.hir.vmul ins(%alloc_11, %alloc_26 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_11, %alloc_28 : memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
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
          memref.store %43, %alloc_9[%44, %c0] : memref<8x1xf32, strided<[1, 1]>>
        } else {
          %38 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%38, %c0] : memref<8x1xf32, strided<[1, 1]>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>>)
      %28 = arith.muli %14, %c8_i32 : i32
      %29 = arith.subi %c64_i32, %28 : i32
      %30 = arith.subi %29, %24 : i32
      %31 = arith.minsi %30, %c8_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview_15 = memref.subview %alloc_12[0, 0] [%32, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[512, 1]>>
      %33 = arith.addi %24, %28 : i32
      %34 = arith.index_cast %33 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast_4[%21, %23, %34, 0] [1, 1, %32, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
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
    %15 = memref_ext.alloc_workspace() : memref<32x512xbf16>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16>
    %16 = memref_ext.alloc_workspace() : memref<1x32xf32>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32>
    %17 = memref_ext.alloc_workspace() : memref<16x32xf32>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<16x32xf32>
    %18 = memref_ext.alloc_workspace() : memref<16x32xbf16>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<16x32xbf16>
    %19 = memref_ext.alloc_workspace() : memref<16x512xf32>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<16x512xf32>
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
      %20 = arith.divsi %12, %arg9 : i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.remsi %12, %arg9 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = arith.muli %arg18, %c16_i32 : i32
      %25 = arith.index_cast %24 : i32 to index
      %subview = memref.subview %reinterpret_cast[%21, %23, %25, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>>
      %26 = arith.addi %arg11, %c31_i32 : i32
      %27 = arith.divsi %26, %c32_i32 : i32
      scf.for %arg19 = %c0_i32 to %27 step %c1_i32  : i32 {
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
        %35 = arith.divsi %12, %arg9 : i32
        %36 = arith.index_cast %35 : i32 to index
        %37 = arith.remsi %12, %arg9 : i32
        %38 = arith.index_cast %37 : i32 to index
        %39 = arith.muli %arg19, %c32_i32 : i32
        %40 = arith.index_cast %39 : i32 to index
        %41 = arith.subi %arg11, %39 : i32
        %42 = arith.minsi %41, %c32_i32 : i32
        %43 = arith.index_cast %42 : i32 to index
        %subview_36 = memref.subview %reinterpret_cast_5[%36, %38, %40] [1, 1, %43] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_37 = memref.subview %alloc_23[0] [%43] [1] : memref<32xi32, strided<[1]>> to memref<?xi32, strided<[1]>>
        memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>>
        scf.for %arg20 = %c0_i32 to %42 step %c1_i32  : i32 {
          %46 = arith.index_cast %arg20 : i32 to index
          %47 = memref.load %alloc_23[%46] : memref<32xi32, strided<[1]>>
          %48 = arith.cmpi ne, %47, %c-1_i32 : i32
          scf.if %48 {
            %49 = arith.index_cast %arg20 : i32 to index
            memref.store %cst_2, %alloc_24[%c0, %49] : memref<1x32xf32, strided<[32, 1]>>
            %50 = arith.divsi %12, %arg9 : i32
            %51 = arith.index_cast %50 : i32 to index
            %52 = arith.index_cast %47 : i32 to index
            %subview_51 = memref.subview %reinterpret_cast_6[%51, %52, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %subview_52 = memref.subview %alloc_22[%49, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>> to memref<512xbf16, strided<[1], offset: ?>>
            memref.copy %subview_51, %subview_52 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>>
          }
        }
        memref.copy %alloc_22, %15 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16>
        %subview_38 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_39 = memref.subview %16[0, 0] [1, 32] [1, 1] : memref<1x32xf32> to memref<32xf32, strided<[1]>>
        memref.copy %subview_38, %subview_39 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        memref.copy %15, %alloc_17 : memref<32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_17, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_19 : memref<16x32xf32, strided<[32, 1]>>)
        memref.copy %alloc_19, %17 : memref<16x32xf32, strided<[32, 1]>> to memref<16x32xf32>
        %44 = arith.muli %14, %c8_i32 : i32
        %45 = arith.index_cast %44 : i32 to index
        %subview_40 = memref.subview %17[%45, 0] [8, 32] [1, 1] : memref<16x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_40, %alloc_25 : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1]>>
        %subview_41 = memref.subview %16[0, 0] [1, 32] [1, 1] : memref<1x32xf32> to memref<32xf32, strided<[1]>>
        %subview_42 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_41, %subview_42 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        %subview_43 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_44 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_43, %subview_44 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vmul ins(%alloc_25, %cst_1 : memref<8x32xf32, strided<[32, 1]>>, f32) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_26 : memref<8x1xf32, strided<[1, 1]>>)
        %subview_45 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_46 = memref.subview %alloc_30[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_45, %subview_46 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_31 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [1]
        hivm.hir.vsub ins(%alloc_25, %alloc_31 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vexp ins(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        %subview_47 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_48 = memref.subview %alloc_33[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_47, %subview_48 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_33 : memref<1x32xf32, strided<[32, 1]>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [0]
        hivm.hir.vmul ins(%alloc_25, %alloc_34 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmul ins(%alloc_10, %alloc_26 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_35 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_35, %alloc_27 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vcast ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_20 : memref<8x32xbf16, strided<[32, 1]>>)
        %subview_49 = memref.subview %18[%45, 0] [8, 32] [1, 1] : memref<16x32xbf16> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_20, %subview_49 : memref<8x32xbf16, strided<[32, 1]>> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %18, %alloc_18 : memref<16x32xbf16> to memref<16x32xbf16, strided<[32, 1]>>
        memref.copy %15, %alloc_17 : memref<32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 ins(%alloc_18, %alloc_17, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_21 : memref<16x512xf32, strided<[512, 1]>>)
        memref.copy %alloc_21, %19 : memref<16x512xf32, strided<[512, 1]>> to memref<16x512xf32>
        %subview_50 = memref.subview %19[%45, 0] [8, 512] [1, 1] : memref<16x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
        memref.copy %subview_50, %alloc_28 : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<8x512xf32, strided<[512, 1]>>
        hivm.hir.vmul ins(%alloc_11, %alloc_26 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_11, %alloc_28 : memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
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
          memref.store %43, %alloc_9[%44, %c0] : memref<8x1xf32, strided<[1, 1]>>
        } else {
          %38 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%38, %c0] : memref<8x1xf32, strided<[1, 1]>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>>)
      %28 = arith.muli %14, %c8_i32 : i32
      %29 = arith.subi %c64_i32, %28 : i32
      %30 = arith.subi %29, %24 : i32
      %31 = arith.minsi %30, %c8_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview_15 = memref.subview %alloc_12[0, 0] [%32, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[512, 1]>>
      %33 = arith.addi %24, %28 : i32
      %34 = arith.index_cast %33 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast_4[%21, %23, %34, 0] [1, 1, %32, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
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
    %15 = memref_ext.alloc_workspace() : memref<32x512xbf16>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<32x512xbf16>
    %16 = memref_ext.alloc_workspace() : memref<1x32xf32>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<1x32xf32>
    %17 = memref_ext.alloc_workspace() : memref<16x32xf32>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<16x32xf32>
    %18 = memref_ext.alloc_workspace() : memref<16x32xbf16>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<16x32xbf16>
    %19 = memref_ext.alloc_workspace() : memref<16x512xf32>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<16x512xf32>
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
      %20 = arith.divsi %12, %arg9 : i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.remsi %12, %arg9 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = arith.muli %arg18, %c16_i32 : i32
      %25 = arith.index_cast %24 : i32 to index
      %subview = memref.subview %reinterpret_cast[%21, %23, %25, 0] [1, 1, 16, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<16x512xbf16, strided<[512, 1]>>
      %26 = arith.addi %arg11, %c31_i32 : i32
      %27 = arith.divsi %26, %c32_i32 : i32
      scf.for %arg19 = %c0_i32 to %27 step %c1_i32  : i32 {
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
        %35 = arith.divsi %12, %arg9 : i32
        %36 = arith.index_cast %35 : i32 to index
        %37 = arith.remsi %12, %arg9 : i32
        %38 = arith.index_cast %37 : i32 to index
        %39 = arith.muli %arg19, %c32_i32 : i32
        %40 = arith.index_cast %39 : i32 to index
        %41 = arith.subi %arg11, %39 : i32
        %42 = arith.minsi %41, %c32_i32 : i32
        %43 = arith.index_cast %42 : i32 to index
        %subview_36 = memref.subview %reinterpret_cast_5[%36, %38, %40] [1, 1, %43] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_37 = memref.subview %alloc_23[0] [%43] [1] : memref<32xi32, strided<[1]>> to memref<?xi32, strided<[1]>>
        memref.copy %subview_36, %subview_37 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>>
        scf.for %arg20 = %c0_i32 to %42 step %c1_i32  : i32 {
          %46 = arith.index_cast %arg20 : i32 to index
          %47 = memref.load %alloc_23[%46] : memref<32xi32, strided<[1]>>
          %48 = arith.cmpi ne, %47, %c-1_i32 : i32
          scf.if %48 {
            %49 = arith.index_cast %arg20 : i32 to index
            memref.store %cst_2, %alloc_24[%c0, %49] : memref<1x32xf32, strided<[32, 1]>>
            %50 = arith.divsi %12, %arg9 : i32
            %51 = arith.index_cast %50 : i32 to index
            %52 = arith.index_cast %47 : i32 to index
            %subview_51 = memref.subview %reinterpret_cast_6[%51, %52, 0] [1, 1, 512] [1, 1, 1] : memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %subview_52 = memref.subview %alloc_22[%49, 0] [1, 512] [1, 1] : memref<32x512xbf16, strided<[512, 1]>> to memref<512xbf16, strided<[1], offset: ?>>
            memref.copy %subview_51, %subview_52 : memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<512xbf16, strided<[1], offset: ?>>
          }
        }
        memref.copy %alloc_22, %15 : memref<32x512xbf16, strided<[512, 1]>> to memref<32x512xbf16>
        %subview_38 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_39 = memref.subview %16[0, 0] [1, 32] [1, 1] : memref<1x32xf32> to memref<32xf32, strided<[1]>>
        memref.copy %subview_38, %subview_39 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        memref.copy %15, %alloc_17 : memref<32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_17, %true, %c16, %c512, %c32 : memref<16x512xbf16, strided<[512, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_19 : memref<16x32xf32, strided<[32, 1]>>)
        memref.copy %alloc_19, %17 : memref<16x32xf32, strided<[32, 1]>> to memref<16x32xf32>
        %44 = arith.muli %14, %c8_i32 : i32
        %45 = arith.index_cast %44 : i32 to index
        %subview_40 = memref.subview %17[%45, 0] [8, 32] [1, 1] : memref<16x32xf32> to memref<8x32xf32, strided<[32, 1], offset: ?>>
        memref.copy %subview_40, %alloc_25 : memref<8x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1]>>
        %subview_41 = memref.subview %16[0, 0] [1, 32] [1, 1] : memref<1x32xf32> to memref<32xf32, strided<[1]>>
        %subview_42 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_41, %subview_42 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        %subview_43 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_44 = memref.subview %alloc_9[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_43, %subview_44 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vmul ins(%alloc_25, %cst_1 : memref<8x32xf32, strided<[32, 1]>>, f32) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_8 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_29 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_26 : memref<8x1xf32, strided<[1, 1]>>)
        %subview_45 = memref.subview %alloc_8[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        %subview_46 = memref.subview %alloc_30[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>> to memref<8xf32, strided<[1]>>
        memref.copy %subview_45, %subview_46 : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_30 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_31 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [1]
        hivm.hir.vsub ins(%alloc_25, %alloc_31 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vexp ins(%alloc_32 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        %subview_47 = memref.subview %alloc_24[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        %subview_48 = memref.subview %alloc_33[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>> to memref<32xf32, strided<[1]>>
        memref.copy %subview_47, %subview_48 : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[1]>>
        hivm.hir.vbrc ins(%alloc_33 : memref<1x32xf32, strided<[32, 1]>>) outs(%alloc_34 : memref<8x32xf32, strided<[32, 1]>>) broadcast_dims = [0]
        hivm.hir.vmul ins(%alloc_25, %alloc_34 : memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_27 : memref<8x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmul ins(%alloc_10, %alloc_26 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_35 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_35, %alloc_27 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
        hivm.hir.vcast ins(%alloc_25 : memref<8x32xf32, strided<[32, 1]>>) outs(%alloc_20 : memref<8x32xbf16, strided<[32, 1]>>)
        %subview_49 = memref.subview %18[%45, 0] [8, 32] [1, 1] : memref<16x32xbf16> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %alloc_20, %subview_49 : memref<8x32xbf16, strided<[32, 1]>> to memref<8x32xbf16, strided<[32, 1], offset: ?>>
        memref.copy %18, %alloc_18 : memref<16x32xbf16> to memref<16x32xbf16, strided<[32, 1]>>
        memref.copy %15, %alloc_17 : memref<32x512xbf16> to memref<32x512xbf16, strided<[512, 1]>>
        hivm.hir.mmadL1 ins(%alloc_18, %alloc_17, %true, %c16, %c32, %c512 : memref<16x32xbf16, strided<[32, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index) outs(%alloc_21 : memref<16x512xf32, strided<[512, 1]>>)
        memref.copy %alloc_21, %19 : memref<16x512xf32, strided<[512, 1]>> to memref<16x512xf32>
        %subview_50 = memref.subview %19[%45, 0] [8, 512] [1, 1] : memref<16x512xf32> to memref<8x512xf32, strided<[512, 1], offset: ?>>
        memref.copy %subview_50, %alloc_28 : memref<8x512xf32, strided<[512, 1], offset: ?>> to memref<8x512xf32, strided<[512, 1]>>
        hivm.hir.vmul ins(%alloc_11, %alloc_26 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_11, %alloc_28 : memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>)
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
          memref.store %43, %alloc_9[%44, %c0] : memref<8x1xf32, strided<[1, 1]>>
        } else {
          %38 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_9[%38, %c0] : memref<8x1xf32, strided<[1, 1]>>
        }
      }
      hivm.hir.vsub ins(%alloc_9, %alloc_8 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vexp ins(%alloc_13 : memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vadd ins(%alloc_10, %alloc_14 : memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_10 : memref<8x1xf32, strided<[1, 1]>>)
      hivm.hir.vdiv ins(%alloc_11, %alloc_10 : memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_11 : memref<8x512xf32, strided<[512, 1]>>) outs(%alloc_12 : memref<8x512xbf16, strided<[512, 1]>>)
      %28 = arith.muli %14, %c8_i32 : i32
      %29 = arith.subi %c64_i32, %28 : i32
      %30 = arith.subi %29, %24 : i32
      %31 = arith.minsi %30, %c8_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview_15 = memref.subview %alloc_12[0, 0] [%32, 512] [1, 1] : memref<8x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[512, 1]>>
      %33 = arith.addi %24, %28 : i32
      %34 = arith.index_cast %33 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast_4[%21, %23, %34, 0] [1, 1, %32, 512] [1, 1, 1, 1] : memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_15, %subview_16 : memref<?x512xbf16, strided<[512, 1]>> to memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


loc("input.mlir":102:9): error: operand #1 does not dominate this use
// -----// IR Dump After TileLangIRCVSplit Failed (tilelangir-cv-split) ('func.func' operation: @sparseAttnMix) //----- //
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{hacc.arg_type = #hacc.arg_type<ffts_base_address>}, {}, {hacc.arg_type = #hacc.arg_type<workspace>}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}], function_type = (i64, memref<?xi8>, memref<?xi8>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xbf16, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xi32, #hivm.address_space<gm>>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (), sym_name = "sparseAttnMix"}> ({
  ^bb0(%arg0: i64, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32):
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
    %44 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<32x512xbf16>
    "annotation.mark"(%44) {hivm.multi_buffer = 2 : i32} : (memref<32x512xbf16>) -> ()
    %45 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<1x32xf32>
    "annotation.mark"(%45) {hivm.multi_buffer = 2 : i32} : (memref<1x32xf32>) -> ()
    %46 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<16x32xf32>
    "annotation.mark"(%46) {hivm.multi_buffer = 2 : i32} : (memref<16x32xf32>) -> ()
    %47 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<16x32xbf16>
    "annotation.mark"(%47) {hivm.multi_buffer = 2 : i32} : (memref<16x32xbf16>) -> ()
    %48 = "memref_ext.alloc_workspace"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<16x512xf32>
    "annotation.mark"(%48) {hivm.multi_buffer = 2 : i32} : (memref<16x512xf32>) -> ()
    "scf.for"(%20, %19, %23) ({
    ^bb0(%arg18: i32):
      %49 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x512xbf16, strided<[512, 1]>>
      %50 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>>
      %51 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>>
      %52 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>>
      %53 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x512xf32, strided<[512, 1]>>
      %54 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x512xbf16, strided<[512, 1]>>
      %55 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>>
      %56 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>>
      "hivm.hir.vbrc"(%4, %53) <{broadcast_dims = array<i64>}> : (f32, memref<8x512xf32, strided<[512, 1]>>) -> ()
      "hivm.hir.vbrc"(%4, %52) <{broadcast_dims = array<i64>}> : (f32, memref<8x1xf32, strided<[1, 1]>>) -> ()
      "hivm.hir.vbrc"(%18, %50) <{broadcast_dims = array<i64>}> : (f32, memref<8x1xf32, strided<[1, 1]>>) -> ()
      %57 = "arith.divsi"(%41, %arg9) : (i32, i32) -> i32
      %58 = "arith.index_cast"(%57) : (i32) -> index
      %59 = "arith.remsi"(%41, %arg9) : (i32, i32) -> i32
      %60 = "arith.index_cast"(%59) : (i32) -> index
      %61 = "arith.muli"(%arg18, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %62 = "arith.index_cast"(%61) : (i32) -> index
      %63 = "memref.subview"(%28, %58, %60, %62) <{operandSegmentSizes = array<i32: 1, 3, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 16, 512>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>, index, index, index) -> memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%63, %49) : (memref<16x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<16x512xbf16, strided<[512, 1]>>) -> ()
      %64 = "arith.addi"(%arg11, %16) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %65 = "arith.divsi"(%64, %15) : (i32, i32) -> i32
      "scf.for"(%20, %65, %23) ({
      ^bb0(%arg20: i32):
        %86 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32xi32, strided<[1]>>
        "hivm.hir.vbrc"(%4, %110) <{broadcast_dims = array<i64>}> : (f32, memref<1x32xf32, strided<[32, 1]>>) -> ()
        %87 = "arith.divsi"(%41, %arg9) : (i32, i32) -> i32
        %88 = "arith.index_cast"(%87) : (i32) -> index
        %89 = "arith.remsi"(%41, %arg9) : (i32, i32) -> i32
        %90 = "arith.index_cast"(%89) : (i32) -> index
        %91 = "arith.muli"(%arg20, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %92 = "arith.index_cast"(%91) : (i32) -> index
        %93 = "arith.subi"(%arg11, %91) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %94 = "arith.minsi"(%93, %15) : (i32, i32) -> i32
        %95 = "arith.index_cast"(%94) : (i32) -> index
        %96 = "memref.subview"(%34, %88, %90, %92, %95) <{operandSegmentSizes = array<i32: 1, 3, 1, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 1, 1, -9223372036854775808>, static_strides = array<i64: 1, 1, 1>}> : (memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>, index, index, index, index) -> memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %97 = "memref.subview"(%86, %95) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (memref<32xi32, strided<[1]>>, index) -> memref<?xi32, strided<[1]>>
        "memref.copy"(%96, %97) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[1]>>) -> ()
        %98 = "memref.subview"(%45) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 32>, static_strides = array<i64: 1, 1>}> : (memref<1x32xf32>) -> memref<32xf32, strided<[1]>>
        %99 = "arith.muli"(%43, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %100 = "arith.index_cast"(%99) : (i32) -> index
        %101 = "memref.subview"(%50) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 8, 1>, static_strides = array<i64: 1, 1>}> : (memref<8x1xf32, strided<[1, 1]>>) -> memref<8xf32, strided<[1]>>
        %102 = "memref.subview"(%51) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 8, 1>, static_strides = array<i64: 1, 1>}> : (memref<8x1xf32, strided<[1, 1]>>) -> memref<8xf32, strided<[1]>>
        "memref.copy"(%101, %102) : (memref<8xf32, strided<[1]>>, memref<8xf32, strided<[1]>>) -> ()
        "scope.scope"() ({
          %130 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x512xbf16, strided<[512, 1]>>
          "hivm.hir.vbrc"(%3, %130) <{broadcast_dims = array<i64>}> : (bf16, memref<32x512xbf16, strided<[512, 1]>>) -> ()
          "scf.for"(%20, %94, %23) ({
          ^bb0(%arg21: i32):
            %131 = "arith.index_cast"(%arg21) : (i32) -> index
            %132 = "memref.load"(%86, %131) : (memref<32xi32, strided<[1]>>, index) -> i32
            %133 = "arith.cmpi"(%132, %14) <{predicate = 1 : i64}> : (i32, i32) -> i1
            "scf.if"(%133) ({
              %134 = "arith.index_cast"(%arg21) : (i32) -> index
              "memref.store"(%13, %110, %2, %134) : (f32, memref<1x32xf32, strided<[32, 1]>>, index, index) -> ()
              %135 = "arith.divsi"(%41, %arg9) : (i32, i32) -> i32
              %136 = "arith.index_cast"(%135) : (i32) -> index
              %137 = "arith.index_cast"(%132) : (i32) -> index
              %138 = "memref.subview"(%38, %136, %137) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>}> : (memref<?x?x512xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>, index, index) -> memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %139 = "memref.subview"(%130, %134) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 1, 512>, static_strides = array<i64: 1, 1>}> : (memref<32x512xbf16, strided<[512, 1]>>, index) -> memref<512xbf16, strided<[1], offset: ?>>
              "memref.copy"(%138, %139) : (memref<512xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<512xbf16, strided<[1], offset: ?>>) -> ()
              "scf.yield"() : () -> ()
            }, {
            }) : (i1) -> ()
            "scf.yield"() : () -> ()
          }) : (i32, i32, i32) -> ()
          "memref.copy"(%130, %44) : (memref<32x512xbf16, strided<[512, 1]>>, memref<32x512xbf16>) -> ()
          "scope.return"() : () -> ()
        }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : () -> ()
        "scope.scope"() ({
          %105 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x512xbf16, strided<[512, 1]>>
          %106 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x32xbf16, strided<[32, 1]>>
          %107 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x32xf32, strided<[32, 1]>>
          %108 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x32xbf16, strided<[32, 1]>>
          %109 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x512xf32, strided<[512, 1]>>
          %110 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32xf32, strided<[32, 1]>>
          %111 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x32xf32, strided<[32, 1]>>
          %112 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>>
          %113 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>>
          %114 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>>
          %115 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>>
          %116 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x32xf32, strided<[32, 1]>>
          %117 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x32xf32, strided<[32, 1]>>
          %118 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32xf32, strided<[32, 1]>>
          %119 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x32xf32, strided<[32, 1]>>
          %120 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x1xf32, strided<[1, 1]>>
          %121 = "memref.subview"(%110) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 32>, static_strides = array<i64: 1, 1>}> : (memref<1x32xf32, strided<[32, 1]>>) -> memref<32xf32, strided<[1]>>
          "memref.copy"(%121, %98) : (memref<32xf32, strided<[1]>>, memref<32xf32, strided<[1]>>) -> ()
          "memref.copy"(%44, %105) : (memref<32x512xbf16>, memref<32x512xbf16, strided<[512, 1]>>) -> ()
          "hivm.hir.mmadL1"(%49, %105, %12, %1, %7, %0, %107) <{b_transpose, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> : (memref<16x512xbf16, strided<[512, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index, memref<16x32xf32, strided<[32, 1]>>) -> ()
          "memref.copy"(%107, %46) : (memref<16x32xf32, strided<[32, 1]>>, memref<16x32xf32>) -> ()
          %122 = "memref.subview"(%46, %100) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 8, 32>, static_strides = array<i64: 1, 1>}> : (memref<16x32xf32>, index) -> memref<8x32xf32, strided<[32, 1], offset: ?>>
          "memref.copy"(%122, %111) : (memref<8x32xf32, strided<[32, 1], offset: ?>>, memref<8x32xf32, strided<[32, 1]>>) -> ()
          %123 = "memref.subview"(%45) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 32>, static_strides = array<i64: 1, 1>}> : (memref<1x32xf32>) -> memref<32xf32, strided<[1]>>
          %124 = "memref.subview"(%110) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 32>, static_strides = array<i64: 1, 1>}> : (memref<1x32xf32, strided<[32, 1]>>) -> memref<32xf32, strided<[1]>>
          "memref.copy"(%123, %124) : (memref<32xf32, strided<[1]>>, memref<32xf32, strided<[1]>>) -> ()
          "hivm.hir.vmul"(%111, %10, %111) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x32xf32, strided<[32, 1]>>, f32, memref<8x32xf32, strided<[32, 1]>>) -> ()
          "hivm.hir.vreduce"(%111, %50) <{arith = #hivm.reduce_op<max>, operandSegmentSizes = array<i32: 1, 1, 0, 0>, reduce_dims = array<i64: 1>}> : (memref<8x32xf32, strided<[32, 1]>>, memref<8x1xf32, strided<[1, 1]>>) -> ()
          "hivm.hir.vsub"(%51, %50, %114) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) -> ()
          "hivm.hir.vexp"(%114, %112) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) -> ()
          %125 = "memref.subview"(%50) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 8, 1>, static_strides = array<i64: 1, 1>}> : (memref<8x1xf32, strided<[1, 1]>>) -> memref<8xf32, strided<[1]>>
          %126 = "memref.subview"(%115) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 8, 1>, static_strides = array<i64: 1, 1>}> : (memref<8x1xf32, strided<[1, 1]>>) -> memref<8xf32, strided<[1]>>
          "memref.copy"(%125, %126) : (memref<8xf32, strided<[1]>>, memref<8xf32, strided<[1]>>) -> ()
          "hivm.hir.vbrc"(%115, %116) <{broadcast_dims = array<i64: 1>}> : (memref<8x1xf32, strided<[1, 1]>>, memref<8x32xf32, strided<[32, 1]>>) -> ()
          "hivm.hir.vsub"(%111, %116, %117) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) -> ()
          "hivm.hir.vexp"(%117, %111) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) -> ()
          %127 = "memref.subview"(%110) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 32>, static_strides = array<i64: 1, 1>}> : (memref<1x32xf32, strided<[32, 1]>>) -> memref<32xf32, strided<[1]>>
          %128 = "memref.subview"(%118) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 32>, static_strides = array<i64: 1, 1>}> : (memref<1x32xf32, strided<[32, 1]>>) -> memref<32xf32, strided<[1]>>
          "memref.copy"(%127, %128) : (memref<32xf32, strided<[1]>>, memref<32xf32, strided<[1]>>) -> ()
          "hivm.hir.vbrc"(%118, %119) <{broadcast_dims = array<i64: 0>}> : (memref<1x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) -> ()
          "hivm.hir.vmul"(%111, %119, %111) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>, memref<8x32xf32, strided<[32, 1]>>) -> ()
          "hivm.hir.vreduce"(%111, %113) <{arith = #hivm.reduce_op<sum>, operandSegmentSizes = array<i32: 1, 1, 0, 0>, reduce_dims = array<i64: 1>}> : (memref<8x32xf32, strided<[32, 1]>>, memref<8x1xf32, strided<[1, 1]>>) -> ()
          "hivm.hir.vmul"(%52, %112, %120) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) -> ()
          "hivm.hir.vadd"(%120, %113, %52) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) -> ()
          "hivm.hir.vcast"(%111, %108) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<8x32xf32, strided<[32, 1]>>, memref<8x32xbf16, strided<[32, 1]>>) -> ()
          %129 = "memref.subview"(%47, %100) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 8, 32>, static_strides = array<i64: 1, 1>}> : (memref<16x32xbf16>, index) -> memref<8x32xbf16, strided<[32, 1], offset: ?>>
          "memref.copy"(%108, %129) : (memref<8x32xbf16, strided<[32, 1]>>, memref<8x32xbf16, strided<[32, 1], offset: ?>>) -> ()
          "memref.copy"(%47, %106) : (memref<16x32xbf16>, memref<16x32xbf16, strided<[32, 1]>>) -> ()
          "memref.copy"(%44, %105) : (memref<32x512xbf16>, memref<32x512xbf16, strided<[512, 1]>>) -> ()
          "hivm.hir.mmadL1"(%106, %105, %12, %1, %0, %7, %109) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> : (memref<16x32xbf16, strided<[32, 1]>>, memref<32x512xbf16, strided<[512, 1]>>, i1, index, index, index, memref<16x512xf32, strided<[512, 1]>>) -> ()
          "memref.copy"(%109, %48) : (memref<16x512xf32, strided<[512, 1]>>, memref<16x512xf32>) -> ()
          "hivm.hir.vmul"(%53, %112, %53) <{broadcast = array<i64: 1>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>, memref<8x512xf32, strided<[512, 1]>>) -> ()
          "scope.return"() : () -> ()
        }) {hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>} : () -> ()
        "scope.scope"() ({
          %103 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x512xf32, strided<[512, 1]>>
          %104 = "memref.subview"(%48, %100) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 8, 512>, static_strides = array<i64: 1, 1>}> : (memref<16x512xf32>, index) -> memref<8x512xf32, strided<[512, 1], offset: ?>>
          "memref.copy"(%104, %103) : (memref<8x512xf32, strided<[512, 1], offset: ?>>, memref<8x512xf32, strided<[512, 1]>>) -> ()
          "hivm.hir.vadd"(%53, %103, %53) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>, memref<8x512xf32, strided<[512, 1]>>) -> ()
          "scope.return"() : () -> ()
        }) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : () -> ()
        "scf.yield"() : () -> ()
      }) {tilelangir.num_stages = 2 : i32} : (i32, i32, i32) -> ()
      "scf.for"(%20, %11, %23) ({
      ^bb0(%arg19: i32):
        %75 = "arith.muli"(%arg18, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %76 = "arith.addi"(%75, %43) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %77 = "arith.cmpi"(%76, %11) <{predicate = 2 : i64}> : (i32, i32) -> i1
        "scf.if"(%77) ({
          %79 = "arith.muli"(%arg18, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %80 = "arith.muli"(%43, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %81 = "arith.addi"(%79, %80) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %82 = "arith.addi"(%81, %arg19) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %83 = "arith.index_cast"(%82) : (i32) -> index
          %84 = "memref.load"(%39, %83) : (memref<64xf32, strided<[1]>, #hivm.address_space<gm>>, index) -> f32
          %85 = "arith.index_cast"(%arg19) : (i32) -> index
          "memref.store"(%84, %51, %85, %2) : (f32, memref<8x1xf32, strided<[1, 1]>>, index, index) -> ()
          "scf.yield"() : () -> ()
        }, {
          %78 = "arith.index_cast"(%arg19) : (i32) -> index
          "memref.store"(%18, %51, %78, %2) : (f32, memref<8x1xf32, strided<[1, 1]>>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (i1) -> ()
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "hivm.hir.vsub"(%51, %50, %55) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) -> ()
      "hivm.hir.vexp"(%55, %56) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 1, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) -> ()
      "hivm.hir.vadd"(%52, %56, %52) <{broadcast = array<i64>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>, memref<8x1xf32, strided<[1, 1]>>) -> ()
      "hivm.hir.vdiv"(%53, %52, %53) <{broadcast = array<i64: 1>, operandSegmentSizes = array<i32: 2, 1, 0>, transpose = array<i64>}> : (memref<8x512xf32, strided<[512, 1]>>, memref<8x1xf32, strided<[1, 1]>>, memref<8x512xf32, strided<[512, 1]>>) -> ()
      "hivm.hir.vcast"(%53, %54) <{broadcast = array<i64>, cast = #hivm.cast<cast_signed>, operandSegmentSizes = array<i32: 1, 1, 0>, round_mode = #hivm.round_mode<rint>, transpose = array<i64>}> : (memref<8x512xf32, strided<[512, 1]>>, memref<8x512xbf16, strided<[512, 1]>>) -> ()
      %66 = "arith.muli"(%43, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %67 = "arith.subi"(%21, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %68 = "arith.subi"(%67, %61) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %69 = "arith.minsi"(%68, %11) : (i32, i32) -> i32
      %70 = "arith.index_cast"(%69) : (i32) -> index
      %71 = "memref.subview"(%54, %70) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, 512>, static_strides = array<i64: 1, 1>}> : (memref<8x512xbf16, strided<[512, 1]>>, index) -> memref<?x512xbf16, strided<[512, 1]>>
      %72 = "arith.addi"(%61, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %73 = "arith.index_cast"(%72) : (i32) -> index
      %74 = "memref.subview"(%29, %58, %60, %73, %70) <{operandSegmentSizes = array<i32: 1, 3, 1, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808, 0>, static_sizes = array<i64: 1, 1, -9223372036854775808, 512>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<?x?x64x512xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>, index, index, index, index) -> memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      "memref.copy"(%71, %74) : (memref<?x512xbf16, strided<[512, 1]>>, memref<?x512xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) -> ()
      "scf.yield"() : () -> ()
    }) : (i32, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} : () -> ()
}) {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} : () -> ()



Traceback (most recent call last):
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/examples/deepseek_v4/inference/sparse_attn_mix_open_final.py", line 406, in <module>
    run_test()
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/examples/deepseek_v4/inference/sparse_attn_mix_open_final.py", line 398, in run_test
    output = sparse_attn(**data["inputs"])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/z00910011/tilelang-ascend-test/tilelang-ascend/examples/deepseek_v4/inference/sparse_attn_mix_open_final.py", line 277, in sparse_attn
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
[ERROR] 2026-04-28-19:29:32 (PID:367979, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
