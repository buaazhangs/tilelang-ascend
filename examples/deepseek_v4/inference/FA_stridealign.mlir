// -----// IR Dump Before MarkStrideAlign (hivm-mark-stride-align) //----- //
func.func @flash_attention_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c8_i64 = arith.constant 8 : i64
  %c16_i64 = arith.constant 16 : i64
  %c8_i32 = arith.constant 8 : i32
  %c3145728 = arith.constant 3145728 : index
  %c2097152 = arith.constant 2097152 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c8192_i32 = arith.constant 8192 : i32
  %cst_0 = arith.constant 0.0883883461 : f32
  %c64_i32 = arith.constant 64 : i32
  %cst_1 = arith.constant 0xFF800000 : f32
  %c3072_i32 = arith.constant 3072 : i32
  %c24_i32 = arith.constant 24 : i32
  %c87_i32 = arith.constant 87 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.set_ffts_base_addr %arg0
  %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = hivm.hir.get_sub_block_idx -> i64
  %5 = arith.trunci %4 : i64 to i32
  %6 = hivm.hir.get_block_idx -> i64
  %7 = arith.index_cast %6 : i64 to index
  %8 = affine.apply affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>(%7)[%c0]
  %view = memref.view %arg2[%8][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
  %9 = hivm.hir.get_block_idx -> i64
  %10 = arith.index_cast %9 : i64 to index
  %11 = affine.apply affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>(%10)[%c2097152]
  %view_2 = memref.view %arg2[%11][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
  %12 = hivm.hir.get_block_idx -> i64
  %13 = arith.index_cast %12 : i64 to index
  %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>(%13)[%c3145728]
  %view_3 = memref.view %arg2[%14][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
  %15 = arith.subi %c87_i32, %3 : i32
  %16 = arith.divsi %15, %c24_i32 : i32
  scf.for %arg10 = %c0_i32 to %16 step %c1_i32  : i32 {
    %alloc = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %alloc_4 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %alloc_5 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %alloc_6 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %17 = arith.muli %arg10, %c3072_i32 : i32
    %18 = arith.muli %3, %c128_i32 : i32
    %19 = arith.addi %17, %18 : i32
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
    hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
    scf.for %arg11 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %28 = arith.extsi %arg11 : i32 to i64
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %28 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    }
    scf.for %arg11 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc_8 = memref.alloc() : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
      %28 = arith.muli %5, %c64_i32 : i32
      %29 = arith.index_cast %28 : i32 to index
      scf.for %arg12 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %30 = arith.extsi %arg12 : i32 to i64
        %31 = arith.addi %30, %c8_i64 : i64
        %32 = arith.remsi %31, %c16_i64 : i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
        %33 = arith.index_cast %arg12 : i32 to index
        %subview_9 = memref.subview %view[%33, 0, %29, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_10 = memref.subview %view_2[%33, 0, %29, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_11 = memref.collapse_shape %subview_10 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_12 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        %alloc_13 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
        %alloc_14 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_15 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_16 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_17 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        hivm.hir.load ins(%collapse_shape : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.vmul ins(%alloc_12, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vreduce <max> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
        hivm.hir.vmax ins(%alloc, %alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vsub ins(%alloc_12, %alloc_16 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
        hivm.hir.vexp ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vreduce <sum> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
        hivm.hir.store ins(%alloc_13 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_11 : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.vsub ins(%alloc, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        %34 = arith.index_cast %arg12 : i32 to index
        %subview_18 = memref.subview %alloc_8[%34, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
        %collapse_shape_19 = memref.collapse_shape %subview_18 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
        hivm.hir.vexp ins(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
        %35 = arith.index_cast %arg12 : i32 to index
        %subview_20 = memref.subview %alloc_8[%35, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
        %collapse_shape_21 = memref.collapse_shape %subview_20 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
        hivm.hir.vmul ins(%alloc_4, %collapse_shape_21 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vadd ins(%alloc_4, %alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vbrc ins(%cst : f32) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vadd ins(%alloc_17, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg12 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %30 = arith.extsi %arg12 : i32 to i64
        %31 = arith.addi %30, %c8_i64 : i64
        %32 = arith.remsi %31, %c16_i64 : i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %32
        %33 = arith.index_cast %arg12 : i32 to index
        %subview_9 = memref.subview %view_3[%33, 0, %29, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_10 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
        hivm.hir.load ins(%collapse_shape : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_10 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %34 = arith.index_cast %arg12 : i32 to index
        %subview_11 = memref.subview %alloc_8[%34, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
        %collapse_shape_12 = memref.collapse_shape %subview_11 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
        hivm.hir.vmul ins(%alloc_5, %collapse_shape_12 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_5, %alloc_10 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %30 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
    } {tilelangir.num_stages = 8 : i32}
    hivm.hir.vdiv ins(%alloc_5, %alloc_4 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.vcast ins(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
    %20 = arith.muli %5, %c64_i32 : i32
    %21 = arith.subi %c8192_i32, %20 : i32
    %22 = arith.subi %21, %18 : i32
    %23 = arith.subi %22, %17 : i32
    %24 = arith.minsi %23, %c64_i32 : i32
    %25 = arith.index_cast %24 : i32 to index
    %subview = memref.subview %alloc_6[0, 0] [%25, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %26 = arith.addi %19, %20 : i32
    %27 = arith.index_cast %26 : i32 to index
    %subview_7 = memref.subview %reinterpret_cast[%27, 0] [%25, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
  }
  return
}

// -----// IR Dump After MarkStrideAlign (hivm-mark-stride-align) //----- //
func.func @flash_attention_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c8_i64 = arith.constant 8 : i64
  %c16_i64 = arith.constant 16 : i64
  %c8_i32 = arith.constant 8 : i32
  %c3145728 = arith.constant 3145728 : index
  %c2097152 = arith.constant 2097152 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c8192_i32 = arith.constant 8192 : i32
  %cst_0 = arith.constant 0.0883883461 : f32
  %c64_i32 = arith.constant 64 : i32
  %cst_1 = arith.constant 0xFF800000 : f32
  %c3072_i32 = arith.constant 3072 : i32
  %c24_i32 = arith.constant 24 : i32
  %c87_i32 = arith.constant 87 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.set_ffts_base_addr %arg0
  %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = hivm.hir.get_sub_block_idx -> i64
  %5 = arith.trunci %4 : i64 to i32
  %6 = hivm.hir.get_block_idx -> i64
  %7 = arith.index_cast %6 : i64 to index
  %8 = affine.apply affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>(%7)[%c0]
  %view = memref.view %arg2[%8][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
  %9 = hivm.hir.get_block_idx -> i64
  %10 = arith.index_cast %9 : i64 to index
  %11 = affine.apply affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>(%10)[%c2097152]
  %view_2 = memref.view %arg2[%11][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
  %12 = hivm.hir.get_block_idx -> i64
  %13 = arith.index_cast %12 : i64 to index
  %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>(%13)[%c3145728]
  %view_3 = memref.view %arg2[%14][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
  %15 = arith.subi %c87_i32, %3 : i32
  %16 = arith.divsi %15, %c24_i32 : i32
  scf.for %arg10 = %c0_i32 to %16 step %c1_i32  : i32 {
    %alloc = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %alloc_4 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %alloc_5 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %alloc_6 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %17 = arith.muli %arg10, %c3072_i32 : i32
    %18 = arith.muli %3, %c128_i32 : i32
    %19 = arith.addi %17, %18 : i32
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
    hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
    scf.for %arg11 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %28 = arith.extsi %arg11 : i32 to i64
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %28 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    }
    scf.for %arg11 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc_8 = memref.alloc() : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
      %28 = arith.muli %5, %c64_i32 : i32
      %29 = arith.index_cast %28 : i32 to index
      scf.for %arg12 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %30 = arith.extsi %arg12 : i32 to i64
        %31 = arith.addi %30, %c8_i64 : i64
        %32 = arith.remsi %31, %c16_i64 : i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %30
        %33 = arith.index_cast %arg12 : i32 to index
        %subview_9 = memref.subview %view[%33, 0, %29, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_10 = memref.subview %view_2[%33, 0, %29, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_11 = memref.collapse_shape %subview_10 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_12 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        %alloc_13 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
        %alloc_14 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_15 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_16 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_17 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        hivm.hir.load ins(%collapse_shape : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.vmul ins(%alloc_12, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vreduce <max> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
        hivm.hir.vmax ins(%alloc, %alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vsub ins(%alloc_12, %alloc_16 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
        hivm.hir.vexp ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vcast ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vreduce <sum> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
        hivm.hir.store ins(%alloc_13 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_11 : memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.vsub ins(%alloc, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        %34 = arith.index_cast %arg12 : i32 to index
        %subview_18 = memref.subview %alloc_8[%34, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
        %collapse_shape_19 = memref.collapse_shape %subview_18 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
        hivm.hir.vexp ins(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
        %35 = arith.index_cast %arg12 : i32 to index
        %subview_20 = memref.subview %alloc_8[%35, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
        %collapse_shape_21 = memref.collapse_shape %subview_20 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
        hivm.hir.vmul ins(%alloc_4, %collapse_shape_21 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vadd ins(%alloc_4, %alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vbrc ins(%cst : f32) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        hivm.hir.vadd ins(%alloc_17, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %32 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg12 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %30 = arith.extsi %arg12 : i32 to i64
        %31 = arith.addi %30, %c8_i64 : i64
        %32 = arith.remsi %31, %c16_i64 : i64
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %32
        %33 = arith.index_cast %arg12 : i32 to index
        %subview_9 = memref.subview %view_3[%33, 0, %29, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_10 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
        hivm.hir.load ins(%collapse_shape : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_10 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %34 = arith.index_cast %arg12 : i32 to index
        %subview_11 = memref.subview %alloc_8[%34, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
        %collapse_shape_12 = memref.collapse_shape %subview_11 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
        hivm.hir.vmul ins(%alloc_5, %collapse_shape_12 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_5, %alloc_10 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %30 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
    } {tilelangir.num_stages = 8 : i32}
    hivm.hir.vdiv ins(%alloc_5, %alloc_4 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.vcast ins(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
    %20 = arith.muli %5, %c64_i32 : i32
    %21 = arith.subi %c8192_i32, %20 : i32
    %22 = arith.subi %21, %18 : i32
    %23 = arith.subi %22, %17 : i32
    %24 = arith.minsi %23, %c64_i32 : i32
    %25 = arith.index_cast %24 : i32 to index
    %subview = memref.subview %alloc_6[0, 0] [%25, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %26 = arith.addi %19, %20 : i32
    %27 = arith.index_cast %26 : i32 to index
    %subview_7 = memref.subview %reinterpret_cast[%27, 0] [%25, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
  }
  return
}
