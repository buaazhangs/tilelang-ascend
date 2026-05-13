// -----// IR Dump Before EnableStrideAlign (hivm-enable-stride-align) //----- //
func.func @sparseAttnMix_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %c1_i32 = arith.constant 1 : i32
  %c4_i64 = arith.constant 4 : i64
  %c2_i64 = arith.constant 2 : i64
  %c16_i64 = arith.constant 16 : i64
  %c57344 = arith.constant 57344 : index
  %c40960 = arith.constant 40960 : index
  %c8192 = arith.constant 8192 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c2048 = arith.constant 2048 : index
  %c2048_i32 = arith.constant 2048 : i32
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %cst_1 = arith.constant 0.176776692 : f32
  %cst_2 = arith.constant 1.000000e+00 : f32
  %c-1_i32 = arith.constant -1 : i32
  %c31_i32 = arith.constant 31 : i32
  %cst_3 = arith.constant 0xFF800000 : f32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c32_i32 = arith.constant 32 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.set_ffts_base_addr %arg0
  %2 = arith.index_cast %arg8 : i32 to index
  %3 = arith.index_cast %arg9 : i32 to index
  %4 = arith.muli %arg9, %c2048_i32 : i32
  %5 = arith.index_cast %4 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%2, %3, 64, 32], strides: [%5, %c2048, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x32xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
  %6 = arith.index_cast %arg11 : i32 to index
  %7 = arith.index_cast %arg11 : i32 to index
  %8 = arith.muli %arg9, %arg11 : i32
  %9 = arith.index_cast %8 : i32 to index
  %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%2, %3, %6], strides: [%9, %7, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
  %10 = arith.index_cast %arg10 : i32 to index
  %11 = arith.muli %arg10, %c32_i32 : i32
  %12 = arith.index_cast %11 : i32 to index
  %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%2, %10, 32], strides: [%12, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x32xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
  %13 = hivm.hir.get_block_idx -> i64
  %14 = arith.trunci %13 : i64 to i32
  %15 = hivm.hir.get_sub_block_idx -> i64
  %16 = arith.trunci %15 : i64 to i32
  %17 = hivm.hir.get_block_idx -> i64
  %18 = arith.index_cast %17 : i64 to index
  %19 = affine.apply affine_map<(d0)[s0] -> (d0 * 90112 + s0)>(%18)[%c0]
  %view = memref.view %arg2[%19][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x32xbf16, #hivm.address_space<gm>>
  %20 = hivm.hir.get_block_idx -> i64
  %21 = arith.index_cast %20 : i64 to index
  %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 90112 + s0)>(%21)[%c8192]
  %view_7 = memref.view %arg2[%22][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x64x32xf32, #hivm.address_space<gm>>
  %23 = hivm.hir.get_block_idx -> i64
  %24 = arith.index_cast %23 : i64 to index
  %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 90112 + s0)>(%24)[%c40960]
  %view_8 = memref.view %arg2[%25][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x64x32xbf16, #hivm.address_space<gm>>
  %26 = hivm.hir.get_block_idx -> i64
  %27 = arith.index_cast %26 : i64 to index
  %28 = affine.apply affine_map<(d0)[s0] -> (d0 * 90112 + s0)>(%27)[%c57344]
  %view_9 = memref.view %arg2[%28][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x64x32xf32, #hivm.address_space<gm>>
  %alloc = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
  %alloc_10 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
  %alloc_11 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
  %alloc_12 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
  %alloc_13 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
  %alloc_14 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
  %alloc_15 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_12 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
  hivm.hir.vbrc ins(%cst_3 : f32) outs(%alloc : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
  %29 = arith.divsi %14, %arg9 : i32
  %30 = arith.index_cast %29 : i32 to index
  %31 = arith.remsi %14, %arg9 : i32
  %32 = arith.index_cast %31 : i32 to index
  %33 = arith.addi %arg11, %c31_i32 : i32
  %34 = arith.divsi %33, %c32_i32 : i32
  %35 = arith.divsi %34, %c2_i32 : i32
  scf.for %arg15 = %c0_i32 to %35 step %c1_i32  : i32 {
    %alloc_17 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_18 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_19 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<ub>>
    %alloc_20 = memref.alloc() : memref<2x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>>
    %alloc_21 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_22 = memref.alloc() : memref<2x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>>
    %alloc_23 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %alloc_24 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_25 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %alloc_26 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %alloc_27 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_28 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_29 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_30 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_31 = memref.alloc() : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %43 = arith.extsi %arg16 : i32 to i64
      %44 = arith.muli %arg15, %c2_i32 : i32
      %45 = arith.addi %44, %arg16 : i32
      %46 = arith.index_cast %arg16 : i32 to index
      hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_18 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
      %47 = arith.index_cast %arg16 : i32 to index
      %subview_32 = memref.subview %alloc_20[%47, 0, 0] [1, 1, 32] [1, 1, 1] : memref<2x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>> to memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %subview_32 [[0, 1], [2]] : memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>> into memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape : memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>)
      %48 = arith.divsi %14, %arg9 : i32
      %49 = arith.index_cast %48 : i32 to index
      %50 = arith.remsi %14, %arg9 : i32
      %51 = arith.index_cast %50 : i32 to index
      %52 = arith.muli %45, %c32_i32 : i32
      %53 = arith.subi %arg11, %52 : i32
      %54 = arith.minsi %53, %c32_i32 : i32
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.muli %45, %c32_i32 : i32
      %57 = arith.index_cast %56 : i32 to index
      %subview_33 = memref.subview %reinterpret_cast_4[%49, %51, %57] [1, 1, %55] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %subview_34 = memref.subview %alloc_19[0] [%55] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      annotation.mark %subview_34 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_33 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_34 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      scf.for %arg17 = %c0_i32 to %54 step %c1_i32  : i32 {
        %58 = arith.index_cast %arg17 : i32 to index
        %59 = memref.load %alloc_19[%58] : memref<32xi32, strided<[1]>, #hivm.address_space<ub>>
        %60 = arith.cmpi ne, %59, %c-1_i32 : i32
        scf.if %60 {
          %61 = arith.index_cast %arg17 : i32 to index
          %62 = arith.index_cast %arg16 : i32 to index
          %subview_37 = memref.subview %alloc_20[%62, 0, 0] [1, 1, 32] [1, 1, 1] : memref<2x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>> to memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_38 = memref.collapse_shape %subview_37 [[0, 1], [2]] : memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>> into memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
          memref.store %cst_2, %collapse_shape_38[%c0, %61] : memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
          %63 = arith.divsi %14, %arg9 : i32
          %64 = arith.index_cast %63 : i32 to index
          %65 = arith.index_cast %59 : i32 to index
          %subview_39 = memref.subview %reinterpret_cast_5[%64, %65, 0] [1, 1, 32] [1, 1, 1] : memref<?x?x32xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<32xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_40 = memref.subview %alloc_18[%61, 0] [1, 32] [1, 1] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
          annotation.mark %subview_40 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<32xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.load ins(%subview_39 : memref<32xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_40 : memref<32xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        }
      }
      %subview_35 = memref.subview %view[%46, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x32x32xbf16, #hivm.address_space<gm>> to memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_36 = memref.collapse_shape %subview_35 [[0, 1, 2], [3]] : memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.store ins(%alloc_18 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_36 : memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %43 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    }
    %41 = arith.muli %16, %c32_i32 : i32
    %42 = arith.index_cast %41 : i32 to index
    scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %43 = arith.extsi %arg16 : i32 to i64
      %44 = arith.addi %43, %c2_i64 : i64
      %45 = arith.remsi %44, %c16_i64 : i64
      %46 = arith.addi %43, %c2_i64 : i64
      %47 = arith.remsi %46, %c16_i64 : i64
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %45
      %48 = arith.index_cast %arg16 : i32 to index
      %subview_32 = memref.subview %view_7[%48, 0, %42, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x64x32xf32, #hivm.address_space<gm>> to memref<1x1x32x32xf32, strided<[4096, 2048, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape = memref.collapse_shape %subview_32 [[0, 1, 2], [3]] : memref<1x1x32x32xf32, strided<[4096, 2048, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%collapse_shape : memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_21 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %subview_33 = memref.subview %alloc[0, 0] [32, 1] [1, 1] : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_34 = memref.subview %alloc_10[0, 0] [32, 1] [1, 1] : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.copy ins(%subview_33 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_34 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_21, %cst_1 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_21 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vreduce <max> ins(%alloc_21 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
      hivm.hir.vsub ins(%alloc_10, %alloc : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %49 = arith.index_cast %arg16 : i32 to index
      %subview_35 = memref.subview %alloc_22[%49, 0, 0] [1, 32, 1] [1, 1, 1] : memref<2x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
      %collapse_shape_36 = memref.collapse_shape %subview_35 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.vexp ins(%alloc_25 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_36 : memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
      %subview_37 = memref.subview %alloc[0, 0] [32, 1] [1, 1] : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_38 = memref.subview %alloc_26[0, 0] [32, 1] [1, 1] : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.copy ins(%subview_37 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_38 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%alloc_26 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
      hivm.hir.vsub ins(%alloc_21, %alloc_27 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_28 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
      %50 = arith.index_cast %arg16 : i32 to index
      %subview_39 = memref.subview %alloc_20[%50, 0, 0] [1, 1, 32] [1, 1, 1] : memref<2x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>> to memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>>
      %collapse_shape_40 = memref.collapse_shape %subview_39 [[0, 1], [2]] : memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>> into memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_41 = memref.subview %collapse_shape_40[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_42 = memref.subview %alloc_29[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.copy ins(%subview_41 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_42 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%alloc_29 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
      hivm.hir.vmul ins(%alloc_21, %alloc_30 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vreduce <sum> ins(%alloc_21 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_23 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
      %51 = arith.index_cast %arg16 : i32 to index
      %subview_43 = memref.subview %alloc_22[%51, 0, 0] [1, 32, 1] [1, 1, 1] : memref<2x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
      %collapse_shape_44 = memref.collapse_shape %subview_43 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%alloc_11, %collapse_shape_44 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_31, %alloc_23 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vcast ins(%alloc_21 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
      %subview_45 = memref.subview %view_8[%48, 0, %42, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x64x32xbf16, #hivm.address_space<gm>> to memref<1x1x32x32xbf16, strided<[4096, 2048, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_46 = memref.collapse_shape %subview_45 [[0, 1, 2], [3]] : memref<1x1x32x32xbf16, strided<[4096, 2048, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.store ins(%alloc_17 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_46 : memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %47 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    }
    scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %43 = arith.extsi %arg16 : i32 to i64
      %44 = arith.addi %43, %c4_i64 : i64
      %45 = arith.remsi %44, %c16_i64 : i64
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %45
      %46 = arith.index_cast %arg16 : i32 to index
      %subview_32 = memref.subview %view_9[%46, 0, %42, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x64x32xf32, #hivm.address_space<gm>> to memref<1x1x32x32xf32, strided<[4096, 2048, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape = memref.collapse_shape %subview_32 [[0, 1, 2], [3]] : memref<1x1x32x32xf32, strided<[4096, 2048, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%collapse_shape : memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_24 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %47 = arith.index_cast %arg16 : i32 to index
      %subview_33 = memref.subview %alloc_22[%47, 0, 0] [1, 32, 1] [1, 1, 1] : memref<2x32x1xf32, strided<[32, 1, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
      %collapse_shape_34 = memref.collapse_shape %subview_33 [[0, 1], [2]] : memref<1x32x1xf32, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%alloc_12, %collapse_shape_34 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vadd ins(%alloc_12, %alloc_24 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
    }
  } {tilelangir.num_stages = 2 : i32}
  scf.for %arg15 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
    %41 = arith.cmpi slt, %16, %c2_i32 : i32
    scf.if %41 {
      %42 = arith.muli %16, %c32_i32 : i32
      %43 = arith.addi %42, %arg15 : i32
      %44 = arith.index_cast %43 : i32 to index
      %45 = memref.load %reinterpret_cast_6[%44] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
      %46 = arith.index_cast %arg15 : i32 to index
      memref.store %45, %alloc_10[%46, %c0] : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    } else {
      %42 = arith.index_cast %arg15 : i32 to index
      memref.store %cst_3, %alloc_10[%42, %c0] : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    }
  }
  hivm.hir.vsub ins(%alloc_10, %alloc : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
  hivm.hir.vexp ins(%alloc_14 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
  hivm.hir.vadd ins(%alloc_11, %alloc_15 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
  hivm.hir.vdiv ins(%alloc_12, %alloc_11 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast = [1]
  hivm.hir.vcast ins(%alloc_12 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
  %36 = arith.muli %16, %c32_i32 : i32
  %37 = arith.subi %c64_i32, %36 : i32
  %38 = arith.minsi %37, %c32_i32 : i32
  %39 = arith.index_cast %38 : i32 to index
  %subview = memref.subview %alloc_13[0, 0] [%39, 32] [1, 1] : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<?x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
  annotation.mark %subview {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
  %40 = arith.index_cast %36 : i32 to index
  %subview_16 = memref.subview %reinterpret_cast[%30, %32, %40, 0] [1, 1, %39, 32] [1, 1, 1, 1] : memref<?x?x64x32xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x32xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview : memref<?x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%subview_16 : memref<?x32xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>)
  return
}

// -----// IR Dump After EnableStrideAlign (hivm-enable-stride-align) //----- //
func.func @sparseAttnMix_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>>, %arg4: memref<?xbf16, #hivm.address_space<gm>>, %arg5: memref<?xbf16, #hivm.address_space<gm>>, %arg6: memref<?xf32, #hivm.address_space<gm>>, %arg7: memref<?xi32, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix"} {
  %c1_i32 = arith.constant 1 : i32
  %c4_i64 = arith.constant 4 : i64
  %c2_i64 = arith.constant 2 : i64
  %c16_i64 = arith.constant 16 : i64
  %c57344 = arith.constant 57344 : index
  %c40960 = arith.constant 40960 : index
  %c8192 = arith.constant 8192 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c2048 = arith.constant 2048 : index
  %c2048_i32 = arith.constant 2048 : i32
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %cst_1 = arith.constant 0.176776692 : f32
  %cst_2 = arith.constant 1.000000e+00 : f32
  %c-1_i32 = arith.constant -1 : i32
  %c31_i32 = arith.constant 31 : i32
  %cst_3 = arith.constant 0xFF800000 : f32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c32_i32 = arith.constant 32 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.set_ffts_base_addr %arg0
  %2 = arith.index_cast %arg8 : i32 to index
  %3 = arith.index_cast %arg9 : i32 to index
  %4 = arith.muli %arg9, %c2048_i32 : i32
  %5 = arith.index_cast %4 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%2, %3, 64, 32], strides: [%5, %c2048, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x32xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
  %6 = arith.index_cast %arg11 : i32 to index
  %7 = arith.index_cast %arg11 : i32 to index
  %8 = arith.muli %arg9, %arg11 : i32
  %9 = arith.index_cast %8 : i32 to index
  %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%2, %3, %6], strides: [%9, %7, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
  %10 = arith.index_cast %arg10 : i32 to index
  %11 = arith.muli %arg10, %c32_i32 : i32
  %12 = arith.index_cast %11 : i32 to index
  %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%2, %10, 32], strides: [%12, %c32, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x32xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
  %13 = hivm.hir.get_block_idx -> i64
  %14 = arith.trunci %13 : i64 to i32
  %15 = hivm.hir.get_sub_block_idx -> i64
  %16 = arith.trunci %15 : i64 to i32
  %17 = hivm.hir.get_block_idx -> i64
  %18 = arith.index_cast %17 : i64 to index
  %19 = affine.apply affine_map<(d0)[s0] -> (d0 * 90112 + s0)>(%18)[%c0]
  %view = memref.view %arg2[%19][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x32xbf16, #hivm.address_space<gm>>
  %20 = hivm.hir.get_block_idx -> i64
  %21 = arith.index_cast %20 : i64 to index
  %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 90112 + s0)>(%21)[%c8192]
  %view_7 = memref.view %arg2[%22][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x64x32xf32, #hivm.address_space<gm>>
  %23 = hivm.hir.get_block_idx -> i64
  %24 = arith.index_cast %23 : i64 to index
  %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 90112 + s0)>(%24)[%c40960]
  %view_8 = memref.view %arg2[%25][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x64x32xbf16, #hivm.address_space<gm>>
  %26 = hivm.hir.get_block_idx -> i64
  %27 = arith.index_cast %26 : i64 to index
  %28 = affine.apply affine_map<(d0)[s0] -> (d0 * 90112 + s0)>(%27)[%c57344]
  %view_9 = memref.view %arg2[%28][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x64x32xf32, #hivm.address_space<gm>>
  %alloc = memref.alloc() : memref<32x8x1xf32, #hivm.address_space<ub>>
  %subview = memref.subview %alloc[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %alloc_10 = memref.alloc() : memref<32x8x1xf32, #hivm.address_space<ub>>
  %subview_11 = memref.subview %alloc_10[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %alloc_12 = memref.alloc() : memref<32x8x1xf32, #hivm.address_space<ub>>
  %subview_13 = memref.subview %alloc_12[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %alloc_14 = memref.alloc() : memref<32x32x8xf32, #hivm.address_space<ub>>
  %subview_15 = memref.subview %alloc_14[0, 0, 0] [32, 32, 1] [1, 1, 1] : memref<32x32x8xf32, #hivm.address_space<ub>> to memref<32x32xf32, strided<[256, 8]>, #hivm.address_space<ub>>
  %alloc_16 = memref.alloc() : memref<32x32x16xbf16, #hivm.address_space<ub>>
  %subview_17 = memref.subview %alloc_16[0, 0, 0] [32, 32, 1] [1, 1, 1] : memref<32x32x16xbf16, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[512, 16]>, #hivm.address_space<ub>>
  %alloc_18 = memref.alloc() : memref<32x8x1xf32, #hivm.address_space<ub>>
  %subview_19 = memref.subview %alloc_18[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %alloc_20 = memref.alloc() : memref<32x8x1xf32, #hivm.address_space<ub>>
  %subview_21 = memref.subview %alloc_20[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%subview_15 : memref<32x32xf32, strided<[256, 8]>, #hivm.address_space<ub>>)
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%subview_13 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  hivm.hir.vbrc ins(%cst_3 : f32) outs(%subview : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  %29 = arith.divsi %14, %arg9 : i32
  %30 = arith.index_cast %29 : i32 to index
  %31 = arith.remsi %14, %arg9 : i32
  %32 = arith.index_cast %31 : i32 to index
  %33 = arith.addi %arg11, %c31_i32 : i32
  %34 = arith.divsi %33, %c32_i32 : i32
  %35 = arith.divsi %34, %c2_i32 : i32
  scf.for %arg15 = %c0_i32 to %35 step %c1_i32  : i32 {
    %alloc_24 = memref.alloc() : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_25 = memref.alloc() : memref<32x32x16xbf16, #hivm.address_space<ub>>
    %subview_26 = memref.subview %alloc_25[0, 0, 0] [32, 32, 1] [1, 1, 1] : memref<32x32x16xbf16, #hivm.address_space<ub>> to memref<32x32xbf16, strided<[512, 16]>, #hivm.address_space<ub>>
    %alloc_27 = memref.alloc() : memref<32x8xi32, #hivm.address_space<ub>>
    %subview_28 = memref.subview %alloc_27[0, 0] [32, 1] [1, 1] : memref<32x8xi32, #hivm.address_space<ub>> to memref<32xi32, strided<[8]>, #hivm.address_space<ub>>
    %alloc_29 = memref.alloc() : memref<2x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>>
    %alloc_30 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_31 = memref.alloc() : memref<2x32x8x1xf32, #hivm.address_space<ub>>
    %subview_32 = memref.subview %alloc_31[0, 0, 0, 0] [2, 32, 1, 1] [1, 1, 1, 1] : memref<2x32x8x1xf32, #hivm.address_space<ub>> to memref<2x32x1xf32, strided<[256, 8, 1]>, #hivm.address_space<ub>>
    %alloc_33 = memref.alloc() : memref<32x8x1xf32, #hivm.address_space<ub>>
    %subview_34 = memref.subview %alloc_33[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %alloc_35 = memref.alloc() : memref<32x32x8xf32, #hivm.address_space<ub>>
    %subview_36 = memref.subview %alloc_35[0, 0, 0] [32, 32, 1] [1, 1, 1] : memref<32x32x8xf32, #hivm.address_space<ub>> to memref<32x32xf32, strided<[256, 8]>, #hivm.address_space<ub>>
    %alloc_37 = memref.alloc() : memref<32x8x1xf32, #hivm.address_space<ub>>
    %subview_38 = memref.subview %alloc_37[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %alloc_39 = memref.alloc() : memref<32x8x1xf32, #hivm.address_space<ub>>
    %subview_40 = memref.subview %alloc_39[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %alloc_41 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_42 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_43 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_44 = memref.alloc() : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %alloc_45 = memref.alloc() : memref<32x8x1xf32, #hivm.address_space<ub>>
    %subview_46 = memref.subview %alloc_45[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %43 = arith.extsi %arg16 : i32 to i64
      %44 = arith.muli %arg15, %c2_i32 : i32
      %45 = arith.addi %44, %arg16 : i32
      %46 = arith.index_cast %arg16 : i32 to index
      hivm.hir.vbrc ins(%cst : bf16) outs(%subview_26 : memref<32x32xbf16, strided<[512, 16]>, #hivm.address_space<ub>>)
      %47 = arith.index_cast %arg16 : i32 to index
      %subview_47 = memref.subview %alloc_29[%47, 0, 0] [1, 1, 32] [1, 1, 1] : memref<2x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>> to memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %subview_47 [[0, 1], [2]] : memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>> into memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape : memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>)
      %48 = arith.divsi %14, %arg9 : i32
      %49 = arith.index_cast %48 : i32 to index
      %50 = arith.remsi %14, %arg9 : i32
      %51 = arith.index_cast %50 : i32 to index
      %52 = arith.muli %45, %c32_i32 : i32
      %53 = arith.subi %arg11, %52 : i32
      %54 = arith.minsi %53, %c32_i32 : i32
      %55 = arith.index_cast %54 : i32 to index
      %56 = arith.muli %45, %c32_i32 : i32
      %57 = arith.index_cast %56 : i32 to index
      %subview_48 = memref.subview %reinterpret_cast_4[%49, %51, %57] [1, 1, %55] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %subview_49 = memref.subview %subview_28[0] [%55] [1] : memref<32xi32, strided<[8]>, #hivm.address_space<ub>> to memref<?xi32, strided<[8]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_48 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_49 : memref<?xi32, strided<[8]>, #hivm.address_space<ub>>) left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      scf.for %arg17 = %c0_i32 to %54 step %c1_i32  : i32 {
        %58 = arith.index_cast %arg17 : i32 to index
        %59 = memref.load %subview_28[%58] : memref<32xi32, strided<[8]>, #hivm.address_space<ub>>
        %60 = arith.cmpi ne, %59, %c-1_i32 : i32
        scf.if %60 {
          %61 = arith.index_cast %arg17 : i32 to index
          %62 = arith.index_cast %arg16 : i32 to index
          %subview_52 = memref.subview %alloc_29[%62, 0, 0] [1, 1, 32] [1, 1, 1] : memref<2x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>> to memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_53 = memref.collapse_shape %subview_52 [[0, 1], [2]] : memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>> into memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
          memref.store %cst_2, %collapse_shape_53[%c0, %61] : memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
          %63 = arith.divsi %14, %arg9 : i32
          %64 = arith.index_cast %63 : i32 to index
          %65 = arith.index_cast %59 : i32 to index
          %subview_54 = memref.subview %reinterpret_cast_5[%64, %65, 0] [1, 1, 32] [1, 1, 1] : memref<?x?x32xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<32xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_55 = memref.subview %subview_26[%61, 0] [1, 32] [1, 1] : memref<32x32xbf16, strided<[512, 16]>, #hivm.address_space<ub>> to memref<32xbf16, strided<[16], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.load ins(%subview_54 : memref<32xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_55 : memref<32xbf16, strided<[16], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        }
      }
      %subview_50 = memref.subview %view[%46, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x32x32xbf16, #hivm.address_space<gm>> to memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_51 = memref.collapse_shape %subview_50 [[0, 1, 2], [3]] : memref<1x1x32x32xbf16, strided<[2048, 1024, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.store ins(%subview_26 : memref<32x32xbf16, strided<[512, 16]>, #hivm.address_space<ub>>) outs(%collapse_shape_51 : memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %43 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    }
    %41 = arith.muli %16, %c32_i32 : i32
    %42 = arith.index_cast %41 : i32 to index
    scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %43 = arith.extsi %arg16 : i32 to i64
      %44 = arith.addi %43, %c2_i64 : i64
      %45 = arith.remsi %44, %c16_i64 : i64
      %46 = arith.addi %43, %c2_i64 : i64
      %47 = arith.remsi %46, %c16_i64 : i64
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %45
      %48 = arith.index_cast %arg16 : i32 to index
      %subview_47 = memref.subview %view_7[%48, 0, %42, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x64x32xf32, #hivm.address_space<gm>> to memref<1x1x32x32xf32, strided<[4096, 2048, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape = memref.collapse_shape %subview_47 [[0, 1, 2], [3]] : memref<1x1x32x32xf32, strided<[4096, 2048, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%collapse_shape : memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_30 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %subview_48 = memref.subview %subview[0, 0] [32, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
      %subview_49 = memref.subview %subview_11[0, 0] [32, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
      hivm.hir.copy ins(%subview_48 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%subview_49 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%alloc_30, %cst_1 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_30 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vreduce <max> ins(%alloc_30 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%subview : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
      hivm.hir.vsub ins(%subview_11, %subview : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_38 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      %49 = arith.index_cast %arg16 : i32 to index
      %subview_50 = memref.subview %subview_32[%49, 0, 0] [1, 32, 1] [1, 1, 1] : memref<2x32x1xf32, strided<[256, 8, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[256, 8, 1], offset: ?>, #hivm.address_space<ub>>
      %collapse_shape_51 = memref.collapse_shape %subview_50 [[0, 1], [2]] : memref<1x32x1xf32, strided<[256, 8, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.vexp ins(%subview_38 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_51 : memref<32x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>)
      %subview_52 = memref.subview %subview[0, 0] [32, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
      %subview_53 = memref.subview %subview_40[0, 0] [32, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
      hivm.hir.copy ins(%subview_52 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%subview_53 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%subview_40 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%alloc_41 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
      hivm.hir.vsub ins(%alloc_30, %alloc_41 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_42 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_42 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
      %50 = arith.index_cast %arg16 : i32 to index
      %subview_54 = memref.subview %alloc_29[%50, 0, 0] [1, 1, 32] [1, 1, 1] : memref<2x1x32xf32, strided<[32, 32, 1]>, #hivm.address_space<ub>> to memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>>
      %collapse_shape_55 = memref.collapse_shape %subview_54 [[0, 1], [2]] : memref<1x1x32xf32, strided<[32, 32, 1], offset: ?>, #hivm.address_space<ub>> into memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_56 = memref.subview %collapse_shape_55[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_57 = memref.subview %alloc_43[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.copy ins(%subview_56 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_57 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%alloc_43 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_44 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
      hivm.hir.vmul ins(%alloc_30, %alloc_44 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vreduce <sum> ins(%alloc_30 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%subview_34 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
      %51 = arith.index_cast %arg16 : i32 to index
      %subview_58 = memref.subview %subview_32[%51, 0, 0] [1, 32, 1] [1, 1, 1] : memref<2x32x1xf32, strided<[256, 8, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[256, 8, 1], offset: ?>, #hivm.address_space<ub>>
      %collapse_shape_59 = memref.collapse_shape %subview_58 [[0, 1], [2]] : memref<1x32x1xf32, strided<[256, 8, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%subview_13, %collapse_shape_59 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_46 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%subview_46, %subview_34 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_13 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vcast ins(%alloc_30 : memref<32x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_24 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
      %subview_60 = memref.subview %view_8[%48, 0, %42, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x64x32xbf16, #hivm.address_space<gm>> to memref<1x1x32x32xbf16, strided<[4096, 2048, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_61 = memref.collapse_shape %subview_60 [[0, 1, 2], [3]] : memref<1x1x32x32xbf16, strided<[4096, 2048, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.store ins(%alloc_24 : memref<32x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_61 : memref<32x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %47 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
    }
    scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %43 = arith.extsi %arg16 : i32 to i64
      %44 = arith.addi %43, %c4_i64 : i64
      %45 = arith.remsi %44, %c16_i64 : i64
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %45
      %46 = arith.index_cast %arg16 : i32 to index
      %subview_47 = memref.subview %view_9[%46, 0, %42, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x64x32xf32, #hivm.address_space<gm>> to memref<1x1x32x32xf32, strided<[4096, 2048, 32, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape = memref.collapse_shape %subview_47 [[0, 1, 2], [3]] : memref<1x1x32x32xf32, strided<[4096, 2048, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%collapse_shape : memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_36 : memref<32x32xf32, strided<[256, 8]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %47 = arith.index_cast %arg16 : i32 to index
      %subview_48 = memref.subview %subview_32[%47, 0, 0] [1, 32, 1] [1, 1, 1] : memref<2x32x1xf32, strided<[256, 8, 1]>, #hivm.address_space<ub>> to memref<1x32x1xf32, strided<[256, 8, 1], offset: ?>, #hivm.address_space<ub>>
      %collapse_shape_49 = memref.collapse_shape %subview_48 [[0, 1], [2]] : memref<1x32x1xf32, strided<[256, 8, 1], offset: ?>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%subview_15, %collapse_shape_49 : memref<32x32xf32, strided<[256, 8]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_15 : memref<32x32xf32, strided<[256, 8]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vadd ins(%subview_15, %subview_36 : memref<32x32xf32, strided<[256, 8]>, #hivm.address_space<ub>>, memref<32x32xf32, strided<[256, 8]>, #hivm.address_space<ub>>) outs(%subview_15 : memref<32x32xf32, strided<[256, 8]>, #hivm.address_space<ub>>)
    }
  } {tilelangir.num_stages = 2 : i32}
  scf.for %arg15 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
    %41 = arith.cmpi slt, %16, %c2_i32 : i32
    scf.if %41 {
      %42 = arith.muli %16, %c32_i32 : i32
      %43 = arith.addi %42, %arg15 : i32
      %44 = arith.index_cast %43 : i32 to index
      %45 = memref.load %reinterpret_cast_6[%44] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
      %46 = arith.index_cast %arg15 : i32 to index
      memref.store %45, %subview_11[%46, %c0] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    } else {
      %42 = arith.index_cast %arg15 : i32 to index
      memref.store %cst_3, %subview_11[%42, %c0] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    }
  }
  hivm.hir.vsub ins(%subview_11, %subview : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_19 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  hivm.hir.vexp ins(%subview_19 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_21 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  hivm.hir.vadd ins(%subview_13, %subview_21 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_13 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  hivm.hir.vdiv ins(%subview_15, %subview_13 : memref<32x32xf32, strided<[256, 8]>, #hivm.address_space<ub>>, memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%subview_15 : memref<32x32xf32, strided<[256, 8]>, #hivm.address_space<ub>>) broadcast = [1]
  hivm.hir.vcast ins(%subview_15 : memref<32x32xf32, strided<[256, 8]>, #hivm.address_space<ub>>) outs(%subview_17 : memref<32x32xbf16, strided<[512, 16]>, #hivm.address_space<ub>>)
  %36 = arith.muli %16, %c32_i32 : i32
  %37 = arith.subi %c64_i32, %36 : i32
  %38 = arith.minsi %37, %c32_i32 : i32
  %39 = arith.index_cast %38 : i32 to index
  %subview_22 = memref.subview %subview_17[0, 0] [%39, 32] [1, 1] : memref<32x32xbf16, strided<[512, 16]>, #hivm.address_space<ub>> to memref<?x32xbf16, strided<[512, 16]>, #hivm.address_space<ub>>
  %40 = arith.index_cast %36 : i32 to index
  %subview_23 = memref.subview %reinterpret_cast[%30, %32, %40, 0] [1, 1, %39, 32] [1, 1, 1, 1] : memref<?x?x64x32xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x32xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_22 : memref<?x32xbf16, strided<[512, 16]>, #hivm.address_space<ub>>) outs(%subview_23 : memref<?x32xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>)
  return
}