module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @sparseAttnMix_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c20480 = arith.constant 20480 : index
    return %c20480 : index
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
    %c16384 = arith.constant 16384 : index
    %c12288 = arith.constant 12288 : index
    %c4096 = arith.constant 4096 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 2.500000e-01 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c1024_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 16], strides: [%3, %c1024, %c16, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x16xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c16_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = hivm.hir.get_block_idx -> i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = affine.apply affine_map<(d0)[s0] -> (d0 * 20480 + s0)>(%16)[%c0]
    %view = memref.view %arg2[%17][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x16xbf16, #hivm.address_space<gm>>
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = affine.apply affine_map<(d0)[s0] -> (d0 * 20480 + s0)>(%19)[%c4096]
    %view_5 = memref.view %arg2[%20][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = affine.apply affine_map<(d0)[s0] -> (d0 * 20480 + s0)>(%22)[%c12288]
    %view_6 = memref.view %arg2[%23][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = affine.apply affine_map<(d0)[s0] -> (d0 * 20480 + s0)>(%25)[%c16384]
    %view_7 = memref.view %arg2[%26][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x16xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<16x16xbf16, strided<[16, 1]>, #hivm.address_space<cbuf>>
      %27 = arith.divsi %12, %arg9 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.remsi %12, %arg9 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.muli %arg18, %c16_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %subview = memref.subview %reinterpret_cast[%28, %30, %32, 0] [1, 1, 16, 16] [1, 1, 1, 1] : memref<?x?x64x16xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<16x16xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<16x16xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16x16xbf16, strided<[16, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      %33 = arith.addi %arg11, %c31_i32 : i32
      %34 = arith.divsi %33, %c32_i32 : i32
      %35 = arith.divsi %34, %c2_i32 : i32
      scf.for %arg19 = %c0_i32 to %35 step %c1_i32  : i32 {
        %alloc_8 = memref.alloc() : memref<32x16xbf16, strided<[16, 1]>, #hivm.address_space<cbuf>>
        %alloc_9 = memref.alloc() : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>
        %alloc_10 = memref.alloc() : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>
        %alloc_11 = memref.alloc() : memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<cc>>
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c2_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %45
          %48 = arith.index_cast %arg20 : i32 to index
          %subview_12 = memref.subview %view[%48, 0, 0, 0] [1, 1, 32, 16] [1, 1, 1, 1] : memref<2x2x32x16xbf16, #hivm.address_space<gm>> to memref<1x1x32x16xbf16, strided<[1024, 512, 16, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_12 [[0, 1, 2], [3]] : memref<1x1x32x16xbf16, strided<[1024, 512, 16, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x16xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<32x16xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_8 : memref<32x16xbf16, strided<[16, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_8, %true, %c16, %c16, %c32 : memref<16x16xbf16, strided<[16, 1]>, #hivm.address_space<cbuf>>, memref<32x16xbf16, strided<[16, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_10 : memref<16x32xf32, strided<[32, 1]>, #hivm.address_space<cc>>)
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
          hivm.hir.mmadL1 ins(%alloc_9, %alloc_8, %true, %c16, %c32, %c16 : memref<16x32xbf16, strided<[32, 1]>, #hivm.address_space<cbuf>>, memref<32x16xbf16, strided<[16, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_11 : memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<cc>>)
          %subview_13 = memref.subview %view_7[%50, 0, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf32, #hivm.address_space<gm>> to memref<1x1x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_14 = memref.collapse_shape %subview_13 [[0, 1, 2], [3]] : memref<1x1x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_11 : memref<16x16xf32, strided<[16, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_14 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %49 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
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
    %c16384 = arith.constant 16384 : index
    %c12288 = arith.constant 12288 : index
    %c4096 = arith.constant 4096 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 2.500000e-01 : f32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg8 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = arith.muli %arg9, %c1024_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 16], strides: [%3, %c1024, %c16, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x64x16xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>>
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.muli %arg9, %arg11 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xi32, #hivm.address_space<gm>> to memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.muli %arg10, %c16_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %8, 16], strides: [%10, %c16, %c1] : memref<?xbf16, #hivm.address_space<gm>> to memref<?x?x16xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64], strides: [%c1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = hivm.hir.get_sub_block_idx -> i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = hivm.hir.get_block_idx -> i64
    %16 = arith.index_cast %15 : i64 to index
    %17 = affine.apply affine_map<(d0)[s0] -> (d0 * 20480 + s0)>(%16)[%c0]
    %view = memref.view %arg2[%17][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x32x16xbf16, #hivm.address_space<gm>>
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = affine.apply affine_map<(d0)[s0] -> (d0 * 20480 + s0)>(%19)[%c4096]
    %view_7 = memref.view %arg2[%20][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xf32, #hivm.address_space<gm>>
    %21 = hivm.hir.get_block_idx -> i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = affine.apply affine_map<(d0)[s0] -> (d0 * 20480 + s0)>(%22)[%c12288]
    %view_8 = memref.view %arg2[%23][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x32xbf16, #hivm.address_space<gm>>
    %24 = hivm.hir.get_block_idx -> i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = affine.apply affine_map<(d0)[s0] -> (d0 * 20480 + s0)>(%25)[%c16384]
    %view_9 = memref.view %arg2[%26][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x16x16xf32, #hivm.address_space<gm>>
    scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<8x16xbf16, strided<[16, 1]>, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_15 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_12 : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
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
        %alloc_17 = memref.alloc() : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_18 = memref.alloc() : memref<32x16xbf16, strided<[16, 1]>, #hivm.address_space<ub>>
        %alloc_19 = memref.alloc() : memref<32xi32, strided<[1]>, #hivm.address_space<ub>>
        %alloc_20 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_21 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_22 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_23 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        %alloc_25 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_26 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %alloc_27 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_28 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_29 = memref.alloc() : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_30 = memref.alloc() : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
        %alloc_31 = memref.alloc() : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.muli %arg19, %c2_i32 : i32
          %47 = arith.addi %46, %arg20 : i32
          %48 = arith.index_cast %arg20 : i32 to index
          hivm.hir.vbrc ins(%cst : bf16) outs(%alloc_18 : memref<32x16xbf16, strided<[16, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_20 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
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
          %subview_32 = memref.subview %reinterpret_cast_4[%50, %52, %58] [1, 1, %56] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %subview_33 = memref.subview %alloc_19[0] [%56] [1] : memref<32xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_32, %subview_33 : memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          scf.for %arg21 = %c0_i32 to %55 step %c1_i32  : i32 {
            %59 = arith.index_cast %arg21 : i32 to index
            %60 = memref.load %alloc_19[%59] : memref<32xi32, strided<[1]>, #hivm.address_space<ub>>
            %61 = arith.cmpi ne, %60, %c-1_i32 : i32
            scf.if %61 {
              %62 = arith.index_cast %arg21 : i32 to index
              memref.store %cst_2, %alloc_20[%c0, %62] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
              %63 = arith.divsi %12, %arg9 : i32
              %64 = arith.index_cast %63 : i32 to index
              %65 = arith.index_cast %60 : i32 to index
              %subview_35 = memref.subview %reinterpret_cast_5[%64, %65, 0] [1, 1, 16] [1, 1, 1] : memref<?x?x16xbf16, strided<[?, ?, ?]>, #hivm.address_space<gm>> to memref<16xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
              %subview_36 = memref.subview %alloc_18[%62, 0] [1, 16] [1, 1] : memref<32x16xbf16, strided<[16, 1]>, #hivm.address_space<ub>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
              memref.copy %subview_35, %subview_36 : memref<16xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<16xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
            }
          }
          %subview_34 = memref.subview %view[%48, 0, 0, 0] [1, 1, 32, 16] [1, 1, 1, 1] : memref<2x2x32x16xbf16, #hivm.address_space<gm>> to memref<1x1x32x16xbf16, strided<[1024, 512, 16, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_34 [[0, 1, 2], [3]] : memref<1x1x32x16xbf16, strided<[1024, 512, 16, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x16xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_18, %collapse_shape : memref<32x16xbf16, strided<[16, 1]>, #hivm.address_space<ub>> to memref<32x16xbf16, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
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
          %subview_32 = memref.subview %view_7[%50, 0, %44, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xf32, #hivm.address_space<gm>> to memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_32 [[0, 1, 2], [3]] : memref<1x1x8x32xf32, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_21 : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
          %subview_33 = memref.subview %alloc[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_34 = memref.subview %alloc_10[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_33, %subview_34 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_21, %cst_1 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_21 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_21 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vsub ins(%alloc_10, %alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_25 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_25 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %subview_35 = memref.subview %alloc[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_36 = memref.subview %alloc_26[0, 0] [8, 1] [1, 1] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_35, %subview_36 : memref<8xf32, strided<[1]>, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_26 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]
          hivm.hir.vsub ins(%alloc_21, %alloc_27 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_28 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_37 = memref.subview %alloc_20[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_38 = memref.subview %alloc_29[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          memref.copy %subview_37, %subview_38 : memref<32xf32, strided<[1]>, #hivm.address_space<ub>> to memref<32xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%alloc_29 : memref<1x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_dims = [0]
          hivm.hir.vmul ins(%alloc_21, %alloc_30 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>, memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_21 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_23 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmul ins(%alloc_11, %alloc_22 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_31, %alloc_23 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_21 : memref<8x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>>)
          %subview_39 = memref.subview %view_8[%50, 0, %44, 0] [1, 1, 8, 32] [1, 1, 1, 1] : memref<2x2x16x32xbf16, #hivm.address_space<gm>> to memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_40 = memref.collapse_shape %subview_39 [[0, 1, 2], [3]] : memref<1x1x8x32xbf16, strided<[1024, 512, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_17, %collapse_shape_40 : memref<8x32xbf16, strided<[32, 1]>, #hivm.address_space<ub>> to memref<8x32xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %49 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %45 = arith.extsi %arg20 : i32 to i64
          %46 = arith.addi %45, %c4_i64 : i64
          %47 = arith.remsi %46, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %47
          %48 = arith.index_cast %arg20 : i32 to index
          %subview_32 = memref.subview %view_9[%48, 0, %44, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<2x2x16x16xf32, #hivm.address_space<gm>> to memref<1x1x8x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_32 [[0, 1, 2], [3]] : memref<1x1x8x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<gm>> into memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %collapse_shape, %alloc_24 : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_12, %alloc_22 : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_12, %alloc_24 : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>, memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>)
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
          %51 = memref.load %reinterpret_cast_6[%50] : memref<64xf32, strided<[1]>, #hivm.address_space<gm>>
          %52 = arith.index_cast %arg19 : i32 to index
          memref.store %51, %alloc_10[%52, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        } else {
          %46 = arith.index_cast %arg19 : i32 to index
          memref.store %cst_3, %alloc_10[%46, %c0] : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.vsub ins(%alloc_10, %alloc : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%alloc_14 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_11, %alloc_15 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_12, %alloc_11 : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_12 : memref<8x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<8x16xbf16, strided<[16, 1]>, #hivm.address_space<ub>>)
      %36 = arith.muli %14, %c8_i32 : i32
      %37 = arith.subi %c64_i32, %36 : i32
      %38 = arith.subi %37, %31 : i32
      %39 = arith.minsi %38, %c8_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %subview = memref.subview %alloc_13[0, 0] [%40, 16] [1, 1] : memref<8x16xbf16, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x16xbf16, strided<[16, 1]>, #hivm.address_space<ub>>
      %41 = arith.addi %31, %36 : i32
      %42 = arith.index_cast %41 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast[%28, %30, %42, 0] [1, 1, %40, 16] [1, 1, 1, 1] : memref<?x?x64x16xbf16, strided<[?, ?, ?, ?]>, #hivm.address_space<gm>> to memref<?x16xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %subview_16 : memref<?x16xbf16, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x16xbf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}
