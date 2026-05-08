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