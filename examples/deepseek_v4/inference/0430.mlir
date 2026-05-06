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