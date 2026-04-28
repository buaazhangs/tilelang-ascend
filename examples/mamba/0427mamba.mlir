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