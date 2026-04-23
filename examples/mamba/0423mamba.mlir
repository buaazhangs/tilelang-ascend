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

