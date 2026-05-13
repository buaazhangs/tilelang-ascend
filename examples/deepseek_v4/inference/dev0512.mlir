module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @sparseAttn(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16>, %arg4: memref<?xbf16>, %arg5: memref<?xbf16>, %arg6: memref<?xf32>, %arg7: memref<?xi32>, %arg8: memref<?xbf16>, %arg9: memref<?xf32>, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.part_of_mix, mix_mode = "mix"} {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c2048 = arith.constant 2048 : index
    %c2048_i32 = arith.constant 2048 : i32
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %cst_1 = arith.constant 0.176776692 : f32
    %true = arith.constant true
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c31_i32 = arith.constant 31 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.index_cast %arg10 : i32 to index
    %1 = arith.index_cast %arg11 : i32 to index
    %2 = arith.muli %arg11, %c2048_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1, 64, 32], strides: [%3, %c2048, %c32, %c1] : memref<?xbf16> to memref<?x?x64x32xbf16, strided<[?, ?, ?, ?]>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1, 64, 32], strides: [%3, %c2048, %c32, %c1] : memref<?xbf16> to memref<?x?x64x32xbf16, strided<[?, ?, ?, ?]>>
    %4 = arith.index_cast %arg14 : i32 to index
    %5 = arith.index_cast %arg14 : i32 to index
    %6 = arith.muli %arg11, %arg14 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [%0, %1, %4], strides: [%7, %5, %c1] : memref<?xf32> to memref<?x?x?xf32, strided<[?, ?, ?]>>
    %8 = arith.index_cast %arg13 : i32 to index
    %9 = arith.index_cast %arg13 : i32 to index
    %10 = arith.muli %arg11, %arg13 : i32
    %11 = arith.index_cast %10 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [%0, %1, %8], strides: [%11, %9, %c1] : memref<?xi32> to memref<?x?x?xi32, strided<[?, ?, ?]>>
    %12 = arith.muli %arg14, %c32_i32 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = arith.muli %arg11, %12 : i32
    %15 = arith.index_cast %14 : i32 to index
    %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [%0, %1, %4, 32], strides: [%15, %13, %c32, %c1] : memref<?xbf16> to memref<?x?x?x32xbf16, strided<[?, ?, ?, ?]>>
    %16 = arith.index_cast %arg12 : i32 to index
    %17 = arith.muli %arg12, %c32_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %16, 32], strides: [%18, %c32, %c1] : memref<?xbf16> to memref<?x?x32xbf16, strided<[?, ?, ?]>>
    %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [64, 1], strides: [%c1, %c1] : memref<?xf32> to memref<64x1xf32, strided<[1, 1]>>
    %19 = hivm.hir.get_block_idx -> i64
    %20 = arith.trunci %19 : i64 to i32
    scf.for %arg21 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %21 = tensor.empty() : tensor<16x1xf32>
      %22 = tensor.empty() : tensor<16x32xf32>
      %23 = tensor.empty() : tensor<16x1xf32>
      %24 = tensor.empty() : tensor<16x1xf32>
      %25 = tensor.empty() : tensor<16x1xf32>
      %26 = tensor.empty() : tensor<16x1xf32>
      %27 = tensor.empty() : tensor<16x32xf32>
      %28 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%22 : tensor<16x32xf32>) -> tensor<16x32xf32>
      %29 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%24 : tensor<16x1xf32>) -> tensor<16x1xf32>
      %30 = hivm.hir.vbrc ins(%cst_3 : f32) outs(%23 : tensor<16x1xf32>) -> tensor<16x1xf32>
      %31 = arith.divsi %20, %arg11 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = arith.remsi %20, %arg11 : i32
      %34 = arith.index_cast %33 : i32 to index
      %35 = arith.muli %arg21, %c16_i32 : i32
      %36 = arith.index_cast %35 : i32 to index
      %subview = memref.subview %reinterpret_cast[%32, %34, %36, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<?x?x64x32xbf16, strided<[?, ?, ?, ?]>> to memref<16x32xbf16, strided<[?, ?], offset: ?>>
      %alloc = memref.alloc() : memref<16x32xbf16>
      memref.copy %subview, %alloc : memref<16x32xbf16, strided<[?, ?], offset: ?>> to memref<16x32xbf16>
      %37 = bufferization.to_tensor %alloc restrict : memref<16x32xbf16>
      %38 = arith.addi %arg13, %c31_i32 : i32
      %39 = arith.divsi %38, %c32_i32 : i32
      %40:3 = scf.for %arg22 = %c0_i32 to %39 step %c1_i32 iter_args(%arg23 = %30, %arg24 = %29, %arg25 = %28) -> (tensor<16x1xf32>, tensor<16x1xf32>, tensor<16x32xf32>)  : i32 {
        %49 = tensor.empty() : tensor<32x32xbf16>
        %50 = tensor.empty() : tensor<16x32xf32>
        %51 = tensor.empty() : tensor<16x1xf32>
        %52 = tensor.empty() : tensor<16x1xf32>
        %53 = tensor.empty() : tensor<16x1xf32>
        %54 = tensor.empty() : tensor<1x32xf32>
        %55 = tensor.empty() : tensor<16x1xf32>
        %56 = tensor.empty() : tensor<16x32xf32>
        %57 = tensor.empty() : tensor<16x32xf32>
        %58 = tensor.empty() : tensor<16x32xf32>
        %59 = tensor.empty() : tensor<16x1xf32>
        %60 = tensor.empty() : tensor<16x32xf32>
        %61 = arith.cmpi eq, %arg21, %c0_i32 : i32
        %62 = scf.if %61 -> (tensor<1x32xf32>) {
          %90 = arith.divsi %20, %arg11 : i32
          %91 = arith.index_cast %90 : i32 to index
          %92 = arith.remsi %20, %arg11 : i32
          %93 = arith.index_cast %92 : i32 to index
          %94 = arith.muli %arg22, %c32_i32 : i32
          %95 = arith.index_cast %94 : i32 to index
          %subview_18 = memref.subview %reinterpret_cast_6[%91, %93, %95] [1, 1, 32] [1, 1, 1] : memref<?x?x?xi32, strided<[?, ?, ?]>> to memref<32xi32, strided<[?], offset: ?>>
          %alloc_19 = memref.alloc() : memref<32xi32>
          memref.copy %subview_18, %alloc_19 : memref<32xi32, strided<[?], offset: ?>> to memref<32xi32>
          %96 = bufferization.to_tensor %alloc_19 restrict : memref<32xi32>
          %97 = hivm.hir.vbrc ins(%cst : bf16) outs(%49 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
          %98 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%54 : tensor<1x32xf32>) -> tensor<1x32xf32>
          %99 = arith.subi %arg13, %94 : i32
          %100 = arith.minsi %99, %c32_i32 : i32
          %101:2 = scf.for %arg26 = %c0_i32 to %100 step %c1_i32 iter_args(%arg27 = %98, %arg28 = %97) -> (tensor<1x32xf32>, tensor<32x32xbf16>)  : i32 {
            %102 = arith.index_cast %arg26 : i32 to index
            %extracted = tensor.extract %96[%102] : tensor<32xi32>
            %103 = arith.cmpi ne, %extracted, %c-1_i32 : i32
            %104:2 = scf.if %103 -> (tensor<1x32xf32>, tensor<32x32xbf16>) {
              %105 = arith.index_cast %arg26 : i32 to index
              %inserted = tensor.insert %cst_2 into %arg27[%c0, %105] : tensor<1x32xf32>
              %106 = arith.divsi %20, %arg11 : i32
              %107 = arith.index_cast %106 : i32 to index
              %108 = arith.index_cast %extracted : i32 to index
              %subview_22 = memref.subview %reinterpret_cast_8[%107, %108, 0] [1, 1, 32] [1, 1, 1] : memref<?x?x32xbf16, strided<[?, ?, ?]>> to memref<32xbf16, strided<[?], offset: ?>>
              %alloc_23 = memref.alloc() : memref<32xbf16>
              memref.copy %subview_22, %alloc_23 : memref<32xbf16, strided<[?], offset: ?>> to memref<32xbf16>
              %109 = bufferization.to_tensor %alloc_23 restrict : memref<32xbf16>
              %inserted_slice_24 = tensor.insert_slice %109 into %arg28[%105, 0] [1, 32] [1, 1] : tensor<32xbf16> into tensor<32x32xbf16>
              scf.yield %inserted, %inserted_slice_24 : tensor<1x32xf32>, tensor<32x32xbf16>
            } else {
              scf.yield %arg27, %arg28 : tensor<1x32xf32>, tensor<32x32xbf16>
            }
            scf.yield %104#0, %104#1 : tensor<1x32xf32>, tensor<32x32xbf16>
          }
          %subview_20 = memref.subview %reinterpret_cast_5[%91, %93, %95] [1, 1, 32] [1, 1, 1] : memref<?x?x?xf32, strided<[?, ?, ?]>> to memref<1x32xf32, strided<[?, ?], offset: ?>>
          bufferization.materialize_in_destination %101#0 in writable %subview_20 : (tensor<1x32xf32>, memref<1x32xf32, strided<[?, ?], offset: ?>>) -> ()
          %subview_21 = memref.subview %reinterpret_cast_7[%91, %93, %95, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<?x?x?x32xbf16, strided<[?, ?, ?, ?]>> to memref<32x32xbf16, strided<[?, ?], offset: ?>>
          bufferization.materialize_in_destination %101#1 in writable %subview_21 : (tensor<32x32xbf16>, memref<32x32xbf16, strided<[?, ?], offset: ?>>) -> ()
          scf.yield %101#0 : tensor<1x32xf32>
        } else {
          scf.yield %54 : tensor<1x32xf32>
        }
        %63 = arith.divsi %20, %arg11 : i32
        %64 = arith.index_cast %63 : i32 to index
        %65 = arith.remsi %20, %arg11 : i32
        %66 = arith.index_cast %65 : i32 to index
        %67 = arith.muli %arg22, %c32_i32 : i32
        %68 = arith.index_cast %67 : i32 to index
        %subview_13 = memref.subview %reinterpret_cast_7[%64, %66, %68, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<?x?x?x32xbf16, strided<[?, ?, ?, ?]>> to memref<32x32xbf16, strided<[?, ?], offset: ?>>
        %alloc_14 = memref.alloc() : memref<32x32xbf16>
        memref.copy %subview_13, %alloc_14 : memref<32x32xbf16, strided<[?, ?], offset: ?>> to memref<32x32xbf16>
        %69 = bufferization.to_tensor %alloc_14 restrict : memref<32x32xbf16>
        %70 = hivm.hir.mmadL1 {b_transpose} ins(%37, %69, %true, %c16, %c32, %c32 : tensor<16x32xbf16>, tensor<32x32xbf16>, i1, index, index, index) outs(%50 : tensor<16x32xf32>) -> tensor<16x32xf32>
        %71 = hivm.hir.vmul ins(%70, %cst_1 : tensor<16x32xf32>, f32) outs(%70 : tensor<16x32xf32>) -> tensor<16x32xf32>
        %72 = hivm.hir.vadd ins(%arg23, %cst_0 : tensor<16x1xf32>, f32) outs(%51 : tensor<16x1xf32>) -> tensor<16x1xf32>
        %73 = hivm.hir.vreduce <max> ins(%71 : tensor<16x32xf32>) outs(%arg23 : tensor<16x1xf32>) reduce_dims = [1] -> tensor<16x1xf32>
        %74 = hivm.hir.vsub ins(%72, %73 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%55 : tensor<16x1xf32>) -> tensor<16x1xf32>
        %75 = hivm.hir.vexp ins(%74 : tensor<16x1xf32>) outs(%52 : tensor<16x1xf32>) -> tensor<16x1xf32>
        %76 = hivm.hir.vbrc ins(%73 : tensor<16x1xf32>) outs(%56 : tensor<16x32xf32>) broadcast_dims = [1] -> tensor<16x32xf32>
        %77 = hivm.hir.vsub ins(%71, %76 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%57 : tensor<16x32xf32>) -> tensor<16x32xf32>
        %78 = hivm.hir.vexp ins(%77 : tensor<16x32xf32>) outs(%71 : tensor<16x32xf32>) -> tensor<16x32xf32>
        %subview_15 = memref.subview %reinterpret_cast_5[%64, %66, %68] [1, 1, 32] [1, 1, 1] : memref<?x?x?xf32, strided<[?, ?, ?]>> to memref<32xf32, strided<[?], offset: ?>>
        %alloc_16 = memref.alloc() : memref<32xf32>
        memref.copy %subview_15, %alloc_16 : memref<32xf32, strided<[?], offset: ?>> to memref<32xf32>
        %79 = bufferization.to_tensor %alloc_16 restrict : memref<32xf32>
        %inserted_slice_17 = tensor.insert_slice %79 into %62[0, 0] [1, 32] [1, 1] : tensor<32xf32> into tensor<1x32xf32>
        %80 = hivm.hir.vbrc ins(%inserted_slice_17 : tensor<1x32xf32>) outs(%58 : tensor<16x32xf32>) broadcast_dims = [0] -> tensor<16x32xf32>
        %81 = hivm.hir.vmul ins(%78, %80 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%78 : tensor<16x32xf32>) -> tensor<16x32xf32>
        %82 = hivm.hir.vreduce <sum> ins(%81 : tensor<16x32xf32>) outs(%53 : tensor<16x1xf32>) reduce_dims = [1] -> tensor<16x1xf32>
        %83 = hivm.hir.vmul ins(%arg24, %75 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%59 : tensor<16x1xf32>) -> tensor<16x1xf32>
        %84 = hivm.hir.vadd ins(%83, %82 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%arg24 : tensor<16x1xf32>) -> tensor<16x1xf32>
        %85 = tensor.empty() : tensor<16x32xbf16>
        %86 = hivm.hir.vcast ins(%81 : tensor<16x32xf32>) outs(%85 : tensor<16x32xbf16>) -> tensor<16x32xbf16>
        %87 = hivm.hir.vbrc ins(%75 : tensor<16x1xf32>) outs(%60 : tensor<16x32xf32>) broadcast_dims = [1] -> tensor<16x32xf32>
        %88 = hivm.hir.vmul ins(%arg25, %87 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%arg25 : tensor<16x32xf32>) -> tensor<16x32xf32>
        %89 = hivm.hir.mmadL1 ins(%86, %69, %false, %c16, %c32, %c32 : tensor<16x32xbf16>, tensor<32x32xbf16>, i1, index, index, index) outs(%88 : tensor<16x32xf32>) -> tensor<16x32xf32>
        scf.yield %73, %84, %89 : tensor<16x1xf32>, tensor<16x1xf32>, tensor<16x32xf32>
      }
      %subview_10 = memref.subview %reinterpret_cast_9[%36, 0] [16, 1] [1, 1] : memref<64x1xf32, strided<[1, 1]>> to memref<16xf32, strided<[1], offset: ?>>
      %alloc_11 = memref.alloc() : memref<16xf32>
      memref.copy %subview_10, %alloc_11 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32>
      %41 = bufferization.to_tensor %alloc_11 restrict : memref<16xf32>
      %inserted_slice = tensor.insert_slice %41 into %21[0, 0] [16, 1] [1, 1] : tensor<16xf32> into tensor<16x1xf32>
      %42 = hivm.hir.vsub ins(%inserted_slice, %40#0 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%25 : tensor<16x1xf32>) -> tensor<16x1xf32>
      %43 = hivm.hir.vexp ins(%42 : tensor<16x1xf32>) outs(%26 : tensor<16x1xf32>) -> tensor<16x1xf32>
      %44 = hivm.hir.vadd ins(%40#1, %43 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%40#1 : tensor<16x1xf32>) -> tensor<16x1xf32>
      %45 = hivm.hir.vbrc ins(%44 : tensor<16x1xf32>) outs(%27 : tensor<16x32xf32>) broadcast_dims = [1] -> tensor<16x32xf32>
      %46 = hivm.hir.vdiv ins(%40#2, %45 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%40#2 : tensor<16x32xf32>) -> tensor<16x32xf32>
      %47 = tensor.empty() : tensor<16x32xbf16>
      %48 = hivm.hir.vcast ins(%46 : tensor<16x32xf32>) outs(%47 : tensor<16x32xbf16>) -> tensor<16x32xbf16>
      %subview_12 = memref.subview %reinterpret_cast_4[%32, %34, %36, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<?x?x64x32xbf16, strided<[?, ?, ?, ?]>> to memref<16x32xbf16, strided<[?, ?], offset: ?>>
      bufferization.materialize_in_destination %48 in writable %subview_12 : (tensor<16x32xbf16>, memref<16x32xbf16, strided<[?, ?], offset: ?>>) -> ()
    }
    return
  }
}
