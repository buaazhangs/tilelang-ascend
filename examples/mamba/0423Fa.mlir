2026-04-23 20:19:48  [TileLang:tilelang.env:WARNING]: Loading tilelang libs from dev root: /home/z00910011/tilelang-ascend/build
====== TVM IR ======
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def flash_attention(Q: T.Buffer((8192, 128), "float16"), K: T.Buffer((8192, 128), "float16"), V: T.Buffer((8192, 128), "float16"), Output: T.Buffer((8192, 128), "float16")):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "stackvm", "tag": ""}, "keys": [], "kind": "npuir", "tag": ""})})
        kernel_id = T.launch_thread("blockIdx.x", 24)
        vid = T.launch_thread("blockIdx.y", 48)
        Workspace1 = T.decl_buffer((2, 128, 256), scope="global.workspace;multi_buffer=8")
        Workspace2 = T.decl_buffer((2, 128, 256), "float16", scope="global.workspace;multi_buffer=8")
        Workspace3 = T.decl_buffer((2, 128, 128), scope="global.workspace;multi_buffer=8")
        for task_id in range((87 - kernel_id) // 24):
            Q_shared = T.decl_buffer((128, 128), "float16", scope="shared.flat")
            acc_m = T.decl_buffer((64, 1), scope="shared.flat")
            acc_l = T.decl_buffer((64, 1), scope="shared.flat")
            acc_o = T.decl_buffer((64, 128), scope="shared.flat")
            O_cast = T.decl_buffer((64, 128), "float16", scope="shared.flat")
            T.copy(T.region(Q[task_id * 3072 + kernel_id * 128, 0], 1, 128, 128), T.region(Q_shared[0, 0], 2, 128, 128))
            T.npuir_brc(0, T.region(acc_o[0, 0], 2, 64, 128))
            T.npuir_brc(0, T.region(acc_l[0, 0], 2, 64, 1))
            T.npuir_brc(T.float32("-inf"), T.region(acc_m[0, 0], 2, 64, 1))
            for k in T.serial(32, annotations={"num_stages": 8}):
                K_shared = T.decl_buffer((256, 128), "float16", scope="shared.flat")
                V_shared = T.decl_buffer((256, 128), "float16", scope="shared.flat")
                scores = T.decl_buffer((128, 256), scope="local.fragment")
                socres_l1 = T.decl_buffer((128, 256), "float16", scope="shared.flat")
                acc_o_l0c = T.decl_buffer((128, 128), scope="local.fragment")
                scores_ub = T.decl_buffer((64, 256), scope="shared.flat")
                scores_cast = T.decl_buffer((64, 256), "float16", scope="shared.flat")
                acc_o_ub = T.decl_buffer((64, 128), scope="shared.flat")
                local_max = T.decl_buffer((64, 1), scope="shared.flat")
                local_sum = T.decl_buffer((64, 1), scope="shared.flat")
                new_max = T.decl_buffer((64, 1), scope="shared.flat")
                correction = T.decl_buffer((64, 1), scope="shared.flat;multi_buffer=8")
                tmp1 = T.decl_buffer((64, 1), scope="shared.flat")
                T.copy(T.region(K[k * 256, 0], 1, 256, 128), T.region(K_shared[0, 0], 2, 256, 128))
                T.npuir_dot(T.region(Q_shared[0, 0], 1, 128, 128), T.region(K_shared[0, 0], 1, 128, 256), T.region(scores[0, 0], 3, 128, 256), T.bool(True), T.bool(False), T.bool(True))
                T.copy(T.region(scores[0, 0], 1, 128, 256), T.region(Workspace1[0, 0, 0], 2, 1, 128, 256))
                T.copy(T.region(Workspace1[0, vid * 64, 0], 1, 1, 64, 256), T.region(scores_ub[0, 0], 2, 64, 256))
                T.npuir_mul(T.region(scores_ub[0, 0], 1, 64, 256), T.float32(0.088388347648318447), T.region(scores_ub[0, 0], 2, 64, 256))
                T.npuir_reduce(T.region(scores_ub[0, 0], 1, 64, 256), T.region(local_max[0, 0], 2, 64, 1), "1", "max")
                T.npuir_max(T.region(acc_m[0, 0], 1, 64, 1), T.region(local_max[0, 0], 1, 64, 1), T.region(new_max[0, 0], 2, 64, 1))
                T.npuir_sub(T.region(scores_ub[0, 0], 1, 64, 256), T.region(new_max[0, 0], 1, 64, 1), T.region(scores_ub[0, 0], 2, 64, 256))
                T.npuir_exp(T.region(scores_ub[0, 0], 1, 64, 256), T.region(scores_ub[0, 0], 2, 64, 256))
                T.npuir_cast(T.region(scores_ub[0, 0], 1, 64, 256), T.region(scores_cast[0, 0], 2, 64, 256), "rint")
                T.npuir_reduce(T.region(scores_ub[0, 0], 1, 64, 256), T.region(local_sum[0, 0], 2, 64, 1), "1", "sum")
                T.copy(T.region(scores_cast[0, 0], 1, 64, 256), T.region(Workspace2[0, vid * 64, 0], 2, 1, 64, 256))
                T.npuir_sub(T.region(acc_m[0, 0], 1, 64, 1), T.region(new_max[0, 0], 1, 64, 1), T.region(tmp1[0, 0], 2, 64, 1))
                T.npuir_exp(T.region(tmp1[0, 0], 1, 64, 1), T.region(correction[0, 0], 2, 64, 1))
                T.npuir_mul(T.region(acc_l[0, 0], 1, 64, 1), T.region(correction[0, 0], 1, 64, 1), T.region(acc_l[0, 0], 2, 64, 1))
                T.npuir_add(T.region(acc_l[0, 0], 1, 64, 1), T.region(local_sum[0, 0], 1, 64, 1), T.region(acc_l[0, 0], 2, 64, 1))
                T.npuir_brc(0, T.region(tmp1[0, 0], 2, 64, 1))
                T.npuir_add(T.region(tmp1[0, 0], 1, 64, 1), T.region(new_max[0, 0], 1, 64, 1), T.region(acc_m[0, 0], 2, 64, 1))
                T.copy(T.region(Workspace2[0, 0, 0], 1, 1, 128, 256), T.region(socres_l1[0, 0], 2, 128, 256))
                T.copy(T.region(V[k * 256, 0], 1, 256, 128), T.region(V_shared[0, 0], 2, 256, 128))
                T.npuir_dot(T.region(socres_l1[0, 0], 1, 128, 256), T.region(V_shared[0, 0], 1, 256, 128), T.region(acc_o_l0c[0, 0], 3, 128, 128), T.bool(True), T.bool(False), T.bool(False))
                T.copy(T.region(acc_o_l0c[0, 0], 1, 128, 128), T.region(Workspace3[0, 0, 0], 2, 1, 128, 128))
                T.copy(T.region(Workspace3[0, vid * 64, 0], 1, 1, 64, 128), T.region(acc_o_ub[0, 0], 2, 64, 128))
                T.npuir_mul(T.region(acc_o[0, 0], 1, 64, 128), T.region(correction[0, 0], 1, 64, 1), T.region(acc_o[0, 0], 2, 64, 128))
                T.npuir_add(T.region(acc_o[0, 0], 1, 64, 128), T.region(acc_o_ub[0, 0], 1, 64, 128), T.region(acc_o[0, 0], 2, 64, 128))
            T.npuir_div(T.region(acc_o[0, 0], 1, 64, 128), T.region(acc_l[0, 0], 1, 64, 1), T.region(acc_o[0, 0], 2, 64, 128))
            T.npuir_cast(T.region(acc_o[0, 0], 1, 64, 128), T.region(O_cast[0, 0], 2, 64, 128), "rint")
            T.copy(T.region(O_cast[0, 0], 1, T.min(64, 8192 - vid * 64 - kernel_id * 128 - task_id * 3072), 128), T.region(Output[task_id * 3072 + kernel_id * 128 + vid * 64, 0], 2, T.min(64, 8192 - vid * 64 - kernel_id * 128 - task_id * 3072), 128))

====== npuir ======
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c128_i32 = arith.constant 128 : i32
    %1 = arith.muli %c128_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%2, %0] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %3 = hivm.hir.get_block_idx -> i64
    %4 = arith.trunci %3 : i64 to i32
    %5 = hivm.hir.get_sub_block_idx -> i64
    %6 = arith.trunci %5 : i64 to i32
    %7 = memref_ext.alloc_workspace() : memref<2x128x256xf32>
    annotation.mark %7 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf32>
    %8 = memref_ext.alloc_workspace() : memref<2x128x256xf16>
    annotation.mark %8 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf16>
    %9 = memref_ext.alloc_workspace() : memref<2x128x128xf32>
    annotation.mark %9 {hivm.multi_buffer = 8 : i32} : memref<2x128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c87_i32 = arith.constant 87 : i32
    %10 = arith.subi %c87_i32, %4 : i32
    %c24_i32 = arith.constant 24 : i32
    %11 = arith.divsi %10, %c24_i32 : i32
    %c1_i32_3 = arith.constant 1 : i32
    scf.for %arg13 = %c0_i32 to %11 step %c1_i32_3  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>>
      %alloc_4 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_6 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
      %alloc_7 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
      %c3072_i32 = arith.constant 3072 : i32
      %12 = arith.muli %arg13, %c3072_i32 : i32
      %c128_i32_8 = arith.constant 128 : i32
      %13 = arith.muli %4, %c128_i32_8 : i32
      %14 = arith.addi %12, %13 : i32
      %15 = arith.index_cast %14 : i32 to index
      %subview = memref.subview %reinterpret_cast[%15, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>>
      %c0_i32_9 = arith.constant 0 : i32
      %16 = arith.sitofp %c0_i32_9 : i32 to f32
      hivm.hir.vbrc ins(%16 : f32) outs(%alloc_6 : memref<64x128xf32, strided<[128, 1]>>)
      %17 = arith.sitofp %c0_i32_9 : i32 to f32
      hivm.hir.vbrc ins(%17 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
      %cst = arith.constant 0xFF800000 : f32
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>>)
      %c32_i32 = arith.constant 32 : i32
      %c1_i32_10 = arith.constant 1 : i32
      scf.for %arg14 = %c0_i32_9 to %c32_i32 step %c1_i32_10  : i32 {
        %alloc_13 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
        %alloc_14 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
        %alloc_15 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>>
        %alloc_16 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>>
        %alloc_17 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>>
        %alloc_18 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>>
        %alloc_19 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>>
        %alloc_20 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
        %alloc_21 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_22 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_23 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_24 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_24 {hivm.multi_buffer = 8 : i32} : memref<64x1xf32, strided<[1, 1]>>
        %alloc_25 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %c256_i32 = arith.constant 256 : i32
        %26 = arith.muli %arg14, %c256_i32 : i32
        %27 = arith.index_cast %26 : i32 to index
        %subview_26 = memref.subview %reinterpret_cast_1[%27, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_26, %alloc_13 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
        %true = arith.constant true
        %c128_i32_27 = arith.constant 128 : i32
        %28 = arith.index_cast %c128_i32_27 : i32 to index
        %29 = arith.index_cast %c256_i32 : i32 to index
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_13, %true, %28, %28, %29 : memref<128x128xf16, strided<[128, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_15 : memref<128x256xf32, strided<[256, 1]>>)
        %subview_28 = memref.subview %7[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<128x256xf32, strided<[256, 1]>>
        memref.copy %alloc_15, %subview_28 : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
        %c64_i32_29 = arith.constant 64 : i32
        %30 = arith.muli %6, %c64_i32_29 : i32
        %31 = arith.index_cast %30 : i32 to index
        %subview_30 = memref.subview %7[0, %31, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<64x256xf32, strided<[256, 1], offset: ?>>
        memref.copy %subview_30, %alloc_18 : memref<64x256xf32, strided<[256, 1], offset: ?>> to memref<64x256xf32, strided<[256, 1]>>
        %cst_31 = arith.constant 0.0883883461 : f32
        hivm.hir.vmul ins(%alloc_18, %cst_31 : memref<64x256xf32, strided<[256, 1]>>, f32) outs(%alloc_18 : memref<64x256xf32, strided<[256, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_18 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_21 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmax ins(%alloc_4, %alloc_21 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vsub ins(%alloc_18, %alloc_23 : memref<64x256xf32, strided<[256, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_18 : memref<64x256xf32, strided<[256, 1]>>) broadcast = [1]
        hivm.hir.vexp ins(%alloc_18 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_18 : memref<64x256xf32, strided<[256, 1]>>)
        hivm.hir.vcast ins(%alloc_18 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_19 : memref<64x256xf16, strided<[256, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_18 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_22 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        %subview_32 = memref.subview %8[0, %31, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<64x256xf16, strided<[256, 1], offset: ?>>
        memref.copy %alloc_19, %subview_32 : memref<64x256xf16, strided<[256, 1]>> to memref<64x256xf16, strided<[256, 1], offset: ?>>
        hivm.hir.vsub ins(%alloc_4, %alloc_23 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_25 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_25 : memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_24 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vmul ins(%alloc_5, %alloc_24 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_5, %alloc_22 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
        %c0_i32_33 = arith.constant 0 : i32
        %32 = arith.sitofp %c0_i32_33 : i32 to f32
        hivm.hir.vbrc ins(%32 : f32) outs(%alloc_25 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_25, %alloc_23 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>>)
        %subview_34 = memref.subview %8[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<128x256xf16, strided<[256, 1]>>
        memref.copy %subview_34, %alloc_16 : memref<128x256xf16, strided<[256, 1]>> to memref<128x256xf16, strided<[256, 1]>>
        %subview_35 = memref.subview %reinterpret_cast_0[%27, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_35, %alloc_14 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
        hivm.hir.mmadL1 ins(%alloc_16, %alloc_14, %true, %28, %29, %28 : memref<128x256xf16, strided<[256, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_17 : memref<128x128xf32, strided<[128, 1]>>)
        %subview_36 = memref.subview %9[0, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<128x128xf32, strided<[128, 1]>>
        memref.copy %alloc_17, %subview_36 : memref<128x128xf32, strided<[128, 1]>> to memref<128x128xf32, strided<[128, 1]>>
        %subview_37 = memref.subview %9[0, %31, 0] [1, 64, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
        memref.copy %subview_37, %alloc_20 : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x128xf32, strided<[128, 1]>>
        hivm.hir.vmul ins(%alloc_6, %alloc_24 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_6, %alloc_20 : memref<64x128xf32, strided<[128, 1]>>, memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_6 : memref<64x128xf32, strided<[128, 1]>>)
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_6, %alloc_5 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_6 : memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_7 : memref<64x128xf16, strided<[128, 1]>>)
      %c64_i32 = arith.constant 64 : i32
      %c8192_i32 = arith.constant 8192 : i32
      %18 = arith.muli %6, %c64_i32 : i32
      %19 = arith.subi %c8192_i32, %18 : i32
      %20 = arith.subi %19, %13 : i32
      %21 = arith.subi %20, %12 : i32
      %22 = arith.minsi %c64_i32, %21 : i32
      %23 = arith.index_cast %22 : i32 to index
      %subview_11 = memref.subview %alloc_7[0, 0] [%23, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1]>>
      %24 = arith.addi %14, %18 : i32
      %25 = arith.index_cast %24 : i32 to index
      %subview_12 = memref.subview %reinterpret_cast_2[%25, 0] [%23, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_11, %subview_12 : memref<?x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}
// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x128x256xf32>
    annotation.mark %4 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf32>
    %5 = memref_ext.alloc_workspace() : memref<2x128x256xf16>
    annotation.mark %5 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf16>
    %6 = memref_ext.alloc_workspace() : memref<2x128x128xf32>
    annotation.mark %6 {hivm.multi_buffer = 8 : i32} : memref<2x128x128xf32>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
      scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        %alloc_11 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
        %alloc_12 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
        %alloc_13 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>>
        %alloc_14 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>>
        %alloc_15 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>>
        %alloc_16 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>>
        %alloc_17 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>>
        %alloc_18 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
        %alloc_19 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_20 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_21 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_22 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_22 {hivm.multi_buffer = 8 : i32} : memref<64x1xf32, strided<[1, 1]>>
        %alloc_23 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %21 = arith.muli %arg14, %c256_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %subview_24 = memref.subview %reinterpret_cast_3[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_24, %alloc_11 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_11, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_13 : memref<128x256xf32, strided<[256, 1]>>)
        %subview_25 = memref.subview %4[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<128x256xf32, strided<[256, 1]>>
        memref.copy %alloc_13, %subview_25 : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
        %23 = arith.muli %3, %c64_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_26 = memref.subview %4[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<64x256xf32, strided<[256, 1], offset: ?>>
        memref.copy %subview_26, %alloc_16 : memref<64x256xf32, strided<[256, 1], offset: ?>> to memref<64x256xf32, strided<[256, 1]>>
        hivm.hir.vmul ins(%alloc_16, %cst_0 : memref<64x256xf32, strided<[256, 1]>>, f32) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_19 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmax ins(%alloc_5, %alloc_19 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_21 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vsub ins(%alloc_16, %alloc_21 : memref<64x256xf32, strided<[256, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) broadcast = [1]
        hivm.hir.vexp ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>)
        hivm.hir.vcast ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_17 : memref<64x256xf16, strided<[256, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_20 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        %subview_27 = memref.subview %5[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<64x256xf16, strided<[256, 1], offset: ?>>
        memref.copy %alloc_17, %subview_27 : memref<64x256xf16, strided<[256, 1]>> to memref<64x256xf16, strided<[256, 1], offset: ?>>
        hivm.hir.vsub ins(%alloc_5, %alloc_21 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_22 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vmul ins(%alloc_6, %alloc_22 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_6, %alloc_20 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vbrc ins(%cst : f32) outs(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_23, %alloc_21 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
        %subview_28 = memref.subview %5[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<128x256xf16, strided<[256, 1]>>
        memref.copy %subview_28, %alloc_14 : memref<128x256xf16, strided<[256, 1]>> to memref<128x256xf16, strided<[256, 1]>>
        %subview_29 = memref.subview %reinterpret_cast_2[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_29, %alloc_12 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
        hivm.hir.mmadL1 ins(%alloc_14, %alloc_12, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_15 : memref<128x128xf32, strided<[128, 1]>>)
        %subview_30 = memref.subview %6[0, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<128x128xf32, strided<[128, 1]>>
        memref.copy %alloc_15, %subview_30 : memref<128x128xf32, strided<[128, 1]>> to memref<128x128xf32, strided<[128, 1]>>
        %subview_31 = memref.subview %6[0, %24, 0] [1, 64, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
        memref.copy %subview_31, %alloc_18 : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x128xf32, strided<[128, 1]>>
        hivm.hir.vmul ins(%alloc_7, %alloc_22 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_7, %alloc_18 : memref<64x128xf32, strided<[128, 1]>>, memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>>)
      %13 = arith.muli %3, %c64_i32 : i32
      %14 = arith.subi %c8192_i32, %13 : i32
      %15 = arith.subi %14, %10 : i32
      %16 = arith.subi %15, %9 : i32
      %17 = arith.minsi %16, %c64_i32 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview_9 = memref.subview %alloc_8[0, 0] [%18, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1]>>
      %19 = arith.addi %11, %13 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%20, 0] [%18, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After AdaptTritonKernel (adapt-triton-kernel) ('builtin.module' operation) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x128x256xf32>
    annotation.mark %4 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf32>
    %5 = memref_ext.alloc_workspace() : memref<2x128x256xf16>
    annotation.mark %5 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf16>
    %6 = memref_ext.alloc_workspace() : memref<2x128x128xf32>
    annotation.mark %6 {hivm.multi_buffer = 8 : i32} : memref<2x128x128xf32>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
      scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        %alloc_11 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
        %alloc_12 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
        %alloc_13 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>>
        %alloc_14 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>>
        %alloc_15 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>>
        %alloc_16 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>>
        %alloc_17 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>>
        %alloc_18 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
        %alloc_19 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_20 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_21 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_22 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_22 {hivm.multi_buffer = 8 : i32} : memref<64x1xf32, strided<[1, 1]>>
        %alloc_23 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %21 = arith.muli %arg14, %c256_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %subview_24 = memref.subview %reinterpret_cast_3[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_24, %alloc_11 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_11, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_13 : memref<128x256xf32, strided<[256, 1]>>)
        %subview_25 = memref.subview %4[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<128x256xf32, strided<[256, 1]>>
        memref.copy %alloc_13, %subview_25 : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
        %23 = arith.muli %3, %c64_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_26 = memref.subview %4[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<64x256xf32, strided<[256, 1], offset: ?>>
        memref.copy %subview_26, %alloc_16 : memref<64x256xf32, strided<[256, 1], offset: ?>> to memref<64x256xf32, strided<[256, 1]>>
        hivm.hir.vmul ins(%alloc_16, %cst_0 : memref<64x256xf32, strided<[256, 1]>>, f32) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_19 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmax ins(%alloc_5, %alloc_19 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_21 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vsub ins(%alloc_16, %alloc_21 : memref<64x256xf32, strided<[256, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) broadcast = [1]
        hivm.hir.vexp ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>)
        hivm.hir.vcast ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_17 : memref<64x256xf16, strided<[256, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_20 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        %subview_27 = memref.subview %5[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<64x256xf16, strided<[256, 1], offset: ?>>
        memref.copy %alloc_17, %subview_27 : memref<64x256xf16, strided<[256, 1]>> to memref<64x256xf16, strided<[256, 1], offset: ?>>
        hivm.hir.vsub ins(%alloc_5, %alloc_21 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_22 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vmul ins(%alloc_6, %alloc_22 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_6, %alloc_20 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vbrc ins(%cst : f32) outs(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_23, %alloc_21 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
        %subview_28 = memref.subview %5[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<128x256xf16, strided<[256, 1]>>
        memref.copy %subview_28, %alloc_14 : memref<128x256xf16, strided<[256, 1]>> to memref<128x256xf16, strided<[256, 1]>>
        %subview_29 = memref.subview %reinterpret_cast_2[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_29, %alloc_12 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
        hivm.hir.mmadL1 ins(%alloc_14, %alloc_12, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_15 : memref<128x128xf32, strided<[128, 1]>>)
        %subview_30 = memref.subview %6[0, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<128x128xf32, strided<[128, 1]>>
        memref.copy %alloc_15, %subview_30 : memref<128x128xf32, strided<[128, 1]>> to memref<128x128xf32, strided<[128, 1]>>
        %subview_31 = memref.subview %6[0, %24, 0] [1, 64, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
        memref.copy %subview_31, %alloc_18 : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x128xf32, strided<[128, 1]>>
        hivm.hir.vmul ins(%alloc_7, %alloc_22 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_7, %alloc_18 : memref<64x128xf32, strided<[128, 1]>>, memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>>)
      %13 = arith.muli %3, %c64_i32 : i32
      %14 = arith.subi %c8192_i32, %13 : i32
      %15 = arith.subi %14, %10 : i32
      %16 = arith.subi %15, %9 : i32
      %17 = arith.minsi %16, %c64_i32 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview_9 = memref.subview %alloc_8[0, 0] [%18, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1]>>
      %19 = arith.addi %11, %13 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%20, 0] [%18, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRInsertWorkspace (tilelangir-insert-workspace) ('func.func' operation: @flash_attention) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x128x256xf32>
    annotation.mark %4 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf32>
    %5 = memref_ext.alloc_workspace() : memref<2x128x256xf16>
    annotation.mark %5 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf16>
    %6 = memref_ext.alloc_workspace() : memref<2x128x128xf32>
    annotation.mark %6 {hivm.multi_buffer = 8 : i32} : memref<2x128x128xf32>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
      scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        %alloc_11 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
        %alloc_12 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
        %alloc_13 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>>
        %alloc_14 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>>
        %alloc_15 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>>
        %alloc_16 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>>
        %alloc_17 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>>
        %alloc_18 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
        %alloc_19 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_20 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_21 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_22 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_22 {hivm.multi_buffer = 8 : i32} : memref<64x1xf32, strided<[1, 1]>>
        %alloc_23 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %21 = arith.muli %arg14, %c256_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %subview_24 = memref.subview %reinterpret_cast_3[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_24, %alloc_11 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_11, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_13 : memref<128x256xf32, strided<[256, 1]>>)
        %subview_25 = memref.subview %4[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<128x256xf32, strided<[256, 1]>>
        memref.copy %alloc_13, %subview_25 : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
        %23 = arith.muli %3, %c64_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_26 = memref.subview %4[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<64x256xf32, strided<[256, 1], offset: ?>>
        memref.copy %subview_26, %alloc_16 : memref<64x256xf32, strided<[256, 1], offset: ?>> to memref<64x256xf32, strided<[256, 1]>>
        hivm.hir.vmul ins(%alloc_16, %cst_0 : memref<64x256xf32, strided<[256, 1]>>, f32) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_19 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmax ins(%alloc_5, %alloc_19 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_21 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vsub ins(%alloc_16, %alloc_21 : memref<64x256xf32, strided<[256, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) broadcast = [1]
        hivm.hir.vexp ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>)
        hivm.hir.vcast ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_17 : memref<64x256xf16, strided<[256, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_20 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        %subview_27 = memref.subview %5[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<64x256xf16, strided<[256, 1], offset: ?>>
        memref.copy %alloc_17, %subview_27 : memref<64x256xf16, strided<[256, 1]>> to memref<64x256xf16, strided<[256, 1], offset: ?>>
        hivm.hir.vsub ins(%alloc_5, %alloc_21 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_22 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vmul ins(%alloc_6, %alloc_22 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_6, %alloc_20 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vbrc ins(%cst : f32) outs(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_23, %alloc_21 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
        %subview_28 = memref.subview %5[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<128x256xf16, strided<[256, 1]>>
        memref.copy %subview_28, %alloc_14 : memref<128x256xf16, strided<[256, 1]>> to memref<128x256xf16, strided<[256, 1]>>
        %subview_29 = memref.subview %reinterpret_cast_2[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_29, %alloc_12 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
        hivm.hir.mmadL1 ins(%alloc_14, %alloc_12, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_15 : memref<128x128xf32, strided<[128, 1]>>)
        %subview_30 = memref.subview %6[0, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<128x128xf32, strided<[128, 1]>>
        memref.copy %alloc_15, %subview_30 : memref<128x128xf32, strided<[128, 1]>> to memref<128x128xf32, strided<[128, 1]>>
        %subview_31 = memref.subview %6[0, %24, 0] [1, 64, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
        memref.copy %subview_31, %alloc_18 : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x128xf32, strided<[128, 1]>>
        hivm.hir.vmul ins(%alloc_7, %alloc_22 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_7, %alloc_18 : memref<64x128xf32, strided<[128, 1]>>, memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>>)
      %13 = arith.muli %3, %c64_i32 : i32
      %14 = arith.subi %c8192_i32, %13 : i32
      %15 = arith.subi %14, %10 : i32
      %16 = arith.subi %15, %9 : i32
      %17 = arith.minsi %16, %c64_i32 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview_9 = memref.subview %alloc_8[0, 0] [%18, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1]>>
      %19 = arith.addi %11, %13 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%20, 0] [%18, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRMarkMultiBuffer (tilelangir-mark-multibuffer) ('func.func' operation: @flash_attention) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x128x256xf32>
    annotation.mark %4 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf32>
    %5 = memref_ext.alloc_workspace() : memref<2x128x256xf16>
    annotation.mark %5 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf16>
    %6 = memref_ext.alloc_workspace() : memref<2x128x128xf32>
    annotation.mark %6 {hivm.multi_buffer = 8 : i32} : memref<2x128x128xf32>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
      scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        %alloc_11 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
        %alloc_12 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
        %alloc_13 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>>
        %alloc_14 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>>
        %alloc_15 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>>
        %alloc_16 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>>
        %alloc_17 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>>
        %alloc_18 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
        %alloc_19 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_20 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_21 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %alloc_22 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_22 {hivm.multi_buffer = 8 : i32} : memref<64x1xf32, strided<[1, 1]>>
        %alloc_23 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        %21 = arith.muli %arg14, %c256_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %subview_24 = memref.subview %reinterpret_cast_3[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_24, %alloc_11 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
        hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_11, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_13 : memref<128x256xf32, strided<[256, 1]>>)
        %subview_25 = memref.subview %4[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<128x256xf32, strided<[256, 1]>>
        memref.copy %alloc_13, %subview_25 : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
        %23 = arith.muli %3, %c64_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        %subview_26 = memref.subview %4[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<64x256xf32, strided<[256, 1], offset: ?>>
        memref.copy %subview_26, %alloc_16 : memref<64x256xf32, strided<[256, 1], offset: ?>> to memref<64x256xf32, strided<[256, 1]>>
        hivm.hir.vmul ins(%alloc_16, %cst_0 : memref<64x256xf32, strided<[256, 1]>>, f32) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>)
        hivm.hir.vreduce <max> ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_19 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        hivm.hir.vmax ins(%alloc_5, %alloc_19 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_21 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vsub ins(%alloc_16, %alloc_21 : memref<64x256xf32, strided<[256, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) broadcast = [1]
        hivm.hir.vexp ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>)
        hivm.hir.vcast ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_17 : memref<64x256xf16, strided<[256, 1]>>)
        hivm.hir.vreduce <sum> ins(%alloc_16 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_20 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
        %subview_27 = memref.subview %5[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<64x256xf16, strided<[256, 1], offset: ?>>
        memref.copy %alloc_17, %subview_27 : memref<64x256xf16, strided<[256, 1]>> to memref<64x256xf16, strided<[256, 1], offset: ?>>
        hivm.hir.vsub ins(%alloc_5, %alloc_21 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vexp ins(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_22 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vmul ins(%alloc_6, %alloc_22 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_6, %alloc_20 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vbrc ins(%cst : f32) outs(%alloc_23 : memref<64x1xf32, strided<[1, 1]>>)
        hivm.hir.vadd ins(%alloc_23, %alloc_21 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
        %subview_28 = memref.subview %5[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<128x256xf16, strided<[256, 1]>>
        memref.copy %subview_28, %alloc_14 : memref<128x256xf16, strided<[256, 1]>> to memref<128x256xf16, strided<[256, 1]>>
        %subview_29 = memref.subview %reinterpret_cast_2[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        memref.copy %subview_29, %alloc_12 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
        hivm.hir.mmadL1 ins(%alloc_14, %alloc_12, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_15 : memref<128x128xf32, strided<[128, 1]>>)
        %subview_30 = memref.subview %6[0, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<128x128xf32, strided<[128, 1]>>
        memref.copy %alloc_15, %subview_30 : memref<128x128xf32, strided<[128, 1]>> to memref<128x128xf32, strided<[128, 1]>>
        %subview_31 = memref.subview %6[0, %24, 0] [1, 64, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
        memref.copy %subview_31, %alloc_18 : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x128xf32, strided<[128, 1]>>
        hivm.hir.vmul ins(%alloc_7, %alloc_22 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
        hivm.hir.vadd ins(%alloc_7, %alloc_18 : memref<64x128xf32, strided<[128, 1]>>, memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>>)
      %13 = arith.muli %3, %c64_i32 : i32
      %14 = arith.subi %c8192_i32, %13 : i32
      %15 = arith.subi %14, %10 : i32
      %16 = arith.subi %15, %9 : i32
      %17 = arith.minsi %16, %c64_i32 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview_9 = memref.subview %alloc_8[0, 0] [%18, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1]>>
      %19 = arith.addi %11, %13 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%20, 0] [%18, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRCVSplit (tilelangir-cv-split) ('func.func' operation: @flash_attention) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x128x256xf32>
    annotation.mark %4 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf32>
    %5 = memref_ext.alloc_workspace() : memref<2x128x256xf16>
    annotation.mark %5 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf16>
    %6 = memref_ext.alloc_workspace() : memref<2x128x128xf32>
    annotation.mark %6 {hivm.multi_buffer = 8 : i32} : memref<2x128x128xf32>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
      scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        %alloc_11 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
        annotation.mark %alloc_11 {hivm.multi_buffer = 8 : i32} : memref<64x1xf32, strided<[1, 1]>>
        %21 = arith.muli %arg14, %c256_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = arith.muli %3, %c64_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        scope.scope : () -> () {
          %alloc_12 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
          %alloc_13 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>>
          %subview_14 = memref.subview %reinterpret_cast_3[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_14, %alloc_12 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_12, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_13 : memref<128x256xf32, strided<[256, 1]>>)
          %subview_15 = memref.subview %4[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<128x256xf32, strided<[256, 1]>>
          memref.copy %alloc_13, %subview_15 : memref<128x256xf32, strided<[256, 1]>> to memref<128x256xf32, strided<[256, 1]>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %alloc_12 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>>
          %alloc_13 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>>
          %alloc_14 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
          %alloc_15 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
          %alloc_16 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
          %alloc_17 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>>
          %subview_18 = memref.subview %4[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf32> to memref<64x256xf32, strided<[256, 1], offset: ?>>
          memref.copy %subview_18, %alloc_12 : memref<64x256xf32, strided<[256, 1], offset: ?>> to memref<64x256xf32, strided<[256, 1]>>
          hivm.hir.vmul ins(%alloc_12, %cst_0 : memref<64x256xf32, strided<[256, 1]>>, f32) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>>)
          hivm.hir.vreduce <max> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_14 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_14 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_16 : memref<64x1xf32, strided<[1, 1]>>)
          hivm.hir.vsub ins(%alloc_12, %alloc_16 : memref<64x256xf32, strided<[256, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>>)
          hivm.hir.vcast ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_13 : memref<64x256xf16, strided<[256, 1]>>)
          hivm.hir.vreduce <sum> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>>) outs(%alloc_15 : memref<64x1xf32, strided<[1, 1]>>) reduce_dims = [1]
          %subview_19 = memref.subview %5[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<64x256xf16, strided<[256, 1], offset: ?>>
          memref.copy %alloc_13, %subview_19 : memref<64x256xf16, strided<[256, 1]>> to memref<64x256xf16, strided<[256, 1], offset: ?>>
          hivm.hir.vsub ins(%alloc_5, %alloc_16 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>>)
          hivm.hir.vexp ins(%alloc_17 : memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_11 : memref<64x1xf32, strided<[1, 1]>>)
          hivm.hir.vmul ins(%alloc_6, %alloc_11 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_15 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>>)
          hivm.hir.vadd ins(%alloc_17, %alloc_16 : memref<64x1xf32, strided<[1, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>>)
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %alloc_12 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>>
          %alloc_13 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>>
          %alloc_14 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>>
          %subview_15 = memref.subview %5[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf16> to memref<128x256xf16, strided<[256, 1]>>
          memref.copy %subview_15, %alloc_13 : memref<128x256xf16, strided<[256, 1]>> to memref<128x256xf16, strided<[256, 1]>>
          %subview_16 = memref.subview %reinterpret_cast_2[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_16, %alloc_12 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>>
          hivm.hir.mmadL1 ins(%alloc_13, %alloc_12, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>>, memref<256x128xf16, strided<[128, 1]>>, i1, index, index, index) outs(%alloc_14 : memref<128x128xf32, strided<[128, 1]>>)
          %subview_17 = memref.subview %6[0, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<128x128xf32, strided<[128, 1]>>
          memref.copy %alloc_14, %subview_17 : memref<128x128xf32, strided<[128, 1]>> to memref<128x128xf32, strided<[128, 1]>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %alloc_12 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>>
          %subview_13 = memref.subview %6[0, %24, 0] [1, 64, 128] [1, 1, 1] : memref<2x128x128xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
          memref.copy %subview_13, %alloc_12 : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x128xf32, strided<[128, 1]>>
          hivm.hir.vmul ins(%alloc_7, %alloc_11 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_12 : memref<64x128xf32, strided<[128, 1]>>, memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>)
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>>, memref<64x1xf32, strided<[1, 1]>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>>)
      %13 = arith.muli %3, %c64_i32 : i32
      %14 = arith.subi %c8192_i32, %13 : i32
      %15 = arith.subi %14, %10 : i32
      %16 = arith.subi %15, %9 : i32
      %17 = arith.minsi %16, %c64_i32 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview_9 = memref.subview %alloc_8[0, 0] [%18, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1]>>
      %19 = arith.addi %11, %13 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%20, 0] [%18, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRInferMemScope (tilelangir-infer-mem-scope) ('func.func' operation: @flash_attention) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x128x256xf32, #hivm.address_space<gm>>
    annotation.mark %4 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() : memref<2x128x256xf16, #hivm.address_space<gm>>
    annotation.mark %5 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf16, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() : memref<2x128x128xf32, #hivm.address_space<gm>>
    annotation.mark %6 {hivm.multi_buffer = 8 : i32} : memref<2x128x128xf32, #hivm.address_space<gm>>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        %alloc_11 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        annotation.mark %alloc_11 {hivm.multi_buffer = 8 : i32} : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %21 = arith.muli %arg14, %c256_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = arith.muli %3, %c64_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        scope.scope : () -> () {
          %alloc_12 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_13 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          %subview_14 = memref.subview %reinterpret_cast_3[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_14, %alloc_12 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_12, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_13 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          %subview_15 = memref.subview %4[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf32, #hivm.address_space<gm>> to memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<gm>>
          memref.copy %alloc_13, %subview_15 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>> to memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<gm>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %alloc_12 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_13 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_14 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_15 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_16 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_17 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %subview_18 = memref.subview %4[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf32, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_18, %alloc_12 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_12, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_12, %alloc_16 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          %subview_19 = memref.subview %5[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf16, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_13, %subview_19 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc_5, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vmul ins(%alloc_6, %alloc_11 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_17, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %alloc_12 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_13 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_14 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          %subview_15 = memref.subview %5[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf16, #hivm.address_space<gm>> to memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<gm>>
          memref.copy %subview_15, %alloc_13 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %subview_16 = memref.subview %reinterpret_cast_2[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_16, %alloc_12 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 ins(%alloc_13, %alloc_12, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_14 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          %subview_17 = memref.subview %6[0, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32, #hivm.address_space<gm>> to memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<gm>>
          memref.copy %alloc_14, %subview_17 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>> to memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<gm>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %alloc_12 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %subview_13 = memref.subview %6[0, %24, 0] [1, 64, 128] [1, 1, 1] : memref<2x128x128xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_13, %alloc_12 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_7, %alloc_11 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_12 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %13 = arith.muli %3, %c64_i32 : i32
      %14 = arith.subi %c8192_i32, %13 : i32
      %15 = arith.subi %14, %10 : i32
      %16 = arith.subi %15, %9 : i32
      %17 = arith.minsi %16, %c64_i32 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview_9 = memref.subview %alloc_8[0, 0] [%18, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %19 = arith.addi %11, %13 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%20, 0] [%18, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRMergeCopyChains (tilelangir-merge-copy-chains) ('func.func' operation: @flash_attention) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<2x128x256xf32, #hivm.address_space<gm>>
    annotation.mark %4 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() : memref<2x128x256xf16, #hivm.address_space<gm>>
    annotation.mark %5 {hivm.multi_buffer = 8 : i32} : memref<2x128x256xf16, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() : memref<2x128x128xf32, #hivm.address_space<gm>>
    annotation.mark %6 {hivm.multi_buffer = 8 : i32} : memref<2x128x128xf32, #hivm.address_space<gm>>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        %alloc_11 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        annotation.mark %alloc_11 {hivm.multi_buffer = 8 : i32} : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %21 = arith.muli %arg14, %c256_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = arith.muli %3, %c64_i32 : i32
        %24 = arith.index_cast %23 : i32 to index
        scope.scope : () -> () {
          %alloc_12 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_13 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          %subview_14 = memref.subview %reinterpret_cast_3[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_14, %alloc_12 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_12, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_13 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          %subview_15 = memref.subview %4[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf32, #hivm.address_space<gm>> to memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<gm>>
          memref.copy %alloc_13, %subview_15 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>> to memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<gm>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %alloc_12 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_13 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_14 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_15 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_16 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_17 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %subview_18 = memref.subview %4[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf32, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_18, %alloc_12 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_12, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_12, %alloc_16 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          %subview_19 = memref.subview %5[0, %24, 0] [1, 64, 256] [1, 1, 1] : memref<2x128x256xf16, #hivm.address_space<gm>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %alloc_13, %subview_19 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc_5, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vmul ins(%alloc_6, %alloc_11 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_17, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          %alloc_12 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_13 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_14 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          %subview_15 = memref.subview %5[0, 0, 0] [1, 128, 256] [1, 1, 1] : memref<2x128x256xf16, #hivm.address_space<gm>> to memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<gm>>
          memref.copy %subview_15, %alloc_13 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %subview_16 = memref.subview %reinterpret_cast_2[%22, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_16, %alloc_12 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 ins(%alloc_13, %alloc_12, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_14 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          %subview_17 = memref.subview %6[0, 0, 0] [1, 128, 128] [1, 1, 1] : memref<2x128x128xf32, #hivm.address_space<gm>> to memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<gm>>
          memref.copy %alloc_14, %subview_17 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>> to memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<gm>>
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %alloc_12 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %subview_13 = memref.subview %6[0, %24, 0] [1, 64, 128] [1, 1, 1] : memref<2x128x128xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          memref.copy %subview_13, %alloc_12 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_7, %alloc_11 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_12 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
          scope.return
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %13 = arith.muli %3, %c64_i32 : i32
      %14 = arith.subi %c8192_i32, %13 : i32
      %15 = arith.subi %14, %10 : i32
      %16 = arith.subi %15, %9 : i32
      %17 = arith.minsi %16, %c64_i32 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview_9 = memref.subview %alloc_8[0, 0] [%18, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %19 = arith.addi %11, %13 : i32
      %20 = arith.index_cast %19 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%20, 0] [%18, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


  [getIndexFactor] Checking value: %54 = "arith.index_cast"(%53) : (i32) -> index
  [getIndexFactor] Following IndexCastOp
  [getIndexFactor] Checking value: %53 = "arith.muli"(%arg14, %8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  [getIndexFactor] Checking value: <block argument> of type 'i32' at index: 0
  [getIndexFactor] Found direct match (factor=1)
  [getIndexFactor] Found mul: 256 * 1 = 256
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: %56 = "arith.index_cast"(%55) : (i32) -> index
  [getIndexFactor] Checking value: %56 = "arith.index_cast"(%55) : (i32) -> index
  [getIndexFactor] Following IndexCastOp
  [getIndexFactor] Checking value: %55 = "arith.muli"(%24, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  [getIndexFactor] Checking value: %24 = "arith.trunci"(%23) : (i64) -> i32
  [getIndexFactor] No match
  [getIndexFactor] No match
  [adjustWorkspaceSubviewOp] No factor found, keeping original
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: %56 = "arith.index_cast"(%55) : (i32) -> index
  [getIndexFactor] Checking value: %56 = "arith.index_cast"(%55) : (i32) -> index
  [getIndexFactor] Following IndexCastOp
  [getIndexFactor] Checking value: %55 = "arith.muli"(%24, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  [getIndexFactor] Checking value: %24 = "arith.trunci"(%23) : (i64) -> i32
  [getIndexFactor] No match
  [getIndexFactor] No match
  [adjustWorkspaceSubviewOp] No factor found, keeping original
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [getIndexFactor] Checking value: %54 = "arith.index_cast"(%53) : (i32) -> index
  [getIndexFactor] Following IndexCastOp
  [getIndexFactor] Checking value: %53 = "arith.muli"(%arg14, %8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  [getIndexFactor] Checking value: <block argument> of type 'i32' at index: 0
  [getIndexFactor] Found direct match (factor=1)
  [getIndexFactor] Found mul: 256 * 1 = 256
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[0]: constant 0
  [adjustWorkspaceSubviewOp] Processing offset[1]: %56 = "arith.index_cast"(%55) : (i32) -> index
  [getIndexFactor] Checking value: %56 = "arith.index_cast"(%55) : (i32) -> index
  [getIndexFactor] Following IndexCastOp
  [getIndexFactor] Checking value: %55 = "arith.muli"(%24, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  [getIndexFactor] Checking value: %24 = "arith.trunci"(%23) : (i64) -> i32
  [getIndexFactor] No match
  [getIndexFactor] No match
  [adjustWorkspaceSubviewOp] No factor found, keeping original
  [adjustWorkspaceSubviewOp] Processing offset[2]: constant 0
WARNING: Expected 6 subviews to be processed, but got 0.
This implies some subviews were skipped or the pass crashed early.
// -----// IR Dump After TileLangIREnableMultiBuffer (tilelangir-enable-multi-buffer) ('builtin.module' operation) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() : memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() : memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %c8_i32 = arith.constant 8 : i32
      %13 = arith.divsi %c32_i32, %c8_i32 : i32
      scf.for %arg14 = %c0_i32 to %13 step %c1_i32  : i32 {
        %alloc_11 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        annotation.mark %alloc_11 {hivm.multi_buffer = 8 : i32} : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %22 = arith.muli %arg14, %c256_i32 : i32
        %23 = arith.index_cast %22 : i32 to index
        %24 = arith.muli %3, %c64_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %c0_i32_12 = arith.constant 0 : i32
        %c8_i32_13 = arith.constant 8 : i32
        %c1_i32_14 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_12 to %c8_i32_13 step %c1_i32_14  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %c256_i32_25 = arith.constant 256 : i32
          %30 = arith.muli %27, %c256_i32_25 : i32
          %31 = arith.index_cast %30 : i32 to index
          %subview_26 = memref.subview %reinterpret_cast_3[%31, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_27 = memref.subview %4[%29, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_28 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_29 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          memref.copy %subview_26, %alloc_28 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_28, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_29 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          memref.copy %alloc_29, %collapse_shape : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>> to memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_15 = arith.constant 0 : i32
        %c8_i32_16 = arith.constant 8 : i32
        %c1_i32_17 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_15 to %c8_i32_16 step %c1_i32_17  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %subview_25 = memref.subview %4[%29, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_26 = memref.subview %5[%29, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_27 = memref.collapse_shape %subview_26 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_28 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_29 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_30 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_31 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_32 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_33 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_28 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_28, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_30 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_28, %alloc_32 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          memref.copy %alloc_29, %collapse_shape_27 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc_5, %alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vexp ins(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_11 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vmul ins(%alloc_6, %alloc_11 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_31 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_33, %alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_18 = arith.constant 0 : i32
        %c8_i32_19 = arith.constant 8 : i32
        %c1_i32_20 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_18 to %c8_i32_19 step %c1_i32_20  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %c256_i32_25 = arith.constant 256 : i32
          %30 = arith.muli %27, %c256_i32_25 : i32
          %31 = arith.index_cast %30 : i32 to index
          %subview_26 = memref.subview %reinterpret_cast_2[%31, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_27 = memref.subview %5[%29, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_28 = memref.subview %6[%29, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_29 = memref.collapse_shape %subview_28 [[0, 1, 2], [3]] : memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_30 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_31 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_32 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          memref.copy %collapse_shape, %alloc_31 : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          memref.copy %subview_26, %alloc_30 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 ins(%alloc_31, %alloc_30, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_32 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          memref.copy %alloc_32, %collapse_shape_29 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>> to memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_21 = arith.constant 0 : i32
        %c8_i32_22 = arith.constant 8 : i32
        %c1_i32_23 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_21 to %c8_i32_22 step %c1_i32_23  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %subview_25 = memref.subview %6[%29, 0, %25, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_26 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_26 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_7, %alloc_11 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_26 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %14 = arith.muli %3, %c64_i32 : i32
      %15 = arith.subi %c8192_i32, %14 : i32
      %16 = arith.subi %15, %10 : i32
      %17 = arith.subi %16, %9 : i32
      %18 = arith.minsi %17, %c64_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %subview_9 = memref.subview %alloc_8[0, 0] [%19, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %20 = arith.addi %11, %14 : i32
      %21 = arith.index_cast %20 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%21, 0] [%19, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIREnableLocalBuffer (tilelangir-enable-local-buffer) ('builtin.module' operation) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() : memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() : memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %c8_i32 = arith.constant 8 : i32
      %13 = arith.divsi %c32_i32, %c8_i32 : i32
      scf.for %arg14 = %c0_i32 to %13 step %c1_i32  : i32 {
        %alloc_11 = memref.alloc() : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
        %22 = arith.muli %arg14, %c256_i32 : i32
        %23 = arith.index_cast %22 : i32 to index
        %24 = arith.muli %3, %c64_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %c0_i32_12 = arith.constant 0 : i32
        %c8_i32_13 = arith.constant 8 : i32
        %c1_i32_14 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_12 to %c8_i32_13 step %c1_i32_14  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %c256_i32_25 = arith.constant 256 : i32
          %30 = arith.muli %27, %c256_i32_25 : i32
          %31 = arith.index_cast %30 : i32 to index
          %subview_26 = memref.subview %reinterpret_cast_3[%31, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_27 = memref.subview %4[%29, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_28 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_29 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          memref.copy %subview_26, %alloc_28 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_28, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_29 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          memref.copy %alloc_29, %collapse_shape : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>> to memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_15 = arith.constant 0 : i32
        %c8_i32_16 = arith.constant 8 : i32
        %c1_i32_17 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_15 to %c8_i32_16 step %c1_i32_17  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %subview_25 = memref.subview %4[%29, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_26 = memref.subview %5[%29, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_27 = memref.collapse_shape %subview_26 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_28 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_29 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_30 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_31 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_32 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_33 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_28 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_28, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_30 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_28, %alloc_32 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          memref.copy %alloc_29, %collapse_shape_27 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc_5, %alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %30 = arith.index_cast %arg15 : i32 to index
          %subview_34 = memref.subview %alloc_11[%30, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_35 = memref.collapse_shape %subview_34 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_35 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
          %31 = arith.index_cast %arg15 : i32 to index
          %subview_36 = memref.subview %alloc_11[%31, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_37 = memref.collapse_shape %subview_36 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_6, %collapse_shape_37 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_31 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_33, %alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_18 = arith.constant 0 : i32
        %c8_i32_19 = arith.constant 8 : i32
        %c1_i32_20 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_18 to %c8_i32_19 step %c1_i32_20  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %c256_i32_25 = arith.constant 256 : i32
          %30 = arith.muli %27, %c256_i32_25 : i32
          %31 = arith.index_cast %30 : i32 to index
          %subview_26 = memref.subview %reinterpret_cast_2[%31, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_27 = memref.subview %5[%29, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_28 = memref.subview %6[%29, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_29 = memref.collapse_shape %subview_28 [[0, 1, 2], [3]] : memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_30 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_31 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_32 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          memref.copy %collapse_shape, %alloc_31 : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          memref.copy %subview_26, %alloc_30 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          hivm.hir.mmadL1 ins(%alloc_31, %alloc_30, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_32 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          memref.copy %alloc_32, %collapse_shape_29 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>> to memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_21 = arith.constant 0 : i32
        %c8_i32_22 = arith.constant 8 : i32
        %c1_i32_23 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_21 to %c8_i32_22 step %c1_i32_23  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %subview_25 = memref.subview %6[%29, 0, %25, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_26 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_26 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %30 = arith.index_cast %arg15 : i32 to index
          %subview_27 = memref.subview %alloc_11[%30, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_7, %collapse_shape_28 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_26 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %14 = arith.muli %3, %c64_i32 : i32
      %15 = arith.subi %c8192_i32, %14 : i32
      %16 = arith.subi %15, %10 : i32
      %17 = arith.subi %16, %9 : i32
      %18 = arith.minsi %17, %c64_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %subview_9 = memref.subview %alloc_8[0, 0] [%19, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %20 = arith.addi %11, %14 : i32
      %21 = arith.index_cast %20 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%21, 0] [%19, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRSpecializeCube (tilelangir-specialize-cube) ('func.func' operation: @flash_attention) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() : memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() : memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() : memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %c8_i32 = arith.constant 8 : i32
      %13 = arith.divsi %c32_i32, %c8_i32 : i32
      scf.for %arg14 = %c0_i32 to %13 step %c1_i32  : i32 {
        %alloc_11 = memref.alloc() : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
        %22 = arith.muli %arg14, %c256_i32 : i32
        %23 = arith.index_cast %22 : i32 to index
        %24 = arith.muli %3, %c64_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %c0_i32_12 = arith.constant 0 : i32
        %c8_i32_13 = arith.constant 8 : i32
        %c1_i32_14 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_12 to %c8_i32_13 step %c1_i32_14  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %c256_i32_25 = arith.constant 256 : i32
          %30 = arith.muli %27, %c256_i32_25 : i32
          %31 = arith.index_cast %30 : i32 to index
          %subview_26 = memref.subview %reinterpret_cast_3[%31, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_27 = memref.subview %4[%29, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_28 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_29 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_28 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_28, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_29 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_29 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_15 = arith.constant 0 : i32
        %c8_i32_16 = arith.constant 8 : i32
        %c1_i32_17 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_15 to %c8_i32_16 step %c1_i32_17  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %subview_25 = memref.subview %4[%29, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_26 = memref.subview %5[%29, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_27 = memref.collapse_shape %subview_26 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_28 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_29 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_30 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_31 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_32 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_33 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_28 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_28, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_30 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_28, %alloc_32 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          memref.copy %alloc_29, %collapse_shape_27 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc_5, %alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %30 = arith.index_cast %arg15 : i32 to index
          %subview_34 = memref.subview %alloc_11[%30, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_35 = memref.collapse_shape %subview_34 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_35 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
          %31 = arith.index_cast %arg15 : i32 to index
          %subview_36 = memref.subview %alloc_11[%31, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_37 = memref.collapse_shape %subview_36 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_6, %collapse_shape_37 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_31 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_33, %alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_18 = arith.constant 0 : i32
        %c8_i32_19 = arith.constant 8 : i32
        %c1_i32_20 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_18 to %c8_i32_19 step %c1_i32_20  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %c256_i32_25 = arith.constant 256 : i32
          %30 = arith.muli %27, %c256_i32_25 : i32
          %31 = arith.index_cast %30 : i32 to index
          %subview_26 = memref.subview %reinterpret_cast_2[%31, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_27 = memref.subview %5[%29, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_28 = memref.subview %6[%29, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_29 = memref.collapse_shape %subview_28 [[0, 1, 2], [3]] : memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_30 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_31 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_32 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_31 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_30 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_31, %alloc_30, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_32 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_32 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_29 : memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_21 = arith.constant 0 : i32
        %c8_i32_22 = arith.constant 8 : i32
        %c1_i32_23 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_21 to %c8_i32_22 step %c1_i32_23  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %subview_25 = memref.subview %6[%29, 0, %25, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_26 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_26 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %30 = arith.index_cast %arg15 : i32 to index
          %subview_27 = memref.subview %alloc_11[%30, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_7, %collapse_shape_28 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_26 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %14 = arith.muli %3, %c64_i32 : i32
      %15 = arith.subi %c8192_i32, %14 : i32
      %16 = arith.subi %15, %10 : i32
      %17 = arith.subi %16, %9 : i32
      %18 = arith.minsi %17, %c64_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %subview_9 = memref.subview %alloc_8[0, 0] [%19, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %20 = arith.addi %11, %14 : i32
      %21 = arith.index_cast %20 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%21, 0] [%19, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After BindWorkSpaceArg (hivm-bind-workspace-arg) ('func.func' operation: @flash_attention) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %5 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %6 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %c8_i32 = arith.constant 8 : i32
      %13 = arith.divsi %c32_i32, %c8_i32 : i32
      scf.for %arg14 = %c0_i32 to %13 step %c1_i32  : i32 {
        %alloc_11 = memref.alloc() : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
        %22 = arith.muli %arg14, %c256_i32 : i32
        %23 = arith.index_cast %22 : i32 to index
        %24 = arith.muli %3, %c64_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %c0_i32_12 = arith.constant 0 : i32
        %c8_i32_13 = arith.constant 8 : i32
        %c1_i32_14 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_12 to %c8_i32_13 step %c1_i32_14  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %c256_i32_25 = arith.constant 256 : i32
          %30 = arith.muli %27, %c256_i32_25 : i32
          %31 = arith.index_cast %30 : i32 to index
          %subview_26 = memref.subview %reinterpret_cast_3[%31, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_27 = memref.subview %4[%29, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_28 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_29 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_28 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_28, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_29 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_29 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_15 = arith.constant 0 : i32
        %c8_i32_16 = arith.constant 8 : i32
        %c1_i32_17 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_15 to %c8_i32_16 step %c1_i32_17  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %subview_25 = memref.subview %4[%29, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_26 = memref.subview %5[%29, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_27 = memref.collapse_shape %subview_26 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_28 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_29 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_30 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_31 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_32 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_33 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_28 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_28, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_30 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_28, %alloc_32 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          memref.copy %alloc_29, %collapse_shape_27 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc_5, %alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %30 = arith.index_cast %arg15 : i32 to index
          %subview_34 = memref.subview %alloc_11[%30, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_35 = memref.collapse_shape %subview_34 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_35 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
          %31 = arith.index_cast %arg15 : i32 to index
          %subview_36 = memref.subview %alloc_11[%31, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_37 = memref.collapse_shape %subview_36 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_6, %collapse_shape_37 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_31 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_33, %alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_18 = arith.constant 0 : i32
        %c8_i32_19 = arith.constant 8 : i32
        %c1_i32_20 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_18 to %c8_i32_19 step %c1_i32_20  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %c256_i32_25 = arith.constant 256 : i32
          %30 = arith.muli %27, %c256_i32_25 : i32
          %31 = arith.index_cast %30 : i32 to index
          %subview_26 = memref.subview %reinterpret_cast_2[%31, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_27 = memref.subview %5[%29, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_28 = memref.subview %6[%29, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_29 = memref.collapse_shape %subview_28 [[0, 1, 2], [3]] : memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_30 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_31 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_32 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_31 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_30 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_31, %alloc_30, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_32 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_32 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_29 : memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_21 = arith.constant 0 : i32
        %c8_i32_22 = arith.constant 8 : i32
        %c1_i32_23 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_21 to %c8_i32_22 step %c1_i32_23  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %subview_25 = memref.subview %6[%29, 0, %25, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_26 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_26 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %30 = arith.index_cast %arg15 : i32 to index
          %subview_27 = memref.subview %alloc_11[%30, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_7, %collapse_shape_28 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_26 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %14 = arith.muli %3, %c64_i32 : i32
      %15 = arith.subi %c8192_i32, %14 : i32
      %16 = arith.subi %15, %10 : i32
      %17 = arith.subi %16, %9 : i32
      %18 = arith.minsi %17, %c64_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %subview_9 = memref.subview %alloc_8[0, 0] [%19, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %20 = arith.addi %11, %14 : i32
      %21 = arith.index_cast %20 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%21, 0] [%19, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRPlanWorkspaceMemory (tilelangir-plan-workspace-memory) ('func.func' operation: @flash_attention) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %c0 = arith.constant 0 : index
    %4 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %c2097152 = arith.constant 2097152 : index
    %5 = memref_ext.alloc_workspace() from %arg2 offset = [%c2097152] : from memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %c3145728 = arith.constant 3145728 : index
    %6 = memref_ext.alloc_workspace() from %arg2 offset = [%c3145728] : from memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %c8_i32 = arith.constant 8 : i32
      %13 = arith.divsi %c32_i32, %c8_i32 : i32
      scf.for %arg14 = %c0_i32 to %13 step %c1_i32  : i32 {
        %alloc_11 = memref.alloc() : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
        %22 = arith.muli %arg14, %c256_i32 : i32
        %23 = arith.index_cast %22 : i32 to index
        %24 = arith.muli %3, %c64_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %c0_i32_12 = arith.constant 0 : i32
        %c8_i32_13 = arith.constant 8 : i32
        %c1_i32_14 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_12 to %c8_i32_13 step %c1_i32_14  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %c256_i32_25 = arith.constant 256 : i32
          %30 = arith.muli %27, %c256_i32_25 : i32
          %31 = arith.index_cast %30 : i32 to index
          %subview_26 = memref.subview %reinterpret_cast_3[%31, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_27 = memref.subview %4[%29, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_28 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_29 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_28 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_28, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_29 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_29 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_15 = arith.constant 0 : i32
        %c8_i32_16 = arith.constant 8 : i32
        %c1_i32_17 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_15 to %c8_i32_16 step %c1_i32_17  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %subview_25 = memref.subview %4[%29, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_26 = memref.subview %5[%29, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_27 = memref.collapse_shape %subview_26 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_28 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_29 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_30 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_31 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_32 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_33 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_28 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_28, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_30 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_30 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_28, %alloc_32 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_29 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_28 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          memref.copy %alloc_29, %collapse_shape_27 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc_5, %alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %30 = arith.index_cast %arg15 : i32 to index
          %subview_34 = memref.subview %alloc_11[%30, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_35 = memref.collapse_shape %subview_34 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_35 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
          %31 = arith.index_cast %arg15 : i32 to index
          %subview_36 = memref.subview %alloc_11[%31, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_37 = memref.collapse_shape %subview_36 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_6, %collapse_shape_37 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_31 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_33, %alloc_32 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_18 = arith.constant 0 : i32
        %c8_i32_19 = arith.constant 8 : i32
        %c1_i32_20 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_18 to %c8_i32_19 step %c1_i32_20  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %c256_i32_25 = arith.constant 256 : i32
          %30 = arith.muli %27, %c256_i32_25 : i32
          %31 = arith.index_cast %30 : i32 to index
          %subview_26 = memref.subview %reinterpret_cast_2[%31, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_27 = memref.subview %5[%29, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_27 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_28 = memref.subview %6[%29, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_29 = memref.collapse_shape %subview_28 [[0, 1, 2], [3]] : memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_30 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_31 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_32 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_31 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_30 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_31, %alloc_30, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_32 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_32 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_29 : memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_21 = arith.constant 0 : i32
        %c8_i32_22 = arith.constant 8 : i32
        %c1_i32_23 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_21 to %c8_i32_22 step %c1_i32_23  : i32 {
          %c8_i32_24 = arith.constant 8 : i32
          %26 = arith.muli %arg14, %c8_i32_24 : i32
          %27 = arith.addi %26, %arg15 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = arith.index_cast %arg15 : i32 to index
          %subview_25 = memref.subview %6[%29, 0, %25, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_25 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_26 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_26 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %30 = arith.index_cast %arg15 : i32 to index
          %subview_27 = memref.subview %alloc_11[%30, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_28 = memref.collapse_shape %subview_27 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_7, %collapse_shape_28 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_26 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %14 = arith.muli %3, %c64_i32 : i32
      %15 = arith.subi %c8192_i32, %14 : i32
      %16 = arith.subi %15, %10 : i32
      %17 = arith.subi %16, %9 : i32
      %18 = arith.minsi %17, %c64_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %subview_9 = memref.subview %alloc_8[0, 0] [%19, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %20 = arith.addi %11, %14 : i32
      %21 = arith.index_cast %20 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_4[%21, 0] [%19, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_9, %subview_10 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRInsertCVSync (tilelangir-insert-cv-sync) ('builtin.module' operation) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %c0 = arith.constant 0 : index
    %4 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %c2097152 = arith.constant 2097152 : index
    %5 = memref_ext.alloc_workspace() from %arg2 offset = [%c2097152] : from memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %c3145728 = arith.constant 3145728 : index
    %6 = memref_ext.alloc_workspace() from %arg2 offset = [%c3145728] : from memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %c8_i32 = arith.constant 8 : i32
      %13 = arith.divsi %c32_i32, %c8_i32 : i32
      %c0_i32_9 = arith.constant 0 : i32
      %c1_i32_10 = arith.constant 1 : i32
      %c8_i32_11 = arith.constant 8 : i32
      scf.for %arg14 = %c0_i32_9 to %c8_i32_11 step %c1_i32_10  : i32 {
        %22 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %22 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scf.for %arg14 = %c0_i32 to %13 step %c1_i32  : i32 {
        %alloc_14 = memref.alloc() : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
        %22 = arith.muli %arg14, %c256_i32 : i32
        %23 = arith.index_cast %22 : i32 to index
        %24 = arith.muli %3, %c64_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %c0_i32_15 = arith.constant 0 : i32
        %c8_i32_16 = arith.constant 8 : i32
        %c1_i32_17 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_15 to %c8_i32_16 step %c1_i32_17  : i32 {
          %26 = arith.extsi %arg15 : i32 to i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %26
          %c8_i32_27 = arith.constant 8 : i32
          %27 = arith.muli %arg14, %c8_i32_27 : i32
          %28 = arith.addi %27, %arg15 : i32
          %29 = arith.index_cast %28 : i32 to index
          %30 = arith.index_cast %arg15 : i32 to index
          %c256_i32_28 = arith.constant 256 : i32
          %31 = arith.muli %28, %c256_i32_28 : i32
          %32 = arith.index_cast %31 : i32 to index
          %subview_29 = memref.subview %reinterpret_cast_3[%32, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_30 = memref.subview %4[%30, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_30 [[0, 1, 2], [3]] : memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_31 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_32 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_29 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_31 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_31, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_32 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_32 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %26 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_18 = arith.constant 0 : i32
        %c8_i32_19 = arith.constant 8 : i32
        %c1_i32_20 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_18 to %c8_i32_19 step %c1_i32_20  : i32 {
          %26 = arith.extsi %arg15 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c8_i64 = arith.constant 8 : i64
          %27 = arith.addi %26, %c8_i64 : i64
          %28 = arith.remsi %27, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %26
          %c8_i32_27 = arith.constant 8 : i32
          %29 = arith.muli %arg14, %c8_i32_27 : i32
          %30 = arith.addi %29, %arg15 : i32
          %31 = arith.index_cast %30 : i32 to index
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_28 = memref.subview %4[%32, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_28 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_29 = memref.subview %5[%32, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_30 = memref.collapse_shape %subview_29 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_31 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_32 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_33 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_34 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_35 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_36 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_31 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_31, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_35 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_31, %alloc_35 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_34 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          memref.copy %alloc_32, %collapse_shape_30 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc_5, %alloc_35 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_36 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_37 = memref.subview %alloc_14[%33, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_38 = memref.collapse_shape %subview_37 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%alloc_36 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_38 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
          %34 = arith.index_cast %arg15 : i32 to index
          %subview_39 = memref.subview %alloc_14[%34, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_40 = memref.collapse_shape %subview_39 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_6, %collapse_shape_40 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_34 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_36 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_36, %alloc_35 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %28 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_21 = arith.constant 0 : i32
        %c8_i32_22 = arith.constant 8 : i32
        %c1_i32_23 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_21 to %c8_i32_22 step %c1_i32_23  : i32 {
          %26 = arith.extsi %arg15 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c8_i64 = arith.constant 8 : i64
          %27 = arith.addi %26, %c8_i64 : i64
          %28 = arith.remsi %27, %c16_i64 : i64
          %c8_i64_27 = arith.constant 8 : i64
          %29 = arith.addi %26, %c8_i64_27 : i64
          %30 = arith.remsi %29, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %28
          %c8_i32_28 = arith.constant 8 : i32
          %31 = arith.muli %arg14, %c8_i32_28 : i32
          %32 = arith.addi %31, %arg15 : i32
          %33 = arith.index_cast %32 : i32 to index
          %34 = arith.index_cast %arg15 : i32 to index
          %c256_i32_29 = arith.constant 256 : i32
          %35 = arith.muli %32, %c256_i32_29 : i32
          %36 = arith.index_cast %35 : i32 to index
          %subview_30 = memref.subview %reinterpret_cast_2[%36, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_31 = memref.subview %5[%34, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_32 = memref.subview %6[%34, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_33 = memref.collapse_shape %subview_32 [[0, 1, 2], [3]] : memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_34 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_35 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_36 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_35 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.nd2nz {dst_continuous} ins(%subview_30 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_34 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_35, %alloc_34, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_36 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_36 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_33 : memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %30 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_24 = arith.constant 0 : i32
        %c8_i32_25 = arith.constant 8 : i32
        %c1_i32_26 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_24 to %c8_i32_25 step %c1_i32_26  : i32 {
          %26 = arith.extsi %arg15 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c8_i64 = arith.constant 8 : i64
          %27 = arith.addi %26, %c8_i64 : i64
          %28 = arith.remsi %27, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %28
          %c8_i32_27 = arith.constant 8 : i32
          %29 = arith.muli %arg14, %c8_i32_27 : i32
          %30 = arith.addi %29, %arg15 : i32
          %31 = arith.index_cast %30 : i32 to index
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_28 = memref.subview %6[%32, 0, %25, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_28 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_29 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_29 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_30 = memref.subview %alloc_14[%33, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_31 = memref.collapse_shape %subview_30 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_7, %collapse_shape_31 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_29 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %26 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 8 : i32}
      scf.for %arg14 = %c0_i32_9 to %c8_i32_11 step %c1_i32_10  : i32 {
        %22 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %22
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %14 = arith.muli %3, %c64_i32 : i32
      %15 = arith.subi %c8192_i32, %14 : i32
      %16 = arith.subi %15, %10 : i32
      %17 = arith.subi %16, %9 : i32
      %18 = arith.minsi %17, %c64_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %subview_12 = memref.subview %alloc_8[0, 0] [%19, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %20 = arith.addi %11, %14 : i32
      %21 = arith.index_cast %20 : i32 to index
      %subview_13 = memref.subview %reinterpret_cast_4[%21, 0] [%19, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_12, %subview_13 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After InsertInferWorkSpaceSizeFunc (hivm-insert-infer-workspace-size-func) ('func.func' operation: @flash_attention) //----- //
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c4194304 = arith.constant 4194304 : index
    return %c4194304 : index
  }
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %c0 = arith.constant 0 : index
    %4 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %c2097152 = arith.constant 2097152 : index
    %5 = memref_ext.alloc_workspace() from %arg2 offset = [%c2097152] : from memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %c3145728 = arith.constant 3145728 : index
    %6 = memref_ext.alloc_workspace() from %arg2 offset = [%c3145728] : from memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %7 = arith.subi %c87_i32, %1 : i32
    %8 = arith.divsi %7, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %8 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %9 = arith.muli %arg13, %c3072_i32 : i32
      %10 = arith.muli %1, %c128_i32 : i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %subview = memref.subview %reinterpret_cast[%12, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      %c8_i32 = arith.constant 8 : i32
      %13 = arith.divsi %c32_i32, %c8_i32 : i32
      %c0_i32_9 = arith.constant 0 : i32
      %c1_i32_10 = arith.constant 1 : i32
      %c8_i32_11 = arith.constant 8 : i32
      scf.for %arg14 = %c0_i32_9 to %c8_i32_11 step %c1_i32_10  : i32 {
        %22 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %22 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scf.for %arg14 = %c0_i32 to %13 step %c1_i32  : i32 {
        %alloc_14 = memref.alloc() : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
        %22 = arith.muli %arg14, %c256_i32 : i32
        %23 = arith.index_cast %22 : i32 to index
        %24 = arith.muli %3, %c64_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %c0_i32_15 = arith.constant 0 : i32
        %c8_i32_16 = arith.constant 8 : i32
        %c1_i32_17 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_15 to %c8_i32_16 step %c1_i32_17  : i32 {
          %26 = arith.extsi %arg15 : i32 to i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %26
          %c8_i32_27 = arith.constant 8 : i32
          %27 = arith.muli %arg14, %c8_i32_27 : i32
          %28 = arith.addi %27, %arg15 : i32
          %29 = arith.index_cast %28 : i32 to index
          %30 = arith.index_cast %arg15 : i32 to index
          %c256_i32_28 = arith.constant 256 : i32
          %31 = arith.muli %28, %c256_i32_28 : i32
          %32 = arith.index_cast %31 : i32 to index
          %subview_29 = memref.subview %reinterpret_cast_3[%32, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_30 = memref.subview %4[%30, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_30 [[0, 1, 2], [3]] : memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_31 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_32 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_29 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_31 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_31, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_32 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_32 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %26 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_18 = arith.constant 0 : i32
        %c8_i32_19 = arith.constant 8 : i32
        %c1_i32_20 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_18 to %c8_i32_19 step %c1_i32_20  : i32 {
          %26 = arith.extsi %arg15 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c8_i64 = arith.constant 8 : i64
          %27 = arith.addi %26, %c8_i64 : i64
          %28 = arith.remsi %27, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %26
          %c8_i32_27 = arith.constant 8 : i32
          %29 = arith.muli %arg14, %c8_i32_27 : i32
          %30 = arith.addi %29, %arg15 : i32
          %31 = arith.index_cast %30 : i32 to index
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_28 = memref.subview %4[%32, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_28 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_29 = memref.subview %5[%32, 0, %25, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_30 = memref.collapse_shape %subview_29 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_31 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_32 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_33 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_34 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_35 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_36 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_31 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_31, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_5, %alloc_33 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_35 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_31, %alloc_35 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_32 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_31 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_34 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          memref.copy %alloc_32, %collapse_shape_30 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc_5, %alloc_35 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_36 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_37 = memref.subview %alloc_14[%33, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_38 = memref.collapse_shape %subview_37 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%alloc_36 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_38 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
          %34 = arith.index_cast %arg15 : i32 to index
          %subview_39 = memref.subview %alloc_14[%34, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_40 = memref.collapse_shape %subview_39 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_6, %collapse_shape_40 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_6, %alloc_34 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_36 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_36, %alloc_35 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %28 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        %c0_i32_21 = arith.constant 0 : i32
        %c8_i32_22 = arith.constant 8 : i32
        %c1_i32_23 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_21 to %c8_i32_22 step %c1_i32_23  : i32 {
          %26 = arith.extsi %arg15 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c8_i64 = arith.constant 8 : i64
          %27 = arith.addi %26, %c8_i64 : i64
          %28 = arith.remsi %27, %c16_i64 : i64
          %c8_i64_27 = arith.constant 8 : i64
          %29 = arith.addi %26, %c8_i64_27 : i64
          %30 = arith.remsi %29, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %28
          %c8_i32_28 = arith.constant 8 : i32
          %31 = arith.muli %arg14, %c8_i32_28 : i32
          %32 = arith.addi %31, %arg15 : i32
          %33 = arith.index_cast %32 : i32 to index
          %34 = arith.index_cast %arg15 : i32 to index
          %c256_i32_29 = arith.constant 256 : i32
          %35 = arith.muli %32, %c256_i32_29 : i32
          %36 = arith.index_cast %35 : i32 to index
          %subview_30 = memref.subview %reinterpret_cast_2[%36, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_31 = memref.subview %5[%34, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_31 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_32 = memref.subview %6[%34, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_33 = memref.collapse_shape %subview_32 [[0, 1, 2], [3]] : memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_34 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_35 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_36 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_35 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.nd2nz {dst_continuous} ins(%subview_30 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_34 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_35, %alloc_34, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_36 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_36 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_33 : memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %30 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        %c0_i32_24 = arith.constant 0 : i32
        %c8_i32_25 = arith.constant 8 : i32
        %c1_i32_26 = arith.constant 1 : i32
        scf.for %arg15 = %c0_i32_24 to %c8_i32_25 step %c1_i32_26  : i32 {
          %26 = arith.extsi %arg15 : i32 to i64
          %c16_i64 = arith.constant 16 : i64
          %c8_i64 = arith.constant 8 : i64
          %27 = arith.addi %26, %c8_i64 : i64
          %28 = arith.remsi %27, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %28
          %c8_i32_27 = arith.constant 8 : i32
          %29 = arith.muli %arg14, %c8_i32_27 : i32
          %30 = arith.addi %29, %arg15 : i32
          %31 = arith.index_cast %30 : i32 to index
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_28 = memref.subview %6[%32, 0, %25, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_28 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_29 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_29 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_30 = memref.subview %alloc_14[%33, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_31 = memref.collapse_shape %subview_30 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_7, %collapse_shape_31 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_7, %alloc_29 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %26 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 8 : i32}
      scf.for %arg14 = %c0_i32_9 to %c8_i32_11 step %c1_i32_10  : i32 {
        %22 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %22
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %14 = arith.muli %3, %c64_i32 : i32
      %15 = arith.subi %c8192_i32, %14 : i32
      %16 = arith.subi %15, %10 : i32
      %17 = arith.subi %16, %9 : i32
      %18 = arith.minsi %17, %c64_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %subview_12 = memref.subview %alloc_8[0, 0] [%19, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %20 = arith.addi %11, %14 : i32
      %21 = arith.index_cast %20 : i32 to index
      %subview_13 = memref.subview %reinterpret_cast_4[%21, 0] [%19, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_12, %subview_13 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After LowerMemRefExt (lower-memref-ext) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @flash_attention_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c4194304 = arith.constant 4194304 : index
    return %c4194304 : index
  }
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "aic"} {
    %c4_i32 = arith.constant 4 : i32
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3145728 = arith.constant 3145728 : index
    %c2097152 = arith.constant 2097152 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map(%8)[%c2097152]
    %view_5 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map(%11)[%c3145728]
    %view_6 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c87_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_7 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %15 = arith.muli %arg13, %c3072_i32 : i32
      %16 = arith.muli %1, %c128_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview = memref.subview %reinterpret_cast[%18, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_9 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_8 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_7 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %27 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %alloc_13 = memref.alloc() : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
        %27 = arith.muli %3, %c64_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %29
          %30 = arith.muli %arg14, %c8_i32 : i32
          %31 = arith.addi %30, %arg15 : i32
          %32 = arith.index_cast %arg15 : i32 to index
          %33 = arith.muli %31, %c256_i32 : i32
          %34 = arith.index_cast %33 : i32 to index
          %subview_14 = memref.subview %reinterpret_cast_3[%34, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_15 = memref.subview %view[%32, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_15 [[0, 1, 2], [3]] : memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_16 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_17 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_14 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_16 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_16, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_17 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_17 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %29 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          %30 = arith.addi %29, %c8_i64 : i64
          %31 = arith.remsi %30, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %29
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_14 = memref.subview %view[%32, 0, %28, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_14 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_15 = memref.subview %view_5[%32, 0, %28, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_16 = memref.collapse_shape %subview_15 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_17 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_18 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_19 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_20 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_21 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_22 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_17 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_17, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_17 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_17 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_19 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc_7, %alloc_19 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_21 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_17, %alloc_21 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_17 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_17 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_18 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_17 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_20 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          memref.copy %alloc_18, %collapse_shape_16 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc_7, %alloc_21 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_22 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_23 = memref.subview %alloc_13[%33, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_24 = memref.collapse_shape %subview_23 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%alloc_22 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
          %34 = arith.index_cast %arg15 : i32 to index
          %subview_25 = memref.subview %alloc_13[%34, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_26 = memref.collapse_shape %subview_25 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_8, %collapse_shape_26 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_8, %alloc_20 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_22 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_22, %alloc_21 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %31 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          %30 = arith.addi %29, %c8_i64 : i64
          %31 = arith.remsi %30, %c16_i64 : i64
          %32 = arith.addi %29, %c8_i64 : i64
          %33 = arith.remsi %32, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %31
          %34 = arith.muli %arg14, %c8_i32 : i32
          %35 = arith.addi %34, %arg15 : i32
          %36 = arith.index_cast %arg15 : i32 to index
          %37 = arith.muli %35, %c256_i32 : i32
          %38 = arith.index_cast %37 : i32 to index
          %subview_14 = memref.subview %reinterpret_cast_2[%38, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_15 = memref.subview %view_5[%36, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_15 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_16 = memref.subview %view_6[%36, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_17 = memref.collapse_shape %subview_16 [[0, 1, 2], [3]] : memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_18 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_19 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_20 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_19 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.nd2nz {dst_continuous} ins(%subview_14 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_18 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_19, %alloc_18, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_20 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_20 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_17 : memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %33 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          %30 = arith.addi %29, %c8_i64 : i64
          %31 = arith.remsi %30, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %31
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_14 = memref.subview %view_6[%32, 0, %28, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_14 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_15 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_15 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_16 = memref.subview %alloc_13[%33, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_17 = memref.collapse_shape %subview_16 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_9, %collapse_shape_17 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_9, %alloc_15 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %29 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      } {tilelangir.num_stages = 8 : i32}
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %27
      } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
      hivm.hir.vdiv ins(%alloc_9, %alloc_8 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_9 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_9 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %19 = arith.muli %3, %c64_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c64_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview_11 = memref.subview %alloc_10[0, 0] [%24, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
      %subview_12 = memref.subview %reinterpret_cast_4[%26, 0] [%24, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_11, %subview_12 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRSplitMixKernel (tilelangir-split-mix-kernel) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @flash_attention_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c4194304 = arith.constant 4194304 : index
    return %c4194304 : index
  }
  func.func @flash_attention_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i32 = arith.constant 4 : i32
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3145728 = arith.constant 3145728 : index
    %c2097152 = arith.constant 2097152 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map(%8)[%c2097152]
    %view_4 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map(%11)[%c3145728]
    %view_5 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c87_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %15 = arith.muli %arg13, %c3072_i32 : i32
      %16 = arith.muli %1, %c128_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview = memref.subview %reinterpret_cast[%18, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %27 = arith.muli %3, %c64_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %29
          %30 = arith.muli %arg14, %c8_i32 : i32
          %31 = arith.addi %30, %arg15 : i32
          %32 = arith.index_cast %arg15 : i32 to index
          %33 = arith.muli %31, %c256_i32 : i32
          %34 = arith.index_cast %33 : i32 to index
          %subview_6 = memref.subview %reinterpret_cast_3[%34, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_7 = memref.subview %view[%32, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_7 [[0, 1, 2], [3]] : memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_8 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_9 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_8 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_8, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_9 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_9 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %29 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          %30 = arith.addi %29, %c8_i64 : i64
          %31 = arith.remsi %30, %c16_i64 : i64
          %32 = arith.addi %29, %c8_i64 : i64
          %33 = arith.remsi %32, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %31
          %34 = arith.muli %arg14, %c8_i32 : i32
          %35 = arith.addi %34, %arg15 : i32
          %36 = arith.index_cast %arg15 : i32 to index
          %37 = arith.muli %35, %c256_i32 : i32
          %38 = arith.index_cast %37 : i32 to index
          %subview_6 = memref.subview %reinterpret_cast_2[%38, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_7 = memref.subview %view_4[%36, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_7 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_8 = memref.subview %view_5[%36, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_9 = memref.collapse_shape %subview_8 [[0, 1, 2], [3]] : memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_10 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_11 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_12 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_11 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_10 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_11, %alloc_10, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_12 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_12 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_9 : memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %33 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 8 : i32}
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %27
      }
      %19 = arith.muli %3, %c64_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c64_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
    }
    return
  }
  func.func @flash_attention_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i32 = arith.constant 4 : i32
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3145728 = arith.constant 3145728 : index
    %c2097152 = arith.constant 2097152 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map(%8)[%c2097152]
    %view_2 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map(%11)[%c3145728]
    %view_3 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c87_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %15 = arith.muli %arg13, %c3072_i32 : i32
      %16 = arith.muli %1, %c128_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %27 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %alloc_8 = memref.alloc() : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
        %27 = arith.muli %3, %c64_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          %30 = arith.addi %29, %c8_i64 : i64
          %31 = arith.remsi %30, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %29
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_9 = memref.subview %view[%32, 0, %28, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_10 = memref.subview %view_2[%32, 0, %28, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_11 = memref.collapse_shape %subview_10 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_12 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_13 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_14 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_15 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_16 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_17 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_12 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_12, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc, %alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_12, %alloc_16 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          memref.copy %alloc_13, %collapse_shape_11 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_18 = memref.subview %alloc_8[%33, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_19 = memref.collapse_shape %subview_18 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
          %34 = arith.index_cast %arg15 : i32 to index
          %subview_20 = memref.subview %alloc_8[%34, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_21 = memref.collapse_shape %subview_20 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_4, %collapse_shape_21 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_4, %alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_17, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %31 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          %30 = arith.addi %29, %c8_i64 : i64
          %31 = arith.remsi %30, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %31
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_9 = memref.subview %view_3[%32, 0, %28, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_10 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_10 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_11 = memref.subview %alloc_8[%33, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_12 = memref.collapse_shape %subview_11 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_5, %collapse_shape_12 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_5, %alloc_10 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %29 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_5, %alloc_4 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %19 = arith.muli %3, %c64_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c64_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %alloc_6[0, 0] [%24, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
      %subview_7 = memref.subview %reinterpret_cast[%26, 0] [%24, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %subview_7 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}


// -----// IR Dump After TileLangIRWrapHostFunction (tilelangir-wrap-host-function) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @flash_attention_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c4194304 = arith.constant 4194304 : index
    return %c4194304 : index
  }
  func.func @flash_attention_get_kernel_num_args() -> i32 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c4_i32 = arith.constant 4 : i32
    return %c4_i32 : i32
  }
  func.func @flash_attention_get_kernel_arg_type(%arg0: i32) -> i32 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c-1_i32 = arith.constant -1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c129_i32 = arith.constant 129 : i32
    %0 = arith.cmpi eq, %arg0, %c3_i32 : i32
    %1 = arith.select %0, %c129_i32, %c-1_i32 : i32
    %c2_i32 = arith.constant 2 : i32
    %c129_i32_0 = arith.constant 129 : i32
    %2 = arith.cmpi eq, %arg0, %c2_i32 : i32
    %3 = arith.select %2, %c129_i32_0, %1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c129_i32_1 = arith.constant 129 : i32
    %4 = arith.cmpi eq, %arg0, %c1_i32 : i32
    %5 = arith.select %4, %c129_i32_1, %3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c129_i32_2 = arith.constant 129 : i32
    %6 = arith.cmpi eq, %arg0, %c0_i32 : i32
    %7 = arith.select %6, %c129_i32_2, %5 : i32
    return %7 : i32
  }
  func.func @flash_attention_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i32 = arith.constant 4 : i32
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3145728 = arith.constant 3145728 : index
    %c2097152 = arith.constant 2097152 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map(%8)[%c2097152]
    %view_4 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map(%11)[%c3145728]
    %view_5 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c87_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %15 = arith.muli %arg13, %c3072_i32 : i32
      %16 = arith.muli %1, %c128_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview = memref.subview %reinterpret_cast[%18, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %27 = arith.muli %3, %c64_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %29
          %30 = arith.muli %arg14, %c8_i32 : i32
          %31 = arith.addi %30, %arg15 : i32
          %32 = arith.index_cast %arg15 : i32 to index
          %33 = arith.muli %31, %c256_i32 : i32
          %34 = arith.index_cast %33 : i32 to index
          %subview_6 = memref.subview %reinterpret_cast_3[%34, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_7 = memref.subview %view[%32, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_7 [[0, 1, 2], [3]] : memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_8 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_9 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_8 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_8, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_9 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_9 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %29 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          %30 = arith.addi %29, %c8_i64 : i64
          %31 = arith.remsi %30, %c16_i64 : i64
          %32 = arith.addi %29, %c8_i64 : i64
          %33 = arith.remsi %32, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %31
          %34 = arith.muli %arg14, %c8_i32 : i32
          %35 = arith.addi %34, %arg15 : i32
          %36 = arith.index_cast %arg15 : i32 to index
          %37 = arith.muli %35, %c256_i32 : i32
          %38 = arith.index_cast %37 : i32 to index
          %subview_6 = memref.subview %reinterpret_cast_2[%38, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_7 = memref.subview %view_4[%36, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_7 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_8 = memref.subview %view_5[%36, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_9 = memref.collapse_shape %subview_8 [[0, 1, 2], [3]] : memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_10 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_11 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_12 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_11 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_10 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_11, %alloc_10, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_12 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_12 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_9 : memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %33 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 8 : i32}
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %27
      }
      %19 = arith.muli %3, %c64_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c64_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
    }
    return
  }
  func.func @flash_attention_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i32 = arith.constant 4 : i32
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3145728 = arith.constant 3145728 : index
    %c2097152 = arith.constant 2097152 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply #map(%8)[%c2097152]
    %view_2 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply #map(%11)[%c3145728]
    %view_3 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c87_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %15 = arith.muli %arg13, %c3072_i32 : i32
      %16 = arith.muli %1, %c128_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %27 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %alloc_8 = memref.alloc() : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
        %27 = arith.muli %3, %c64_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          %30 = arith.addi %29, %c8_i64 : i64
          %31 = arith.remsi %30, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %29
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_9 = memref.subview %view[%32, 0, %28, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_10 = memref.subview %view_2[%32, 0, %28, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_11 = memref.collapse_shape %subview_10 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_12 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_13 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_14 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_15 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_16 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_17 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_12 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_12, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc, %alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_12, %alloc_16 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          memref.copy %alloc_13, %collapse_shape_11 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_18 = memref.subview %alloc_8[%33, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_19 = memref.collapse_shape %subview_18 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
          %34 = arith.index_cast %arg15 : i32 to index
          %subview_20 = memref.subview %alloc_8[%34, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_21 = memref.collapse_shape %subview_20 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_4, %collapse_shape_21 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_4, %alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_17, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %31 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          %30 = arith.addi %29, %c8_i64 : i64
          %31 = arith.remsi %30, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %31
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_9 = memref.subview %view_3[%32, 0, %28, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_10 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_10 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_11 = memref.subview %alloc_8[%33, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_12 = memref.collapse_shape %subview_11 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_5, %collapse_shape_12 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_5, %alloc_10 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %29 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_5, %alloc_4 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %19 = arith.muli %3, %c64_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c64_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %alloc_6[0, 0] [%24, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
      %subview_7 = memref.subview %reinterpret_cast[%26, 0] [%24, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %subview_7 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}



====== final npuir ======
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @flash_attention_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c4194304 = arith.constant 4194304 : index
    return %c4194304 : index
  }
  func.func @flash_attention_get_kernel_num_args() -> i32 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c4_i32 = arith.constant 4 : i32
    return %c4_i32 : i32
  }
  func.func @flash_attention_get_kernel_arg_type(%arg0: i32) -> i32 attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c-1_i32 = arith.constant -1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c129_i32 = arith.constant 129 : i32
    %0 = arith.cmpi eq, %arg0, %c3_i32 : i32
    %1 = arith.select %0, %c129_i32, %c-1_i32 : i32
    %c2_i32 = arith.constant 2 : i32
    %c129_i32_0 = arith.constant 129 : i32
    %2 = arith.cmpi eq, %arg0, %c2_i32 : i32
    %3 = arith.select %2, %c129_i32_0, %1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c129_i32_1 = arith.constant 129 : i32
    %4 = arith.cmpi eq, %arg0, %c1_i32 : i32
    %5 = arith.select %4, %c129_i32_1, %3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c129_i32_2 = arith.constant 129 : i32
    %6 = arith.cmpi eq, %arg0, %c0_i32 : i32
    %7 = arith.select %6, %c129_i32_2, %5 : i32
    return %7 : i32
  }
  func.func @flash_attention_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i32 = arith.constant 4 : i32
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3145728 = arith.constant 3145728 : index
    %c2097152 = arith.constant 2097152 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>(%8)[%c2097152]
    %view_4 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>(%11)[%c3145728]
    %view_5 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c87_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %15 = arith.muli %arg13, %c3072_i32 : i32
      %16 = arith.muli %1, %c128_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      %subview = memref.subview %reinterpret_cast[%18, 0] [128, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %27 = arith.muli %3, %c64_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %29
          %30 = arith.muli %arg14, %c8_i32 : i32
          %31 = arith.addi %30, %arg15 : i32
          %32 = arith.index_cast %arg15 : i32 to index
          %33 = arith.muli %31, %c256_i32 : i32
          %34 = arith.index_cast %33 : i32 to index
          %subview_6 = memref.subview %reinterpret_cast_3[%34, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_7 = memref.subview %view[%32, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_7 [[0, 1, 2], [3]] : memref<1x1x128x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_8 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_9 = memref.alloc() : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_8 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_8, %true, %c128, %c128, %c256 : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_9 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_9 : memref<128x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape : memref<128x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %29 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          %30 = arith.addi %29, %c8_i64 : i64
          %31 = arith.remsi %30, %c16_i64 : i64
          %32 = arith.addi %29, %c8_i64 : i64
          %33 = arith.remsi %32, %c16_i64 : i64
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %31
          %34 = arith.muli %arg14, %c8_i32 : i32
          %35 = arith.addi %34, %arg15 : i32
          %36 = arith.index_cast %arg15 : i32 to index
          %37 = arith.muli %35, %c256_i32 : i32
          %38 = arith.index_cast %37 : i32 to index
          %subview_6 = memref.subview %reinterpret_cast_2[%38, 0] [256, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_7 = memref.subview %view_4[%36, 0, 0, 0] [1, 1, 128, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_7 [[0, 1, 2], [3]] : memref<1x1x128x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_8 = memref.subview %view_5[%36, 0, 0, 0] [1, 1, 128, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_9 = memref.collapse_shape %subview_8 [[0, 1, 2], [3]] : memref<1x1x128x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_10 = memref.alloc() : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
          %alloc_11 = memref.alloc() : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
          %alloc_12 = memref.alloc() : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>
          hivm.hir.nd2nz {dst_continuous} ins(%collapse_shape : memref<128x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_11 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_10 : memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>) init_out_buffer = false
          hivm.hir.mmadL1 ins(%alloc_11, %alloc_10, %true, %c128, %c256, %c128 : memref<128x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>, memref<256x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_12 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>)
          hivm.hir.fixpipe {enable_nz2nd} ins(%alloc_12 : memref<128x128xf32, strided<[128, 1]>, #hivm.address_space<cc>>) outs(%collapse_shape_9 : memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %33 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 8 : i32}
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %27
      }
      %19 = arith.muli %3, %c64_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c64_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
    }
    return
  }
  func.func @flash_attention_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c4_i32 = arith.constant 4 : i32
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c8_i32 = arith.constant 8 : i32
    %c3145728 = arith.constant 3145728 : index
    %c2097152 = arith.constant 2097152 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c8192_i32 = arith.constant 8192 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c3072_i32 = arith.constant 3072 : i32
    %c24_i32 = arith.constant 24 : i32
    %c87_i32 = arith.constant 87 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [8192, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>(%5)[%c0]
    %view = memref.view %arg2[%6][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf32, #hivm.address_space<gm>>
    %7 = hivm.hir.get_block_idx -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.apply affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>(%8)[%c2097152]
    %view_2 = memref.view %arg2[%9][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x256xf16, #hivm.address_space<gm>>
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.index_cast %10 : i64 to index
    %12 = affine.apply affine_map<(d0)[s0] -> (d0 * 4194304 + s0)>(%11)[%c3145728]
    %view_3 = memref.view %arg2[%12][] : memref<?xi8, #hivm.address_space<gm>> to memref<8x2x128x128xf32, #hivm.address_space<gm>>
    %13 = arith.subi %c87_i32, %1 : i32
    %14 = arith.divsi %13, %c24_i32 : i32
    scf.for %arg13 = %c0_i32 to %14 step %c1_i32  : i32 {
      %alloc = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %15 = arith.muli %arg13, %c3072_i32 : i32
      %16 = arith.muli %1, %c128_i32 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
      scf.for %arg14 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %27 = arith.extsi %arg14 : i32 to i64
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %27 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
      }
      scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %alloc_8 = memref.alloc() : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>>
        %27 = arith.muli %3, %c64_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          %30 = arith.addi %29, %c8_i64 : i64
          %31 = arith.remsi %30, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %29
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_9 = memref.subview %view[%32, 0, %28, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf32, #hivm.address_space<gm>> to memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x256xf32, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_10 = memref.subview %view_2[%32, 0, %28, 0] [1, 1, 64, 256] [1, 1, 1, 1] : memref<8x2x128x256xf16, #hivm.address_space<gm>> to memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_11 = memref.collapse_shape %subview_10 [[0, 1, 2], [3]] : memref<1x1x64x256xf16, strided<[65536, 32768, 256, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_12 = memref.alloc() : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_13 = memref.alloc() : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
          %alloc_14 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_15 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_16 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          %alloc_17 = memref.alloc() : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_12 : memref<64x256xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_12, %cst_0 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, f32) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <max> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          hivm.hir.vmax ins(%alloc, %alloc_14 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vsub ins(%alloc_12, %alloc_16 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vexp ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vcast ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_13 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vreduce <sum> ins(%alloc_12 : memref<64x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) reduce_dims = [1]
          memref.copy %alloc_13, %collapse_shape_11 : memref<64x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<64x256xf16, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.vsub ins(%alloc, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_18 = memref.subview %alloc_8[%33, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_19 = memref.collapse_shape %subview_18 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
          %34 = arith.index_cast %arg15 : i32 to index
          %subview_20 = memref.subview %alloc_8[%34, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_21 = memref.collapse_shape %subview_20 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_4, %collapse_shape_21 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_4, %alloc_15 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_4 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_17 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%alloc_17, %alloc_16 : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc : memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %31 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
        scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
          %29 = arith.extsi %arg15 : i32 to i64
          %30 = arith.addi %29, %c8_i64 : i64
          %31 = arith.remsi %30, %c16_i64 : i64
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %31
          %32 = arith.index_cast %arg15 : i32 to index
          %subview_9 = memref.subview %view_3[%32, 0, %28, 0] [1, 1, 64, 128] [1, 1, 1, 1] : memref<8x2x128x128xf32, #hivm.address_space<gm>> to memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_9 [[0, 1, 2], [3]] : memref<1x1x64x128xf32, strided<[32768, 16384, 128, 1], offset: ?>, #hivm.address_space<gm>> into memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_10 = memref.alloc() : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          memref.copy %collapse_shape, %alloc_10 : memref<64x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          %33 = arith.index_cast %arg15 : i32 to index
          %subview_11 = memref.subview %alloc_8[%33, 0, 0] [1, 64, 1] [1, 1, 1] : memref<8x64x1xf32, strided<[64, 1, 1]>, #hivm.address_space<ub>> to memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %collapse_shape_12 = memref.collapse_shape %subview_11 [[0, 1], [2]] : memref<1x64x1xf32, strided<[64, 1, 1], offset: ?>, #hivm.address_space<ub>> into memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%alloc_5, %collapse_shape_12 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.vadd ins(%alloc_5, %alloc_10 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %29 syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
        }
      } {tilelangir.num_stages = 8 : i32}
      hivm.hir.vdiv ins(%alloc_5, %alloc_4 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<64x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) outs(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) broadcast = [1]
      hivm.hir.vcast ins(%alloc_5 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%alloc_6 : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>)
      %19 = arith.muli %3, %c64_i32 : i32
      %20 = arith.subi %c8192_i32, %19 : i32
      %21 = arith.subi %20, %16 : i32
      %22 = arith.subi %21, %15 : i32
      %23 = arith.minsi %22, %c64_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview = memref.subview %alloc_6[0, 0] [%24, 128] [1, 1] : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      %25 = arith.addi %17, %19 : i32
      %26 = arith.index_cast %25 : i32 to index
      %subview_7 = memref.subview %reinterpret_cast[%26, 0] [%24, 128] [1, 1] : memref<8192x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview, %subview_7 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    }
    return
  }
}[W423 20:19:55.674814063 ToKernelNpu.cpp:164] Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())

AscendNPU IR compile success: 
output:
tensor([[-0.0434,  0.0065,  0.0169,  ..., -0.0086, -0.0411,  0.0077],
        [ 0.0022,  0.0187,  0.0090,  ..., -0.0168, -0.0145, -0.0054],
        [-0.0285,  0.0068,  0.0259,  ..., -0.0107, -0.0491, -0.0075],
        ...,
        [-0.0198,  0.0368, -0.0188,  ..., -0.0251, -0.0406,  0.0435],
        [-0.0058,  0.0101, -0.0259,  ...,  0.0006, -0.0093,  0.0235],
        [-0.0233,  0.0013,  0.0186,  ...,  0.0107, -0.0558,  0.0039]],
       device='npu:10', dtype=torch.float16)
ref_output:
tensor([[-0.0434,  0.0065,  0.0169,  ..., -0.0086, -0.0411,  0.0077],
        [ 0.0022,  0.0187,  0.0090,  ..., -0.0168, -0.0145, -0.0054],
        [-0.0285,  0.0068,  0.0259,  ..., -0.0107, -0.0491, -0.0074],
        ...,
        [-0.0198,  0.0368, -0.0188,  ..., -0.0251, -0.0406,  0.0435],
        [-0.0058,  0.0101, -0.0259,  ...,  0.0006, -0.0093,  0.0235],
        [-0.0233,  0.0013,  0.0186,  ...,  0.0107, -0.0558,  0.0039]],
       device='npu:10', dtype=torch.float16)
All check passed.
