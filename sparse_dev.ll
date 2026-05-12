; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: alwaysinline
define private void @broadcast_scalar_float_to_1d(float %0, ptr addrspace(6) %1, ptr addrspace(6) %2, i64 %3, i64 %4, i64 %5) #0 {
  %7 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %1, 0
  %8 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %7, ptr addrspace(6) %2, 1
  %9 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %8, i64 %3, 2
  %10 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %9, i64 %4, 3, 0
  %11 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %10, i64 %5, 4, 0
  %12 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %11, ptr %12, align 8
  call void @_mlir_ciface_broadcast_scalar_float_to_1d(float %0, ptr %12)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_broadcast_scalar_float_to_1d(float, ptr) #0

; Function Attrs: alwaysinline
define private void @copy_ubuf_to_ubuf_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10, float %11) #0 {
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, ptr addrspace(6) %1, 1
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %2, 2
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %3, 3, 0
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, i64 %4, 4, 0
  %18 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, ptr %18, align 8
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, ptr addrspace(6) %6, 1
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, i64 %7, 2
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, i64 %8, 3, 0
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %9, 4, 0
  %24 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, ptr %24, align 8
  call void @_mlir_ciface_copy_ubuf_to_ubuf_1d_float(ptr %18, ptr %24, i32 %10, float %11)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_copy_ubuf_to_ubuf_1d_float(ptr, ptr, i32, float) #0

; Function Attrs: alwaysinline
define private void @load_gm_to_ubuf_1d_int32_t(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10, i32 %11, i64 %12) #0 {
  %14 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0
  %15 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %14, ptr addrspace(1) %1, 1
  %16 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %15, i64 %2, 2
  %17 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %16, i64 %3, 3, 0
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %17, i64 %4, 4, 0
  %19 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, ptr %19, align 8
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, ptr addrspace(6) %6, 1
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, i64 %7, 2
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %8, 3, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %9, 4, 0
  %25 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, ptr %25, align 8
  call void @_mlir_ciface_load_gm_to_ubuf_1d_int32_t(ptr %19, ptr %25, i32 %10, i32 %11, i64 %12)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_load_gm_to_ubuf_1d_int32_t(ptr, ptr, i32, i32, i64) #0

; Function Attrs: alwaysinline
define private void @broadcast_scalar_bfloat16_t_to_1d(bfloat %0, ptr addrspace(6) %1, ptr addrspace(6) %2, i64 %3, i64 %4, i64 %5) #0 {
  %7 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %1, 0
  %8 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %7, ptr addrspace(6) %2, 1
  %9 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %8, i64 %3, 2
  %10 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %9, i64 %4, 3, 0
  %11 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %10, i64 %5, 4, 0
  %12 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %11, ptr %12, align 8
  call void @_mlir_ciface_broadcast_scalar_bfloat16_t_to_1d(bfloat %0, ptr %12)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_broadcast_scalar_bfloat16_t_to_1d(bfloat, ptr) #0

; Function Attrs: alwaysinline
define private void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10, bfloat %11, i64 %12) #0 {
  %14 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0
  %15 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %14, ptr addrspace(1) %1, 1
  %16 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %15, i64 %2, 2
  %17 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %16, i64 %3, 3, 0
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %17, i64 %4, 4, 0
  %19 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, ptr %19, align 8
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, ptr addrspace(6) %6, 1
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, i64 %7, 2
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %8, 3, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %9, 4, 0
  %25 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, ptr %25, align 8
  call void @_mlir_ciface_load_gm_to_ubuf_1d_bfloat16_t(ptr %19, ptr %25, i32 %10, bfloat %11, i64 %12)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_load_gm_to_ubuf_1d_bfloat16_t(ptr, ptr, i32, bfloat, i64) #0

; Function Attrs: alwaysinline
define private void @store_ubuf_to_gm_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i64 %7, i64 %8, i64 %9, i32 %10) #0 {
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr addrspace(6) %1, 1
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, i64 %2, 2
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %3, 3, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %4, 4, 0
  %17 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr %17, align 8
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %5, 0
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, ptr addrspace(1) %6, 1
  %20 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, i64 %7, 2
  %21 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %20, i64 %8, 3, 0
  %22 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %21, i64 %9, 4, 0
  %23 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %22, ptr %23, align 8
  call void @_mlir_ciface_store_ubuf_to_gm_1d_float(ptr %17, ptr %23, i32 %10)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_store_ubuf_to_gm_1d_float(ptr, ptr, i32) #0

; Function Attrs: alwaysinline
define private void @store_ubuf_to_gm_1d_bfloat16_t(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i64 %7, i64 %8, i64 %9, i32 %10) #0 {
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr addrspace(6) %1, 1
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, i64 %2, 2
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %3, 3, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %4, 4, 0
  %17 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr %17, align 8
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %5, 0
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, ptr addrspace(1) %6, 1
  %20 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, i64 %7, 2
  %21 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %20, i64 %8, 3, 0
  %22 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %21, i64 %9, 4, 0
  %23 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %22, ptr %23, align 8
  call void @_mlir_ciface_store_ubuf_to_gm_1d_bfloat16_t(ptr %17, ptr %23, i32 %10)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_store_ubuf_to_gm_1d_bfloat16_t(ptr, ptr, i32) #0

; Function Attrs: alwaysinline
define private void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10, float %11, i64 %12) #0 {
  %14 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0
  %15 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %14, ptr addrspace(1) %1, 1
  %16 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %15, i64 %2, 2
  %17 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %16, i64 %3, 3, 0
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %17, i64 %4, 4, 0
  %19 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, ptr %19, align 8
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, ptr addrspace(6) %6, 1
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, i64 %7, 2
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %8, 3, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %9, 4, 0
  %25 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, ptr %25, align 8
  call void @_mlir_ciface_load_gm_to_ubuf_1d_float(ptr %19, ptr %25, i32 %10, float %11, i64 %12)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_load_gm_to_ubuf_1d_float(ptr, ptr, i32, float, i64) #0

; Function Attrs: alwaysinline
define private void @vmuls_vs_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, float %5, ptr addrspace(6) %6, ptr addrspace(6) %7, i64 %8, i64 %9, i64 %10, ptr addrspace(6) %11, ptr addrspace(6) %12, i64 %13, i64 %14, i64 %15) #0 {
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, ptr addrspace(6) %1, 1
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, i64 %2, 2
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %3, 3, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, i64 %4, 4, 0
  %22 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr %22, align 8
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %6, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, ptr addrspace(6) %7, 1
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %8, 2
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %9, 3, 0
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, i64 %10, 4, 0
  %28 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr %28, align 8
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %11, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, ptr addrspace(6) %12, 1
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %13, 2
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %14, 3, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, i64 %15, 4, 0
  %34 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr %34, align 8
  call void @_mlir_ciface_vmuls_vs_1d_float(ptr %22, float %5, ptr %28, ptr %34)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vmuls_vs_1d_float(ptr, float, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vadds_vs_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, float %5, ptr addrspace(6) %6, ptr addrspace(6) %7, i64 %8, i64 %9, i64 %10, ptr addrspace(6) %11, ptr addrspace(6) %12, i64 %13, i64 %14, i64 %15) #0 {
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, ptr addrspace(6) %1, 1
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, i64 %2, 2
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %3, 3, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, i64 %4, 4, 0
  %22 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr %22, align 8
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %6, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, ptr addrspace(6) %7, 1
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %8, 2
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %9, 3, 0
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, i64 %10, 4, 0
  %28 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr %28, align 8
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %11, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, ptr addrspace(6) %12, 1
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %13, 2
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %14, 3, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, i64 %15, 4, 0
  %34 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr %34, align 8
  call void @_mlir_ciface_vadds_vs_1d_float(ptr %22, float %5, ptr %28, ptr %34)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vadds_vs_1d_float(ptr, float, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @enablevc_reduce_max_ar_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr addrspace(6) %14, ptr addrspace(6) %15, i64 %16, i64 %17, i64 %18, float %19) #0 {
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, ptr addrspace(6) %1, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %22, i64 %2, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %24, i64 %5, 4, 0
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %25, i64 %4, 3, 1
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %26, i64 %6, 4, 1
  %28 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, ptr %28, align 8
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, ptr addrspace(6) %8, 1
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %9, 2
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, i64 %10, 3, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %32, i64 %12, 4, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %33, i64 %11, 3, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %34, i64 %13, 4, 1
  %36 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %35, ptr %36, align 8
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %14, 0
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, ptr addrspace(6) %15, 1
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %38, i64 %16, 2
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, i64 %17, 3, 0
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, i64 %18, 4, 0
  %42 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, ptr %42, align 8
  call void @_mlir_ciface_enablevc_reduce_max_ar_float(ptr %28, ptr %36, ptr %42, float %19)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_enablevc_reduce_max_ar_float(ptr, ptr, ptr, float) #0

; Function Attrs: alwaysinline
define private void @vsub_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14, ptr addrspace(6) %15, ptr addrspace(6) %16, i64 %17, i64 %18, i64 %19) #0 {
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %1, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %2, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %4, 4, 0
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr addrspace(6) %6, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, i64 %7, 2
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %8, 3, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %9, 4, 0
  %32 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, ptr %32, align 8
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr addrspace(6) %11, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %34, i64 %12, 2
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %35, i64 %13, 3, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %36, i64 %14, 4, 0
  %38 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, ptr %38, align 8
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %15, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, ptr addrspace(6) %16, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, i64 %17, 2
  %42 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, i64 %18, 3, 0
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %42, i64 %19, 4, 0
  %44 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %43, ptr %44, align 8
  call void @_mlir_ciface_vsub_1d_float(ptr %26, ptr %32, ptr %38, ptr %44)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vsub_1d_float(ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vexp_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14) #0 {
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr addrspace(6) %1, 1
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0
  %21 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, ptr %21, align 8
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, ptr addrspace(6) %6, 1
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %7, 2
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %8, 3, 0
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, i64 %9, 4, 0
  %27 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %26, ptr %27, align 8
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, ptr addrspace(6) %11, 1
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %12, 2
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %13, 3, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, i64 %14, 4, 0
  %33 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %32, ptr %33, align 8
  call void @_mlir_ciface_vexp_1d_float(ptr %21, ptr %27, ptr %33)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vexp_1d_float(ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vsub_2d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr addrspace(6) %14, ptr addrspace(6) %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, ptr addrspace(6) %21, ptr addrspace(6) %22, i64 %23, i64 %24, i64 %25) #0 {
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, ptr addrspace(6) %1, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, i64 %2, 2
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, i64 %3, 3, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %5, 4, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, i64 %4, 3, 1
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %32, i64 %6, 4, 1
  %34 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %33, ptr %34, align 8
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %35, ptr addrspace(6) %8, 1
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %36, i64 %9, 2
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %37, i64 %10, 3, 0
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %38, i64 %12, 4, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %39, i64 %11, 3, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %40, i64 %13, 4, 1
  %42 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %41, ptr %42, align 8
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %14, 0
  %44 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %43, ptr addrspace(6) %15, 1
  %45 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %44, i64 %16, 2
  %46 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %45, i64 %17, 3, 0
  %47 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %46, i64 %19, 4, 0
  %48 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %47, i64 %18, 3, 1
  %49 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %48, i64 %20, 4, 1
  %50 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %49, ptr %50, align 8
  %51 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %21, 0
  %52 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %51, ptr addrspace(6) %22, 1
  %53 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %52, i64 %23, 2
  %54 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %53, i64 %24, 3, 0
  %55 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %54, i64 %25, 4, 0
  %56 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %55, ptr %56, align 8
  call void @_mlir_ciface_vsub_2d_float(ptr %34, ptr %42, ptr %50, ptr %56)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vsub_2d_float(ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vmul_2d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr addrspace(6) %14, ptr addrspace(6) %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, ptr addrspace(6) %21, ptr addrspace(6) %22, i64 %23, i64 %24, i64 %25) #0 {
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, ptr addrspace(6) %1, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, i64 %2, 2
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, i64 %3, 3, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %5, 4, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, i64 %4, 3, 1
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %32, i64 %6, 4, 1
  %34 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %33, ptr %34, align 8
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %35, ptr addrspace(6) %8, 1
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %36, i64 %9, 2
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %37, i64 %10, 3, 0
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %38, i64 %12, 4, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %39, i64 %11, 3, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %40, i64 %13, 4, 1
  %42 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %41, ptr %42, align 8
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %14, 0
  %44 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %43, ptr addrspace(6) %15, 1
  %45 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %44, i64 %16, 2
  %46 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %45, i64 %17, 3, 0
  %47 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %46, i64 %19, 4, 0
  %48 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %47, i64 %18, 3, 1
  %49 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %48, i64 %20, 4, 1
  %50 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %49, ptr %50, align 8
  %51 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %21, 0
  %52 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %51, ptr addrspace(6) %22, 1
  %53 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %52, i64 %23, 2
  %54 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %53, i64 %24, 3, 0
  %55 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %54, i64 %25, 4, 0
  %56 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %55, ptr %56, align 8
  call void @_mlir_ciface_vmul_2d_float(ptr %34, ptr %42, ptr %50, ptr %56)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vmul_2d_float(ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @enablevc_reduce_sum_ar_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr addrspace(6) %14, ptr addrspace(6) %15, i64 %16, i64 %17, i64 %18, float %19) #0 {
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, ptr addrspace(6) %1, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %22, i64 %2, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %24, i64 %5, 4, 0
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %25, i64 %4, 3, 1
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %26, i64 %6, 4, 1
  %28 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, ptr %28, align 8
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, ptr addrspace(6) %8, 1
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %9, 2
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, i64 %10, 3, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %32, i64 %12, 4, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %33, i64 %11, 3, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %34, i64 %13, 4, 1
  %36 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %35, ptr %36, align 8
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %14, 0
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, ptr addrspace(6) %15, 1
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %38, i64 %16, 2
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, i64 %17, 3, 0
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, i64 %18, 4, 0
  %42 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, ptr %42, align 8
  call void @_mlir_ciface_enablevc_reduce_sum_ar_float(ptr %28, ptr %36, ptr %42, float %19)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_enablevc_reduce_sum_ar_float(ptr, ptr, ptr, float) #0

; Function Attrs: alwaysinline
define private void @vmul_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14, ptr addrspace(6) %15, ptr addrspace(6) %16, i64 %17, i64 %18, i64 %19) #0 {
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %1, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %2, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %4, 4, 0
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr addrspace(6) %6, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, i64 %7, 2
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %8, 3, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %9, 4, 0
  %32 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, ptr %32, align 8
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr addrspace(6) %11, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %34, i64 %12, 2
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %35, i64 %13, 3, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %36, i64 %14, 4, 0
  %38 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, ptr %38, align 8
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %15, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, ptr addrspace(6) %16, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, i64 %17, 2
  %42 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, i64 %18, 3, 0
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %42, i64 %19, 4, 0
  %44 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %43, ptr %44, align 8
  call void @_mlir_ciface_vmul_1d_float(ptr %26, ptr %32, ptr %38, ptr %44)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vmul_1d_float(ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vadd_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, ptr addrspace(6) %10, ptr addrspace(6) %11, i64 %12, i64 %13, i64 %14, ptr addrspace(6) %15, ptr addrspace(6) %16, i64 %17, i64 %18, i64 %19) #0 {
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %1, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %2, 2
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %4, 4, 0
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %27, ptr addrspace(6) %6, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %28, i64 %7, 2
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %29, i64 %8, 3, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %30, i64 %9, 4, 0
  %32 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %31, ptr %32, align 8
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %10, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %33, ptr addrspace(6) %11, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %34, i64 %12, 2
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %35, i64 %13, 3, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %36, i64 %14, 4, 0
  %38 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, ptr %38, align 8
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %15, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, ptr addrspace(6) %16, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, i64 %17, 2
  %42 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %41, i64 %18, 3, 0
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %42, i64 %19, 4, 0
  %44 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %43, ptr %44, align 8
  call void @_mlir_ciface_vadd_1d_float(ptr %26, ptr %32, ptr %38, ptr %44)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vadd_1d_float(ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10) #0 {
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr addrspace(6) %1, 1
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, i64 %2, 2
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %3, 3, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %4, 4, 0
  %17 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr %17, align 8
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %18, ptr addrspace(6) %6, 1
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %19, i64 %7, 2
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %20, i64 %8, 3, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, i64 %9, 4, 0
  %23 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, ptr %23, align 8
  call void @_mlir_ciface_vcast_float_to_bfloat16_t_1d_with_mode(ptr %17, ptr %23, i32 %10)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vcast_float_to_bfloat16_t_1d_with_mode(ptr, ptr, i32) #0

; Function Attrs: alwaysinline
define private void @vdiv_2d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr addrspace(6) %14, ptr addrspace(6) %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, ptr addrspace(6) %21, ptr addrspace(6) %22, i64 %23, i64 %24, i64 %25) #0 {
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, ptr addrspace(6) %1, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, i64 %2, 2
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, i64 %3, 3, 0
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %5, 4, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, i64 %4, 3, 1
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %32, i64 %6, 4, 1
  %34 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %33, ptr %34, align 8
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %35, ptr addrspace(6) %8, 1
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %36, i64 %9, 2
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %37, i64 %10, 3, 0
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %38, i64 %12, 4, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %39, i64 %11, 3, 1
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %40, i64 %13, 4, 1
  %42 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %41, ptr %42, align 8
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %14, 0
  %44 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %43, ptr addrspace(6) %15, 1
  %45 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %44, i64 %16, 2
  %46 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %45, i64 %17, 3, 0
  %47 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %46, i64 %19, 4, 0
  %48 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %47, i64 %18, 3, 1
  %49 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %48, i64 %20, 4, 1
  %50 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %49, ptr %50, align 8
  %51 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %21, 0
  %52 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %51, ptr addrspace(6) %22, 1
  %53 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %52, i64 %23, 2
  %54 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %53, i64 %24, 3, 0
  %55 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %54, i64 %25, 4, 0
  %56 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %55, ptr %56, align 8
  call void @_mlir_ciface_vdiv_2d_float(ptr %34, ptr %42, ptr %50, ptr %56)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vdiv_2d_float(ptr, ptr, ptr, ptr) #0

define dso_local void @sparseAttn_mix_aiv(i64 %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, ptr addrspace(1) %7, ptr addrspace(1) %8, ptr addrspace(1) %9, i32 %10, i32 %11, i32 %12, i32 %13, i32 %14, i32 %15, i32 %16, i32 %17) {
  call void @llvm.hivm.SET.FFTS.BASE.ADDR(i64 %0)
  %19 = call i64 @llvm.hivm.GET.CTRL()
  %20 = call i64 @llvm.hivm.SBITSET0(i64 %19, i64 56)
  call void @llvm.hivm.SET.CTRL(i64 %20)
  call void @llvm.hivm.SET.FFTS.BASE.ADDR(i64 %0)
  %21 = mul i32 %11, 2048
  %22 = sext i32 %21 to i64
  %23 = sext i32 %14 to i64
  %24 = mul i32 %11, %14
  %25 = sext i32 %24 to i64
  %26 = sext i32 %13 to i64
  %27 = mul i32 %11, %13
  %28 = sext i32 %27 to i64
  %29 = mul i32 %14, 32
  %30 = sext i32 %29 to i64
  %31 = mul i32 %11, %29
  %32 = sext i32 %31 to i64
  %33 = mul i32 %12, 32
  %34 = sext i32 %33 to i64
  %35 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %36 = trunc i64 %35 to i32
  call void @llvm.hivm.SET.CROSS.CORE(i64 4, i64 545)
  call void @llvm.hivm.SET.CROSS.CORE(i64 4, i64 1057)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 3)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @broadcast_scalar_float_to_1d(float 0.000000e+00, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 512, i64 1)
  call void @broadcast_scalar_float_to_1d(float 0.000000e+00, ptr addrspace(6) inttoptr (i64 2048 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2048 to ptr addrspace(6)), i64 0, i64 16, i64 1)
  call void @broadcast_scalar_float_to_1d(float 0xFFF0000000000000, ptr addrspace(6) inttoptr (i64 2112 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2112 to ptr addrspace(6)), i64 0, i64 16, i64 1)
  %37 = srem i32 %36, %11
  %38 = sext i32 %37 to i64
  %39 = add i32 %13, 31
  %40 = sdiv i32 %39, 32
  %41 = mul i64 %35, 5120
  %42 = getelementptr i8, ptr addrspace(1) %2, i64 %41
  %43 = mul i64 %35, 5120
  %44 = add i64 %43, 2048
  %45 = getelementptr i8, ptr addrspace(1) %2, i64 %44
  %46 = call i64 @llvm.hivm.GET.SUBBLOCKID()
  %47 = icmp eq i64 %46, 0
  %48 = mul i64 %35, 5120
  %49 = add i64 %48, 3072
  %50 = getelementptr i8, ptr addrspace(1) %2, i64 %49
  br label %51

51:                                               ; preds = %168, %18
  %52 = phi i32 [ %169, %168 ], [ 0, %18 ]
  %53 = icmp slt i32 %52, 4
  br i1 %53, label %54, label %170

54:                                               ; preds = %51
  %55 = sdiv i32 %36, %11
  %56 = sext i32 %55 to i64
  %57 = mul i32 %52, 16
  %58 = sext i32 %57 to i64
  %59 = icmp eq i32 %52, 0
  call void @llvm.hivm.BARRIER(i64 1)
  call void @copy_ubuf_to_ubuf_1d_float(ptr addrspace(6) inttoptr (i64 2112 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2112 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), i64 0, i64 16, i64 1, i32 0, float 0.000000e+00)
  call void @copy_ubuf_to_ubuf_1d_float(ptr addrspace(6) inttoptr (i64 2048 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2048 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 2432 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2432 to ptr addrspace(6)), i64 0, i64 16, i64 1, i32 0, float 0.000000e+00)
  call void @copy_ubuf_to_ubuf_1d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 512, i64 1, ptr addrspace(6) inttoptr (i64 12160 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12160 to ptr addrspace(6)), i64 0, i64 512, i64 1, i32 0, float 0.000000e+00)
  br label %60

60:                                               ; preds = %127, %54
  %61 = phi i32 [ %144, %127 ], [ 0, %54 ]
  %62 = phi { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } [ { ptr addrspace(6) inttoptr (i64 12160 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12160 to ptr addrspace(6)), i64 0, [2 x i64] [i64 16, i64 32], [2 x i64] [i64 32, i64 1] }, %127 ], [ { ptr addrspace(6) inttoptr (i64 12160 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12160 to ptr addrspace(6)), i64 0, [2 x i64] [i64 16, i64 32], [2 x i64] [i64 32, i64 1] }, %54 ]
  %63 = icmp slt i32 %61, %40
  br i1 %63, label %64, label %145

64:                                               ; preds = %60
  call void @llvm.hivm.WAIT.FLAG.DEV.REG(i64 0)
  br i1 %59, label %65, label %107

65:                                               ; preds = %64
  %66 = mul i32 %61, 32
  %67 = sext i32 %66 to i64
  %68 = mul i64 %56, %28
  %69 = mul i64 %38, %26
  %70 = add i64 %68, %69
  %71 = add i64 %70, %67
  call void @llvm.hivm.BARRIER(i64 4)
  call void @load_gm_to_ubuf_1d_int32_t(ptr addrspace(1) %7, ptr addrspace(1) %7, i64 %71, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 2496 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2496 to ptr addrspace(6)), i64 0, i64 32, i64 1, i32 0, i32 0, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @broadcast_scalar_bfloat16_t_to_1d(bfloat 0xR0000, ptr addrspace(6) inttoptr (i64 2624 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2624 to ptr addrspace(6)), i64 0, i64 1024, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @broadcast_scalar_float_to_1d(float 0.000000e+00, ptr addrspace(6) inttoptr (i64 4672 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4672 to ptr addrspace(6)), i64 0, i64 32, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 0, i64 0)
  %72 = sub i32 %13, %66
  %73 = icmp slt i32 %72, 32
  %74 = select i1 %73, i32 %72, i32 32
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %75

75:                                               ; preds = %91, %65
  %76 = phi i32 [ %92, %91 ], [ 0, %65 ]
  %77 = icmp slt i32 %76, %74
  br i1 %77, label %78, label %93

78:                                               ; preds = %75
  %79 = sext i32 %76 to i64
  %80 = getelementptr i32, ptr addrspace(6) inttoptr (i64 2496 to ptr addrspace(6)), i64 %79
  %81 = load i32, ptr addrspace(6) %80, align 4
  %82 = icmp ne i32 %81, -1
  br i1 %82, label %83, label %91

83:                                               ; preds = %78
  %84 = add i64 0, %79
  %85 = getelementptr float, ptr addrspace(6) inttoptr (i64 4672 to ptr addrspace(6)), i64 %84
  store float 1.000000e+00, ptr addrspace(6) %85, align 4
  %86 = sext i32 %81 to i64
  %87 = mul i64 %56, %34
  %88 = mul i64 %86, 32
  %89 = add i64 %87, %88
  %90 = mul i64 %79, 32
  call void @llvm.hivm.BARRIER(i64 4)
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %89, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 2624 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2624 to ptr addrspace(6)), i64 %90, i64 32, i64 1, i32 0, bfloat 0xR0000, i64 0)
  br label %91

91:                                               ; preds = %83, %78
  %92 = add i32 %76, 1
  br label %75

93:                                               ; preds = %75
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  br i1 %47, label %94, label %99

94:                                               ; preds = %93
  %95 = mul i64 %56, %25
  %96 = mul i64 %38, %23
  %97 = add i64 %95, %96
  %98 = add i64 %97, %67
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) inttoptr (i64 4672 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4672 to ptr addrspace(6)), i64 0, i64 32, i64 1, ptr addrspace(1) %9, ptr addrspace(1) %9, i64 %98, i64 32, i64 1, i32 0)
  br label %99

99:                                               ; preds = %94, %93
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 5, i64 0)
  br i1 %47, label %100, label %106

100:                                              ; preds = %99
  %101 = mul i64 %56, %32
  %102 = mul i64 %38, %30
  %103 = add i64 %101, %102
  %104 = mul i64 %67, 32
  %105 = add i64 %103, %104
  call void @store_ubuf_to_gm_1d_bfloat16_t(ptr addrspace(6) inttoptr (i64 2624 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2624 to ptr addrspace(6)), i64 0, i64 1024, i64 1, ptr addrspace(1) %8, ptr addrspace(1) %8, i64 %105, i64 1024, i64 1, i32 0)
  br label %106

106:                                              ; preds = %100, %99
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  br label %108

107:                                              ; preds = %64
  br label %108

108:                                              ; preds = %106, %107
  %109 = phi { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } [ { ptr addrspace(6) inttoptr (i64 2176 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2176 to ptr addrspace(6)), i64 0, [2 x i64] [i64 1, i64 32], [2 x i64] [i64 32, i64 1] }, %107 ], [ { ptr addrspace(6) inttoptr (i64 4672 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4672 to ptr addrspace(6)), i64 0, [2 x i64] [i64 1, i64 32], [2 x i64] [i64 32, i64 1] }, %106 ]
  br label %110

110:                                              ; preds = %108
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 2)
  call void @llvm.hivm.SET.CROSS.CORE(i64 5, i64 289)
  %111 = mul i32 %61, 32
  %112 = sext i32 %111 to i64
  call void @llvm.hivm.WAIT.FLAG.DEV.REG(i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %2, ptr addrspace(1) %42, i64 0, i64 512, i64 1, ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), i64 0, i64 512, i64 1, i32 0, float 0.000000e+00, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.SET.CROSS.CORE(i64 4, i64 545)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vmuls_vs_1d_float(ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), i64 0, i64 512, i64 1, float 0x3FC6A09E60000000, ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), i64 0, i64 512, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadds_vs_1d_float(ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), i64 0, i64 16, i64 1, float 0.000000e+00, ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @enablevc_reduce_max_ar_float(ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), i64 0, i64 16, i64 32, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), i64 0, i64 16, i64 1, i64 1, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1, float 0xFFF0000000000000)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vsub_1d_float(ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vexp_1d_float(ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @vsub_2d_float(ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), i64 0, i64 16, i64 32, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), i64 0, i64 16, i64 1, i64 1, i64 1, ptr addrspace(6) inttoptr (i64 6912 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 6912 to ptr addrspace(6)), i64 0, i64 16, i64 32, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 8960 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8960 to ptr addrspace(6)), i64 0, i64 128, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vexp_1d_float(ptr addrspace(6) inttoptr (i64 6912 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 6912 to ptr addrspace(6)), i64 0, i64 512, i64 1, ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), i64 0, i64 512, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  %113 = mul i64 %56, %25
  %114 = mul i64 %38, %23
  %115 = add i64 %113, %114
  %116 = add i64 %115, %112
  %117 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %109, 0
  %118 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %109, 1
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 2)
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %9, ptr addrspace(1) %9, i64 %116, i64 32, i64 1, ptr addrspace(6) %117, ptr addrspace(6) %118, i64 0, i64 32, i64 1, i32 0, float 0.000000e+00, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  %119 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %109, 0
  %120 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %109, 1
  %121 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %109, 2
  %122 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %109, 3, 0
  %123 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %109, 3, 1
  %124 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %109, 4, 0
  %125 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %109, 4, 1
  call void @vmul_2d_float(ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), i64 0, i64 16, i64 32, i64 32, i64 1, ptr addrspace(6) %119, ptr addrspace(6) %120, i64 %121, i64 %122, i64 %123, i64 %124, i64 %125, ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), i64 0, i64 16, i64 32, i64 32, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @enablevc_reduce_sum_ar_float(ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), i64 0, i64 16, i64 32, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 9472 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 9472 to ptr addrspace(6)), i64 0, i64 16, i64 1, i64 1, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1, float 0.000000e+00)
  call void @vmul_1d_float(ptr addrspace(6) inttoptr (i64 2432 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2432 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 9536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 9536 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) inttoptr (i64 9536 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 9536 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 9472 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 9472 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 2432 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2432 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), i64 0, i64 512, i64 1, ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), i64 0, i64 512, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.DEV.REG(i64 3)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 1)
  br i1 %47, label %126, label %127

126:                                              ; preds = %110
  call void @store_ubuf_to_gm_1d_bfloat16_t(ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 4800 to ptr addrspace(6)), i64 0, i64 512, i64 1, ptr addrspace(1) %2, ptr addrspace(1) %45, i64 0, i64 512, i64 1, i32 0)
  br label %127

127:                                              ; preds = %126, %110
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.SET.CROSS.CORE(i64 5, i64 289)
  %128 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 0
  %129 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 1
  %130 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 2
  %131 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 3, 0
  %132 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 3, 1
  %133 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 4, 0
  %134 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 4, 1
  %135 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 0
  %136 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 1
  %137 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 2
  %138 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 3, 0
  %139 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 3, 1
  %140 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 4, 0
  %141 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 4, 1
  call void @vmul_2d_float(ptr addrspace(6) %128, ptr addrspace(6) %129, i64 %130, i64 %131, i64 %132, i64 %133, i64 %134, ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 6848 to ptr addrspace(6)), i64 0, i64 16, i64 1, i64 1, i64 1, ptr addrspace(6) %135, ptr addrspace(6) %136, i64 %137, i64 %138, i64 %139, i64 %140, i64 %141, ptr addrspace(6) inttoptr (i64 9600 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 9600 to ptr addrspace(6)), i64 0, i64 128, i64 1)
  call void @llvm.hivm.WAIT.FLAG.DEV.REG(i64 1)
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %2, ptr addrspace(1) %50, i64 0, i64 512, i64 1, ptr addrspace(6) inttoptr (i64 10112 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 10112 to ptr addrspace(6)), i64 0, i64 512, i64 1, i32 0, float 0.000000e+00, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 2)
  call void @llvm.hivm.SET.CROSS.CORE(i64 4, i64 1057)
  %142 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 0
  %143 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 1
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 2)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) inttoptr (i64 10112 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 10112 to ptr addrspace(6)), i64 0, i64 512, i64 1, ptr addrspace(6) %142, ptr addrspace(6) %143, i64 0, i64 512, i64 1, ptr addrspace(6) inttoptr (i64 12160 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 12160 to ptr addrspace(6)), i64 0, i64 512, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  %144 = add i32 %61, 1
  br label %60

145:                                              ; preds = %60
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 3)
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %6, ptr addrspace(1) %6, i64 %58, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 2304 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2304 to ptr addrspace(6)), i64 0, i64 16, i64 1, i32 0, float 0.000000e+00, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 3)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 3)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vsub_1d_float(ptr addrspace(6) inttoptr (i64 2304 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2304 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 3)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vexp_1d_float(ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_1d_float(ptr addrspace(6) inttoptr (i64 2432 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2432 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2368 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) inttoptr (i64 2432 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2432 to ptr addrspace(6)), i64 0, i64 16, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  %146 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 0
  %147 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 1
  %148 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 2
  %149 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 3, 0
  %150 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 3, 1
  %151 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 4, 0
  %152 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 4, 1
  %153 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 0
  %154 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 1
  %155 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 2
  %156 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 3, 0
  %157 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 3, 1
  %158 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 4, 0
  %159 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 4, 1
  call void @vdiv_2d_float(ptr addrspace(6) %146, ptr addrspace(6) %147, i64 %148, i64 %149, i64 %150, i64 %151, i64 %152, ptr addrspace(6) inttoptr (i64 2432 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 2432 to ptr addrspace(6)), i64 0, i64 16, i64 1, i64 1, i64 1, ptr addrspace(6) %153, ptr addrspace(6) %154, i64 %155, i64 %156, i64 %157, i64 %158, i64 %159, ptr addrspace(6) inttoptr (i64 14208 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 14208 to ptr addrspace(6)), i64 0, i64 128, i64 1)
  %160 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 0
  %161 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %62, 1
  call void @llvm.hivm.BARRIER(i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) %160, ptr addrspace(6) %161, i64 0, i64 512, i64 1, ptr addrspace(6) inttoptr (i64 14720 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 14720 to ptr addrspace(6)), i64 0, i64 512, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 2)
  br i1 %47, label %162, label %168

162:                                              ; preds = %145
  %163 = mul i64 %56, %22
  %164 = mul i64 %38, 2048
  %165 = add i64 %163, %164
  %166 = mul i64 %58, 32
  %167 = add i64 %165, %166
  call void @store_ubuf_to_gm_1d_bfloat16_t(ptr addrspace(6) inttoptr (i64 14720 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 14720 to ptr addrspace(6)), i64 0, i64 512, i64 1, ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %167, i64 512, i64 1, i32 0)
  br label %168

168:                                              ; preds = %162, %145
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  %169 = add i32 %52, 1
  br label %51

170:                                              ; preds = %51
  call void @llvm.hivm.BARRIER(i64 6)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 3)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.DEV.REG(i64 0)
  call void @llvm.hivm.WAIT.FLAG.DEV.REG(i64 3)
  ret void
}

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.FFTS.BASE.ADDR(i64) #1

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #1

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET0(i64, i64) #2

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #1

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #2

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CROSS.CORE(i64, i64) #1

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.SUBBLOCKID() #2

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #1

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.WAIT.FLAG.DEV.REG(i64) #1

attributes #0 = { alwaysinline }
attributes #1 = { nounwind  inaccessiblememonly }
attributes #2 = { nounwind readnone  }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @sparseAttn_mix_aiv, !"kernel", i32 1}

