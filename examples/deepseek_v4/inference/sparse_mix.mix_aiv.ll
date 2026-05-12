; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: alwaysinline
define private void @broadcast_scalar_float_to_2d(float %0, ptr addrspace(6) %1, ptr addrspace(6) %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7) #0 {
  %9 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %1, 0
  %10 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %9, ptr addrspace(6) %2, 1
  %11 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %10, i64 %3, 2
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %11, i64 %4, 3, 0
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %12, i64 %6, 4, 0
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %13, i64 %5, 3, 1
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %14, i64 %7, 4, 1
  %16 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %15, ptr %16, align 8
  call void @_mlir_ciface_broadcast_scalar_float_to_2d(float %0, ptr %16)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_broadcast_scalar_float_to_2d(float, ptr) #0

; Function Attrs: alwaysinline
define private void @broadcast_scalar_bfloat16_t_to_2d(bfloat %0, ptr addrspace(6) %1, ptr addrspace(6) %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7) #0 {
  %9 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %1, 0
  %10 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %9, ptr addrspace(6) %2, 1
  %11 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %10, i64 %3, 2
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %11, i64 %4, 3, 0
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %12, i64 %6, 4, 0
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %13, i64 %5, 3, 1
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %14, i64 %7, 4, 1
  %16 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %15, ptr %16, align 8
  call void @_mlir_ciface_broadcast_scalar_bfloat16_t_to_2d(bfloat %0, ptr %16)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_broadcast_scalar_bfloat16_t_to_2d(bfloat, ptr) #0

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
define private void @copy_ubuf_to_ubuf_2d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i32 %14, float %15) #0 {
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %17, ptr addrspace(6) %1, 1
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %18, i64 %2, 2
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %19, i64 %3, 3, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %20, i64 %5, 4, 0
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, i64 %4, 3, 1
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %22, i64 %6, 4, 1
  %24 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %23, ptr %24, align 8
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %25, ptr addrspace(6) %8, 1
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %26, i64 %9, 2
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, i64 %10, 3, 0
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, i64 %12, 4, 0
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, i64 %11, 3, 1
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %13, 4, 1
  %32 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, ptr %32, align 8
  call void @_mlir_ciface_copy_ubuf_to_ubuf_2d_float(ptr %24, ptr %32, i32 %14, float %15)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_copy_ubuf_to_ubuf_2d_float(ptr, ptr, i32, float) #0

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
define private void @vexp_2d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr addrspace(6) %14, ptr addrspace(6) %15, i64 %16, i64 %17, i64 %18) #0 {
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %20, ptr addrspace(6) %1, 1
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, i64 %2, 2
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %22, i64 %3, 3, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %23, i64 %5, 4, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %24, i64 %4, 3, 1
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %25, i64 %6, 4, 1
  %27 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %26, ptr %27, align 8
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, ptr addrspace(6) %8, 1
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, i64 %9, 2
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %10, 3, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, i64 %12, 4, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %32, i64 %11, 3, 1
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %33, i64 %13, 4, 1
  %35 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %34, ptr %35, align 8
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %14, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %36, ptr addrspace(6) %15, 1
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, i64 %16, 2
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %38, i64 %17, 3, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, i64 %18, 4, 0
  %41 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, ptr %41, align 8
  call void @_mlir_ciface_vexp_2d_float(ptr %27, ptr %35, ptr %41)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vexp_2d_float(ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @broadcast_last_axis_align_2d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr addrspace(6) %14, ptr addrspace(6) %15, i64 %16, i64 %17, i64 %18) #0 {
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %20, ptr addrspace(6) %1, 1
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, i64 %2, 2
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %22, i64 %3, 3, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %23, i64 %5, 4, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %24, i64 %4, 3, 1
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %25, i64 %6, 4, 1
  %27 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %26, ptr %27, align 8
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, ptr addrspace(6) %8, 1
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, i64 %9, 2
  %31 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, i64 %10, 3, 0
  %32 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %31, i64 %12, 4, 0
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %32, i64 %11, 3, 1
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %33, i64 %13, 4, 1
  %35 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %34, ptr %35, align 8
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %14, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %36, ptr addrspace(6) %15, 1
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %37, i64 %16, 2
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %38, i64 %17, 3, 0
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %39, i64 %18, 4, 0
  %41 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %40, ptr %41, align 8
  call void @_mlir_ciface_broadcast_last_axis_align_2d_float(ptr %27, ptr %35, ptr %41)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_broadcast_last_axis_align_2d_float(ptr, ptr, ptr) #0

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
define private void @broadcast_first_axis_align_2d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) #0 {
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %15, ptr addrspace(6) %1, 1
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %16, i64 %2, 2
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %17, i64 %3, 3, 0
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %18, i64 %5, 4, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %19, i64 %4, 3, 1
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %20, i64 %6, 4, 1
  %22 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, ptr %22, align 8
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %23, ptr addrspace(6) %8, 1
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %24, i64 %9, 2
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %25, i64 %10, 3, 0
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %26, i64 %12, 4, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, i64 %11, 3, 1
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, i64 %13, 4, 1
  %30 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, ptr %30, align 8
  call void @_mlir_ciface_broadcast_first_axis_align_2d_float(ptr %22, ptr %30)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_broadcast_first_axis_align_2d_float(ptr, ptr) #0

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
define private void @vadd_2d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr addrspace(6) %14, ptr addrspace(6) %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, ptr addrspace(6) %21, ptr addrspace(6) %22, i64 %23, i64 %24, i64 %25) #0 {
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
  call void @_mlir_ciface_vadd_2d_float(ptr %34, ptr %42, ptr %50, ptr %56)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vadd_2d_float(ptr, ptr, ptr, ptr) #0

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
define private void @vmul_3d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr addrspace(6) %9, ptr addrspace(6) %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, ptr addrspace(6) %18, ptr addrspace(6) %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, ptr addrspace(6) %27, ptr addrspace(6) %28, i64 %29, i64 %30, i64 %31) #0 {
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } undef, ptr addrspace(6) %0, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %33, ptr addrspace(6) %1, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %34, i64 %2, 2
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %35, i64 %3, 3, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %36, i64 %6, 4, 0
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %37, i64 %4, 3, 1
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %38, i64 %7, 4, 1
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %39, i64 %5, 3, 2
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %40, i64 %8, 4, 2
  %42 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %41, ptr %42, align 8
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } undef, ptr addrspace(6) %9, 0
  %44 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %43, ptr addrspace(6) %10, 1
  %45 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %44, i64 %11, 2
  %46 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %45, i64 %12, 3, 0
  %47 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %46, i64 %15, 4, 0
  %48 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %47, i64 %13, 3, 1
  %49 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %48, i64 %16, 4, 1
  %50 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %49, i64 %14, 3, 2
  %51 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %50, i64 %17, 4, 2
  %52 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %51, ptr %52, align 8
  %53 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } undef, ptr addrspace(6) %18, 0
  %54 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %53, ptr addrspace(6) %19, 1
  %55 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %54, i64 %20, 2
  %56 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %55, i64 %21, 3, 0
  %57 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %56, i64 %24, 4, 0
  %58 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %57, i64 %22, 3, 1
  %59 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %58, i64 %25, 4, 1
  %60 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %59, i64 %23, 3, 2
  %61 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %60, i64 %26, 4, 2
  %62 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %61, ptr %62, align 8
  %63 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %27, 0
  %64 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %63, ptr addrspace(6) %28, 1
  %65 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %64, i64 %29, 2
  %66 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %65, i64 %30, 3, 0
  %67 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %66, i64 %31, 4, 0
  %68 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %67, ptr %68, align 8
  call void @_mlir_ciface_vmul_3d_float(ptr %42, ptr %52, ptr %62, ptr %68)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vmul_3d_float(ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vdiv_3d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr addrspace(6) %9, ptr addrspace(6) %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, ptr addrspace(6) %18, ptr addrspace(6) %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, ptr addrspace(6) %27, ptr addrspace(6) %28, i64 %29, i64 %30, i64 %31) #0 {
  %33 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } undef, ptr addrspace(6) %0, 0
  %34 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %33, ptr addrspace(6) %1, 1
  %35 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %34, i64 %2, 2
  %36 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %35, i64 %3, 3, 0
  %37 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %36, i64 %6, 4, 0
  %38 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %37, i64 %4, 3, 1
  %39 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %38, i64 %7, 4, 1
  %40 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %39, i64 %5, 3, 2
  %41 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %40, i64 %8, 4, 2
  %42 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %41, ptr %42, align 8
  %43 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } undef, ptr addrspace(6) %9, 0
  %44 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %43, ptr addrspace(6) %10, 1
  %45 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %44, i64 %11, 2
  %46 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %45, i64 %12, 3, 0
  %47 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %46, i64 %15, 4, 0
  %48 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %47, i64 %13, 3, 1
  %49 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %48, i64 %16, 4, 1
  %50 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %49, i64 %14, 3, 2
  %51 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %50, i64 %17, 4, 2
  %52 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %51, ptr %52, align 8
  %53 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } undef, ptr addrspace(6) %18, 0
  %54 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %53, ptr addrspace(6) %19, 1
  %55 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %54, i64 %20, 2
  %56 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %55, i64 %21, 3, 0
  %57 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %56, i64 %24, 4, 0
  %58 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %57, i64 %22, 3, 1
  %59 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %58, i64 %25, 4, 1
  %60 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %59, i64 %23, 3, 2
  %61 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %60, i64 %26, 4, 2
  %62 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [3 x i64], [3 x i64] } %61, ptr %62, align 8
  %63 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %27, 0
  %64 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %63, ptr addrspace(6) %28, 1
  %65 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %64, i64 %29, 2
  %66 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %65, i64 %30, 3, 0
  %67 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %66, i64 %31, 4, 0
  %68 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %67, ptr %68, align 8
  call void @_mlir_ciface_vdiv_3d_float(ptr %42, ptr %52, ptr %62, ptr %68)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vdiv_3d_float(ptr, ptr, ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @vcast_float_to_bfloat16_t_2d_with_mode(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(6) %7, ptr addrspace(6) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i32 %14) #0 {
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %0, 0
  %17 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %16, ptr addrspace(6) %1, 1
  %18 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %19, i64 %5, 4, 0
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %20, i64 %4, 3, 1
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %21, i64 %6, 4, 1
  %23 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %22, ptr %23, align 8
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(6) %7, 0
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %24, ptr addrspace(6) %8, 1
  %26 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %25, i64 %9, 2
  %27 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %26, i64 %10, 3, 0
  %28 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %27, i64 %12, 4, 0
  %29 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %28, i64 %11, 3, 1
  %30 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %29, i64 %13, 4, 1
  %31 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(6), ptr addrspace(6), i64, [2 x i64], [2 x i64] } %30, ptr %31, align 8
  call void @_mlir_ciface_vcast_float_to_bfloat16_t_2d_with_mode(ptr %23, ptr %31, i32 %14)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_vcast_float_to_bfloat16_t_2d_with_mode(ptr, ptr, i32) #0

define dso_local void @sparseAttnMix_mix_aiv(i64 %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, ptr addrspace(1) %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, i32 %14) {
  %16 = call i64 @llvm.hivm.GET.CTRL()
  %17 = call i64 @llvm.hivm.SBITSET0(i64 %16, i64 56)
  call void @llvm.hivm.SET.CTRL(i64 %17)
  call void @llvm.hivm.SET.FFTS.BASE.ADDR(i64 %0)
  %18 = mul i32 %9, 2048
  %19 = sext i32 %18 to i64
  %20 = sext i32 %11 to i64
  %21 = mul i32 %9, %11
  %22 = sext i32 %21 to i64
  %23 = mul i32 %10, 32
  %24 = sext i32 %23 to i64
  %25 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %26 = trunc i64 %25 to i32
  %27 = call i64 @llvm.hivm.GET.SUBBLOCKID()
  %28 = trunc i64 %27 to i32
  %29 = mul i64 %25, 90112
  %30 = getelementptr i8, ptr addrspace(1) %2, i64 %29
  %31 = mul i64 %25, 90112
  %32 = add i64 %31, 8192
  %33 = getelementptr i8, ptr addrspace(1) %2, i64 %32
  %34 = mul i64 %25, 90112
  %35 = add i64 %34, 40960
  %36 = getelementptr i8, ptr addrspace(1) %2, i64 %35
  %37 = mul i64 %25, 90112
  %38 = add i64 %37, 57344
  %39 = getelementptr i8, ptr addrspace(1) %2, i64 %38
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 2)
  call void @broadcast_scalar_float_to_2d(float 0.000000e+00, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 1024, i64 1, i64 8, i64 1)
  call void @broadcast_scalar_float_to_2d(float 0.000000e+00, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1)
  call void @broadcast_scalar_float_to_2d(float 0xFFF0000000000000, ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1)
  %40 = sdiv i32 %26, %9
  %41 = sext i32 %40 to i64
  %42 = srem i32 %26, %9
  %43 = sext i32 %42 to i64
  %44 = add i32 %11, 31
  %45 = sdiv i32 %44, 32
  %46 = sdiv i32 %45, 2
  br label %47

47:                                               ; preds = %132, %15
  %48 = phi i32 [ %133, %132 ], [ 0, %15 ]
  %49 = icmp slt i32 %48, %46
  br i1 %49, label %50, label %134

50:                                               ; preds = %47
  br label %51

51:                                               ; preds = %91, %50
  %52 = phi i32 [ %95, %91 ], [ 0, %50 ]
  %53 = icmp slt i32 %52, 2
  br i1 %53, label %54, label %96

54:                                               ; preds = %51
  %55 = sext i32 %52 to i64
  %56 = mul i32 %48, 2
  %57 = add i32 %56, %52
  %58 = sext i32 %52 to i64
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @broadcast_scalar_bfloat16_t_to_2d(bfloat 0xR0000, ptr addrspace(6) inttoptr (i64 72704 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 72704 to ptr addrspace(6)), i64 0, i64 1024, i64 1, i64 16, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  %59 = mul i64 %58, 32
  %60 = mul i64 %58, 32
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 1, i64 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @broadcast_scalar_float_to_1d(float 0.000000e+00, ptr addrspace(6) inttoptr (i64 155776 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 155776 to ptr addrspace(6)), i64 %60, i64 32, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 0, i64 0)
  %61 = mul i32 %57, 32
  %62 = sub i32 %11, %61
  %63 = icmp slt i32 %62, 32
  %64 = select i1 %63, i32 %62, i32 32
  %65 = sext i32 %64 to i64
  %66 = sext i32 %61 to i64
  %67 = mul i64 %41, %22
  %68 = mul i64 %43, %20
  %69 = add i64 %67, %68
  %70 = add i64 %69, %66
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 4, i64 0)
  call void @llvm.hivm.BARRIER(i64 4)
  call void @load_gm_to_ubuf_1d_int32_t(ptr addrspace(1) %7, ptr addrspace(1) %7, i64 %70, i64 %65, i64 1, ptr addrspace(6) inttoptr (i64 115712 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 115712 to ptr addrspace(6)), i64 0, i64 %65, i64 8, i32 0, i32 0, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 0, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  br label %71

71:                                               ; preds = %89, %54
  %72 = phi i32 [ %90, %89 ], [ 0, %54 ]
  %73 = icmp slt i32 %72, %64
  br i1 %73, label %74, label %91

74:                                               ; preds = %71
  %75 = sext i32 %72 to i64
  %76 = mul i64 %75, 8
  %77 = getelementptr i32, ptr addrspace(6) inttoptr (i64 115712 to ptr addrspace(6)), i64 %76
  %78 = load i32, ptr addrspace(6) %77, align 4
  %79 = icmp ne i32 %78, -1
  br i1 %79, label %80, label %89

80:                                               ; preds = %74
  %81 = getelementptr float, ptr addrspace(6) inttoptr (i64 155776 to ptr addrspace(6)), i64 %59
  %82 = add i64 0, %75
  %83 = getelementptr float, ptr addrspace(6) %81, i64 %82
  store float 1.000000e+00, ptr addrspace(6) %83, align 4
  %84 = sext i32 %78 to i64
  %85 = mul i64 %41, %24
  %86 = mul i64 %84, 32
  %87 = add i64 %85, %86
  %88 = mul i64 %75, 512
  call void @llvm.hivm.BARRIER(i64 4)
  call void @load_gm_to_ubuf_1d_bfloat16_t(ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %87, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 72704 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 72704 to ptr addrspace(6)), i64 %88, i64 32, i64 16, i32 2, bfloat 0xR0000, i64 0)
  br label %89

89:                                               ; preds = %80, %74
  %90 = add i32 %72, 1
  br label %71

91:                                               ; preds = %71
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 4, i64 0)
  %92 = mul i64 %58, 2048
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 5, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @store_ubuf_to_gm_1d_bfloat16_t(ptr addrspace(6) inttoptr (i64 72704 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 72704 to ptr addrspace(6)), i64 0, i64 1024, i64 16, ptr addrspace(1) %2, ptr addrspace(1) %30, i64 %92, i64 1024, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  %93 = shl i64 %55, 8
  %94 = or i64 %93, 33
  call void @llvm.hivm.SET.CROSS.CORE(i64 5, i64 %94)
  %95 = add i32 %52, 1
  br label %51

96:                                               ; preds = %51
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 1, i64 1)
  %97 = mul i32 %28, 32
  %98 = sext i32 %97 to i64
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 1, i64 1)
  br label %99

99:                                               ; preds = %102, %96
  %100 = phi i32 [ %117, %102 ], [ 0, %96 ]
  %101 = icmp slt i32 %100, 2
  br i1 %101, label %102, label %118

102:                                              ; preds = %99
  %103 = sext i32 %100 to i64
  %104 = add i64 %103, 2
  %105 = srem i64 %104, 16
  call void @llvm.hivm.WAIT.FLAG.DEV.REG(i64 %105)
  %106 = sext i32 %100 to i64
  %107 = mul i64 %106, 4096
  %108 = mul i64 %98, 32
  %109 = add i64 %107, %108
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %2, ptr addrspace(1) %33, i64 %109, i64 1024, i64 1, ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), i64 0, i64 1024, i64 1, i32 0, float 0.000000e+00, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @copy_ubuf_to_ubuf_2d_float(ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 36864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 36864 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, i32 0, float 0.000000e+00)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vmuls_vs_1d_float(ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), i64 0, i64 1024, i64 1, float 0x3FC6A09E60000000, ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), i64 0, i64 1024, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @enablevc_reduce_max_ar_float(ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), i64 0, i64 32, i64 32, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1, float 0xFFF0000000000000)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vsub_2d_float(ptr addrspace(6) inttoptr (i64 36864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 36864 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 109568 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 109568 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  %110 = mul i64 %106, 256
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vexp_2d_float(ptr addrspace(6) inttoptr (i64 109568 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 109568 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 153728 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 153728 to ptr addrspace(6)), i64 %110, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @copy_ubuf_to_ubuf_2d_float(ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 110592 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 110592 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, i32 0, float 0.000000e+00)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @broadcast_last_axis_align_2d_float(ptr addrspace(6) inttoptr (i64 110592 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 110592 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 149632 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 149632 to ptr addrspace(6)), i64 0, i64 32, i64 32, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 162176 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 162176 to ptr addrspace(6)), i64 0, i64 256, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vsub_1d_float(ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), i64 0, i64 1024, i64 1, ptr addrspace(6) inttoptr (i64 149632 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 149632 to ptr addrspace(6)), i64 0, i64 1024, i64 1, ptr addrspace(6) inttoptr (i64 111616 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 111616 to ptr addrspace(6)), i64 0, i64 1024, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vexp_1d_float(ptr addrspace(6) inttoptr (i64 111616 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 111616 to ptr addrspace(6)), i64 0, i64 1024, i64 1, ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), i64 0, i64 1024, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  %111 = mul i64 %106, 32
  call void @copy_ubuf_to_ubuf_1d_float(ptr addrspace(6) inttoptr (i64 155776 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 155776 to ptr addrspace(6)), i64 %111, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 149504 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 149504 to ptr addrspace(6)), i64 0, i64 32, i64 1, i32 0, float 0.000000e+00)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @broadcast_first_axis_align_2d_float(ptr addrspace(6) inttoptr (i64 149504 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 149504 to ptr addrspace(6)), i64 0, i64 1, i64 32, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 105472 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 105472 to ptr addrspace(6)), i64 0, i64 32, i64 32, i64 32, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_1d_float(ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), i64 0, i64 1024, i64 1, ptr addrspace(6) inttoptr (i64 105472 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 105472 to ptr addrspace(6)), i64 0, i64 1024, i64 1, ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), i64 0, i64 1024, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @enablevc_reduce_sum_ar_float(ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), i64 0, i64 32, i64 32, i64 32, i64 1, ptr addrspace(6) inttoptr (i64 161152 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 161152 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1, float 0.000000e+00)
  call void @vmul_2d_float(ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 153728 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 153728 to ptr addrspace(6)), i64 %110, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 160128 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 160128 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_2d_float(ptr addrspace(6) inttoptr (i64 160128 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 160128 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 161152 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 161152 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @vcast_float_to_bfloat16_t_1d_with_mode(ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 156032 to ptr addrspace(6)), i64 0, i64 1024, i64 1, ptr addrspace(6) inttoptr (i64 70656 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 70656 to ptr addrspace(6)), i64 0, i64 1024, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  %112 = mul i64 %106, 4096
  %113 = mul i64 %98, 32
  %114 = add i64 %112, %113
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 1)
  call void @store_ubuf_to_gm_1d_bfloat16_t(ptr addrspace(6) inttoptr (i64 70656 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 70656 to ptr addrspace(6)), i64 0, i64 1024, i64 1, ptr addrspace(1) %2, ptr addrspace(1) %36, i64 %114, i64 1024, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  %115 = shl i64 %105, 8
  %116 = or i64 %115, 33
  call void @llvm.hivm.SET.CROSS.CORE(i64 5, i64 %116)
  %117 = add i32 %100, 1
  br label %99

118:                                              ; preds = %99
  br label %119

119:                                              ; preds = %122, %118
  %120 = phi i32 [ %131, %122 ], [ 0, %118 ]
  %121 = icmp slt i32 %120, 2
  br i1 %121, label %122, label %132

122:                                              ; preds = %119
  %123 = sext i32 %120 to i64
  %124 = add i64 %123, 4
  %125 = srem i64 %124, 16
  call void @llvm.hivm.WAIT.FLAG.DEV.REG(i64 %125)
  %126 = sext i32 %120 to i64
  %127 = mul i64 %126, 4096
  %128 = mul i64 %98, 32
  %129 = add i64 %127, %128
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 2)
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %2, ptr addrspace(1) %39, i64 %129, i64 1024, i64 1, ptr addrspace(6) inttoptr (i64 116736 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 116736 to ptr addrspace(6)), i64 0, i64 1024, i64 8, i32 0, float 0.000000e+00, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 1)
  %130 = mul i64 %126, 256
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vmul_3d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 32, i64 32, i64 1, i64 256, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 153728 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 153728 to ptr addrspace(6)), i64 %130, i64 32, i64 1, i64 1, i64 8, i64 1, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 32, i64 32, i64 1, i64 256, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 1)
  call void @vadd_2d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 1024, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 116736 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 116736 to ptr addrspace(6)), i64 0, i64 1024, i64 1, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 1024, i64 1, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 2)
  %131 = add i32 %120, 1
  br label %119

132:                                              ; preds = %119
  %133 = add i32 %48, 1
  br label %47

134:                                              ; preds = %47
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 0, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 0, i64 1)
  br label %135

135:                                              ; preds = %155, %134
  %136 = phi i32 [ %156, %155 ], [ 0, %134 ]
  %137 = icmp slt i32 %136, 32
  br i1 %137, label %138, label %157

138:                                              ; preds = %135
  %139 = icmp slt i32 %28, 2
  br i1 %139, label %140, label %150

140:                                              ; preds = %138
  %141 = mul i32 %28, 32
  %142 = add i32 %141, %136
  %143 = sext i32 %142 to i64
  %144 = getelementptr float, ptr addrspace(1) %6, i64 %143
  %145 = load float, ptr addrspace(1) %144, align 4
  %146 = sext i32 %136 to i64
  %147 = mul i64 %146, 8
  %148 = add i64 %147, 0
  %149 = getelementptr float, ptr addrspace(6) inttoptr (i64 36864 to ptr addrspace(6)), i64 %148
  store float %145, ptr addrspace(6) %149, align 4
  br label %155

150:                                              ; preds = %138
  %151 = sext i32 %136 to i64
  %152 = mul i64 %151, 8
  %153 = add i64 %152, 0
  %154 = getelementptr float, ptr addrspace(6) inttoptr (i64 36864 to ptr addrspace(6)), i64 %153
  store float 0xFFF0000000000000, ptr addrspace(6) %154, align 4
  br label %155

155:                                              ; preds = %140, %150
  %156 = add i32 %136, 1
  br label %135

157:                                              ; preds = %135
  call void @llvm.hivm.SET.FLAG.IMM(i64 0, i64 1, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 1, i64 2)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vsub_2d_float(ptr addrspace(6) inttoptr (i64 36864 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 36864 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 33792 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 34816 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 34816 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vexp_2d_float(ptr addrspace(6) inttoptr (i64 34816 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 34816 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 35840 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 35840 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vadd_2d_float(ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 35840 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 35840 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vdiv_3d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 32, i64 32, i64 1, i64 256, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), i64 0, i64 32, i64 1, i64 1, i64 8, i64 1, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 32, i64 32, i64 1, i64 256, i64 8, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 0, i64 1)
  call void @llvm.hivm.BARRIER(i64 1)
  call void @vcast_float_to_bfloat16_t_2d_with_mode(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 1024, i64 1, i64 8, i64 1, ptr addrspace(6) inttoptr (i64 37888 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 37888 to ptr addrspace(6)), i64 0, i64 1024, i64 1, i64 16, i64 1, i32 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 2)
  %158 = mul i32 %28, 32
  %159 = sub i32 64, %158
  %160 = icmp slt i32 %159, 32
  %161 = select i1 %160, i32 %159, i32 32
  %162 = sext i32 %161 to i64
  %163 = sext i32 %158 to i64
  %164 = mul i64 %162, 32
  %165 = mul i64 %41, %19
  %166 = mul i64 %43, 2048
  %167 = add i64 %165, %166
  %168 = mul i64 %163, 32
  %169 = add i64 %167, %168
  %170 = mul i64 %162, 32
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 2)
  call void @store_ubuf_to_gm_1d_bfloat16_t(ptr addrspace(6) inttoptr (i64 37888 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 37888 to ptr addrspace(6)), i64 0, i64 %164, i64 16, ptr addrspace(1) %5, ptr addrspace(1) %5, i64 %169, i64 %170, i64 1, i32 0)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 0, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 2)
  ret void
}

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #1

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET0(i64, i64) #2

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #1

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.FFTS.BASE.ADDR(i64) #1

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #2

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.SUBBLOCKID() #2

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #1

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.WAIT.FLAG.DEV.REG(i64) #1

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CROSS.CORE(i64, i64) #1

attributes #0 = { alwaysinline }
attributes #1 = { nounwind  inaccessiblememonly }
attributes #2 = { nounwind readnone  }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @sparseAttnMix_mix_aiv, !"kernel", i32 1}

