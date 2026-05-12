; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define dso_local void @sparseAttnMix_mix_aic(i64 %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, ptr addrspace(1) %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, i32 %14) {
  %16 = call i64 @llvm.hivm.GET.CTRL()
  %17 = call i64 @llvm.hivm.SBITSET0(i64 %16, i64 56)
  call void @llvm.hivm.SET.CTRL(i64 %17)
  call void @llvm.hivm.SET.FFTS.BASE.ADDR(i64 %0)
  %18 = mul i32 %9, 2048
  %19 = sext i32 %18 to i64
  %20 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %21 = trunc i64 %20 to i32
  %22 = mul i64 %20, 90112
  %23 = getelementptr i8, ptr addrspace(1) %2, i64 %22
  %24 = mul i64 %20, 90112
  %25 = add i64 %24, 8192
  %26 = getelementptr i8, ptr addrspace(1) %2, i64 %25
  %27 = mul i64 %20, 90112
  %28 = add i64 %27, 40960
  %29 = getelementptr i8, ptr addrspace(1) %2, i64 %28
  %30 = mul i64 %20, 90112
  %31 = add i64 %30, 57344
  %32 = getelementptr i8, ptr addrspace(1) %2, i64 %31
  %33 = sdiv i32 %21, %9
  %34 = sext i32 %33 to i64
  %35 = srem i32 %21, %9
  %36 = sext i32 %35 to i64
  %37 = mul i64 %34, %19
  %38 = mul i64 %36, 2048
  %39 = add i64 %37, %38
  call void @llvm.hivm.SET.FLAG.IMM(i64 2, i64 3, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 2, i64 3, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 3, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 3, i64 4, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 10, i64 2, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 3, i64 4, i64 2)
  call void @llvm.hivm.SET.FLAG.IMM(i64 10, i64 2, i64 1)
  call void @nd2nz_bfloat16_t(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %39, i64 64, i64 32, i64 32, i64 1, ptr addrspace(2) null, ptr addrspace(2) null, i64 0, i64 2, i64 4, i64 16, i64 16, i64 1024, i64 256, i64 16, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 3, i64 0)
  %40 = add i32 %11, 31
  %41 = sdiv i32 %40, 32
  %42 = sdiv i32 %41, 2
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 3, i64 0)
  br label %43

43:                                               ; preds = %86, %15
  %44 = phi i32 [ %87, %86 ], [ 0, %15 ]
  %45 = icmp slt i32 %44, %42
  br i1 %45, label %46, label %88

46:                                               ; preds = %43
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 0)
  br label %47

47:                                               ; preds = %50, %46
  %48 = phi i32 [ %64, %50 ], [ 0, %46 ]
  %49 = icmp slt i32 %48, 2
  br i1 %49, label %50, label %65

50:                                               ; preds = %47
  %51 = sext i32 %48 to i64
  %52 = sext i32 %44 to i64
  %53 = mul i64 %52, 2
  %54 = add i64 %51, %53
  %55 = sext i32 %48 to i64
  %56 = add i64 %55, 2
  %57 = srem i64 %56, 16
  call void @llvm.hivm.WAIT.FLAG.DEV.REG(i64 %55)
  %58 = sext i32 %48 to i64
  %59 = mul i64 %58, 2048
  %60 = mul i64 %58, 1024
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 1)
  call void @nd2nz_bfloat16_t(ptr addrspace(1) %2, ptr addrspace(1) %23, i64 %59, i64 32, i64 32, i64 32, i64 1, ptr addrspace(2) inttoptr (i64 8192 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 8192 to ptr addrspace(2)), i64 %60, i64 2, i64 2, i64 16, i64 16, i64 512, i64 256, i64 16, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 3, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 10, i64 2, i64 0)
  call void @mma_tile_bfloat16_t_to_float_tb(ptr addrspace(2) null, ptr addrspace(2) null, i64 0, i64 2, i64 4, i64 16, i64 16, i64 1024, i64 256, i64 16, i64 1, ptr addrspace(2) inttoptr (i64 8192 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 8192 to ptr addrspace(2)), i64 %60, i64 2, i64 2, i64 16, i64 16, i64 512, i64 256, i64 16, i64 1, i1 true, i64 64, i64 32, i64 32, ptr addrspace(5) null, ptr addrspace(5) null, i64 0, i64 2, i64 4, i64 16, i64 16, i64 1024, i64 256, i64 16, i64 1, i64 -1, i64 1, i64 -1, i64 1, i64 %54, i64 0, i64 1, i8 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 2, i64 10, i64 0)
  %61 = mul i64 %58, 4096
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 2, i64 10, i64 0)
  call void @fixpipe_nz2nd_float_to_float_4d_to_2d(ptr addrspace(5) null, ptr addrspace(5) null, i64 0, i64 2, i64 4, i64 16, i64 16, i64 1024, i64 256, i64 16, i64 1, ptr addrspace(1) %2, ptr addrspace(1) %26, i64 %61, i64 64, i64 32, i64 32, i64 1, i64 0, i64 0, i1 false, i8 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 10, i64 2, i64 0)
  %62 = shl i64 %57, 8
  %63 = or i64 %62, 33
  call void @llvm.hivm.SET.CROSS.CORE(i64 10, i64 %63)
  %64 = add i32 %48, 1
  br label %47

65:                                               ; preds = %47
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 3, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 3, i64 2)
  br label %66

66:                                               ; preds = %69, %65
  %67 = phi i32 [ %85, %69 ], [ 0, %65 ]
  %68 = icmp slt i32 %67, 2
  br i1 %68, label %69, label %86

69:                                               ; preds = %66
  %70 = sext i32 %67 to i64
  %71 = sext i32 %44 to i64
  %72 = mul i64 %71, 2
  %73 = add i64 %70, %72
  %74 = sext i32 %67 to i64
  %75 = add i64 %74, 2
  %76 = srem i64 %75, 16
  %77 = add i64 %74, 4
  %78 = srem i64 %77, 16
  call void @llvm.hivm.WAIT.FLAG.DEV.REG(i64 %76)
  %79 = sext i32 %67 to i64
  %80 = mul i64 %79, 4096
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 2)
  call void @nd2nz_bfloat16_t(ptr addrspace(1) %2, ptr addrspace(1) %29, i64 %80, i64 64, i64 32, i64 32, i64 1, ptr addrspace(2) inttoptr (i64 4096 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 4096 to ptr addrspace(2)), i64 0, i64 2, i64 4, i64 16, i64 16, i64 1024, i64 256, i64 16, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 3, i64 3)
  %81 = mul i64 %79, 1024
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 10, i64 2, i64 1)
  call void @mma_tile_bfloat16_t_to_float(ptr addrspace(2) inttoptr (i64 4096 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 4096 to ptr addrspace(2)), i64 0, i64 2, i64 4, i64 16, i64 16, i64 1024, i64 256, i64 16, i64 1, ptr addrspace(2) inttoptr (i64 8192 to ptr addrspace(2)), ptr addrspace(2) inttoptr (i64 8192 to ptr addrspace(2)), i64 %81, i64 2, i64 2, i64 16, i64 16, i64 512, i64 256, i64 16, i64 1, i1 true, i64 64, i64 32, i64 32, ptr addrspace(5) inttoptr (i64 8192 to ptr addrspace(5)), ptr addrspace(5) inttoptr (i64 8192 to ptr addrspace(5)), i64 0, i64 2, i64 4, i64 16, i64 16, i64 1024, i64 256, i64 16, i64 1, i64 3, i64 -1, i64 2, i64 -1, i64 %73, i64 0, i64 1, i8 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 2, i64 10, i64 1)
  %82 = mul i64 %79, 4096
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 2, i64 10, i64 1)
  call void @fixpipe_nz2nd_float_to_float_4d_to_2d(ptr addrspace(5) inttoptr (i64 8192 to ptr addrspace(5)), ptr addrspace(5) inttoptr (i64 8192 to ptr addrspace(5)), i64 0, i64 2, i64 4, i64 16, i64 16, i64 1024, i64 256, i64 16, i64 1, ptr addrspace(1) %2, ptr addrspace(1) %32, i64 %82, i64 64, i64 32, i64 32, i64 1, i64 0, i64 0, i1 false, i8 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 10, i64 2, i64 1)
  %83 = shl i64 %78, 8
  %84 = or i64 %83, 33
  call void @llvm.hivm.SET.CROSS.CORE(i64 10, i64 %84)
  %85 = add i32 %67, 1
  br label %66

86:                                               ; preds = %66
  call void @llvm.hivm.SET.FLAG.IMM(i64 3, i64 4, i64 0)
  %87 = add i32 %44, 1
  br label %43

88:                                               ; preds = %43
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 2, i64 3, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 2, i64 3, i64 1)
  call void @llvm.hivm.BARRIER(i64 6)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 10, i64 2, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 3, i64 4, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 10, i64 2, i64 1)
  ret void
}

; Function Attrs: alwaysinline
define private void @nd2nz_bfloat16_t(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr addrspace(2) %7, ptr addrspace(2) %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17) #0 {
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(1) %0, 0
  %20 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %19, ptr addrspace(1) %1, 1
  %21 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %20, i64 %2, 2
  %22 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %21, i64 %3, 3, 0
  %23 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %22, i64 %5, 4, 0
  %24 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %23, i64 %4, 3, 1
  %25 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %24, i64 %6, 4, 1
  %26 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %25, ptr %26, align 8
  %27 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %7, 0
  %28 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %27, ptr addrspace(2) %8, 1
  %29 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %28, i64 %9, 2
  %30 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %29, i64 %10, 3, 0
  %31 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %30, i64 %14, 4, 0
  %32 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %31, i64 %11, 3, 1
  %33 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %32, i64 %15, 4, 1
  %34 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %33, i64 %12, 3, 2
  %35 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %34, i64 %16, 4, 2
  %36 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %35, i64 %13, 3, 3
  %37 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %36, i64 %17, 4, 3
  %38 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %37, ptr %38, align 8
  call void @_mlir_ciface_nd2nz_bfloat16_t(ptr %26, ptr %38)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_nd2nz_bfloat16_t(ptr, ptr) #0

; Function Attrs: alwaysinline
define private void @mma_tile_bfloat16_t_to_float_tb(ptr addrspace(2) %0, ptr addrspace(2) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr addrspace(2) %11, ptr addrspace(2) %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i1 %22, i64 %23, i64 %24, i64 %25, ptr addrspace(5) %26, ptr addrspace(5) %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36, i64 %37, i64 %38, i64 %39, i64 %40, i64 %41, i64 %42, i64 %43, i8 %44) #0 {
  %46 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %0, 0
  %47 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %46, ptr addrspace(2) %1, 1
  %48 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %47, i64 %2, 2
  %49 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %48, i64 %3, 3, 0
  %50 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %49, i64 %7, 4, 0
  %51 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %50, i64 %4, 3, 1
  %52 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %51, i64 %8, 4, 1
  %53 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %52, i64 %5, 3, 2
  %54 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %53, i64 %9, 4, 2
  %55 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %54, i64 %6, 3, 3
  %56 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %55, i64 %10, 4, 3
  %57 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %56, ptr %57, align 8
  %58 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %11, 0
  %59 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %58, ptr addrspace(2) %12, 1
  %60 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %59, i64 %13, 2
  %61 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %60, i64 %14, 3, 0
  %62 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %61, i64 %18, 4, 0
  %63 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %62, i64 %15, 3, 1
  %64 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %63, i64 %19, 4, 1
  %65 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %64, i64 %16, 3, 2
  %66 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %65, i64 %20, 4, 2
  %67 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %66, i64 %17, 3, 3
  %68 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %67, i64 %21, 4, 3
  %69 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %68, ptr %69, align 8
  %70 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(5) %26, 0
  %71 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %70, ptr addrspace(5) %27, 1
  %72 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %71, i64 %28, 2
  %73 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %72, i64 %29, 3, 0
  %74 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %73, i64 %33, 4, 0
  %75 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %74, i64 %30, 3, 1
  %76 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %75, i64 %34, 4, 1
  %77 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %76, i64 %31, 3, 2
  %78 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %77, i64 %35, 4, 2
  %79 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %78, i64 %32, 3, 3
  %80 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %79, i64 %36, 4, 3
  %81 = alloca { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %80, ptr %81, align 8
  call void @_mlir_ciface_mma_tile_bfloat16_t_to_float_tb(ptr %57, ptr %69, i1 %22, i64 %23, i64 %24, i64 %25, ptr %81, i64 %37, i64 %38, i64 %39, i64 %40, i64 %41, i64 %42, i64 %43, i8 %44)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_mma_tile_bfloat16_t_to_float_tb(ptr, ptr, i1, i64, i64, i64, ptr, i64, i64, i64, i64, i64, i64, i64, i8) #0

; Function Attrs: alwaysinline
define private void @fixpipe_nz2nd_float_to_float_4d_to_2d(ptr addrspace(5) %0, ptr addrspace(5) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr addrspace(1) %11, ptr addrspace(1) %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i1 %20, i8 %21) #0 {
  %23 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(5) %0, 0
  %24 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %23, ptr addrspace(5) %1, 1
  %25 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %24, i64 %2, 2
  %26 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %25, i64 %3, 3, 0
  %27 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %26, i64 %7, 4, 0
  %28 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %27, i64 %4, 3, 1
  %29 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %28, i64 %8, 4, 1
  %30 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %29, i64 %5, 3, 2
  %31 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %30, i64 %9, 4, 2
  %32 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %31, i64 %6, 3, 3
  %33 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %32, i64 %10, 4, 3
  %34 = alloca { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %33, ptr %34, align 8
  %35 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } undef, ptr addrspace(1) %11, 0
  %36 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %35, ptr addrspace(1) %12, 1
  %37 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %36, i64 %13, 2
  %38 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %37, i64 %14, 3, 0
  %39 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %38, i64 %16, 4, 0
  %40 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %39, i64 %15, 3, 1
  %41 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %40, i64 %17, 4, 1
  %42 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr addrspace(1), ptr addrspace(1), i64, [2 x i64], [2 x i64] } %41, ptr %42, align 8
  call void @_mlir_ciface_fixpipe_nz2nd_float_to_float_4d_to_2d(ptr %34, ptr %42, i64 %18, i64 %19, i1 %20, i8 %21)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_fixpipe_nz2nd_float_to_float_4d_to_2d(ptr, ptr, i64, i64, i1, i8) #0

; Function Attrs: alwaysinline
define private void @mma_tile_bfloat16_t_to_float(ptr addrspace(2) %0, ptr addrspace(2) %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr addrspace(2) %11, ptr addrspace(2) %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i1 %22, i64 %23, i64 %24, i64 %25, ptr addrspace(5) %26, ptr addrspace(5) %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36, i64 %37, i64 %38, i64 %39, i64 %40, i64 %41, i64 %42, i64 %43, i8 %44) #0 {
  %46 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %0, 0
  %47 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %46, ptr addrspace(2) %1, 1
  %48 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %47, i64 %2, 2
  %49 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %48, i64 %3, 3, 0
  %50 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %49, i64 %7, 4, 0
  %51 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %50, i64 %4, 3, 1
  %52 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %51, i64 %8, 4, 1
  %53 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %52, i64 %5, 3, 2
  %54 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %53, i64 %9, 4, 2
  %55 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %54, i64 %6, 3, 3
  %56 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %55, i64 %10, 4, 3
  %57 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %56, ptr %57, align 8
  %58 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(2) %11, 0
  %59 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %58, ptr addrspace(2) %12, 1
  %60 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %59, i64 %13, 2
  %61 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %60, i64 %14, 3, 0
  %62 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %61, i64 %18, 4, 0
  %63 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %62, i64 %15, 3, 1
  %64 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %63, i64 %19, 4, 1
  %65 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %64, i64 %16, 3, 2
  %66 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %65, i64 %20, 4, 2
  %67 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %66, i64 %17, 3, 3
  %68 = insertvalue { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %67, i64 %21, 4, 3
  %69 = alloca { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(2), ptr addrspace(2), i64, [4 x i64], [4 x i64] } %68, ptr %69, align 8
  %70 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } undef, ptr addrspace(5) %26, 0
  %71 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %70, ptr addrspace(5) %27, 1
  %72 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %71, i64 %28, 2
  %73 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %72, i64 %29, 3, 0
  %74 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %73, i64 %33, 4, 0
  %75 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %74, i64 %30, 3, 1
  %76 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %75, i64 %34, 4, 1
  %77 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %76, i64 %31, 3, 2
  %78 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %77, i64 %35, 4, 2
  %79 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %78, i64 %32, 3, 3
  %80 = insertvalue { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %79, i64 %36, 4, 3
  %81 = alloca { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr addrspace(5), ptr addrspace(5), i64, [4 x i64], [4 x i64] } %80, ptr %81, align 8
  call void @_mlir_ciface_mma_tile_bfloat16_t_to_float(ptr %57, ptr %69, i1 %22, i64 %23, i64 %24, i64 %25, ptr %81, i64 %37, i64 %38, i64 %39, i64 %40, i64 %41, i64 %42, i64 %43, i8 %44)
  ret void
}

; Function Attrs: alwaysinline
declare dso_local void @_mlir_ciface_mma_tile_bfloat16_t_to_float(ptr, ptr, i1, i64, i64, i64, ptr, i64, i64, i64, i64, i64, i64, i64, i8) #0

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
!1 = !{ptr @sparseAttnMix_mix_aic, !"kernel", i32 1}

