; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_reduce_fusion(ptr noalias align 16 dereferenceable(262144) %0, ptr noalias align 256 dereferenceable(1024) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = udiv i32 %3, 32
  %6 = mul i32 %5, 256
  %7 = mul i32 %4, 2048
  %8 = add i32 %6, %7
  %9 = urem i32 %3, 32
  %10 = add i32 %8, %9
  %11 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %10
  %12 = load double, ptr %11, align 8, !invariant.load !3
  %13 = call double @region_0_1_reduce_sum_5_0(double 0.000000e+00, double %12)
  %14 = add i32 %10, 32
  %15 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %14
  %16 = load double, ptr %15, align 8, !invariant.load !3
  %17 = call double @region_0_1_reduce_sum_5_0(double %13, double %16)
  %18 = add i32 %10, 64
  %19 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %18
  %20 = load double, ptr %19, align 8, !invariant.load !3
  %21 = call double @region_0_1_reduce_sum_5_0(double %17, double %20)
  %22 = add i32 %10, 96
  %23 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %22
  %24 = load double, ptr %23, align 8, !invariant.load !3
  %25 = call double @region_0_1_reduce_sum_5_0(double %21, double %24)
  %26 = add i32 %10, 128
  %27 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %26
  %28 = load double, ptr %27, align 8, !invariant.load !3
  %29 = call double @region_0_1_reduce_sum_5_0(double %25, double %28)
  %30 = add i32 %10, 160
  %31 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %30
  %32 = load double, ptr %31, align 8, !invariant.load !3
  %33 = call double @region_0_1_reduce_sum_5_0(double %29, double %32)
  %34 = add i32 %10, 192
  %35 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %34
  %36 = load double, ptr %35, align 8, !invariant.load !3
  %37 = call double @region_0_1_reduce_sum_5_0(double %33, double %36)
  %38 = add i32 %10, 224
  %39 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %38
  %40 = load double, ptr %39, align 8, !invariant.load !3
  %41 = call double @region_0_1_reduce_sum_5_0(double %37, double %40)
  %42 = bitcast double %41 to i64
  %43 = bitcast i64 %42 to <2 x i32>
  %44 = extractelement <2 x i32> %43, i32 0
  %45 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %44, i32 16, i32 31)
  %46 = insertelement <2 x i32> undef, i32 %45, i32 0
  %47 = extractelement <2 x i32> %43, i32 1
  %48 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %47, i32 16, i32 31)
  %49 = insertelement <2 x i32> %46, i32 %48, i32 1
  %50 = bitcast <2 x i32> %49 to double
  %51 = call double @region_0_1_reduce_sum_5_0(double %41, double %50)
  %52 = bitcast double %51 to i64
  %53 = bitcast i64 %52 to <2 x i32>
  %54 = extractelement <2 x i32> %53, i32 0
  %55 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %54, i32 8, i32 31)
  %56 = insertelement <2 x i32> undef, i32 %55, i32 0
  %57 = extractelement <2 x i32> %53, i32 1
  %58 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %57, i32 8, i32 31)
  %59 = insertelement <2 x i32> %56, i32 %58, i32 1
  %60 = bitcast <2 x i32> %59 to double
  %61 = call double @region_0_1_reduce_sum_5_0(double %51, double %60)
  %62 = bitcast double %61 to i64
  %63 = bitcast i64 %62 to <2 x i32>
  %64 = extractelement <2 x i32> %63, i32 0
  %65 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %64, i32 4, i32 31)
  %66 = insertelement <2 x i32> undef, i32 %65, i32 0
  %67 = extractelement <2 x i32> %63, i32 1
  %68 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %67, i32 4, i32 31)
  %69 = insertelement <2 x i32> %66, i32 %68, i32 1
  %70 = bitcast <2 x i32> %69 to double
  %71 = call double @region_0_1_reduce_sum_5_0(double %61, double %70)
  %72 = bitcast double %71 to i64
  %73 = bitcast i64 %72 to <2 x i32>
  %74 = extractelement <2 x i32> %73, i32 0
  %75 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %74, i32 2, i32 31)
  %76 = insertelement <2 x i32> undef, i32 %75, i32 0
  %77 = extractelement <2 x i32> %73, i32 1
  %78 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %77, i32 2, i32 31)
  %79 = insertelement <2 x i32> %76, i32 %78, i32 1
  %80 = bitcast <2 x i32> %79 to double
  %81 = call double @region_0_1_reduce_sum_5_0(double %71, double %80)
  %82 = bitcast double %81 to i64
  %83 = bitcast i64 %82 to <2 x i32>
  %84 = extractelement <2 x i32> %83, i32 0
  %85 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %84, i32 1, i32 31)
  %86 = insertelement <2 x i32> undef, i32 %85, i32 0
  %87 = extractelement <2 x i32> %83, i32 1
  %88 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %87, i32 1, i32 31)
  %89 = insertelement <2 x i32> %86, i32 %88, i32 1
  %90 = bitcast <2 x i32> %89 to double
  %91 = call double @region_0_1_reduce_sum_5_0(double %81, double %90)
  %92 = icmp eq i32 %9, 0
  %93 = icmp sle i32 %3, 224
  %94 = and i1 %92, %93
  %95 = mul i32 %4, 8
  %96 = add i32 %95, %5
  br i1 %94, label %97, label %99

97:                                               ; preds = %2
  %98 = getelementptr inbounds [128 x double], ptr %1, i32 0, i32 %96
  store double %91, ptr %98, align 8
  br label %99

99:                                               ; preds = %97, %2
  ret void
}

define internal double @region_0_1_reduce_sum_5_0(double %0, double %1) {
  %3 = fadd nsz double %0, %1
  ret double %3
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

define ptx_kernel void @input_reduce_fusion_1(ptr noalias align 256 dereferenceable(1024) %0, ptr noalias align 256 dereferenceable(8) %1) #3 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %4 = getelementptr inbounds [128 x double], ptr %0, i32 0, i32 %3
  %5 = load double, ptr %4, align 8, !invariant.load !3
  %6 = call double @region_0_1_reduce_sum_5_01(double 0.000000e+00, double %5)
  %7 = add i32 %3, 32
  %8 = getelementptr inbounds [128 x double], ptr %0, i32 0, i32 %7
  %9 = load double, ptr %8, align 8, !invariant.load !3
  %10 = call double @region_0_1_reduce_sum_5_01(double %6, double %9)
  %11 = add i32 %3, 64
  %12 = getelementptr inbounds [128 x double], ptr %0, i32 0, i32 %11
  %13 = load double, ptr %12, align 8, !invariant.load !3
  %14 = call double @region_0_1_reduce_sum_5_01(double %10, double %13)
  %15 = add i32 %3, 96
  %16 = getelementptr inbounds [128 x double], ptr %0, i32 0, i32 %15
  %17 = load double, ptr %16, align 8, !invariant.load !3
  %18 = call double @region_0_1_reduce_sum_5_01(double %14, double %17)
  %19 = bitcast double %18 to i64
  %20 = bitcast i64 %19 to <2 x i32>
  %21 = extractelement <2 x i32> %20, i32 0
  %22 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %21, i32 16, i32 31)
  %23 = insertelement <2 x i32> undef, i32 %22, i32 0
  %24 = extractelement <2 x i32> %20, i32 1
  %25 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %24, i32 16, i32 31)
  %26 = insertelement <2 x i32> %23, i32 %25, i32 1
  %27 = bitcast <2 x i32> %26 to double
  %28 = call double @region_0_1_reduce_sum_5_01(double %18, double %27)
  %29 = bitcast double %28 to i64
  %30 = bitcast i64 %29 to <2 x i32>
  %31 = extractelement <2 x i32> %30, i32 0
  %32 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %31, i32 8, i32 31)
  %33 = insertelement <2 x i32> undef, i32 %32, i32 0
  %34 = extractelement <2 x i32> %30, i32 1
  %35 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %34, i32 8, i32 31)
  %36 = insertelement <2 x i32> %33, i32 %35, i32 1
  %37 = bitcast <2 x i32> %36 to double
  %38 = call double @region_0_1_reduce_sum_5_01(double %28, double %37)
  %39 = bitcast double %38 to i64
  %40 = bitcast i64 %39 to <2 x i32>
  %41 = extractelement <2 x i32> %40, i32 0
  %42 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %41, i32 4, i32 31)
  %43 = insertelement <2 x i32> undef, i32 %42, i32 0
  %44 = extractelement <2 x i32> %40, i32 1
  %45 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %44, i32 4, i32 31)
  %46 = insertelement <2 x i32> %43, i32 %45, i32 1
  %47 = bitcast <2 x i32> %46 to double
  %48 = call double @region_0_1_reduce_sum_5_01(double %38, double %47)
  %49 = bitcast double %48 to i64
  %50 = bitcast i64 %49 to <2 x i32>
  %51 = extractelement <2 x i32> %50, i32 0
  %52 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %51, i32 2, i32 31)
  %53 = insertelement <2 x i32> undef, i32 %52, i32 0
  %54 = extractelement <2 x i32> %50, i32 1
  %55 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %54, i32 2, i32 31)
  %56 = insertelement <2 x i32> %53, i32 %55, i32 1
  %57 = bitcast <2 x i32> %56 to double
  %58 = call double @region_0_1_reduce_sum_5_01(double %48, double %57)
  %59 = bitcast double %58 to i64
  %60 = bitcast i64 %59 to <2 x i32>
  %61 = extractelement <2 x i32> %60, i32 0
  %62 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %61, i32 1, i32 31)
  %63 = insertelement <2 x i32> undef, i32 %62, i32 0
  %64 = extractelement <2 x i32> %60, i32 1
  %65 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %64, i32 1, i32 31)
  %66 = insertelement <2 x i32> %63, i32 %65, i32 1
  %67 = bitcast <2 x i32> %66 to double
  %68 = call double @region_0_1_reduce_sum_5_01(double %58, double %67)
  %69 = icmp eq i32 %3, 0
  br i1 %69, label %70, label %72

70:                                               ; preds = %2
  %71 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  store double %68, ptr %71, align 8
  br label %72

72:                                               ; preds = %70, %2
  ret void
}

define internal double @region_0_1_reduce_sum_5_01(double %0, double %1) {
  %3 = fadd nsz double %0, %1
  ret double %3
}

attributes #0 = { "nvvm.reqntid"="256,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { "nvvm.reqntid"="32,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 256}
!2 = !{i32 0, i32 16}
!3 = !{}
!4 = !{i32 0, i32 32}
