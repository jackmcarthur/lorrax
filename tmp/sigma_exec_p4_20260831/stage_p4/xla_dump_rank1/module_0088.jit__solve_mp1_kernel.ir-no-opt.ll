; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_reduce_fusion_2(ptr noalias align 16 dereferenceable(196608) %0, ptr noalias align 256 dereferenceable(1024) %1, ptr noalias align 256 dereferenceable(1024) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %6 = udiv i32 %4, 32
  %7 = mul i32 %6, 192
  %8 = mul i32 %5, 1536
  %9 = add i32 %7, %8
  %10 = urem i32 %4, 32
  %11 = add i32 %9, %10
  %12 = getelementptr inbounds [24576 x double], ptr %0, i32 0, i32 %11
  %13 = load double, ptr %12, align 8, !invariant.load !3
  %14 = call double @region_1_2_reduce_max_5_0(double 0xFFF0000000000000, double %13)
  %15 = call double @region_0_1_reduce_min_5_0(double 0x7FF0000000000000, double %13)
  %16 = add i32 %11, 32
  %17 = getelementptr inbounds [24576 x double], ptr %0, i32 0, i32 %16
  %18 = load double, ptr %17, align 8, !invariant.load !3
  %19 = call double @region_1_2_reduce_max_5_0(double %14, double %18)
  %20 = call double @region_0_1_reduce_min_5_0(double %15, double %18)
  %21 = add i32 %11, 64
  %22 = getelementptr inbounds [24576 x double], ptr %0, i32 0, i32 %21
  %23 = load double, ptr %22, align 8, !invariant.load !3
  %24 = call double @region_1_2_reduce_max_5_0(double %19, double %23)
  %25 = call double @region_0_1_reduce_min_5_0(double %20, double %23)
  %26 = add i32 %11, 96
  %27 = getelementptr inbounds [24576 x double], ptr %0, i32 0, i32 %26
  %28 = load double, ptr %27, align 8, !invariant.load !3
  %29 = call double @region_1_2_reduce_max_5_0(double %24, double %28)
  %30 = call double @region_0_1_reduce_min_5_0(double %25, double %28)
  %31 = add i32 %11, 128
  %32 = getelementptr inbounds [24576 x double], ptr %0, i32 0, i32 %31
  %33 = load double, ptr %32, align 8, !invariant.load !3
  %34 = call double @region_1_2_reduce_max_5_0(double %29, double %33)
  %35 = call double @region_0_1_reduce_min_5_0(double %30, double %33)
  %36 = add i32 %11, 160
  %37 = getelementptr inbounds [24576 x double], ptr %0, i32 0, i32 %36
  %38 = load double, ptr %37, align 8, !invariant.load !3
  %39 = call double @region_1_2_reduce_max_5_0(double %34, double %38)
  %40 = call double @region_0_1_reduce_min_5_0(double %35, double %38)
  %41 = bitcast double %39 to i64
  %42 = bitcast i64 %41 to <2 x i32>
  %43 = extractelement <2 x i32> %42, i32 0
  %44 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %43, i32 16, i32 31)
  %45 = insertelement <2 x i32> undef, i32 %44, i32 0
  %46 = extractelement <2 x i32> %42, i32 1
  %47 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %46, i32 16, i32 31)
  %48 = insertelement <2 x i32> %45, i32 %47, i32 1
  %49 = bitcast <2 x i32> %48 to double
  %50 = call double @region_1_2_reduce_max_5_0(double %39, double %49)
  %51 = bitcast double %50 to i64
  %52 = bitcast i64 %51 to <2 x i32>
  %53 = extractelement <2 x i32> %52, i32 0
  %54 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %53, i32 8, i32 31)
  %55 = insertelement <2 x i32> undef, i32 %54, i32 0
  %56 = extractelement <2 x i32> %52, i32 1
  %57 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %56, i32 8, i32 31)
  %58 = insertelement <2 x i32> %55, i32 %57, i32 1
  %59 = bitcast <2 x i32> %58 to double
  %60 = call double @region_1_2_reduce_max_5_0(double %50, double %59)
  %61 = bitcast double %60 to i64
  %62 = bitcast i64 %61 to <2 x i32>
  %63 = extractelement <2 x i32> %62, i32 0
  %64 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %63, i32 4, i32 31)
  %65 = insertelement <2 x i32> undef, i32 %64, i32 0
  %66 = extractelement <2 x i32> %62, i32 1
  %67 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %66, i32 4, i32 31)
  %68 = insertelement <2 x i32> %65, i32 %67, i32 1
  %69 = bitcast <2 x i32> %68 to double
  %70 = call double @region_1_2_reduce_max_5_0(double %60, double %69)
  %71 = bitcast double %70 to i64
  %72 = bitcast i64 %71 to <2 x i32>
  %73 = extractelement <2 x i32> %72, i32 0
  %74 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %73, i32 2, i32 31)
  %75 = insertelement <2 x i32> undef, i32 %74, i32 0
  %76 = extractelement <2 x i32> %72, i32 1
  %77 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %76, i32 2, i32 31)
  %78 = insertelement <2 x i32> %75, i32 %77, i32 1
  %79 = bitcast <2 x i32> %78 to double
  %80 = call double @region_1_2_reduce_max_5_0(double %70, double %79)
  %81 = bitcast double %80 to i64
  %82 = bitcast i64 %81 to <2 x i32>
  %83 = extractelement <2 x i32> %82, i32 0
  %84 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %83, i32 1, i32 31)
  %85 = insertelement <2 x i32> undef, i32 %84, i32 0
  %86 = extractelement <2 x i32> %82, i32 1
  %87 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %86, i32 1, i32 31)
  %88 = insertelement <2 x i32> %85, i32 %87, i32 1
  %89 = bitcast <2 x i32> %88 to double
  %90 = call double @region_1_2_reduce_max_5_0(double %80, double %89)
  %91 = bitcast double %40 to i64
  %92 = bitcast i64 %91 to <2 x i32>
  %93 = extractelement <2 x i32> %92, i32 0
  %94 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %93, i32 16, i32 31)
  %95 = insertelement <2 x i32> undef, i32 %94, i32 0
  %96 = extractelement <2 x i32> %92, i32 1
  %97 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %96, i32 16, i32 31)
  %98 = insertelement <2 x i32> %95, i32 %97, i32 1
  %99 = bitcast <2 x i32> %98 to double
  %100 = call double @region_0_1_reduce_min_5_0(double %40, double %99)
  %101 = bitcast double %100 to i64
  %102 = bitcast i64 %101 to <2 x i32>
  %103 = extractelement <2 x i32> %102, i32 0
  %104 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %103, i32 8, i32 31)
  %105 = insertelement <2 x i32> undef, i32 %104, i32 0
  %106 = extractelement <2 x i32> %102, i32 1
  %107 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %106, i32 8, i32 31)
  %108 = insertelement <2 x i32> %105, i32 %107, i32 1
  %109 = bitcast <2 x i32> %108 to double
  %110 = call double @region_0_1_reduce_min_5_0(double %100, double %109)
  %111 = bitcast double %110 to i64
  %112 = bitcast i64 %111 to <2 x i32>
  %113 = extractelement <2 x i32> %112, i32 0
  %114 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %113, i32 4, i32 31)
  %115 = insertelement <2 x i32> undef, i32 %114, i32 0
  %116 = extractelement <2 x i32> %112, i32 1
  %117 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %116, i32 4, i32 31)
  %118 = insertelement <2 x i32> %115, i32 %117, i32 1
  %119 = bitcast <2 x i32> %118 to double
  %120 = call double @region_0_1_reduce_min_5_0(double %110, double %119)
  %121 = bitcast double %120 to i64
  %122 = bitcast i64 %121 to <2 x i32>
  %123 = extractelement <2 x i32> %122, i32 0
  %124 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %123, i32 2, i32 31)
  %125 = insertelement <2 x i32> undef, i32 %124, i32 0
  %126 = extractelement <2 x i32> %122, i32 1
  %127 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %126, i32 2, i32 31)
  %128 = insertelement <2 x i32> %125, i32 %127, i32 1
  %129 = bitcast <2 x i32> %128 to double
  %130 = call double @region_0_1_reduce_min_5_0(double %120, double %129)
  %131 = bitcast double %130 to i64
  %132 = bitcast i64 %131 to <2 x i32>
  %133 = extractelement <2 x i32> %132, i32 0
  %134 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %133, i32 1, i32 31)
  %135 = insertelement <2 x i32> undef, i32 %134, i32 0
  %136 = extractelement <2 x i32> %132, i32 1
  %137 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %136, i32 1, i32 31)
  %138 = insertelement <2 x i32> %135, i32 %137, i32 1
  %139 = bitcast <2 x i32> %138 to double
  %140 = call double @region_0_1_reduce_min_5_0(double %130, double %139)
  %141 = icmp eq i32 %10, 0
  %142 = icmp sle i32 %4, 224
  %143 = and i1 %141, %142
  %144 = mul i32 %5, 8
  %145 = add i32 %144, %6
  br i1 %143, label %146, label %149

146:                                              ; preds = %3
  %147 = getelementptr inbounds [128 x double], ptr %1, i32 0, i32 %145
  store double %90, ptr %147, align 8
  %148 = getelementptr inbounds [128 x double], ptr %2, i32 0, i32 %145
  store double %140, ptr %148, align 8
  br label %149

149:                                              ; preds = %146, %3
  ret void
}

define internal double @region_1_2_reduce_max_5_0(double %0, double %1) {
  %3 = call double @llvm.maximum.f64(double %0, double %1)
  ret double %3
}

define internal double @region_0_1_reduce_min_5_0(double %0, double %1) {
  %3 = call double @llvm.minimum.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #3

define ptx_kernel void @input_reduce_fusion_3(ptr noalias align 256 dereferenceable(1024) %0, ptr noalias align 256 dereferenceable(8) %1) #4 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %4 = getelementptr inbounds [128 x double], ptr %0, i32 0, i32 %3
  %5 = load double, ptr %4, align 8, !invariant.load !3
  %6 = call double @region_0_1_reduce_min_5_01(double 0x7FF0000000000000, double %5)
  %7 = add i32 %3, 32
  %8 = getelementptr inbounds [128 x double], ptr %0, i32 0, i32 %7
  %9 = load double, ptr %8, align 8, !invariant.load !3
  %10 = call double @region_0_1_reduce_min_5_01(double %6, double %9)
  %11 = add i32 %3, 64
  %12 = getelementptr inbounds [128 x double], ptr %0, i32 0, i32 %11
  %13 = load double, ptr %12, align 8, !invariant.load !3
  %14 = call double @region_0_1_reduce_min_5_01(double %10, double %13)
  %15 = add i32 %3, 96
  %16 = getelementptr inbounds [128 x double], ptr %0, i32 0, i32 %15
  %17 = load double, ptr %16, align 8, !invariant.load !3
  %18 = call double @region_0_1_reduce_min_5_01(double %14, double %17)
  %19 = bitcast double %18 to i64
  %20 = bitcast i64 %19 to <2 x i32>
  %21 = extractelement <2 x i32> %20, i32 0
  %22 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %21, i32 16, i32 31)
  %23 = insertelement <2 x i32> undef, i32 %22, i32 0
  %24 = extractelement <2 x i32> %20, i32 1
  %25 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %24, i32 16, i32 31)
  %26 = insertelement <2 x i32> %23, i32 %25, i32 1
  %27 = bitcast <2 x i32> %26 to double
  %28 = call double @region_0_1_reduce_min_5_01(double %18, double %27)
  %29 = bitcast double %28 to i64
  %30 = bitcast i64 %29 to <2 x i32>
  %31 = extractelement <2 x i32> %30, i32 0
  %32 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %31, i32 8, i32 31)
  %33 = insertelement <2 x i32> undef, i32 %32, i32 0
  %34 = extractelement <2 x i32> %30, i32 1
  %35 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %34, i32 8, i32 31)
  %36 = insertelement <2 x i32> %33, i32 %35, i32 1
  %37 = bitcast <2 x i32> %36 to double
  %38 = call double @region_0_1_reduce_min_5_01(double %28, double %37)
  %39 = bitcast double %38 to i64
  %40 = bitcast i64 %39 to <2 x i32>
  %41 = extractelement <2 x i32> %40, i32 0
  %42 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %41, i32 4, i32 31)
  %43 = insertelement <2 x i32> undef, i32 %42, i32 0
  %44 = extractelement <2 x i32> %40, i32 1
  %45 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %44, i32 4, i32 31)
  %46 = insertelement <2 x i32> %43, i32 %45, i32 1
  %47 = bitcast <2 x i32> %46 to double
  %48 = call double @region_0_1_reduce_min_5_01(double %38, double %47)
  %49 = bitcast double %48 to i64
  %50 = bitcast i64 %49 to <2 x i32>
  %51 = extractelement <2 x i32> %50, i32 0
  %52 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %51, i32 2, i32 31)
  %53 = insertelement <2 x i32> undef, i32 %52, i32 0
  %54 = extractelement <2 x i32> %50, i32 1
  %55 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %54, i32 2, i32 31)
  %56 = insertelement <2 x i32> %53, i32 %55, i32 1
  %57 = bitcast <2 x i32> %56 to double
  %58 = call double @region_0_1_reduce_min_5_01(double %48, double %57)
  %59 = bitcast double %58 to i64
  %60 = bitcast i64 %59 to <2 x i32>
  %61 = extractelement <2 x i32> %60, i32 0
  %62 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %61, i32 1, i32 31)
  %63 = insertelement <2 x i32> undef, i32 %62, i32 0
  %64 = extractelement <2 x i32> %60, i32 1
  %65 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %64, i32 1, i32 31)
  %66 = insertelement <2 x i32> %63, i32 %65, i32 1
  %67 = bitcast <2 x i32> %66 to double
  %68 = call double @region_0_1_reduce_min_5_01(double %58, double %67)
  %69 = icmp eq i32 %3, 0
  br i1 %69, label %70, label %72

70:                                               ; preds = %2
  %71 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  store double %68, ptr %71, align 8
  br label %72

72:                                               ; preds = %70, %2
  ret void
}

define internal double @region_0_1_reduce_min_5_01(double %0, double %1) {
  %3 = call double @llvm.minimum.f64(double %0, double %1)
  ret double %3
}

define ptx_kernel void @loop_subtract_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2) #5 {
  %4 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %5 = load double, ptr %4, align 8, !invariant.load !3
  %6 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %7 = load double, ptr %6, align 8, !invariant.load !3
  %8 = fmul double %5, 1.600000e+01
  %9 = fsub double %7, %8
  %10 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  store double %9, ptr %10, align 8
  ret void
}

define ptx_kernel void @input_reduce_fusion_4(ptr noalias align 256 dereferenceable(1024) %0, ptr noalias align 256 dereferenceable(8) %1) #4 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %4 = getelementptr inbounds [128 x double], ptr %0, i32 0, i32 %3
  %5 = load double, ptr %4, align 8, !invariant.load !3
  %6 = call double @region_1_2_reduce_max_5_02(double 0xFFF0000000000000, double %5)
  %7 = add i32 %3, 32
  %8 = getelementptr inbounds [128 x double], ptr %0, i32 0, i32 %7
  %9 = load double, ptr %8, align 8, !invariant.load !3
  %10 = call double @region_1_2_reduce_max_5_02(double %6, double %9)
  %11 = add i32 %3, 64
  %12 = getelementptr inbounds [128 x double], ptr %0, i32 0, i32 %11
  %13 = load double, ptr %12, align 8, !invariant.load !3
  %14 = call double @region_1_2_reduce_max_5_02(double %10, double %13)
  %15 = add i32 %3, 96
  %16 = getelementptr inbounds [128 x double], ptr %0, i32 0, i32 %15
  %17 = load double, ptr %16, align 8, !invariant.load !3
  %18 = call double @region_1_2_reduce_max_5_02(double %14, double %17)
  %19 = bitcast double %18 to i64
  %20 = bitcast i64 %19 to <2 x i32>
  %21 = extractelement <2 x i32> %20, i32 0
  %22 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %21, i32 16, i32 31)
  %23 = insertelement <2 x i32> undef, i32 %22, i32 0
  %24 = extractelement <2 x i32> %20, i32 1
  %25 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %24, i32 16, i32 31)
  %26 = insertelement <2 x i32> %23, i32 %25, i32 1
  %27 = bitcast <2 x i32> %26 to double
  %28 = call double @region_1_2_reduce_max_5_02(double %18, double %27)
  %29 = bitcast double %28 to i64
  %30 = bitcast i64 %29 to <2 x i32>
  %31 = extractelement <2 x i32> %30, i32 0
  %32 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %31, i32 8, i32 31)
  %33 = insertelement <2 x i32> undef, i32 %32, i32 0
  %34 = extractelement <2 x i32> %30, i32 1
  %35 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %34, i32 8, i32 31)
  %36 = insertelement <2 x i32> %33, i32 %35, i32 1
  %37 = bitcast <2 x i32> %36 to double
  %38 = call double @region_1_2_reduce_max_5_02(double %28, double %37)
  %39 = bitcast double %38 to i64
  %40 = bitcast i64 %39 to <2 x i32>
  %41 = extractelement <2 x i32> %40, i32 0
  %42 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %41, i32 4, i32 31)
  %43 = insertelement <2 x i32> undef, i32 %42, i32 0
  %44 = extractelement <2 x i32> %40, i32 1
  %45 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %44, i32 4, i32 31)
  %46 = insertelement <2 x i32> %43, i32 %45, i32 1
  %47 = bitcast <2 x i32> %46 to double
  %48 = call double @region_1_2_reduce_max_5_02(double %38, double %47)
  %49 = bitcast double %48 to i64
  %50 = bitcast i64 %49 to <2 x i32>
  %51 = extractelement <2 x i32> %50, i32 0
  %52 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %51, i32 2, i32 31)
  %53 = insertelement <2 x i32> undef, i32 %52, i32 0
  %54 = extractelement <2 x i32> %50, i32 1
  %55 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %54, i32 2, i32 31)
  %56 = insertelement <2 x i32> %53, i32 %55, i32 1
  %57 = bitcast <2 x i32> %56 to double
  %58 = call double @region_1_2_reduce_max_5_02(double %48, double %57)
  %59 = bitcast double %58 to i64
  %60 = bitcast i64 %59 to <2 x i32>
  %61 = extractelement <2 x i32> %60, i32 0
  %62 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %61, i32 1, i32 31)
  %63 = insertelement <2 x i32> undef, i32 %62, i32 0
  %64 = extractelement <2 x i32> %60, i32 1
  %65 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %64, i32 1, i32 31)
  %66 = insertelement <2 x i32> %63, i32 %65, i32 1
  %67 = bitcast <2 x i32> %66 to double
  %68 = call double @region_1_2_reduce_max_5_02(double %58, double %67)
  %69 = icmp eq i32 %3, 0
  br i1 %69, label %70, label %72

70:                                               ; preds = %2
  %71 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  store double %68, ptr %71, align 8
  br label %72

72:                                               ; preds = %70, %2
  ret void
}

define internal double @region_1_2_reduce_max_5_02(double %0, double %1) {
  %3 = call double @llvm.maximum.f64(double %0, double %1)
  ret double %3
}

define ptx_kernel void @loop_add_fusion_1(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2) #5 {
  %4 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %5 = load double, ptr %4, align 8, !invariant.load !3
  %6 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %7 = load double, ptr %6, align 8, !invariant.load !3
  %8 = fmul double %5, 1.600000e+01
  %9 = fadd double %7, %8
  %10 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  store double %9, ptr %10, align 8
  ret void
}

define ptx_kernel void @loop_compare_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(1) %1) #5 {
  %3 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %4 = load i64, ptr %3, align 4, !invariant.load !3
  %5 = icmp slt i64 %4, 64
  %6 = zext i1 %5 to i8
  %7 = getelementptr inbounds [1 x i8], ptr %1, i32 0, i32 0
  store i8 %6, ptr %7, align 1
  ret void
}

define ptx_kernel void @loop_add_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1) #5 {
  %3 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %4 = load i64, ptr %3, align 4
  %5 = add i64 %4, 1
  store i64 %5, ptr %3, align 4
  ret void
}

define ptx_kernel void @input_reduce_fusion_1(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 16 dereferenceable(196608) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(4096) %5) #0 {
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %9 = udiv i32 %7, 32
  %10 = mul i32 %9, 48
  %11 = mul i32 %8, 384
  %12 = add i32 %10, %11
  %13 = urem i32 %7, 32
  %14 = add i32 %12, %13
  %15 = getelementptr inbounds [24576 x double], ptr %2, i32 0, i32 %14
  %16 = load double, ptr %15, align 8, !invariant.load !3
  %17 = getelementptr inbounds [1 x double], ptr %3, i32 0, i32 0
  %18 = load double, ptr %17, align 8, !invariant.load !3
  %19 = getelementptr inbounds [1 x double], ptr %4, i32 0, i32 0
  %20 = load double, ptr %19, align 8, !invariant.load !3
  %21 = fadd double %18, %20
  %22 = fmul double %21, 5.000000e-01
  %23 = fsub double %16, %22
  %24 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %25 = load double, ptr %24, align 8, !invariant.load !3
  %26 = fmul double %25, 2.000000e+00
  %27 = fdiv double %23, %26
  %28 = fmul double %27, %27
  %29 = fneg double %28
  %30 = call double @__nv_exp(double %29)
  %31 = call double @__nv_erf(double %27)
  %32 = call double @__nv_fabs(double %27)
  %33 = fcmp one double %32, 0x7FF0000000000000
  %34 = fmul double %27, %30
  %35 = fsub double 1.000000e+00, %31
  %36 = select i1 %33, double %34, double 0.000000e+00
  %37 = fmul double %35, 5.000000e-01
  %38 = fmul double %36, 0x3FD20DD750429B6D
  %39 = fsub double %37, %38
  %40 = call double @__nv_fabs(double %39)
  %41 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %42 = load double, ptr %41, align 8, !invariant.load !3
  %43 = fcmp olt double %40, %42
  %44 = select i1 %43, double 0.000000e+00, double %39
  %45 = fsub double 1.000000e+00, %44
  %46 = call double @__nv_fabs(double %45)
  %47 = fcmp olt double %46, %42
  %48 = select i1 %47, double 1.000000e+00, double %44
  %49 = call double @region_3_5_reduce_sum_5_0(double 0.000000e+00, double %48)
  %50 = icmp sle i32 %13, 15
  br i1 %50, label %51, label %84

51:                                               ; preds = %6
  %52 = add i32 %14, 32
  %53 = getelementptr inbounds [24576 x double], ptr %2, i32 0, i32 %52
  %54 = load double, ptr %53, align 8, !invariant.load !3
  %55 = load double, ptr %17, align 8, !invariant.load !3
  %56 = load double, ptr %19, align 8, !invariant.load !3
  %57 = fadd double %55, %56
  %58 = fmul double %57, 5.000000e-01
  %59 = fsub double %54, %58
  %60 = load double, ptr %24, align 8, !invariant.load !3
  %61 = fmul double %60, 2.000000e+00
  %62 = fdiv double %59, %61
  %63 = fmul double %62, %62
  %64 = fneg double %63
  %65 = call double @__nv_exp(double %64)
  %66 = call double @__nv_erf(double %62)
  %67 = call double @__nv_fabs(double %62)
  %68 = fcmp one double %67, 0x7FF0000000000000
  %69 = fmul double %62, %65
  %70 = fsub double 1.000000e+00, %66
  %71 = select i1 %68, double %69, double 0.000000e+00
  %72 = fmul double %70, 5.000000e-01
  %73 = fmul double %71, 0x3FD20DD750429B6D
  %74 = fsub double %72, %73
  %75 = call double @__nv_fabs(double %74)
  %76 = load double, ptr %41, align 8, !invariant.load !3
  %77 = fcmp olt double %75, %76
  %78 = select i1 %77, double 0.000000e+00, double %74
  %79 = fsub double 1.000000e+00, %78
  %80 = call double @__nv_fabs(double %79)
  %81 = fcmp olt double %80, %76
  %82 = select i1 %81, double 1.000000e+00, double %78
  %83 = call double @region_3_5_reduce_sum_5_0(double %49, double %82)
  br label %85

84:                                               ; preds = %6
  br label %85

85:                                               ; preds = %51, %84
  %86 = phi double [ %49, %84 ], [ %83, %51 ]
  br label %87

87:                                               ; preds = %85
  %88 = bitcast double %86 to i64
  %89 = bitcast i64 %88 to <2 x i32>
  %90 = extractelement <2 x i32> %89, i32 0
  %91 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %90, i32 16, i32 31)
  %92 = insertelement <2 x i32> undef, i32 %91, i32 0
  %93 = extractelement <2 x i32> %89, i32 1
  %94 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %93, i32 16, i32 31)
  %95 = insertelement <2 x i32> %92, i32 %94, i32 1
  %96 = bitcast <2 x i32> %95 to double
  %97 = call double @region_3_5_reduce_sum_5_0(double %86, double %96)
  %98 = bitcast double %97 to i64
  %99 = bitcast i64 %98 to <2 x i32>
  %100 = extractelement <2 x i32> %99, i32 0
  %101 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %100, i32 8, i32 31)
  %102 = insertelement <2 x i32> undef, i32 %101, i32 0
  %103 = extractelement <2 x i32> %99, i32 1
  %104 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %103, i32 8, i32 31)
  %105 = insertelement <2 x i32> %102, i32 %104, i32 1
  %106 = bitcast <2 x i32> %105 to double
  %107 = call double @region_3_5_reduce_sum_5_0(double %97, double %106)
  %108 = bitcast double %107 to i64
  %109 = bitcast i64 %108 to <2 x i32>
  %110 = extractelement <2 x i32> %109, i32 0
  %111 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %110, i32 4, i32 31)
  %112 = insertelement <2 x i32> undef, i32 %111, i32 0
  %113 = extractelement <2 x i32> %109, i32 1
  %114 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %113, i32 4, i32 31)
  %115 = insertelement <2 x i32> %112, i32 %114, i32 1
  %116 = bitcast <2 x i32> %115 to double
  %117 = call double @region_3_5_reduce_sum_5_0(double %107, double %116)
  %118 = bitcast double %117 to i64
  %119 = bitcast i64 %118 to <2 x i32>
  %120 = extractelement <2 x i32> %119, i32 0
  %121 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %120, i32 2, i32 31)
  %122 = insertelement <2 x i32> undef, i32 %121, i32 0
  %123 = extractelement <2 x i32> %119, i32 1
  %124 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %123, i32 2, i32 31)
  %125 = insertelement <2 x i32> %122, i32 %124, i32 1
  %126 = bitcast <2 x i32> %125 to double
  %127 = call double @region_3_5_reduce_sum_5_0(double %117, double %126)
  %128 = bitcast double %127 to i64
  %129 = bitcast i64 %128 to <2 x i32>
  %130 = extractelement <2 x i32> %129, i32 0
  %131 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %130, i32 1, i32 31)
  %132 = insertelement <2 x i32> undef, i32 %131, i32 0
  %133 = extractelement <2 x i32> %129, i32 1
  %134 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %133, i32 1, i32 31)
  %135 = insertelement <2 x i32> %132, i32 %134, i32 1
  %136 = bitcast <2 x i32> %135 to double
  %137 = call double @region_3_5_reduce_sum_5_0(double %127, double %136)
  %138 = icmp eq i32 %13, 0
  %139 = icmp sle i32 %7, 224
  %140 = and i1 %138, %139
  %141 = mul i32 %8, 8
  %142 = add i32 %141, %9
  br i1 %140, label %143, label %145

143:                                              ; preds = %87
  %144 = getelementptr inbounds [512 x double], ptr %5, i32 0, i32 %142
  store double %137, ptr %144, align 8
  br label %145

145:                                              ; preds = %143, %87
  ret void
}

declare double @__nv_exp(double)

declare double @__nv_erf(double)

declare double @__nv_fabs(double)

define internal double @region_3_5_reduce_sum_5_0(double %0, double %1) {
  %3 = fadd nsz double %0, %1
  ret double %3
}

define ptx_kernel void @input_reduce_fusion(ptr noalias align 16 dereferenceable(4096) %0, ptr noalias align 256 dereferenceable(4096) %1, ptr noalias align 256 dereferenceable(8) %2) #4 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %5 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %4
  %6 = load double, ptr %5, align 8, !invariant.load !3
  %7 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %4
  %8 = load double, ptr %7, align 8, !invariant.load !3
  %9 = fmul double %6, %8
  %10 = call double @region_3_5_clone_reduce_sum_6_0(double 0.000000e+00, double %9)
  %11 = add i32 %4, 32
  %12 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %11
  %13 = load double, ptr %12, align 8, !invariant.load !3
  %14 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %11
  %15 = load double, ptr %14, align 8, !invariant.load !3
  %16 = fmul double %13, %15
  %17 = call double @region_3_5_clone_reduce_sum_6_0(double %10, double %16)
  %18 = add i32 %4, 64
  %19 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %18
  %20 = load double, ptr %19, align 8, !invariant.load !3
  %21 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %18
  %22 = load double, ptr %21, align 8, !invariant.load !3
  %23 = fmul double %20, %22
  %24 = call double @region_3_5_clone_reduce_sum_6_0(double %17, double %23)
  %25 = add i32 %4, 96
  %26 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %25
  %27 = load double, ptr %26, align 8, !invariant.load !3
  %28 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %25
  %29 = load double, ptr %28, align 8, !invariant.load !3
  %30 = fmul double %27, %29
  %31 = call double @region_3_5_clone_reduce_sum_6_0(double %24, double %30)
  %32 = add i32 %4, 128
  %33 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %32
  %34 = load double, ptr %33, align 8, !invariant.load !3
  %35 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %32
  %36 = load double, ptr %35, align 8, !invariant.load !3
  %37 = fmul double %34, %36
  %38 = call double @region_3_5_clone_reduce_sum_6_0(double %31, double %37)
  %39 = add i32 %4, 160
  %40 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %39
  %41 = load double, ptr %40, align 8, !invariant.load !3
  %42 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %39
  %43 = load double, ptr %42, align 8, !invariant.load !3
  %44 = fmul double %41, %43
  %45 = call double @region_3_5_clone_reduce_sum_6_0(double %38, double %44)
  %46 = add i32 %4, 192
  %47 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %46
  %48 = load double, ptr %47, align 8, !invariant.load !3
  %49 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %46
  %50 = load double, ptr %49, align 8, !invariant.load !3
  %51 = fmul double %48, %50
  %52 = call double @region_3_5_clone_reduce_sum_6_0(double %45, double %51)
  %53 = add i32 %4, 224
  %54 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %53
  %55 = load double, ptr %54, align 8, !invariant.load !3
  %56 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %53
  %57 = load double, ptr %56, align 8, !invariant.load !3
  %58 = fmul double %55, %57
  %59 = call double @region_3_5_clone_reduce_sum_6_0(double %52, double %58)
  %60 = add i32 %4, 256
  %61 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %60
  %62 = load double, ptr %61, align 8, !invariant.load !3
  %63 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %60
  %64 = load double, ptr %63, align 8, !invariant.load !3
  %65 = fmul double %62, %64
  %66 = call double @region_3_5_clone_reduce_sum_6_0(double %59, double %65)
  %67 = add i32 %4, 288
  %68 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %67
  %69 = load double, ptr %68, align 8, !invariant.load !3
  %70 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %67
  %71 = load double, ptr %70, align 8, !invariant.load !3
  %72 = fmul double %69, %71
  %73 = call double @region_3_5_clone_reduce_sum_6_0(double %66, double %72)
  %74 = add i32 %4, 320
  %75 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %74
  %76 = load double, ptr %75, align 8, !invariant.load !3
  %77 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %74
  %78 = load double, ptr %77, align 8, !invariant.load !3
  %79 = fmul double %76, %78
  %80 = call double @region_3_5_clone_reduce_sum_6_0(double %73, double %79)
  %81 = add i32 %4, 352
  %82 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %81
  %83 = load double, ptr %82, align 8, !invariant.load !3
  %84 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %81
  %85 = load double, ptr %84, align 8, !invariant.load !3
  %86 = fmul double %83, %85
  %87 = call double @region_3_5_clone_reduce_sum_6_0(double %80, double %86)
  %88 = add i32 %4, 384
  %89 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %88
  %90 = load double, ptr %89, align 8, !invariant.load !3
  %91 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %88
  %92 = load double, ptr %91, align 8, !invariant.load !3
  %93 = fmul double %90, %92
  %94 = call double @region_3_5_clone_reduce_sum_6_0(double %87, double %93)
  %95 = add i32 %4, 416
  %96 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %95
  %97 = load double, ptr %96, align 8, !invariant.load !3
  %98 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %95
  %99 = load double, ptr %98, align 8, !invariant.load !3
  %100 = fmul double %97, %99
  %101 = call double @region_3_5_clone_reduce_sum_6_0(double %94, double %100)
  %102 = add i32 %4, 448
  %103 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %102
  %104 = load double, ptr %103, align 8, !invariant.load !3
  %105 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %102
  %106 = load double, ptr %105, align 8, !invariant.load !3
  %107 = fmul double %104, %106
  %108 = call double @region_3_5_clone_reduce_sum_6_0(double %101, double %107)
  %109 = add i32 %4, 480
  %110 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %109
  %111 = load double, ptr %110, align 8, !invariant.load !3
  %112 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %109
  %113 = load double, ptr %112, align 8, !invariant.load !3
  %114 = fmul double %111, %113
  %115 = call double @region_3_5_clone_reduce_sum_6_0(double %108, double %114)
  %116 = bitcast double %115 to i64
  %117 = bitcast i64 %116 to <2 x i32>
  %118 = extractelement <2 x i32> %117, i32 0
  %119 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %118, i32 16, i32 31)
  %120 = insertelement <2 x i32> undef, i32 %119, i32 0
  %121 = extractelement <2 x i32> %117, i32 1
  %122 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %121, i32 16, i32 31)
  %123 = insertelement <2 x i32> %120, i32 %122, i32 1
  %124 = bitcast <2 x i32> %123 to double
  %125 = call double @region_3_5_clone_reduce_sum_6_0(double %115, double %124)
  %126 = bitcast double %125 to i64
  %127 = bitcast i64 %126 to <2 x i32>
  %128 = extractelement <2 x i32> %127, i32 0
  %129 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %128, i32 8, i32 31)
  %130 = insertelement <2 x i32> undef, i32 %129, i32 0
  %131 = extractelement <2 x i32> %127, i32 1
  %132 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %131, i32 8, i32 31)
  %133 = insertelement <2 x i32> %130, i32 %132, i32 1
  %134 = bitcast <2 x i32> %133 to double
  %135 = call double @region_3_5_clone_reduce_sum_6_0(double %125, double %134)
  %136 = bitcast double %135 to i64
  %137 = bitcast i64 %136 to <2 x i32>
  %138 = extractelement <2 x i32> %137, i32 0
  %139 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %138, i32 4, i32 31)
  %140 = insertelement <2 x i32> undef, i32 %139, i32 0
  %141 = extractelement <2 x i32> %137, i32 1
  %142 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %141, i32 4, i32 31)
  %143 = insertelement <2 x i32> %140, i32 %142, i32 1
  %144 = bitcast <2 x i32> %143 to double
  %145 = call double @region_3_5_clone_reduce_sum_6_0(double %135, double %144)
  %146 = bitcast double %145 to i64
  %147 = bitcast i64 %146 to <2 x i32>
  %148 = extractelement <2 x i32> %147, i32 0
  %149 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %148, i32 2, i32 31)
  %150 = insertelement <2 x i32> undef, i32 %149, i32 0
  %151 = extractelement <2 x i32> %147, i32 1
  %152 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %151, i32 2, i32 31)
  %153 = insertelement <2 x i32> %150, i32 %152, i32 1
  %154 = bitcast <2 x i32> %153 to double
  %155 = call double @region_3_5_clone_reduce_sum_6_0(double %145, double %154)
  %156 = bitcast double %155 to i64
  %157 = bitcast i64 %156 to <2 x i32>
  %158 = extractelement <2 x i32> %157, i32 0
  %159 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %158, i32 1, i32 31)
  %160 = insertelement <2 x i32> undef, i32 %159, i32 0
  %161 = extractelement <2 x i32> %157, i32 1
  %162 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %161, i32 1, i32 31)
  %163 = insertelement <2 x i32> %160, i32 %162, i32 1
  %164 = bitcast <2 x i32> %163 to double
  %165 = call double @region_3_5_clone_reduce_sum_6_0(double %155, double %164)
  %166 = icmp eq i32 %4, 0
  br i1 %166, label %167, label %169

167:                                              ; preds = %3
  %168 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  store double %165, ptr %168, align 8
  br label %169

169:                                              ; preds = %167, %3
  ret void
}

define internal double @region_3_5_clone_reduce_sum_6_0(double %0, double %1) {
  %3 = fadd nsz double %0, %1
  ret double %3
}

define ptx_kernel void @loop_select_fusion_1(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 16 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(8) %5) #5 {
  %7 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  %8 = load double, ptr %7, align 8, !invariant.load !3
  %9 = getelementptr inbounds [1 x double], ptr %3, i32 0, i32 0
  %10 = load double, ptr %9, align 8, !invariant.load !3
  %11 = getelementptr inbounds [1 x double], ptr %4, i32 0, i32 0
  %12 = load double, ptr %11, align 8, !invariant.load !3
  %13 = fmul double %8, %10
  %14 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %15 = load double, ptr %14, align 8, !invariant.load !3
  %16 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %17 = load double, ptr %16, align 8, !invariant.load !3
  %18 = fadd double %12, %17
  %19 = fcmp olt double %13, %15
  %20 = fmul double %18, 5.000000e-01
  %21 = select i1 %19, double %17, double %20
  %22 = getelementptr inbounds [1 x double], ptr %5, i32 0, i32 0
  store double %21, ptr %22, align 8
  ret void
}

define ptx_kernel void @loop_select_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 16 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(8) %5) #5 {
  %7 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  %8 = load double, ptr %7, align 8, !invariant.load !3
  %9 = getelementptr inbounds [1 x double], ptr %3, i32 0, i32 0
  %10 = load double, ptr %9, align 8, !invariant.load !3
  %11 = getelementptr inbounds [1 x double], ptr %4, i32 0, i32 0
  %12 = load double, ptr %11, align 8, !invariant.load !3
  %13 = fmul double %8, %10
  %14 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %15 = load double, ptr %14, align 8, !invariant.load !3
  %16 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %17 = load double, ptr %16, align 8
  %18 = fadd double %17, %12
  %19 = fcmp olt double %13, %15
  %20 = fmul double %18, 5.000000e-01
  %21 = select i1 %19, double %20, double %17
  store double %21, ptr %16, align 8
  ret void
}

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2) #5 {
  %4 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %5 = load double, ptr %4, align 8, !invariant.load !3
  %6 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %7 = load double, ptr %6, align 8, !invariant.load !3
  %8 = fadd double %5, %7
  %9 = fmul double %8, 5.000000e-01
  %10 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  store double %9, ptr %10, align 8
  ret void
}

define ptx_kernel void @loop_select_fusion_2(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 16 dereferenceable(196608) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 16 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(196608) %4) #6 {
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !7
  %8 = mul i32 %6, 128
  %9 = add i32 %8, %7
  %10 = getelementptr inbounds [24576 x double], ptr %1, i32 0, i32 %9
  %11 = load double, ptr %10, align 8, !invariant.load !3
  %12 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  %13 = load double, ptr %12, align 8, !invariant.load !3
  %14 = fsub double %11, %13
  %15 = getelementptr inbounds [1 x double], ptr %3, i32 0, i32 0
  %16 = load double, ptr %15, align 8, !invariant.load !3
  %17 = fmul double %16, 2.000000e+00
  %18 = fdiv double %14, %17
  %19 = fmul double %18, %18
  %20 = fneg double %19
  %21 = call double @__nv_exp(double %20)
  %22 = call double @__nv_erf(double %18)
  %23 = call double @__nv_fabs(double %18)
  %24 = fcmp one double %23, 0x7FF0000000000000
  %25 = fmul double %18, %21
  %26 = fsub double 1.000000e+00, %22
  %27 = select i1 %24, double %25, double 0.000000e+00
  %28 = fmul double %26, 5.000000e-01
  %29 = fmul double %27, 0x3FD20DD750429B6D
  %30 = fsub double %28, %29
  %31 = call double @__nv_fabs(double %30)
  %32 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %33 = load double, ptr %32, align 8, !invariant.load !3
  %34 = fcmp olt double %31, %33
  %35 = select i1 %34, double 0.000000e+00, double %30
  %36 = fsub double 1.000000e+00, %35
  %37 = call double @__nv_fabs(double %36)
  %38 = fcmp olt double %37, %33
  %39 = select i1 %38, double 1.000000e+00, double %35
  %40 = getelementptr inbounds [24576 x double], ptr %4, i32 0, i32 %9
  store double %39, ptr %40, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="256,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { "nvvm.reqntid"="32,1,1" }
attributes #5 = { "nvvm.reqntid"="1,1,1" }
attributes #6 = { "nvvm.reqntid"="128,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 256}
!2 = !{i32 0, i32 16}
!3 = !{}
!4 = !{i32 0, i32 32}
!5 = !{i32 0, i32 64}
!6 = !{i32 0, i32 192}
!7 = !{i32 0, i32 128}
