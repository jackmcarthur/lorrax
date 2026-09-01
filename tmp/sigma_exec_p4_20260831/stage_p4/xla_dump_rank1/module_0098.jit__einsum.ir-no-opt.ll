; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_reduce_fusion_1(ptr noalias align 16 dereferenceable(196608) %0, ptr noalias align 256 dereferenceable(4096) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = udiv i32 %3, 32
  %6 = mul i32 %5, 48
  %7 = mul i32 %4, 384
  %8 = add i32 %6, %7
  %9 = urem i32 %3, 32
  %10 = add i32 %8, %9
  %11 = getelementptr inbounds [24576 x double], ptr %0, i32 0, i32 %10
  %12 = load double, ptr %11, align 8, !invariant.load !3
  %13 = call double @region_0_1_reduce_sum_5_0(double 0.000000e+00, double %12)
  %14 = icmp sle i32 %9, 15
  br i1 %14, label %15, label %20

15:                                               ; preds = %2
  %16 = add i32 %10, 32
  %17 = getelementptr inbounds [24576 x double], ptr %0, i32 0, i32 %16
  %18 = load double, ptr %17, align 8, !invariant.load !3
  %19 = call double @region_0_1_reduce_sum_5_0(double %13, double %18)
  br label %21

20:                                               ; preds = %2
  br label %21

21:                                               ; preds = %15, %20
  %22 = phi double [ %13, %20 ], [ %19, %15 ]
  br label %23

23:                                               ; preds = %21
  %24 = bitcast double %22 to i64
  %25 = bitcast i64 %24 to <2 x i32>
  %26 = extractelement <2 x i32> %25, i32 0
  %27 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %26, i32 16, i32 31)
  %28 = insertelement <2 x i32> undef, i32 %27, i32 0
  %29 = extractelement <2 x i32> %25, i32 1
  %30 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %29, i32 16, i32 31)
  %31 = insertelement <2 x i32> %28, i32 %30, i32 1
  %32 = bitcast <2 x i32> %31 to double
  %33 = call double @region_0_1_reduce_sum_5_0(double %22, double %32)
  %34 = bitcast double %33 to i64
  %35 = bitcast i64 %34 to <2 x i32>
  %36 = extractelement <2 x i32> %35, i32 0
  %37 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %36, i32 8, i32 31)
  %38 = insertelement <2 x i32> undef, i32 %37, i32 0
  %39 = extractelement <2 x i32> %35, i32 1
  %40 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %39, i32 8, i32 31)
  %41 = insertelement <2 x i32> %38, i32 %40, i32 1
  %42 = bitcast <2 x i32> %41 to double
  %43 = call double @region_0_1_reduce_sum_5_0(double %33, double %42)
  %44 = bitcast double %43 to i64
  %45 = bitcast i64 %44 to <2 x i32>
  %46 = extractelement <2 x i32> %45, i32 0
  %47 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %46, i32 4, i32 31)
  %48 = insertelement <2 x i32> undef, i32 %47, i32 0
  %49 = extractelement <2 x i32> %45, i32 1
  %50 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %49, i32 4, i32 31)
  %51 = insertelement <2 x i32> %48, i32 %50, i32 1
  %52 = bitcast <2 x i32> %51 to double
  %53 = call double @region_0_1_reduce_sum_5_0(double %43, double %52)
  %54 = bitcast double %53 to i64
  %55 = bitcast i64 %54 to <2 x i32>
  %56 = extractelement <2 x i32> %55, i32 0
  %57 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %56, i32 2, i32 31)
  %58 = insertelement <2 x i32> undef, i32 %57, i32 0
  %59 = extractelement <2 x i32> %55, i32 1
  %60 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %59, i32 2, i32 31)
  %61 = insertelement <2 x i32> %58, i32 %60, i32 1
  %62 = bitcast <2 x i32> %61 to double
  %63 = call double @region_0_1_reduce_sum_5_0(double %53, double %62)
  %64 = bitcast double %63 to i64
  %65 = bitcast i64 %64 to <2 x i32>
  %66 = extractelement <2 x i32> %65, i32 0
  %67 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %66, i32 1, i32 31)
  %68 = insertelement <2 x i32> undef, i32 %67, i32 0
  %69 = extractelement <2 x i32> %65, i32 1
  %70 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %69, i32 1, i32 31)
  %71 = insertelement <2 x i32> %68, i32 %70, i32 1
  %72 = bitcast <2 x i32> %71 to double
  %73 = call double @region_0_1_reduce_sum_5_0(double %63, double %72)
  %74 = icmp eq i32 %9, 0
  %75 = icmp sle i32 %3, 224
  %76 = and i1 %74, %75
  %77 = mul i32 %4, 8
  %78 = add i32 %77, %5
  br i1 %76, label %79, label %81

79:                                               ; preds = %23
  %80 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %78
  store double %73, ptr %80, align 8
  br label %81

81:                                               ; preds = %79, %23
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

define ptx_kernel void @input_reduce_fusion(ptr noalias align 16 dereferenceable(4096) %0, ptr noalias align 256 dereferenceable(4096) %1, ptr noalias align 256 dereferenceable(8) %2) #3 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %5 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %4
  %6 = load double, ptr %5, align 8, !invariant.load !3
  %7 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %4
  %8 = load double, ptr %7, align 8, !invariant.load !3
  %9 = fmul double %6, %8
  %10 = call double @region_0_1_clone_reduce_sum_2_0(double 0.000000e+00, double %9)
  %11 = add i32 %4, 32
  %12 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %11
  %13 = load double, ptr %12, align 8, !invariant.load !3
  %14 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %11
  %15 = load double, ptr %14, align 8, !invariant.load !3
  %16 = fmul double %13, %15
  %17 = call double @region_0_1_clone_reduce_sum_2_0(double %10, double %16)
  %18 = add i32 %4, 64
  %19 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %18
  %20 = load double, ptr %19, align 8, !invariant.load !3
  %21 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %18
  %22 = load double, ptr %21, align 8, !invariant.load !3
  %23 = fmul double %20, %22
  %24 = call double @region_0_1_clone_reduce_sum_2_0(double %17, double %23)
  %25 = add i32 %4, 96
  %26 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %25
  %27 = load double, ptr %26, align 8, !invariant.load !3
  %28 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %25
  %29 = load double, ptr %28, align 8, !invariant.load !3
  %30 = fmul double %27, %29
  %31 = call double @region_0_1_clone_reduce_sum_2_0(double %24, double %30)
  %32 = add i32 %4, 128
  %33 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %32
  %34 = load double, ptr %33, align 8, !invariant.load !3
  %35 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %32
  %36 = load double, ptr %35, align 8, !invariant.load !3
  %37 = fmul double %34, %36
  %38 = call double @region_0_1_clone_reduce_sum_2_0(double %31, double %37)
  %39 = add i32 %4, 160
  %40 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %39
  %41 = load double, ptr %40, align 8, !invariant.load !3
  %42 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %39
  %43 = load double, ptr %42, align 8, !invariant.load !3
  %44 = fmul double %41, %43
  %45 = call double @region_0_1_clone_reduce_sum_2_0(double %38, double %44)
  %46 = add i32 %4, 192
  %47 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %46
  %48 = load double, ptr %47, align 8, !invariant.load !3
  %49 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %46
  %50 = load double, ptr %49, align 8, !invariant.load !3
  %51 = fmul double %48, %50
  %52 = call double @region_0_1_clone_reduce_sum_2_0(double %45, double %51)
  %53 = add i32 %4, 224
  %54 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %53
  %55 = load double, ptr %54, align 8, !invariant.load !3
  %56 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %53
  %57 = load double, ptr %56, align 8, !invariant.load !3
  %58 = fmul double %55, %57
  %59 = call double @region_0_1_clone_reduce_sum_2_0(double %52, double %58)
  %60 = add i32 %4, 256
  %61 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %60
  %62 = load double, ptr %61, align 8, !invariant.load !3
  %63 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %60
  %64 = load double, ptr %63, align 8, !invariant.load !3
  %65 = fmul double %62, %64
  %66 = call double @region_0_1_clone_reduce_sum_2_0(double %59, double %65)
  %67 = add i32 %4, 288
  %68 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %67
  %69 = load double, ptr %68, align 8, !invariant.load !3
  %70 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %67
  %71 = load double, ptr %70, align 8, !invariant.load !3
  %72 = fmul double %69, %71
  %73 = call double @region_0_1_clone_reduce_sum_2_0(double %66, double %72)
  %74 = add i32 %4, 320
  %75 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %74
  %76 = load double, ptr %75, align 8, !invariant.load !3
  %77 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %74
  %78 = load double, ptr %77, align 8, !invariant.load !3
  %79 = fmul double %76, %78
  %80 = call double @region_0_1_clone_reduce_sum_2_0(double %73, double %79)
  %81 = add i32 %4, 352
  %82 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %81
  %83 = load double, ptr %82, align 8, !invariant.load !3
  %84 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %81
  %85 = load double, ptr %84, align 8, !invariant.load !3
  %86 = fmul double %83, %85
  %87 = call double @region_0_1_clone_reduce_sum_2_0(double %80, double %86)
  %88 = add i32 %4, 384
  %89 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %88
  %90 = load double, ptr %89, align 8, !invariant.load !3
  %91 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %88
  %92 = load double, ptr %91, align 8, !invariant.load !3
  %93 = fmul double %90, %92
  %94 = call double @region_0_1_clone_reduce_sum_2_0(double %87, double %93)
  %95 = add i32 %4, 416
  %96 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %95
  %97 = load double, ptr %96, align 8, !invariant.load !3
  %98 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %95
  %99 = load double, ptr %98, align 8, !invariant.load !3
  %100 = fmul double %97, %99
  %101 = call double @region_0_1_clone_reduce_sum_2_0(double %94, double %100)
  %102 = add i32 %4, 448
  %103 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %102
  %104 = load double, ptr %103, align 8, !invariant.load !3
  %105 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %102
  %106 = load double, ptr %105, align 8, !invariant.load !3
  %107 = fmul double %104, %106
  %108 = call double @region_0_1_clone_reduce_sum_2_0(double %101, double %107)
  %109 = add i32 %4, 480
  %110 = getelementptr inbounds [512 x double], ptr %0, i32 0, i32 %109
  %111 = load double, ptr %110, align 8, !invariant.load !3
  %112 = getelementptr inbounds [512 x double], ptr %1, i32 0, i32 %109
  %113 = load double, ptr %112, align 8, !invariant.load !3
  %114 = fmul double %111, %113
  %115 = call double @region_0_1_clone_reduce_sum_2_0(double %108, double %114)
  %116 = bitcast double %115 to i64
  %117 = bitcast i64 %116 to <2 x i32>
  %118 = extractelement <2 x i32> %117, i32 0
  %119 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %118, i32 16, i32 31)
  %120 = insertelement <2 x i32> undef, i32 %119, i32 0
  %121 = extractelement <2 x i32> %117, i32 1
  %122 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %121, i32 16, i32 31)
  %123 = insertelement <2 x i32> %120, i32 %122, i32 1
  %124 = bitcast <2 x i32> %123 to double
  %125 = call double @region_0_1_clone_reduce_sum_2_0(double %115, double %124)
  %126 = bitcast double %125 to i64
  %127 = bitcast i64 %126 to <2 x i32>
  %128 = extractelement <2 x i32> %127, i32 0
  %129 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %128, i32 8, i32 31)
  %130 = insertelement <2 x i32> undef, i32 %129, i32 0
  %131 = extractelement <2 x i32> %127, i32 1
  %132 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %131, i32 8, i32 31)
  %133 = insertelement <2 x i32> %130, i32 %132, i32 1
  %134 = bitcast <2 x i32> %133 to double
  %135 = call double @region_0_1_clone_reduce_sum_2_0(double %125, double %134)
  %136 = bitcast double %135 to i64
  %137 = bitcast i64 %136 to <2 x i32>
  %138 = extractelement <2 x i32> %137, i32 0
  %139 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %138, i32 4, i32 31)
  %140 = insertelement <2 x i32> undef, i32 %139, i32 0
  %141 = extractelement <2 x i32> %137, i32 1
  %142 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %141, i32 4, i32 31)
  %143 = insertelement <2 x i32> %140, i32 %142, i32 1
  %144 = bitcast <2 x i32> %143 to double
  %145 = call double @region_0_1_clone_reduce_sum_2_0(double %135, double %144)
  %146 = bitcast double %145 to i64
  %147 = bitcast i64 %146 to <2 x i32>
  %148 = extractelement <2 x i32> %147, i32 0
  %149 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %148, i32 2, i32 31)
  %150 = insertelement <2 x i32> undef, i32 %149, i32 0
  %151 = extractelement <2 x i32> %147, i32 1
  %152 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %151, i32 2, i32 31)
  %153 = insertelement <2 x i32> %150, i32 %152, i32 1
  %154 = bitcast <2 x i32> %153 to double
  %155 = call double @region_0_1_clone_reduce_sum_2_0(double %145, double %154)
  %156 = bitcast double %155 to i64
  %157 = bitcast i64 %156 to <2 x i32>
  %158 = extractelement <2 x i32> %157, i32 0
  %159 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %158, i32 1, i32 31)
  %160 = insertelement <2 x i32> undef, i32 %159, i32 0
  %161 = extractelement <2 x i32> %157, i32 1
  %162 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %161, i32 1, i32 31)
  %163 = insertelement <2 x i32> %160, i32 %162, i32 1
  %164 = bitcast <2 x i32> %163 to double
  %165 = call double @region_0_1_clone_reduce_sum_2_0(double %155, double %164)
  %166 = icmp eq i32 %4, 0
  br i1 %166, label %167, label %169

167:                                              ; preds = %3
  %168 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  store double %165, ptr %168, align 8
  br label %169

169:                                              ; preds = %167, %3
  ret void
}

define internal double @region_0_1_clone_reduce_sum_2_0(double %0, double %1) {
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
!2 = !{i32 0, i32 64}
!3 = !{}
!4 = !{i32 0, i32 32}
