; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [24 x i64] undef
@shared_01 = private addrspace(3) global [24 x double] undef

declare double @__nv_fabs(double)

define ptx_kernel void @loop_compare_not_select_fusion(ptr noalias align 16 dereferenceable(98304) %0, ptr noalias align 256 dereferenceable(12288) %1, ptr noalias align 256 dereferenceable(98304) %2, ptr noalias align 256 dereferenceable(12288) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = mul i32 %5, 128
  %8 = add i32 %7, %6
  %9 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %8
  %10 = load double, ptr %9, align 8, !invariant.load !3
  %11 = call double @__nv_fabs(double %10)
  %12 = fcmp one double %11, 0x7FF0000000000000
  %13 = zext i1 %12 to i8
  %14 = icmp eq i8 %13, 0
  %15 = zext i1 %14 to i8
  %16 = select i1 %12, double %11, double 0.000000e+00
  %17 = fcmp une double %10, %10
  %18 = zext i1 %17 to i8
  %19 = getelementptr inbounds [12288 x i8], ptr %1, i32 0, i32 %8
  store i8 %15, ptr %19, align 1
  %20 = getelementptr inbounds [12288 x double], ptr %2, i32 0, i32 %8
  store double %16, ptr %20, align 8
  %21 = getelementptr inbounds [12288 x i8], ptr %3, i32 0, i32 %8
  store i8 %18, ptr %21, align 1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

define ptx_kernel void @input_reduce_fusion_2(ptr noalias align 256 dereferenceable(12288) %0, ptr noalias align 256 dereferenceable(8) %1) #2 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %4 = mul i32 %3, 4
  %5 = getelementptr inbounds [12288 x i8], ptr %0, i32 0, i32 %4
  %6 = load <4 x i8>, ptr %5, align 1, !invariant.load !3
  %7 = extractelement <4 x i8> %6, i64 0
  %8 = sext i8 %7 to i64
  %9 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 0, i64 %8)
  %10 = extractelement <4 x i8> %6, i64 1
  %11 = sext i8 %10 to i64
  %12 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %9, i64 %11)
  %13 = extractelement <4 x i8> %6, i64 2
  %14 = sext i8 %13 to i64
  %15 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %12, i64 %14)
  %16 = extractelement <4 x i8> %6, i64 3
  %17 = sext i8 %16 to i64
  %18 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %15, i64 %17)
  %19 = add i32 %4, 3072
  %20 = getelementptr inbounds [12288 x i8], ptr %0, i32 0, i32 %19
  %21 = load <4 x i8>, ptr %20, align 1, !invariant.load !3
  %22 = extractelement <4 x i8> %21, i64 0
  %23 = sext i8 %22 to i64
  %24 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %18, i64 %23)
  %25 = extractelement <4 x i8> %21, i64 1
  %26 = sext i8 %25 to i64
  %27 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %24, i64 %26)
  %28 = extractelement <4 x i8> %21, i64 2
  %29 = sext i8 %28 to i64
  %30 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %27, i64 %29)
  %31 = extractelement <4 x i8> %21, i64 3
  %32 = sext i8 %31 to i64
  %33 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %30, i64 %32)
  %34 = add i32 %4, 6144
  %35 = getelementptr inbounds [12288 x i8], ptr %0, i32 0, i32 %34
  %36 = load <4 x i8>, ptr %35, align 1, !invariant.load !3
  %37 = extractelement <4 x i8> %36, i64 0
  %38 = sext i8 %37 to i64
  %39 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %33, i64 %38)
  %40 = extractelement <4 x i8> %36, i64 1
  %41 = sext i8 %40 to i64
  %42 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %39, i64 %41)
  %43 = extractelement <4 x i8> %36, i64 2
  %44 = sext i8 %43 to i64
  %45 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %42, i64 %44)
  %46 = extractelement <4 x i8> %36, i64 3
  %47 = sext i8 %46 to i64
  %48 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %45, i64 %47)
  %49 = add i32 %4, 9216
  %50 = getelementptr inbounds [12288 x i8], ptr %0, i32 0, i32 %49
  %51 = load <4 x i8>, ptr %50, align 1, !invariant.load !3
  %52 = extractelement <4 x i8> %51, i64 0
  %53 = sext i8 %52 to i64
  %54 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %48, i64 %53)
  %55 = extractelement <4 x i8> %51, i64 1
  %56 = sext i8 %55 to i64
  %57 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %54, i64 %56)
  %58 = extractelement <4 x i8> %51, i64 2
  %59 = sext i8 %58 to i64
  %60 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %57, i64 %59)
  %61 = extractelement <4 x i8> %51, i64 3
  %62 = sext i8 %61 to i64
  %63 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %60, i64 %62)
  %64 = bitcast i64 %63 to <2 x i32>
  %65 = extractelement <2 x i32> %64, i32 0
  %66 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %65, i32 16, i32 31)
  %67 = insertelement <2 x i32> undef, i32 %66, i32 0
  %68 = extractelement <2 x i32> %64, i32 1
  %69 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %68, i32 16, i32 31)
  %70 = insertelement <2 x i32> %67, i32 %69, i32 1
  %71 = bitcast <2 x i32> %70 to i64
  %72 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %63, i64 %71)
  %73 = bitcast i64 %72 to <2 x i32>
  %74 = extractelement <2 x i32> %73, i32 0
  %75 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %74, i32 8, i32 31)
  %76 = insertelement <2 x i32> undef, i32 %75, i32 0
  %77 = extractelement <2 x i32> %73, i32 1
  %78 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %77, i32 8, i32 31)
  %79 = insertelement <2 x i32> %76, i32 %78, i32 1
  %80 = bitcast <2 x i32> %79 to i64
  %81 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %72, i64 %80)
  %82 = bitcast i64 %81 to <2 x i32>
  %83 = extractelement <2 x i32> %82, i32 0
  %84 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %83, i32 4, i32 31)
  %85 = insertelement <2 x i32> undef, i32 %84, i32 0
  %86 = extractelement <2 x i32> %82, i32 1
  %87 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %86, i32 4, i32 31)
  %88 = insertelement <2 x i32> %85, i32 %87, i32 1
  %89 = bitcast <2 x i32> %88 to i64
  %90 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %81, i64 %89)
  %91 = bitcast i64 %90 to <2 x i32>
  %92 = extractelement <2 x i32> %91, i32 0
  %93 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %92, i32 2, i32 31)
  %94 = insertelement <2 x i32> undef, i32 %93, i32 0
  %95 = extractelement <2 x i32> %91, i32 1
  %96 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %95, i32 2, i32 31)
  %97 = insertelement <2 x i32> %94, i32 %96, i32 1
  %98 = bitcast <2 x i32> %97 to i64
  %99 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %90, i64 %98)
  %100 = bitcast i64 %99 to <2 x i32>
  %101 = extractelement <2 x i32> %100, i32 0
  %102 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %101, i32 1, i32 31)
  %103 = insertelement <2 x i32> undef, i32 %102, i32 0
  %104 = extractelement <2 x i32> %100, i32 1
  %105 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %104, i32 1, i32 31)
  %106 = insertelement <2 x i32> %103, i32 %105, i32 1
  %107 = bitcast <2 x i32> %106 to i64
  %108 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %99, i64 %107)
  %109 = urem i32 %3, 32
  %110 = icmp eq i32 %109, 0
  br i1 %110, label %111, label %114

111:                                              ; preds = %2
  %112 = udiv i32 %3, 32
  %113 = getelementptr inbounds [24 x i64], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %112
  store i64 %108, ptr %113, align 4
  br label %114

114:                                              ; preds = %111, %2
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %115 = icmp sle i32 %3, 31
  br i1 %115, label %116, label %175

116:                                              ; preds = %114
  %117 = icmp sle i32 %3, 23
  br i1 %117, label %118, label %121

118:                                              ; preds = %116
  %119 = getelementptr inbounds [24 x i64], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %3
  %120 = load i64, ptr %119, align 4
  br label %122

121:                                              ; preds = %116
  br label %122

122:                                              ; preds = %118, %121
  %123 = phi i64 [ 0, %121 ], [ %120, %118 ]
  br label %124

124:                                              ; preds = %122
  %125 = bitcast i64 %123 to <2 x i32>
  %126 = extractelement <2 x i32> %125, i32 0
  %127 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %126, i32 16, i32 31)
  %128 = insertelement <2 x i32> undef, i32 %127, i32 0
  %129 = extractelement <2 x i32> %125, i32 1
  %130 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %129, i32 16, i32 31)
  %131 = insertelement <2 x i32> %128, i32 %130, i32 1
  %132 = bitcast <2 x i32> %131 to i64
  %133 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %123, i64 %132)
  %134 = bitcast i64 %133 to <2 x i32>
  %135 = extractelement <2 x i32> %134, i32 0
  %136 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %135, i32 8, i32 31)
  %137 = insertelement <2 x i32> undef, i32 %136, i32 0
  %138 = extractelement <2 x i32> %134, i32 1
  %139 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %138, i32 8, i32 31)
  %140 = insertelement <2 x i32> %137, i32 %139, i32 1
  %141 = bitcast <2 x i32> %140 to i64
  %142 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %133, i64 %141)
  %143 = bitcast i64 %142 to <2 x i32>
  %144 = extractelement <2 x i32> %143, i32 0
  %145 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %144, i32 4, i32 31)
  %146 = insertelement <2 x i32> undef, i32 %145, i32 0
  %147 = extractelement <2 x i32> %143, i32 1
  %148 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %147, i32 4, i32 31)
  %149 = insertelement <2 x i32> %146, i32 %148, i32 1
  %150 = bitcast <2 x i32> %149 to i64
  %151 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %142, i64 %150)
  %152 = bitcast i64 %151 to <2 x i32>
  %153 = extractelement <2 x i32> %152, i32 0
  %154 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %153, i32 2, i32 31)
  %155 = insertelement <2 x i32> undef, i32 %154, i32 0
  %156 = extractelement <2 x i32> %152, i32 1
  %157 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %156, i32 2, i32 31)
  %158 = insertelement <2 x i32> %155, i32 %157, i32 1
  %159 = bitcast <2 x i32> %158 to i64
  %160 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %151, i64 %159)
  %161 = bitcast i64 %160 to <2 x i32>
  %162 = extractelement <2 x i32> %161, i32 0
  %163 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %162, i32 1, i32 31)
  %164 = insertelement <2 x i32> undef, i32 %163, i32 0
  %165 = extractelement <2 x i32> %161, i32 1
  %166 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %165, i32 1, i32 31)
  %167 = insertelement <2 x i32> %164, i32 %166, i32 1
  %168 = bitcast <2 x i32> %167 to i64
  %169 = call i64 @region_0_1_clone_reduce_sum_18_0(i64 %160, i64 %168)
  %170 = sitofp i64 %169 to double
  %171 = icmp eq i32 %3, 0
  br i1 %171, label %172, label %174

172:                                              ; preds = %124
  %173 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  store double %170, ptr %173, align 8
  br label %174

174:                                              ; preds = %172, %124
  br label %175

175:                                              ; preds = %174, %114
  ret void
}

define internal i64 @region_0_1_clone_reduce_sum_18_0(i64 %0, i64 %1) {
  %3 = add i64 %0, %1
  ret i64 %3
}

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #3

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #4

define ptx_kernel void @input_reduce_fusion(ptr noalias align 256 dereferenceable(98304) %0, ptr noalias align 256 dereferenceable(8) %1) #2 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %4 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %3
  %5 = load double, ptr %4, align 8, !invariant.load !3
  %6 = call double @region_2_3_reduce_max_2_0(double 0xFFF0000000000000, double %5)
  %7 = add i32 %3, 768
  %8 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %7
  %9 = load double, ptr %8, align 8, !invariant.load !3
  %10 = call double @region_2_3_reduce_max_2_0(double %6, double %9)
  %11 = add i32 %3, 1536
  %12 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %11
  %13 = load double, ptr %12, align 8, !invariant.load !3
  %14 = call double @region_2_3_reduce_max_2_0(double %10, double %13)
  %15 = add i32 %3, 2304
  %16 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %15
  %17 = load double, ptr %16, align 8, !invariant.load !3
  %18 = call double @region_2_3_reduce_max_2_0(double %14, double %17)
  %19 = add i32 %3, 3072
  %20 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %19
  %21 = load double, ptr %20, align 8, !invariant.load !3
  %22 = call double @region_2_3_reduce_max_2_0(double %18, double %21)
  %23 = add i32 %3, 3840
  %24 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %23
  %25 = load double, ptr %24, align 8, !invariant.load !3
  %26 = call double @region_2_3_reduce_max_2_0(double %22, double %25)
  %27 = add i32 %3, 4608
  %28 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %27
  %29 = load double, ptr %28, align 8, !invariant.load !3
  %30 = call double @region_2_3_reduce_max_2_0(double %26, double %29)
  %31 = add i32 %3, 5376
  %32 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %31
  %33 = load double, ptr %32, align 8, !invariant.load !3
  %34 = call double @region_2_3_reduce_max_2_0(double %30, double %33)
  %35 = add i32 %3, 6144
  %36 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %35
  %37 = load double, ptr %36, align 8, !invariant.load !3
  %38 = call double @region_2_3_reduce_max_2_0(double %34, double %37)
  %39 = add i32 %3, 6912
  %40 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %39
  %41 = load double, ptr %40, align 8, !invariant.load !3
  %42 = call double @region_2_3_reduce_max_2_0(double %38, double %41)
  %43 = add i32 %3, 7680
  %44 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %43
  %45 = load double, ptr %44, align 8, !invariant.load !3
  %46 = call double @region_2_3_reduce_max_2_0(double %42, double %45)
  %47 = add i32 %3, 8448
  %48 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %47
  %49 = load double, ptr %48, align 8, !invariant.load !3
  %50 = call double @region_2_3_reduce_max_2_0(double %46, double %49)
  %51 = add i32 %3, 9216
  %52 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %51
  %53 = load double, ptr %52, align 8, !invariant.load !3
  %54 = call double @region_2_3_reduce_max_2_0(double %50, double %53)
  %55 = add i32 %3, 9984
  %56 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %55
  %57 = load double, ptr %56, align 8, !invariant.load !3
  %58 = call double @region_2_3_reduce_max_2_0(double %54, double %57)
  %59 = add i32 %3, 10752
  %60 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %59
  %61 = load double, ptr %60, align 8, !invariant.load !3
  %62 = call double @region_2_3_reduce_max_2_0(double %58, double %61)
  %63 = add i32 %3, 11520
  %64 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %63
  %65 = load double, ptr %64, align 8, !invariant.load !3
  %66 = call double @region_2_3_reduce_max_2_0(double %62, double %65)
  %67 = bitcast double %66 to i64
  %68 = bitcast i64 %67 to <2 x i32>
  %69 = extractelement <2 x i32> %68, i32 0
  %70 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %69, i32 16, i32 31)
  %71 = insertelement <2 x i32> undef, i32 %70, i32 0
  %72 = extractelement <2 x i32> %68, i32 1
  %73 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %72, i32 16, i32 31)
  %74 = insertelement <2 x i32> %71, i32 %73, i32 1
  %75 = bitcast <2 x i32> %74 to double
  %76 = call double @region_2_3_reduce_max_2_0(double %66, double %75)
  %77 = bitcast double %76 to i64
  %78 = bitcast i64 %77 to <2 x i32>
  %79 = extractelement <2 x i32> %78, i32 0
  %80 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %79, i32 8, i32 31)
  %81 = insertelement <2 x i32> undef, i32 %80, i32 0
  %82 = extractelement <2 x i32> %78, i32 1
  %83 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %82, i32 8, i32 31)
  %84 = insertelement <2 x i32> %81, i32 %83, i32 1
  %85 = bitcast <2 x i32> %84 to double
  %86 = call double @region_2_3_reduce_max_2_0(double %76, double %85)
  %87 = bitcast double %86 to i64
  %88 = bitcast i64 %87 to <2 x i32>
  %89 = extractelement <2 x i32> %88, i32 0
  %90 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %89, i32 4, i32 31)
  %91 = insertelement <2 x i32> undef, i32 %90, i32 0
  %92 = extractelement <2 x i32> %88, i32 1
  %93 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %92, i32 4, i32 31)
  %94 = insertelement <2 x i32> %91, i32 %93, i32 1
  %95 = bitcast <2 x i32> %94 to double
  %96 = call double @region_2_3_reduce_max_2_0(double %86, double %95)
  %97 = bitcast double %96 to i64
  %98 = bitcast i64 %97 to <2 x i32>
  %99 = extractelement <2 x i32> %98, i32 0
  %100 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %99, i32 2, i32 31)
  %101 = insertelement <2 x i32> undef, i32 %100, i32 0
  %102 = extractelement <2 x i32> %98, i32 1
  %103 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %102, i32 2, i32 31)
  %104 = insertelement <2 x i32> %101, i32 %103, i32 1
  %105 = bitcast <2 x i32> %104 to double
  %106 = call double @region_2_3_reduce_max_2_0(double %96, double %105)
  %107 = bitcast double %106 to i64
  %108 = bitcast i64 %107 to <2 x i32>
  %109 = extractelement <2 x i32> %108, i32 0
  %110 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %109, i32 1, i32 31)
  %111 = insertelement <2 x i32> undef, i32 %110, i32 0
  %112 = extractelement <2 x i32> %108, i32 1
  %113 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %112, i32 1, i32 31)
  %114 = insertelement <2 x i32> %111, i32 %113, i32 1
  %115 = bitcast <2 x i32> %114 to double
  %116 = call double @region_2_3_reduce_max_2_0(double %106, double %115)
  %117 = urem i32 %3, 32
  %118 = icmp eq i32 %117, 0
  br i1 %118, label %119, label %122

119:                                              ; preds = %2
  %120 = udiv i32 %3, 32
  %121 = getelementptr inbounds [24 x double], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %120
  store double %116, ptr %121, align 8
  br label %122

122:                                              ; preds = %119, %2
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %123 = icmp sle i32 %3, 31
  br i1 %123, label %124, label %187

124:                                              ; preds = %122
  %125 = icmp sle i32 %3, 23
  br i1 %125, label %126, label %129

126:                                              ; preds = %124
  %127 = getelementptr inbounds [24 x double], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %3
  %128 = load double, ptr %127, align 8
  br label %130

129:                                              ; preds = %124
  br label %130

130:                                              ; preds = %126, %129
  %131 = phi double [ 0xFFF0000000000000, %129 ], [ %128, %126 ]
  br label %132

132:                                              ; preds = %130
  %133 = bitcast double %131 to i64
  %134 = bitcast i64 %133 to <2 x i32>
  %135 = extractelement <2 x i32> %134, i32 0
  %136 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %135, i32 16, i32 31)
  %137 = insertelement <2 x i32> undef, i32 %136, i32 0
  %138 = extractelement <2 x i32> %134, i32 1
  %139 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %138, i32 16, i32 31)
  %140 = insertelement <2 x i32> %137, i32 %139, i32 1
  %141 = bitcast <2 x i32> %140 to double
  %142 = call double @region_2_3_reduce_max_2_0(double %131, double %141)
  %143 = bitcast double %142 to i64
  %144 = bitcast i64 %143 to <2 x i32>
  %145 = extractelement <2 x i32> %144, i32 0
  %146 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %145, i32 8, i32 31)
  %147 = insertelement <2 x i32> undef, i32 %146, i32 0
  %148 = extractelement <2 x i32> %144, i32 1
  %149 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %148, i32 8, i32 31)
  %150 = insertelement <2 x i32> %147, i32 %149, i32 1
  %151 = bitcast <2 x i32> %150 to double
  %152 = call double @region_2_3_reduce_max_2_0(double %142, double %151)
  %153 = bitcast double %152 to i64
  %154 = bitcast i64 %153 to <2 x i32>
  %155 = extractelement <2 x i32> %154, i32 0
  %156 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %155, i32 4, i32 31)
  %157 = insertelement <2 x i32> undef, i32 %156, i32 0
  %158 = extractelement <2 x i32> %154, i32 1
  %159 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %158, i32 4, i32 31)
  %160 = insertelement <2 x i32> %157, i32 %159, i32 1
  %161 = bitcast <2 x i32> %160 to double
  %162 = call double @region_2_3_reduce_max_2_0(double %152, double %161)
  %163 = bitcast double %162 to i64
  %164 = bitcast i64 %163 to <2 x i32>
  %165 = extractelement <2 x i32> %164, i32 0
  %166 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %165, i32 2, i32 31)
  %167 = insertelement <2 x i32> undef, i32 %166, i32 0
  %168 = extractelement <2 x i32> %164, i32 1
  %169 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %168, i32 2, i32 31)
  %170 = insertelement <2 x i32> %167, i32 %169, i32 1
  %171 = bitcast <2 x i32> %170 to double
  %172 = call double @region_2_3_reduce_max_2_0(double %162, double %171)
  %173 = bitcast double %172 to i64
  %174 = bitcast i64 %173 to <2 x i32>
  %175 = extractelement <2 x i32> %174, i32 0
  %176 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %175, i32 1, i32 31)
  %177 = insertelement <2 x i32> undef, i32 %176, i32 0
  %178 = extractelement <2 x i32> %174, i32 1
  %179 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %178, i32 1, i32 31)
  %180 = insertelement <2 x i32> %177, i32 %179, i32 1
  %181 = bitcast <2 x i32> %180 to double
  %182 = call double @region_2_3_reduce_max_2_0(double %172, double %181)
  %183 = icmp eq i32 %3, 0
  br i1 %183, label %184, label %186

184:                                              ; preds = %132
  %185 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  store double %182, ptr %185, align 8
  br label %186

186:                                              ; preds = %184, %132
  br label %187

187:                                              ; preds = %186, %122
  ret void
}

define internal double @region_2_3_reduce_max_2_0(double %0, double %1) {
  %3 = call double @llvm.maximum.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #5

define ptx_kernel void @input_concatenate_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(24) %3) #6 {
  %5 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  %6 = load double, ptr %5, align 8, !invariant.load !3
  %7 = getelementptr inbounds [3 x double], ptr %3, i32 0, i32 0
  store double %6, ptr %7, align 8
  %8 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %9 = load double, ptr %8, align 8, !invariant.load !3
  %10 = getelementptr inbounds [3 x double], ptr %3, i32 0, i32 1
  store double %9, ptr %10, align 8
  %11 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %12 = load double, ptr %11, align 8, !invariant.load !3
  %13 = getelementptr inbounds [3 x double], ptr %3, i32 0, i32 2
  store double %12, ptr %13, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { "nvvm.reqntid"="768,1,1" }
attributes #3 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #4 = { convergent nocallback nounwind }
attributes #5 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 96}
!2 = !{i32 0, i32 128}
!3 = !{}
!4 = !{i32 0, i32 768}
