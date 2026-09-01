; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x double] undef

declare double @__nv_sqrt(double)

define ptx_kernel void @loop_multiply_fusion_1(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(16) %1) #0 {
  %3 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %4 = load double, ptr %3, align 8, !invariant.load !1
  %5 = fdiv double 3.276800e+04, %4
  %6 = call double @__nv_sqrt(double %5)
  %7 = fmul double %6, 0x4066A09E667F3BCD
  %8 = fmul double %6, 0.000000e+00
  %9 = fadd double %8, 0.000000e+00
  %10 = insertvalue { double, double } poison, double %7, 0
  %11 = insertvalue { double, double } %10, double %9, 1
  %12 = getelementptr inbounds [1 x { double, double }], ptr %1, i32 0, i32 0
  store { double, double } %11, ptr %12, align 8
  ret void
}

define ptx_kernel void @input_reduce_fusion(ptr noalias align 256 dereferenceable(16777216) %0, ptr noalias align 256 dereferenceable(16) %1, ptr noalias align 16 dereferenceable(128) %2, ptr noalias align 256 dereferenceable(262144) %3) #1 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %7 = udiv i32 %5, 64
  %8 = udiv i32 %5, 32
  %9 = mul i32 %8, 32768
  %10 = mul i32 %6, 32
  %11 = add i32 %9, %10
  %12 = urem i32 %5, 32
  %13 = add i32 %11, %12
  %14 = getelementptr inbounds [1048576 x { double, double }], ptr %0, i32 0, i32 %13
  %15 = load { double, double }, ptr %14, align 8, !invariant.load !1
  %16 = getelementptr inbounds [1 x { double, double }], ptr %1, i32 0, i32 0
  %17 = load { double, double }, ptr %16, align 8, !invariant.load !1
  %18 = extractvalue { double, double } %15, 0
  %19 = extractvalue { double, double } %15, 1
  %20 = extractvalue { double, double } %17, 0
  %21 = extractvalue { double, double } %17, 1
  %22 = fmul double %18, %20
  %23 = fmul double %19, %21
  %24 = fsub double %22, %23
  %25 = fmul double %19, %20
  %26 = fmul double %18, %21
  %27 = fadd double %25, %26
  %28 = fneg double %27
  %29 = fmul double %24, %24
  %30 = fmul double %28, %27
  %31 = fsub double %29, %30
  %32 = getelementptr inbounds [16 x double], ptr %2, i32 0, i32 %7
  %33 = load double, ptr %32, align 8, !invariant.load !1
  %34 = fmul double %33, %31
  %35 = call double @region_1_3_reduce_sum_5_0(double 0.000000e+00, double %34)
  %36 = mul i32 %12, 33
  %37 = add i32 %36, %8
  %38 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %37
  store double %35, ptr %38, align 8
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %39 = mul i32 %8, 33
  %40 = add i32 %39, %12
  %41 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %40
  %42 = load double, ptr %41, align 8
  %43 = bitcast double %42 to i64
  %44 = bitcast i64 %43 to <2 x i32>
  %45 = extractelement <2 x i32> %44, i32 0
  %46 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %45, i32 16, i32 31)
  %47 = insertelement <2 x i32> undef, i32 %46, i32 0
  %48 = extractelement <2 x i32> %44, i32 1
  %49 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %48, i32 16, i32 31)
  %50 = insertelement <2 x i32> %47, i32 %49, i32 1
  %51 = bitcast <2 x i32> %50 to double
  %52 = call double @region_1_3_reduce_sum_5_0(double %42, double %51)
  %53 = bitcast double %52 to i64
  %54 = bitcast i64 %53 to <2 x i32>
  %55 = extractelement <2 x i32> %54, i32 0
  %56 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %55, i32 8, i32 31)
  %57 = insertelement <2 x i32> undef, i32 %56, i32 0
  %58 = extractelement <2 x i32> %54, i32 1
  %59 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %58, i32 8, i32 31)
  %60 = insertelement <2 x i32> %57, i32 %59, i32 1
  %61 = bitcast <2 x i32> %60 to double
  %62 = call double @region_1_3_reduce_sum_5_0(double %52, double %61)
  %63 = bitcast double %62 to i64
  %64 = bitcast i64 %63 to <2 x i32>
  %65 = extractelement <2 x i32> %64, i32 0
  %66 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %65, i32 4, i32 31)
  %67 = insertelement <2 x i32> undef, i32 %66, i32 0
  %68 = extractelement <2 x i32> %64, i32 1
  %69 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %68, i32 4, i32 31)
  %70 = insertelement <2 x i32> %67, i32 %69, i32 1
  %71 = bitcast <2 x i32> %70 to double
  %72 = call double @region_1_3_reduce_sum_5_0(double %62, double %71)
  %73 = bitcast double %72 to i64
  %74 = bitcast i64 %73 to <2 x i32>
  %75 = extractelement <2 x i32> %74, i32 0
  %76 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %75, i32 2, i32 31)
  %77 = insertelement <2 x i32> undef, i32 %76, i32 0
  %78 = extractelement <2 x i32> %74, i32 1
  %79 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %78, i32 2, i32 31)
  %80 = insertelement <2 x i32> %77, i32 %79, i32 1
  %81 = bitcast <2 x i32> %80 to double
  %82 = call double @region_1_3_reduce_sum_5_0(double %72, double %81)
  %83 = bitcast double %82 to i64
  %84 = bitcast i64 %83 to <2 x i32>
  %85 = extractelement <2 x i32> %84, i32 0
  %86 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %85, i32 1, i32 31)
  %87 = insertelement <2 x i32> undef, i32 %86, i32 0
  %88 = extractelement <2 x i32> %84, i32 1
  %89 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %88, i32 1, i32 31)
  %90 = insertelement <2 x i32> %87, i32 %89, i32 1
  %91 = bitcast <2 x i32> %90 to double
  %92 = call double @region_1_3_reduce_sum_5_0(double %82, double %91)
  %93 = icmp eq i32 %12, 0
  %94 = icmp sle i32 %5, 992
  %95 = and i1 %93, %94
  br i1 %95, label %96, label %99

96:                                               ; preds = %4
  %97 = add i32 %10, %8
  %98 = getelementptr inbounds [32768 x double], ptr %3, i32 0, i32 %97
  store double %92, ptr %98, align 8
  br label %99

99:                                               ; preds = %96, %4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

define internal double @region_1_3_reduce_sum_5_0(double %0, double %1) {
  %3 = fadd nsz double %0, %1
  ret double %3
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #4

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 dereferenceable(262144) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 16 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(262144) %3) #5 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %7 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %8 = load double, ptr %7, align 8, !invariant.load !1
  %9 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  %10 = load double, ptr %9, align 8, !invariant.load !1
  %11 = fmul double %8, %10
  %12 = mul i32 %5, 128
  %13 = add i32 %12, %6
  %14 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %13
  %15 = load double, ptr %14, align 8
  %16 = fmul double %11, %15
  store double %16, ptr %14, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }
attributes #1 = { "nvvm.reqntid"="1024,1,1" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nounwind }
attributes #4 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #5 = { "nvvm.reqntid"="128,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
!2 = !{i32 0, i32 1024}
!3 = !{i32 0, i32 256}
!4 = !{i32 0, i32 128}
