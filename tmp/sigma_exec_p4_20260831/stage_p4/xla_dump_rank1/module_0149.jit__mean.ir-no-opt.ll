; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x double] undef

define ptx_kernel void @input_reduce_fusion(ptr noalias align 16 dereferenceable(12582912) %0, ptr noalias align 256 dereferenceable(262144) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = udiv i32 %3, 32
  %6 = mul i32 %5, 32768
  %7 = mul i32 %4, 32
  %8 = add i32 %6, %7
  %9 = urem i32 %3, 32
  %10 = add i32 %8, %9
  %11 = getelementptr inbounds [1572864 x double], ptr %0, i32 0, i32 %10
  %12 = load double, ptr %11, align 8, !invariant.load !2
  %13 = call double @region_0_1_reduce_sum_5_0(double 0.000000e+00, double %12)
  %14 = icmp sle i32 %3, 511
  br i1 %14, label %15, label %20

15:                                               ; preds = %2
  %16 = add i32 %10, 1048576
  %17 = getelementptr inbounds [1572864 x double], ptr %0, i32 0, i32 %16
  %18 = load double, ptr %17, align 8, !invariant.load !2
  %19 = call double @region_0_1_reduce_sum_5_0(double %13, double %18)
  br label %21

20:                                               ; preds = %2
  br label %21

21:                                               ; preds = %15, %20
  %22 = phi double [ %13, %20 ], [ %19, %15 ]
  br label %23

23:                                               ; preds = %21
  %24 = mul i32 %9, 33
  %25 = add i32 %24, %5
  %26 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %25
  store double %22, ptr %26, align 8
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %27 = mul i32 %5, 33
  %28 = add i32 %27, %9
  %29 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %28
  %30 = load double, ptr %29, align 8
  %31 = bitcast double %30 to i64
  %32 = bitcast i64 %31 to <2 x i32>
  %33 = extractelement <2 x i32> %32, i32 0
  %34 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %33, i32 16, i32 31)
  %35 = insertelement <2 x i32> undef, i32 %34, i32 0
  %36 = extractelement <2 x i32> %32, i32 1
  %37 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %36, i32 16, i32 31)
  %38 = insertelement <2 x i32> %35, i32 %37, i32 1
  %39 = bitcast <2 x i32> %38 to double
  %40 = call double @region_0_1_reduce_sum_5_0(double %30, double %39)
  %41 = bitcast double %40 to i64
  %42 = bitcast i64 %41 to <2 x i32>
  %43 = extractelement <2 x i32> %42, i32 0
  %44 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %43, i32 8, i32 31)
  %45 = insertelement <2 x i32> undef, i32 %44, i32 0
  %46 = extractelement <2 x i32> %42, i32 1
  %47 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %46, i32 8, i32 31)
  %48 = insertelement <2 x i32> %45, i32 %47, i32 1
  %49 = bitcast <2 x i32> %48 to double
  %50 = call double @region_0_1_reduce_sum_5_0(double %40, double %49)
  %51 = bitcast double %50 to i64
  %52 = bitcast i64 %51 to <2 x i32>
  %53 = extractelement <2 x i32> %52, i32 0
  %54 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %53, i32 4, i32 31)
  %55 = insertelement <2 x i32> undef, i32 %54, i32 0
  %56 = extractelement <2 x i32> %52, i32 1
  %57 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %56, i32 4, i32 31)
  %58 = insertelement <2 x i32> %55, i32 %57, i32 1
  %59 = bitcast <2 x i32> %58 to double
  %60 = call double @region_0_1_reduce_sum_5_0(double %50, double %59)
  %61 = bitcast double %60 to i64
  %62 = bitcast i64 %61 to <2 x i32>
  %63 = extractelement <2 x i32> %62, i32 0
  %64 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %63, i32 2, i32 31)
  %65 = insertelement <2 x i32> undef, i32 %64, i32 0
  %66 = extractelement <2 x i32> %62, i32 1
  %67 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %66, i32 2, i32 31)
  %68 = insertelement <2 x i32> %65, i32 %67, i32 1
  %69 = bitcast <2 x i32> %68 to double
  %70 = call double @region_0_1_reduce_sum_5_0(double %60, double %69)
  %71 = bitcast double %70 to i64
  %72 = bitcast i64 %71 to <2 x i32>
  %73 = extractelement <2 x i32> %72, i32 0
  %74 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %73, i32 1, i32 31)
  %75 = insertelement <2 x i32> undef, i32 %74, i32 0
  %76 = extractelement <2 x i32> %72, i32 1
  %77 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %76, i32 1, i32 31)
  %78 = insertelement <2 x i32> %75, i32 %77, i32 1
  %79 = bitcast <2 x i32> %78 to double
  %80 = call double @region_0_1_reduce_sum_5_0(double %70, double %79)
  %81 = icmp eq i32 %9, 0
  %82 = icmp sle i32 %3, 992
  %83 = and i1 %81, %82
  %84 = add i32 %7, %5
  br i1 %83, label %85, label %87

85:                                               ; preds = %23
  %86 = getelementptr inbounds [32768 x double], ptr %1, i32 0, i32 %84
  store double %80, ptr %86, align 8
  br label %87

87:                                               ; preds = %85, %23
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

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #3

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 dereferenceable(262144) %0, ptr noalias align 256 dereferenceable(262144) %1) #4 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %5 = mul i32 %3, 128
  %6 = add i32 %5, %4
  %7 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %6
  %8 = load double, ptr %7, align 8
  %9 = fmul double %8, 0x3F95555555555555
  store double %9, ptr %7, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="1024,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #4 = { "nvvm.reqntid"="128,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 1024}
!2 = !{}
!3 = !{i32 0, i32 256}
!4 = !{i32 0, i32 128}
