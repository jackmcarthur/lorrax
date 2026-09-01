; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @wrapped_transpose(ptr noalias align 16 dereferenceable(1179648) %0, ptr noalias align 256 dereferenceable(1179648) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = urem i32 %3, 32
  %6 = icmp sle i32 %5, 11
  br i1 %6, label %7, label %53

7:                                                ; preds = %2
  %8 = udiv i32 %3, 32
  %9 = mul i32 %8, 12
  %10 = mul i32 %4, 384
  %11 = add i32 %9, %10
  %12 = add i32 %11, %5
  %13 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %12
  %14 = load { double, double }, ptr %13, align 8, !invariant.load !3
  %15 = mul i32 %5, 33
  %16 = add i32 %15, %8
  %17 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %16
  store { double, double } %14, ptr %17, align 8
  %18 = add i32 %12, 48
  %19 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %18
  %20 = load { double, double }, ptr %19, align 8, !invariant.load !3
  %21 = add i32 %16, 4
  %22 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %21
  store { double, double } %20, ptr %22, align 8
  %23 = add i32 %12, 96
  %24 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %23
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !3
  %26 = add i32 %16, 8
  %27 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %26
  store { double, double } %25, ptr %27, align 8
  %28 = add i32 %12, 144
  %29 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %28
  %30 = load { double, double }, ptr %29, align 8, !invariant.load !3
  %31 = add i32 %16, 12
  %32 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %31
  store { double, double } %30, ptr %32, align 8
  %33 = add i32 %12, 192
  %34 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %33
  %35 = load { double, double }, ptr %34, align 8, !invariant.load !3
  %36 = add i32 %16, 16
  %37 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %36
  store { double, double } %35, ptr %37, align 8
  %38 = add i32 %12, 240
  %39 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %38
  %40 = load { double, double }, ptr %39, align 8, !invariant.load !3
  %41 = add i32 %16, 20
  %42 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %41
  store { double, double } %40, ptr %42, align 8
  %43 = add i32 %12, 288
  %44 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %43
  %45 = load { double, double }, ptr %44, align 8, !invariant.load !3
  %46 = add i32 %16, 24
  %47 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %46
  store { double, double } %45, ptr %47, align 8
  %48 = add i32 %12, 336
  %49 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %48
  %50 = load { double, double }, ptr %49, align 8, !invariant.load !3
  %51 = add i32 %16, 28
  %52 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %51
  store { double, double } %50, ptr %52, align 8
  br label %53

53:                                               ; preds = %7, %2
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %54 = udiv i32 %3, 32
  %55 = mul i32 %54, 33
  %56 = add i32 %55, %5
  %57 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %56
  %58 = load { double, double }, ptr %57, align 8
  %59 = mul i32 %54, 6144
  %60 = mul i32 %4, 32
  %61 = add i32 %59, %60
  %62 = add i32 %61, %5
  %63 = getelementptr inbounds [73728 x { double, double }], ptr %1, i32 0, i32 %62
  store { double, double } %58, ptr %63, align 8
  %64 = add i32 %56, 132
  %65 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %64
  %66 = load { double, double }, ptr %65, align 8
  %67 = add i32 %62, 24576
  %68 = getelementptr inbounds [73728 x { double, double }], ptr %1, i32 0, i32 %67
  store { double, double } %66, ptr %68, align 8
  %69 = add i32 %56, 264
  %70 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %69
  %71 = load { double, double }, ptr %70, align 8
  %72 = add i32 %62, 49152
  %73 = getelementptr inbounds [73728 x { double, double }], ptr %1, i32 0, i32 %72
  store { double, double } %71, ptr %73, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

define ptx_kernel void @loop_transpose_fusion_1(ptr noalias align 256 dereferenceable(2359296) %0, ptr noalias align 256 dereferenceable(2359296) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = mul i32 %3, 128
  %6 = add i32 %5, %4
  %7 = urem i32 %6, 24
  %8 = mul i32 %7, 6144
  %9 = udiv i32 %6, 24
  %10 = urem i32 %9, 512
  %11 = mul i32 %10, 12
  %12 = add i32 %8, %11
  %13 = udiv i32 %3, 96
  %14 = add i32 %12, %13
  %15 = getelementptr inbounds [147456 x { double, double }], ptr %0, i32 0, i32 %14
  %16 = load { double, double }, ptr %15, align 8, !invariant.load !3
  %17 = getelementptr inbounds [147456 x { double, double }], ptr %1, i32 0, i32 %6
  store { double, double } %16, ptr %17, align 8
  ret void
}

define ptx_kernel void @loop_transpose_fusion(ptr noalias align 256 dereferenceable(4718592) %0, ptr noalias align 256 dereferenceable(4718592) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = mul i32 %3, 128
  %6 = add i32 %5, %4
  %7 = urem i32 %6, 6
  %8 = mul i32 %7, 4
  %9 = udiv i32 %6, 6
  %10 = urem i32 %9, 24
  %11 = mul i32 %10, 12288
  %12 = add i32 %8, %11
  %13 = udiv i32 %6, 144
  %14 = mul i32 %13, 24
  %15 = add i32 %12, %14
  %16 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %15
  %17 = load { double, double }, ptr %16, align 8, !invariant.load !3
  %18 = mul i32 %4, 4
  %19 = mul i32 %3, 512
  %20 = add i32 %18, %19
  %21 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %20
  store { double, double } %17, ptr %21, align 8
  %22 = add i32 %15, 1
  %23 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %22
  %24 = load { double, double }, ptr %23, align 8, !invariant.load !3
  %25 = add i32 %20, 1
  %26 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %25
  store { double, double } %24, ptr %26, align 8
  %27 = add i32 %15, 2
  %28 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %27
  %29 = load { double, double }, ptr %28, align 8, !invariant.load !3
  %30 = add i32 %20, 2
  %31 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %30
  store { double, double } %29, ptr %31, align 8
  %32 = add i32 %15, 3
  %33 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %32
  %34 = load { double, double }, ptr %33, align 8, !invariant.load !3
  %35 = add i32 %20, 3
  %36 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %35
  store { double, double } %34, ptr %36, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 128}
!2 = !{i32 0, i32 192}
!3 = !{}
!4 = !{i32 0, i32 1152}
!5 = !{i32 0, i32 576}
