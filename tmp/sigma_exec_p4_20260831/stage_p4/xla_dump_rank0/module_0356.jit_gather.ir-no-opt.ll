; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias align 16 dereferenceable(178361600) %0, ptr noalias align 16 dereferenceable(4) %1, ptr noalias align 256 dereferenceable(44590400) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = icmp sle i32 %7, 696724
  br i1 %8, label %9, label %37

9:                                                ; preds = %3
  %10 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  %11 = load i32, ptr %10, align 4, !invariant.load !3
  %12 = call i32 @llvm.smin.i32(i32 %11, i32 3)
  %13 = call i32 @llvm.smax.i32(i32 %12, i32 0)
  %14 = mul i32 %5, 4
  %15 = mul i32 %4, 512
  %16 = add i32 %14, %15
  %17 = mul i32 %13, 2786900
  %18 = add i32 %16, %17
  %19 = getelementptr inbounds [11147600 x { double, double }], ptr %0, i32 0, i32 %18
  %20 = load { double, double }, ptr %19, align 8, !invariant.load !3
  %21 = getelementptr inbounds [2786900 x { double, double }], ptr %2, i32 0, i32 %16
  store { double, double } %20, ptr %21, align 8
  %22 = add i32 %18, 1
  %23 = getelementptr inbounds [11147600 x { double, double }], ptr %0, i32 0, i32 %22
  %24 = load { double, double }, ptr %23, align 8, !invariant.load !3
  %25 = add i32 %16, 1
  %26 = getelementptr inbounds [2786900 x { double, double }], ptr %2, i32 0, i32 %25
  store { double, double } %24, ptr %26, align 8
  %27 = add i32 %18, 2
  %28 = getelementptr inbounds [11147600 x { double, double }], ptr %0, i32 0, i32 %27
  %29 = load { double, double }, ptr %28, align 8, !invariant.load !3
  %30 = add i32 %16, 2
  %31 = getelementptr inbounds [2786900 x { double, double }], ptr %2, i32 0, i32 %30
  store { double, double } %29, ptr %31, align 8
  %32 = add i32 %18, 3
  %33 = getelementptr inbounds [11147600 x { double, double }], ptr %0, i32 0, i32 %32
  %34 = load { double, double }, ptr %33, align 8, !invariant.load !3
  %35 = add i32 %16, 3
  %36 = getelementptr inbounds [2786900 x { double, double }], ptr %2, i32 0, i32 %35
  store { double, double } %34, ptr %36, align 8
  br label %37

37:                                               ; preds = %9, %3
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 5444}
!2 = !{i32 0, i32 128}
!3 = !{}
