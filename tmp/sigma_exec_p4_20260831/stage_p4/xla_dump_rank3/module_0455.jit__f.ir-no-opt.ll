; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_gather(ptr noalias align 16 dereferenceable(267264) %0, ptr noalias align 256 dereferenceable(2048) %1, ptr noalias align 256 dereferenceable(4718592) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = udiv i32 %7, 144
  %9 = getelementptr inbounds [512 x i32], ptr %1, i32 0, i32 %8
  %10 = load i32, ptr %9, align 4, !invariant.load !3
  %11 = call i32 @llvm.smin.i32(i32 %10, i32 28)
  %12 = call i32 @llvm.smax.i32(i32 %11, i32 0)
  %13 = urem i32 %7, 6
  %14 = mul i32 %13, 4
  %15 = udiv i32 %7, 6
  %16 = urem i32 %15, 24
  %17 = mul i32 %16, 24
  %18 = add i32 %14, %17
  %19 = mul i32 %12, 576
  %20 = add i32 %18, %19
  %21 = getelementptr inbounds [16704 x { double, double }], ptr %0, i32 0, i32 %20
  %22 = load { double, double }, ptr %21, align 8, !invariant.load !3
  %23 = mul i32 %5, 4
  %24 = mul i32 %4, 512
  %25 = add i32 %23, %24
  %26 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %25
  store { double, double } %22, ptr %26, align 8
  %27 = add i32 %20, 1
  %28 = getelementptr inbounds [16704 x { double, double }], ptr %0, i32 0, i32 %27
  %29 = load { double, double }, ptr %28, align 8, !invariant.load !3
  %30 = add i32 %25, 1
  %31 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %30
  store { double, double } %29, ptr %31, align 8
  %32 = add i32 %20, 2
  %33 = getelementptr inbounds [16704 x { double, double }], ptr %0, i32 0, i32 %32
  %34 = load { double, double }, ptr %33, align 8, !invariant.load !3
  %35 = add i32 %25, 2
  %36 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %35
  store { double, double } %34, ptr %36, align 8
  %37 = add i32 %20, 3
  %38 = getelementptr inbounds [16704 x { double, double }], ptr %0, i32 0, i32 %37
  %39 = load { double, double }, ptr %38, align 8, !invariant.load !3
  %40 = add i32 %25, 3
  %41 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %40
  store { double, double } %39, ptr %41, align 8
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
!1 = !{i32 0, i32 576}
!2 = !{i32 0, i32 128}
!3 = !{}
