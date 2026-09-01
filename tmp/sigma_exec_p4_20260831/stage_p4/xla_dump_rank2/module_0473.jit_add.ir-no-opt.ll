; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_add_fusion(ptr noalias align 16 dereferenceable(1179648) %0, ptr noalias align 16 dereferenceable(4718592) %1, ptr noalias align 256 dereferenceable(16) %2, ptr noalias align 256 dereferenceable(4) %3, ptr noalias align 256 dereferenceable(16) %4, ptr noalias align 256 dereferenceable(1179648) %5) #0 {
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %9 = mul i32 %7, 128
  %10 = add i32 %9, %8
  %11 = udiv i32 %10, 12
  %12 = urem i32 %11, 12
  %13 = urem i32 %10, 12
  %14 = getelementptr inbounds [1 x i32], ptr %3, i32 0, i32 0
  %15 = load i32, ptr %14, align 4, !invariant.load !3
  %16 = call i32 @llvm.umin.i32(i32 %15, i32 3)
  %17 = getelementptr inbounds [4 x i32], ptr %4, i32 0, i32 %16
  %18 = load i32, ptr %17, align 4, !invariant.load !3
  %19 = call i32 @llvm.smin.i32(i32 %18, i32 12)
  %20 = call i32 @llvm.smax.i32(i32 %19, i32 0)
  %21 = add i32 %12, %20
  %22 = getelementptr inbounds [4 x i32], ptr %2, i32 0, i32 %16
  %23 = load i32, ptr %22, align 4, !invariant.load !3
  %24 = call i32 @llvm.smin.i32(i32 %23, i32 12)
  %25 = call i32 @llvm.smax.i32(i32 %24, i32 0)
  %26 = add i32 %13, %25
  %27 = udiv i32 %10, 144
  %28 = mul i32 %27, 576
  %29 = mul i32 %21, 24
  %30 = add i32 %28, %29
  %31 = add i32 %30, %26
  %32 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %31
  %33 = load { double, double }, ptr %32, align 8, !invariant.load !3
  %34 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %10
  %35 = load { double, double }, ptr %34, align 8, !invariant.load !3
  %36 = extractvalue { double, double } %33, 0
  %37 = extractvalue { double, double } %35, 0
  %38 = fadd double %36, %37
  %39 = extractvalue { double, double } %33, 1
  %40 = extractvalue { double, double } %35, 1
  %41 = fadd double %39, %40
  %42 = insertvalue { double, double } poison, double %38, 0
  %43 = insertvalue { double, double } %42, double %41, 1
  %44 = getelementptr inbounds [73728 x { double, double }], ptr %5, i32 0, i32 %10
  store { double, double } %43, ptr %44, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #2

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
