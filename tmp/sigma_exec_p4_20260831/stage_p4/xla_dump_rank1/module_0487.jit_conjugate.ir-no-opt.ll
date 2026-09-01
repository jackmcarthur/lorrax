; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_complex_fusion(ptr noalias align 16 dereferenceable(4718592) %0, ptr noalias align 256 dereferenceable(4718592) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %5 = mul i32 %4, 4
  %6 = mul i32 %3, 512
  %7 = add i32 %5, %6
  %8 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %7
  %9 = load { double, double }, ptr %8, align 8, !invariant.load !3
  %10 = extractvalue { double, double } %9, 1
  %11 = extractvalue { double, double } %9, 0
  %12 = fneg double %10
  %13 = insertvalue { double, double } poison, double %11, 0
  %14 = insertvalue { double, double } %13, double %12, 1
  %15 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %7
  store { double, double } %14, ptr %15, align 8
  %16 = add i32 %7, 1
  %17 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %16
  %18 = load { double, double }, ptr %17, align 8, !invariant.load !3
  %19 = extractvalue { double, double } %18, 1
  %20 = extractvalue { double, double } %18, 0
  %21 = fneg double %19
  %22 = insertvalue { double, double } poison, double %20, 0
  %23 = insertvalue { double, double } %22, double %21, 1
  %24 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %16
  store { double, double } %23, ptr %24, align 8
  %25 = add i32 %7, 2
  %26 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %25
  %27 = load { double, double }, ptr %26, align 8, !invariant.load !3
  %28 = extractvalue { double, double } %27, 1
  %29 = extractvalue { double, double } %27, 0
  %30 = fneg double %28
  %31 = insertvalue { double, double } poison, double %29, 0
  %32 = insertvalue { double, double } %31, double %30, 1
  %33 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %25
  store { double, double } %32, ptr %33, align 8
  %34 = add i32 %7, 3
  %35 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %34
  %36 = load { double, double }, ptr %35, align 8, !invariant.load !3
  %37 = extractvalue { double, double } %36, 1
  %38 = extractvalue { double, double } %36, 0
  %39 = fneg double %37
  %40 = insertvalue { double, double } poison, double %38, 0
  %41 = insertvalue { double, double } %40, double %39, 1
  %42 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %34
  store { double, double } %41, ptr %42, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 576}
!2 = !{i32 0, i32 128}
!3 = !{}
