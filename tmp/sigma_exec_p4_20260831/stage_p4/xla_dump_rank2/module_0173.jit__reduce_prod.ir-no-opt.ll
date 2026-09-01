; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_reduce_fusion(ptr noalias align 16 dereferenceable(48) %0, ptr noalias align 256 dereferenceable(16) %1) #0 {
  %3 = getelementptr inbounds [3 x { double, double }], ptr %0, i32 0, i32 0
  %4 = load { double, double }, ptr %3, align 8, !invariant.load !1
  %5 = extractvalue { double, double } %4, 0
  %6 = extractvalue { double, double } %4, 1
  %7 = fmul double %6, 0.000000e+00
  %8 = fsub double %5, %7
  %9 = fmul double %5, 0.000000e+00
  %10 = fadd double %9, %6
  %11 = getelementptr inbounds [3 x { double, double }], ptr %0, i32 0, i32 1
  %12 = load { double, double }, ptr %11, align 8, !invariant.load !1
  %13 = extractvalue { double, double } %12, 0
  %14 = extractvalue { double, double } %12, 1
  %15 = fmul double %8, %13
  %16 = fmul double %10, %14
  %17 = fsub double %15, %16
  %18 = fmul double %10, %13
  %19 = fmul double %8, %14
  %20 = fadd double %18, %19
  %21 = getelementptr inbounds [3 x { double, double }], ptr %0, i32 0, i32 2
  %22 = load { double, double }, ptr %21, align 8, !invariant.load !1
  %23 = extractvalue { double, double } %22, 0
  %24 = extractvalue { double, double } %22, 1
  %25 = fmul double %17, %23
  %26 = fmul double %20, %24
  %27 = fsub double %25, %26
  %28 = fmul double %20, %23
  %29 = fmul double %17, %24
  %30 = fadd double %28, %29
  %31 = insertvalue { double, double } poison, double %27, 0
  %32 = insertvalue { double, double } %31, double %30, 1
  %33 = getelementptr inbounds [1 x { double, double }], ptr %1, i32 0, i32 0
  store { double, double } %32, ptr %33, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
