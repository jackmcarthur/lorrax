; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_fmod(double, double)

define ptx_kernel void @loop_select_fusion(ptr noalias align 16 dereferenceable(256) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(256) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = getelementptr inbounds [32 x double], ptr %0, i32 0, i32 %4
  %6 = load double, ptr %5, align 8, !invariant.load !2
  %7 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %8 = load i64, ptr %7, align 4, !invariant.load !2
  %9 = sitofp i64 %8 to double
  %10 = call double @__nv_fmod(double %6, double %9)
  %11 = fcmp olt double %10, 0.000000e+00
  %12 = icmp slt i64 %8, 0
  %13 = icmp ne i1 %11, %12
  %14 = fcmp une double %10, 0.000000e+00
  %15 = and i1 %13, %14
  %16 = fadd double %10, %9
  %17 = select i1 %15, double %16, double %10
  %18 = getelementptr inbounds [32 x double], ptr %2, i32 0, i32 %4
  store double %17, ptr %18, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="32,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 32}
!2 = !{}
