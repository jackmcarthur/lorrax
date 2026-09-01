; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_sqrt(double)

define ptx_kernel void @wrapped_sqrt(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1) #0 {
  %3 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %4 = load double, ptr %3, align 8, !invariant.load !1
  %5 = call double @__nv_sqrt(double %4)
  %6 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  store double %5, ptr %6, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
