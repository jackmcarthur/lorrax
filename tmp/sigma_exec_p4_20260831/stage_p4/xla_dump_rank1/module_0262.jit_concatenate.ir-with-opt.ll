; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_concatenate(ptr noalias readonly align 16 captures(none) dereferenceable(4) %0, ptr noalias readonly align 16 captures(none) dereferenceable(4) %1, ptr noalias readonly align 16 captures(none) dereferenceable(4) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(12) initializes((0, 12)) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %3 to ptr addrspace(1)
  %7 = addrspacecast ptr %1 to ptr addrspace(1)
  %8 = addrspacecast ptr %2 to ptr addrspace(1)
  %9 = load i32, ptr addrspace(1) %5, align 16, !invariant.load !2
  %10 = load i32, ptr addrspace(1) %7, align 16, !invariant.load !2
  %11 = insertelement <2 x i32> poison, i32 %9, i32 0
  %12 = insertelement <2 x i32> %11, i32 %10, i32 1
  store <2 x i32> %12, ptr addrspace(1) %6, align 256
  %13 = load i32, ptr addrspace(1) %8, align 16, !invariant.load !2
  %14 = getelementptr inbounds i8, ptr addrspace(1) %6, i64 8
  store i32 %13, ptr addrspace(1) %14, align 8
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
