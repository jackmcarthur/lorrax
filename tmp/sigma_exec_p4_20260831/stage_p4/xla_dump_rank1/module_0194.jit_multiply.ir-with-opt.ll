; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(256) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(256) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = zext nneg i32 %7 to i64
  %9 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %8
  %10 = load double, ptr addrspace(1) %9, align 8, !invariant.load !3
  %11 = load i64, ptr addrspace(1) %5, align 16, !invariant.load !3
  %12 = sitofp i64 %11 to double
  %13 = fmul double %10, %12
  %14 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %8
  store double %13, ptr addrspace(1) %14, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="32,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 32}
!3 = !{}
