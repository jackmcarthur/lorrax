; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(32768) %0, ptr noalias readonly align 16 captures(none) dereferenceable(262144) %1, ptr noalias readonly align 16 captures(none) dereferenceable(8) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(262144) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = addrspacecast ptr %1 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %10 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %11 = shl nuw nsw i32 %9, 7
  %12 = or disjoint i32 %11, %10
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds i8, ptr addrspace(1) %5, i64 %13
  %15 = load i8, ptr addrspace(1) %14, align 1, !invariant.load !4
  %16 = load double, ptr addrspace(1) %6, align 16, !invariant.load !4
  %17 = getelementptr inbounds double, ptr addrspace(1) %7, i64 %13
  %18 = load double, ptr addrspace(1) %17, align 8, !invariant.load !4
  %19 = trunc i8 %15 to i1
  %20 = select i1 %19, double %16, double %18
  %21 = getelementptr inbounds double, ptr addrspace(1) %8, i64 %13
  store double %20, ptr addrspace(1) %21, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 256}
!3 = !{i32 0, i32 128}
!4 = !{}
