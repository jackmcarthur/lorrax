; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_add_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(262144) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8192) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(262144) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %11
  %13 = load double, ptr addrspace(1) %12, align 8, !invariant.load !4
  %14 = shl nuw nsw i32 %7, 2
  %15 = and i32 %14, 992
  %16 = and i32 %8, 31
  %17 = or disjoint i32 %15, %16
  %18 = zext nneg i32 %17 to i64
  %19 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %18
  %20 = load double, ptr addrspace(1) %19, align 8, !invariant.load !4
  %21 = fadd double %13, %20
  %22 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %11
  store double %21, ptr addrspace(1) %22, align 8
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
