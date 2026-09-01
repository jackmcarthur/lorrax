; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_add_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(6291456) %0, ptr noalias readonly align 16 captures(none) dereferenceable(4) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(6291456) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = load i32, ptr addrspace(1) %4, align 16, !invariant.load !4
  %10 = shl nuw nsw i32 %8, 2
  %11 = shl nuw nsw i32 %7, 9
  %12 = or disjoint i32 %10, %11
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %13
  %15 = load <4 x i32>, ptr addrspace(1) %14, align 16, !invariant.load !4
  %16 = extractelement <4 x i32> %15, i32 0
  %17 = extractelement <4 x i32> %15, i32 1
  %18 = extractelement <4 x i32> %15, i32 2
  %19 = extractelement <4 x i32> %15, i32 3
  %20 = add i32 %16, %9
  %21 = add i32 %17, %9
  %22 = add i32 %18, %9
  %23 = add i32 %19, %9
  %24 = insertelement <4 x i32> poison, i32 %20, i64 0
  %25 = insertelement <4 x i32> %24, i32 %21, i64 1
  %26 = insertelement <4 x i32> %25, i32 %22, i64 2
  %27 = insertelement <4 x i32> %26, i32 %23, i64 3
  %28 = getelementptr inbounds i32, ptr addrspace(1) %6, i64 %13
  store <4 x i32> %27, ptr addrspace(1) %28, align 16
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
!2 = !{i32 0, i32 3072}
!3 = !{i32 0, i32 128}
!4 = !{}
