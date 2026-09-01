; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_select(ptr noalias readonly align 16 captures(none) dereferenceable(1572864) %0, ptr noalias readonly align 16 captures(none) dereferenceable(6291456) %1, ptr noalias readonly align 16 captures(none) dereferenceable(6291456) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(6291456) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %10 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %11 = shl nuw nsw i32 %10, 2
  %12 = shl nuw nsw i32 %9, 9
  %13 = or disjoint i32 %11, %12
  %14 = zext nneg i32 %13 to i64
  %15 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %14
  %16 = getelementptr inbounds i32, ptr addrspace(1) %6, i64 %14
  %17 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 %14
  %18 = load <32 x i1>, ptr addrspace(1) %17, align 4, !invariant.load !4
  %19 = load <4 x i32>, ptr addrspace(1) %16, align 16, !invariant.load !4
  %20 = extractelement <4 x i32> %19, i32 0
  %21 = extractelement <4 x i32> %19, i32 1
  %22 = extractelement <4 x i32> %19, i32 2
  %23 = extractelement <4 x i32> %19, i32 3
  %24 = load <4 x i32>, ptr addrspace(1) %15, align 16, !invariant.load !4
  %25 = extractelement <4 x i32> %24, i32 0
  %26 = extractelement <4 x i32> %24, i32 1
  %27 = extractelement <4 x i32> %24, i32 2
  %28 = extractelement <4 x i32> %24, i32 3
  %29 = extractelement <32 x i1> %18, i64 0
  %30 = select i1 %29, i32 %20, i32 %25
  %31 = extractelement <32 x i1> %18, i64 8
  %32 = select i1 %31, i32 %21, i32 %26
  %33 = extractelement <32 x i1> %18, i64 16
  %34 = select i1 %33, i32 %22, i32 %27
  %35 = extractelement <32 x i1> %18, i64 24
  %36 = select i1 %35, i32 %23, i32 %28
  %37 = insertelement <4 x i32> poison, i32 %30, i64 0
  %38 = insertelement <4 x i32> %37, i32 %32, i64 1
  %39 = insertelement <4 x i32> %38, i32 %34, i64 2
  %40 = insertelement <4 x i32> %39, i32 %36, i64 3
  %41 = getelementptr inbounds i32, ptr addrspace(1) %8, i64 %14
  store <4 x i32> %40, ptr addrspace(1) %41, align 16
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
