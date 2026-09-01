; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_compare_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(6291456) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(1572864) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = load i64, ptr addrspace(1) %4, align 16, !invariant.load !4
  %10 = trunc i64 %9 to i32
  %11 = shl nuw nsw i32 %8, 2
  %12 = shl nuw nsw i32 %7, 9
  %13 = or disjoint i32 %11, %12
  %14 = zext nneg i32 %13 to i64
  %15 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 %14
  %16 = load <4 x i32>, ptr addrspace(1) %15, align 16, !invariant.load !4
  %17 = extractelement <4 x i32> %16, i32 0
  %18 = extractelement <4 x i32> %16, i32 1
  %19 = extractelement <4 x i32> %16, i32 2
  %20 = extractelement <4 x i32> %16, i32 3
  %21 = icmp slt i32 %17, %10
  %22 = zext i1 %21 to i8
  %23 = icmp slt i32 %18, %10
  %24 = zext i1 %23 to i8
  %25 = icmp slt i32 %19, %10
  %26 = zext i1 %25 to i8
  %27 = icmp slt i32 %20, %10
  %28 = zext i1 %27 to i8
  %29 = insertelement <4 x i8> poison, i8 %22, i64 0
  %30 = insertelement <4 x i8> %29, i8 %24, i64 1
  %31 = insertelement <4 x i8> %30, i8 %26, i64 2
  %32 = insertelement <4 x i8> %31, i8 %28, i64 3
  %33 = getelementptr inbounds i8, ptr addrspace(1) %6, i64 %14
  store <4 x i8> %32, ptr addrspace(1) %33, align 4
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
