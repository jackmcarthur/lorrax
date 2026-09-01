; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_slice(ptr noalias readonly align 16 captures(none) dereferenceable(243793920) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(121896960) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %7 = shl nuw nsw i32 %5, 7
  %8 = or disjoint i32 %7, %6
  %9 = udiv i32 %8, 6
  %10 = shl nuw nsw i32 %8, 2
  %11 = mul nuw nsw i32 %9, 24
  %12 = add nuw nsw i32 %11, %10
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %13
  %15 = load <2 x double>, ptr addrspace(1) %14, align 16, !invariant.load !4
  %.unpack20 = extractelement <2 x double> %15, i32 0
  %.unpack221 = extractelement <2 x double> %15, i32 1
  %16 = shl nuw nsw i32 %6, 2
  %17 = shl nuw nsw i32 %5, 9
  %18 = or disjoint i32 %16, %17
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %19
  %21 = insertelement <2 x double> poison, double %.unpack20, i32 0
  %22 = insertelement <2 x double> %21, double %.unpack221, i32 1
  store <2 x double> %22, ptr addrspace(1) %20, align 64
  %23 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 16
  %24 = load <2 x double>, ptr addrspace(1) %23, align 16, !invariant.load !4
  %.unpack522 = extractelement <2 x double> %24, i32 0
  %.unpack723 = extractelement <2 x double> %24, i32 1
  %25 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 16
  %26 = insertelement <2 x double> poison, double %.unpack522, i32 0
  %27 = insertelement <2 x double> %26, double %.unpack723, i32 1
  store <2 x double> %27, ptr addrspace(1) %25, align 16
  %28 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 32
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack1024 = extractelement <2 x double> %29, i32 0
  %.unpack1225 = extractelement <2 x double> %29, i32 1
  %30 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 32
  %31 = insertelement <2 x double> poison, double %.unpack1024, i32 0
  %32 = insertelement <2 x double> %31, double %.unpack1225, i32 1
  store <2 x double> %32, ptr addrspace(1) %30, align 32
  %33 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 48
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !4
  %.unpack1526 = extractelement <2 x double> %34, i32 0
  %.unpack1727 = extractelement <2 x double> %34, i32 1
  %35 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 48
  %36 = insertelement <2 x double> poison, double %.unpack1526, i32 0
  %37 = insertelement <2 x double> %36, double %.unpack1727, i32 1
  store <2 x double> %37, ptr addrspace(1) %35, align 16
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
!2 = !{i32 0, i32 14880}
!3 = !{i32 0, i32 128}
!4 = !{}
