; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(178361600) %0, ptr noalias readonly align 16 captures(none) dereferenceable(4) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(44590400) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = icmp samesign ult i32 %10, 696725
  br i1 %11, label %12, label %43

12:                                               ; preds = %3
  %13 = load i32, ptr addrspace(1) %4, align 16, !invariant.load !4
  %14 = tail call i32 @llvm.smax.i32(i32 %13, i32 0)
  %15 = tail call i32 @llvm.umin.i32(i32 %14, i32 3)
  %16 = shl nuw nsw i32 %8, 2
  %17 = shl nuw nsw i32 %7, 9
  %18 = or disjoint i32 %16, %17
  %19 = mul nuw nsw i32 %15, 2786900
  %20 = add nuw nsw i32 %19, %18
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %21
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !4
  %.unpack20 = extractelement <2 x double> %23, i32 0
  %.unpack221 = extractelement <2 x double> %23, i32 1
  %24 = zext nneg i32 %18 to i64
  %25 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %24
  %26 = insertelement <2 x double> poison, double %.unpack20, i32 0
  %27 = insertelement <2 x double> %26, double %.unpack221, i32 1
  store <2 x double> %27, ptr addrspace(1) %25, align 64
  %28 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 16
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack522 = extractelement <2 x double> %29, i32 0
  %.unpack723 = extractelement <2 x double> %29, i32 1
  %30 = getelementptr inbounds i8, ptr addrspace(1) %25, i64 16
  %31 = insertelement <2 x double> poison, double %.unpack522, i32 0
  %32 = insertelement <2 x double> %31, double %.unpack723, i32 1
  store <2 x double> %32, ptr addrspace(1) %30, align 16
  %33 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 32
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !4
  %.unpack1024 = extractelement <2 x double> %34, i32 0
  %.unpack1225 = extractelement <2 x double> %34, i32 1
  %35 = getelementptr inbounds i8, ptr addrspace(1) %25, i64 32
  %36 = insertelement <2 x double> poison, double %.unpack1024, i32 0
  %37 = insertelement <2 x double> %36, double %.unpack1225, i32 1
  store <2 x double> %37, ptr addrspace(1) %35, align 32
  %38 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 48
  %39 = load <2 x double>, ptr addrspace(1) %38, align 16, !invariant.load !4
  %.unpack1526 = extractelement <2 x double> %39, i32 0
  %.unpack1727 = extractelement <2 x double> %39, i32 1
  %40 = getelementptr inbounds i8, ptr addrspace(1) %25, i64 48
  %41 = insertelement <2 x double> poison, double %.unpack1526, i32 0
  %42 = insertelement <2 x double> %41, double %.unpack1727, i32 1
  store <2 x double> %42, ptr addrspace(1) %40, align 16
  br label %43

43:                                               ; preds = %12, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 5444}
!3 = !{i32 0, i32 128}
!4 = !{}
