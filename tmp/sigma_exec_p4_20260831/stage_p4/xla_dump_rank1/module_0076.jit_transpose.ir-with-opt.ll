; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_transpose(ptr noalias readonly align 16 captures(none) dereferenceable(243793920) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(243793920) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %7 = shl nuw nsw i32 %5, 7
  %8 = or disjoint i32 %7, %6
  %9 = udiv i32 %8, 12
  %10 = mul i32 %9, 12
  %.decomposed = sub i32 %8, %10
  %11 = shl nuw nsw i32 %.decomposed, 3
  %12 = urem i32 %9, 310
  %13 = mul nuw nsw i32 %12, 96
  %14 = udiv i32 %8, 7440
  %15 = mul nuw nsw i32 %14, 29760
  %16 = add nuw nsw i32 %15, %11
  %17 = udiv i32 %8, 3720
  %18 = and i32 %17, 1
  %19 = or disjoint i32 %16, %18
  %20 = add nuw nsw i32 %19, %13
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %21
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !4
  %.unpack20 = extractelement <2 x double> %23, i32 0
  %.unpack221 = extractelement <2 x double> %23, i32 1
  %24 = shl nuw nsw i32 %6, 2
  %25 = shl nuw nsw i32 %5, 9
  %26 = or disjoint i32 %24, %25
  %27 = zext nneg i32 %26 to i64
  %28 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %27
  %29 = insertelement <2 x double> poison, double %.unpack20, i32 0
  %30 = insertelement <2 x double> %29, double %.unpack221, i32 1
  store <2 x double> %30, ptr addrspace(1) %28, align 64
  %31 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 32
  %32 = load <2 x double>, ptr addrspace(1) %31, align 16, !invariant.load !4
  %.unpack522 = extractelement <2 x double> %32, i32 0
  %.unpack723 = extractelement <2 x double> %32, i32 1
  %33 = getelementptr inbounds i8, ptr addrspace(1) %28, i64 16
  %34 = insertelement <2 x double> poison, double %.unpack522, i32 0
  %35 = insertelement <2 x double> %34, double %.unpack723, i32 1
  store <2 x double> %35, ptr addrspace(1) %33, align 16
  %36 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 64
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !4
  %.unpack1024 = extractelement <2 x double> %37, i32 0
  %.unpack1225 = extractelement <2 x double> %37, i32 1
  %38 = getelementptr inbounds i8, ptr addrspace(1) %28, i64 32
  %39 = insertelement <2 x double> poison, double %.unpack1024, i32 0
  %40 = insertelement <2 x double> %39, double %.unpack1225, i32 1
  store <2 x double> %40, ptr addrspace(1) %38, align 32
  %41 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 96
  %42 = load <2 x double>, ptr addrspace(1) %41, align 16, !invariant.load !4
  %.unpack1526 = extractelement <2 x double> %42, i32 0
  %.unpack1727 = extractelement <2 x double> %42, i32 1
  %43 = getelementptr inbounds i8, ptr addrspace(1) %28, i64 48
  %44 = insertelement <2 x double> poison, double %.unpack1526, i32 0
  %45 = insertelement <2 x double> %44, double %.unpack1727, i32 1
  store <2 x double> %45, ptr addrspace(1) %43, align 16
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
!2 = !{i32 0, i32 29760}
!3 = !{i32 0, i32 128}
!4 = !{}
