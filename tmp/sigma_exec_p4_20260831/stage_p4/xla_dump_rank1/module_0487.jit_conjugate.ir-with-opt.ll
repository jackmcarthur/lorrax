; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_complex_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(4718592) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4718592) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %10
  %12 = load <2 x double>, ptr addrspace(1) %11, align 16, !invariant.load !4
  %.unpack26 = extractelement <2 x double> %12, i32 0
  %.unpack227 = extractelement <2 x double> %12, i32 1
  %13 = fneg double %.unpack227
  %14 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %10
  %15 = insertelement <2 x double> poison, double %.unpack26, i32 0
  %16 = insertelement <2 x double> %15, double %13, i32 1
  store <2 x double> %16, ptr addrspace(1) %14, align 64
  %17 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 16
  %18 = load <2 x double>, ptr addrspace(1) %17, align 16, !invariant.load !4
  %.unpack528 = extractelement <2 x double> %18, i32 0
  %.unpack729 = extractelement <2 x double> %18, i32 1
  %19 = fneg double %.unpack729
  %20 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 16
  %21 = insertelement <2 x double> poison, double %.unpack528, i32 0
  %22 = insertelement <2 x double> %21, double %19, i32 1
  store <2 x double> %22, ptr addrspace(1) %20, align 16
  %23 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 32
  %24 = load <2 x double>, ptr addrspace(1) %23, align 16, !invariant.load !4
  %.unpack1030 = extractelement <2 x double> %24, i32 0
  %.unpack1231 = extractelement <2 x double> %24, i32 1
  %25 = fneg double %.unpack1231
  %26 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 32
  %27 = insertelement <2 x double> poison, double %.unpack1030, i32 0
  %28 = insertelement <2 x double> %27, double %25, i32 1
  store <2 x double> %28, ptr addrspace(1) %26, align 32
  %29 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 48
  %30 = load <2 x double>, ptr addrspace(1) %29, align 16, !invariant.load !4
  %.unpack1532 = extractelement <2 x double> %30, i32 0
  %.unpack1733 = extractelement <2 x double> %30, i32 1
  %31 = fneg double %.unpack1733
  %32 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 48
  %33 = insertelement <2 x double> poison, double %.unpack1532, i32 0
  %34 = insertelement <2 x double> %33, double %31, i32 1
  store <2 x double> %34, ptr addrspace(1) %32, align 16
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
!2 = !{i32 0, i32 576}
!3 = !{i32 0, i32 128}
!4 = !{}
