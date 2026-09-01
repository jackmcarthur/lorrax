; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_add(ptr noalias readonly align 16 captures(none) dereferenceable(4718592) %0, ptr noalias readonly align 16 captures(none) dereferenceable(4718592) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(4718592) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %8, 2
  %10 = shl nuw nsw i32 %7, 9
  %11 = or disjoint i32 %9, %10
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %12
  %14 = load <2 x double>, ptr addrspace(1) %13, align 16, !invariant.load !4
  %.unpack41 = extractelement <2 x double> %14, i32 0
  %.unpack242 = extractelement <2 x double> %14, i32 1
  %15 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %12
  %16 = load <2 x double>, ptr addrspace(1) %15, align 16, !invariant.load !4
  %.unpack349 = extractelement <2 x double> %16, i32 0
  %.unpack550 = extractelement <2 x double> %16, i32 1
  %17 = fadd double %.unpack41, %.unpack349
  %18 = fadd double %.unpack242, %.unpack550
  %19 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %12
  %20 = insertelement <2 x double> poison, double %17, i32 0
  %21 = insertelement <2 x double> %20, double %18, i32 1
  store <2 x double> %21, ptr addrspace(1) %19, align 64
  %22 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 16
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !4
  %.unpack843 = extractelement <2 x double> %23, i32 0
  %.unpack1044 = extractelement <2 x double> %23, i32 1
  %24 = getelementptr inbounds i8, ptr addrspace(1) %15, i64 16
  %25 = load <2 x double>, ptr addrspace(1) %24, align 16, !invariant.load !4
  %.unpack1151 = extractelement <2 x double> %25, i32 0
  %.unpack1352 = extractelement <2 x double> %25, i32 1
  %26 = fadd double %.unpack843, %.unpack1151
  %27 = fadd double %.unpack1044, %.unpack1352
  %28 = getelementptr inbounds i8, ptr addrspace(1) %19, i64 16
  %29 = insertelement <2 x double> poison, double %26, i32 0
  %30 = insertelement <2 x double> %29, double %27, i32 1
  store <2 x double> %30, ptr addrspace(1) %28, align 16
  %31 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 32
  %32 = load <2 x double>, ptr addrspace(1) %31, align 16, !invariant.load !4
  %.unpack1645 = extractelement <2 x double> %32, i32 0
  %.unpack1846 = extractelement <2 x double> %32, i32 1
  %33 = getelementptr inbounds i8, ptr addrspace(1) %15, i64 32
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !4
  %.unpack1953 = extractelement <2 x double> %34, i32 0
  %.unpack2154 = extractelement <2 x double> %34, i32 1
  %35 = fadd double %.unpack1645, %.unpack1953
  %36 = fadd double %.unpack1846, %.unpack2154
  %37 = getelementptr inbounds i8, ptr addrspace(1) %19, i64 32
  %38 = insertelement <2 x double> poison, double %35, i32 0
  %39 = insertelement <2 x double> %38, double %36, i32 1
  store <2 x double> %39, ptr addrspace(1) %37, align 32
  %40 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 48
  %41 = load <2 x double>, ptr addrspace(1) %40, align 16, !invariant.load !4
  %.unpack2447 = extractelement <2 x double> %41, i32 0
  %.unpack2648 = extractelement <2 x double> %41, i32 1
  %42 = getelementptr inbounds i8, ptr addrspace(1) %15, i64 48
  %43 = load <2 x double>, ptr addrspace(1) %42, align 16, !invariant.load !4
  %.unpack2755 = extractelement <2 x double> %43, i32 0
  %.unpack2956 = extractelement <2 x double> %43, i32 1
  %44 = fadd double %.unpack2447, %.unpack2755
  %45 = fadd double %.unpack2648, %.unpack2956
  %46 = getelementptr inbounds i8, ptr addrspace(1) %19, i64 48
  %47 = insertelement <2 x double> poison, double %44, i32 0
  %48 = insertelement <2 x double> %47, double %45, i32 1
  store <2 x double> %48, ptr addrspace(1) %46, align 16
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
