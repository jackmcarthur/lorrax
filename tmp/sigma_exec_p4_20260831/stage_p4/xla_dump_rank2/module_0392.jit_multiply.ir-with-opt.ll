; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(4718592) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(4718592) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = load double, ptr addrspace(1) %4, align 16, !invariant.load !4
  %10 = shl nuw nsw i32 %8, 2
  %11 = shl nuw nsw i32 %7, 9
  %12 = or disjoint i32 %10, %11
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %13
  %15 = load <2 x double>, ptr addrspace(1) %14, align 16, !invariant.load !4
  %.unpack26 = extractelement <2 x double> %15, i32 0
  %.unpack227 = extractelement <2 x double> %15, i32 1
  %16 = fmul double %9, %.unpack26
  %17 = fmul double %.unpack227, 0.000000e+00
  %18 = fsub double %16, %17
  %19 = fmul double %.unpack26, 0.000000e+00
  %20 = fmul double %9, %.unpack227
  %21 = fadd double %19, %20
  %22 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %13
  %23 = insertelement <2 x double> poison, double %18, i32 0
  %24 = insertelement <2 x double> %23, double %21, i32 1
  store <2 x double> %24, ptr addrspace(1) %22, align 64
  %25 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 16
  %26 = load <2 x double>, ptr addrspace(1) %25, align 16, !invariant.load !4
  %.unpack528 = extractelement <2 x double> %26, i32 0
  %.unpack729 = extractelement <2 x double> %26, i32 1
  %27 = fmul double %9, %.unpack528
  %28 = fmul double %.unpack729, 0.000000e+00
  %29 = fsub double %27, %28
  %30 = fmul double %.unpack528, 0.000000e+00
  %31 = fmul double %9, %.unpack729
  %32 = fadd double %30, %31
  %33 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 16
  %34 = insertelement <2 x double> poison, double %29, i32 0
  %35 = insertelement <2 x double> %34, double %32, i32 1
  store <2 x double> %35, ptr addrspace(1) %33, align 16
  %36 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 32
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !4
  %.unpack1030 = extractelement <2 x double> %37, i32 0
  %.unpack1231 = extractelement <2 x double> %37, i32 1
  %38 = fmul double %9, %.unpack1030
  %39 = fmul double %.unpack1231, 0.000000e+00
  %40 = fsub double %38, %39
  %41 = fmul double %.unpack1030, 0.000000e+00
  %42 = fmul double %9, %.unpack1231
  %43 = fadd double %41, %42
  %44 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 32
  %45 = insertelement <2 x double> poison, double %40, i32 0
  %46 = insertelement <2 x double> %45, double %43, i32 1
  store <2 x double> %46, ptr addrspace(1) %44, align 32
  %47 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 48
  %48 = load <2 x double>, ptr addrspace(1) %47, align 16, !invariant.load !4
  %.unpack1532 = extractelement <2 x double> %48, i32 0
  %.unpack1733 = extractelement <2 x double> %48, i32 1
  %49 = fmul double %9, %.unpack1532
  %50 = fmul double %.unpack1733, 0.000000e+00
  %51 = fsub double %49, %50
  %52 = fmul double %.unpack1532, 0.000000e+00
  %53 = fmul double %9, %.unpack1733
  %54 = fadd double %52, %53
  %55 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 48
  %56 = insertelement <2 x double> poison, double %51, i32 0
  %57 = insertelement <2 x double> %56, double %54, i32 1
  store <2 x double> %57, ptr addrspace(1) %55, align 16
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
