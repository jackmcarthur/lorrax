; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(178361600) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(178361600) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = icmp samesign ult i32 %10, 2786900
  br i1 %11, label %12, label %62

12:                                               ; preds = %3
  %13 = load double, ptr addrspace(1) %4, align 16, !invariant.load !4
  %14 = shl nuw nsw i32 %8, 2
  %15 = shl nuw nsw i32 %7, 9
  %16 = or disjoint i32 %14, %15
  %17 = zext nneg i32 %16 to i64
  %18 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %17
  %19 = load <2 x double>, ptr addrspace(1) %18, align 16, !invariant.load !4
  %.unpack26 = extractelement <2 x double> %19, i32 0
  %.unpack227 = extractelement <2 x double> %19, i32 1
  %20 = fmul double %13, %.unpack26
  %21 = fmul double %.unpack227, 0.000000e+00
  %22 = fsub double %20, %21
  %23 = fmul double %13, %.unpack227
  %24 = fmul double %.unpack26, 0.000000e+00
  %25 = fadd double %24, %23
  %26 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %17
  %27 = insertelement <2 x double> poison, double %22, i32 0
  %28 = insertelement <2 x double> %27, double %25, i32 1
  store <2 x double> %28, ptr addrspace(1) %26, align 64
  %29 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 16
  %30 = load <2 x double>, ptr addrspace(1) %29, align 16, !invariant.load !4
  %.unpack528 = extractelement <2 x double> %30, i32 0
  %.unpack729 = extractelement <2 x double> %30, i32 1
  %31 = fmul double %13, %.unpack528
  %32 = fmul double %.unpack729, 0.000000e+00
  %33 = fsub double %31, %32
  %34 = fmul double %13, %.unpack729
  %35 = fmul double %.unpack528, 0.000000e+00
  %36 = fadd double %35, %34
  %37 = getelementptr inbounds i8, ptr addrspace(1) %26, i64 16
  %38 = insertelement <2 x double> poison, double %33, i32 0
  %39 = insertelement <2 x double> %38, double %36, i32 1
  store <2 x double> %39, ptr addrspace(1) %37, align 16
  %40 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 32
  %41 = load <2 x double>, ptr addrspace(1) %40, align 16, !invariant.load !4
  %.unpack1030 = extractelement <2 x double> %41, i32 0
  %.unpack1231 = extractelement <2 x double> %41, i32 1
  %42 = fmul double %13, %.unpack1030
  %43 = fmul double %.unpack1231, 0.000000e+00
  %44 = fsub double %42, %43
  %45 = fmul double %13, %.unpack1231
  %46 = fmul double %.unpack1030, 0.000000e+00
  %47 = fadd double %46, %45
  %48 = getelementptr inbounds i8, ptr addrspace(1) %26, i64 32
  %49 = insertelement <2 x double> poison, double %44, i32 0
  %50 = insertelement <2 x double> %49, double %47, i32 1
  store <2 x double> %50, ptr addrspace(1) %48, align 32
  %51 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 48
  %52 = load <2 x double>, ptr addrspace(1) %51, align 16, !invariant.load !4
  %.unpack1532 = extractelement <2 x double> %52, i32 0
  %.unpack1733 = extractelement <2 x double> %52, i32 1
  %53 = fmul double %13, %.unpack1532
  %54 = fmul double %.unpack1733, 0.000000e+00
  %55 = fsub double %53, %54
  %56 = fmul double %13, %.unpack1733
  %57 = fmul double %.unpack1532, 0.000000e+00
  %58 = fadd double %57, %56
  %59 = getelementptr inbounds i8, ptr addrspace(1) %26, i64 48
  %60 = insertelement <2 x double> poison, double %55, i32 0
  %61 = insertelement <2 x double> %60, double %58, i32 1
  store <2 x double> %61, ptr addrspace(1) %59, align 16
  br label %62

62:                                               ; preds = %12, %3
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
!2 = !{i32 0, i32 21773}
!3 = !{i32 0, i32 128}
!4 = !{}
