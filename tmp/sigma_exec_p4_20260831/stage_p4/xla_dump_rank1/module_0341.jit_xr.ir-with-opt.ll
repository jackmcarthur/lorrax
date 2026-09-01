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
  %7 = shl nuw nsw i32 %6, 1
  %8 = shl nuw nsw i32 %5, 8
  %9 = or disjoint i32 %7, %8
  %10 = udiv i32 %9, 155
  %11 = trunc i32 %10 to i1
  %12 = select i1 %11, i32 310, i32 0
  %13 = shl nuw nsw i32 %5, 7
  %14 = or disjoint i32 %13, %6
  %15 = udiv i32 %14, 155
  %16 = urem i32 %15, 24
  %17 = mul nuw nsw i32 %16, 620
  %18 = udiv i32 %14, 3720
  %19 = mul nuw nsw i32 %18, 29760
  %20 = add nuw nsw i32 %17, %19
  %21 = add nuw nsw i32 %20, %12
  %22 = shl nuw nsw i32 %6, 2
  %23 = shl nuw nsw i32 %5, 9
  %24 = or disjoint i32 %22, %23
  %25 = urem i32 %24, 310
  %26 = add nuw nsw i32 %21, %25
  %27 = zext nneg i32 %26 to i64
  %28 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %27
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack29 = extractelement <2 x double> %29, i32 0
  %.unpack230 = extractelement <2 x double> %29, i32 1
  %30 = zext nneg i32 %24 to i64
  %31 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %30
  %32 = insertelement <2 x double> poison, double %.unpack29, i32 0
  %33 = insertelement <2 x double> %32, double %.unpack230, i32 1
  store <2 x double> %33, ptr addrspace(1) %31, align 64
  %34 = or disjoint i32 %24, 1
  %35 = urem i32 %34, 310
  %36 = add nuw nsw i32 %21, %35
  %37 = zext nneg i32 %36 to i64
  %38 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %37
  %39 = load <2 x double>, ptr addrspace(1) %38, align 16, !invariant.load !4
  %.unpack527 = extractelement <2 x double> %39, i32 0
  %.unpack728 = extractelement <2 x double> %39, i32 1
  %40 = getelementptr inbounds i8, ptr addrspace(1) %31, i64 16
  %41 = insertelement <2 x double> poison, double %.unpack527, i32 0
  %42 = insertelement <2 x double> %41, double %.unpack728, i32 1
  store <2 x double> %42, ptr addrspace(1) %40, align 16
  %43 = or disjoint i32 %9, 1
  %44 = udiv i32 %43, 155
  %45 = trunc i32 %44 to i1
  %46 = select i1 %45, i32 310, i32 0
  %47 = or disjoint i32 %24, 2
  %48 = urem i32 %47, 310
  %49 = add nuw nsw i32 %20, %48
  %50 = add nuw nsw i32 %49, %46
  %51 = zext nneg i32 %50 to i64
  %52 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %51
  %53 = load <2 x double>, ptr addrspace(1) %52, align 16, !invariant.load !4
  %.unpack1025 = extractelement <2 x double> %53, i32 0
  %.unpack1226 = extractelement <2 x double> %53, i32 1
  %54 = getelementptr inbounds i8, ptr addrspace(1) %31, i64 32
  %55 = insertelement <2 x double> poison, double %.unpack1025, i32 0
  %56 = insertelement <2 x double> %55, double %.unpack1226, i32 1
  store <2 x double> %56, ptr addrspace(1) %54, align 32
  %57 = or disjoint i32 %24, 3
  %58 = udiv i32 %57, 310
  %59 = trunc i32 %58 to i1
  %60 = select i1 %59, i32 310, i32 0
  %61 = mul i32 %58, 310
  %.decomposed = sub i32 %57, %61
  %62 = add nuw nsw i32 %20, %.decomposed
  %63 = add nuw nsw i32 %62, %60
  %64 = zext nneg i32 %63 to i64
  %65 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %64
  %66 = load <2 x double>, ptr addrspace(1) %65, align 16, !invariant.load !4
  %.unpack1523 = extractelement <2 x double> %66, i32 0
  %.unpack1724 = extractelement <2 x double> %66, i32 1
  %67 = getelementptr inbounds i8, ptr addrspace(1) %31, i64 48
  %68 = insertelement <2 x double> poison, double %.unpack1523, i32 0
  %69 = insertelement <2 x double> %68, double %.unpack1724, i32 1
  store <2 x double> %69, ptr addrspace(1) %67, align 16
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
