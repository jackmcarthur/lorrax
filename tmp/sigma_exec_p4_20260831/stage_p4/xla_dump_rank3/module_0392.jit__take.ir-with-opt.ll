; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(24772608) %0, ptr noalias readonly align 16 captures(none) dereferenceable(232) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(1403136) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = icmp samesign ult i32 %10, 87696
  br i1 %11, label %12, label %42

12:                                               ; preds = %3
  %13 = udiv i32 %10, 144
  %.lhs.trunc = trunc nuw nsw i32 %13 to i16
  %14 = urem i16 %.lhs.trunc, 29
  %15 = zext nneg i16 %14 to i64
  %16 = getelementptr inbounds i64, ptr addrspace(1) %4, i64 %15
  %17 = load i64, ptr addrspace(1) %16, align 8, !invariant.load !4
  %18 = lshr i64 %17, 54
  %19 = and i64 %18, 512
  %20 = add i64 %19, %17
  %21 = trunc i64 %20 to i32
  %22 = icmp ult i32 %21, 512
  %23 = tail call i32 @llvm.smax.i32(i32 %21, i32 0)
  %24 = tail call i32 @llvm.umin.i32(i32 %23, i32 511)
  %25 = udiv i32 %10, 12
  %.lhs.trunc5 = trunc nuw nsw i32 %25 to i16
  %26 = urem i16 %.lhs.trunc5, 12
  %narrow = mul nuw nsw i16 %26, 12
  %27 = zext nneg i16 %narrow to i32
  %28 = udiv i32 %10, 4176
  %29 = mul nuw nsw i32 %28, 73728
  %30 = mul i32 %25, 12
  %.decomposed = sub i32 %10, %30
  %31 = mul nuw nsw i32 %24, 144
  %32 = or disjoint i32 %29, %.decomposed
  %33 = add nuw nsw i32 %32, %27
  %34 = add nuw nsw i32 %33, %31
  %35 = zext nneg i32 %34 to i64
  %36 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %35
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !4
  %.unpack7 = extractelement <2 x double> %37, i32 0
  %.unpack28 = extractelement <2 x double> %37, i32 1
  %38 = zext nneg i32 %10 to i64
  %39 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %38
  %.elt = select i1 %22, double %.unpack7, double 0x7FF8000000000000
  %.elt4 = select i1 %22, double %.unpack28, double 0.000000e+00
  %40 = insertelement <2 x double> poison, double %.elt, i32 0
  %41 = insertelement <2 x double> %40, double %.elt4, i32 1
  store <2 x double> %41, ptr addrspace(1) %39, align 16
  br label %42

42:                                               ; preds = %12, %3
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
!2 = !{i32 0, i32 686}
!3 = !{i32 0, i32 128}
!4 = !{}
