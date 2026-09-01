; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(1179648) %0, ptr noalias readonly align 16 captures(none) dereferenceable(232) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(66816) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = icmp samesign ult i32 %10, 4176
  br i1 %11, label %12, label %34

12:                                               ; preds = %3
  %.lhs.trunc = trunc nuw nsw i32 %10 to i16
  %13 = udiv i16 %.lhs.trunc, 144
  %14 = zext nneg i16 %13 to i64
  %15 = getelementptr inbounds i64, ptr addrspace(1) %4, i64 %14
  %16 = load i64, ptr addrspace(1) %15, align 8, !invariant.load !4
  %17 = lshr i64 %16, 54
  %18 = and i64 %17, 512
  %19 = add i64 %18, %16
  %20 = trunc i64 %19 to i32
  %21 = icmp ult i32 %20, 512
  %22 = tail call i32 @llvm.smax.i32(i32 %20, i32 0)
  %23 = tail call i32 @llvm.umin.i32(i32 %22, i32 511)
  %24 = mul i16 %13, 144
  %urem5.decomposed = sub i16 %.lhs.trunc, %24
  %urem.zext = zext nneg i16 %urem5.decomposed to i32
  %25 = mul nuw nsw i32 %23, 144
  %26 = add nuw nsw i32 %25, %urem.zext
  %27 = zext nneg i32 %26 to i64
  %28 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %27
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack6 = extractelement <2 x double> %29, i32 0
  %.unpack27 = extractelement <2 x double> %29, i32 1
  %30 = zext nneg i32 %10 to i64
  %31 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %30
  %.elt = select i1 %21, double %.unpack6, double 0x7FF8000000000000
  %.elt4 = select i1 %21, double %.unpack27, double 0.000000e+00
  %32 = insertelement <2 x double> poison, double %.elt, i32 0
  %33 = insertelement <2 x double> %32, double %.elt4, i32 1
  store <2 x double> %33, ptr addrspace(1) %31, align 16
  br label %34

34:                                               ; preds = %12, %3
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
!2 = !{i32 0, i32 33}
!3 = !{i32 0, i32 128}
!4 = !{}
