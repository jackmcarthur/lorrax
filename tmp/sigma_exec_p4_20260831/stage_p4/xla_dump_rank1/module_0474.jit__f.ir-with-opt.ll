; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_2_0 = local_unnamed_addr addrspace(1) global [2048 x i8] zeroinitializer, align 256

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_gather(ptr noalias readonly align 16 captures(none) dereferenceable(267264) %0, ptr noalias readonly align 256 captures(none) dereferenceable(2048) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(4718592) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = udiv i32 %10, 144
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %12
  %14 = load i32, ptr addrspace(1) %13, align 4, !invariant.load !4
  %15 = tail call i32 @llvm.smax.i32(i32 %14, i32 0)
  %16 = tail call i32 @llvm.umin.i32(i32 %15, i32 28)
  %17 = udiv i32 %10, 6
  %18 = mul i32 %17, 6
  %.decomposed = sub i32 %10, %18
  %19 = shl nuw nsw i32 %.decomposed, 2
  %20 = urem i32 %17, 24
  %21 = mul nuw nsw i32 %20, 24
  %22 = add nuw nsw i32 %21, %19
  %23 = mul nuw nsw i32 %16, 576
  %24 = add nuw nsw i32 %22, %23
  %25 = zext nneg i32 %24 to i64
  %26 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %25
  %27 = load <2 x double>, ptr addrspace(1) %26, align 16, !invariant.load !4
  %.unpack20 = extractelement <2 x double> %27, i32 0
  %.unpack221 = extractelement <2 x double> %27, i32 1
  %28 = shl nuw nsw i32 %8, 2
  %29 = shl nuw nsw i32 %7, 9
  %30 = or disjoint i32 %28, %29
  %31 = zext nneg i32 %30 to i64
  %32 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %31
  %33 = insertelement <2 x double> poison, double %.unpack20, i32 0
  %34 = insertelement <2 x double> %33, double %.unpack221, i32 1
  store <2 x double> %34, ptr addrspace(1) %32, align 64
  %35 = getelementptr inbounds i8, ptr addrspace(1) %26, i64 16
  %36 = load <2 x double>, ptr addrspace(1) %35, align 16, !invariant.load !4
  %.unpack522 = extractelement <2 x double> %36, i32 0
  %.unpack723 = extractelement <2 x double> %36, i32 1
  %37 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 16
  %38 = insertelement <2 x double> poison, double %.unpack522, i32 0
  %39 = insertelement <2 x double> %38, double %.unpack723, i32 1
  store <2 x double> %39, ptr addrspace(1) %37, align 16
  %40 = getelementptr inbounds i8, ptr addrspace(1) %26, i64 32
  %41 = load <2 x double>, ptr addrspace(1) %40, align 16, !invariant.load !4
  %.unpack1024 = extractelement <2 x double> %41, i32 0
  %.unpack1225 = extractelement <2 x double> %41, i32 1
  %42 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 32
  %43 = insertelement <2 x double> poison, double %.unpack1024, i32 0
  %44 = insertelement <2 x double> %43, double %.unpack1225, i32 1
  store <2 x double> %44, ptr addrspace(1) %42, align 32
  %45 = getelementptr inbounds i8, ptr addrspace(1) %26, i64 48
  %46 = load <2 x double>, ptr addrspace(1) %45, align 16, !invariant.load !4
  %.unpack1526 = extractelement <2 x double> %46, i32 0
  %.unpack1727 = extractelement <2 x double> %46, i32 1
  %47 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 48
  %48 = insertelement <2 x double> poison, double %.unpack1526, i32 0
  %49 = insertelement <2 x double> %48, double %.unpack1727, i32 1
  store <2 x double> %49, ptr addrspace(1) %47, align 16
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
!2 = !{i32 0, i32 576}
!3 = !{i32 0, i32 128}
!4 = !{}
