; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_2_0 = local_unnamed_addr addrspace(1) global [64 x i8] zeroinitializer, align 256
@buffer_for_constant_1_0 = local_unnamed_addr addrspace(1) global [64 x i8] zeroinitializer, align 256

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_add_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(1179648) %0, ptr noalias readonly align 16 captures(none) dereferenceable(4718592) %1, ptr noalias readonly align 256 captures(none) dereferenceable(16) %2, ptr noalias readonly align 256 captures(none) dereferenceable(4) %3, ptr noalias readonly align 256 captures(none) dereferenceable(16) %4, ptr noalias writeonly align 256 captures(none) dereferenceable(1179648) %5) local_unnamed_addr #0 {
  %7 = addrspacecast ptr %3 to ptr addrspace(1)
  %8 = addrspacecast ptr %4 to ptr addrspace(1)
  %9 = addrspacecast ptr %2 to ptr addrspace(1)
  %10 = addrspacecast ptr %1 to ptr addrspace(1)
  %11 = addrspacecast ptr %0 to ptr addrspace(1)
  %12 = addrspacecast ptr %5 to ptr addrspace(1)
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %14 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %15 = shl nuw nsw i32 %13, 7
  %16 = or disjoint i32 %15, %14
  %17 = udiv i32 %16, 12
  %18 = urem i32 %17, 12
  %19 = mul i32 %17, 12
  %.decomposed = sub i32 %16, %19
  %20 = load i32, ptr addrspace(1) %7, align 256, !invariant.load !4
  %21 = tail call i32 @llvm.umin.i32(i32 %20, i32 3)
  %22 = zext nneg i32 %21 to i64
  %23 = getelementptr inbounds i32, ptr addrspace(1) %8, i64 %22
  %24 = load i32, ptr addrspace(1) %23, align 4, !invariant.load !4
  %25 = tail call i32 @llvm.smax.i32(i32 %24, i32 0)
  %26 = tail call i32 @llvm.umin.i32(i32 %25, i32 12)
  %27 = add nuw nsw i32 %26, %18
  %28 = getelementptr inbounds i32, ptr addrspace(1) %9, i64 %22
  %29 = load i32, ptr addrspace(1) %28, align 4, !invariant.load !4
  %30 = tail call i32 @llvm.smax.i32(i32 %29, i32 0)
  %31 = tail call i32 @llvm.umin.i32(i32 %30, i32 12)
  %32 = udiv i32 %16, 144
  %33 = mul nuw nsw i32 %32, 576
  %34 = mul nuw nsw i32 %27, 24
  %35 = or disjoint i32 %33, %.decomposed
  %36 = add nuw nsw i32 %35, %31
  %37 = add nuw nsw i32 %36, %34
  %38 = zext nneg i32 %37 to i64
  %39 = getelementptr inbounds { double, double }, ptr addrspace(1) %10, i64 %38
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !4
  %.unpack8 = extractelement <2 x double> %40, i32 0
  %.unpack29 = extractelement <2 x double> %40, i32 1
  %41 = zext nneg i32 %16 to i64
  %42 = getelementptr inbounds { double, double }, ptr addrspace(1) %11, i64 %41
  %43 = load <2 x double>, ptr addrspace(1) %42, align 16, !invariant.load !4
  %.unpack310 = extractelement <2 x double> %43, i32 0
  %.unpack511 = extractelement <2 x double> %43, i32 1
  %44 = fadd double %.unpack8, %.unpack310
  %45 = fadd double %.unpack29, %.unpack511
  %46 = getelementptr inbounds { double, double }, ptr addrspace(1) %12, i64 %41
  %47 = insertelement <2 x double> poison, double %44, i32 0
  %48 = insertelement <2 x double> %47, double %45, i32 1
  store <2 x double> %48, ptr addrspace(1) %46, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 576}
!3 = !{i32 0, i32 128}
!4 = !{}
