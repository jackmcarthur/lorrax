; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_2_0 = local_unnamed_addr addrspace(1) global [2048 x i8] zeroinitializer, align 256

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_gather(ptr noalias readonly align 16 captures(none) dereferenceable(66816) %0, ptr noalias readonly align 256 captures(none) dereferenceable(2048) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(1179648) %2) local_unnamed_addr #0 {
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
  %17 = mul nuw nsw i32 %16, 144
  %18 = mul i32 %11, 144
  %urem.decomposed = sub i32 %10, %18
  %19 = add nuw nsw i32 %urem.decomposed, %17
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %20
  %22 = load <2 x double>, ptr addrspace(1) %21, align 16, !invariant.load !4
  %.unpack5 = extractelement <2 x double> %22, i32 0
  %.unpack26 = extractelement <2 x double> %22, i32 1
  %23 = zext nneg i32 %10 to i64
  %24 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %23
  %25 = insertelement <2 x double> poison, double %.unpack5, i32 0
  %26 = insertelement <2 x double> %25, double %.unpack26, i32 1
  store <2 x double> %26, ptr addrspace(1) %24, align 16
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
