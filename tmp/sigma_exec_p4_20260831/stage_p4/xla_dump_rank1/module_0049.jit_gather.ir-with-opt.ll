; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(787251200) %0, ptr noalias readonly align 16 captures(none) dereferenceable(4) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(1537600) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = icmp samesign ult i32 %10, 96100
  br i1 %11, label %12, label %25

12:                                               ; preds = %3
  %13 = load i32, ptr addrspace(1) %4, align 16, !invariant.load !4
  %14 = tail call i32 @llvm.smax.i32(i32 %13, i32 0)
  %15 = tail call i32 @llvm.umin.i32(i32 %14, i32 511)
  %16 = mul nuw nsw i32 %15, 96100
  %17 = add nuw nsw i32 %16, %10
  %18 = zext nneg i32 %17 to i64
  %19 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %18
  %20 = load <2 x double>, ptr addrspace(1) %19, align 16, !invariant.load !4
  %.unpack5 = extractelement <2 x double> %20, i32 0
  %.unpack26 = extractelement <2 x double> %20, i32 1
  %21 = zext nneg i32 %10 to i64
  %22 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %21
  %23 = insertelement <2 x double> poison, double %.unpack5, i32 0
  %24 = insertelement <2 x double> %23, double %.unpack26, i32 1
  store <2 x double> %24, ptr addrspace(1) %22, align 16
  br label %25

25:                                               ; preds = %12, %3
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
!2 = !{i32 0, i32 751}
!3 = !{i32 0, i32 128}
!4 = !{}
