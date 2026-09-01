; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_broadcast(ptr noalias readonly align 16 captures(none) dereferenceable(16) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4718592) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %7 = load <2 x double>, ptr addrspace(1) %3, align 16, !invariant.load !4
  %.unpack11 = extractelement <2 x double> %7, i32 0
  %.unpack212 = extractelement <2 x double> %7, i32 1
  %8 = shl nuw nsw i32 %6, 2
  %9 = shl nuw nsw i32 %5, 9
  %10 = or disjoint i32 %8, %9
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %11
  %13 = insertelement <2 x double> poison, double %.unpack11, i32 0
  %14 = insertelement <2 x double> %13, double %.unpack212, i32 1
  store <2 x double> %14, ptr addrspace(1) %12, align 64
  %15 = getelementptr inbounds i8, ptr addrspace(1) %12, i64 16
  %16 = insertelement <2 x double> poison, double %.unpack11, i32 0
  %17 = insertelement <2 x double> %16, double %.unpack212, i32 1
  store <2 x double> %17, ptr addrspace(1) %15, align 16
  %18 = getelementptr inbounds i8, ptr addrspace(1) %12, i64 32
  %19 = insertelement <2 x double> poison, double %.unpack11, i32 0
  %20 = insertelement <2 x double> %19, double %.unpack212, i32 1
  store <2 x double> %20, ptr addrspace(1) %18, align 32
  %21 = getelementptr inbounds i8, ptr addrspace(1) %12, i64 48
  %22 = insertelement <2 x double> poison, double %.unpack11, i32 0
  %23 = insertelement <2 x double> %22, double %.unpack212, i32 1
  store <2 x double> %23, ptr addrspace(1) %21, align 16
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
