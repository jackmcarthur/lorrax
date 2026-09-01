; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_concatenate(ptr noalias readonly align 16 captures(none) dereferenceable(787251200) %0, ptr noalias readonly align 16 captures(none) dereferenceable(787251200) %1, ptr noalias readonly align 16 captures(none) dereferenceable(787251200) %2, ptr noalias readonly align 16 captures(none) dereferenceable(787251200) %3, ptr noalias writeonly align 256 captures(none) dereferenceable(3149004800) %4) local_unnamed_addr #0 {
  %6 = addrspacecast ptr %0 to ptr addrspace(1)
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = addrspacecast ptr %1 to ptr addrspace(1)
  %9 = addrspacecast ptr %2 to ptr addrspace(1)
  %10 = addrspacecast ptr %3 to ptr addrspace(1)
  %11 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %12 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %13 = shl nuw nsw i32 %11, 7
  %14 = or disjoint i32 %13, %12
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %15
  %17 = load <2 x double>, ptr addrspace(1) %16, align 16, !invariant.load !4
  %.unpack20 = extractelement <2 x double> %17, i32 0
  %.unpack221 = extractelement <2 x double> %17, i32 1
  %18 = getelementptr inbounds { double, double }, ptr addrspace(1) %7, i64 %15
  %19 = insertelement <2 x double> poison, double %.unpack20, i32 0
  %20 = insertelement <2 x double> %19, double %.unpack221, i32 1
  store <2 x double> %20, ptr addrspace(1) %18, align 16
  %21 = getelementptr inbounds { double, double }, ptr addrspace(1) %8, i64 %15
  %22 = load <2 x double>, ptr addrspace(1) %21, align 16, !invariant.load !4
  %.unpack522 = extractelement <2 x double> %22, i32 0
  %.unpack723 = extractelement <2 x double> %22, i32 1
  %23 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 787251200
  %24 = insertelement <2 x double> poison, double %.unpack522, i32 0
  %25 = insertelement <2 x double> %24, double %.unpack723, i32 1
  store <2 x double> %25, ptr addrspace(1) %23, align 16
  %26 = getelementptr inbounds { double, double }, ptr addrspace(1) %9, i64 %15
  %27 = load <2 x double>, ptr addrspace(1) %26, align 16, !invariant.load !4
  %.unpack1024 = extractelement <2 x double> %27, i32 0
  %.unpack1225 = extractelement <2 x double> %27, i32 1
  %28 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 1574502400
  %29 = insertelement <2 x double> poison, double %.unpack1024, i32 0
  %30 = insertelement <2 x double> %29, double %.unpack1225, i32 1
  store <2 x double> %30, ptr addrspace(1) %28, align 16
  %31 = getelementptr inbounds { double, double }, ptr addrspace(1) %10, i64 %15
  %32 = load <2 x double>, ptr addrspace(1) %31, align 16, !invariant.load !4
  %.unpack1526 = extractelement <2 x double> %32, i32 0
  %.unpack1727 = extractelement <2 x double> %32, i32 1
  %33 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 2361753600
  %34 = insertelement <2 x double> poison, double %.unpack1526, i32 0
  %35 = insertelement <2 x double> %34, double %.unpack1727, i32 1
  store <2 x double> %35, ptr addrspace(1) %33, align 16
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
!2 = !{i32 0, i32 384400}
!3 = !{i32 0, i32 128}
!4 = !{}
