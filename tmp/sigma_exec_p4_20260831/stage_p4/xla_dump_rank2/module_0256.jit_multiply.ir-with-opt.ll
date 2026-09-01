; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(524288) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(524288) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = load double, ptr addrspace(1) %4, align 16, !invariant.load !4
  %10 = shl nuw nsw i32 %7, 7
  %11 = or disjoint i32 %10, %8
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %12
  %14 = load <2 x double>, ptr addrspace(1) %13, align 16, !invariant.load !4
  %.unpack5 = extractelement <2 x double> %14, i32 0
  %.unpack26 = extractelement <2 x double> %14, i32 1
  %15 = fmul double %9, %.unpack5
  %16 = fmul double %.unpack26, 0.000000e+00
  %17 = fsub double %15, %16
  %18 = fmul double %.unpack5, 0.000000e+00
  %19 = fmul double %9, %.unpack26
  %20 = fadd double %18, %19
  %21 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %12
  %22 = insertelement <2 x double> poison, double %17, i32 0
  %23 = insertelement <2 x double> %22, double %20, i32 1
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
!2 = !{i32 0, i32 256}
!3 = !{i32 0, i32 128}
!4 = !{}
