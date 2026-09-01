; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_reduce_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(48) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(16) initializes((0, 16)) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = load <2 x double>, ptr addrspace(1) %3, align 16, !invariant.load !2
  %.unpack11 = extractelement <2 x double> %5, i32 0
  %.unpack212 = extractelement <2 x double> %5, i32 1
  %6 = fmul double %.unpack212, 0.000000e+00
  %7 = fsub double %.unpack11, %6
  %8 = fmul double %.unpack11, 0.000000e+00
  %9 = fadd double %8, %.unpack212
  %10 = getelementptr inbounds i8, ptr addrspace(1) %3, i64 16
  %11 = load <2 x double>, ptr addrspace(1) %10, align 16, !invariant.load !2
  %.unpack313 = extractelement <2 x double> %11, i32 0
  %.unpack514 = extractelement <2 x double> %11, i32 1
  %12 = fmul double %.unpack313, %7
  %13 = fmul double %9, %.unpack514
  %14 = fsub double %12, %13
  %15 = fmul double %9, %.unpack313
  %16 = fmul double %7, %.unpack514
  %17 = fadd double %15, %16
  %18 = getelementptr inbounds i8, ptr addrspace(1) %3, i64 32
  %19 = load <2 x double>, ptr addrspace(1) %18, align 16, !invariant.load !2
  %.unpack615 = extractelement <2 x double> %19, i32 0
  %.unpack816 = extractelement <2 x double> %19, i32 1
  %20 = fmul double %.unpack615, %14
  %21 = fmul double %17, %.unpack816
  %22 = fsub double %20, %21
  %23 = fmul double %.unpack615, %17
  %24 = fmul double %14, %.unpack816
  %25 = fadd double %23, %24
  %26 = insertelement <2 x double> poison, double %22, i32 0
  %27 = insertelement <2 x double> %26, double %25, i32 1
  store <2 x double> %27, ptr addrspace(1) %4, align 256
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
