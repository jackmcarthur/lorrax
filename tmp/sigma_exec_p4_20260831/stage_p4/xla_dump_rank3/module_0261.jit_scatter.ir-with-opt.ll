; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(524288) %0, ptr noalias readonly align 16 captures(none) dereferenceable(12) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(16) initializes((0, 16)) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = call <4 x i32> @llvm.masked.load.v4i32.p1(ptr addrspace(1) align 16 %4, <4 x i1> <i1 true, i1 true, i1 true, i1 false>, <4 x i32> poison), !invariant.load !2
  %8 = extractelement <4 x i32> %7, i32 0
  %9 = extractelement <4 x i32> %7, i32 1
  %10 = extractelement <4 x i32> %7, i32 2
  %Extend5 = extractelement <4 x i32> %7, i32 3
  %11 = tail call i32 @llvm.smax.i32(i32 %8, i32 0)
  %12 = tail call i32 @llvm.umin.i32(i32 %11, i32 31)
  %13 = tail call i32 @llvm.smax.i32(i32 %9, i32 0)
  %14 = tail call i32 @llvm.umin.i32(i32 %13, i32 31)
  %15 = tail call i32 @llvm.smax.i32(i32 %10, i32 0)
  %16 = tail call i32 @llvm.umin.i32(i32 %15, i32 31)
  %17 = shl nuw nsw i32 %12, 10
  %18 = shl nuw nsw i32 %14, 5
  %19 = or disjoint i32 %18, %17
  %20 = or disjoint i32 %19, %16
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %21
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !2
  %.unpack6 = extractelement <2 x double> %23, i32 0
  %.unpack27 = extractelement <2 x double> %23, i32 1
  %24 = insertelement <2 x double> poison, double %.unpack6, i32 0
  %25 = insertelement <2 x double> %24, double %.unpack27, i32 1
  store <2 x double> %25, ptr addrspace(1) %6, align 256
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_dynamic_update_slice_fusion(ptr noalias writeonly align 256 captures(none) dereferenceable(524288) %0, ptr noalias readonly align 256 captures(none) dereferenceable(16) %1, ptr noalias readonly align 16 captures(none) dereferenceable(16) %2, ptr noalias readonly align 16 captures(none) dereferenceable(12) %3, ptr noalias readnone align 256 captures(none) dereferenceable(524288) %4) local_unnamed_addr #0 {
  %6 = addrspacecast ptr %3 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %1 to ptr addrspace(1)
  %9 = addrspacecast ptr %0 to ptr addrspace(1)
  %10 = call <4 x i32> @llvm.masked.load.v4i32.p1(ptr addrspace(1) align 16 %6, <4 x i1> <i1 true, i1 true, i1 true, i1 false>, <4 x i32> poison), !invariant.load !2
  %11 = extractelement <4 x i32> %10, i32 0
  %12 = extractelement <4 x i32> %10, i32 1
  %13 = extractelement <4 x i32> %10, i32 2
  %Extend8 = extractelement <4 x i32> %10, i32 3
  %14 = tail call i32 @llvm.smax.i32(i32 %11, i32 0)
  %15 = tail call i32 @llvm.umin.i32(i32 %14, i32 31)
  %16 = tail call i32 @llvm.smax.i32(i32 %12, i32 0)
  %17 = tail call i32 @llvm.umin.i32(i32 %16, i32 31)
  %18 = tail call i32 @llvm.smax.i32(i32 %13, i32 0)
  %19 = tail call i32 @llvm.umin.i32(i32 %18, i32 31)
  %20 = or i32 %12, %11
  %21 = or i32 %20, %13
  %22 = icmp ult i32 %21, 32
  %23 = load <2 x double>, ptr addrspace(1) %7, align 16, !invariant.load !2
  %.unpack9 = extractelement <2 x double> %23, i32 0
  %.unpack210 = extractelement <2 x double> %23, i32 1
  %24 = load <2 x double>, ptr addrspace(1) %8, align 256, !invariant.load !2
  %.unpack311 = extractelement <2 x double> %24, i32 0
  %.unpack512 = extractelement <2 x double> %24, i32 1
  %25 = shl nuw nsw i32 %15, 10
  %26 = shl nuw nsw i32 %17, 5
  %27 = or disjoint i32 %26, %25
  %28 = or disjoint i32 %27, %19
  %29 = zext nneg i32 %28 to i64
  %30 = getelementptr inbounds { double, double }, ptr addrspace(1) %9, i64 %29
  %.elt = select i1 %22, double %.unpack9, double %.unpack311
  %.elt7 = select i1 %22, double %.unpack210, double %.unpack512
  %31 = insertelement <2 x double> poison, double %.elt, i32 0
  %32 = insertelement <2 x double> %31, double %.elt7, i32 1
  store <2 x double> %32, ptr addrspace(1) %30, align 16
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.masked.load.v4i32.p1(ptr addrspace(1) captures(none), <4 x i1>, <4 x i32>) #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
