; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_gather_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(262144) %0, ptr noalias readonly align 16 captures(none) dereferenceable(6291456) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(12582912) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %8, 2
  %10 = shl nuw nsw i32 %7, 9
  %11 = or disjoint i32 %9, %10
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %12
  %14 = load <4 x i32>, ptr addrspace(1) %13, align 16, !invariant.load !4
  %15 = extractelement <4 x i32> %14, i32 0
  %16 = extractelement <4 x i32> %14, i32 1
  %17 = extractelement <4 x i32> %14, i32 2
  %18 = extractelement <4 x i32> %14, i32 3
  %19 = tail call i32 @llvm.smax.i32(i32 %15, i32 0)
  %20 = tail call i32 @llvm.umin.i32(i32 %19, i32 32767)
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %21
  %23 = load double, ptr addrspace(1) %22, align 8, !invariant.load !4
  %24 = tail call i32 @llvm.smax.i32(i32 %16, i32 0)
  %25 = tail call i32 @llvm.umin.i32(i32 %24, i32 32767)
  %26 = zext nneg i32 %25 to i64
  %27 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %26
  %28 = load double, ptr addrspace(1) %27, align 8, !invariant.load !4
  %29 = tail call i32 @llvm.smax.i32(i32 %17, i32 0)
  %30 = tail call i32 @llvm.umin.i32(i32 %29, i32 32767)
  %31 = zext nneg i32 %30 to i64
  %32 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %31
  %33 = load double, ptr addrspace(1) %32, align 8, !invariant.load !4
  %34 = tail call i32 @llvm.smax.i32(i32 %18, i32 0)
  %35 = tail call i32 @llvm.umin.i32(i32 %34, i32 32767)
  %36 = zext nneg i32 %35 to i64
  %37 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %36
  %38 = load double, ptr addrspace(1) %37, align 8, !invariant.load !4
  %39 = insertelement <4 x double> poison, double %23, i64 0
  %40 = insertelement <4 x double> %39, double %28, i64 1
  %41 = insertelement <4 x double> %40, double %33, i64 2
  %42 = insertelement <4 x double> %41, double %38, i64 3
  %43 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %12
  store <4 x double> %42, ptr addrspace(1) %43, align 32
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
!2 = !{i32 0, i32 3072}
!3 = !{i32 0, i32 128}
!4 = !{}
