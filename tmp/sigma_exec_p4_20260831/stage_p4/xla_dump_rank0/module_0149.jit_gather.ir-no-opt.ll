; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_gather_fusion(ptr noalias align 16 dereferenceable(262144) %0, ptr noalias align 16 dereferenceable(6291456) %1, ptr noalias align 256 dereferenceable(12582912) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %5, 4
  %7 = mul i32 %4, 512
  %8 = add i32 %6, %7
  %9 = getelementptr inbounds [1572864 x i32], ptr %1, i32 0, i32 %8
  %10 = load <4 x i32>, ptr %9, align 4, !invariant.load !3
  %11 = extractelement <4 x i32> %10, i64 0
  %12 = call i32 @llvm.smin.i32(i32 %11, i32 32767)
  %13 = call i32 @llvm.smax.i32(i32 %12, i32 0)
  %14 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %13
  %15 = load double, ptr %14, align 8, !invariant.load !3
  %16 = extractelement <4 x i32> %10, i64 1
  %17 = call i32 @llvm.smin.i32(i32 %16, i32 32767)
  %18 = call i32 @llvm.smax.i32(i32 %17, i32 0)
  %19 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %18
  %20 = load double, ptr %19, align 8, !invariant.load !3
  %21 = extractelement <4 x i32> %10, i64 2
  %22 = call i32 @llvm.smin.i32(i32 %21, i32 32767)
  %23 = call i32 @llvm.smax.i32(i32 %22, i32 0)
  %24 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %23
  %25 = load double, ptr %24, align 8, !invariant.load !3
  %26 = extractelement <4 x i32> %10, i64 3
  %27 = call i32 @llvm.smin.i32(i32 %26, i32 32767)
  %28 = call i32 @llvm.smax.i32(i32 %27, i32 0)
  %29 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %28
  %30 = load double, ptr %29, align 8, !invariant.load !3
  %31 = insertelement <4 x double> poison, double %15, i32 0
  %32 = insertelement <4 x double> %31, double %20, i32 1
  %33 = insertelement <4 x double> %32, double %25, i32 2
  %34 = insertelement <4 x double> %33, double %30, i32 3
  %35 = getelementptr inbounds [1572864 x double], ptr %2, i32 0, i32 %8
  store <4 x double> %34, ptr %35, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 3072}
!2 = !{i32 0, i32 128}
!3 = !{}
