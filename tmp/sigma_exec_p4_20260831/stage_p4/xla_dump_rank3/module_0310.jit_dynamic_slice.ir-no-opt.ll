; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias align 16 dereferenceable(196608) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(98304) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = sext i32 %4 to i64
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = sext i32 %6 to i64
  %8 = mul i64 %5, 128
  %9 = add i64 %8, %7
  %10 = urem i64 %9, 24
  %11 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %12 = load i64, ptr %11, align 4, !invariant.load !3
  %13 = call i64 @llvm.smin.i64(i64 %12, i64 24)
  %14 = call i64 @llvm.smax.i64(i64 %13, i64 0)
  %15 = add i64 %10, %14
  %16 = udiv i64 %9, 24
  %17 = mul i64 %16, 48
  %18 = add i64 %17, %15
  %19 = getelementptr inbounds [24576 x double], ptr %0, i32 0, i64 %18
  %20 = load double, ptr %19, align 8, !invariant.load !3
  %21 = getelementptr inbounds [12288 x double], ptr %2, i32 0, i64 %9
  store double %20, ptr %21, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #2

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 96}
!2 = !{i32 0, i32 128}
!3 = !{}
