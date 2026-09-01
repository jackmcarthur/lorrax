; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_gather(ptr noalias align 16 dereferenceable(66816) %0, ptr noalias align 256 dereferenceable(2048) %1, ptr noalias align 256 dereferenceable(1179648) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = udiv i32 %7, 144
  %9 = getelementptr inbounds [512 x i32], ptr %1, i32 0, i32 %8
  %10 = load i32, ptr %9, align 4, !invariant.load !3
  %11 = call i32 @llvm.smin.i32(i32 %10, i32 28)
  %12 = call i32 @llvm.smax.i32(i32 %11, i32 0)
  %13 = udiv i32 %7, 12
  %14 = urem i32 %13, 12
  %15 = mul i32 %14, 12
  %16 = mul i32 %12, 144
  %17 = add i32 %15, %16
  %18 = urem i32 %7, 12
  %19 = add i32 %17, %18
  %20 = getelementptr inbounds [4176 x { double, double }], ptr %0, i32 0, i32 %19
  %21 = load { double, double }, ptr %20, align 8, !invariant.load !3
  %22 = getelementptr inbounds [73728 x { double, double }], ptr %2, i32 0, i32 %7
  store { double, double } %21, ptr %22, align 8
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
!1 = !{i32 0, i32 576}
!2 = !{i32 0, i32 128}
!3 = !{}
