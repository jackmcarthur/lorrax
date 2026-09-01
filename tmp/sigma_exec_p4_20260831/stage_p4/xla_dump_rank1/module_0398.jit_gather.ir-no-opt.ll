; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias align 16 dereferenceable(24772608) %0, ptr noalias align 16 dereferenceable(4) %1, ptr noalias align 256 dereferenceable(1179648) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  %7 = load i32, ptr %6, align 4, !invariant.load !3
  %8 = call i32 @llvm.smin.i32(i32 %7, i32 20)
  %9 = call i32 @llvm.smax.i32(i32 %8, i32 0)
  %10 = mul i32 %9, 73728
  %11 = mul i32 %4, 128
  %12 = add i32 %10, %11
  %13 = add i32 %12, %5
  %14 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %13
  %15 = load { double, double }, ptr %14, align 8, !invariant.load !3
  %16 = add i32 %11, %5
  %17 = getelementptr inbounds [73728 x { double, double }], ptr %2, i32 0, i32 %16
  store { double, double } %15, ptr %17, align 8
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
