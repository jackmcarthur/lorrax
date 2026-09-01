; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias align 16 dereferenceable(787251200) %0, ptr noalias align 16 dereferenceable(4) %1, ptr noalias align 256 dereferenceable(1537600) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = icmp sle i32 %7, 96099
  br i1 %8, label %9, label %19

9:                                                ; preds = %3
  %10 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  %11 = load i32, ptr %10, align 4, !invariant.load !3
  %12 = call i32 @llvm.smin.i32(i32 %11, i32 511)
  %13 = call i32 @llvm.smax.i32(i32 %12, i32 0)
  %14 = mul i32 %13, 96100
  %15 = add i32 %7, %14
  %16 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %15
  %17 = load { double, double }, ptr %16, align 8, !invariant.load !3
  %18 = getelementptr inbounds [96100 x { double, double }], ptr %2, i32 0, i32 %7
  store { double, double } %17, ptr %18, align 8
  br label %19

19:                                               ; preds = %9, %3
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
!1 = !{i32 0, i32 751}
!2 = !{i32 0, i32 128}
!3 = !{}
