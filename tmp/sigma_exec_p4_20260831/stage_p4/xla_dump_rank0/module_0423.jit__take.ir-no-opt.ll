; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_select_fusion(ptr noalias align 16 dereferenceable(1179648) %0, ptr noalias align 16 dereferenceable(232) %1, ptr noalias align 256 dereferenceable(66816) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = icmp sle i32 %7, 4175
  br i1 %8, label %9, label %33

9:                                                ; preds = %3
  %10 = udiv i32 %7, 144
  %11 = getelementptr inbounds [29 x i64], ptr %1, i32 0, i32 %10
  %12 = load i64, ptr %11, align 4, !invariant.load !3
  %13 = icmp slt i64 %12, 0
  %14 = add i64 %12, 512
  %15 = select i1 %13, i64 %14, i64 %12
  %16 = trunc i64 %15 to i32
  %17 = icmp sge i32 %16, 0
  %18 = icmp sle i32 %16, 511
  %19 = and i1 %17, %18
  %20 = call i32 @llvm.smin.i32(i32 %16, i32 511)
  %21 = call i32 @llvm.smax.i32(i32 %20, i32 0)
  %22 = udiv i32 %7, 12
  %23 = urem i32 %22, 12
  %24 = mul i32 %23, 12
  %25 = urem i32 %7, 12
  %26 = add i32 %24, %25
  %27 = mul i32 %21, 144
  %28 = add i32 %26, %27
  %29 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %28
  %30 = load { double, double }, ptr %29, align 8, !invariant.load !3
  %31 = select i1 %19, { double, double } %30, { double, double } { double 0x7FF8000000000000, double 0.000000e+00 }
  %32 = getelementptr inbounds [4176 x { double, double }], ptr %2, i32 0, i32 %7
  store { double, double } %31, ptr %32, align 8
  br label %33

33:                                               ; preds = %9, %3
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
!1 = !{i32 0, i32 33}
!2 = !{i32 0, i32 128}
!3 = !{}
