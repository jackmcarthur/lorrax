; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_select_fusion(ptr noalias align 16 dereferenceable(24772608) %0, ptr noalias align 16 dereferenceable(232) %1, ptr noalias align 256 dereferenceable(1403136) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = icmp sle i32 %7, 87695
  br i1 %8, label %9, label %37

9:                                                ; preds = %3
  %10 = udiv i32 %7, 144
  %11 = urem i32 %10, 29
  %12 = getelementptr inbounds [29 x i64], ptr %1, i32 0, i32 %11
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = icmp slt i64 %13, 0
  %15 = add i64 %13, 512
  %16 = select i1 %14, i64 %15, i64 %13
  %17 = trunc i64 %16 to i32
  %18 = icmp sge i32 %17, 0
  %19 = icmp sle i32 %17, 511
  %20 = and i1 %18, %19
  %21 = call i32 @llvm.smin.i32(i32 %17, i32 511)
  %22 = call i32 @llvm.smax.i32(i32 %21, i32 0)
  %23 = udiv i32 %7, 12
  %24 = urem i32 %23, 12
  %25 = mul i32 %24, 12
  %26 = udiv i32 %7, 4176
  %27 = mul i32 %26, 73728
  %28 = add i32 %25, %27
  %29 = urem i32 %7, 12
  %30 = add i32 %28, %29
  %31 = mul i32 %22, 144
  %32 = add i32 %30, %31
  %33 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %32
  %34 = load { double, double }, ptr %33, align 8, !invariant.load !3
  %35 = select i1 %20, { double, double } %34, { double, double } { double 0x7FF8000000000000, double 0.000000e+00 }
  %36 = getelementptr inbounds [87696 x { double, double }], ptr %2, i32 0, i32 %7
  store { double, double } %35, ptr %36, align 8
  br label %37

37:                                               ; preds = %9, %3
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
!1 = !{i32 0, i32 686}
!2 = !{i32 0, i32 128}
!3 = !{}
