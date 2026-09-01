; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_dynamic_slice_fusion(ptr noalias align 256 dereferenceable(524288) %0, ptr noalias align 16 dereferenceable(12) %1, ptr noalias align 256 dereferenceable(16) %2) #0 {
  %4 = getelementptr inbounds [3 x i32], ptr %1, i32 0, i32 0
  %5 = load i32, ptr %4, align 4, !invariant.load !1
  %6 = call i32 @llvm.smin.i32(i32 %5, i32 31)
  %7 = call i32 @llvm.smax.i32(i32 %6, i32 0)
  %8 = getelementptr inbounds [3 x i32], ptr %1, i32 0, i32 1
  %9 = load i32, ptr %8, align 4, !invariant.load !1
  %10 = call i32 @llvm.smin.i32(i32 %9, i32 31)
  %11 = call i32 @llvm.smax.i32(i32 %10, i32 0)
  %12 = getelementptr inbounds [3 x i32], ptr %1, i32 0, i32 2
  %13 = load i32, ptr %12, align 4, !invariant.load !1
  %14 = call i32 @llvm.smin.i32(i32 %13, i32 31)
  %15 = call i32 @llvm.smax.i32(i32 %14, i32 0)
  %16 = mul i32 %7, 1024
  %17 = mul i32 %11, 32
  %18 = add i32 %16, %17
  %19 = add i32 %18, %15
  %20 = getelementptr inbounds [32768 x { double, double }], ptr %0, i32 0, i32 %19
  %21 = load { double, double }, ptr %20, align 8, !invariant.load !1
  %22 = getelementptr inbounds [1 x { double, double }], ptr %2, i32 0, i32 0
  store { double, double } %21, ptr %22, align 8
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #1

define ptx_kernel void @loop_dynamic_update_slice_fusion(ptr noalias align 256 dereferenceable(524288) %0, ptr noalias align 256 dereferenceable(16) %1, ptr noalias align 16 dereferenceable(16) %2, ptr noalias align 16 dereferenceable(12) %3, ptr noalias align 256 dereferenceable(524288) %4) #0 {
  %6 = getelementptr inbounds [3 x i32], ptr %3, i32 0, i32 0
  %7 = load i32, ptr %6, align 4, !invariant.load !1
  %8 = getelementptr inbounds [3 x i32], ptr %3, i32 0, i32 1
  %9 = load i32, ptr %8, align 4, !invariant.load !1
  %10 = getelementptr inbounds [3 x i32], ptr %3, i32 0, i32 2
  %11 = load i32, ptr %10, align 4, !invariant.load !1
  %12 = call i32 @llvm.smin.i32(i32 %7, i32 31)
  %13 = call i32 @llvm.smax.i32(i32 %12, i32 0)
  %14 = call i32 @llvm.smin.i32(i32 %9, i32 31)
  %15 = call i32 @llvm.smax.i32(i32 %14, i32 0)
  %16 = call i32 @llvm.smin.i32(i32 %11, i32 31)
  %17 = call i32 @llvm.smax.i32(i32 %16, i32 0)
  %18 = icmp sge i32 %7, 0
  %19 = icmp sle i32 %7, 31
  %20 = and i1 %18, %19
  %21 = zext i1 %20 to i8
  %22 = and i8 %21, 1
  %23 = icmp sge i32 %9, 0
  %24 = icmp sle i32 %9, 31
  %25 = and i1 %23, %24
  %26 = zext i1 %25 to i8
  %27 = and i8 %22, %26
  %28 = icmp sge i32 %11, 0
  %29 = icmp sle i32 %11, 31
  %30 = and i1 %28, %29
  %31 = zext i1 %30 to i8
  %32 = and i8 %27, %31
  %33 = getelementptr inbounds [1 x { double, double }], ptr %2, i32 0, i32 0
  %34 = load { double, double }, ptr %33, align 8, !invariant.load !1
  %35 = getelementptr inbounds [1 x { double, double }], ptr %1, i32 0, i32 0
  %36 = load { double, double }, ptr %35, align 8, !invariant.load !1
  %37 = trunc i8 %32 to i1
  %38 = select i1 %37, { double, double } %34, { double, double } %36
  %39 = mul i32 %13, 1024
  %40 = mul i32 %15, 32
  %41 = add i32 %39, %40
  %42 = add i32 %41, %17
  %43 = getelementptr inbounds [32768 x { double, double }], ptr %0, i32 0, i32 %42
  store { double, double } %38, ptr %43, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
