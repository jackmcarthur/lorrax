; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_transpose(ptr noalias align 16 dereferenceable(243793920) %0, ptr noalias align 256 dereferenceable(243793920) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %5 = mul i32 %3, 128
  %6 = add i32 %5, %4
  %7 = urem i32 %6, 12
  %8 = mul i32 %7, 8
  %9 = udiv i32 %6, 12
  %10 = urem i32 %9, 310
  %11 = mul i32 %10, 96
  %12 = add i32 %8, %11
  %13 = udiv i32 %6, 7440
  %14 = mul i32 %13, 29760
  %15 = add i32 %12, %14
  %16 = udiv i32 %6, 3720
  %17 = urem i32 %16, 2
  %18 = add i32 %15, %17
  %19 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %18
  %20 = load { double, double }, ptr %19, align 8, !invariant.load !3
  %21 = mul i32 %4, 4
  %22 = mul i32 %3, 512
  %23 = add i32 %21, %22
  %24 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %23
  store { double, double } %20, ptr %24, align 8
  %25 = add i32 %18, 2
  %26 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %25
  %27 = load { double, double }, ptr %26, align 8, !invariant.load !3
  %28 = add i32 %23, 1
  %29 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %28
  store { double, double } %27, ptr %29, align 8
  %30 = add i32 %18, 4
  %31 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %30
  %32 = load { double, double }, ptr %31, align 8, !invariant.load !3
  %33 = add i32 %23, 2
  %34 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %33
  store { double, double } %32, ptr %34, align 8
  %35 = add i32 %18, 6
  %36 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %35
  %37 = load { double, double }, ptr %36, align 8, !invariant.load !3
  %38 = add i32 %23, 3
  %39 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %38
  store { double, double } %37, ptr %39, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 29760}
!2 = !{i32 0, i32 128}
!3 = !{}
