; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_slice(ptr noalias align 16 dereferenceable(243793920) %0, ptr noalias align 256 dereferenceable(121896960) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %5 = mul i32 %3, 128
  %6 = add i32 %5, %4
  %7 = urem i32 %6, 6
  %8 = mul i32 %7, 4
  %9 = udiv i32 %6, 6
  %10 = mul i32 %9, 48
  %11 = add i32 %8, %10
  %12 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %11
  %13 = load { double, double }, ptr %12, align 8, !invariant.load !3
  %14 = mul i32 %4, 4
  %15 = mul i32 %3, 512
  %16 = add i32 %14, %15
  %17 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %16
  store { double, double } %13, ptr %17, align 8
  %18 = add i32 %11, 1
  %19 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %18
  %20 = load { double, double }, ptr %19, align 8, !invariant.load !3
  %21 = add i32 %16, 1
  %22 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %21
  store { double, double } %20, ptr %22, align 8
  %23 = add i32 %11, 2
  %24 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %23
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !3
  %26 = add i32 %16, 2
  %27 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %26
  store { double, double } %25, ptr %27, align 8
  %28 = add i32 %11, 3
  %29 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %28
  %30 = load { double, double }, ptr %29, align 8, !invariant.load !3
  %31 = add i32 %16, 3
  %32 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %31
  store { double, double } %30, ptr %32, align 8
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
!1 = !{i32 0, i32 14880}
!2 = !{i32 0, i32 128}
!3 = !{}
