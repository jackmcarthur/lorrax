; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_slice(ptr noalias align 16 dereferenceable(243793920) %0, ptr noalias align 256 dereferenceable(121896960) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %5 = mul i32 %4, 2
  %6 = mul i32 %3, 256
  %7 = add i32 %5, %6
  %8 = udiv i32 %7, 155
  %9 = urem i32 %8, 2
  %10 = mul i32 %9, 310
  %11 = mul i32 %3, 128
  %12 = add i32 %11, %4
  %13 = udiv i32 %12, 155
  %14 = urem i32 %13, 24
  %15 = mul i32 %14, 620
  %16 = add i32 %10, %15
  %17 = udiv i32 %12, 3720
  %18 = mul i32 %17, 29760
  %19 = add i32 %16, %18
  %20 = mul i32 %4, 4
  %21 = mul i32 %3, 512
  %22 = add i32 %20, %21
  %23 = urem i32 %22, 310
  %24 = add i32 %19, %23
  %25 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %24
  %26 = load { double, double }, ptr %25, align 8, !invariant.load !3
  %27 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %22
  store { double, double } %26, ptr %27, align 8
  %28 = add i32 %22, 1
  %29 = urem i32 %28, 310
  %30 = add i32 %19, %29
  %31 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %30
  %32 = load { double, double }, ptr %31, align 8, !invariant.load !3
  %33 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %28
  store { double, double } %32, ptr %33, align 8
  %34 = add i32 %7, 1
  %35 = udiv i32 %34, 155
  %36 = urem i32 %35, 2
  %37 = mul i32 %36, 310
  %38 = add i32 %37, %15
  %39 = add i32 %38, %18
  %40 = add i32 %22, 2
  %41 = urem i32 %40, 310
  %42 = add i32 %39, %41
  %43 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %42
  %44 = load { double, double }, ptr %43, align 8, !invariant.load !3
  %45 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %40
  store { double, double } %44, ptr %45, align 8
  %46 = add i32 %22, 3
  %47 = udiv i32 %46, 310
  %48 = urem i32 %47, 2
  %49 = mul i32 %48, 310
  %50 = add i32 %49, %15
  %51 = add i32 %50, %18
  %52 = urem i32 %46, 310
  %53 = add i32 %51, %52
  %54 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %53
  %55 = load { double, double }, ptr %54, align 8, !invariant.load !3
  %56 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %46
  store { double, double } %55, ptr %56, align 8
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
