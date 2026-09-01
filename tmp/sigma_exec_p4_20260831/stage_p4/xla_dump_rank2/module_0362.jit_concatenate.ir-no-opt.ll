; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_concatenate(ptr noalias align 16 dereferenceable(787251200) %0, ptr noalias align 16 dereferenceable(787251200) %1, ptr noalias align 16 dereferenceable(787251200) %2, ptr noalias align 16 dereferenceable(787251200) %3, ptr noalias align 256 dereferenceable(3149004800) %4) #0 {
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = mul i32 %6, 128
  %9 = add i32 %8, %7
  %10 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %9
  %11 = load { double, double }, ptr %10, align 8, !invariant.load !3
  %12 = getelementptr inbounds [196812800 x { double, double }], ptr %4, i32 0, i32 %9
  store { double, double } %11, ptr %12, align 8
  %13 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %9
  %14 = load { double, double }, ptr %13, align 8, !invariant.load !3
  %15 = add i32 %9, 49203200
  %16 = getelementptr inbounds [196812800 x { double, double }], ptr %4, i32 0, i32 %15
  store { double, double } %14, ptr %16, align 8
  %17 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i32 %9
  %18 = load { double, double }, ptr %17, align 8, !invariant.load !3
  %19 = add i32 %9, 98406400
  %20 = getelementptr inbounds [196812800 x { double, double }], ptr %4, i32 0, i32 %19
  store { double, double } %18, ptr %20, align 8
  %21 = getelementptr inbounds [49203200 x { double, double }], ptr %3, i32 0, i32 %9
  %22 = load { double, double }, ptr %21, align 8, !invariant.load !3
  %23 = add i32 %9, 147609600
  %24 = getelementptr inbounds [196812800 x { double, double }], ptr %4, i32 0, i32 %23
  store { double, double } %22, ptr %24, align 8
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
!1 = !{i32 0, i32 384400}
!2 = !{i32 0, i32 128}
!3 = !{}
