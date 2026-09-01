; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 16 dereferenceable(1179648) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(1179648) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %7 = load double, ptr %6, align 8, !invariant.load !3
  %8 = mul i32 %4, 128
  %9 = add i32 %8, %5
  %10 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %9
  %11 = load { double, double }, ptr %10, align 8, !invariant.load !3
  %12 = extractvalue { double, double } %11, 0
  %13 = extractvalue { double, double } %11, 1
  %14 = fmul double %7, %12
  %15 = fmul double %13, 0.000000e+00
  %16 = fsub double %14, %15
  %17 = fmul double %12, 0.000000e+00
  %18 = fmul double %7, %13
  %19 = fadd double %17, %18
  %20 = insertvalue { double, double } poison, double %16, 0
  %21 = insertvalue { double, double } %20, double %19, 1
  %22 = getelementptr inbounds [73728 x { double, double }], ptr %2, i32 0, i32 %9
  store { double, double } %21, ptr %22, align 8
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
!1 = !{i32 0, i32 576}
!2 = !{i32 0, i32 128}
!3 = !{}
