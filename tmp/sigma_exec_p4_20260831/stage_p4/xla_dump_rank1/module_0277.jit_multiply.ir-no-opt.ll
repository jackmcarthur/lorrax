; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_multiply(ptr noalias align 16 dereferenceable(262144) %0, ptr noalias align 16 dereferenceable(262144) %1, ptr noalias align 256 dereferenceable(262144) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %7
  %9 = load double, ptr %8, align 8, !invariant.load !3
  %10 = getelementptr inbounds [32768 x double], ptr %1, i32 0, i32 %7
  %11 = load double, ptr %10, align 8, !invariant.load !3
  %12 = fmul double %9, %11
  %13 = getelementptr inbounds [32768 x double], ptr %2, i32 0, i32 %7
  store double %12, ptr %13, align 8
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
!1 = !{i32 0, i32 256}
!2 = !{i32 0, i32 128}
!3 = !{}
