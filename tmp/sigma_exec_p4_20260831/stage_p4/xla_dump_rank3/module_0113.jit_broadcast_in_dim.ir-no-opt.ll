; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_broadcast(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(262144) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %5 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %6 = load double, ptr %5, align 8, !invariant.load !3
  %7 = mul i32 %3, 128
  %8 = add i32 %7, %4
  %9 = getelementptr inbounds [32768 x double], ptr %1, i32 0, i32 %8
  store double %6, ptr %9, align 8
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
