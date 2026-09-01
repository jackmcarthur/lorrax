; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_broadcast(ptr noalias align 16 dereferenceable(16) %0, ptr noalias align 256 dereferenceable(4718592) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %5 = getelementptr inbounds [1 x { double, double }], ptr %0, i32 0, i32 0
  %6 = load { double, double }, ptr %5, align 8, !invariant.load !3
  %7 = mul i32 %4, 4
  %8 = mul i32 %3, 512
  %9 = add i32 %7, %8
  %10 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %9
  store { double, double } %6, ptr %10, align 8
  %11 = add i32 %9, 1
  %12 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %11
  store { double, double } %6, ptr %12, align 8
  %13 = add i32 %9, 2
  %14 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %13
  store { double, double } %6, ptr %14, align 8
  %15 = add i32 %9, 3
  %16 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %15
  store { double, double } %6, ptr %16, align 8
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
