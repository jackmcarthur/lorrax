; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_and_fusion(ptr noalias align 16 dereferenceable(32) %0, ptr noalias align 16 dereferenceable(32) %1, ptr noalias align 256 dereferenceable(1024) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 4
  %7 = udiv i32 %5, 32
  %8 = add i32 %6, %7
  %9 = urem i32 %5, 32
  %10 = getelementptr inbounds [32 x i8], ptr %0, i32 0, i32 %8
  %11 = load i8, ptr %10, align 1, !invariant.load !3
  %12 = getelementptr inbounds [32 x i8], ptr %1, i32 0, i32 %9
  %13 = load i8, ptr %12, align 1, !invariant.load !3
  %14 = and i8 %11, %13
  %15 = mul i32 %4, 128
  %16 = add i32 %15, %5
  %17 = getelementptr inbounds [1024 x i8], ptr %2, i32 0, i32 %16
  store i8 %14, ptr %17, align 1
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
!1 = !{i32 0, i32 8}
!2 = !{i32 0, i32 128}
!3 = !{}
