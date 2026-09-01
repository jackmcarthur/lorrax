; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_compare_fusion(ptr noalias align 16 dereferenceable(256) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(32) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = getelementptr inbounds [32 x i64], ptr %0, i32 0, i32 %4
  %6 = load i64, ptr %5, align 4, !invariant.load !2
  %7 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %8 = load i64, ptr %7, align 4, !invariant.load !2
  %9 = icmp eq i64 %6, %8
  %10 = zext i1 %9 to i8
  %11 = getelementptr inbounds [32 x i8], ptr %2, i32 0, i32 %4
  store i8 %10, ptr %11, align 1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="32,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 32}
!2 = !{}
