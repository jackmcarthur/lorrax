; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_add_fusion(ptr noalias align 16 dereferenceable(6291456) %0, ptr noalias align 16 dereferenceable(4) %1, ptr noalias align 256 dereferenceable(6291456) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  %7 = load i32, ptr %6, align 4, !invariant.load !3
  %8 = mul i32 %5, 4
  %9 = mul i32 %4, 512
  %10 = add i32 %8, %9
  %11 = getelementptr inbounds [1572864 x i32], ptr %0, i32 0, i32 %10
  %12 = load <4 x i32>, ptr %11, align 4, !invariant.load !3
  %13 = extractelement <4 x i32> %12, i64 0
  %14 = add i32 %13, %7
  %15 = extractelement <4 x i32> %12, i64 1
  %16 = add i32 %15, %7
  %17 = extractelement <4 x i32> %12, i64 2
  %18 = add i32 %17, %7
  %19 = extractelement <4 x i32> %12, i64 3
  %20 = add i32 %19, %7
  %21 = insertelement <4 x i32> poison, i32 %14, i32 0
  %22 = insertelement <4 x i32> %21, i32 %16, i32 1
  %23 = insertelement <4 x i32> %22, i32 %18, i32 2
  %24 = insertelement <4 x i32> %23, i32 %20, i32 3
  %25 = getelementptr inbounds [1572864 x i32], ptr %2, i32 0, i32 %10
  store <4 x i32> %24, ptr %25, align 4
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
!1 = !{i32 0, i32 3072}
!2 = !{i32 0, i32 128}
!3 = !{}
