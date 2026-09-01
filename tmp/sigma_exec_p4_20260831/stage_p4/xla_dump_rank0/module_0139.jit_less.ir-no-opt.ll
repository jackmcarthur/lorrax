; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_compare_fusion(ptr noalias align 16 dereferenceable(6291456) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(1572864) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %7 = load i64, ptr %6, align 4, !invariant.load !3
  %8 = trunc i64 %7 to i32
  %9 = mul i32 %5, 4
  %10 = mul i32 %4, 512
  %11 = add i32 %9, %10
  %12 = getelementptr inbounds [1572864 x i32], ptr %0, i32 0, i32 %11
  %13 = load <4 x i32>, ptr %12, align 4, !invariant.load !3
  %14 = extractelement <4 x i32> %13, i64 0
  %15 = icmp slt i32 %14, %8
  %16 = zext i1 %15 to i8
  %17 = extractelement <4 x i32> %13, i64 1
  %18 = icmp slt i32 %17, %8
  %19 = zext i1 %18 to i8
  %20 = extractelement <4 x i32> %13, i64 2
  %21 = icmp slt i32 %20, %8
  %22 = zext i1 %21 to i8
  %23 = extractelement <4 x i32> %13, i64 3
  %24 = icmp slt i32 %23, %8
  %25 = zext i1 %24 to i8
  %26 = insertelement <4 x i8> poison, i8 %16, i32 0
  %27 = insertelement <4 x i8> %26, i8 %19, i32 1
  %28 = insertelement <4 x i8> %27, i8 %22, i32 2
  %29 = insertelement <4 x i8> %28, i8 %25, i32 3
  %30 = getelementptr inbounds [1572864 x i8], ptr %2, i32 0, i32 %11
  store <4 x i8> %29, ptr %30, align 1
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
