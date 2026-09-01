; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_select(ptr noalias align 16 dereferenceable(1572864) %0, ptr noalias align 16 dereferenceable(6291456) %1, ptr noalias align 16 dereferenceable(6291456) %2, ptr noalias align 256 dereferenceable(6291456) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = mul i32 %6, 4
  %8 = mul i32 %5, 512
  %9 = add i32 %7, %8
  %10 = getelementptr inbounds [1572864 x i32], ptr %2, i32 0, i32 %9
  %11 = load <4 x i32>, ptr %10, align 4, !invariant.load !3
  %12 = getelementptr inbounds [1572864 x i32], ptr %1, i32 0, i32 %9
  %13 = load <4 x i32>, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds [1572864 x i8], ptr %0, i32 0, i32 %9
  %15 = load <4 x i8>, ptr %14, align 1, !invariant.load !3
  %16 = extractelement <4 x i8> %15, i64 0
  %17 = extractelement <4 x i32> %13, i64 0
  %18 = extractelement <4 x i32> %11, i64 0
  %19 = trunc i8 %16 to i1
  %20 = select i1 %19, i32 %17, i32 %18
  %21 = extractelement <4 x i8> %15, i64 1
  %22 = extractelement <4 x i32> %13, i64 1
  %23 = extractelement <4 x i32> %11, i64 1
  %24 = trunc i8 %21 to i1
  %25 = select i1 %24, i32 %22, i32 %23
  %26 = extractelement <4 x i8> %15, i64 2
  %27 = extractelement <4 x i32> %13, i64 2
  %28 = extractelement <4 x i32> %11, i64 2
  %29 = trunc i8 %26 to i1
  %30 = select i1 %29, i32 %27, i32 %28
  %31 = extractelement <4 x i8> %15, i64 3
  %32 = extractelement <4 x i32> %13, i64 3
  %33 = extractelement <4 x i32> %11, i64 3
  %34 = trunc i8 %31 to i1
  %35 = select i1 %34, i32 %32, i32 %33
  %36 = insertelement <4 x i32> poison, i32 %20, i32 0
  %37 = insertelement <4 x i32> %36, i32 %25, i32 1
  %38 = insertelement <4 x i32> %37, i32 %30, i32 2
  %39 = insertelement <4 x i32> %38, i32 %35, i32 3
  %40 = getelementptr inbounds [1572864 x i32], ptr %3, i32 0, i32 %9
  store <4 x i32> %39, ptr %40, align 4
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
