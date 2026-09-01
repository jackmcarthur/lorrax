; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_gather_fusion(ptr noalias align 16 dereferenceable(131072) %0, ptr noalias align 16 dereferenceable(784384) %1, ptr noalias align 256 dereferenceable(16777216) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = urem i32 %4, 64
  %7 = mul i32 %6, 512
  %8 = mul i32 %5, 4
  %9 = add i32 %7, %8
  %10 = getelementptr inbounds [32768 x i32], ptr %0, i32 0, i32 %9
  %11 = load <4 x i32>, ptr %10, align 4, !invariant.load !3
  %12 = extractelement <4 x i32> %11, i64 0
  %13 = call i32 @llvm.smin.i32(i32 %12, i32 1532)
  %14 = call i32 @llvm.smax.i32(i32 %13, i32 0)
  %15 = icmp sle i32 %14, 1531
  br i1 %15, label %16, label %22

16:                                               ; preds = %3
  %17 = udiv i32 %4, 64
  %18 = mul i32 %17, 1532
  %19 = add i32 %18, %14
  %20 = getelementptr inbounds [49024 x { double, double }], ptr %1, i32 0, i32 %19
  %21 = load { double, double }, ptr %20, align 8, !invariant.load !3
  br label %23

22:                                               ; preds = %3
  br label %23

23:                                               ; preds = %16, %22
  %24 = phi { double, double } [ zeroinitializer, %22 ], [ %21, %16 ]
  br label %25

25:                                               ; preds = %23
  %26 = mul i32 %4, 512
  %27 = add i32 %8, %26
  %28 = getelementptr inbounds [1048576 x { double, double }], ptr %2, i32 0, i32 %27
  store { double, double } %24, ptr %28, align 8
  %29 = extractelement <4 x i32> %11, i64 1
  %30 = call i32 @llvm.smin.i32(i32 %29, i32 1532)
  %31 = call i32 @llvm.smax.i32(i32 %30, i32 0)
  %32 = icmp sle i32 %31, 1531
  br i1 %32, label %33, label %39

33:                                               ; preds = %25
  %34 = udiv i32 %4, 64
  %35 = mul i32 %34, 1532
  %36 = add i32 %35, %31
  %37 = getelementptr inbounds [49024 x { double, double }], ptr %1, i32 0, i32 %36
  %38 = load { double, double }, ptr %37, align 8, !invariant.load !3
  br label %40

39:                                               ; preds = %25
  br label %40

40:                                               ; preds = %33, %39
  %41 = phi { double, double } [ zeroinitializer, %39 ], [ %38, %33 ]
  br label %42

42:                                               ; preds = %40
  %43 = add i32 %27, 1
  %44 = getelementptr inbounds [1048576 x { double, double }], ptr %2, i32 0, i32 %43
  store { double, double } %41, ptr %44, align 8
  %45 = extractelement <4 x i32> %11, i64 2
  %46 = call i32 @llvm.smin.i32(i32 %45, i32 1532)
  %47 = call i32 @llvm.smax.i32(i32 %46, i32 0)
  %48 = icmp sle i32 %47, 1531
  br i1 %48, label %49, label %55

49:                                               ; preds = %42
  %50 = udiv i32 %4, 64
  %51 = mul i32 %50, 1532
  %52 = add i32 %51, %47
  %53 = getelementptr inbounds [49024 x { double, double }], ptr %1, i32 0, i32 %52
  %54 = load { double, double }, ptr %53, align 8, !invariant.load !3
  br label %56

55:                                               ; preds = %42
  br label %56

56:                                               ; preds = %49, %55
  %57 = phi { double, double } [ zeroinitializer, %55 ], [ %54, %49 ]
  br label %58

58:                                               ; preds = %56
  %59 = add i32 %27, 2
  %60 = getelementptr inbounds [1048576 x { double, double }], ptr %2, i32 0, i32 %59
  store { double, double } %57, ptr %60, align 8
  %61 = extractelement <4 x i32> %11, i64 3
  %62 = call i32 @llvm.smin.i32(i32 %61, i32 1532)
  %63 = call i32 @llvm.smax.i32(i32 %62, i32 0)
  %64 = icmp sle i32 %63, 1531
  br i1 %64, label %65, label %71

65:                                               ; preds = %58
  %66 = udiv i32 %4, 64
  %67 = mul i32 %66, 1532
  %68 = add i32 %67, %63
  %69 = getelementptr inbounds [49024 x { double, double }], ptr %1, i32 0, i32 %68
  %70 = load { double, double }, ptr %69, align 8, !invariant.load !3
  br label %72

71:                                               ; preds = %58
  br label %72

72:                                               ; preds = %65, %71
  %73 = phi { double, double } [ zeroinitializer, %71 ], [ %70, %65 ]
  br label %74

74:                                               ; preds = %72
  %75 = add i32 %27, 3
  %76 = getelementptr inbounds [1048576 x { double, double }], ptr %2, i32 0, i32 %75
  store { double, double } %73, ptr %76, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 2048}
!2 = !{i32 0, i32 128}
!3 = !{}
