; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 16 dereferenceable(178361600) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(178361600) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = icmp sle i32 %7, 2786899
  br i1 %8, label %9, label %70

9:                                                ; preds = %3
  %10 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %11 = load double, ptr %10, align 8, !invariant.load !3
  %12 = mul i32 %5, 4
  %13 = mul i32 %4, 512
  %14 = add i32 %12, %13
  %15 = getelementptr inbounds [11147600 x { double, double }], ptr %0, i32 0, i32 %14
  %16 = load { double, double }, ptr %15, align 8, !invariant.load !3
  %17 = extractvalue { double, double } %16, 0
  %18 = extractvalue { double, double } %16, 1
  %19 = fmul double %17, %11
  %20 = fmul double %18, 0.000000e+00
  %21 = fsub double %19, %20
  %22 = fmul double %18, %11
  %23 = fmul double %17, 0.000000e+00
  %24 = fadd double %22, %23
  %25 = insertvalue { double, double } poison, double %21, 0
  %26 = insertvalue { double, double } %25, double %24, 1
  %27 = getelementptr inbounds [11147600 x { double, double }], ptr %2, i32 0, i32 %14
  store { double, double } %26, ptr %27, align 8
  %28 = add i32 %14, 1
  %29 = getelementptr inbounds [11147600 x { double, double }], ptr %0, i32 0, i32 %28
  %30 = load { double, double }, ptr %29, align 8, !invariant.load !3
  %31 = extractvalue { double, double } %30, 0
  %32 = extractvalue { double, double } %30, 1
  %33 = fmul double %31, %11
  %34 = fmul double %32, 0.000000e+00
  %35 = fsub double %33, %34
  %36 = fmul double %32, %11
  %37 = fmul double %31, 0.000000e+00
  %38 = fadd double %36, %37
  %39 = insertvalue { double, double } poison, double %35, 0
  %40 = insertvalue { double, double } %39, double %38, 1
  %41 = getelementptr inbounds [11147600 x { double, double }], ptr %2, i32 0, i32 %28
  store { double, double } %40, ptr %41, align 8
  %42 = add i32 %14, 2
  %43 = getelementptr inbounds [11147600 x { double, double }], ptr %0, i32 0, i32 %42
  %44 = load { double, double }, ptr %43, align 8, !invariant.load !3
  %45 = extractvalue { double, double } %44, 0
  %46 = extractvalue { double, double } %44, 1
  %47 = fmul double %45, %11
  %48 = fmul double %46, 0.000000e+00
  %49 = fsub double %47, %48
  %50 = fmul double %46, %11
  %51 = fmul double %45, 0.000000e+00
  %52 = fadd double %50, %51
  %53 = insertvalue { double, double } poison, double %49, 0
  %54 = insertvalue { double, double } %53, double %52, 1
  %55 = getelementptr inbounds [11147600 x { double, double }], ptr %2, i32 0, i32 %42
  store { double, double } %54, ptr %55, align 8
  %56 = add i32 %14, 3
  %57 = getelementptr inbounds [11147600 x { double, double }], ptr %0, i32 0, i32 %56
  %58 = load { double, double }, ptr %57, align 8, !invariant.load !3
  %59 = extractvalue { double, double } %58, 0
  %60 = extractvalue { double, double } %58, 1
  %61 = fmul double %59, %11
  %62 = fmul double %60, 0.000000e+00
  %63 = fsub double %61, %62
  %64 = fmul double %60, %11
  %65 = fmul double %59, 0.000000e+00
  %66 = fadd double %64, %65
  %67 = insertvalue { double, double } poison, double %63, 0
  %68 = insertvalue { double, double } %67, double %66, 1
  %69 = getelementptr inbounds [11147600 x { double, double }], ptr %2, i32 0, i32 %56
  store { double, double } %68, ptr %69, align 8
  br label %70

70:                                               ; preds = %9, %3
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
!1 = !{i32 0, i32 21773}
!2 = !{i32 0, i32 128}
!3 = !{}
