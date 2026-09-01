; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 16 dereferenceable(4718592) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(4718592) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  %7 = load double, ptr %6, align 8, !invariant.load !3
  %8 = mul i32 %5, 4
  %9 = mul i32 %4, 512
  %10 = add i32 %8, %9
  %11 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %10
  %12 = load { double, double }, ptr %11, align 8, !invariant.load !3
  %13 = extractvalue { double, double } %12, 0
  %14 = extractvalue { double, double } %12, 1
  %15 = fmul double %7, %13
  %16 = fmul double %14, 0.000000e+00
  %17 = fsub double %15, %16
  %18 = fmul double %13, 0.000000e+00
  %19 = fmul double %7, %14
  %20 = fadd double %18, %19
  %21 = insertvalue { double, double } poison, double %17, 0
  %22 = insertvalue { double, double } %21, double %20, 1
  %23 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %10
  store { double, double } %22, ptr %23, align 8
  %24 = add i32 %10, 1
  %25 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %24
  %26 = load { double, double }, ptr %25, align 8, !invariant.load !3
  %27 = extractvalue { double, double } %26, 0
  %28 = extractvalue { double, double } %26, 1
  %29 = fmul double %7, %27
  %30 = fmul double %28, 0.000000e+00
  %31 = fsub double %29, %30
  %32 = fmul double %27, 0.000000e+00
  %33 = fmul double %7, %28
  %34 = fadd double %32, %33
  %35 = insertvalue { double, double } poison, double %31, 0
  %36 = insertvalue { double, double } %35, double %34, 1
  %37 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %24
  store { double, double } %36, ptr %37, align 8
  %38 = add i32 %10, 2
  %39 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %38
  %40 = load { double, double }, ptr %39, align 8, !invariant.load !3
  %41 = extractvalue { double, double } %40, 0
  %42 = extractvalue { double, double } %40, 1
  %43 = fmul double %7, %41
  %44 = fmul double %42, 0.000000e+00
  %45 = fsub double %43, %44
  %46 = fmul double %41, 0.000000e+00
  %47 = fmul double %7, %42
  %48 = fadd double %46, %47
  %49 = insertvalue { double, double } poison, double %45, 0
  %50 = insertvalue { double, double } %49, double %48, 1
  %51 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %38
  store { double, double } %50, ptr %51, align 8
  %52 = add i32 %10, 3
  %53 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %52
  %54 = load { double, double }, ptr %53, align 8, !invariant.load !3
  %55 = extractvalue { double, double } %54, 0
  %56 = extractvalue { double, double } %54, 1
  %57 = fmul double %7, %55
  %58 = fmul double %56, 0.000000e+00
  %59 = fsub double %57, %58
  %60 = fmul double %55, 0.000000e+00
  %61 = fmul double %7, %56
  %62 = fadd double %60, %61
  %63 = insertvalue { double, double } poison, double %59, 0
  %64 = insertvalue { double, double } %63, double %62, 1
  %65 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %52
  store { double, double } %64, ptr %65, align 8
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
