; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_fabs(double)

declare double @__nv_sqrt(double)

define ptx_kernel void @loop_complex_fusion(ptr noalias align 16 dereferenceable(16) %0, ptr noalias align 256 dereferenceable(16) %1) #0 {
  %3 = getelementptr inbounds [1 x { double, double }], ptr %0, i32 0, i32 0
  %4 = load { double, double }, ptr %3, align 8, !invariant.load !1
  %5 = extractvalue { double, double } %4, 0
  %6 = extractvalue { double, double } %4, 1
  %7 = call double @__nv_fabs(double %5)
  %8 = call double @__nv_fabs(double %6)
  %9 = fcmp olt double %7, %8
  %10 = fdiv double %7, %8
  %11 = fdiv double %8, %7
  %12 = fcmp oeq double %7, %8
  %13 = select i1 %9, double %10, double %11
  %14 = select i1 %12, double 1.000000e+00, double %13
  %15 = call double @__nv_fabs(double %14)
  %16 = call double @llvm.minimum.f64(double %15, double 1.000000e+00)
  %17 = call double @llvm.maximum.f64(double %15, double 1.000000e+00)
  %18 = fdiv double %16, %17
  %19 = fmul double %18, %18
  %20 = fmul double %17, %19
  %21 = fadd double %19, 1.000000e+00
  %22 = call double @llvm.minimum.f64(double %7, double %8)
  %23 = call double @llvm.maximum.f64(double %7, double %8)
  %24 = fdiv double %22, %23
  %25 = call double @__nv_sqrt(double %21)
  %26 = fcmp oeq double %25, 1.000000e+00
  %27 = fcmp ogt double %19, 0.000000e+00
  %28 = fmul double %20, 5.000000e-01
  %29 = fmul double %24, %24
  %30 = fmul double %23, %29
  %31 = fadd double %29, 1.000000e+00
  %32 = and i1 %26, %27
  %33 = fadd double %17, %28
  %34 = fmul double %17, %25
  %35 = call double @__nv_sqrt(double %31)
  %36 = fcmp oeq double %35, 1.000000e+00
  %37 = fcmp ogt double %29, 0.000000e+00
  %38 = fmul double %30, 5.000000e-01
  %39 = fcmp oeq double %17, %16
  %40 = fmul double %17, 0x3FF6A09E667F3BCD
  %41 = select i1 %32, double %33, double %34
  %42 = and i1 %36, %37
  %43 = fadd double %23, %38
  %44 = fmul double %23, %35
  %45 = call double @__nv_sqrt(double %7)
  %46 = call double @__nv_sqrt(double %8)
  %47 = fdiv double %45, %46
  %48 = fdiv double %46, %45
  %49 = select i1 %39, double %40, double %41
  %50 = fcmp oeq double %23, %22
  %51 = fmul double %23, 0x3FF6A09E667F3BCD
  %52 = select i1 %42, double %43, double %44
  %53 = select i1 %9, double %47, double %48
  %54 = fadd double %49, 1.000000e+00
  %55 = fadd double %14, %49
  %56 = select i1 %50, double %51, double %52
  %57 = select i1 %12, double 1.000000e+00, double %53
  %58 = call double @__nv_sqrt(double %54)
  %59 = call double @__nv_sqrt(double %55)
  %60 = fmul double %58, 0x3FE6A09E667F3BCC
  %61 = fmul double %59, 0x3FE6A09E667F3BCC
  %62 = fadd double %56, %7
  %63 = fmul double %46, %57
  %64 = fmul double %58, 0x3FF6A09E667F3BCD
  %65 = fmul double %59, 0x3FF6A09E667F3BCD
  %66 = fmul double %45, %60
  %67 = fmul double %46, %61
  %68 = fmul double %62, 5.000000e-01
  %69 = call double @__nv_sqrt(double %68)
  %70 = fcmp oeq double %69, 0.000000e+00
  %71 = fcmp oeq double %69, 0x7FF0000000000000
  %72 = fcmp ogt double %7, %8
  %73 = fdiv double %63, %64
  %74 = fdiv double %46, %65
  %75 = fmul double %69, 2.000000e+00
  %76 = select i1 %72, double %66, double %67
  %77 = or i1 %70, %71
  %78 = select i1 %72, double %73, double %74
  %79 = fdiv double %8, %75
  %80 = fmul double %45, 0x3FF19435CAFFA9F8
  %81 = select i1 %77, double %76, double %69
  %82 = fmul double %46, 0x3FDD203138F6C828
  %83 = select i1 %77, double %78, double %79
  %84 = select i1 %12, double %80, double %81
  %85 = fneg double %84
  %86 = fcmp olt double %6, 0.000000e+00
  %87 = select i1 %12, double %82, double %83
  %88 = fneg double %87
  %89 = fcmp oge double %5, 0.000000e+00
  %90 = fcmp olt double %5, 0.000000e+00
  %91 = select i1 %86, double %85, double %84
  %92 = select i1 %86, double %88, double %87
  %93 = select i1 %89, double %84, double %87
  %94 = select i1 %90, double %91, double %92
  %95 = insertvalue { double, double } poison, double %93, 0
  %96 = insertvalue { double, double } %95, double %94, 1
  %97 = getelementptr inbounds [1 x { double, double }], ptr %1, i32 0, i32 0
  store { double, double } %96, ptr %97, align 8
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #1

attributes #0 = { "nvvm.reqntid"="1,1,1" }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
