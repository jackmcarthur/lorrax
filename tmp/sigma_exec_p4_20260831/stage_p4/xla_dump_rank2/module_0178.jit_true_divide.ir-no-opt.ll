; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_fabs(double)

declare double @__nv_copysign(double, double)

define ptx_kernel void @loop_divide_fusion(ptr noalias align 16 dereferenceable(16) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(16) %2) #0 {
  %4 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %5 = load i64, ptr %4, align 4, !invariant.load !1
  %6 = sitofp i64 %5 to double
  %7 = getelementptr inbounds [1 x { double, double }], ptr %0, i32 0, i32 0
  %8 = load { double, double }, ptr %7, align 8, !invariant.load !1
  %9 = extractvalue { double, double } %8, 0
  %10 = extractvalue { double, double } %8, 1
  %11 = fdiv double %9, %10
  %12 = fmul double %11, %9
  %13 = fadd double %10, %12
  %14 = fmul double %6, %11
  %15 = fadd double %14, 0.000000e+00
  %16 = fdiv double %15, %13
  %17 = fmul double %11, 0.000000e+00
  %18 = fsub double %17, %6
  %19 = fdiv double %18, %13
  %20 = fdiv double %10, %9
  %21 = fmul double %20, %10
  %22 = fadd double %9, %21
  %23 = fmul double %20, 0.000000e+00
  %24 = fadd double %6, %23
  %25 = fdiv double %24, %22
  %26 = fmul double %6, %20
  %27 = fsub double 0.000000e+00, %26
  %28 = fdiv double %27, %22
  %29 = call double @__nv_fabs(double %9)
  %30 = fcmp oeq double %29, 0.000000e+00
  %31 = call double @__nv_fabs(double %10)
  %32 = fcmp oeq double %31, 0.000000e+00
  %33 = and i1 %30, %32
  %34 = call double @__nv_copysign(double 0x7FF0000000000000, double %9)
  %35 = fmul double %34, %6
  %36 = fmul double %34, 0.000000e+00
  %37 = fcmp one double %29, 0x7FF0000000000000
  %38 = fcmp one double %31, 0x7FF0000000000000
  %39 = and i1 %37, %38
  %40 = call double @__nv_fabs(double %6)
  %41 = fcmp oeq double %40, 0x7FF0000000000000
  %42 = and i1 %41, %39
  %43 = select i1 %41, double 1.000000e+00, double 0.000000e+00
  %44 = call double @__nv_copysign(double %43, double %6)
  %45 = fmul double %44, %9
  %46 = fmul double %10, 0.000000e+00
  %47 = fadd double %45, %46
  %48 = fmul double %47, 0x7FF0000000000000
  %49 = fmul double %44, %10
  %50 = fmul double %9, 0.000000e+00
  %51 = fsub double %50, %49
  %52 = fmul double %51, 0x7FF0000000000000
  %53 = fcmp one double %40, 0x7FF0000000000000
  %54 = fcmp oeq double %29, 0x7FF0000000000000
  %55 = fcmp oeq double %31, 0x7FF0000000000000
  %56 = or i1 %54, %55
  %57 = and i1 %53, %56
  %58 = select i1 %54, double 1.000000e+00, double 0.000000e+00
  %59 = call double @__nv_copysign(double %58, double %9)
  %60 = select i1 %55, double 1.000000e+00, double 0.000000e+00
  %61 = call double @__nv_copysign(double %60, double %10)
  %62 = fmul double %6, %59
  %63 = fmul double %61, 0.000000e+00
  %64 = fadd double %62, %63
  %65 = fmul double %64, 0.000000e+00
  %66 = fmul double %59, 0.000000e+00
  %67 = fmul double %6, %61
  %68 = fsub double %66, %67
  %69 = fmul double %68, 0.000000e+00
  %70 = fcmp olt double %29, %31
  %71 = select i1 %70, double %16, double %25
  %72 = select i1 %70, double %19, double %28
  %73 = select i1 %57, double %65, double %71
  %74 = select i1 %57, double %69, double %72
  %75 = select i1 %42, double %48, double %73
  %76 = select i1 %42, double %52, double %74
  %77 = select i1 %33, double %35, double %75
  %78 = select i1 %33, double %36, double %76
  %79 = fcmp uno double %71, 0.000000e+00
  %80 = fcmp uno double %72, 0.000000e+00
  %81 = and i1 %79, %80
  %82 = select i1 %81, double %77, double %71
  %83 = select i1 %81, double %78, double %72
  %84 = insertvalue { double, double } poison, double %82, 0
  %85 = insertvalue { double, double } %84, double %83, 1
  %86 = getelementptr inbounds [1 x { double, double }], ptr %2, i32 0, i32 0
  store { double, double } %85, ptr %86, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
