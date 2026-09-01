; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_fabs(double)

declare double @__nv_copysign(double, double)

define ptx_kernel void @loop_divide_fusion(ptr noalias align 16 dereferenceable(524288) %0, ptr noalias align 16 dereferenceable(262144) %1, ptr noalias align 256 dereferenceable(524288) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = getelementptr inbounds [32768 x double], ptr %1, i32 0, i32 %7
  %9 = load double, ptr %8, align 8, !invariant.load !3
  %10 = getelementptr inbounds [32768 x { double, double }], ptr %0, i32 0, i32 %7
  %11 = load { double, double }, ptr %10, align 8, !invariant.load !3
  %12 = extractvalue { double, double } %11, 0
  %13 = extractvalue { double, double } %11, 1
  %14 = fdiv double %9, 0.000000e+00
  %15 = fmul double %14, %9
  %16 = fadd double %15, 0.000000e+00
  %17 = fmul double %12, %14
  %18 = fadd double %17, %13
  %19 = fdiv double %18, %16
  %20 = fmul double %13, %14
  %21 = fsub double %20, %12
  %22 = fdiv double %21, %16
  %23 = fdiv double 0.000000e+00, %9
  %24 = fmul double %23, 0.000000e+00
  %25 = fadd double %9, %24
  %26 = fmul double %13, %23
  %27 = fadd double %12, %26
  %28 = fdiv double %27, %25
  %29 = fmul double %12, %23
  %30 = fsub double %13, %29
  %31 = fdiv double %30, %25
  %32 = call double @__nv_fabs(double %9)
  %33 = fcmp oeq double %32, 0.000000e+00
  %34 = fcmp ord double %12, 0.000000e+00
  %35 = fcmp ord double %13, 0.000000e+00
  %36 = or i1 %34, %35
  %37 = and i1 %36, %33
  %38 = call double @__nv_copysign(double 0x7FF0000000000000, double %9)
  %39 = fmul double %38, %12
  %40 = fmul double %38, %13
  %41 = fcmp one double %32, 0x7FF0000000000000
  %42 = call double @__nv_fabs(double %12)
  %43 = fcmp oeq double %42, 0x7FF0000000000000
  %44 = call double @__nv_fabs(double %13)
  %45 = fcmp oeq double %44, 0x7FF0000000000000
  %46 = or i1 %43, %45
  %47 = and i1 %46, %41
  %48 = select i1 %43, double 1.000000e+00, double 0.000000e+00
  %49 = call double @__nv_copysign(double %48, double %12)
  %50 = select i1 %45, double 1.000000e+00, double 0.000000e+00
  %51 = call double @__nv_copysign(double %50, double %13)
  %52 = fmul double %49, %9
  %53 = fmul double %51, 0.000000e+00
  %54 = fadd double %52, %53
  %55 = fmul double %54, 0x7FF0000000000000
  %56 = fmul double %49, 0.000000e+00
  %57 = fmul double %51, %9
  %58 = fsub double %57, %56
  %59 = fmul double %58, 0x7FF0000000000000
  %60 = fcmp one double %42, 0x7FF0000000000000
  %61 = fcmp one double %44, 0x7FF0000000000000
  %62 = and i1 %60, %61
  %63 = fcmp oeq double %32, 0x7FF0000000000000
  %64 = and i1 %62, %63
  %65 = select i1 %63, double 1.000000e+00, double 0.000000e+00
  %66 = call double @__nv_copysign(double %65, double %9)
  %67 = fmul double %12, %66
  %68 = fmul double %13, 0.000000e+00
  %69 = fadd double %67, %68
  %70 = fmul double %69, 0.000000e+00
  %71 = fmul double %13, %66
  %72 = fmul double %12, 0.000000e+00
  %73 = fsub double %71, %72
  %74 = fmul double %73, 0.000000e+00
  %75 = fcmp olt double %32, 0.000000e+00
  %76 = select i1 %75, double %19, double %28
  %77 = select i1 %75, double %22, double %31
  %78 = select i1 %64, double %70, double %76
  %79 = select i1 %64, double %74, double %77
  %80 = select i1 %47, double %55, double %78
  %81 = select i1 %47, double %59, double %79
  %82 = select i1 %37, double %39, double %80
  %83 = select i1 %37, double %40, double %81
  %84 = fcmp uno double %76, 0.000000e+00
  %85 = fcmp uno double %77, 0.000000e+00
  %86 = and i1 %84, %85
  %87 = select i1 %86, double %82, double %76
  %88 = select i1 %86, double %83, double %77
  %89 = insertvalue { double, double } poison, double %87, 0
  %90 = insertvalue { double, double } %89, double %88, 1
  %91 = getelementptr inbounds [32768 x { double, double }], ptr %2, i32 0, i32 %7
  store { double, double } %90, ptr %91, align 8
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
!1 = !{i32 0, i32 256}
!2 = !{i32 0, i32 128}
!3 = !{}
