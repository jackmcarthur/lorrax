; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_exp(double)

declare double @__nv_sin(double)

declare double @__nv_cos(double)

define ptx_kernel void @loop_select_fusion(ptr noalias align 16 dereferenceable(98304) %0, ptr noalias align 16 dereferenceable(98304) %1, ptr noalias align 16 dereferenceable(8) %2, ptr noalias align 16 dereferenceable(16) %3, ptr noalias align 256 dereferenceable(196608) %4) #0 {
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = mul i32 %6, 128
  %9 = add i32 %8, %7
  %10 = getelementptr inbounds [12288 x double], ptr %1, i32 0, i32 %9
  %11 = load double, ptr %10, align 8, !invariant.load !3
  %12 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  %13 = load double, ptr %12, align 8, !invariant.load !3
  %14 = fsub double %11, %13
  %15 = getelementptr inbounds [1 x { double, double }], ptr %3, i32 0, i32 0
  %16 = load { double, double }, ptr %15, align 8, !invariant.load !3
  %17 = extractvalue { double, double } %16, 0
  %18 = extractvalue { double, double } %16, 1
  %19 = fmul double %17, 0.000000e+00
  %20 = fsub double %19, %18
  %21 = fmul double %18, 0.000000e+00
  %22 = fadd double %21, %17
  %23 = fneg double %20
  %24 = fneg double %22
  %25 = fmul double %23, %14
  %26 = fmul double %24, 0.000000e+00
  %27 = fsub double %25, %26
  %28 = fmul double %24, %14
  %29 = fmul double %23, 0.000000e+00
  %30 = fadd double %28, %29
  %31 = fmul double %27, 5.000000e-01
  %32 = call double @__nv_exp(double %31)
  %33 = call double @__nv_sin(double %30)
  %34 = fmul double %32, %33
  %35 = call double @__nv_cos(double %30)
  %36 = fmul double %32, %35
  %37 = call double @__nv_exp(double %27)
  %38 = fmul double %34, %32
  %39 = fmul double %37, %33
  %40 = fcmp oeq double %37, 0x7FF0000000000000
  %41 = fmul double %36, %32
  %42 = fmul double %37, %35
  %43 = fcmp oeq double %30, 0.000000e+00
  %44 = select i1 %40, double %38, double %39
  %45 = select i1 %40, double %41, double %42
  %46 = select i1 %43, double 0.000000e+00, double %44
  %47 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %9
  %48 = load double, ptr %47, align 8, !invariant.load !3
  %49 = fcmp une double %48, 0.000000e+00
  %50 = fmul double %45, %48
  %51 = fmul double %46, 0.000000e+00
  %52 = fsub double %50, %51
  %53 = fmul double %46, %48
  %54 = fmul double %45, 0.000000e+00
  %55 = fadd double %53, %54
  %56 = insertvalue { double, double } poison, double %52, 0
  %57 = insertvalue { double, double } %56, double %55, 1
  %58 = select i1 %49, { double, double } %57, { double, double } zeroinitializer
  %59 = getelementptr inbounds [12288 x { double, double }], ptr %4, i32 0, i32 %9
  store { double, double } %58, ptr %59, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 dereferenceable(196608) %0, ptr noalias align 16 dereferenceable(121896960) %1, ptr noalias align 256 dereferenceable(121896960) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = udiv i32 %7, 155
  %9 = getelementptr inbounds [12288 x { double, double }], ptr %0, i32 0, i32 %8
  %10 = load { double, double }, ptr %9, align 8, !invariant.load !3
  %11 = mul i32 %5, 4
  %12 = mul i32 %4, 512
  %13 = add i32 %11, %12
  %14 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %13
  %15 = load { double, double }, ptr %14, align 8, !invariant.load !3
  %16 = extractvalue { double, double } %15, 1
  %17 = extractvalue { double, double } %15, 0
  %18 = fneg double %16
  %19 = extractvalue { double, double } %10, 0
  %20 = extractvalue { double, double } %10, 1
  %21 = fmul double %19, %17
  %22 = fmul double %20, %18
  %23 = fsub double %21, %22
  %24 = fmul double %20, %17
  %25 = fmul double %19, %18
  %26 = fadd double %24, %25
  %27 = insertvalue { double, double } poison, double %23, 0
  %28 = insertvalue { double, double } %27, double %26, 1
  %29 = getelementptr inbounds [7618560 x { double, double }], ptr %2, i32 0, i32 %13
  store { double, double } %28, ptr %29, align 8
  %30 = add i32 %13, 1
  %31 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %30
  %32 = load { double, double }, ptr %31, align 8, !invariant.load !3
  %33 = extractvalue { double, double } %32, 1
  %34 = extractvalue { double, double } %32, 0
  %35 = fneg double %33
  %36 = fmul double %19, %34
  %37 = fmul double %20, %35
  %38 = fsub double %36, %37
  %39 = fmul double %20, %34
  %40 = fmul double %19, %35
  %41 = fadd double %39, %40
  %42 = insertvalue { double, double } poison, double %38, 0
  %43 = insertvalue { double, double } %42, double %41, 1
  %44 = getelementptr inbounds [7618560 x { double, double }], ptr %2, i32 0, i32 %30
  store { double, double } %43, ptr %44, align 8
  %45 = add i32 %13, 2
  %46 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %45
  %47 = load { double, double }, ptr %46, align 8, !invariant.load !3
  %48 = extractvalue { double, double } %47, 1
  %49 = extractvalue { double, double } %47, 0
  %50 = fneg double %48
  %51 = fmul double %19, %49
  %52 = fmul double %20, %50
  %53 = fsub double %51, %52
  %54 = fmul double %20, %49
  %55 = fmul double %19, %50
  %56 = fadd double %54, %55
  %57 = insertvalue { double, double } poison, double %53, 0
  %58 = insertvalue { double, double } %57, double %56, 1
  %59 = getelementptr inbounds [7618560 x { double, double }], ptr %2, i32 0, i32 %45
  store { double, double } %58, ptr %59, align 8
  %60 = add i32 %13, 3
  %61 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %60
  %62 = load { double, double }, ptr %61, align 8, !invariant.load !3
  %63 = extractvalue { double, double } %62, 1
  %64 = extractvalue { double, double } %62, 0
  %65 = fneg double %63
  %66 = fmul double %19, %64
  %67 = fmul double %20, %65
  %68 = fsub double %66, %67
  %69 = fmul double %20, %64
  %70 = fmul double %19, %65
  %71 = fadd double %69, %70
  %72 = insertvalue { double, double } poison, double %68, 0
  %73 = insertvalue { double, double } %72, double %71, 1
  %74 = getelementptr inbounds [7618560 x { double, double }], ptr %2, i32 0, i32 %60
  store { double, double } %73, ptr %74, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 96}
!2 = !{i32 0, i32 128}
!3 = !{}
!4 = !{i32 0, i32 14880}
