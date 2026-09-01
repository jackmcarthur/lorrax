; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_add_fusion(ptr noalias align 16 dereferenceable(24772608) %0, ptr noalias align 16 dereferenceable(1179648) %1, ptr noalias align 16 dereferenceable(336) %2, ptr noalias align 16 dereferenceable(24772608) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = udiv i32 %5, 144
  %8 = getelementptr inbounds [21 x { double, double }], ptr %2, i32 0, i32 %7
  %9 = load { double, double }, ptr %8, align 8, !invariant.load !3
  %10 = mul i32 %5, 128
  %11 = add i32 %10, %6
  %12 = urem i32 %11, 3
  %13 = mul i32 %12, 4
  %14 = udiv i32 %11, 3
  %15 = urem i32 %14, 12
  %16 = mul i32 %15, 12
  %17 = add i32 %13, %16
  %18 = udiv i32 %11, 36
  %19 = urem i32 %18, 512
  %20 = mul i32 %19, 144
  %21 = add i32 %17, %20
  %22 = getelementptr inbounds [73728 x { double, double }], ptr %1, i32 0, i32 %21
  %23 = load { double, double }, ptr %22, align 8, !invariant.load !3
  %24 = mul i32 %6, 4
  %25 = mul i32 %5, 512
  %26 = add i32 %24, %25
  %27 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %26
  %28 = load { double, double }, ptr %27, align 8
  %29 = extractvalue { double, double } %9, 0
  %30 = extractvalue { double, double } %9, 1
  %31 = extractvalue { double, double } %23, 0
  %32 = extractvalue { double, double } %23, 1
  %33 = fmul double %29, %31
  %34 = fmul double %30, %32
  %35 = fsub double %33, %34
  %36 = fmul double %30, %31
  %37 = fmul double %29, %32
  %38 = fadd double %36, %37
  %39 = extractvalue { double, double } %28, 0
  %40 = fadd double %39, %35
  %41 = extractvalue { double, double } %28, 1
  %42 = fadd double %41, %38
  %43 = insertvalue { double, double } poison, double %40, 0
  %44 = insertvalue { double, double } %43, double %42, 1
  store { double, double } %44, ptr %27, align 8
  %45 = add i32 %21, 1
  %46 = getelementptr inbounds [73728 x { double, double }], ptr %1, i32 0, i32 %45
  %47 = load { double, double }, ptr %46, align 8, !invariant.load !3
  %48 = add i32 %26, 1
  %49 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %48
  %50 = load { double, double }, ptr %49, align 8
  %51 = extractvalue { double, double } %47, 0
  %52 = extractvalue { double, double } %47, 1
  %53 = fmul double %29, %51
  %54 = fmul double %30, %52
  %55 = fsub double %53, %54
  %56 = fmul double %30, %51
  %57 = fmul double %29, %52
  %58 = fadd double %56, %57
  %59 = extractvalue { double, double } %50, 0
  %60 = fadd double %59, %55
  %61 = extractvalue { double, double } %50, 1
  %62 = fadd double %61, %58
  %63 = insertvalue { double, double } poison, double %60, 0
  %64 = insertvalue { double, double } %63, double %62, 1
  store { double, double } %64, ptr %49, align 8
  %65 = add i32 %21, 2
  %66 = getelementptr inbounds [73728 x { double, double }], ptr %1, i32 0, i32 %65
  %67 = load { double, double }, ptr %66, align 8, !invariant.load !3
  %68 = add i32 %26, 2
  %69 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %68
  %70 = load { double, double }, ptr %69, align 8
  %71 = extractvalue { double, double } %67, 0
  %72 = extractvalue { double, double } %67, 1
  %73 = fmul double %29, %71
  %74 = fmul double %30, %72
  %75 = fsub double %73, %74
  %76 = fmul double %30, %71
  %77 = fmul double %29, %72
  %78 = fadd double %76, %77
  %79 = extractvalue { double, double } %70, 0
  %80 = fadd double %79, %75
  %81 = extractvalue { double, double } %70, 1
  %82 = fadd double %81, %78
  %83 = insertvalue { double, double } poison, double %80, 0
  %84 = insertvalue { double, double } %83, double %82, 1
  store { double, double } %84, ptr %69, align 8
  %85 = add i32 %21, 3
  %86 = getelementptr inbounds [73728 x { double, double }], ptr %1, i32 0, i32 %85
  %87 = load { double, double }, ptr %86, align 8, !invariant.load !3
  %88 = add i32 %26, 3
  %89 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %88
  %90 = load { double, double }, ptr %89, align 8
  %91 = extractvalue { double, double } %87, 0
  %92 = extractvalue { double, double } %87, 1
  %93 = fmul double %29, %91
  %94 = fmul double %30, %92
  %95 = fsub double %93, %94
  %96 = fmul double %30, %91
  %97 = fmul double %29, %92
  %98 = fadd double %96, %97
  %99 = extractvalue { double, double } %90, 0
  %100 = fadd double %99, %95
  %101 = extractvalue { double, double } %90, 1
  %102 = fadd double %101, %98
  %103 = insertvalue { double, double } poison, double %100, 0
  %104 = insertvalue { double, double } %103, double %102, 1
  store { double, double } %104, ptr %89, align 8
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
!1 = !{i32 0, i32 3024}
!2 = !{i32 0, i32 128}
!3 = !{}
