; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_select_fusion(ptr noalias align 16 dereferenceable(8530176) %0, ptr noalias align 16 dereferenceable(116) %1, ptr noalias align 256 dereferenceable(8530176) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = icmp sle i32 %7, 133283
  br i1 %8, label %9, label %100

9:                                                ; preds = %3
  %10 = udiv i32 %7, 4596
  %11 = udiv i32 %7, 766
  %12 = urem i32 %11, 6
  %13 = udiv i32 %7, 383
  %14 = urem i32 %13, 2
  %15 = getelementptr inbounds [29 x i32], ptr %1, i32 0, i32 %10
  %16 = load i32, ptr %15, align 4, !invariant.load !3
  %17 = icmp slt i32 %16, 0
  %18 = add i32 %16, 29
  %19 = select i1 %17, i32 %18, i32 %16
  %20 = icmp sge i32 %19, 0
  %21 = icmp sle i32 %19, 28
  %22 = and i1 %20, %21
  %23 = call i32 @llvm.smin.i32(i32 %19, i32 28)
  %24 = call i32 @llvm.smax.i32(i32 %23, i32 0)
  %25 = urem i32 %7, 383
  %26 = mul i32 %25, 4
  %27 = call { double, double } @fused_select_convert_element_type_10_6(ptr %0, ptr %1, i32 %12, i32 %14, i32 %24, i32 %26, i32 1)
  %28 = call { double, double } @fused_select_convert_element_type_10_6(ptr %0, ptr %1, i32 %12, i32 %14, i32 %24, i32 %26, i32 0)
  %29 = extractvalue { double, double } %27, 0
  %30 = extractvalue { double, double } %27, 1
  %31 = fmul double %29, 0.000000e+00
  %32 = fsub double %31, %30
  %33 = fmul double %30, 0.000000e+00
  %34 = fadd double %33, %29
  %35 = extractvalue { double, double } %28, 0
  %36 = fadd double %35, %32
  %37 = extractvalue { double, double } %28, 1
  %38 = fadd double %37, %34
  %39 = insertvalue { double, double } poison, double %36, 0
  %40 = insertvalue { double, double } %39, double %38, 1
  %41 = select i1 %22, { double, double } %40, { double, double } { double 0x7FF8000000000000, double 0.000000e+00 }
  %42 = mul i32 %5, 4
  %43 = mul i32 %4, 512
  %44 = add i32 %42, %43
  %45 = getelementptr inbounds [533136 x { double, double }], ptr %2, i32 0, i32 %44
  store { double, double } %41, ptr %45, align 8
  %46 = add i32 %26, 1
  %47 = call { double, double } @fused_select_convert_element_type_10_6(ptr %0, ptr %1, i32 %12, i32 %14, i32 %24, i32 %46, i32 1)
  %48 = call { double, double } @fused_select_convert_element_type_10_6(ptr %0, ptr %1, i32 %12, i32 %14, i32 %24, i32 %46, i32 0)
  %49 = extractvalue { double, double } %47, 0
  %50 = extractvalue { double, double } %47, 1
  %51 = fmul double %49, 0.000000e+00
  %52 = fsub double %51, %50
  %53 = fmul double %50, 0.000000e+00
  %54 = fadd double %53, %49
  %55 = extractvalue { double, double } %48, 0
  %56 = fadd double %55, %52
  %57 = extractvalue { double, double } %48, 1
  %58 = fadd double %57, %54
  %59 = insertvalue { double, double } poison, double %56, 0
  %60 = insertvalue { double, double } %59, double %58, 1
  %61 = select i1 %22, { double, double } %60, { double, double } { double 0x7FF8000000000000, double 0.000000e+00 }
  %62 = add i32 %44, 1
  %63 = getelementptr inbounds [533136 x { double, double }], ptr %2, i32 0, i32 %62
  store { double, double } %61, ptr %63, align 8
  %64 = add i32 %26, 2
  %65 = call { double, double } @fused_select_convert_element_type_10_6(ptr %0, ptr %1, i32 %12, i32 %14, i32 %24, i32 %64, i32 1)
  %66 = call { double, double } @fused_select_convert_element_type_10_6(ptr %0, ptr %1, i32 %12, i32 %14, i32 %24, i32 %64, i32 0)
  %67 = extractvalue { double, double } %65, 0
  %68 = extractvalue { double, double } %65, 1
  %69 = fmul double %67, 0.000000e+00
  %70 = fsub double %69, %68
  %71 = fmul double %68, 0.000000e+00
  %72 = fadd double %71, %67
  %73 = extractvalue { double, double } %66, 0
  %74 = fadd double %73, %70
  %75 = extractvalue { double, double } %66, 1
  %76 = fadd double %75, %72
  %77 = insertvalue { double, double } poison, double %74, 0
  %78 = insertvalue { double, double } %77, double %76, 1
  %79 = select i1 %22, { double, double } %78, { double, double } { double 0x7FF8000000000000, double 0.000000e+00 }
  %80 = add i32 %44, 2
  %81 = getelementptr inbounds [533136 x { double, double }], ptr %2, i32 0, i32 %80
  store { double, double } %79, ptr %81, align 8
  %82 = add i32 %26, 3
  %83 = call { double, double } @fused_select_convert_element_type_10_6(ptr %0, ptr %1, i32 %12, i32 %14, i32 %24, i32 %82, i32 1)
  %84 = call { double, double } @fused_select_convert_element_type_10_6(ptr %0, ptr %1, i32 %12, i32 %14, i32 %24, i32 %82, i32 0)
  %85 = extractvalue { double, double } %83, 0
  %86 = extractvalue { double, double } %83, 1
  %87 = fmul double %85, 0.000000e+00
  %88 = fsub double %87, %86
  %89 = fmul double %86, 0.000000e+00
  %90 = fadd double %89, %85
  %91 = extractvalue { double, double } %84, 0
  %92 = fadd double %91, %88
  %93 = extractvalue { double, double } %84, 1
  %94 = fadd double %93, %90
  %95 = insertvalue { double, double } poison, double %92, 0
  %96 = insertvalue { double, double } %95, double %94, 1
  %97 = select i1 %22, { double, double } %96, { double, double } { double 0x7FF8000000000000, double 0.000000e+00 }
  %98 = add i32 %44, 3
  %99 = getelementptr inbounds [533136 x { double, double }], ptr %2, i32 0, i32 %98
  store { double, double } %97, ptr %99, align 8
  br label %100

100:                                              ; preds = %9, %3
  ret void
}

define internal { double, double } @fused_select_convert_element_type_10_6(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6) {
  %8 = mul i32 %2, 177712
  %9 = mul i32 %3, 88856
  %10 = add i32 %8, %9
  %11 = mul i32 %4, 3064
  %12 = add i32 %10, %11
  %13 = mul i32 %5, 2
  %14 = add i32 %12, %13
  %15 = add i32 %14, %6
  %16 = getelementptr inbounds [1066272 x double], ptr %0, i32 0, i32 %15
  %17 = load double, ptr %16, align 8, !invariant.load !3
  %18 = insertvalue { double, double } poison, double %17, 0
  %19 = insertvalue { double, double } %18, double 0.000000e+00, 1
  ret { double, double } %19
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
!1 = !{i32 0, i32 1042}
!2 = !{i32 0, i32 128}
!3 = !{}
