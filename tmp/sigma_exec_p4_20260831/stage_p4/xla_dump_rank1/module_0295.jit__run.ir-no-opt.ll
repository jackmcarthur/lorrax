; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_broadcast_fusion(ptr noalias align 256 dereferenceable(66816) %0) #0 {
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %4 = mul i32 %2, 128
  %5 = add i32 %4, %3
  %6 = icmp sle i32 %5, 4175
  br i1 %6, label %7, label %9

7:                                                ; preds = %1
  %8 = getelementptr inbounds [4176 x { double, double }], ptr %0, i32 0, i32 %5
  store { double, double } zeroinitializer, ptr %8, align 8
  br label %9

9:                                                ; preds = %7, %1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

define ptx_kernel void @loop_compare_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(1) %1) #2 {
  %3 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %4 = load i64, ptr %3, align 4, !invariant.load !3
  %5 = icmp slt i64 %4, 29
  %6 = zext i1 %5 to i8
  %7 = getelementptr inbounds [1 x i8], ptr %1, i32 0, i32 0
  store i8 %6, ptr %7, align 1
  ret void
}

define ptx_kernel void @loop_gather_fusion(ptr noalias align 256 dereferenceable(294144) %0, ptr noalias align 16 dereferenceable(3801088) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(6291456) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %6 = sext i32 %5 to i64
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = sext i32 %7 to i64
  %9 = getelementptr inbounds [1 x i64], ptr %2, i32 0, i32 0
  %10 = load i64, ptr %9, align 4, !invariant.load !3
  %11 = call i64 @llvm.smin.i64(i64 %10, i64 28)
  %12 = call i64 @llvm.smax.i64(i64 %11, i64 0)
  %13 = urem i64 %6, 64
  %14 = mul i64 %13, 512
  %15 = mul i64 %12, 32768
  %16 = add i64 %14, %15
  %17 = mul i64 %8, 4
  %18 = add i64 %16, %17
  %19 = getelementptr inbounds [950272 x i32], ptr %1, i32 0, i64 %18
  %20 = load <4 x i32>, ptr %19, align 4, !invariant.load !3
  %21 = extractelement <4 x i32> %20, i64 0
  %22 = sext i32 %21 to i64
  %23 = call i64 @llvm.smin.i64(i64 %22, i64 1532)
  %24 = call i64 @llvm.smax.i64(i64 %23, i64 0)
  %25 = icmp sle i64 %24, 1531
  br i1 %25, label %26, label %32

26:                                               ; preds = %4
  %27 = udiv i64 %6, 64
  %28 = mul i64 %27, 1532
  %29 = add i64 %28, %24
  %30 = getelementptr inbounds [18384 x { double, double }], ptr %0, i32 0, i64 %29
  %31 = load { double, double }, ptr %30, align 8, !invariant.load !3
  br label %33

32:                                               ; preds = %4
  br label %33

33:                                               ; preds = %26, %32
  %34 = phi { double, double } [ zeroinitializer, %32 ], [ %31, %26 ]
  br label %35

35:                                               ; preds = %33
  %36 = mul i64 %6, 512
  %37 = add i64 %17, %36
  %38 = getelementptr inbounds [393216 x { double, double }], ptr %3, i32 0, i64 %37
  store { double, double } %34, ptr %38, align 8
  %39 = extractelement <4 x i32> %20, i64 1
  %40 = sext i32 %39 to i64
  %41 = call i64 @llvm.smin.i64(i64 %40, i64 1532)
  %42 = call i64 @llvm.smax.i64(i64 %41, i64 0)
  %43 = icmp sle i64 %42, 1531
  br i1 %43, label %44, label %50

44:                                               ; preds = %35
  %45 = udiv i64 %6, 64
  %46 = mul i64 %45, 1532
  %47 = add i64 %46, %42
  %48 = getelementptr inbounds [18384 x { double, double }], ptr %0, i32 0, i64 %47
  %49 = load { double, double }, ptr %48, align 8, !invariant.load !3
  br label %51

50:                                               ; preds = %35
  br label %51

51:                                               ; preds = %44, %50
  %52 = phi { double, double } [ zeroinitializer, %50 ], [ %49, %44 ]
  br label %53

53:                                               ; preds = %51
  %54 = add i64 %37, 1
  %55 = getelementptr inbounds [393216 x { double, double }], ptr %3, i32 0, i64 %54
  store { double, double } %52, ptr %55, align 8
  %56 = extractelement <4 x i32> %20, i64 2
  %57 = sext i32 %56 to i64
  %58 = call i64 @llvm.smin.i64(i64 %57, i64 1532)
  %59 = call i64 @llvm.smax.i64(i64 %58, i64 0)
  %60 = icmp sle i64 %59, 1531
  br i1 %60, label %61, label %67

61:                                               ; preds = %53
  %62 = udiv i64 %6, 64
  %63 = mul i64 %62, 1532
  %64 = add i64 %63, %59
  %65 = getelementptr inbounds [18384 x { double, double }], ptr %0, i32 0, i64 %64
  %66 = load { double, double }, ptr %65, align 8, !invariant.load !3
  br label %68

67:                                               ; preds = %53
  br label %68

68:                                               ; preds = %61, %67
  %69 = phi { double, double } [ zeroinitializer, %67 ], [ %66, %61 ]
  br label %70

70:                                               ; preds = %68
  %71 = add i64 %37, 2
  %72 = getelementptr inbounds [393216 x { double, double }], ptr %3, i32 0, i64 %71
  store { double, double } %69, ptr %72, align 8
  %73 = extractelement <4 x i32> %20, i64 3
  %74 = sext i32 %73 to i64
  %75 = call i64 @llvm.smin.i64(i64 %74, i64 1532)
  %76 = call i64 @llvm.smax.i64(i64 %75, i64 0)
  %77 = icmp sle i64 %76, 1531
  br i1 %77, label %78, label %84

78:                                               ; preds = %70
  %79 = udiv i64 %6, 64
  %80 = mul i64 %79, 1532
  %81 = add i64 %80, %76
  %82 = getelementptr inbounds [18384 x { double, double }], ptr %0, i32 0, i64 %81
  %83 = load { double, double }, ptr %82, align 8, !invariant.load !3
  br label %85

84:                                               ; preds = %70
  br label %85

85:                                               ; preds = %78, %84
  %86 = phi { double, double } [ zeroinitializer, %84 ], [ %83, %78 ]
  br label %87

87:                                               ; preds = %85
  %88 = add i64 %37, 3
  %89 = getelementptr inbounds [393216 x { double, double }], ptr %3, i32 0, i64 %88
  store { double, double } %86, ptr %89, align 8
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #3

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 dereferenceable(6291456) %0, ptr noalias align 16 dereferenceable(524288) %1, ptr noalias align 256 dereferenceable(6291456) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %5, 4
  %7 = mul i32 %4, 512
  %8 = add i32 %6, %7
  %9 = getelementptr inbounds [393216 x { double, double }], ptr %0, i32 0, i32 %8
  %10 = load { double, double }, ptr %9, align 8
  %11 = udiv i32 %4, 2
  %12 = urem i32 %11, 32
  %13 = mul i32 %12, 1024
  %14 = urem i32 %4, 2
  %15 = mul i32 %14, 512
  %16 = add i32 %13, %15
  %17 = add i32 %16, %6
  %18 = getelementptr inbounds [32768 x { double, double }], ptr %1, i32 0, i32 %17
  %19 = load { double, double }, ptr %18, align 8, !invariant.load !3
  %20 = extractvalue { double, double } %19, 0
  %21 = extractvalue { double, double } %19, 1
  %22 = fmul double %20, 0x40A00C3EA4553987
  %23 = fmul double %21, 0.000000e+00
  %24 = fsub double %22, %23
  %25 = fmul double %21, 0x40A00C3EA4553987
  %26 = fmul double %20, 0.000000e+00
  %27 = fadd double %25, %26
  %28 = extractvalue { double, double } %10, 0
  %29 = extractvalue { double, double } %10, 1
  %30 = fmul double %28, %24
  %31 = fmul double %29, %27
  %32 = fsub double %30, %31
  %33 = fmul double %29, %24
  %34 = fmul double %28, %27
  %35 = fadd double %33, %34
  %36 = insertvalue { double, double } poison, double %32, 0
  %37 = insertvalue { double, double } %36, double %35, 1
  store { double, double } %37, ptr %9, align 8
  %38 = add i32 %8, 1
  %39 = getelementptr inbounds [393216 x { double, double }], ptr %0, i32 0, i32 %38
  %40 = load { double, double }, ptr %39, align 8
  %41 = add i32 %17, 1
  %42 = getelementptr inbounds [32768 x { double, double }], ptr %1, i32 0, i32 %41
  %43 = load { double, double }, ptr %42, align 8, !invariant.load !3
  %44 = extractvalue { double, double } %43, 0
  %45 = extractvalue { double, double } %43, 1
  %46 = fmul double %44, 0x40A00C3EA4553987
  %47 = fmul double %45, 0.000000e+00
  %48 = fsub double %46, %47
  %49 = fmul double %45, 0x40A00C3EA4553987
  %50 = fmul double %44, 0.000000e+00
  %51 = fadd double %49, %50
  %52 = extractvalue { double, double } %40, 0
  %53 = extractvalue { double, double } %40, 1
  %54 = fmul double %52, %48
  %55 = fmul double %53, %51
  %56 = fsub double %54, %55
  %57 = fmul double %53, %48
  %58 = fmul double %52, %51
  %59 = fadd double %57, %58
  %60 = insertvalue { double, double } poison, double %56, 0
  %61 = insertvalue { double, double } %60, double %59, 1
  store { double, double } %61, ptr %39, align 8
  %62 = add i32 %8, 2
  %63 = getelementptr inbounds [393216 x { double, double }], ptr %0, i32 0, i32 %62
  %64 = load { double, double }, ptr %63, align 8
  %65 = add i32 %17, 2
  %66 = getelementptr inbounds [32768 x { double, double }], ptr %1, i32 0, i32 %65
  %67 = load { double, double }, ptr %66, align 8, !invariant.load !3
  %68 = extractvalue { double, double } %67, 0
  %69 = extractvalue { double, double } %67, 1
  %70 = fmul double %68, 0x40A00C3EA4553987
  %71 = fmul double %69, 0.000000e+00
  %72 = fsub double %70, %71
  %73 = fmul double %69, 0x40A00C3EA4553987
  %74 = fmul double %68, 0.000000e+00
  %75 = fadd double %73, %74
  %76 = extractvalue { double, double } %64, 0
  %77 = extractvalue { double, double } %64, 1
  %78 = fmul double %76, %72
  %79 = fmul double %77, %75
  %80 = fsub double %78, %79
  %81 = fmul double %77, %72
  %82 = fmul double %76, %75
  %83 = fadd double %81, %82
  %84 = insertvalue { double, double } poison, double %80, 0
  %85 = insertvalue { double, double } %84, double %83, 1
  store { double, double } %85, ptr %63, align 8
  %86 = add i32 %8, 3
  %87 = getelementptr inbounds [393216 x { double, double }], ptr %0, i32 0, i32 %86
  %88 = load { double, double }, ptr %87, align 8
  %89 = add i32 %17, 3
  %90 = getelementptr inbounds [32768 x { double, double }], ptr %1, i32 0, i32 %89
  %91 = load { double, double }, ptr %90, align 8, !invariant.load !3
  %92 = extractvalue { double, double } %91, 0
  %93 = extractvalue { double, double } %91, 1
  %94 = fmul double %92, 0x40A00C3EA4553987
  %95 = fmul double %93, 0.000000e+00
  %96 = fsub double %94, %95
  %97 = fmul double %93, 0x40A00C3EA4553987
  %98 = fmul double %92, 0.000000e+00
  %99 = fadd double %97, %98
  %100 = extractvalue { double, double } %88, 0
  %101 = extractvalue { double, double } %88, 1
  %102 = fmul double %100, %96
  %103 = fmul double %101, %99
  %104 = fsub double %102, %103
  %105 = fmul double %101, %96
  %106 = fmul double %100, %99
  %107 = fadd double %105, %106
  %108 = insertvalue { double, double } poison, double %104, 0
  %109 = insertvalue { double, double } %108, double %107, 1
  store { double, double } %109, ptr %87, align 8
  ret void
}

define ptx_kernel void @loop_gather_fusion_1(ptr noalias align 256 dereferenceable(6291456) %0, ptr noalias align 16 dereferenceable(533136) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(294144) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %6 = sext i32 %5 to i64
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = sext i32 %7 to i64
  %9 = mul i64 %6, 128
  %10 = add i64 %9, %8
  %11 = icmp sle i64 %10, 18383
  br i1 %11, label %12, label %55

12:                                               ; preds = %4
  %13 = urem i64 %10, 1532
  %14 = call i32 @fused_gather_1_bitcast_7_9(ptr %0, ptr %1, ptr %2, i64 %13, i64 0)
  %15 = icmp slt i32 %14, 0
  %16 = add i32 %14, 32
  %17 = select i1 %15, i32 %16, i32 %14
  %18 = sext i32 %17 to i64
  %19 = call i64 @llvm.smin.i64(i64 %18, i64 31)
  %20 = call i64 @llvm.smax.i64(i64 %19, i64 0)
  %21 = call i32 @fused_gather_1_bitcast_7_9(ptr %0, ptr %1, ptr %2, i64 %13, i64 1)
  %22 = icmp slt i32 %21, 0
  %23 = add i32 %21, 32
  %24 = select i1 %22, i32 %23, i32 %21
  %25 = sext i32 %24 to i64
  %26 = call i64 @llvm.smin.i64(i64 %25, i64 31)
  %27 = call i64 @llvm.smax.i64(i64 %26, i64 0)
  %28 = call i32 @fused_gather_1_bitcast_7_9(ptr %0, ptr %1, ptr %2, i64 %13, i64 2)
  %29 = icmp slt i32 %28, 0
  %30 = add i32 %28, 32
  %31 = select i1 %29, i32 %30, i32 %28
  %32 = sext i32 %31 to i64
  %33 = call i64 @llvm.smin.i64(i64 %32, i64 31)
  %34 = call i64 @llvm.smax.i64(i64 %33, i64 0)
  %35 = udiv i64 %10, 1532
  %36 = mul i64 %35, 32768
  %37 = mul i64 %20, 1024
  %38 = add i64 %36, %37
  %39 = mul i64 %27, 32
  %40 = add i64 %38, %39
  %41 = add i64 %40, %34
  %42 = getelementptr inbounds [393216 x { double, double }], ptr %0, i32 0, i64 %41
  %43 = load { double, double }, ptr %42, align 8, !invariant.load !3
  %44 = extractvalue { double, double } %43, 0
  %45 = extractvalue { double, double } %43, 1
  %46 = fmul double %44, 0x3F7FCF3D6F094292
  %47 = fmul double %45, 0.000000e+00
  %48 = fsub double %46, %47
  %49 = fmul double %45, 0x3F7FCF3D6F094292
  %50 = fmul double %44, 0.000000e+00
  %51 = fadd double %49, %50
  %52 = insertvalue { double, double } poison, double %48, 0
  %53 = insertvalue { double, double } %52, double %51, 1
  %54 = getelementptr inbounds [18384 x { double, double }], ptr %3, i32 0, i64 %10
  store { double, double } %53, ptr %54, align 8
  br label %55

55:                                               ; preds = %12, %4
  ret void
}

define internal i32 @fused_gather_1_bitcast_7_9(ptr noalias %0, ptr noalias %1, ptr noalias %2, i64 %3, i64 %4) {
  %6 = getelementptr inbounds [1 x i64], ptr %2, i32 0, i32 0
  %7 = load i64, ptr %6, align 4, !invariant.load !3
  %8 = call i64 @llvm.smin.i64(i64 %7, i64 28)
  %9 = call i64 @llvm.smax.i64(i64 %8, i64 0)
  %10 = mul i64 %9, 4596
  %11 = mul i64 %3, 3
  %12 = add i64 %10, %11
  %13 = add i64 %12, %4
  %14 = getelementptr inbounds [133284 x i32], ptr %1, i32 0, i64 %13
  %15 = load i32, ptr %14, align 4, !invariant.load !3
  ret i32 %15
}

define ptx_kernel void @loop_complex_multiply_fusion(ptr noalias align 256 dereferenceable(588288) %0, ptr noalias align 16 dereferenceable(355424) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(588288) %3, ptr noalias align 256 dereferenceable(588288) %4, ptr noalias align 256 dereferenceable(588288) %5) #0 {
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %8 = sext i32 %7 to i64
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %10 = sext i32 %9 to i64
  %11 = mul i64 %8, 128
  %12 = add i64 %11, %10
  %13 = icmp sle i64 %12, 36767
  br i1 %13, label %14, label %51

14:                                               ; preds = %6
  %15 = getelementptr inbounds [36768 x { double, double }], ptr %0, i32 0, i64 %12
  %16 = load { double, double }, ptr %15, align 8, !invariant.load !3
  %17 = getelementptr inbounds [1 x i64], ptr %2, i32 0, i32 0
  %18 = load i64, ptr %17, align 4, !invariant.load !3
  %19 = call i64 @llvm.smin.i64(i64 %18, i64 28)
  %20 = call i64 @llvm.smax.i64(i64 %19, i64 0)
  %21 = urem i64 %12, 1532
  %22 = mul i64 %20, 1532
  %23 = add i64 %21, %22
  %24 = getelementptr inbounds [44428 x double], ptr %1, i32 0, i64 %23
  %25 = load double, ptr %24, align 8, !invariant.load !3
  %26 = extractvalue { double, double } %16, 0
  %27 = extractvalue { double, double } %16, 1
  %28 = fmul double %26, %25
  %29 = fmul double %27, 0.000000e+00
  %30 = fsub double %28, %29
  %31 = fmul double %27, %25
  %32 = fmul double %26, 0.000000e+00
  %33 = fadd double %31, %32
  %34 = insertvalue { double, double } poison, double %30, 0
  %35 = insertvalue { double, double } %34, double %33, 1
  %36 = getelementptr inbounds [36768 x { double, double }], ptr %3, i32 0, i64 %12
  %37 = load { double, double }, ptr %36, align 8, !invariant.load !3
  %38 = extractvalue { double, double } %37, 0
  %39 = extractvalue { double, double } %37, 1
  %40 = fmul double %38, %25
  %41 = fmul double %39, 0.000000e+00
  %42 = fsub double %40, %41
  %43 = fmul double %39, %25
  %44 = fmul double %38, 0.000000e+00
  %45 = fadd double %43, %44
  %46 = fneg double %45
  %47 = insertvalue { double, double } poison, double %42, 0
  %48 = insertvalue { double, double } %47, double %46, 1
  %49 = getelementptr inbounds [36768 x { double, double }], ptr %4, i32 0, i64 %12
  store { double, double } %35, ptr %49, align 8
  %50 = getelementptr inbounds [36768 x { double, double }], ptr %5, i32 0, i64 %12
  store { double, double } %48, ptr %50, align 8
  br label %51

51:                                               ; preds = %14, %6
  ret void
}

define ptx_kernel void @loop_add_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1) #2 {
  %3 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %4 = load i64, ptr %3, align 4
  %5 = add i64 %4, 1
  store i64 %5, ptr %3, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { "nvvm.reqntid"="1,1,1" }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 33}
!2 = !{i32 0, i32 128}
!3 = !{}
!4 = !{i32 0, i32 768}
!5 = !{i32 0, i32 144}
!6 = !{i32 0, i32 288}
