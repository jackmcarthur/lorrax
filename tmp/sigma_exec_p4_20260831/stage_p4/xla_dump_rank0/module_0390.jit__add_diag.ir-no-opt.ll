; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_1 = private addrspace(3) global [1056 x double] undef
@shared_0 = private addrspace(3) global [1056 x double] undef
@shared_11 = private addrspace(3) global [1056 x double] undef
@shared_02 = private addrspace(3) global [1056 x double] undef
@shared_03 = private addrspace(3) global [1056 x double] undef
@shared_04 = private addrspace(3) global [1056 x double] undef

define ptx_kernel void @loop_and_fusion(ptr noalias align 256 dereferenceable(4) %0, ptr noalias align 256 dereferenceable(24) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = sext i32 %3 to i64
  %5 = getelementptr inbounds [1 x i32], ptr %0, i32 0, i32 0
  %6 = load i32, ptr %5, align 4, !invariant.load !2
  %7 = lshr i32 %6, 1
  %8 = and i32 %7, 1
  %9 = mul i32 %8, 12
  %10 = sext i32 %9 to i64
  %11 = sub i64 %4, %10
  %12 = icmp sge i64 %11, 0
  %13 = icmp slt i64 %11, 12
  %14 = and i32 %6, 1
  %15 = mul i32 %14, 12
  %16 = sext i32 %15 to i64
  %17 = and i1 %12, %13
  %18 = sub i64 %4, %16
  %19 = icmp sge i64 %18, 0
  %20 = and i1 %17, %19
  %21 = icmp slt i64 %18, 12
  %22 = and i1 %20, %21
  %23 = zext i1 %22 to i8
  %24 = getelementptr inbounds [24 x i8], ptr %1, i32 0, i32 %3
  store i8 %23, ptr %24, align 1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

define ptx_kernel void @input_transpose_fusion(ptr noalias align 16 dereferenceable(4128768) %0, ptr noalias align 256 dereferenceable(24) %1, ptr noalias align 256 dereferenceable(2064384) %2, ptr noalias align 256 dereferenceable(2064384) %3) #2 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %7 = urem i32 %5, 32
  %8 = icmp sle i32 %7, 23
  br i1 %8, label %9, label %90

9:                                                ; preds = %4
  %10 = getelementptr inbounds [24 x i8], ptr %1, i32 0, i32 %7
  %11 = load i8, ptr %10, align 1, !invariant.load !2
  %12 = trunc i8 %11 to i1
  %13 = udiv i32 %5, 32
  %14 = mul i32 %13, 24
  %15 = mul i32 %6, 768
  %16 = add i32 %14, %15
  %17 = add i32 %16, %7
  %18 = getelementptr inbounds [258048 x { double, double }], ptr %0, i32 0, i32 %17
  %19 = load { double, double }, ptr %18, align 8, !invariant.load !2
  %20 = select i1 %12, { double, double } %19, { double, double } zeroinitializer
  %21 = extractvalue { double, double } %20, 0
  %22 = mul i32 %7, 33
  %23 = add i32 %22, %13
  %24 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %23
  store double %21, ptr %24, align 8
  %25 = extractvalue { double, double } %20, 1
  %26 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %23
  store double %25, ptr %26, align 8
  %27 = add i32 %17, 96
  %28 = getelementptr inbounds [258048 x { double, double }], ptr %0, i32 0, i32 %27
  %29 = load { double, double }, ptr %28, align 8, !invariant.load !2
  %30 = select i1 %12, { double, double } %29, { double, double } zeroinitializer
  %31 = extractvalue { double, double } %30, 0
  %32 = add i32 %23, 4
  %33 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %32
  store double %31, ptr %33, align 8
  %34 = extractvalue { double, double } %30, 1
  %35 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %32
  store double %34, ptr %35, align 8
  %36 = add i32 %17, 192
  %37 = getelementptr inbounds [258048 x { double, double }], ptr %0, i32 0, i32 %36
  %38 = load { double, double }, ptr %37, align 8, !invariant.load !2
  %39 = select i1 %12, { double, double } %38, { double, double } zeroinitializer
  %40 = extractvalue { double, double } %39, 0
  %41 = add i32 %23, 8
  %42 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %41
  store double %40, ptr %42, align 8
  %43 = extractvalue { double, double } %39, 1
  %44 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %41
  store double %43, ptr %44, align 8
  %45 = add i32 %17, 288
  %46 = getelementptr inbounds [258048 x { double, double }], ptr %0, i32 0, i32 %45
  %47 = load { double, double }, ptr %46, align 8, !invariant.load !2
  %48 = select i1 %12, { double, double } %47, { double, double } zeroinitializer
  %49 = extractvalue { double, double } %48, 0
  %50 = add i32 %23, 12
  %51 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %50
  store double %49, ptr %51, align 8
  %52 = extractvalue { double, double } %48, 1
  %53 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %50
  store double %52, ptr %53, align 8
  %54 = add i32 %17, 384
  %55 = getelementptr inbounds [258048 x { double, double }], ptr %0, i32 0, i32 %54
  %56 = load { double, double }, ptr %55, align 8, !invariant.load !2
  %57 = select i1 %12, { double, double } %56, { double, double } zeroinitializer
  %58 = extractvalue { double, double } %57, 0
  %59 = add i32 %23, 16
  %60 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %59
  store double %58, ptr %60, align 8
  %61 = extractvalue { double, double } %57, 1
  %62 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %59
  store double %61, ptr %62, align 8
  %63 = add i32 %17, 480
  %64 = getelementptr inbounds [258048 x { double, double }], ptr %0, i32 0, i32 %63
  %65 = load { double, double }, ptr %64, align 8, !invariant.load !2
  %66 = select i1 %12, { double, double } %65, { double, double } zeroinitializer
  %67 = extractvalue { double, double } %66, 0
  %68 = add i32 %23, 20
  %69 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %68
  store double %67, ptr %69, align 8
  %70 = extractvalue { double, double } %66, 1
  %71 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %68
  store double %70, ptr %71, align 8
  %72 = add i32 %17, 576
  %73 = getelementptr inbounds [258048 x { double, double }], ptr %0, i32 0, i32 %72
  %74 = load { double, double }, ptr %73, align 8, !invariant.load !2
  %75 = select i1 %12, { double, double } %74, { double, double } zeroinitializer
  %76 = extractvalue { double, double } %75, 0
  %77 = add i32 %23, 24
  %78 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %77
  store double %76, ptr %78, align 8
  %79 = extractvalue { double, double } %75, 1
  %80 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %77
  store double %79, ptr %80, align 8
  %81 = add i32 %17, 672
  %82 = getelementptr inbounds [258048 x { double, double }], ptr %0, i32 0, i32 %81
  %83 = load { double, double }, ptr %82, align 8, !invariant.load !2
  %84 = select i1 %12, { double, double } %83, { double, double } zeroinitializer
  %85 = extractvalue { double, double } %84, 0
  %86 = add i32 %23, 28
  %87 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %86
  store double %85, ptr %87, align 8
  %88 = extractvalue { double, double } %84, 1
  %89 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %86
  store double %88, ptr %89, align 8
  br label %90

90:                                               ; preds = %9, %4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %91 = udiv i32 %5, 32
  %92 = mul i32 %91, 33
  %93 = add i32 %92, %7
  %94 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %93
  %95 = load double, ptr %94, align 8
  %96 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %93
  %97 = load double, ptr %96, align 8
  %98 = mul i32 %91, 10752
  %99 = mul i32 %6, 32
  %100 = add i32 %98, %99
  %101 = add i32 %100, %7
  %102 = getelementptr inbounds [258048 x double], ptr %2, i32 0, i32 %101
  store double %95, ptr %102, align 8
  %103 = getelementptr inbounds [258048 x double], ptr %3, i32 0, i32 %101
  store double %97, ptr %103, align 8
  %104 = add i32 %93, 132
  %105 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %104
  %106 = load double, ptr %105, align 8
  %107 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %104
  %108 = load double, ptr %107, align 8
  %109 = add i32 %101, 43008
  %110 = getelementptr inbounds [258048 x double], ptr %2, i32 0, i32 %109
  store double %106, ptr %110, align 8
  %111 = getelementptr inbounds [258048 x double], ptr %3, i32 0, i32 %109
  store double %108, ptr %111, align 8
  %112 = add i32 %93, 264
  %113 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %112
  %114 = load double, ptr %113, align 8
  %115 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %112
  %116 = load double, ptr %115, align 8
  %117 = add i32 %101, 86016
  %118 = getelementptr inbounds [258048 x double], ptr %2, i32 0, i32 %117
  store double %114, ptr %118, align 8
  %119 = getelementptr inbounds [258048 x double], ptr %3, i32 0, i32 %117
  store double %116, ptr %119, align 8
  %120 = add i32 %93, 396
  %121 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %120
  %122 = load double, ptr %121, align 8
  %123 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %120
  %124 = load double, ptr %123, align 8
  %125 = add i32 %101, 129024
  %126 = getelementptr inbounds [258048 x double], ptr %2, i32 0, i32 %125
  store double %122, ptr %126, align 8
  %127 = getelementptr inbounds [258048 x double], ptr %3, i32 0, i32 %125
  store double %124, ptr %127, align 8
  %128 = add i32 %93, 528
  %129 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %128
  %130 = load double, ptr %129, align 8
  %131 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %128
  %132 = load double, ptr %131, align 8
  %133 = add i32 %101, 172032
  %134 = getelementptr inbounds [258048 x double], ptr %2, i32 0, i32 %133
  store double %130, ptr %134, align 8
  %135 = getelementptr inbounds [258048 x double], ptr %3, i32 0, i32 %133
  store double %132, ptr %135, align 8
  %136 = add i32 %93, 660
  %137 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %136
  %138 = load double, ptr %137, align 8
  %139 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %136
  %140 = load double, ptr %139, align 8
  %141 = add i32 %101, 215040
  %142 = getelementptr inbounds [258048 x double], ptr %2, i32 0, i32 %141
  store double %138, ptr %142, align 8
  %143 = getelementptr inbounds [258048 x double], ptr %3, i32 0, i32 %141
  store double %140, ptr %143, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

define ptx_kernel void @input_transpose_fusion_2(ptr noalias align 16 dereferenceable(24772608) %0, ptr noalias align 256 dereferenceable(12386304) %1, ptr noalias align 256 dereferenceable(12386304) %2) #2 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %6 = urem i32 %5, 5
  %7 = mul i32 %6, 32
  %8 = urem i32 %4, 32
  %9 = add i32 %7, %8
  %10 = icmp sle i32 %9, 143
  br i1 %10, label %11, label %83

11:                                               ; preds = %3
  %12 = udiv i32 %5, 5
  %13 = mul i32 %12, 4608
  %14 = add i32 %7, %13
  %15 = udiv i32 %4, 32
  %16 = mul i32 %15, 144
  %17 = add i32 %14, %16
  %18 = add i32 %17, %8
  %19 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %18
  %20 = load { double, double }, ptr %19, align 8, !invariant.load !2
  %21 = extractvalue { double, double } %20, 0
  %22 = mul i32 %8, 33
  %23 = add i32 %22, %15
  %24 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %23
  store double %21, ptr %24, align 8
  %25 = extractvalue { double, double } %20, 1
  %26 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %23
  store double %25, ptr %26, align 8
  %27 = add i32 %18, 576
  %28 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %27
  %29 = load { double, double }, ptr %28, align 8, !invariant.load !2
  %30 = extractvalue { double, double } %29, 0
  %31 = add i32 %23, 4
  %32 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %31
  store double %30, ptr %32, align 8
  %33 = extractvalue { double, double } %29, 1
  %34 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %31
  store double %33, ptr %34, align 8
  %35 = add i32 %18, 1152
  %36 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %35
  %37 = load { double, double }, ptr %36, align 8, !invariant.load !2
  %38 = extractvalue { double, double } %37, 0
  %39 = add i32 %23, 8
  %40 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %39
  store double %38, ptr %40, align 8
  %41 = extractvalue { double, double } %37, 1
  %42 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %39
  store double %41, ptr %42, align 8
  %43 = add i32 %18, 1728
  %44 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %43
  %45 = load { double, double }, ptr %44, align 8, !invariant.load !2
  %46 = extractvalue { double, double } %45, 0
  %47 = add i32 %23, 12
  %48 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %47
  store double %46, ptr %48, align 8
  %49 = extractvalue { double, double } %45, 1
  %50 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %47
  store double %49, ptr %50, align 8
  %51 = add i32 %18, 2304
  %52 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %51
  %53 = load { double, double }, ptr %52, align 8, !invariant.load !2
  %54 = extractvalue { double, double } %53, 0
  %55 = add i32 %23, 16
  %56 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %55
  store double %54, ptr %56, align 8
  %57 = extractvalue { double, double } %53, 1
  %58 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %55
  store double %57, ptr %58, align 8
  %59 = add i32 %18, 2880
  %60 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %59
  %61 = load { double, double }, ptr %60, align 8, !invariant.load !2
  %62 = extractvalue { double, double } %61, 0
  %63 = add i32 %23, 20
  %64 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %63
  store double %62, ptr %64, align 8
  %65 = extractvalue { double, double } %61, 1
  %66 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %63
  store double %65, ptr %66, align 8
  %67 = add i32 %18, 3456
  %68 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %67
  %69 = load { double, double }, ptr %68, align 8, !invariant.load !2
  %70 = extractvalue { double, double } %69, 0
  %71 = add i32 %23, 24
  %72 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %71
  store double %70, ptr %72, align 8
  %73 = extractvalue { double, double } %69, 1
  %74 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %71
  store double %73, ptr %74, align 8
  %75 = add i32 %18, 4032
  %76 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %75
  %77 = load { double, double }, ptr %76, align 8, !invariant.load !2
  %78 = extractvalue { double, double } %77, 0
  %79 = add i32 %23, 28
  %80 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %79
  store double %78, ptr %80, align 8
  %81 = extractvalue { double, double } %77, 1
  %82 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %79
  store double %81, ptr %82, align 8
  br label %83

83:                                               ; preds = %11, %3
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %84 = udiv i32 %4, 32
  %85 = mul i32 %84, 33
  %86 = add i32 %85, %8
  %87 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %86
  %88 = load double, ptr %87, align 8
  %89 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %86
  %90 = load double, ptr %89, align 8
  %91 = mul i32 %6, 344064
  %92 = udiv i32 %5, 5
  %93 = mul i32 %92, 32
  %94 = add i32 %91, %93
  %95 = mul i32 %84, 10752
  %96 = add i32 %94, %95
  %97 = add i32 %96, %8
  %98 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %97
  store double %88, ptr %98, align 8
  %99 = getelementptr inbounds [1548288 x double], ptr %2, i32 0, i32 %97
  store double %90, ptr %99, align 8
  %100 = add i32 %86, 132
  %101 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %100
  %102 = load double, ptr %101, align 8
  %103 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %100
  %104 = load double, ptr %103, align 8
  %105 = add i32 %97, 43008
  %106 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %105
  store double %102, ptr %106, align 8
  %107 = getelementptr inbounds [1548288 x double], ptr %2, i32 0, i32 %105
  store double %104, ptr %107, align 8
  %108 = add i32 %86, 264
  %109 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %108
  %110 = load double, ptr %109, align 8
  %111 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %108
  %112 = load double, ptr %111, align 8
  %113 = add i32 %97, 86016
  %114 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %113
  store double %110, ptr %114, align 8
  %115 = getelementptr inbounds [1548288 x double], ptr %2, i32 0, i32 %113
  store double %112, ptr %115, align 8
  %116 = add i32 %86, 396
  %117 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %116
  %118 = load double, ptr %117, align 8
  %119 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %116
  %120 = load double, ptr %119, align 8
  %121 = add i32 %97, 129024
  %122 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %121
  store double %118, ptr %122, align 8
  %123 = getelementptr inbounds [1548288 x double], ptr %2, i32 0, i32 %121
  store double %120, ptr %123, align 8
  %124 = mul i32 %6, 8
  %125 = add i32 %124, 4
  %126 = icmp sle i32 %125, 35
  br i1 %126, label %127, label %136

127:                                              ; preds = %83
  %128 = add i32 %86, 528
  %129 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %128
  %130 = load double, ptr %129, align 8
  %131 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %128
  %132 = load double, ptr %131, align 8
  %133 = add i32 %97, 172032
  %134 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %133
  store double %130, ptr %134, align 8
  %135 = getelementptr inbounds [1548288 x double], ptr %2, i32 0, i32 %133
  store double %132, ptr %135, align 8
  br label %136

136:                                              ; preds = %127, %83
  %137 = add i32 %124, 5
  %138 = icmp sle i32 %137, 35
  br i1 %138, label %139, label %148

139:                                              ; preds = %136
  %140 = add i32 %86, 660
  %141 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %140
  %142 = load double, ptr %141, align 8
  %143 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %140
  %144 = load double, ptr %143, align 8
  %145 = add i32 %97, 215040
  %146 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %145
  store double %142, ptr %146, align 8
  %147 = getelementptr inbounds [1548288 x double], ptr %2, i32 0, i32 %145
  store double %144, ptr %147, align 8
  br label %148

148:                                              ; preds = %139, %136
  %149 = add i32 %124, 6
  %150 = icmp sle i32 %149, 35
  br i1 %150, label %151, label %160

151:                                              ; preds = %148
  %152 = add i32 %86, 792
  %153 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %152
  %154 = load double, ptr %153, align 8
  %155 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %152
  %156 = load double, ptr %155, align 8
  %157 = add i32 %97, 258048
  %158 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %157
  store double %154, ptr %158, align 8
  %159 = getelementptr inbounds [1548288 x double], ptr %2, i32 0, i32 %157
  store double %156, ptr %159, align 8
  br label %160

160:                                              ; preds = %151, %148
  %161 = add i32 %124, 7
  %162 = icmp sle i32 %161, 35
  br i1 %162, label %163, label %172

163:                                              ; preds = %160
  %164 = add i32 %86, 924
  %165 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_11 to ptr), i32 0, i32 %164
  %166 = load double, ptr %165, align 8
  %167 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_02 to ptr), i32 0, i32 %164
  %168 = load double, ptr %167, align 8
  %169 = add i32 %97, 301056
  %170 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %169
  store double %166, ptr %170, align 8
  %171 = getelementptr inbounds [1548288 x double], ptr %2, i32 0, i32 %169
  store double %168, ptr %171, align 8
  br label %172

172:                                              ; preds = %163, %160
  ret void
}

define ptx_kernel void @input_scatter_fusion(ptr noalias align 256 dereferenceable(12386304) %0, ptr noalias align 256 dereferenceable(2064384) %1, ptr noalias align 256 dereferenceable(4) %2, ptr noalias align 256 dereferenceable(12386304) %3) #2 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %7 = udiv i32 %6, 21
  %8 = call i32 @fused_scatter_concatenate_3_3(ptr %0, ptr %1, ptr %2, i32 %7, i32 0)
  %9 = call i32 @fused_scatter_concatenate_3_3(ptr %0, ptr %1, ptr %2, i32 %7, i32 1)
  %10 = icmp ule i32 %8, 11
  %11 = icmp ule i32 %9, 11
  %12 = and i1 %10, %11
  br i1 %12, label %13, label %41

13:                                               ; preds = %4
  %14 = mul i32 %5, 4
  %15 = mul i32 %6, 512
  %16 = add i32 %14, %15
  %17 = getelementptr inbounds [258048 x double], ptr %1, i32 0, i32 %16
  %18 = load <4 x double>, ptr %17, align 8, !invariant.load !2
  %19 = extractelement <4 x double> %18, i64 0
  %20 = urem i32 %6, 21
  %21 = mul i32 %20, 512
  %22 = mul i32 %8, 129024
  %23 = add i32 %21, %22
  %24 = mul i32 %9, 10752
  %25 = add i32 %23, %24
  %26 = add i32 %25, %14
  %27 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %26
  %28 = atomicrmw fadd ptr %27, double %19 monotonic, align 8
  %29 = extractelement <4 x double> %18, i64 1
  %30 = add i32 %26, 1
  %31 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %30
  %32 = atomicrmw fadd ptr %31, double %29 monotonic, align 8
  %33 = extractelement <4 x double> %18, i64 2
  %34 = add i32 %26, 2
  %35 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %34
  %36 = atomicrmw fadd ptr %35, double %33 monotonic, align 8
  %37 = extractelement <4 x double> %18, i64 3
  %38 = add i32 %26, 3
  %39 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %38
  %40 = atomicrmw fadd ptr %39, double %37 monotonic, align 8
  br label %41

41:                                               ; preds = %13, %4
  ret void
}

define internal i32 @fused_scatter_concatenate_3_3(ptr noalias %0, ptr noalias %1, ptr noalias %2, i32 %3, i32 %4) {
  %6 = icmp ult i32 %4, 1
  br i1 %6, label %7, label %22

7:                                                ; preds = %5
  %8 = zext i32 %3 to i64
  %9 = getelementptr inbounds [1 x i32], ptr %2, i32 0, i32 0
  %10 = load i32, ptr %9, align 4, !invariant.load !2
  %11 = lshr i32 %10, 1
  %12 = and i32 %11, 1
  %13 = mul i32 %12, 12
  %14 = sext i32 %13 to i64
  %15 = sub i64 %8, %14
  %16 = call i64 @llvm.smax.i64(i64 %15, i64 0)
  %17 = call i64 @llvm.smin.i64(i64 %16, i64 11)
  %18 = icmp slt i64 %17, 0
  %19 = add i64 %17, 12
  %20 = select i1 %18, i64 %19, i64 %17
  %21 = trunc i64 %20 to i32
  br label %36

22:                                               ; preds = %5
  %23 = zext i32 %3 to i64
  %24 = getelementptr inbounds [1 x i32], ptr %2, i32 0, i32 0
  %25 = load i32, ptr %24, align 4, !invariant.load !2
  %26 = and i32 %25, 1
  %27 = mul i32 %26, 12
  %28 = sext i32 %27 to i64
  %29 = sub i64 %23, %28
  %30 = call i64 @llvm.smax.i64(i64 %29, i64 0)
  %31 = call i64 @llvm.smin.i64(i64 %30, i64 11)
  %32 = icmp slt i64 %31, 0
  %33 = add i64 %31, 12
  %34 = select i1 %32, i64 %33, i64 %31
  %35 = trunc i64 %34 to i32
  br label %36

36:                                               ; preds = %7, %22
  %37 = phi i32 [ %35, %22 ], [ %21, %7 ]
  br label %38

38:                                               ; preds = %36
  ret i32 %37
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #4

define ptx_kernel void @wrapped_transpose(ptr noalias align 256 dereferenceable(12386304) %0, ptr noalias align 256 dereferenceable(12386304) %1) #2 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %5 = udiv i32 %4, 16
  %6 = urem i32 %5, 21
  %7 = mul i32 %6, 512
  %8 = urem i32 %4, 16
  %9 = mul i32 %8, 32
  %10 = add i32 %7, %9
  %11 = udiv i32 %4, 336
  %12 = mul i32 %11, 344064
  %13 = add i32 %10, %12
  %14 = udiv i32 %3, 32
  %15 = mul i32 %14, 10752
  %16 = add i32 %13, %15
  %17 = urem i32 %3, 32
  %18 = add i32 %16, %17
  %19 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %18
  %20 = load double, ptr %19, align 8, !invariant.load !2
  %21 = mul i32 %17, 33
  %22 = add i32 %21, %14
  %23 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %22
  store double %20, ptr %23, align 8
  %24 = add i32 %18, 43008
  %25 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %24
  %26 = load double, ptr %25, align 8, !invariant.load !2
  %27 = add i32 %22, 4
  %28 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %27
  store double %26, ptr %28, align 8
  %29 = add i32 %18, 86016
  %30 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %29
  %31 = load double, ptr %30, align 8, !invariant.load !2
  %32 = add i32 %22, 8
  %33 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %32
  store double %31, ptr %33, align 8
  %34 = add i32 %18, 129024
  %35 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %34
  %36 = load double, ptr %35, align 8, !invariant.load !2
  %37 = add i32 %22, 12
  %38 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %37
  store double %36, ptr %38, align 8
  %39 = mul i32 %11, 8
  %40 = add i32 %39, 4
  %41 = icmp sle i32 %40, 35
  br i1 %41, label %42, label %48

42:                                               ; preds = %2
  %43 = add i32 %18, 172032
  %44 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %43
  %45 = load double, ptr %44, align 8, !invariant.load !2
  %46 = add i32 %22, 16
  %47 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %46
  store double %45, ptr %47, align 8
  br label %48

48:                                               ; preds = %42, %2
  %49 = add i32 %39, 5
  %50 = icmp sle i32 %49, 35
  br i1 %50, label %51, label %57

51:                                               ; preds = %48
  %52 = add i32 %18, 215040
  %53 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %52
  %54 = load double, ptr %53, align 8, !invariant.load !2
  %55 = add i32 %22, 20
  %56 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %55
  store double %54, ptr %56, align 8
  br label %57

57:                                               ; preds = %51, %48
  %58 = add i32 %39, 6
  %59 = icmp sle i32 %58, 35
  br i1 %59, label %60, label %66

60:                                               ; preds = %57
  %61 = add i32 %18, 258048
  %62 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %61
  %63 = load double, ptr %62, align 8, !invariant.load !2
  %64 = add i32 %22, 24
  %65 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %64
  store double %63, ptr %65, align 8
  br label %66

66:                                               ; preds = %60, %57
  %67 = add i32 %39, 7
  %68 = icmp sle i32 %67, 35
  br i1 %68, label %69, label %75

69:                                               ; preds = %66
  %70 = add i32 %18, 301056
  %71 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %70
  %72 = load double, ptr %71, align 8, !invariant.load !2
  %73 = add i32 %22, 28
  %74 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %73
  store double %72, ptr %74, align 8
  br label %75

75:                                               ; preds = %69, %66
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %76 = mul i32 %11, 32
  %77 = add i32 %76, %17
  %78 = icmp sle i32 %77, 143
  br i1 %78, label %79, label %127

79:                                               ; preds = %75
  %80 = mul i32 %14, 33
  %81 = add i32 %80, %17
  %82 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %81
  %83 = load double, ptr %82, align 8
  %84 = mul i32 %6, 73728
  %85 = mul i32 %8, 4608
  %86 = add i32 %84, %85
  %87 = add i32 %86, %76
  %88 = mul i32 %14, 144
  %89 = add i32 %87, %88
  %90 = add i32 %89, %17
  %91 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %90
  store double %83, ptr %91, align 8
  %92 = add i32 %81, 132
  %93 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %92
  %94 = load double, ptr %93, align 8
  %95 = add i32 %90, 576
  %96 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %95
  store double %94, ptr %96, align 8
  %97 = add i32 %81, 264
  %98 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %97
  %99 = load double, ptr %98, align 8
  %100 = add i32 %90, 1152
  %101 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %100
  store double %99, ptr %101, align 8
  %102 = add i32 %81, 396
  %103 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %102
  %104 = load double, ptr %103, align 8
  %105 = add i32 %90, 1728
  %106 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %105
  store double %104, ptr %106, align 8
  %107 = add i32 %81, 528
  %108 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %107
  %109 = load double, ptr %108, align 8
  %110 = add i32 %90, 2304
  %111 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %110
  store double %109, ptr %111, align 8
  %112 = add i32 %81, 660
  %113 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %112
  %114 = load double, ptr %113, align 8
  %115 = add i32 %90, 2880
  %116 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %115
  store double %114, ptr %116, align 8
  %117 = add i32 %81, 792
  %118 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %117
  %119 = load double, ptr %118, align 8
  %120 = add i32 %90, 3456
  %121 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %120
  store double %119, ptr %121, align 8
  %122 = add i32 %81, 924
  %123 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_03 to ptr), i32 0, i32 %122
  %124 = load double, ptr %123, align 8
  %125 = add i32 %90, 4032
  %126 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %125
  store double %124, ptr %126, align 8
  br label %127

127:                                              ; preds = %79, %75
  ret void
}

define ptx_kernel void @input_transpose_fusion_1(ptr noalias align 256 dereferenceable(12386304) %0, ptr noalias align 256 dereferenceable(12386304) %1, ptr noalias align 256 dereferenceable(24772608) %2) #2 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %6 = udiv i32 %5, 16
  %7 = urem i32 %6, 21
  %8 = mul i32 %7, 512
  %9 = urem i32 %5, 16
  %10 = mul i32 %9, 32
  %11 = add i32 %8, %10
  %12 = udiv i32 %5, 336
  %13 = mul i32 %12, 344064
  %14 = add i32 %11, %13
  %15 = udiv i32 %4, 32
  %16 = mul i32 %15, 10752
  %17 = add i32 %14, %16
  %18 = urem i32 %4, 32
  %19 = add i32 %17, %18
  %20 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %19
  %21 = load double, ptr %20, align 8, !invariant.load !2
  %22 = mul i32 %18, 33
  %23 = add i32 %22, %15
  %24 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %23
  store double %21, ptr %24, align 8
  %25 = add i32 %19, 43008
  %26 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %25
  %27 = load double, ptr %26, align 8, !invariant.load !2
  %28 = add i32 %23, 4
  %29 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %28
  store double %27, ptr %29, align 8
  %30 = add i32 %19, 86016
  %31 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %30
  %32 = load double, ptr %31, align 8, !invariant.load !2
  %33 = add i32 %23, 8
  %34 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %33
  store double %32, ptr %34, align 8
  %35 = add i32 %19, 129024
  %36 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %35
  %37 = load double, ptr %36, align 8, !invariant.load !2
  %38 = add i32 %23, 12
  %39 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %38
  store double %37, ptr %39, align 8
  %40 = mul i32 %12, 8
  %41 = add i32 %40, 4
  %42 = icmp sle i32 %41, 35
  br i1 %42, label %43, label %49

43:                                               ; preds = %3
  %44 = add i32 %19, 172032
  %45 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %44
  %46 = load double, ptr %45, align 8, !invariant.load !2
  %47 = add i32 %23, 16
  %48 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %47
  store double %46, ptr %48, align 8
  br label %49

49:                                               ; preds = %43, %3
  %50 = add i32 %40, 5
  %51 = icmp sle i32 %50, 35
  br i1 %51, label %52, label %58

52:                                               ; preds = %49
  %53 = add i32 %19, 215040
  %54 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %53
  %55 = load double, ptr %54, align 8, !invariant.load !2
  %56 = add i32 %23, 20
  %57 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %56
  store double %55, ptr %57, align 8
  br label %58

58:                                               ; preds = %52, %49
  %59 = add i32 %40, 6
  %60 = icmp sle i32 %59, 35
  br i1 %60, label %61, label %67

61:                                               ; preds = %58
  %62 = add i32 %19, 258048
  %63 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %62
  %64 = load double, ptr %63, align 8, !invariant.load !2
  %65 = add i32 %23, 24
  %66 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %65
  store double %64, ptr %66, align 8
  br label %67

67:                                               ; preds = %61, %58
  %68 = add i32 %40, 7
  %69 = icmp sle i32 %68, 35
  br i1 %69, label %70, label %76

70:                                               ; preds = %67
  %71 = add i32 %19, 301056
  %72 = getelementptr inbounds [1548288 x double], ptr %1, i32 0, i32 %71
  %73 = load double, ptr %72, align 8, !invariant.load !2
  %74 = add i32 %23, 28
  %75 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %74
  store double %73, ptr %75, align 8
  br label %76

76:                                               ; preds = %70, %67
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %77 = mul i32 %12, 32
  %78 = add i32 %77, %18
  %79 = icmp sle i32 %78, 143
  br i1 %79, label %80, label %160

80:                                               ; preds = %76
  %81 = mul i32 %15, 33
  %82 = add i32 %81, %18
  %83 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %82
  %84 = load double, ptr %83, align 8
  %85 = mul i32 %7, 73728
  %86 = mul i32 %9, 4608
  %87 = add i32 %85, %86
  %88 = add i32 %87, %77
  %89 = mul i32 %15, 144
  %90 = add i32 %88, %89
  %91 = add i32 %90, %18
  %92 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %91
  %93 = load double, ptr %92, align 8, !invariant.load !2
  %94 = insertvalue { double, double } poison, double %93, 0
  %95 = insertvalue { double, double } %94, double %84, 1
  %96 = getelementptr inbounds [1548288 x { double, double }], ptr %2, i32 0, i32 %91
  store { double, double } %95, ptr %96, align 8
  %97 = add i32 %82, 132
  %98 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %97
  %99 = load double, ptr %98, align 8
  %100 = add i32 %91, 576
  %101 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %100
  %102 = load double, ptr %101, align 8, !invariant.load !2
  %103 = insertvalue { double, double } poison, double %102, 0
  %104 = insertvalue { double, double } %103, double %99, 1
  %105 = getelementptr inbounds [1548288 x { double, double }], ptr %2, i32 0, i32 %100
  store { double, double } %104, ptr %105, align 8
  %106 = add i32 %82, 264
  %107 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %106
  %108 = load double, ptr %107, align 8
  %109 = add i32 %91, 1152
  %110 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %109
  %111 = load double, ptr %110, align 8, !invariant.load !2
  %112 = insertvalue { double, double } poison, double %111, 0
  %113 = insertvalue { double, double } %112, double %108, 1
  %114 = getelementptr inbounds [1548288 x { double, double }], ptr %2, i32 0, i32 %109
  store { double, double } %113, ptr %114, align 8
  %115 = add i32 %82, 396
  %116 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %115
  %117 = load double, ptr %116, align 8
  %118 = add i32 %91, 1728
  %119 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %118
  %120 = load double, ptr %119, align 8, !invariant.load !2
  %121 = insertvalue { double, double } poison, double %120, 0
  %122 = insertvalue { double, double } %121, double %117, 1
  %123 = getelementptr inbounds [1548288 x { double, double }], ptr %2, i32 0, i32 %118
  store { double, double } %122, ptr %123, align 8
  %124 = add i32 %82, 528
  %125 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %124
  %126 = load double, ptr %125, align 8
  %127 = add i32 %91, 2304
  %128 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %127
  %129 = load double, ptr %128, align 8, !invariant.load !2
  %130 = insertvalue { double, double } poison, double %129, 0
  %131 = insertvalue { double, double } %130, double %126, 1
  %132 = getelementptr inbounds [1548288 x { double, double }], ptr %2, i32 0, i32 %127
  store { double, double } %131, ptr %132, align 8
  %133 = add i32 %82, 660
  %134 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %133
  %135 = load double, ptr %134, align 8
  %136 = add i32 %91, 2880
  %137 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %136
  %138 = load double, ptr %137, align 8, !invariant.load !2
  %139 = insertvalue { double, double } poison, double %138, 0
  %140 = insertvalue { double, double } %139, double %135, 1
  %141 = getelementptr inbounds [1548288 x { double, double }], ptr %2, i32 0, i32 %136
  store { double, double } %140, ptr %141, align 8
  %142 = add i32 %82, 792
  %143 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %142
  %144 = load double, ptr %143, align 8
  %145 = add i32 %91, 3456
  %146 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %145
  %147 = load double, ptr %146, align 8, !invariant.load !2
  %148 = insertvalue { double, double } poison, double %147, 0
  %149 = insertvalue { double, double } %148, double %144, 1
  %150 = getelementptr inbounds [1548288 x { double, double }], ptr %2, i32 0, i32 %145
  store { double, double } %149, ptr %150, align 8
  %151 = add i32 %82, 924
  %152 = getelementptr inbounds [1056 x double], ptr addrspacecast (ptr addrspace(3) @shared_04 to ptr), i32 0, i32 %151
  %153 = load double, ptr %152, align 8
  %154 = add i32 %91, 4032
  %155 = getelementptr inbounds [1548288 x double], ptr %0, i32 0, i32 %154
  %156 = load double, ptr %155, align 8, !invariant.load !2
  %157 = insertvalue { double, double } poison, double %156, 0
  %158 = insertvalue { double, double } %157, double %153, 1
  %159 = getelementptr inbounds [1548288 x { double, double }], ptr %2, i32 0, i32 %154
  store { double, double } %158, ptr %159, align 8
  br label %160

160:                                              ; preds = %80, %76
  ret void
}

attributes #0 = { "nvvm.reqntid"="24,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { "nvvm.reqntid"="128,1,1" }
attributes #3 = { convergent nocallback nounwind }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 24}
!2 = !{}
!3 = !{i32 0, i32 128}
!4 = !{i32 0, i32 336}
!5 = !{i32 0, i32 1680}
!6 = !{i32 0, i32 504}
