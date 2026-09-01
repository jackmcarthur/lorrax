; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_reduce_fusion(ptr noalias align 16 dereferenceable(262144) %0, ptr noalias align 256 dereferenceable(1024) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = udiv i32 %3, 32
  %6 = mul i32 %5, 256
  %7 = mul i32 %4, 2048
  %8 = add i32 %6, %7
  %9 = urem i32 %3, 32
  %10 = add i32 %8, %9
  %11 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %10
  %12 = load double, ptr %11, align 8, !invariant.load !3
  %13 = call double @region_0_1_reduce_sum_5_0(double 0.000000e+00, double %12)
  %14 = add i32 %10, 32
  %15 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %14
  %16 = load double, ptr %15, align 8, !invariant.load !3
  %17 = call double @region_0_1_reduce_sum_5_0(double %13, double %16)
  %18 = add i32 %10, 64
  %19 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %18
  %20 = load double, ptr %19, align 8, !invariant.load !3
  %21 = call double @region_0_1_reduce_sum_5_0(double %17, double %20)
  %22 = add i32 %10, 96
  %23 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %22
  %24 = load double, ptr %23, align 8, !invariant.load !3
  %25 = call double @region_0_1_reduce_sum_5_0(double %21, double %24)
  %26 = add i32 %10, 128
  %27 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %26
  %28 = load double, ptr %27, align 8, !invariant.load !3
  %29 = call double @region_0_1_reduce_sum_5_0(double %25, double %28)
  %30 = add i32 %10, 160
  %31 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %30
  %32 = load double, ptr %31, align 8, !invariant.load !3
  %33 = call double @region_0_1_reduce_sum_5_0(double %29, double %32)
  %34 = add i32 %10, 192
  %35 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %34
  %36 = load double, ptr %35, align 8, !invariant.load !3
  %37 = call double @region_0_1_reduce_sum_5_0(double %33, double %36)
  %38 = add i32 %10, 224
  %39 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %38
  %40 = load double, ptr %39, align 8, !invariant.load !3
  %41 = call double @region_0_1_reduce_sum_5_0(double %37, double %40)
  %42 = bitcast double %41 to i64
  %43 = bitcast i64 %42 to <2 x i32>
  %44 = extractelement <2 x i32> %43, i32 0
  %45 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %44, i32 16, i32 31)
  %46 = insertelement <2 x i32> undef, i32 %45, i32 0
  %47 = extractelement <2 x i32> %43, i32 1
  %48 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %47, i32 16, i32 31)
  %49 = insertelement <2 x i32> %46, i32 %48, i32 1
  %50 = bitcast <2 x i32> %49 to double
  %51 = call double @region_0_1_reduce_sum_5_0(double %41, double %50)
  %52 = bitcast double %51 to i64
  %53 = bitcast i64 %52 to <2 x i32>
  %54 = extractelement <2 x i32> %53, i32 0
  %55 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %54, i32 8, i32 31)
  %56 = insertelement <2 x i32> undef, i32 %55, i32 0
  %57 = extractelement <2 x i32> %53, i32 1
  %58 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %57, i32 8, i32 31)
  %59 = insertelement <2 x i32> %56, i32 %58, i32 1
  %60 = bitcast <2 x i32> %59 to double
  %61 = call double @region_0_1_reduce_sum_5_0(double %51, double %60)
  %62 = bitcast double %61 to i64
  %63 = bitcast i64 %62 to <2 x i32>
  %64 = extractelement <2 x i32> %63, i32 0
  %65 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %64, i32 4, i32 31)
  %66 = insertelement <2 x i32> undef, i32 %65, i32 0
  %67 = extractelement <2 x i32> %63, i32 1
  %68 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %67, i32 4, i32 31)
  %69 = insertelement <2 x i32> %66, i32 %68, i32 1
  %70 = bitcast <2 x i32> %69 to double
  %71 = call double @region_0_1_reduce_sum_5_0(double %61, double %70)
  %72 = bitcast double %71 to i64
  %73 = bitcast i64 %72 to <2 x i32>
  %74 = extractelement <2 x i32> %73, i32 0
  %75 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %74, i32 2, i32 31)
  %76 = insertelement <2 x i32> undef, i32 %75, i32 0
  %77 = extractelement <2 x i32> %73, i32 1
  %78 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %77, i32 2, i32 31)
  %79 = insertelement <2 x i32> %76, i32 %78, i32 1
  %80 = bitcast <2 x i32> %79 to double
  %81 = call double @region_0_1_reduce_sum_5_0(double %71, double %80)
  %82 = bitcast double %81 to i64
  %83 = bitcast i64 %82 to <2 x i32>
  %84 = extractelement <2 x i32> %83, i32 0
  %85 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %84, i32 1, i32 31)
  %86 = insertelement <2 x i32> undef, i32 %85, i32 0
  %87 = extractelement <2 x i32> %83, i32 1
  %88 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %87, i32 1, i32 31)
  %89 = insertelement <2 x i32> %86, i32 %88, i32 1
  %90 = bitcast <2 x i32> %89 to double
  %91 = call double @region_0_1_reduce_sum_5_0(double %81, double %90)
  %92 = icmp eq i32 %9, 0
  %93 = icmp sle i32 %3, 224
  %94 = and i1 %92, %93
  %95 = mul i32 %4, 8
  %96 = add i32 %95, %5
  br i1 %94, label %97, label %99

97:                                               ; preds = %2
  %98 = getelementptr inbounds [128 x double], ptr %1, i32 0, i32 %96
  store double %91, ptr %98, align 8
  br label %99

99:                                               ; preds = %97, %2
  ret void
}

define internal double @region_0_1_reduce_sum_5_0(double %0, double %1) {
  %3 = fadd nsz double %0, %1
  ret double %3
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

define ptx_kernel void @input_reduce_fusion_1(ptr noalias align 256 dereferenceable(1024) %arg0, ptr noalias align 256 dereferenceable(8) %arg1) #3 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = addrspacecast ptr null to ptr addrspace(1)
  %4 = addrspacecast ptr null to ptr addrspace(1)
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = and i32 %5, 31
  %7 = shl i32 %6, 0
  %8 = or i32 0, %7
  %9 = and i32 %8, 31
  %10 = shl i32 %9, 1
  %11 = or disjoint i32 %10, 0
  %12 = xor i32 0, %11
  %13 = xor i32 %12, 0
  %14 = xor i32 %12, 64
  %15 = add i32 %13, 0
  %16 = add i32 %14, 0
  %17 = sext i32 %15 to i64
  %18 = sext i32 %16 to i64
  %19 = getelementptr double, ptr addrspace(1) %1, i64 %17
  %20 = getelementptr double, ptr addrspace(1) %1, i64 %18
  %21 = call { i64, i64 } asm sideeffect "mov.u64 $0, 0x0;\0A\09mov.u64 $1, 0x0;\0A\09ld.global.v2.b64 { $0, $1 }, [ $2 + 0 ];", "=l,=l,l"(ptr addrspace(1) %19)
  %22 = extractvalue { i64, i64 } %21, 0
  %23 = bitcast i64 %22 to <1 x double>
  %24 = extractvalue { i64, i64 } %21, 1
  %25 = bitcast i64 %24 to <1 x double>
  %26 = extractelement <1 x double> %23, i32 0
  %27 = extractelement <1 x double> %25, i32 0
  %28 = call { i64, i64 } asm sideeffect "mov.u64 $0, 0x0;\0A\09mov.u64 $1, 0x0;\0A\09ld.global.v2.b64 { $0, $1 }, [ $2 + 0 ];", "=l,=l,l"(ptr addrspace(1) %20)
  %29 = extractvalue { i64, i64 } %28, 0
  %30 = bitcast i64 %29 to <1 x double>
  %31 = extractvalue { i64, i64 } %28, 1
  %32 = bitcast i64 %31 to <1 x double>
  %33 = extractelement <1 x double> %30, i32 0
  %34 = extractelement <1 x double> %32, i32 0
  %35 = fadd double %26, %27
  %36 = fadd double %33, %34
  %37 = fadd double %35, %36
  %38 = bitcast double %37 to <2 x float>
  %39 = extractelement <2 x float> %38, i32 0
  %40 = extractelement <2 x float> %38, i32 1
  %41 = bitcast float %39 to i32
  %42 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %41, i32 16, i32 31)
  %43 = bitcast i32 %42 to float
  %44 = bitcast float %40 to i32
  %45 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %44, i32 16, i32 31)
  %46 = bitcast i32 %45 to float
  %47 = insertelement <2 x float> undef, float %43, i32 0
  %48 = insertelement <2 x float> %47, float %46, i32 1
  %49 = bitcast <2 x float> %48 to double
  %50 = fadd double %37, %49
  %51 = bitcast double %50 to <2 x float>
  %52 = extractelement <2 x float> %51, i32 0
  %53 = extractelement <2 x float> %51, i32 1
  %54 = bitcast float %52 to i32
  %55 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %54, i32 8, i32 31)
  %56 = bitcast i32 %55 to float
  %57 = bitcast float %53 to i32
  %58 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %57, i32 8, i32 31)
  %59 = bitcast i32 %58 to float
  %60 = insertelement <2 x float> undef, float %56, i32 0
  %61 = insertelement <2 x float> %60, float %59, i32 1
  %62 = bitcast <2 x float> %61 to double
  %63 = fadd double %50, %62
  %64 = bitcast double %63 to <2 x float>
  %65 = extractelement <2 x float> %64, i32 0
  %66 = extractelement <2 x float> %64, i32 1
  %67 = bitcast float %65 to i32
  %68 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %67, i32 4, i32 31)
  %69 = bitcast i32 %68 to float
  %70 = bitcast float %66 to i32
  %71 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %70, i32 4, i32 31)
  %72 = bitcast i32 %71 to float
  %73 = insertelement <2 x float> undef, float %69, i32 0
  %74 = insertelement <2 x float> %73, float %72, i32 1
  %75 = bitcast <2 x float> %74 to double
  %76 = fadd double %63, %75
  %77 = bitcast double %76 to <2 x float>
  %78 = extractelement <2 x float> %77, i32 0
  %79 = extractelement <2 x float> %77, i32 1
  %80 = bitcast float %78 to i32
  %81 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %80, i32 2, i32 31)
  %82 = bitcast i32 %81 to float
  %83 = bitcast float %79 to i32
  %84 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %83, i32 2, i32 31)
  %85 = bitcast i32 %84 to float
  %86 = insertelement <2 x float> undef, float %82, i32 0
  %87 = insertelement <2 x float> %86, float %85, i32 1
  %88 = bitcast <2 x float> %87 to double
  %89 = fadd double %76, %88
  %90 = bitcast double %89 to <2 x float>
  %91 = extractelement <2 x float> %90, i32 0
  %92 = extractelement <2 x float> %90, i32 1
  %93 = bitcast float %91 to i32
  %94 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %93, i32 1, i32 31)
  %95 = bitcast i32 %94 to float
  %96 = bitcast float %92 to i32
  %97 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %96, i32 1, i32 31)
  %98 = bitcast i32 %97 to float
  %99 = insertelement <2 x float> undef, float %95, i32 0
  %100 = insertelement <2 x float> %99, float %98, i32 1
  %101 = bitcast <2 x float> %100 to double
  %102 = fadd double %89, %101
  %103 = and i32 %6, -1
  %104 = icmp eq i32 %103, 0
  %105 = and i1 %104, true
  %106 = and i1 %105, true
  %107 = insertelement <1 x double> undef, double %102, i32 0
  %108 = bitcast <1 x double> %107 to i64
  call void asm sideeffect "@$2 st.global.b64 [ $1 + 0 ], { $0 };", "l,l,b"(i64 %108, ptr addrspace(1) %2, i1 %106)
  ret void
}

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #2

attributes #0 = { "nvvm.reqntid"="256,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { "nvvm.reqntid"="32,1,1" }

!llvm.module.flags = !{!0}
!nvvm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 256}
!2 = !{i32 0, i32 16}
!3 = !{}
