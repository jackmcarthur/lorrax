; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [24 x i64] undef
@shared_01 = private unnamed_addr addrspace(3) global [24 x double] undef

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_compare_not_select_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(98304) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(12288) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(98304) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(12288) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %10 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %11 = shl nuw nsw i32 %9, 7
  %12 = or disjoint i32 %11, %10
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %13
  %15 = load double, ptr addrspace(1) %14, align 8, !invariant.load !6
  %16 = tail call double @llvm.nvvm.fabs.f64(double %15)
  %17 = fcmp ueq double %16, 0x7FF0000000000000
  %18 = zext i1 %17 to i8
  %19 = select i1 %17, double 0.000000e+00, double %16
  %20 = fcmp uno double %15, 0.000000e+00
  %21 = zext i1 %20 to i8
  %22 = getelementptr inbounds i8, ptr addrspace(1) %6, i64 %13
  store i8 %18, ptr addrspace(1) %22, align 1
  %23 = getelementptr inbounds double, ptr addrspace(1) %7, i64 %13
  store double %19, ptr addrspace(1) %23, align 8
  %24 = getelementptr inbounds i8, ptr addrspace(1) %8, i64 %13
  store i8 %21, ptr addrspace(1) %24, align 1
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_reduce_fusion_2(ptr noalias readonly align 256 captures(none) dereferenceable(12288) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #2 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !7
  %6 = shl nuw nsw i32 %5, 2
  %7 = zext nneg i32 %6 to i64
  %8 = getelementptr inbounds i8, ptr addrspace(1) %3, i64 %7
  %9 = load <4 x i8>, ptr addrspace(1) %8, align 4, !invariant.load !6
  %10 = extractelement <4 x i8> %9, i32 0
  %11 = extractelement <4 x i8> %9, i32 1
  %12 = extractelement <4 x i8> %9, i32 2
  %13 = extractelement <4 x i8> %9, i32 3
  %14 = sext i8 %10 to i64
  %15 = sext i8 %11 to i64
  %16 = add nsw i64 %15, %14
  %17 = sext i8 %12 to i64
  %18 = add nsw i64 %16, %17
  %19 = sext i8 %13 to i64
  %20 = add nsw i64 %18, %19
  %21 = getelementptr inbounds i8, ptr addrspace(1) %8, i64 3072
  %22 = load <4 x i8>, ptr addrspace(1) %21, align 4, !invariant.load !6
  %23 = extractelement <4 x i8> %22, i32 0
  %24 = extractelement <4 x i8> %22, i32 1
  %25 = extractelement <4 x i8> %22, i32 2
  %26 = extractelement <4 x i8> %22, i32 3
  %27 = sext i8 %23 to i64
  %28 = add nsw i64 %20, %27
  %29 = sext i8 %24 to i64
  %30 = add nsw i64 %28, %29
  %31 = sext i8 %25 to i64
  %32 = add nsw i64 %30, %31
  %33 = sext i8 %26 to i64
  %34 = add nsw i64 %32, %33
  %35 = getelementptr inbounds i8, ptr addrspace(1) %8, i64 6144
  %36 = load <4 x i8>, ptr addrspace(1) %35, align 4, !invariant.load !6
  %37 = extractelement <4 x i8> %36, i32 0
  %38 = extractelement <4 x i8> %36, i32 1
  %39 = extractelement <4 x i8> %36, i32 2
  %40 = extractelement <4 x i8> %36, i32 3
  %41 = sext i8 %37 to i64
  %42 = add nsw i64 %34, %41
  %43 = sext i8 %38 to i64
  %44 = add nsw i64 %42, %43
  %45 = sext i8 %39 to i64
  %46 = add nsw i64 %44, %45
  %47 = sext i8 %40 to i64
  %48 = add nsw i64 %46, %47
  %49 = getelementptr inbounds i8, ptr addrspace(1) %8, i64 9216
  %50 = load <4 x i8>, ptr addrspace(1) %49, align 4, !invariant.load !6
  %51 = extractelement <4 x i8> %50, i32 0
  %52 = extractelement <4 x i8> %50, i32 1
  %53 = extractelement <4 x i8> %50, i32 2
  %54 = extractelement <4 x i8> %50, i32 3
  %55 = sext i8 %51 to i64
  %56 = add nsw i64 %48, %55
  %57 = sext i8 %52 to i64
  %58 = add nsw i64 %56, %57
  %59 = sext i8 %53 to i64
  %60 = add nsw i64 %58, %59
  %61 = sext i8 %54 to i64
  %62 = add nsw i64 %60, %61
  %63 = bitcast i64 %62 to <2 x i32>
  %64 = extractelement <2 x i32> %63, i64 0
  %65 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %64, i32 16, i32 31)
  %66 = insertelement <2 x i32> poison, i32 %65, i64 0
  %67 = extractelement <2 x i32> %63, i64 1
  %68 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %67, i32 16, i32 31)
  %69 = insertelement <2 x i32> %66, i32 %68, i64 1
  %70 = bitcast <2 x i32> %69 to i64
  %71 = add i64 %62, %70
  %72 = bitcast i64 %71 to <2 x i32>
  %73 = extractelement <2 x i32> %72, i64 0
  %74 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %73, i32 8, i32 31)
  %75 = insertelement <2 x i32> poison, i32 %74, i64 0
  %76 = extractelement <2 x i32> %72, i64 1
  %77 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %76, i32 8, i32 31)
  %78 = insertelement <2 x i32> %75, i32 %77, i64 1
  %79 = bitcast <2 x i32> %78 to i64
  %80 = add i64 %71, %79
  %81 = bitcast i64 %80 to <2 x i32>
  %82 = extractelement <2 x i32> %81, i64 0
  %83 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %82, i32 4, i32 31)
  %84 = insertelement <2 x i32> poison, i32 %83, i64 0
  %85 = extractelement <2 x i32> %81, i64 1
  %86 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %85, i32 4, i32 31)
  %87 = insertelement <2 x i32> %84, i32 %86, i64 1
  %88 = bitcast <2 x i32> %87 to i64
  %89 = add i64 %80, %88
  %90 = bitcast i64 %89 to <2 x i32>
  %91 = extractelement <2 x i32> %90, i64 0
  %92 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %91, i32 2, i32 31)
  %93 = insertelement <2 x i32> poison, i32 %92, i64 0
  %94 = extractelement <2 x i32> %90, i64 1
  %95 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %94, i32 2, i32 31)
  %96 = insertelement <2 x i32> %93, i32 %95, i64 1
  %97 = bitcast <2 x i32> %96 to i64
  %98 = add i64 %89, %97
  %99 = bitcast i64 %98 to <2 x i32>
  %100 = extractelement <2 x i32> %99, i64 0
  %101 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %100, i32 1, i32 31)
  %102 = extractelement <2 x i32> %99, i64 1
  %103 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %102, i32 1, i32 31)
  %104 = and i32 %5, 31
  %105 = icmp eq i32 %104, 0
  br i1 %105, label %106, label %114

106:                                              ; preds = %2
  %107 = lshr exact i32 %5, 5
  %108 = zext nneg i32 %107 to i64
  %109 = getelementptr inbounds i64, ptr addrspace(3) @shared_0, i64 %108
  %110 = insertelement <2 x i32> poison, i32 %101, i64 0
  %111 = insertelement <2 x i32> %110, i32 %103, i64 1
  %112 = bitcast <2 x i32> %111 to i64
  %113 = add i64 %98, %112
  store i64 %113, ptr addrspace(3) %109, align 4
  br label %114

114:                                              ; preds = %106, %2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %115 = icmp samesign ult i32 %5, 32
  br i1 %115, label %116, label %172

116:                                              ; preds = %114
  %117 = icmp samesign ult i32 %5, 24
  %118 = zext nneg i32 %5 to i64
  %119 = getelementptr inbounds i64, ptr addrspace(3) @shared_0, i64 %118
  br i1 %117, label %120, label %122

120:                                              ; preds = %116
  %121 = load i64, ptr addrspace(3) %119, align 4
  br label %122

122:                                              ; preds = %120, %116
  %123 = phi i64 [ %121, %120 ], [ 0, %116 ]
  %124 = bitcast i64 %123 to <2 x i32>
  %125 = extractelement <2 x i32> %124, i64 0
  %126 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %125, i32 16, i32 31)
  %127 = insertelement <2 x i32> poison, i32 %126, i64 0
  %128 = extractelement <2 x i32> %124, i64 1
  %129 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %128, i32 16, i32 31)
  %130 = insertelement <2 x i32> %127, i32 %129, i64 1
  %131 = bitcast <2 x i32> %130 to i64
  %132 = add i64 %123, %131
  %133 = bitcast i64 %132 to <2 x i32>
  %134 = extractelement <2 x i32> %133, i64 0
  %135 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %134, i32 8, i32 31)
  %136 = insertelement <2 x i32> poison, i32 %135, i64 0
  %137 = extractelement <2 x i32> %133, i64 1
  %138 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %137, i32 8, i32 31)
  %139 = insertelement <2 x i32> %136, i32 %138, i64 1
  %140 = bitcast <2 x i32> %139 to i64
  %141 = add i64 %132, %140
  %142 = bitcast i64 %141 to <2 x i32>
  %143 = extractelement <2 x i32> %142, i64 0
  %144 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %143, i32 4, i32 31)
  %145 = insertelement <2 x i32> poison, i32 %144, i64 0
  %146 = extractelement <2 x i32> %142, i64 1
  %147 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %146, i32 4, i32 31)
  %148 = insertelement <2 x i32> %145, i32 %147, i64 1
  %149 = bitcast <2 x i32> %148 to i64
  %150 = add i64 %141, %149
  %151 = bitcast i64 %150 to <2 x i32>
  %152 = extractelement <2 x i32> %151, i64 0
  %153 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %152, i32 2, i32 31)
  %154 = insertelement <2 x i32> poison, i32 %153, i64 0
  %155 = extractelement <2 x i32> %151, i64 1
  %156 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %155, i32 2, i32 31)
  %157 = insertelement <2 x i32> %154, i32 %156, i64 1
  %158 = bitcast <2 x i32> %157 to i64
  %159 = add i64 %150, %158
  %160 = bitcast i64 %159 to <2 x i32>
  %161 = extractelement <2 x i32> %160, i64 0
  %162 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %161, i32 1, i32 31)
  %163 = extractelement <2 x i32> %160, i64 1
  %164 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %163, i32 1, i32 31)
  %165 = icmp eq i32 %5, 0
  %166 = insertelement <2 x i32> poison, i32 %162, i64 0
  %167 = insertelement <2 x i32> %166, i32 %164, i64 1
  %168 = bitcast <2 x i32> %167 to i64
  %169 = add i64 %159, %168
  %170 = sitofp i64 %169 to double
  br i1 %165, label %171, label %172

171:                                              ; preds = %122
  store double %170, ptr addrspace(1) %4, align 256
  br label %172

172:                                              ; preds = %122, %171, %114
  ret void
}

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #3

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #4

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_reduce_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(98304) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #2 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !7
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds double, ptr addrspace(1) %3, i64 %6
  %8 = load double, ptr addrspace(1) %7, align 8, !invariant.load !6
  %9 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 6144
  %10 = load double, ptr addrspace(1) %9, align 8, !invariant.load !6
  %11 = tail call double @llvm.maximum.f64(double %8, double %10)
  %12 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 12288
  %13 = load double, ptr addrspace(1) %12, align 8, !invariant.load !6
  %14 = tail call double @llvm.maximum.f64(double %11, double %13)
  %15 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 18432
  %16 = load double, ptr addrspace(1) %15, align 8, !invariant.load !6
  %17 = tail call double @llvm.maximum.f64(double %14, double %16)
  %18 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 24576
  %19 = load double, ptr addrspace(1) %18, align 8, !invariant.load !6
  %20 = tail call double @llvm.maximum.f64(double %17, double %19)
  %21 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 30720
  %22 = load double, ptr addrspace(1) %21, align 8, !invariant.load !6
  %23 = tail call double @llvm.maximum.f64(double %20, double %22)
  %24 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 36864
  %25 = load double, ptr addrspace(1) %24, align 8, !invariant.load !6
  %26 = tail call double @llvm.maximum.f64(double %23, double %25)
  %27 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 43008
  %28 = load double, ptr addrspace(1) %27, align 8, !invariant.load !6
  %29 = tail call double @llvm.maximum.f64(double %26, double %28)
  %30 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 49152
  %31 = load double, ptr addrspace(1) %30, align 8, !invariant.load !6
  %32 = tail call double @llvm.maximum.f64(double %29, double %31)
  %33 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 55296
  %34 = load double, ptr addrspace(1) %33, align 8, !invariant.load !6
  %35 = tail call double @llvm.maximum.f64(double %32, double %34)
  %36 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 61440
  %37 = load double, ptr addrspace(1) %36, align 8, !invariant.load !6
  %38 = tail call double @llvm.maximum.f64(double %35, double %37)
  %39 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 67584
  %40 = load double, ptr addrspace(1) %39, align 8, !invariant.load !6
  %41 = tail call double @llvm.maximum.f64(double %38, double %40)
  %42 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 73728
  %43 = load double, ptr addrspace(1) %42, align 8, !invariant.load !6
  %44 = tail call double @llvm.maximum.f64(double %41, double %43)
  %45 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 79872
  %46 = load double, ptr addrspace(1) %45, align 8, !invariant.load !6
  %47 = tail call double @llvm.maximum.f64(double %44, double %46)
  %48 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 86016
  %49 = load double, ptr addrspace(1) %48, align 8, !invariant.load !6
  %50 = tail call double @llvm.maximum.f64(double %47, double %49)
  %51 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 92160
  %52 = load double, ptr addrspace(1) %51, align 8, !invariant.load !6
  %53 = tail call double @llvm.maximum.f64(double %50, double %52)
  %54 = bitcast double %53 to <2 x i32>
  %55 = extractelement <2 x i32> %54, i64 0
  %56 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %55, i32 16, i32 31)
  %57 = insertelement <2 x i32> poison, i32 %56, i64 0
  %58 = extractelement <2 x i32> %54, i64 1
  %59 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %58, i32 16, i32 31)
  %60 = insertelement <2 x i32> %57, i32 %59, i64 1
  %61 = bitcast <2 x i32> %60 to double
  %62 = tail call double @llvm.maximum.f64(double %53, double %61)
  %63 = bitcast double %62 to <2 x i32>
  %64 = extractelement <2 x i32> %63, i64 0
  %65 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %64, i32 8, i32 31)
  %66 = insertelement <2 x i32> poison, i32 %65, i64 0
  %67 = extractelement <2 x i32> %63, i64 1
  %68 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %67, i32 8, i32 31)
  %69 = insertelement <2 x i32> %66, i32 %68, i64 1
  %70 = bitcast <2 x i32> %69 to double
  %71 = tail call double @llvm.maximum.f64(double %62, double %70)
  %72 = bitcast double %71 to <2 x i32>
  %73 = extractelement <2 x i32> %72, i64 0
  %74 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %73, i32 4, i32 31)
  %75 = insertelement <2 x i32> poison, i32 %74, i64 0
  %76 = extractelement <2 x i32> %72, i64 1
  %77 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %76, i32 4, i32 31)
  %78 = insertelement <2 x i32> %75, i32 %77, i64 1
  %79 = bitcast <2 x i32> %78 to double
  %80 = tail call double @llvm.maximum.f64(double %71, double %79)
  %81 = bitcast double %80 to <2 x i32>
  %82 = extractelement <2 x i32> %81, i64 0
  %83 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %82, i32 2, i32 31)
  %84 = insertelement <2 x i32> poison, i32 %83, i64 0
  %85 = extractelement <2 x i32> %81, i64 1
  %86 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %85, i32 2, i32 31)
  %87 = insertelement <2 x i32> %84, i32 %86, i64 1
  %88 = bitcast <2 x i32> %87 to double
  %89 = tail call double @llvm.maximum.f64(double %80, double %88)
  %90 = bitcast double %89 to <2 x i32>
  %91 = extractelement <2 x i32> %90, i64 0
  %92 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %91, i32 1, i32 31)
  %93 = extractelement <2 x i32> %90, i64 1
  %94 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %93, i32 1, i32 31)
  %95 = and i32 %5, 31
  %96 = icmp eq i32 %95, 0
  br i1 %96, label %97, label %106

97:                                               ; preds = %2
  %98 = trunc i64 %6 to i32
  %99 = lshr exact i32 %98, 5
  %100 = zext nneg i32 %99 to i64
  %101 = getelementptr inbounds double, ptr addrspace(3) @shared_01, i64 %100
  %102 = insertelement <2 x i32> poison, i32 %92, i64 0
  %103 = insertelement <2 x i32> %102, i32 %94, i64 1
  %104 = bitcast <2 x i32> %103 to double
  %105 = tail call double @llvm.maximum.f64(double %89, double %104)
  store double %105, ptr addrspace(3) %101, align 8
  br label %106

106:                                              ; preds = %97, %2
  %107 = trunc i64 %6 to i32
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %108 = icmp samesign ult i32 %107, 32
  br i1 %108, label %109, label %165

109:                                              ; preds = %106
  %110 = trunc i64 %6 to i32
  %111 = icmp samesign ult i32 %110, 24
  %112 = getelementptr inbounds double, ptr addrspace(3) @shared_01, i64 %6
  br i1 %111, label %113, label %115

113:                                              ; preds = %109
  %114 = load double, ptr addrspace(3) %112, align 8
  br label %115

115:                                              ; preds = %113, %109
  %116 = phi double [ %114, %113 ], [ 0xFFF0000000000000, %109 ]
  %117 = trunc i64 %6 to i32
  %118 = bitcast double %116 to <2 x i32>
  %119 = extractelement <2 x i32> %118, i64 0
  %120 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %119, i32 16, i32 31)
  %121 = insertelement <2 x i32> poison, i32 %120, i64 0
  %122 = extractelement <2 x i32> %118, i64 1
  %123 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %122, i32 16, i32 31)
  %124 = insertelement <2 x i32> %121, i32 %123, i64 1
  %125 = bitcast <2 x i32> %124 to double
  %126 = tail call double @llvm.maximum.f64(double %116, double %125)
  %127 = bitcast double %126 to <2 x i32>
  %128 = extractelement <2 x i32> %127, i64 0
  %129 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %128, i32 8, i32 31)
  %130 = insertelement <2 x i32> poison, i32 %129, i64 0
  %131 = extractelement <2 x i32> %127, i64 1
  %132 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %131, i32 8, i32 31)
  %133 = insertelement <2 x i32> %130, i32 %132, i64 1
  %134 = bitcast <2 x i32> %133 to double
  %135 = tail call double @llvm.maximum.f64(double %126, double %134)
  %136 = bitcast double %135 to <2 x i32>
  %137 = extractelement <2 x i32> %136, i64 0
  %138 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %137, i32 4, i32 31)
  %139 = insertelement <2 x i32> poison, i32 %138, i64 0
  %140 = extractelement <2 x i32> %136, i64 1
  %141 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %140, i32 4, i32 31)
  %142 = insertelement <2 x i32> %139, i32 %141, i64 1
  %143 = bitcast <2 x i32> %142 to double
  %144 = tail call double @llvm.maximum.f64(double %135, double %143)
  %145 = bitcast double %144 to <2 x i32>
  %146 = extractelement <2 x i32> %145, i64 0
  %147 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %146, i32 2, i32 31)
  %148 = insertelement <2 x i32> poison, i32 %147, i64 0
  %149 = extractelement <2 x i32> %145, i64 1
  %150 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %149, i32 2, i32 31)
  %151 = insertelement <2 x i32> %148, i32 %150, i64 1
  %152 = bitcast <2 x i32> %151 to double
  %153 = tail call double @llvm.maximum.f64(double %144, double %152)
  %154 = bitcast double %153 to <2 x i32>
  %155 = extractelement <2 x i32> %154, i64 0
  %156 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %155, i32 1, i32 31)
  %157 = extractelement <2 x i32> %154, i64 1
  %158 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %157, i32 1, i32 31)
  %159 = icmp eq i32 %117, 0
  br i1 %159, label %160, label %165

160:                                              ; preds = %115
  %161 = insertelement <2 x i32> poison, i32 %156, i64 0
  %162 = insertelement <2 x i32> %161, i32 %158, i64 1
  %163 = bitcast <2 x i32> %162 to double
  %164 = tail call double @llvm.maximum.f64(double %153, double %163)
  store double %164, ptr addrspace(1) %4, align 256
  br label %165

165:                                              ; preds = %115, %160, %106
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #5

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @input_concatenate_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(24) initializes((0, 24)) %3) local_unnamed_addr #6 {
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %3 to ptr addrspace(1)
  %7 = addrspacecast ptr %1 to ptr addrspace(1)
  %8 = addrspacecast ptr %0 to ptr addrspace(1)
  %9 = load double, ptr addrspace(1) %5, align 256, !invariant.load !6
  %10 = load double, ptr addrspace(1) %7, align 256, !invariant.load !6
  %11 = insertelement <2 x double> poison, double %9, i32 0
  %12 = insertelement <2 x double> %11, double %10, i32 1
  store <2 x double> %12, ptr addrspace(1) %6, align 256
  %13 = load double, ptr addrspace(1) %8, align 256, !invariant.load !6
  %14 = getelementptr inbounds i8, ptr addrspace(1) %6, i64 16
  store double %13, ptr addrspace(1) %14, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #5

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { norecurse nounwind "nvvm.reqntid"="768,1,1" }
attributes #3 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #4 = { convergent nocallback nounwind }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 96}
!5 = !{i32 0, i32 128}
!6 = !{}
!7 = !{i32 0, i32 768}
