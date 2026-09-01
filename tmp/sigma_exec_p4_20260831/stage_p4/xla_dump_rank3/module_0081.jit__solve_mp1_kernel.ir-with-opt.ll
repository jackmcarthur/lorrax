; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_36_0 = local_unnamed_addr addrspace(1) constant [64 x i8] zeroinitializer, align 256

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion_2(ptr noalias readonly align 16 captures(none) dereferenceable(196608) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(1024) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(1024) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %9 = lshr i32 %7, 5
  %10 = mul nuw nsw i32 %9, 192
  %11 = mul nuw nsw i32 %8, 1536
  %12 = add nuw nsw i32 %10, %11
  %13 = and i32 %7, 31
  %14 = or disjoint i32 %12, %13
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %15
  %17 = load double, ptr addrspace(1) %16, align 8, !invariant.load !6
  %18 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 256
  %19 = load double, ptr addrspace(1) %18, align 8, !invariant.load !6
  %20 = tail call double @llvm.maximum.f64(double %17, double %19)
  %21 = tail call double @llvm.minimum.f64(double %17, double %19)
  %22 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 512
  %23 = load double, ptr addrspace(1) %22, align 8, !invariant.load !6
  %24 = tail call double @llvm.maximum.f64(double %20, double %23)
  %25 = tail call double @llvm.minimum.f64(double %21, double %23)
  %26 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 768
  %27 = load double, ptr addrspace(1) %26, align 8, !invariant.load !6
  %28 = tail call double @llvm.maximum.f64(double %24, double %27)
  %29 = tail call double @llvm.minimum.f64(double %25, double %27)
  %30 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 1024
  %31 = load double, ptr addrspace(1) %30, align 8, !invariant.load !6
  %32 = tail call double @llvm.maximum.f64(double %28, double %31)
  %33 = tail call double @llvm.minimum.f64(double %29, double %31)
  %34 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 1280
  %35 = load double, ptr addrspace(1) %34, align 8, !invariant.load !6
  %36 = tail call double @llvm.maximum.f64(double %32, double %35)
  %37 = tail call double @llvm.minimum.f64(double %33, double %35)
  %38 = bitcast double %36 to <2 x i32>
  %39 = extractelement <2 x i32> %38, i64 0
  %40 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %39, i32 16, i32 31)
  %41 = insertelement <2 x i32> poison, i32 %40, i64 0
  %42 = extractelement <2 x i32> %38, i64 1
  %43 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %42, i32 16, i32 31)
  %44 = insertelement <2 x i32> %41, i32 %43, i64 1
  %45 = bitcast <2 x i32> %44 to double
  %46 = tail call double @llvm.maximum.f64(double %36, double %45)
  %47 = bitcast double %46 to <2 x i32>
  %48 = extractelement <2 x i32> %47, i64 0
  %49 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %48, i32 8, i32 31)
  %50 = insertelement <2 x i32> poison, i32 %49, i64 0
  %51 = extractelement <2 x i32> %47, i64 1
  %52 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %51, i32 8, i32 31)
  %53 = insertelement <2 x i32> %50, i32 %52, i64 1
  %54 = bitcast <2 x i32> %53 to double
  %55 = tail call double @llvm.maximum.f64(double %46, double %54)
  %56 = bitcast double %55 to <2 x i32>
  %57 = extractelement <2 x i32> %56, i64 0
  %58 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %57, i32 4, i32 31)
  %59 = insertelement <2 x i32> poison, i32 %58, i64 0
  %60 = extractelement <2 x i32> %56, i64 1
  %61 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %60, i32 4, i32 31)
  %62 = insertelement <2 x i32> %59, i32 %61, i64 1
  %63 = bitcast <2 x i32> %62 to double
  %64 = tail call double @llvm.maximum.f64(double %55, double %63)
  %65 = bitcast double %64 to <2 x i32>
  %66 = extractelement <2 x i32> %65, i64 0
  %67 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %66, i32 2, i32 31)
  %68 = insertelement <2 x i32> poison, i32 %67, i64 0
  %69 = extractelement <2 x i32> %65, i64 1
  %70 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %69, i32 2, i32 31)
  %71 = insertelement <2 x i32> %68, i32 %70, i64 1
  %72 = bitcast <2 x i32> %71 to double
  %73 = tail call double @llvm.maximum.f64(double %64, double %72)
  %74 = bitcast double %73 to <2 x i32>
  %75 = extractelement <2 x i32> %74, i64 0
  %76 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %75, i32 1, i32 31)
  %77 = extractelement <2 x i32> %74, i64 1
  %78 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %77, i32 1, i32 31)
  %79 = bitcast double %37 to <2 x i32>
  %80 = extractelement <2 x i32> %79, i64 0
  %81 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %80, i32 16, i32 31)
  %82 = insertelement <2 x i32> poison, i32 %81, i64 0
  %83 = extractelement <2 x i32> %79, i64 1
  %84 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %83, i32 16, i32 31)
  %85 = insertelement <2 x i32> %82, i32 %84, i64 1
  %86 = bitcast <2 x i32> %85 to double
  %87 = tail call double @llvm.minimum.f64(double %37, double %86)
  %88 = bitcast double %87 to <2 x i32>
  %89 = extractelement <2 x i32> %88, i64 0
  %90 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %89, i32 8, i32 31)
  %91 = insertelement <2 x i32> poison, i32 %90, i64 0
  %92 = extractelement <2 x i32> %88, i64 1
  %93 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %92, i32 8, i32 31)
  %94 = insertelement <2 x i32> %91, i32 %93, i64 1
  %95 = bitcast <2 x i32> %94 to double
  %96 = tail call double @llvm.minimum.f64(double %87, double %95)
  %97 = bitcast double %96 to <2 x i32>
  %98 = extractelement <2 x i32> %97, i64 0
  %99 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %98, i32 4, i32 31)
  %100 = insertelement <2 x i32> poison, i32 %99, i64 0
  %101 = extractelement <2 x i32> %97, i64 1
  %102 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %101, i32 4, i32 31)
  %103 = insertelement <2 x i32> %100, i32 %102, i64 1
  %104 = bitcast <2 x i32> %103 to double
  %105 = tail call double @llvm.minimum.f64(double %96, double %104)
  %106 = bitcast double %105 to <2 x i32>
  %107 = extractelement <2 x i32> %106, i64 0
  %108 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %107, i32 2, i32 31)
  %109 = insertelement <2 x i32> poison, i32 %108, i64 0
  %110 = extractelement <2 x i32> %106, i64 1
  %111 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %110, i32 2, i32 31)
  %112 = insertelement <2 x i32> %109, i32 %111, i64 1
  %113 = bitcast <2 x i32> %112 to double
  %114 = tail call double @llvm.minimum.f64(double %105, double %113)
  %115 = bitcast double %114 to <2 x i32>
  %116 = extractelement <2 x i32> %115, i64 0
  %117 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %116, i32 1, i32 31)
  %118 = extractelement <2 x i32> %115, i64 1
  %119 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %118, i32 1, i32 31)
  %120 = icmp eq i32 %13, 0
  %121 = icmp samesign ult i32 %7, 225
  %122 = and i1 %121, %120
  br i1 %122, label %123, label %137

123:                                              ; preds = %3
  %124 = shl nuw nsw i32 %8, 3
  %125 = or disjoint i32 %124, %9
  %126 = zext nneg i32 %125 to i64
  %127 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %126
  %128 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %126
  %129 = insertelement <2 x i32> poison, i32 %117, i64 0
  %130 = insertelement <2 x i32> %129, i32 %119, i64 1
  %131 = bitcast <2 x i32> %130 to double
  %132 = tail call double @llvm.minimum.f64(double %114, double %131)
  %133 = insertelement <2 x i32> poison, i32 %76, i64 0
  %134 = insertelement <2 x i32> %133, i32 %78, i64 1
  %135 = bitcast <2 x i32> %134 to double
  %136 = tail call double @llvm.maximum.f64(double %73, double %135)
  store double %136, ptr addrspace(1) %128, align 8
  store double %132, ptr addrspace(1) %127, align 8
  br label %137

137:                                              ; preds = %123, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #3

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion_3(ptr noalias readonly align 256 captures(none) dereferenceable(1024) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #4 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !7
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds double, ptr addrspace(1) %3, i64 %6
  %8 = load double, ptr addrspace(1) %7, align 8, !invariant.load !6
  %9 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 256
  %10 = load double, ptr addrspace(1) %9, align 8, !invariant.load !6
  %11 = tail call double @llvm.minimum.f64(double %8, double %10)
  %12 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 512
  %13 = load double, ptr addrspace(1) %12, align 8, !invariant.load !6
  %14 = tail call double @llvm.minimum.f64(double %11, double %13)
  %15 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 768
  %16 = load double, ptr addrspace(1) %15, align 8, !invariant.load !6
  %17 = tail call double @llvm.minimum.f64(double %14, double %16)
  %18 = bitcast double %17 to <2 x i32>
  %19 = extractelement <2 x i32> %18, i64 0
  %20 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %19, i32 16, i32 31)
  %21 = insertelement <2 x i32> poison, i32 %20, i64 0
  %22 = extractelement <2 x i32> %18, i64 1
  %23 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %22, i32 16, i32 31)
  %24 = insertelement <2 x i32> %21, i32 %23, i64 1
  %25 = bitcast <2 x i32> %24 to double
  %26 = tail call double @llvm.minimum.f64(double %17, double %25)
  %27 = bitcast double %26 to <2 x i32>
  %28 = extractelement <2 x i32> %27, i64 0
  %29 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %28, i32 8, i32 31)
  %30 = insertelement <2 x i32> poison, i32 %29, i64 0
  %31 = extractelement <2 x i32> %27, i64 1
  %32 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %31, i32 8, i32 31)
  %33 = insertelement <2 x i32> %30, i32 %32, i64 1
  %34 = bitcast <2 x i32> %33 to double
  %35 = tail call double @llvm.minimum.f64(double %26, double %34)
  %36 = bitcast double %35 to <2 x i32>
  %37 = extractelement <2 x i32> %36, i64 0
  %38 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %37, i32 4, i32 31)
  %39 = insertelement <2 x i32> poison, i32 %38, i64 0
  %40 = extractelement <2 x i32> %36, i64 1
  %41 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %40, i32 4, i32 31)
  %42 = insertelement <2 x i32> %39, i32 %41, i64 1
  %43 = bitcast <2 x i32> %42 to double
  %44 = tail call double @llvm.minimum.f64(double %35, double %43)
  %45 = bitcast double %44 to <2 x i32>
  %46 = extractelement <2 x i32> %45, i64 0
  %47 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %46, i32 2, i32 31)
  %48 = insertelement <2 x i32> poison, i32 %47, i64 0
  %49 = extractelement <2 x i32> %45, i64 1
  %50 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %49, i32 2, i32 31)
  %51 = insertelement <2 x i32> %48, i32 %50, i64 1
  %52 = bitcast <2 x i32> %51 to double
  %53 = tail call double @llvm.minimum.f64(double %44, double %52)
  %54 = bitcast double %53 to <2 x i32>
  %55 = extractelement <2 x i32> %54, i64 0
  %56 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %55, i32 1, i32 31)
  %57 = extractelement <2 x i32> %54, i64 1
  %58 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %57, i32 1, i32 31)
  %59 = icmp eq i32 %5, 0
  br i1 %59, label %60, label %65

60:                                               ; preds = %2
  %61 = insertelement <2 x i32> poison, i32 %56, i64 0
  %62 = insertelement <2 x i32> %61, i32 %58, i64 1
  %63 = bitcast <2 x i32> %62 to double
  %64 = tail call double @llvm.minimum.f64(double %53, double %63)
  store double %64, ptr addrspace(1) %4, align 256
  br label %65

65:                                               ; preds = %60, %2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_subtract_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %2) local_unnamed_addr #5 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = load double, ptr addrspace(1) %4, align 16, !invariant.load !6
  %8 = load double, ptr addrspace(1) %5, align 256, !invariant.load !6
  %9 = fmul double %7, 1.600000e+01
  %10 = fsub double %8, %9
  store double %10, ptr addrspace(1) %6, align 256
  ret void
}

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion_4(ptr noalias readonly align 256 captures(none) dereferenceable(1024) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #4 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !7
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds double, ptr addrspace(1) %3, i64 %6
  %8 = load double, ptr addrspace(1) %7, align 8, !invariant.load !6
  %9 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 256
  %10 = load double, ptr addrspace(1) %9, align 8, !invariant.load !6
  %11 = tail call double @llvm.maximum.f64(double %8, double %10)
  %12 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 512
  %13 = load double, ptr addrspace(1) %12, align 8, !invariant.load !6
  %14 = tail call double @llvm.maximum.f64(double %11, double %13)
  %15 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 768
  %16 = load double, ptr addrspace(1) %15, align 8, !invariant.load !6
  %17 = tail call double @llvm.maximum.f64(double %14, double %16)
  %18 = bitcast double %17 to <2 x i32>
  %19 = extractelement <2 x i32> %18, i64 0
  %20 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %19, i32 16, i32 31)
  %21 = insertelement <2 x i32> poison, i32 %20, i64 0
  %22 = extractelement <2 x i32> %18, i64 1
  %23 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %22, i32 16, i32 31)
  %24 = insertelement <2 x i32> %21, i32 %23, i64 1
  %25 = bitcast <2 x i32> %24 to double
  %26 = tail call double @llvm.maximum.f64(double %17, double %25)
  %27 = bitcast double %26 to <2 x i32>
  %28 = extractelement <2 x i32> %27, i64 0
  %29 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %28, i32 8, i32 31)
  %30 = insertelement <2 x i32> poison, i32 %29, i64 0
  %31 = extractelement <2 x i32> %27, i64 1
  %32 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %31, i32 8, i32 31)
  %33 = insertelement <2 x i32> %30, i32 %32, i64 1
  %34 = bitcast <2 x i32> %33 to double
  %35 = tail call double @llvm.maximum.f64(double %26, double %34)
  %36 = bitcast double %35 to <2 x i32>
  %37 = extractelement <2 x i32> %36, i64 0
  %38 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %37, i32 4, i32 31)
  %39 = insertelement <2 x i32> poison, i32 %38, i64 0
  %40 = extractelement <2 x i32> %36, i64 1
  %41 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %40, i32 4, i32 31)
  %42 = insertelement <2 x i32> %39, i32 %41, i64 1
  %43 = bitcast <2 x i32> %42 to double
  %44 = tail call double @llvm.maximum.f64(double %35, double %43)
  %45 = bitcast double %44 to <2 x i32>
  %46 = extractelement <2 x i32> %45, i64 0
  %47 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %46, i32 2, i32 31)
  %48 = insertelement <2 x i32> poison, i32 %47, i64 0
  %49 = extractelement <2 x i32> %45, i64 1
  %50 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %49, i32 2, i32 31)
  %51 = insertelement <2 x i32> %48, i32 %50, i64 1
  %52 = bitcast <2 x i32> %51 to double
  %53 = tail call double @llvm.maximum.f64(double %44, double %52)
  %54 = bitcast double %53 to <2 x i32>
  %55 = extractelement <2 x i32> %54, i64 0
  %56 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %55, i32 1, i32 31)
  %57 = extractelement <2 x i32> %54, i64 1
  %58 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %57, i32 1, i32 31)
  %59 = icmp eq i32 %5, 0
  br i1 %59, label %60, label %65

60:                                               ; preds = %2
  %61 = insertelement <2 x i32> poison, i32 %56, i64 0
  %62 = insertelement <2 x i32> %61, i32 %58, i64 1
  %63 = bitcast <2 x i32> %62 to double
  %64 = tail call double @llvm.maximum.f64(double %53, double %63)
  store double %64, ptr addrspace(1) %4, align 256
  br label %65

65:                                               ; preds = %60, %2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_add_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %2) local_unnamed_addr #5 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = load double, ptr addrspace(1) %4, align 16, !invariant.load !6
  %8 = load double, ptr addrspace(1) %5, align 256, !invariant.load !6
  %9 = fmul double %7, 1.600000e+01
  %10 = fadd double %8, %9
  store double %10, ptr addrspace(1) %6, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_compare_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(1) initializes((0, 1)) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = load i64, ptr addrspace(1) %3, align 256, !invariant.load !6
  %6 = icmp slt i64 %5, 64
  %7 = zext i1 %6 to i8
  store i8 %7, ptr addrspace(1) %4, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_add_fusion(ptr noalias align 256 captures(none) dereferenceable(8) %0, ptr noalias readnone align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = load i64, ptr addrspace(1) %3, align 256
  %5 = add i64 %4, 1
  store i64 %5, ptr addrspace(1) %3, align 256
  ret void
}

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion_1(ptr noalias readonly align 16 captures(none) dereferenceable(8) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias readonly align 16 captures(none) dereferenceable(196608) %2, ptr noalias readonly align 256 captures(none) dereferenceable(8) %3, ptr noalias readonly align 256 captures(none) dereferenceable(8) %4, ptr noalias writeonly align 256 captures(none) dereferenceable(4096) %5) local_unnamed_addr #0 {
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %4 to ptr addrspace(1)
  %10 = addrspacecast ptr %1 to ptr addrspace(1)
  %11 = addrspacecast ptr %0 to ptr addrspace(1)
  %12 = addrspacecast ptr %5 to ptr addrspace(1)
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %14 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !8
  %15 = lshr i32 %13, 5
  %16 = mul nuw nsw i32 %15, 48
  %17 = mul nuw nsw i32 %14, 384
  %18 = and i32 %13, 31
  %19 = or disjoint i32 %18, %17
  %20 = add nuw nsw i32 %19, %16
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds double, ptr addrspace(1) %7, i64 %21
  %23 = load double, ptr addrspace(1) %22, align 8, !invariant.load !6
  %24 = load double, ptr addrspace(1) %8, align 256, !invariant.load !6
  %25 = load double, ptr addrspace(1) %9, align 256, !invariant.load !6
  %26 = fadd double %24, %25
  %27 = fmul double %26, 5.000000e-01
  %28 = fsub double %23, %27
  %29 = load double, ptr addrspace(1) %10, align 16, !invariant.load !6
  %30 = fmul double %29, 2.000000e+00
  %31 = fdiv double %28, %30
  %32 = fneg double %31
  %33 = fmul double %31, %32
  %34 = tail call double @llvm.fma.f64(double %33, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %35 = tail call i32 @llvm.nvvm.d2i.lo(double %34) #9
  %36 = tail call double @llvm.nvvm.add.rn.d(double %34, double 0xC338000000000000) #9
  %37 = tail call double @llvm.fma.f64(double %36, double 0xBFE62E42FEFA39EF, double %33)
  %38 = tail call double @llvm.fma.f64(double %36, double 0xBC7ABC9E3B39803F, double %37)
  %39 = tail call double @llvm.fma.f64(double %38, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %40 = tail call double @llvm.fma.f64(double %39, double %38, double 0x3EC71DEE62401315)
  %41 = tail call double @llvm.fma.f64(double %40, double %38, double 0x3EFA01997C89EB71)
  %42 = tail call double @llvm.fma.f64(double %41, double %38, double 0x3F2A01A014761F65)
  %43 = tail call double @llvm.fma.f64(double %42, double %38, double 0x3F56C16C1852B7AF)
  %44 = tail call double @llvm.fma.f64(double %43, double %38, double 0x3F81111111122322)
  %45 = tail call double @llvm.fma.f64(double %44, double %38, double 0x3FA55555555502A1)
  %46 = tail call double @llvm.fma.f64(double %45, double %38, double 0x3FC5555555555511)
  %47 = tail call double @llvm.fma.f64(double %46, double %38, double 0x3FE000000000000B)
  %48 = tail call double @llvm.fma.f64(double %47, double %38, double 1.000000e+00)
  %49 = tail call double @llvm.fma.f64(double %48, double %38, double 1.000000e+00)
  %50 = tail call i32 @llvm.nvvm.d2i.lo(double %49) #9
  %51 = tail call i32 @llvm.nvvm.d2i.hi(double %49) #9
  %52 = shl i32 %35, 20
  %53 = add i32 %51, %52
  %54 = tail call double @llvm.nvvm.lohi.i2d(i32 %50, i32 %53) #9
  %55 = tail call i32 @llvm.nvvm.d2i.hi(double %33) #9
  %56 = bitcast i32 %55 to float
  %57 = tail call float @llvm.nvvm.fabs.f32(float %56)
  %58 = fcmp olt float %57, 0x4010C46560000000
  br i1 %58, label %__nv_exp.exit, label %__internal_fast_icmp_abs_lt.exit.i

__internal_fast_icmp_abs_lt.exit.i:               ; preds = %6
  %59 = fcmp olt double %33, 0.000000e+00
  %60 = fadd double %33, 0x7FF0000000000000
  %z.0.i = select i1 %59, double 0.000000e+00, double %60
  %61 = fcmp olt float %57, 0x4010E90000000000
  br i1 %61, label %62, label %__nv_exp.exit

62:                                               ; preds = %__internal_fast_icmp_abs_lt.exit.i
  %63 = sdiv i32 %35, 2
  %64 = shl i32 %63, 20
  %65 = add i32 %51, %64
  %66 = tail call double @llvm.nvvm.lohi.i2d(i32 %50, i32 %65) #9
  %67 = sub nsw i32 %35, %63
  %68 = shl i32 %67, 20
  %69 = add nsw i32 %68, 1072693248
  %70 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %69) #9
  %71 = fmul double %70, %66
  br label %__nv_exp.exit

__nv_exp.exit:                                    ; preds = %6, %__internal_fast_icmp_abs_lt.exit.i, %62
  %z.2.i = phi double [ %54, %6 ], [ %71, %62 ], [ %z.0.i, %__internal_fast_icmp_abs_lt.exit.i ]
  %72 = tail call double @llvm.nvvm.fabs.f64(double %31)
  %73 = tail call double @llvm.fma.f64(double %72, double 0xBCF0679AFBA6F279, double 0x3D47088FDB46FA5F)
  %74 = tail call double @llvm.fma.f64(double %73, double %72, double 0xBD8DF9F9B976A9B2)
  %75 = tail call double @llvm.fma.f64(double %74, double %72, double 0x3DC7F1F5590CC332)
  %76 = tail call double @llvm.fma.f64(double %75, double %72, double 0xBDFA28A3CD2D56C4)
  %77 = tail call double @llvm.fma.f64(double %76, double %72, double 0x3E2485EE67835925)
  %78 = tail call double @llvm.fma.f64(double %77, double %72, double 0xBE476DB45919F583)
  %79 = tail call double @llvm.fma.f64(double %78, double %72, double 0x3E62D698D98C8D71)
  %80 = tail call double @llvm.fma.f64(double %79, double %72, double 0xBE720A2C7155D5C6)
  %81 = tail call double @llvm.fma.f64(double %80, double %72, double 0xBE41D29B37CA1397)
  %82 = tail call double @llvm.fma.f64(double %81, double %72, double 0x3EA2EF6CC0F67A49)
  %83 = tail call double @llvm.fma.f64(double %82, double %72, double 0xBEC102B892333B6F)
  %84 = tail call double @llvm.fma.f64(double %83, double %72, double 0x3ECA30375BA9A84E)
  %85 = tail call double @llvm.fma.f64(double %84, double %72, double 0x3ECAAD18DEDEA43E)
  %86 = tail call double @llvm.fma.f64(double %85, double %72, double 0xBEFF05355BC5B225)
  %87 = tail call double @llvm.fma.f64(double %86, double %72, double 0x3F10E37A3108BC8B)
  %88 = tail call double @llvm.fma.f64(double %87, double %72, double 0x3EFB292D828E5CB2)
  %89 = tail call double @llvm.fma.f64(double %88, double %72, double 0xBF4356626EBF9BFA)
  %90 = tail call double @llvm.fma.f64(double %89, double %72, double 0x3F5BCA68F73D6AFC)
  %91 = tail call double @llvm.fma.f64(double %90, double %72, double 0xBF2B6B69EBBC280B)
  %92 = tail call double @llvm.fma.f64(double %91, double %72, double 0xBF9396685912A453)
  %93 = tail call double @llvm.fma.f64(double %92, double %72, double 0x3FBA4F4E2A1ABEF8)
  %94 = tail call double @llvm.fma.f64(double %93, double %72, double 0x3FE45F306DC9C8BB)
  %95 = tail call double @llvm.fma.f64(double %94, double %72, double 0x3FC06EBA8214DB69)
  %96 = tail call double @llvm.fma.f64(double %95, double %72, double %72)
  %97 = fsub double %72, %96
  %98 = tail call double @llvm.fma.f64(double %95, double %72, double %97)
  %99 = fneg double %96
  %100 = fneg double %98
  %101 = fptrunc double %99 to float
  %102 = fmul float %101, 0x3FF7154760000000
  %103 = tail call float @llvm.nvvm.round.f(float %102) #9
  %104 = fpext float %103 to double
  %105 = fneg double %104
  %106 = tail call double @llvm.fma.f64(double %105, double 0x3FE62E42FEFA39EF, double %99)
  %107 = tail call double @llvm.fma.f64(double %106, double 0x3E5AE904A4741B81, double 0x3E928A27F89B6999)
  %108 = tail call double @llvm.fma.f64(double %107, double %106, double 0x3EC71DE715FF7E07)
  %109 = tail call double @llvm.fma.f64(double %108, double %106, double 0x3EFA019A6B0AC45A)
  %110 = tail call double @llvm.fma.f64(double %109, double %106, double 0x3F2A01A017EED94F)
  %111 = tail call double @llvm.fma.f64(double %110, double %106, double 0x3F56C16C17F2A71B)
  %112 = tail call double @llvm.fma.f64(double %111, double %106, double 0x3F811111111173C4)
  %113 = tail call double @llvm.fma.f64(double %112, double %106, double 0x3FA555555555211A)
  %114 = tail call double @llvm.fma.f64(double %113, double %106, double 0x3FC5555555555540)
  %115 = tail call double @llvm.fma.f64(double %114, double %106, double 0x3FE0000000000005)
  %116 = fmul double %106, %115
  %117 = tail call double @llvm.fma.f64(double %116, double %106, double %100)
  %118 = fadd double %106, %117
  %119 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %103)
  %120 = fpext float %119 to double
  %121 = fsub double 1.000000e+00, %120
  %122 = fneg double %118
  %123 = tail call double @llvm.fma.f64(double %122, double %120, double %121)
  %124 = fcmp oge double %72, 0x4017AFB48DC96626
  %poly.0.i = select i1 %124, double 1.000000e+00, double %123
  %125 = tail call double @llvm.copysign.f64(double %poly.0.i, double %31) #9
  %126 = fcmp one double %72, 0x7FF0000000000000
  %127 = fmul double %31, %z.2.i
  %128 = fsub double 1.000000e+00, %125
  %129 = fmul double %128, 5.000000e-01
  %130 = fmul double %127, 0x3FD20DD750429B6D
  %131 = select i1 %126, double %130, double 0.000000e+00
  %132 = fsub double %129, %131
  %133 = tail call double @llvm.nvvm.fabs.f64(double %132)
  %134 = load double, ptr addrspace(1) %11, align 16, !invariant.load !6
  %135 = fcmp olt double %133, %134
  %136 = select i1 %135, double 0.000000e+00, double %132
  %137 = fsub double 1.000000e+00, %136
  %138 = tail call double @llvm.nvvm.fabs.f64(double %137)
  %139 = fcmp olt double %138, %134
  %140 = select i1 %139, double 1.000000e+00, double %136
  %141 = icmp samesign ult i32 %18, 16
  br i1 %141, label %142, label %256

142:                                              ; preds = %__nv_exp.exit
  %143 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 256
  %144 = load double, ptr addrspace(1) %143, align 8, !invariant.load !6
  %145 = fsub double %144, %27
  %146 = fdiv double %145, %30
  %147 = fneg double %146
  %148 = fmul double %146, %147
  %149 = tail call double @llvm.fma.f64(double %148, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %150 = tail call i32 @llvm.nvvm.d2i.lo(double %149) #9
  %151 = tail call double @llvm.nvvm.add.rn.d(double %149, double 0xC338000000000000) #9
  %152 = tail call double @llvm.fma.f64(double %151, double 0xBFE62E42FEFA39EF, double %148)
  %153 = tail call double @llvm.fma.f64(double %151, double 0xBC7ABC9E3B39803F, double %152)
  %154 = tail call double @llvm.fma.f64(double %153, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %155 = tail call double @llvm.fma.f64(double %154, double %153, double 0x3EC71DEE62401315)
  %156 = tail call double @llvm.fma.f64(double %155, double %153, double 0x3EFA01997C89EB71)
  %157 = tail call double @llvm.fma.f64(double %156, double %153, double 0x3F2A01A014761F65)
  %158 = tail call double @llvm.fma.f64(double %157, double %153, double 0x3F56C16C1852B7AF)
  %159 = tail call double @llvm.fma.f64(double %158, double %153, double 0x3F81111111122322)
  %160 = tail call double @llvm.fma.f64(double %159, double %153, double 0x3FA55555555502A1)
  %161 = tail call double @llvm.fma.f64(double %160, double %153, double 0x3FC5555555555511)
  %162 = tail call double @llvm.fma.f64(double %161, double %153, double 0x3FE000000000000B)
  %163 = tail call double @llvm.fma.f64(double %162, double %153, double 1.000000e+00)
  %164 = tail call double @llvm.fma.f64(double %163, double %153, double 1.000000e+00)
  %165 = tail call i32 @llvm.nvvm.d2i.lo(double %164) #9
  %166 = tail call i32 @llvm.nvvm.d2i.hi(double %164) #9
  %167 = shl i32 %150, 20
  %168 = add i32 %166, %167
  %169 = tail call double @llvm.nvvm.lohi.i2d(i32 %165, i32 %168) #9
  %170 = tail call i32 @llvm.nvvm.d2i.hi(double %148) #9
  %171 = bitcast i32 %170 to float
  %172 = tail call float @llvm.nvvm.fabs.f32(float %171)
  %173 = fcmp olt float %172, 0x4010C46560000000
  br i1 %173, label %__nv_exp.exit4, label %__internal_fast_icmp_abs_lt.exit.i1

__internal_fast_icmp_abs_lt.exit.i1:              ; preds = %142
  %174 = fcmp olt double %148, 0.000000e+00
  %175 = fadd double %148, 0x7FF0000000000000
  %z.0.i2 = select i1 %174, double 0.000000e+00, double %175
  %176 = fcmp olt float %172, 0x4010E90000000000
  br i1 %176, label %177, label %__nv_exp.exit4

177:                                              ; preds = %__internal_fast_icmp_abs_lt.exit.i1
  %178 = sdiv i32 %150, 2
  %179 = shl i32 %178, 20
  %180 = add i32 %166, %179
  %181 = tail call double @llvm.nvvm.lohi.i2d(i32 %165, i32 %180) #9
  %182 = sub nsw i32 %150, %178
  %183 = shl i32 %182, 20
  %184 = add nsw i32 %183, 1072693248
  %185 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %184) #9
  %186 = fmul double %185, %181
  br label %__nv_exp.exit4

__nv_exp.exit4:                                   ; preds = %142, %__internal_fast_icmp_abs_lt.exit.i1, %177
  %z.2.i3 = phi double [ %169, %142 ], [ %186, %177 ], [ %z.0.i2, %__internal_fast_icmp_abs_lt.exit.i1 ]
  %187 = tail call double @llvm.nvvm.fabs.f64(double %146)
  %188 = tail call double @llvm.fma.f64(double %187, double 0xBCF0679AFBA6F279, double 0x3D47088FDB46FA5F)
  %189 = tail call double @llvm.fma.f64(double %188, double %187, double 0xBD8DF9F9B976A9B2)
  %190 = tail call double @llvm.fma.f64(double %189, double %187, double 0x3DC7F1F5590CC332)
  %191 = tail call double @llvm.fma.f64(double %190, double %187, double 0xBDFA28A3CD2D56C4)
  %192 = tail call double @llvm.fma.f64(double %191, double %187, double 0x3E2485EE67835925)
  %193 = tail call double @llvm.fma.f64(double %192, double %187, double 0xBE476DB45919F583)
  %194 = tail call double @llvm.fma.f64(double %193, double %187, double 0x3E62D698D98C8D71)
  %195 = tail call double @llvm.fma.f64(double %194, double %187, double 0xBE720A2C7155D5C6)
  %196 = tail call double @llvm.fma.f64(double %195, double %187, double 0xBE41D29B37CA1397)
  %197 = tail call double @llvm.fma.f64(double %196, double %187, double 0x3EA2EF6CC0F67A49)
  %198 = tail call double @llvm.fma.f64(double %197, double %187, double 0xBEC102B892333B6F)
  %199 = tail call double @llvm.fma.f64(double %198, double %187, double 0x3ECA30375BA9A84E)
  %200 = tail call double @llvm.fma.f64(double %199, double %187, double 0x3ECAAD18DEDEA43E)
  %201 = tail call double @llvm.fma.f64(double %200, double %187, double 0xBEFF05355BC5B225)
  %202 = tail call double @llvm.fma.f64(double %201, double %187, double 0x3F10E37A3108BC8B)
  %203 = tail call double @llvm.fma.f64(double %202, double %187, double 0x3EFB292D828E5CB2)
  %204 = tail call double @llvm.fma.f64(double %203, double %187, double 0xBF4356626EBF9BFA)
  %205 = tail call double @llvm.fma.f64(double %204, double %187, double 0x3F5BCA68F73D6AFC)
  %206 = tail call double @llvm.fma.f64(double %205, double %187, double 0xBF2B6B69EBBC280B)
  %207 = tail call double @llvm.fma.f64(double %206, double %187, double 0xBF9396685912A453)
  %208 = tail call double @llvm.fma.f64(double %207, double %187, double 0x3FBA4F4E2A1ABEF8)
  %209 = tail call double @llvm.fma.f64(double %208, double %187, double 0x3FE45F306DC9C8BB)
  %210 = tail call double @llvm.fma.f64(double %209, double %187, double 0x3FC06EBA8214DB69)
  %211 = tail call double @llvm.fma.f64(double %210, double %187, double %187)
  %212 = fsub double %187, %211
  %213 = tail call double @llvm.fma.f64(double %210, double %187, double %212)
  %214 = fneg double %211
  %215 = fneg double %213
  %216 = fptrunc double %214 to float
  %217 = fmul float %216, 0x3FF7154760000000
  %218 = tail call float @llvm.nvvm.round.f(float %217) #9
  %219 = fpext float %218 to double
  %220 = fneg double %219
  %221 = tail call double @llvm.fma.f64(double %220, double 0x3FE62E42FEFA39EF, double %214)
  %222 = tail call double @llvm.fma.f64(double %221, double 0x3E5AE904A4741B81, double 0x3E928A27F89B6999)
  %223 = tail call double @llvm.fma.f64(double %222, double %221, double 0x3EC71DE715FF7E07)
  %224 = tail call double @llvm.fma.f64(double %223, double %221, double 0x3EFA019A6B0AC45A)
  %225 = tail call double @llvm.fma.f64(double %224, double %221, double 0x3F2A01A017EED94F)
  %226 = tail call double @llvm.fma.f64(double %225, double %221, double 0x3F56C16C17F2A71B)
  %227 = tail call double @llvm.fma.f64(double %226, double %221, double 0x3F811111111173C4)
  %228 = tail call double @llvm.fma.f64(double %227, double %221, double 0x3FA555555555211A)
  %229 = tail call double @llvm.fma.f64(double %228, double %221, double 0x3FC5555555555540)
  %230 = tail call double @llvm.fma.f64(double %229, double %221, double 0x3FE0000000000005)
  %231 = fmul double %221, %230
  %232 = tail call double @llvm.fma.f64(double %231, double %221, double %215)
  %233 = fadd double %221, %232
  %234 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %218)
  %235 = fpext float %234 to double
  %236 = fsub double 1.000000e+00, %235
  %237 = fneg double %233
  %238 = tail call double @llvm.fma.f64(double %237, double %235, double %236)
  %239 = fcmp oge double %187, 0x4017AFB48DC96626
  %poly.0.i5 = select i1 %239, double 1.000000e+00, double %238
  %240 = tail call double @llvm.copysign.f64(double %poly.0.i5, double %146) #9
  %241 = fcmp one double %187, 0x7FF0000000000000
  %242 = fmul double %146, %z.2.i3
  %243 = fsub double 1.000000e+00, %240
  %244 = fmul double %243, 5.000000e-01
  %245 = fmul double %242, 0x3FD20DD750429B6D
  %246 = select i1 %241, double %245, double 0.000000e+00
  %247 = fsub double %244, %246
  %248 = tail call double @llvm.nvvm.fabs.f64(double %247)
  %249 = fcmp olt double %248, %134
  %250 = select i1 %249, double 0.000000e+00, double %247
  %251 = fsub double 1.000000e+00, %250
  %252 = tail call double @llvm.nvvm.fabs.f64(double %251)
  %253 = fcmp olt double %252, %134
  %254 = select i1 %253, double 1.000000e+00, double %250
  %255 = fadd nsz double %140, %254
  br label %256

256:                                              ; preds = %__nv_exp.exit4, %__nv_exp.exit
  %257 = phi double [ %255, %__nv_exp.exit4 ], [ %140, %__nv_exp.exit ]
  %258 = bitcast double %257 to <2 x i32>
  %259 = extractelement <2 x i32> %258, i64 0
  %260 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %259, i32 16, i32 31)
  %261 = insertelement <2 x i32> poison, i32 %260, i64 0
  %262 = extractelement <2 x i32> %258, i64 1
  %263 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %262, i32 16, i32 31)
  %264 = insertelement <2 x i32> %261, i32 %263, i64 1
  %265 = bitcast <2 x i32> %264 to double
  %266 = fadd nsz double %257, %265
  %267 = bitcast double %266 to <2 x i32>
  %268 = extractelement <2 x i32> %267, i64 0
  %269 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %268, i32 8, i32 31)
  %270 = insertelement <2 x i32> poison, i32 %269, i64 0
  %271 = extractelement <2 x i32> %267, i64 1
  %272 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %271, i32 8, i32 31)
  %273 = insertelement <2 x i32> %270, i32 %272, i64 1
  %274 = bitcast <2 x i32> %273 to double
  %275 = fadd nsz double %266, %274
  %276 = bitcast double %275 to <2 x i32>
  %277 = extractelement <2 x i32> %276, i64 0
  %278 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %277, i32 4, i32 31)
  %279 = insertelement <2 x i32> poison, i32 %278, i64 0
  %280 = extractelement <2 x i32> %276, i64 1
  %281 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %280, i32 4, i32 31)
  %282 = insertelement <2 x i32> %279, i32 %281, i64 1
  %283 = bitcast <2 x i32> %282 to double
  %284 = fadd nsz double %275, %283
  %285 = bitcast double %284 to <2 x i32>
  %286 = extractelement <2 x i32> %285, i64 0
  %287 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %286, i32 2, i32 31)
  %288 = insertelement <2 x i32> poison, i32 %287, i64 0
  %289 = extractelement <2 x i32> %285, i64 1
  %290 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %289, i32 2, i32 31)
  %291 = insertelement <2 x i32> %288, i32 %290, i64 1
  %292 = bitcast <2 x i32> %291 to double
  %293 = fadd nsz double %284, %292
  %294 = bitcast double %293 to <2 x i32>
  %295 = extractelement <2 x i32> %294, i64 0
  %296 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %295, i32 1, i32 31)
  %297 = extractelement <2 x i32> %294, i64 1
  %298 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %297, i32 1, i32 31)
  %299 = icmp eq i32 %18, 0
  %300 = icmp samesign ult i32 %13, 225
  %301 = and i1 %300, %299
  br i1 %301, label %302, label %311

302:                                              ; preds = %256
  %303 = shl nuw nsw i32 %14, 3
  %304 = or disjoint i32 %303, %15
  %305 = zext nneg i32 %304 to i64
  %306 = getelementptr inbounds double, ptr addrspace(1) %12, i64 %305
  %307 = insertelement <2 x i32> poison, i32 %296, i64 0
  %308 = insertelement <2 x i32> %307, i32 %298, i64 1
  %309 = bitcast <2 x i32> %308 to double
  %310 = fadd nsz double %293, %309
  store double %310, ptr addrspace(1) %306, align 8
  br label %311

311:                                              ; preds = %302, %256
  ret void
}

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(4096) %0, ptr noalias readonly align 256 captures(none) dereferenceable(4096) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(8) %2) local_unnamed_addr #4 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !7
  %8 = zext nneg i32 %7 to i64
  %9 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %8
  %10 = load double, ptr addrspace(1) %9, align 8, !invariant.load !6
  %11 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %8
  %12 = load double, ptr addrspace(1) %11, align 8, !invariant.load !6
  %13 = fmul double %10, %12
  %14 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 256
  %15 = load double, ptr addrspace(1) %14, align 8, !invariant.load !6
  %16 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 256
  %17 = load double, ptr addrspace(1) %16, align 8, !invariant.load !6
  %18 = fmul double %15, %17
  %19 = fadd nsz double %13, %18
  %20 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 512
  %21 = load double, ptr addrspace(1) %20, align 8, !invariant.load !6
  %22 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 512
  %23 = load double, ptr addrspace(1) %22, align 8, !invariant.load !6
  %24 = fmul double %21, %23
  %25 = fadd nsz double %19, %24
  %26 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 768
  %27 = load double, ptr addrspace(1) %26, align 8, !invariant.load !6
  %28 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 768
  %29 = load double, ptr addrspace(1) %28, align 8, !invariant.load !6
  %30 = fmul double %27, %29
  %31 = fadd nsz double %25, %30
  %32 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 1024
  %33 = load double, ptr addrspace(1) %32, align 8, !invariant.load !6
  %34 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 1024
  %35 = load double, ptr addrspace(1) %34, align 8, !invariant.load !6
  %36 = fmul double %33, %35
  %37 = fadd nsz double %31, %36
  %38 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 1280
  %39 = load double, ptr addrspace(1) %38, align 8, !invariant.load !6
  %40 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 1280
  %41 = load double, ptr addrspace(1) %40, align 8, !invariant.load !6
  %42 = fmul double %39, %41
  %43 = fadd nsz double %37, %42
  %44 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 1536
  %45 = load double, ptr addrspace(1) %44, align 8, !invariant.load !6
  %46 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 1536
  %47 = load double, ptr addrspace(1) %46, align 8, !invariant.load !6
  %48 = fmul double %45, %47
  %49 = fadd nsz double %43, %48
  %50 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 1792
  %51 = load double, ptr addrspace(1) %50, align 8, !invariant.load !6
  %52 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 1792
  %53 = load double, ptr addrspace(1) %52, align 8, !invariant.load !6
  %54 = fmul double %51, %53
  %55 = fadd nsz double %49, %54
  %56 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 2048
  %57 = load double, ptr addrspace(1) %56, align 8, !invariant.load !6
  %58 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 2048
  %59 = load double, ptr addrspace(1) %58, align 8, !invariant.load !6
  %60 = fmul double %57, %59
  %61 = fadd nsz double %55, %60
  %62 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 2304
  %63 = load double, ptr addrspace(1) %62, align 8, !invariant.load !6
  %64 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 2304
  %65 = load double, ptr addrspace(1) %64, align 8, !invariant.load !6
  %66 = fmul double %63, %65
  %67 = fadd nsz double %61, %66
  %68 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 2560
  %69 = load double, ptr addrspace(1) %68, align 8, !invariant.load !6
  %70 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 2560
  %71 = load double, ptr addrspace(1) %70, align 8, !invariant.load !6
  %72 = fmul double %69, %71
  %73 = fadd nsz double %67, %72
  %74 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 2816
  %75 = load double, ptr addrspace(1) %74, align 8, !invariant.load !6
  %76 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 2816
  %77 = load double, ptr addrspace(1) %76, align 8, !invariant.load !6
  %78 = fmul double %75, %77
  %79 = fadd nsz double %73, %78
  %80 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 3072
  %81 = load double, ptr addrspace(1) %80, align 8, !invariant.load !6
  %82 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 3072
  %83 = load double, ptr addrspace(1) %82, align 8, !invariant.load !6
  %84 = fmul double %81, %83
  %85 = fadd nsz double %79, %84
  %86 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 3328
  %87 = load double, ptr addrspace(1) %86, align 8, !invariant.load !6
  %88 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 3328
  %89 = load double, ptr addrspace(1) %88, align 8, !invariant.load !6
  %90 = fmul double %87, %89
  %91 = fadd nsz double %85, %90
  %92 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 3584
  %93 = load double, ptr addrspace(1) %92, align 8, !invariant.load !6
  %94 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 3584
  %95 = load double, ptr addrspace(1) %94, align 8, !invariant.load !6
  %96 = fmul double %93, %95
  %97 = fadd nsz double %91, %96
  %98 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 3840
  %99 = load double, ptr addrspace(1) %98, align 8, !invariant.load !6
  %100 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 3840
  %101 = load double, ptr addrspace(1) %100, align 8, !invariant.load !6
  %102 = fmul double %99, %101
  %103 = fadd nsz double %97, %102
  %104 = bitcast double %103 to <2 x i32>
  %105 = extractelement <2 x i32> %104, i64 0
  %106 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %105, i32 16, i32 31)
  %107 = insertelement <2 x i32> poison, i32 %106, i64 0
  %108 = extractelement <2 x i32> %104, i64 1
  %109 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %108, i32 16, i32 31)
  %110 = insertelement <2 x i32> %107, i32 %109, i64 1
  %111 = bitcast <2 x i32> %110 to double
  %112 = fadd nsz double %103, %111
  %113 = bitcast double %112 to <2 x i32>
  %114 = extractelement <2 x i32> %113, i64 0
  %115 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %114, i32 8, i32 31)
  %116 = insertelement <2 x i32> poison, i32 %115, i64 0
  %117 = extractelement <2 x i32> %113, i64 1
  %118 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %117, i32 8, i32 31)
  %119 = insertelement <2 x i32> %116, i32 %118, i64 1
  %120 = bitcast <2 x i32> %119 to double
  %121 = fadd nsz double %112, %120
  %122 = bitcast double %121 to <2 x i32>
  %123 = extractelement <2 x i32> %122, i64 0
  %124 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %123, i32 4, i32 31)
  %125 = insertelement <2 x i32> poison, i32 %124, i64 0
  %126 = extractelement <2 x i32> %122, i64 1
  %127 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %126, i32 4, i32 31)
  %128 = insertelement <2 x i32> %125, i32 %127, i64 1
  %129 = bitcast <2 x i32> %128 to double
  %130 = fadd nsz double %121, %129
  %131 = bitcast double %130 to <2 x i32>
  %132 = extractelement <2 x i32> %131, i64 0
  %133 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %132, i32 2, i32 31)
  %134 = insertelement <2 x i32> poison, i32 %133, i64 0
  %135 = extractelement <2 x i32> %131, i64 1
  %136 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %135, i32 2, i32 31)
  %137 = insertelement <2 x i32> %134, i32 %136, i64 1
  %138 = bitcast <2 x i32> %137 to double
  %139 = fadd nsz double %130, %138
  %140 = bitcast double %139 to <2 x i32>
  %141 = extractelement <2 x i32> %140, i64 0
  %142 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %141, i32 1, i32 31)
  %143 = extractelement <2 x i32> %140, i64 1
  %144 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %143, i32 1, i32 31)
  %145 = icmp eq i32 %7, 0
  %146 = insertelement <2 x i32> poison, i32 %142, i64 0
  %147 = insertelement <2 x i32> %146, i32 %144, i64 1
  %148 = bitcast <2 x i32> %147 to double
  %149 = fadd nsz double %139, %148
  br i1 %145, label %150, label %151

150:                                              ; preds = %3
  store double %149, ptr addrspace(1) %6, align 256
  br label %151

151:                                              ; preds = %150, %3
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias readonly align 16 captures(none) dereferenceable(8) %2, ptr noalias readonly align 256 captures(none) dereferenceable(8) %3, ptr noalias readonly align 256 captures(none) dereferenceable(8) %4, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %5) local_unnamed_addr #5 {
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %4 to ptr addrspace(1)
  %10 = addrspacecast ptr %1 to ptr addrspace(1)
  %11 = addrspacecast ptr %0 to ptr addrspace(1)
  %12 = addrspacecast ptr %5 to ptr addrspace(1)
  %13 = load double, ptr addrspace(1) %7, align 16, !invariant.load !6
  %14 = load double, ptr addrspace(1) %8, align 256, !invariant.load !6
  %15 = load double, ptr addrspace(1) %9, align 256, !invariant.load !6
  %16 = fmul double %13, %14
  %17 = load double, ptr addrspace(1) %10, align 16, !invariant.load !6
  %18 = load double, ptr addrspace(1) %11, align 256, !invariant.load !6
  %19 = fadd double %15, %18
  %20 = fcmp olt double %16, %17
  %21 = fmul double %19, 5.000000e-01
  %22 = select i1 %20, double %18, double %21
  store double %22, ptr addrspace(1) %12, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion(ptr noalias align 256 captures(none) dereferenceable(8) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias readonly align 16 captures(none) dereferenceable(8) %2, ptr noalias readonly align 256 captures(none) dereferenceable(8) %3, ptr noalias readonly align 256 captures(none) dereferenceable(8) %4, ptr noalias readnone align 256 captures(none) dereferenceable(8) %5) local_unnamed_addr #5 {
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %4 to ptr addrspace(1)
  %10 = addrspacecast ptr %1 to ptr addrspace(1)
  %11 = addrspacecast ptr %0 to ptr addrspace(1)
  %12 = load double, ptr addrspace(1) %7, align 16, !invariant.load !6
  %13 = load double, ptr addrspace(1) %8, align 256, !invariant.load !6
  %14 = load double, ptr addrspace(1) %9, align 256, !invariant.load !6
  %15 = fmul double %12, %13
  %16 = load double, ptr addrspace(1) %10, align 16, !invariant.load !6
  %17 = load double, ptr addrspace(1) %11, align 256
  %18 = fadd double %14, %17
  %19 = fcmp olt double %15, %16
  %20 = fmul double %18, 5.000000e-01
  %21 = select i1 %19, double %20, double %17
  store double %21, ptr addrspace(1) %11, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %2) local_unnamed_addr #5 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = load double, ptr addrspace(1) %4, align 256, !invariant.load !6
  %8 = load double, ptr addrspace(1) %5, align 256, !invariant.load !6
  %9 = fadd double %7, %8
  %10 = fmul double %9, 5.000000e-01
  store double %10, ptr addrspace(1) %6, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion_2(ptr noalias readonly align 16 captures(none) dereferenceable(8) %0, ptr noalias readonly align 16 captures(none) dereferenceable(196608) %1, ptr noalias readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias readonly align 16 captures(none) dereferenceable(8) %3, ptr noalias writeonly align 256 captures(none) dereferenceable(196608) %4) local_unnamed_addr #6 {
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %0 to ptr addrspace(1)
  %10 = addrspacecast ptr %4 to ptr addrspace(1)
  %11 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !9
  %12 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !10
  %13 = shl nuw nsw i32 %11, 7
  %14 = or disjoint i32 %13, %12
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %15
  %17 = load double, ptr addrspace(1) %16, align 8, !invariant.load !6
  %18 = load double, ptr addrspace(1) %7, align 256, !invariant.load !6
  %19 = fsub double %17, %18
  %20 = load double, ptr addrspace(1) %8, align 16, !invariant.load !6
  %21 = fmul double %20, 2.000000e+00
  %22 = fdiv double %19, %21
  %23 = fneg double %22
  %24 = fmul double %22, %23
  %25 = tail call double @llvm.fma.f64(double %24, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %26 = tail call i32 @llvm.nvvm.d2i.lo(double %25) #9
  %27 = tail call double @llvm.nvvm.add.rn.d(double %25, double 0xC338000000000000) #9
  %28 = tail call double @llvm.fma.f64(double %27, double 0xBFE62E42FEFA39EF, double %24)
  %29 = tail call double @llvm.fma.f64(double %27, double 0xBC7ABC9E3B39803F, double %28)
  %30 = tail call double @llvm.fma.f64(double %29, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %31 = tail call double @llvm.fma.f64(double %30, double %29, double 0x3EC71DEE62401315)
  %32 = tail call double @llvm.fma.f64(double %31, double %29, double 0x3EFA01997C89EB71)
  %33 = tail call double @llvm.fma.f64(double %32, double %29, double 0x3F2A01A014761F65)
  %34 = tail call double @llvm.fma.f64(double %33, double %29, double 0x3F56C16C1852B7AF)
  %35 = tail call double @llvm.fma.f64(double %34, double %29, double 0x3F81111111122322)
  %36 = tail call double @llvm.fma.f64(double %35, double %29, double 0x3FA55555555502A1)
  %37 = tail call double @llvm.fma.f64(double %36, double %29, double 0x3FC5555555555511)
  %38 = tail call double @llvm.fma.f64(double %37, double %29, double 0x3FE000000000000B)
  %39 = tail call double @llvm.fma.f64(double %38, double %29, double 1.000000e+00)
  %40 = tail call double @llvm.fma.f64(double %39, double %29, double 1.000000e+00)
  %41 = tail call i32 @llvm.nvvm.d2i.lo(double %40) #9
  %42 = tail call i32 @llvm.nvvm.d2i.hi(double %40) #9
  %43 = shl i32 %26, 20
  %44 = add i32 %42, %43
  %45 = tail call double @llvm.nvvm.lohi.i2d(i32 %41, i32 %44) #9
  %46 = tail call i32 @llvm.nvvm.d2i.hi(double %24) #9
  %47 = bitcast i32 %46 to float
  %48 = tail call float @llvm.nvvm.fabs.f32(float %47)
  %49 = fcmp olt float %48, 0x4010C46560000000
  br i1 %49, label %__nv_exp.exit, label %__internal_fast_icmp_abs_lt.exit.i

__internal_fast_icmp_abs_lt.exit.i:               ; preds = %5
  %50 = fcmp olt double %24, 0.000000e+00
  %51 = fadd double %24, 0x7FF0000000000000
  %z.0.i = select i1 %50, double 0.000000e+00, double %51
  %52 = fcmp olt float %48, 0x4010E90000000000
  br i1 %52, label %53, label %__nv_exp.exit

53:                                               ; preds = %__internal_fast_icmp_abs_lt.exit.i
  %54 = sdiv i32 %26, 2
  %55 = shl i32 %54, 20
  %56 = add i32 %42, %55
  %57 = tail call double @llvm.nvvm.lohi.i2d(i32 %41, i32 %56) #9
  %58 = sub nsw i32 %26, %54
  %59 = shl i32 %58, 20
  %60 = add nsw i32 %59, 1072693248
  %61 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %60) #9
  %62 = fmul double %61, %57
  br label %__nv_exp.exit

__nv_exp.exit:                                    ; preds = %5, %__internal_fast_icmp_abs_lt.exit.i, %53
  %z.2.i = phi double [ %45, %5 ], [ %62, %53 ], [ %z.0.i, %__internal_fast_icmp_abs_lt.exit.i ]
  %63 = tail call double @llvm.nvvm.fabs.f64(double %22)
  %64 = tail call double @llvm.fma.f64(double %63, double 0xBCF0679AFBA6F279, double 0x3D47088FDB46FA5F)
  %65 = tail call double @llvm.fma.f64(double %64, double %63, double 0xBD8DF9F9B976A9B2)
  %66 = tail call double @llvm.fma.f64(double %65, double %63, double 0x3DC7F1F5590CC332)
  %67 = tail call double @llvm.fma.f64(double %66, double %63, double 0xBDFA28A3CD2D56C4)
  %68 = tail call double @llvm.fma.f64(double %67, double %63, double 0x3E2485EE67835925)
  %69 = tail call double @llvm.fma.f64(double %68, double %63, double 0xBE476DB45919F583)
  %70 = tail call double @llvm.fma.f64(double %69, double %63, double 0x3E62D698D98C8D71)
  %71 = tail call double @llvm.fma.f64(double %70, double %63, double 0xBE720A2C7155D5C6)
  %72 = tail call double @llvm.fma.f64(double %71, double %63, double 0xBE41D29B37CA1397)
  %73 = tail call double @llvm.fma.f64(double %72, double %63, double 0x3EA2EF6CC0F67A49)
  %74 = tail call double @llvm.fma.f64(double %73, double %63, double 0xBEC102B892333B6F)
  %75 = tail call double @llvm.fma.f64(double %74, double %63, double 0x3ECA30375BA9A84E)
  %76 = tail call double @llvm.fma.f64(double %75, double %63, double 0x3ECAAD18DEDEA43E)
  %77 = tail call double @llvm.fma.f64(double %76, double %63, double 0xBEFF05355BC5B225)
  %78 = tail call double @llvm.fma.f64(double %77, double %63, double 0x3F10E37A3108BC8B)
  %79 = tail call double @llvm.fma.f64(double %78, double %63, double 0x3EFB292D828E5CB2)
  %80 = tail call double @llvm.fma.f64(double %79, double %63, double 0xBF4356626EBF9BFA)
  %81 = tail call double @llvm.fma.f64(double %80, double %63, double 0x3F5BCA68F73D6AFC)
  %82 = tail call double @llvm.fma.f64(double %81, double %63, double 0xBF2B6B69EBBC280B)
  %83 = tail call double @llvm.fma.f64(double %82, double %63, double 0xBF9396685912A453)
  %84 = tail call double @llvm.fma.f64(double %83, double %63, double 0x3FBA4F4E2A1ABEF8)
  %85 = tail call double @llvm.fma.f64(double %84, double %63, double 0x3FE45F306DC9C8BB)
  %86 = tail call double @llvm.fma.f64(double %85, double %63, double 0x3FC06EBA8214DB69)
  %87 = tail call double @llvm.fma.f64(double %86, double %63, double %63)
  %88 = fsub double %63, %87
  %89 = tail call double @llvm.fma.f64(double %86, double %63, double %88)
  %90 = fneg double %87
  %91 = fneg double %89
  %92 = fptrunc double %90 to float
  %93 = fmul float %92, 0x3FF7154760000000
  %94 = tail call float @llvm.nvvm.round.f(float %93) #9
  %95 = fpext float %94 to double
  %96 = fneg double %95
  %97 = tail call double @llvm.fma.f64(double %96, double 0x3FE62E42FEFA39EF, double %90)
  %98 = tail call double @llvm.fma.f64(double %97, double 0x3E5AE904A4741B81, double 0x3E928A27F89B6999)
  %99 = tail call double @llvm.fma.f64(double %98, double %97, double 0x3EC71DE715FF7E07)
  %100 = tail call double @llvm.fma.f64(double %99, double %97, double 0x3EFA019A6B0AC45A)
  %101 = tail call double @llvm.fma.f64(double %100, double %97, double 0x3F2A01A017EED94F)
  %102 = tail call double @llvm.fma.f64(double %101, double %97, double 0x3F56C16C17F2A71B)
  %103 = tail call double @llvm.fma.f64(double %102, double %97, double 0x3F811111111173C4)
  %104 = tail call double @llvm.fma.f64(double %103, double %97, double 0x3FA555555555211A)
  %105 = tail call double @llvm.fma.f64(double %104, double %97, double 0x3FC5555555555540)
  %106 = tail call double @llvm.fma.f64(double %105, double %97, double 0x3FE0000000000005)
  %107 = fmul double %97, %106
  %108 = tail call double @llvm.fma.f64(double %107, double %97, double %91)
  %109 = fadd double %97, %108
  %110 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %94)
  %111 = fpext float %110 to double
  %112 = fsub double 1.000000e+00, %111
  %113 = fneg double %109
  %114 = tail call double @llvm.fma.f64(double %113, double %111, double %112)
  %115 = fcmp oge double %63, 0x4017AFB48DC96626
  %poly.0.i = select i1 %115, double 1.000000e+00, double %114
  %116 = tail call double @llvm.copysign.f64(double %poly.0.i, double %22) #9
  %117 = fcmp one double %63, 0x7FF0000000000000
  %118 = fmul double %22, %z.2.i
  %119 = fsub double 1.000000e+00, %116
  %120 = fmul double %119, 5.000000e-01
  %121 = fmul double %118, 0x3FD20DD750429B6D
  %122 = select i1 %117, double %121, double 0.000000e+00
  %123 = fsub double %120, %122
  %124 = tail call double @llvm.nvvm.fabs.f64(double %123)
  %125 = load double, ptr addrspace(1) %9, align 16, !invariant.load !6
  %126 = fcmp olt double %124, %125
  %127 = select i1 %126, double 0.000000e+00, double %123
  %128 = fsub double 1.000000e+00, %127
  %129 = tail call double @llvm.nvvm.fabs.f64(double %128)
  %130 = fcmp olt double %129, %125
  %131 = select i1 %130, double 1.000000e+00, double %127
  %132 = getelementptr inbounds double, ptr addrspace(1) %10, i64 %15
  store double %131, ptr addrspace(1) %132, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.lo(double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.add.rn.d(double, double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.hi(double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.lohi.i2d(i32, i32) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fabs.f32(float) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.round.f(float) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.ex2.approx.ftz.f32(float) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.copysign.f64(double, double) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fma.f64(double, double, double) #8

attributes #0 = { norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "nvvm.reqntid"="256,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "nvvm.reqntid"="32,1,1" }
attributes #5 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #6 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #7 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 256}
!5 = !{i32 0, i32 16}
!6 = !{}
!7 = !{i32 0, i32 32}
!8 = !{i32 0, i32 64}
!9 = !{i32 0, i32 192}
!10 = !{i32 0, i32 128}
