; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef
@shared_01 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion_1(ptr noalias readonly align 16 captures(none) dereferenceable(4718592) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4718592) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %7 = and i32 %5, 31
  %8 = icmp samesign ult i32 %7, 24
  br i1 %8, label %9, label %37

9:                                                ; preds = %2
  %10 = lshr i32 %5, 5
  %11 = mul nuw nsw i32 %10, 24
  %12 = mul nuw nsw i32 %6, 576
  %13 = or disjoint i32 %7, %12
  %14 = add nuw nsw i32 %13, %11
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %15
  %17 = load <2 x double>, ptr addrspace(1) %16, align 16, !invariant.load !4
  %.unpack100 = extractelement <2 x double> %17, i32 0
  %.unpack2101 = extractelement <2 x double> %17, i32 1
  %18 = mul nuw nsw i32 %7, 33
  %19 = add nuw nsw i32 %18, %10
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %20
  store double %.unpack100, ptr addrspace(3) %21, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 8
  store double %.unpack2101, ptr addrspace(3) %.repack3, align 8
  %22 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 1536
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !4
  %.unpack5102 = extractelement <2 x double> %23, i32 0
  %.unpack7103 = extractelement <2 x double> %23, i32 1
  %24 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 64
  store double %.unpack5102, ptr addrspace(3) %24, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 72
  store double %.unpack7103, ptr addrspace(3) %.repack8, align 8
  %25 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 3072
  %26 = load <2 x double>, ptr addrspace(1) %25, align 16, !invariant.load !4
  %.unpack10104 = extractelement <2 x double> %26, i32 0
  %.unpack12105 = extractelement <2 x double> %26, i32 1
  %27 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 128
  store double %.unpack10104, ptr addrspace(3) %27, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 136
  store double %.unpack12105, ptr addrspace(3) %.repack13, align 8
  %28 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 4608
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack15106 = extractelement <2 x double> %29, i32 0
  %.unpack17107 = extractelement <2 x double> %29, i32 1
  %30 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 192
  store double %.unpack15106, ptr addrspace(3) %30, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 200
  store double %.unpack17107, ptr addrspace(3) %.repack18, align 8
  %31 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 6144
  %32 = load <2 x double>, ptr addrspace(1) %31, align 16, !invariant.load !4
  %.unpack20108 = extractelement <2 x double> %32, i32 0
  %.unpack22109 = extractelement <2 x double> %32, i32 1
  %33 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 256
  store double %.unpack20108, ptr addrspace(3) %33, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 264
  store double %.unpack22109, ptr addrspace(3) %.repack23, align 8
  %34 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 7680
  %35 = load <2 x double>, ptr addrspace(1) %34, align 16, !invariant.load !4
  %.unpack25110 = extractelement <2 x double> %35, i32 0
  %.unpack27111 = extractelement <2 x double> %35, i32 1
  %36 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 320
  store double %.unpack25110, ptr addrspace(3) %36, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 328
  store double %.unpack27111, ptr addrspace(3) %.repack28, align 8
  br label %37

37:                                               ; preds = %9, %2
  %38 = icmp ult i32 %7, 24
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br i1 %38, label %39, label %133

39:                                               ; preds = %37
  %40 = lshr i32 %5, 5
  %41 = mul nuw nsw i32 %40, 33
  %42 = add nuw nsw i32 %41, %7
  %43 = zext nneg i32 %42 to i64
  %44 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %43
  %.unpack30 = load double, ptr addrspace(3) %44, align 8
  %.elt31 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 8
  %.unpack32 = load double, ptr addrspace(3) %.elt31, align 8
  %45 = mul nuw nsw i32 %40, 24
  %46 = mul nuw nsw i32 %6, 576
  %47 = or disjoint i32 %7, %46
  %48 = add nuw nsw i32 %47, %45
  %49 = zext nneg i32 %48 to i64
  %50 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %49
  %51 = load <2 x double>, ptr addrspace(1) %50, align 16, !invariant.load !4
  %.unpack3388 = extractelement <2 x double> %51, i32 0
  %.unpack3589 = extractelement <2 x double> %51, i32 1
  %52 = fadd double %.unpack30, %.unpack3388
  %53 = fsub double %.unpack32, %.unpack3589
  %54 = fmul double %52, 5.000000e-01
  %55 = fmul double %53, 0.000000e+00
  %56 = fsub double %54, %55
  %57 = fmul double %53, 5.000000e-01
  %58 = fmul double %52, 0.000000e+00
  %59 = fadd double %58, %57
  %60 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %49
  %61 = insertelement <2 x double> poison, double %56, i32 0
  %62 = insertelement <2 x double> %61, double %59, i32 1
  store <2 x double> %62, ptr addrspace(1) %60, align 16
  %63 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 2112
  %.unpack38 = load double, ptr addrspace(3) %63, align 8
  %.elt39 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 2120
  %.unpack40 = load double, ptr addrspace(3) %.elt39, align 8
  %64 = getelementptr inbounds i8, ptr addrspace(1) %50, i64 1536
  %65 = load <2 x double>, ptr addrspace(1) %64, align 16, !invariant.load !4
  %.unpack4190 = extractelement <2 x double> %65, i32 0
  %.unpack4391 = extractelement <2 x double> %65, i32 1
  %66 = fadd double %.unpack38, %.unpack4190
  %67 = fsub double %.unpack40, %.unpack4391
  %68 = fmul double %66, 5.000000e-01
  %69 = fmul double %67, 0.000000e+00
  %70 = fsub double %68, %69
  %71 = fmul double %67, 5.000000e-01
  %72 = fmul double %66, 0.000000e+00
  %73 = fadd double %72, %71
  %74 = getelementptr inbounds i8, ptr addrspace(1) %60, i64 1536
  %75 = insertelement <2 x double> poison, double %70, i32 0
  %76 = insertelement <2 x double> %75, double %73, i32 1
  store <2 x double> %76, ptr addrspace(1) %74, align 16
  %77 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 4224
  %.unpack46 = load double, ptr addrspace(3) %77, align 8
  %.elt47 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 4232
  %.unpack48 = load double, ptr addrspace(3) %.elt47, align 8
  %78 = getelementptr inbounds i8, ptr addrspace(1) %50, i64 3072
  %79 = load <2 x double>, ptr addrspace(1) %78, align 16, !invariant.load !4
  %.unpack4992 = extractelement <2 x double> %79, i32 0
  %.unpack5193 = extractelement <2 x double> %79, i32 1
  %80 = fadd double %.unpack46, %.unpack4992
  %81 = fsub double %.unpack48, %.unpack5193
  %82 = fmul double %80, 5.000000e-01
  %83 = fmul double %81, 0.000000e+00
  %84 = fsub double %82, %83
  %85 = fmul double %81, 5.000000e-01
  %86 = fmul double %80, 0.000000e+00
  %87 = fadd double %86, %85
  %88 = getelementptr inbounds i8, ptr addrspace(1) %60, i64 3072
  %89 = insertelement <2 x double> poison, double %84, i32 0
  %90 = insertelement <2 x double> %89, double %87, i32 1
  store <2 x double> %90, ptr addrspace(1) %88, align 16
  %91 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 6336
  %.unpack54 = load double, ptr addrspace(3) %91, align 8
  %.elt55 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 6344
  %.unpack56 = load double, ptr addrspace(3) %.elt55, align 8
  %92 = getelementptr inbounds i8, ptr addrspace(1) %50, i64 4608
  %93 = load <2 x double>, ptr addrspace(1) %92, align 16, !invariant.load !4
  %.unpack5794 = extractelement <2 x double> %93, i32 0
  %.unpack5995 = extractelement <2 x double> %93, i32 1
  %94 = fadd double %.unpack54, %.unpack5794
  %95 = fsub double %.unpack56, %.unpack5995
  %96 = fmul double %94, 5.000000e-01
  %97 = fmul double %95, 0.000000e+00
  %98 = fsub double %96, %97
  %99 = fmul double %95, 5.000000e-01
  %100 = fmul double %94, 0.000000e+00
  %101 = fadd double %100, %99
  %102 = getelementptr inbounds i8, ptr addrspace(1) %60, i64 4608
  %103 = insertelement <2 x double> poison, double %98, i32 0
  %104 = insertelement <2 x double> %103, double %101, i32 1
  store <2 x double> %104, ptr addrspace(1) %102, align 16
  %105 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 8448
  %.unpack62 = load double, ptr addrspace(3) %105, align 8
  %.elt63 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 8456
  %.unpack64 = load double, ptr addrspace(3) %.elt63, align 8
  %106 = getelementptr inbounds i8, ptr addrspace(1) %50, i64 6144
  %107 = load <2 x double>, ptr addrspace(1) %106, align 16, !invariant.load !4
  %.unpack6596 = extractelement <2 x double> %107, i32 0
  %.unpack6797 = extractelement <2 x double> %107, i32 1
  %108 = fadd double %.unpack62, %.unpack6596
  %109 = fsub double %.unpack64, %.unpack6797
  %110 = fmul double %108, 5.000000e-01
  %111 = fmul double %109, 0.000000e+00
  %112 = fsub double %110, %111
  %113 = fmul double %109, 5.000000e-01
  %114 = fmul double %108, 0.000000e+00
  %115 = fadd double %114, %113
  %116 = getelementptr inbounds i8, ptr addrspace(1) %60, i64 6144
  %117 = insertelement <2 x double> poison, double %112, i32 0
  %118 = insertelement <2 x double> %117, double %115, i32 1
  store <2 x double> %118, ptr addrspace(1) %116, align 16
  %119 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 10560
  %.unpack70 = load double, ptr addrspace(3) %119, align 8
  %.elt71 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 10568
  %.unpack72 = load double, ptr addrspace(3) %.elt71, align 8
  %120 = getelementptr inbounds i8, ptr addrspace(1) %50, i64 7680
  %121 = load <2 x double>, ptr addrspace(1) %120, align 16, !invariant.load !4
  %.unpack7398 = extractelement <2 x double> %121, i32 0
  %.unpack7599 = extractelement <2 x double> %121, i32 1
  %122 = fadd double %.unpack70, %.unpack7398
  %123 = fsub double %.unpack72, %.unpack7599
  %124 = fmul double %122, 5.000000e-01
  %125 = fmul double %123, 0.000000e+00
  %126 = fsub double %124, %125
  %127 = fmul double %123, 5.000000e-01
  %128 = fmul double %122, 0.000000e+00
  %129 = fadd double %128, %127
  %130 = getelementptr inbounds i8, ptr addrspace(1) %60, i64 7680
  %131 = insertelement <2 x double> poison, double %126, i32 0
  %132 = insertelement <2 x double> %131, double %129, i32 1
  store <2 x double> %132, ptr addrspace(1) %130, align 16
  br label %133

133:                                              ; preds = %39, %37
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion(ptr noalias align 256 captures(none) dereferenceable(98304) %0, ptr noalias readonly align 256 captures(none) dereferenceable(2048) %1, ptr noalias readnone align 256 captures(none) dereferenceable(98304) %2) local_unnamed_addr #3 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = shl nuw nsw i32 %6, 7
  %9 = or disjoint i32 %8, %7
  %10 = udiv i32 %9, 24
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %11
  %13 = load i32, ptr addrspace(1) %12, align 4, !invariant.load !4
  %14 = icmp eq i32 %13, 0
  %15 = zext nneg i32 %9 to i64
  %16 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %15
  %17 = load double, ptr addrspace(1) %16, align 8
  %18 = select i1 %14, double %17, double 0x7FF8000000000000
  store double %18, ptr addrspace(1) %16, align 8
  ret void
}

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(4718592) %0, ptr noalias readonly align 256 captures(none) dereferenceable(2048) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(4718592) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %9 = and i32 %7, 31
  %10 = icmp samesign ult i32 %9, 24
  br i1 %10, label %11, label %43

11:                                               ; preds = %3
  %12 = zext nneg i32 %8 to i64
  %13 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %12
  %14 = load i32, ptr addrspace(1) %13, align 4, !invariant.load !4
  %15 = icmp eq i32 %14, 0
  %16 = lshr i32 %7, 5
  %17 = mul nuw nsw i32 %16, 24
  %18 = mul nuw nsw i32 %8, 576
  %19 = or disjoint i32 %9, %18
  %20 = add nuw nsw i32 %19, %17
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %21
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !4
  %.unpack82 = extractelement <2 x double> %23, i32 0
  %.unpack283 = extractelement <2 x double> %23, i32 1
  %24 = mul nuw nsw i32 %9, 33
  %25 = add nuw nsw i32 %24, %16
  %26 = zext nneg i32 %25 to i64
  %27 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_01, i64 %26
  %.elt = select i1 %15, double %.unpack82, double 0x7FF8000000000000
  store double %.elt, ptr addrspace(3) %27, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 8
  %.elt4 = select i1 %15, double %.unpack283, double 0x7FF8000000000000
  store double %.elt4, ptr addrspace(3) %.repack3, align 8
  %28 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 1536
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack684 = extractelement <2 x double> %29, i32 0
  %.unpack885 = extractelement <2 x double> %29, i32 1
  %30 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 64
  %.elt9 = select i1 %15, double %.unpack684, double 0x7FF8000000000000
  store double %.elt9, ptr addrspace(3) %30, align 8
  %.repack10 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 72
  %.elt11 = select i1 %15, double %.unpack885, double 0x7FF8000000000000
  store double %.elt11, ptr addrspace(3) %.repack10, align 8
  %31 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 3072
  %32 = load <2 x double>, ptr addrspace(1) %31, align 16, !invariant.load !4
  %.unpack1386 = extractelement <2 x double> %32, i32 0
  %.unpack1587 = extractelement <2 x double> %32, i32 1
  %33 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 128
  %.elt16 = select i1 %15, double %.unpack1386, double 0x7FF8000000000000
  store double %.elt16, ptr addrspace(3) %33, align 8
  %.repack17 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 136
  %.elt18 = select i1 %15, double %.unpack1587, double 0x7FF8000000000000
  store double %.elt18, ptr addrspace(3) %.repack17, align 8
  %34 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 4608
  %35 = load <2 x double>, ptr addrspace(1) %34, align 16, !invariant.load !4
  %.unpack2088 = extractelement <2 x double> %35, i32 0
  %.unpack2289 = extractelement <2 x double> %35, i32 1
  %36 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 192
  %.elt23 = select i1 %15, double %.unpack2088, double 0x7FF8000000000000
  store double %.elt23, ptr addrspace(3) %36, align 8
  %.repack24 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 200
  %.elt25 = select i1 %15, double %.unpack2289, double 0x7FF8000000000000
  store double %.elt25, ptr addrspace(3) %.repack24, align 8
  %37 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 6144
  %38 = load <2 x double>, ptr addrspace(1) %37, align 16, !invariant.load !4
  %.unpack2790 = extractelement <2 x double> %38, i32 0
  %.unpack2991 = extractelement <2 x double> %38, i32 1
  %39 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 256
  %.elt30 = select i1 %15, double %.unpack2790, double 0x7FF8000000000000
  store double %.elt30, ptr addrspace(3) %39, align 8
  %.repack31 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 264
  %.elt32 = select i1 %15, double %.unpack2991, double 0x7FF8000000000000
  store double %.elt32, ptr addrspace(3) %.repack31, align 8
  %40 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 7680
  %41 = load <2 x double>, ptr addrspace(1) %40, align 16, !invariant.load !4
  %.unpack3492 = extractelement <2 x double> %41, i32 0
  %.unpack3693 = extractelement <2 x double> %41, i32 1
  %42 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 320
  %.elt37 = select i1 %15, double %.unpack3492, double 0x7FF8000000000000
  store double %.elt37, ptr addrspace(3) %42, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 328
  %.elt39 = select i1 %15, double %.unpack3693, double 0x7FF8000000000000
  store double %.elt39, ptr addrspace(3) %.repack38, align 8
  br label %43

43:                                               ; preds = %11, %3
  %44 = icmp ult i32 %9, 24
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br i1 %44, label %45, label %79

45:                                               ; preds = %43
  %46 = lshr i32 %7, 5
  %47 = mul nuw nsw i32 %46, 33
  %48 = add nuw nsw i32 %47, %9
  %49 = zext nneg i32 %48 to i64
  %50 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_01, i64 %49
  %.unpack41 = load double, ptr addrspace(3) %50, align 8
  %.elt42 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 8
  %.unpack43 = load double, ptr addrspace(3) %.elt42, align 8
  %51 = mul nuw nsw i32 %46, 24
  %52 = mul nuw nsw i32 %8, 576
  %53 = or disjoint i32 %9, %52
  %54 = add nuw nsw i32 %53, %51
  %55 = zext nneg i32 %54 to i64
  %56 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %55
  %57 = insertelement <2 x double> poison, double %.unpack41, i32 0
  %58 = insertelement <2 x double> %57, double %.unpack43, i32 1
  store <2 x double> %58, ptr addrspace(1) %56, align 16
  %59 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 2112
  %.unpack48 = load double, ptr addrspace(3) %59, align 8
  %.elt49 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 2120
  %.unpack50 = load double, ptr addrspace(3) %.elt49, align 8
  %60 = getelementptr inbounds i8, ptr addrspace(1) %56, i64 1536
  %61 = insertelement <2 x double> poison, double %.unpack48, i32 0
  %62 = insertelement <2 x double> %61, double %.unpack50, i32 1
  store <2 x double> %62, ptr addrspace(1) %60, align 16
  %63 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 4224
  %.unpack55 = load double, ptr addrspace(3) %63, align 8
  %.elt56 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 4232
  %.unpack57 = load double, ptr addrspace(3) %.elt56, align 8
  %64 = getelementptr inbounds i8, ptr addrspace(1) %56, i64 3072
  %65 = insertelement <2 x double> poison, double %.unpack55, i32 0
  %66 = insertelement <2 x double> %65, double %.unpack57, i32 1
  store <2 x double> %66, ptr addrspace(1) %64, align 16
  %67 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 6336
  %.unpack62 = load double, ptr addrspace(3) %67, align 8
  %.elt63 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 6344
  %.unpack64 = load double, ptr addrspace(3) %.elt63, align 8
  %68 = getelementptr inbounds i8, ptr addrspace(1) %56, i64 4608
  %69 = insertelement <2 x double> poison, double %.unpack62, i32 0
  %70 = insertelement <2 x double> %69, double %.unpack64, i32 1
  store <2 x double> %70, ptr addrspace(1) %68, align 16
  %71 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 8448
  %.unpack69 = load double, ptr addrspace(3) %71, align 8
  %.elt70 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 8456
  %.unpack71 = load double, ptr addrspace(3) %.elt70, align 8
  %72 = getelementptr inbounds i8, ptr addrspace(1) %56, i64 6144
  %73 = insertelement <2 x double> poison, double %.unpack69, i32 0
  %74 = insertelement <2 x double> %73, double %.unpack71, i32 1
  store <2 x double> %74, ptr addrspace(1) %72, align 16
  %75 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 10560
  %.unpack76 = load double, ptr addrspace(3) %75, align 8
  %.elt77 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 10568
  %.unpack78 = load double, ptr addrspace(3) %.elt77, align 8
  %76 = getelementptr inbounds i8, ptr addrspace(1) %56, i64 7680
  %77 = insertelement <2 x double> poison, double %.unpack76, i32 0
  %78 = insertelement <2 x double> %77, double %.unpack78, i32 1
  store <2 x double> %78, ptr addrspace(1) %76, align 16
  br label %79

79:                                               ; preds = %45, %43
  ret void
}

attributes #0 = { norecurse nounwind "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 128}
!3 = !{i32 0, i32 512}
!4 = !{}
!5 = !{i32 0, i32 96}
