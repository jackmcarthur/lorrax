; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_1 = private unnamed_addr addrspace(3) global [1056 x double] undef
@shared_0 = private unnamed_addr addrspace(3) global [1056 x double] undef
@shared_11 = private unnamed_addr addrspace(3) global [1056 x double] undef
@shared_02 = private unnamed_addr addrspace(3) global [1056 x double] undef
@shared_03 = private unnamed_addr addrspace(3) global [1056 x double] undef
@shared_04 = private unnamed_addr addrspace(3) global [1056 x double] undef

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_and_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(4) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(24) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = zext nneg i32 %5 to i64
  %7 = load i32, ptr addrspace(1) %3, align 256, !invariant.load !3
  %8 = and i32 %7, 2
  %.not = icmp eq i32 %8, 0
  %.neg = select i1 %.not, i64 0, i64 -12
  %9 = add nsw i64 %.neg, %6
  %10 = trunc i32 %7 to i1
  %.neg1 = select i1 %10, i64 -12, i64 0
  %11 = icmp ult i64 %9, 12
  %12 = add nsw i64 %.neg1, %6
  %13 = icmp ult i64 %12, 12
  %14 = and i1 %13, %11
  %15 = zext i1 %14 to i8
  %16 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %6
  store i8 %15, ptr addrspace(1) %16, align 1
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(4128768) %0, ptr noalias readonly align 256 captures(none) dereferenceable(24) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(2064384) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(2064384) %3) local_unnamed_addr #2 {
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %0 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %10 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %11 = and i32 %9, 31
  %12 = icmp samesign ult i32 %11, 24
  br i1 %12, label %13, label %._crit_edge

._crit_edge:                                      ; preds = %4
  %.pre = lshr i32 %9, 5
  br label %75

13:                                               ; preds = %4
  %14 = zext nneg i32 %11 to i64
  %15 = getelementptr inbounds i8, ptr addrspace(1) %5, i64 %14
  %16 = load i8, ptr addrspace(1) %15, align 1, !invariant.load !3
  %17 = trunc i8 %16 to i1
  %18 = lshr i32 %9, 5
  %19 = mul nuw nsw i32 %18, 24
  %20 = mul nuw nsw i32 %10, 768
  %21 = or disjoint i32 %11, %20
  %22 = add nuw nsw i32 %21, %19
  %23 = zext nneg i32 %22 to i64
  %24 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %23
  %25 = load <2 x double>, ptr addrspace(1) %24, align 16, !invariant.load !3
  %.unpack24 = extractelement <2 x double> %25, i32 0
  %.unpack225 = extractelement <2 x double> %25, i32 1
  %26 = select i1 %17, double %.unpack24, double 0.000000e+00
  %27 = mul nuw nsw i32 %11, 33
  %28 = add nuw nsw i32 %27, %18
  %29 = zext nneg i32 %28 to i64
  %30 = getelementptr inbounds double, ptr addrspace(3) @shared_1, i64 %29
  store double %26, ptr addrspace(3) %30, align 8
  %31 = select i1 %17, double %.unpack225, double 0.000000e+00
  %32 = getelementptr inbounds double, ptr addrspace(3) @shared_0, i64 %29
  store double %31, ptr addrspace(3) %32, align 8
  %33 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 1536
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !3
  %.unpack326 = extractelement <2 x double> %34, i32 0
  %.unpack527 = extractelement <2 x double> %34, i32 1
  %35 = select i1 %17, double %.unpack326, double 0.000000e+00
  %36 = getelementptr inbounds i8, ptr addrspace(3) %30, i64 32
  store double %35, ptr addrspace(3) %36, align 8
  %37 = select i1 %17, double %.unpack527, double 0.000000e+00
  %38 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 32
  store double %37, ptr addrspace(3) %38, align 8
  %39 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 3072
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !3
  %.unpack628 = extractelement <2 x double> %40, i32 0
  %.unpack829 = extractelement <2 x double> %40, i32 1
  %41 = select i1 %17, double %.unpack628, double 0.000000e+00
  %42 = getelementptr inbounds i8, ptr addrspace(3) %30, i64 64
  store double %41, ptr addrspace(3) %42, align 8
  %43 = select i1 %17, double %.unpack829, double 0.000000e+00
  %44 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 64
  store double %43, ptr addrspace(3) %44, align 8
  %45 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 4608
  %46 = load <2 x double>, ptr addrspace(1) %45, align 16, !invariant.load !3
  %.unpack930 = extractelement <2 x double> %46, i32 0
  %.unpack1131 = extractelement <2 x double> %46, i32 1
  %47 = select i1 %17, double %.unpack930, double 0.000000e+00
  %48 = getelementptr inbounds i8, ptr addrspace(3) %30, i64 96
  store double %47, ptr addrspace(3) %48, align 8
  %49 = select i1 %17, double %.unpack1131, double 0.000000e+00
  %50 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 96
  store double %49, ptr addrspace(3) %50, align 8
  %51 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 6144
  %52 = load <2 x double>, ptr addrspace(1) %51, align 16, !invariant.load !3
  %.unpack1232 = extractelement <2 x double> %52, i32 0
  %.unpack1433 = extractelement <2 x double> %52, i32 1
  %53 = select i1 %17, double %.unpack1232, double 0.000000e+00
  %54 = getelementptr inbounds i8, ptr addrspace(3) %30, i64 128
  store double %53, ptr addrspace(3) %54, align 8
  %55 = select i1 %17, double %.unpack1433, double 0.000000e+00
  %56 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 128
  store double %55, ptr addrspace(3) %56, align 8
  %57 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 7680
  %58 = load <2 x double>, ptr addrspace(1) %57, align 16, !invariant.load !3
  %.unpack1534 = extractelement <2 x double> %58, i32 0
  %.unpack1735 = extractelement <2 x double> %58, i32 1
  %59 = select i1 %17, double %.unpack1534, double 0.000000e+00
  %60 = getelementptr inbounds i8, ptr addrspace(3) %30, i64 160
  store double %59, ptr addrspace(3) %60, align 8
  %61 = select i1 %17, double %.unpack1735, double 0.000000e+00
  %62 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 160
  store double %61, ptr addrspace(3) %62, align 8
  %63 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 9216
  %64 = load <2 x double>, ptr addrspace(1) %63, align 16, !invariant.load !3
  %.unpack1836 = extractelement <2 x double> %64, i32 0
  %.unpack2037 = extractelement <2 x double> %64, i32 1
  %65 = select i1 %17, double %.unpack1836, double 0.000000e+00
  %66 = getelementptr inbounds i8, ptr addrspace(3) %30, i64 192
  store double %65, ptr addrspace(3) %66, align 8
  %67 = select i1 %17, double %.unpack2037, double 0.000000e+00
  %68 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 192
  store double %67, ptr addrspace(3) %68, align 8
  %69 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 10752
  %70 = load <2 x double>, ptr addrspace(1) %69, align 16, !invariant.load !3
  %.unpack2138 = extractelement <2 x double> %70, i32 0
  %.unpack2339 = extractelement <2 x double> %70, i32 1
  %71 = select i1 %17, double %.unpack2138, double 0.000000e+00
  %72 = getelementptr inbounds i8, ptr addrspace(3) %30, i64 224
  store double %71, ptr addrspace(3) %72, align 8
  %73 = select i1 %17, double %.unpack2339, double 0.000000e+00
  %74 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 224
  store double %73, ptr addrspace(3) %74, align 8
  br label %75

75:                                               ; preds = %._crit_edge, %13
  %.pre-phi = phi i32 [ %.pre, %._crit_edge ], [ %18, %13 ]
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %76 = mul nuw nsw i32 %.pre-phi, 33
  %77 = add nuw nsw i32 %76, %11
  %78 = zext nneg i32 %77 to i64
  %79 = getelementptr inbounds double, ptr addrspace(3) @shared_1, i64 %78
  %80 = load double, ptr addrspace(3) %79, align 8
  %81 = getelementptr inbounds double, ptr addrspace(3) @shared_0, i64 %78
  %82 = load double, ptr addrspace(3) %81, align 8
  %83 = mul nuw nsw i32 %.pre-phi, 10752
  %84 = shl nuw nsw i32 %10, 5
  %85 = add nuw nsw i32 %83, %84
  %86 = or disjoint i32 %85, %11
  %87 = zext nneg i32 %86 to i64
  %88 = getelementptr inbounds double, ptr addrspace(1) %7, i64 %87
  store double %80, ptr addrspace(1) %88, align 8
  %89 = getelementptr inbounds double, ptr addrspace(1) %8, i64 %87
  store double %82, ptr addrspace(1) %89, align 8
  %90 = getelementptr inbounds i8, ptr addrspace(3) %79, i64 1056
  %91 = load double, ptr addrspace(3) %90, align 8
  %92 = getelementptr inbounds i8, ptr addrspace(3) %81, i64 1056
  %93 = load double, ptr addrspace(3) %92, align 8
  %94 = getelementptr inbounds i8, ptr addrspace(1) %88, i64 344064
  store double %91, ptr addrspace(1) %94, align 8
  %95 = getelementptr inbounds i8, ptr addrspace(1) %89, i64 344064
  store double %93, ptr addrspace(1) %95, align 8
  %96 = getelementptr inbounds i8, ptr addrspace(3) %79, i64 2112
  %97 = load double, ptr addrspace(3) %96, align 8
  %98 = getelementptr inbounds i8, ptr addrspace(3) %81, i64 2112
  %99 = load double, ptr addrspace(3) %98, align 8
  %100 = getelementptr inbounds i8, ptr addrspace(1) %88, i64 688128
  store double %97, ptr addrspace(1) %100, align 8
  %101 = getelementptr inbounds i8, ptr addrspace(1) %89, i64 688128
  store double %99, ptr addrspace(1) %101, align 8
  %102 = getelementptr inbounds i8, ptr addrspace(3) %79, i64 3168
  %103 = load double, ptr addrspace(3) %102, align 8
  %104 = getelementptr inbounds i8, ptr addrspace(3) %81, i64 3168
  %105 = load double, ptr addrspace(3) %104, align 8
  %106 = getelementptr inbounds i8, ptr addrspace(1) %88, i64 1032192
  store double %103, ptr addrspace(1) %106, align 8
  %107 = getelementptr inbounds i8, ptr addrspace(1) %89, i64 1032192
  store double %105, ptr addrspace(1) %107, align 8
  %108 = getelementptr inbounds i8, ptr addrspace(3) %79, i64 4224
  %109 = load double, ptr addrspace(3) %108, align 8
  %110 = getelementptr inbounds i8, ptr addrspace(3) %81, i64 4224
  %111 = load double, ptr addrspace(3) %110, align 8
  %112 = getelementptr inbounds i8, ptr addrspace(1) %88, i64 1376256
  store double %109, ptr addrspace(1) %112, align 8
  %113 = getelementptr inbounds i8, ptr addrspace(1) %89, i64 1376256
  store double %111, ptr addrspace(1) %113, align 8
  %114 = getelementptr inbounds i8, ptr addrspace(3) %79, i64 5280
  %115 = load double, ptr addrspace(3) %114, align 8
  %116 = getelementptr inbounds i8, ptr addrspace(3) %81, i64 5280
  %117 = load double, ptr addrspace(3) %116, align 8
  %118 = getelementptr inbounds i8, ptr addrspace(1) %88, i64 1720320
  store double %115, ptr addrspace(1) %118, align 8
  %119 = getelementptr inbounds i8, ptr addrspace(1) %89, i64 1720320
  store double %117, ptr addrspace(1) %119, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion_2(ptr noalias readonly align 16 captures(none) dereferenceable(24772608) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(12386304) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(12386304) %2) local_unnamed_addr #2 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %9 = udiv i32 %8, 5
  %10 = mul i32 %9, 5
  %.decomposed = sub i32 %8, %10
  %11 = shl nuw nsw i32 %.decomposed, 5
  %12 = and i32 %7, 31
  %13 = or disjoint i32 %11, %12
  %14 = icmp samesign ult i32 %13, 144
  br i1 %14, label %15, label %._crit_edge

._crit_edge:                                      ; preds = %3
  %.pre = lshr i32 %7, 5
  br label %58

15:                                               ; preds = %3
  %16 = mul nuw nsw i32 %9, 4608
  %17 = lshr i32 %7, 5
  %18 = mul nuw nsw i32 %17, 144
  %19 = or disjoint i32 %16, %12
  %20 = or disjoint i32 %19, %11
  %21 = add nuw nsw i32 %20, %18
  %22 = zext nneg i32 %21 to i64
  %23 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %22
  %24 = load <2 x double>, ptr addrspace(1) %23, align 16, !invariant.load !3
  %.unpack28 = extractelement <2 x double> %24, i32 0
  %.unpack229 = extractelement <2 x double> %24, i32 1
  %25 = mul nuw nsw i32 %12, 33
  %26 = add nuw nsw i32 %25, %17
  %27 = zext nneg i32 %26 to i64
  %28 = getelementptr inbounds double, ptr addrspace(3) @shared_11, i64 %27
  store double %.unpack28, ptr addrspace(3) %28, align 8
  %29 = getelementptr inbounds double, ptr addrspace(3) @shared_02, i64 %27
  store double %.unpack229, ptr addrspace(3) %29, align 8
  %30 = getelementptr inbounds i8, ptr addrspace(1) %23, i64 9216
  %31 = load <2 x double>, ptr addrspace(1) %30, align 16, !invariant.load !3
  %.unpack330 = extractelement <2 x double> %31, i32 0
  %.unpack531 = extractelement <2 x double> %31, i32 1
  %32 = getelementptr inbounds i8, ptr addrspace(3) %28, i64 32
  store double %.unpack330, ptr addrspace(3) %32, align 8
  %33 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 32
  store double %.unpack531, ptr addrspace(3) %33, align 8
  %34 = getelementptr inbounds i8, ptr addrspace(1) %23, i64 18432
  %35 = load <2 x double>, ptr addrspace(1) %34, align 16, !invariant.load !3
  %.unpack632 = extractelement <2 x double> %35, i32 0
  %.unpack833 = extractelement <2 x double> %35, i32 1
  %36 = getelementptr inbounds i8, ptr addrspace(3) %28, i64 64
  store double %.unpack632, ptr addrspace(3) %36, align 8
  %37 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 64
  store double %.unpack833, ptr addrspace(3) %37, align 8
  %38 = getelementptr inbounds i8, ptr addrspace(1) %23, i64 27648
  %39 = load <2 x double>, ptr addrspace(1) %38, align 16, !invariant.load !3
  %.unpack934 = extractelement <2 x double> %39, i32 0
  %.unpack1135 = extractelement <2 x double> %39, i32 1
  %40 = getelementptr inbounds i8, ptr addrspace(3) %28, i64 96
  store double %.unpack934, ptr addrspace(3) %40, align 8
  %41 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 96
  store double %.unpack1135, ptr addrspace(3) %41, align 8
  %42 = getelementptr inbounds i8, ptr addrspace(1) %23, i64 36864
  %43 = load <2 x double>, ptr addrspace(1) %42, align 16, !invariant.load !3
  %.unpack1236 = extractelement <2 x double> %43, i32 0
  %.unpack1437 = extractelement <2 x double> %43, i32 1
  %44 = getelementptr inbounds i8, ptr addrspace(3) %28, i64 128
  store double %.unpack1236, ptr addrspace(3) %44, align 8
  %45 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 128
  store double %.unpack1437, ptr addrspace(3) %45, align 8
  %46 = getelementptr inbounds i8, ptr addrspace(1) %23, i64 46080
  %47 = load <2 x double>, ptr addrspace(1) %46, align 16, !invariant.load !3
  %.unpack1538 = extractelement <2 x double> %47, i32 0
  %.unpack1739 = extractelement <2 x double> %47, i32 1
  %48 = getelementptr inbounds i8, ptr addrspace(3) %28, i64 160
  store double %.unpack1538, ptr addrspace(3) %48, align 8
  %49 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 160
  store double %.unpack1739, ptr addrspace(3) %49, align 8
  %50 = getelementptr inbounds i8, ptr addrspace(1) %23, i64 55296
  %51 = load <2 x double>, ptr addrspace(1) %50, align 16, !invariant.load !3
  %.unpack1840 = extractelement <2 x double> %51, i32 0
  %.unpack2041 = extractelement <2 x double> %51, i32 1
  %52 = getelementptr inbounds i8, ptr addrspace(3) %28, i64 192
  store double %.unpack1840, ptr addrspace(3) %52, align 8
  %53 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 192
  store double %.unpack2041, ptr addrspace(3) %53, align 8
  %54 = getelementptr inbounds i8, ptr addrspace(1) %23, i64 64512
  %55 = load <2 x double>, ptr addrspace(1) %54, align 16, !invariant.load !3
  %.unpack2142 = extractelement <2 x double> %55, i32 0
  %.unpack2343 = extractelement <2 x double> %55, i32 1
  %56 = getelementptr inbounds i8, ptr addrspace(3) %28, i64 224
  store double %.unpack2142, ptr addrspace(3) %56, align 8
  %57 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 224
  store double %.unpack2343, ptr addrspace(3) %57, align 8
  br label %58

58:                                               ; preds = %._crit_edge, %15
  %.pre-phi = phi i32 [ %.pre, %._crit_edge ], [ %17, %15 ]
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %59 = mul nuw nsw i32 %.pre-phi, 33
  %60 = add nuw nsw i32 %59, %12
  %61 = zext nneg i32 %60 to i64
  %62 = getelementptr inbounds double, ptr addrspace(3) @shared_11, i64 %61
  %63 = load double, ptr addrspace(3) %62, align 8
  %64 = getelementptr inbounds double, ptr addrspace(3) @shared_02, i64 %61
  %65 = load double, ptr addrspace(3) %64, align 8
  %66 = mul nuw nsw i32 %.decomposed, 344064
  %67 = shl nuw nsw i32 %9, 5
  %68 = mul nuw nsw i32 %.pre-phi, 10752
  %69 = or disjoint i32 %67, %66
  %70 = or disjoint i32 %69, %12
  %71 = add nuw nsw i32 %70, %68
  %72 = zext nneg i32 %71 to i64
  %73 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %72
  store double %63, ptr addrspace(1) %73, align 8
  %74 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %72
  store double %65, ptr addrspace(1) %74, align 8
  %75 = getelementptr inbounds i8, ptr addrspace(3) %62, i64 1056
  %76 = load double, ptr addrspace(3) %75, align 8
  %77 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 1056
  %78 = load double, ptr addrspace(3) %77, align 8
  %79 = getelementptr inbounds i8, ptr addrspace(1) %73, i64 344064
  store double %76, ptr addrspace(1) %79, align 8
  %80 = getelementptr inbounds i8, ptr addrspace(1) %74, i64 344064
  store double %78, ptr addrspace(1) %80, align 8
  %81 = getelementptr inbounds i8, ptr addrspace(3) %62, i64 2112
  %82 = load double, ptr addrspace(3) %81, align 8
  %83 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 2112
  %84 = load double, ptr addrspace(3) %83, align 8
  %85 = getelementptr inbounds i8, ptr addrspace(1) %73, i64 688128
  store double %82, ptr addrspace(1) %85, align 8
  %86 = getelementptr inbounds i8, ptr addrspace(1) %74, i64 688128
  store double %84, ptr addrspace(1) %86, align 8
  %87 = getelementptr inbounds i8, ptr addrspace(3) %62, i64 3168
  %88 = load double, ptr addrspace(3) %87, align 8
  %89 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 3168
  %90 = load double, ptr addrspace(3) %89, align 8
  %91 = getelementptr inbounds i8, ptr addrspace(1) %73, i64 1032192
  store double %88, ptr addrspace(1) %91, align 8
  %92 = getelementptr inbounds i8, ptr addrspace(1) %74, i64 1032192
  store double %90, ptr addrspace(1) %92, align 8
  %93 = icmp samesign ult i32 %.decomposed, 4
  br i1 %93, label %.critedge, label %.critedge25

.critedge:                                        ; preds = %58
  %94 = getelementptr inbounds i8, ptr addrspace(3) %62, i64 4224
  %95 = load double, ptr addrspace(3) %94, align 8
  %96 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 4224
  %97 = load double, ptr addrspace(3) %96, align 8
  %98 = getelementptr inbounds i8, ptr addrspace(1) %73, i64 1376256
  store double %95, ptr addrspace(1) %98, align 8
  %99 = getelementptr inbounds i8, ptr addrspace(1) %74, i64 1376256
  store double %97, ptr addrspace(1) %99, align 8
  %100 = getelementptr inbounds i8, ptr addrspace(3) %62, i64 5280
  %101 = load double, ptr addrspace(3) %100, align 8
  %102 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 5280
  %103 = load double, ptr addrspace(3) %102, align 8
  %104 = getelementptr inbounds i8, ptr addrspace(1) %73, i64 1720320
  store double %101, ptr addrspace(1) %104, align 8
  %105 = getelementptr inbounds i8, ptr addrspace(1) %74, i64 1720320
  store double %103, ptr addrspace(1) %105, align 8
  %106 = getelementptr inbounds i8, ptr addrspace(3) %62, i64 6336
  %107 = load double, ptr addrspace(3) %106, align 8
  %108 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 6336
  %109 = load double, ptr addrspace(3) %108, align 8
  %110 = getelementptr inbounds i8, ptr addrspace(1) %73, i64 2064384
  store double %107, ptr addrspace(1) %110, align 8
  %111 = getelementptr inbounds i8, ptr addrspace(1) %74, i64 2064384
  store double %109, ptr addrspace(1) %111, align 8
  %112 = getelementptr inbounds i8, ptr addrspace(3) %62, i64 7392
  %113 = load double, ptr addrspace(3) %112, align 8
  %114 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 7392
  %115 = load double, ptr addrspace(3) %114, align 8
  %116 = getelementptr inbounds i8, ptr addrspace(1) %73, i64 2408448
  store double %113, ptr addrspace(1) %116, align 8
  %117 = getelementptr inbounds i8, ptr addrspace(1) %74, i64 2408448
  store double %115, ptr addrspace(1) %117, align 8
  br label %.critedge25

.critedge25:                                      ; preds = %58, %.critedge
  ret void
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @input_scatter_fusion(ptr noalias align 256 captures(none) dereferenceable(12386304) %0, ptr noalias readonly align 256 captures(none) dereferenceable(2064384) %1, ptr noalias readonly align 256 captures(none) dereferenceable(4) %2, ptr noalias readnone align 256 captures(none) dereferenceable(12386304) %3) local_unnamed_addr #4 {
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !7
  %9 = udiv i32 %8, 21
  %.val2 = load i32, ptr addrspace(1) %5, align 256, !invariant.load !3
  %10 = and i32 %.val2, 2
  %.not.i = icmp eq i32 %10, 0
  %.neg1.i = select i1 %.not.i, i64 0, i64 -12
  %11 = zext nneg i32 %9 to i64
  %12 = add nsw i64 %.neg1.i, %11
  %13 = tail call i64 @llvm.smax.i64(i64 %12, i64 0)
  %14 = tail call i64 @llvm.umin.i64(i64 %13, i64 11)
  %15 = trunc nuw nsw i64 %14 to i32
  %16 = trunc i32 %.val2 to i1
  %.neg.i = select i1 %16, i64 -12, i64 0
  %17 = add nsw i64 %.neg.i, %11
  %18 = tail call i64 @llvm.smax.i64(i64 %17, i64 0)
  %19 = tail call i64 @llvm.umin.i64(i64 %18, i64 11)
  %20 = trunc nuw nsw i64 %19 to i32
  %21 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %22 = shl nuw nsw i32 %21, 2
  %23 = shl nuw nsw i32 %8, 9
  %24 = or disjoint i32 %22, %23
  %25 = zext nneg i32 %24 to i64
  %26 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %25
  %27 = load <4 x double>, ptr addrspace(1) %26, align 32, !invariant.load !3
  %28 = extractelement <4 x double> %27, i64 0
  %29 = mul i32 %9, 21
  %.decomposed = sub i32 %8, %29
  %30 = shl nuw nsw i32 %.decomposed, 9
  %31 = mul nuw nsw i32 %15, 129024
  %32 = mul nuw nsw i32 %20, 10752
  %33 = or disjoint i32 %30, %22
  %34 = add nuw nsw i32 %33, %32
  %35 = add nuw nsw i32 %34, %31
  %36 = zext nneg i32 %35 to i64
  %37 = getelementptr inbounds double, ptr addrspace(1) %7, i64 %36
  %38 = atomicrmw fadd ptr addrspace(1) %37, double %28 monotonic, align 8
  %39 = extractelement <4 x double> %27, i64 1
  %40 = getelementptr inbounds i8, ptr addrspace(1) %37, i64 8
  %41 = atomicrmw fadd ptr addrspace(1) %40, double %39 monotonic, align 8
  %42 = extractelement <4 x double> %27, i64 2
  %43 = getelementptr inbounds i8, ptr addrspace(1) %37, i64 16
  %44 = atomicrmw fadd ptr addrspace(1) %43, double %42 monotonic, align 8
  %45 = extractelement <4 x double> %27, i64 3
  %46 = getelementptr inbounds i8, ptr addrspace(1) %37, i64 24
  %47 = atomicrmw fadd ptr addrspace(1) %46, double %45 monotonic, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #5

; Function Attrs: norecurse nounwind
define ptx_kernel void @wrapped_transpose(ptr noalias readonly align 256 captures(none) dereferenceable(12386304) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(12386304) %1) local_unnamed_addr #2 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %7 = lshr i32 %6, 4
  %8 = urem i32 %7, 21
  %9 = shl nuw nsw i32 %8, 9
  %10 = and i32 %6, 15
  %11 = shl nuw nsw i32 %10, 5
  %12 = udiv i32 %6, 336
  %13 = mul nuw nsw i32 %12, 344064
  %14 = lshr i32 %5, 5
  %15 = mul nuw nsw i32 %14, 10752
  %16 = and i32 %5, 31
  %17 = or disjoint i32 %11, %13
  %18 = or disjoint i32 %17, %16
  %19 = add nuw nsw i32 %18, %15
  %20 = add nuw nsw i32 %19, %9
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds double, ptr addrspace(1) %3, i64 %21
  %23 = load double, ptr addrspace(1) %22, align 8, !invariant.load !3
  %24 = mul nuw nsw i32 %16, 33
  %25 = add nuw nsw i32 %24, %14
  %26 = zext nneg i32 %25 to i64
  %27 = getelementptr inbounds double, ptr addrspace(3) @shared_03, i64 %26
  store double %23, ptr addrspace(3) %27, align 8
  %28 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 344064
  %29 = load double, ptr addrspace(1) %28, align 8, !invariant.load !3
  %30 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 32
  store double %29, ptr addrspace(3) %30, align 8
  %31 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 688128
  %32 = load double, ptr addrspace(1) %31, align 8, !invariant.load !3
  %33 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 64
  store double %32, ptr addrspace(3) %33, align 8
  %34 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 1032192
  %35 = load double, ptr addrspace(1) %34, align 8, !invariant.load !3
  %36 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 96
  store double %35, ptr addrspace(3) %36, align 8
  %37 = icmp samesign ult i32 %6, 1344
  br i1 %37, label %.critedge, label %.critedge4

.critedge:                                        ; preds = %2
  %38 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 128
  %39 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 1376256
  %40 = load double, ptr addrspace(1) %39, align 8, !invariant.load !3
  store double %40, ptr addrspace(3) %38, align 8
  %41 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 1720320
  %42 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 160
  %43 = load double, ptr addrspace(1) %41, align 8, !invariant.load !3
  store double %43, ptr addrspace(3) %42, align 8
  %44 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 192
  %45 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 2064384
  %46 = load double, ptr addrspace(1) %45, align 8, !invariant.load !3
  store double %46, ptr addrspace(3) %44, align 8
  %47 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 2408448
  %48 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 224
  %49 = load double, ptr addrspace(1) %47, align 8, !invariant.load !3
  store double %49, ptr addrspace(3) %48, align 8
  br label %.critedge4

.critedge4:                                       ; preds = %2, %.critedge
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %50 = shl nuw nsw i32 %12, 5
  %51 = or disjoint i32 %50, %16
  %52 = icmp samesign ult i32 %51, 144
  br i1 %52, label %53, label %89

53:                                               ; preds = %.critedge4
  %54 = mul nuw nsw i32 %14, 33
  %55 = add nuw nsw i32 %54, %16
  %56 = zext nneg i32 %55 to i64
  %57 = getelementptr inbounds double, ptr addrspace(3) @shared_03, i64 %56
  %58 = load double, ptr addrspace(3) %57, align 8
  %59 = mul nuw nsw i32 %8, 73728
  %60 = mul nuw nsw i32 %10, 4608
  %61 = mul nuw nsw i32 %14, 144
  %62 = or disjoint i32 %60, %16
  %63 = or disjoint i32 %62, %50
  %64 = add nuw nsw i32 %63, %61
  %65 = add nuw nsw i32 %64, %59
  %66 = zext nneg i32 %65 to i64
  %67 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %66
  store double %58, ptr addrspace(1) %67, align 8
  %68 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 1056
  %69 = load double, ptr addrspace(3) %68, align 8
  %70 = getelementptr inbounds i8, ptr addrspace(1) %67, i64 4608
  store double %69, ptr addrspace(1) %70, align 8
  %71 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 2112
  %72 = load double, ptr addrspace(3) %71, align 8
  %73 = getelementptr inbounds i8, ptr addrspace(1) %67, i64 9216
  store double %72, ptr addrspace(1) %73, align 8
  %74 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 3168
  %75 = load double, ptr addrspace(3) %74, align 8
  %76 = getelementptr inbounds i8, ptr addrspace(1) %67, i64 13824
  store double %75, ptr addrspace(1) %76, align 8
  %77 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 4224
  %78 = load double, ptr addrspace(3) %77, align 8
  %79 = getelementptr inbounds i8, ptr addrspace(1) %67, i64 18432
  store double %78, ptr addrspace(1) %79, align 8
  %80 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 5280
  %81 = load double, ptr addrspace(3) %80, align 8
  %82 = getelementptr inbounds i8, ptr addrspace(1) %67, i64 23040
  store double %81, ptr addrspace(1) %82, align 8
  %83 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 6336
  %84 = load double, ptr addrspace(3) %83, align 8
  %85 = getelementptr inbounds i8, ptr addrspace(1) %67, i64 27648
  store double %84, ptr addrspace(1) %85, align 8
  %86 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 7392
  %87 = load double, ptr addrspace(3) %86, align 8
  %88 = getelementptr inbounds i8, ptr addrspace(1) %67, i64 32256
  store double %87, ptr addrspace(1) %88, align 8
  br label %89

89:                                               ; preds = %53, %.critedge4
  ret void
}

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(12386304) %0, ptr noalias readonly align 256 captures(none) dereferenceable(12386304) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(24772608) %2) local_unnamed_addr #2 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %9 = lshr i32 %8, 4
  %10 = urem i32 %9, 21
  %11 = shl nuw nsw i32 %10, 9
  %12 = and i32 %8, 15
  %13 = shl nuw nsw i32 %12, 5
  %14 = udiv i32 %8, 336
  %15 = mul nuw nsw i32 %14, 344064
  %16 = lshr i32 %7, 5
  %17 = mul nuw nsw i32 %16, 10752
  %18 = and i32 %7, 31
  %19 = or disjoint i32 %13, %15
  %20 = or disjoint i32 %19, %18
  %21 = add nuw nsw i32 %20, %17
  %22 = add nuw nsw i32 %21, %11
  %23 = zext nneg i32 %22 to i64
  %24 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %23
  %25 = load double, ptr addrspace(1) %24, align 8, !invariant.load !3
  %26 = mul nuw nsw i32 %18, 33
  %27 = add nuw nsw i32 %26, %16
  %28 = zext nneg i32 %27 to i64
  %29 = getelementptr inbounds double, ptr addrspace(3) @shared_04, i64 %28
  store double %25, ptr addrspace(3) %29, align 8
  %30 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 344064
  %31 = load double, ptr addrspace(1) %30, align 8, !invariant.load !3
  %32 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 32
  store double %31, ptr addrspace(3) %32, align 8
  %33 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 688128
  %34 = load double, ptr addrspace(1) %33, align 8, !invariant.load !3
  %35 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 64
  store double %34, ptr addrspace(3) %35, align 8
  %36 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 1032192
  %37 = load double, ptr addrspace(1) %36, align 8, !invariant.load !3
  %38 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 96
  store double %37, ptr addrspace(3) %38, align 8
  %39 = icmp samesign ult i32 %8, 1344
  br i1 %39, label %.critedge, label %.critedge20

.critedge:                                        ; preds = %3
  %40 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 128
  %41 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 1376256
  %42 = load double, ptr addrspace(1) %41, align 8, !invariant.load !3
  store double %42, ptr addrspace(3) %40, align 8
  %43 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 1720320
  %44 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 160
  %45 = load double, ptr addrspace(1) %43, align 8, !invariant.load !3
  store double %45, ptr addrspace(3) %44, align 8
  %46 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 192
  %47 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 2064384
  %48 = load double, ptr addrspace(1) %47, align 8, !invariant.load !3
  store double %48, ptr addrspace(3) %46, align 8
  %49 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 2408448
  %50 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 224
  %51 = load double, ptr addrspace(1) %49, align 8, !invariant.load !3
  store double %51, ptr addrspace(3) %50, align 8
  br label %.critedge20

.critedge20:                                      ; preds = %3, %.critedge
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %52 = shl nuw nsw i32 %14, 5
  %53 = or disjoint i32 %52, %18
  %54 = icmp samesign ult i32 %53, 144
  br i1 %54, label %55, label %123

55:                                               ; preds = %.critedge20
  %56 = mul nuw nsw i32 %16, 33
  %57 = add nuw nsw i32 %56, %18
  %58 = zext nneg i32 %57 to i64
  %59 = getelementptr inbounds double, ptr addrspace(3) @shared_04, i64 %58
  %60 = load double, ptr addrspace(3) %59, align 8
  %61 = mul nuw nsw i32 %10, 73728
  %62 = mul nuw nsw i32 %12, 4608
  %63 = mul nuw nsw i32 %16, 144
  %64 = or disjoint i32 %62, %18
  %65 = or disjoint i32 %64, %52
  %66 = add nuw nsw i32 %65, %63
  %67 = add nuw nsw i32 %66, %61
  %68 = zext nneg i32 %67 to i64
  %69 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %68
  %70 = load double, ptr addrspace(1) %69, align 8, !invariant.load !3
  %71 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %68
  %72 = insertelement <2 x double> poison, double %70, i32 0
  %73 = insertelement <2 x double> %72, double %60, i32 1
  store <2 x double> %73, ptr addrspace(1) %71, align 16
  %74 = getelementptr inbounds i8, ptr addrspace(3) %59, i64 1056
  %75 = load double, ptr addrspace(3) %74, align 8
  %76 = getelementptr inbounds i8, ptr addrspace(1) %69, i64 4608
  %77 = load double, ptr addrspace(1) %76, align 8, !invariant.load !3
  %78 = getelementptr inbounds i8, ptr addrspace(1) %71, i64 9216
  %79 = insertelement <2 x double> poison, double %77, i32 0
  %80 = insertelement <2 x double> %79, double %75, i32 1
  store <2 x double> %80, ptr addrspace(1) %78, align 16
  %81 = getelementptr inbounds i8, ptr addrspace(3) %59, i64 2112
  %82 = load double, ptr addrspace(3) %81, align 8
  %83 = getelementptr inbounds i8, ptr addrspace(1) %69, i64 9216
  %84 = load double, ptr addrspace(1) %83, align 8, !invariant.load !3
  %85 = getelementptr inbounds i8, ptr addrspace(1) %71, i64 18432
  %86 = insertelement <2 x double> poison, double %84, i32 0
  %87 = insertelement <2 x double> %86, double %82, i32 1
  store <2 x double> %87, ptr addrspace(1) %85, align 16
  %88 = getelementptr inbounds i8, ptr addrspace(3) %59, i64 3168
  %89 = load double, ptr addrspace(3) %88, align 8
  %90 = getelementptr inbounds i8, ptr addrspace(1) %69, i64 13824
  %91 = load double, ptr addrspace(1) %90, align 8, !invariant.load !3
  %92 = getelementptr inbounds i8, ptr addrspace(1) %71, i64 27648
  %93 = insertelement <2 x double> poison, double %91, i32 0
  %94 = insertelement <2 x double> %93, double %89, i32 1
  store <2 x double> %94, ptr addrspace(1) %92, align 16
  %95 = getelementptr inbounds i8, ptr addrspace(3) %59, i64 4224
  %96 = load double, ptr addrspace(3) %95, align 8
  %97 = getelementptr inbounds i8, ptr addrspace(1) %69, i64 18432
  %98 = load double, ptr addrspace(1) %97, align 8, !invariant.load !3
  %99 = getelementptr inbounds i8, ptr addrspace(1) %71, i64 36864
  %100 = insertelement <2 x double> poison, double %98, i32 0
  %101 = insertelement <2 x double> %100, double %96, i32 1
  store <2 x double> %101, ptr addrspace(1) %99, align 16
  %102 = getelementptr inbounds i8, ptr addrspace(3) %59, i64 5280
  %103 = load double, ptr addrspace(3) %102, align 8
  %104 = getelementptr inbounds i8, ptr addrspace(1) %69, i64 23040
  %105 = load double, ptr addrspace(1) %104, align 8, !invariant.load !3
  %106 = getelementptr inbounds i8, ptr addrspace(1) %71, i64 46080
  %107 = insertelement <2 x double> poison, double %105, i32 0
  %108 = insertelement <2 x double> %107, double %103, i32 1
  store <2 x double> %108, ptr addrspace(1) %106, align 16
  %109 = getelementptr inbounds i8, ptr addrspace(3) %59, i64 6336
  %110 = load double, ptr addrspace(3) %109, align 8
  %111 = getelementptr inbounds i8, ptr addrspace(1) %69, i64 27648
  %112 = load double, ptr addrspace(1) %111, align 8, !invariant.load !3
  %113 = getelementptr inbounds i8, ptr addrspace(1) %71, i64 55296
  %114 = insertelement <2 x double> poison, double %112, i32 0
  %115 = insertelement <2 x double> %114, double %110, i32 1
  store <2 x double> %115, ptr addrspace(1) %113, align 16
  %116 = getelementptr inbounds i8, ptr addrspace(3) %59, i64 7392
  %117 = load double, ptr addrspace(3) %116, align 8
  %118 = getelementptr inbounds i8, ptr addrspace(1) %69, i64 32256
  %119 = load double, ptr addrspace(1) %118, align 8, !invariant.load !3
  %120 = getelementptr inbounds i8, ptr addrspace(1) %71, i64 64512
  %121 = insertelement <2 x double> poison, double %119, i32 0
  %122 = insertelement <2 x double> %121, double %117, i32 1
  store <2 x double> %122, ptr addrspace(1) %120, align 16
  br label %123

123:                                              ; preds = %55, %.critedge20
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="24,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { norecurse nounwind "nvvm.reqntid"="128,1,1" }
attributes #3 = { convergent nocallback nounwind }
attributes #4 = { mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 24}
!3 = !{}
!4 = !{i32 0, i32 128}
!5 = !{i32 0, i32 336}
!6 = !{i32 0, i32 1680}
!7 = !{i32 0, i32 504}
