; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef
@shared_01 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef
@buffer_for_constant_111_0 = local_unnamed_addr addrspace(1) global [5079040 x i8] zeroinitializer, align 256
@buffer_for_constant_81_0 = local_unnamed_addr addrspace(1) global [5079040 x i8] zeroinitializer, align 256
@buffer_for_constant_68_0 = local_unnamed_addr addrspace(1) global [2048 x i8] zeroinitializer, align 256

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_real_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(5079040) %0, ptr noalias readonly align 256 captures(none) dereferenceable(4) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(1269760) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = load i32, ptr addrspace(1) %4, align 256, !invariant.load !4
  %12 = trunc i32 %11 to i1
  %13 = select i1 %12, i32 310, i32 0
  %14 = urem i32 %10, 310
  %15 = sub nuw nsw i32 %10, %14
  %16 = add nuw nsw i32 %15, %10
  %17 = add nuw nsw i32 %16, %13
  %18 = zext nneg i32 %17 to i64
  %19 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %18
  %.unpack = load double, ptr addrspace(1) %19, align 16, !invariant.load !4
  %20 = zext nneg i32 %10 to i64
  %21 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %20
  store double %.unpack, ptr addrspace(1) %21, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_negate_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(5079040) %0, ptr noalias readonly align 256 captures(none) dereferenceable(4) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(1269760) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = load i32, ptr addrspace(1) %4, align 256, !invariant.load !4
  %12 = trunc i32 %11 to i1
  %13 = select i1 %12, i32 310, i32 0
  %14 = urem i32 %10, 310
  %15 = sub nuw nsw i32 %10, %14
  %16 = add nuw nsw i32 %15, %10
  %17 = add nuw nsw i32 %16, %13
  %18 = zext nneg i32 %17 to i64
  %19 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %18
  %.elt1 = getelementptr inbounds i8, ptr addrspace(1) %19, i64 8
  %.unpack2 = load double, ptr addrspace(1) %.elt1, align 8, !invariant.load !4
  %20 = fneg double %.unpack2
  %21 = zext nneg i32 %10 to i64
  %22 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %21
  store double %20, ptr addrspace(1) %22, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_gather(ptr noalias readonly align 16 captures(none) dereferenceable(44590400) %0, ptr noalias readonly align 256 captures(none) dereferenceable(2048) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(787251200) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %8, 1
  %10 = shl nuw nsw i32 %7, 8
  %11 = or disjoint i32 %9, %10
  %12 = udiv i32 %11, 155
  %13 = and i32 %12, 511
  %14 = zext nneg i32 %13 to i64
  %15 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %14
  %16 = load i32, ptr addrspace(1) %15, align 4, !invariant.load !4
  %17 = tail call i32 @llvm.smax.i32(i32 %16, i32 0)
  %18 = tail call i32 @llvm.umin.i32(i32 %17, i32 28)
  %19 = shl nuw nsw i32 %8, 2
  %20 = shl nuw nsw i32 %7, 9
  %21 = or disjoint i32 %19, %20
  %22 = urem i32 %21, 310
  %23 = mul nuw nsw i32 %22, 310
  %24 = mul nuw nsw i32 %18, 96100
  %25 = udiv i32 %7, 310
  %26 = add nuw nsw i32 %24, %25
  %27 = add nuw nsw i32 %26, %23
  %28 = zext nneg i32 %27 to i64
  %29 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %28
  %30 = load <2 x double>, ptr addrspace(1) %29, align 16, !invariant.load !4
  %.unpack29 = extractelement <2 x double> %30, i32 0
  %.unpack230 = extractelement <2 x double> %30, i32 1
  %31 = zext nneg i32 %21 to i64
  %32 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %31
  %33 = insertelement <2 x double> poison, double %.unpack29, i32 0
  %34 = insertelement <2 x double> %33, double %.unpack230, i32 1
  store <2 x double> %34, ptr addrspace(1) %32, align 64
  %35 = or disjoint i32 %21, 1
  %36 = urem i32 %35, 310
  %37 = mul nuw nsw i32 %36, 310
  %38 = add nuw nsw i32 %26, %37
  %39 = zext nneg i32 %38 to i64
  %40 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %39
  %41 = load <2 x double>, ptr addrspace(1) %40, align 16, !invariant.load !4
  %.unpack527 = extractelement <2 x double> %41, i32 0
  %.unpack728 = extractelement <2 x double> %41, i32 1
  %42 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 16
  %43 = insertelement <2 x double> poison, double %.unpack527, i32 0
  %44 = insertelement <2 x double> %43, double %.unpack728, i32 1
  store <2 x double> %44, ptr addrspace(1) %42, align 16
  %45 = or disjoint i32 %11, 1
  %46 = udiv i32 %45, 155
  %47 = and i32 %46, 511
  %48 = zext nneg i32 %47 to i64
  %49 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %48
  %50 = load i32, ptr addrspace(1) %49, align 4, !invariant.load !4
  %51 = tail call i32 @llvm.smax.i32(i32 %50, i32 0)
  %52 = tail call i32 @llvm.umin.i32(i32 %51, i32 28)
  %53 = or disjoint i32 %21, 2
  %54 = urem i32 %53, 310
  %55 = mul nuw nsw i32 %54, 310
  %56 = mul nuw nsw i32 %52, 96100
  %57 = add nuw nsw i32 %55, %25
  %58 = add nuw nsw i32 %57, %56
  %59 = zext nneg i32 %58 to i64
  %60 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %59
  %61 = load <2 x double>, ptr addrspace(1) %60, align 16, !invariant.load !4
  %.unpack1025 = extractelement <2 x double> %61, i32 0
  %.unpack1226 = extractelement <2 x double> %61, i32 1
  %62 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 32
  %63 = insertelement <2 x double> poison, double %.unpack1025, i32 0
  %64 = insertelement <2 x double> %63, double %.unpack1226, i32 1
  store <2 x double> %64, ptr addrspace(1) %62, align 32
  %65 = or disjoint i32 %21, 3
  %66 = udiv i32 %65, 310
  %67 = and i32 %66, 511
  %68 = zext nneg i32 %67 to i64
  %69 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %68
  %70 = load i32, ptr addrspace(1) %69, align 4, !invariant.load !4
  %71 = tail call i32 @llvm.smax.i32(i32 %70, i32 0)
  %72 = tail call i32 @llvm.umin.i32(i32 %71, i32 28)
  %73 = mul i32 %66, 310
  %.decomposed = sub i32 %65, %73
  %74 = mul nuw nsw i32 %.decomposed, 310
  %75 = mul nuw nsw i32 %72, 96100
  %76 = add nuw nsw i32 %74, %25
  %77 = add nuw nsw i32 %76, %75
  %78 = zext nneg i32 %77 to i64
  %79 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %78
  %80 = load <2 x double>, ptr addrspace(1) %79, align 16, !invariant.load !4
  %.unpack1523 = extractelement <2 x double> %80, i32 0
  %.unpack1724 = extractelement <2 x double> %80, i32 1
  %81 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 48
  %82 = insertelement <2 x double> poison, double %.unpack1523, i32 0
  %83 = insertelement <2 x double> %82, double %.unpack1724, i32 1
  store <2 x double> %83, ptr addrspace(1) %81, align 16
  ret void
}

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(787251200) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(787251200) %1) local_unnamed_addr #3 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %7 = udiv i32 %6, 10
  %8 = mul i32 %7, 10
  %.decomposed = sub i32 %6, %8
  %9 = shl nuw nsw i32 %.decomposed, 5
  %10 = and i32 %5, 31
  %11 = or disjoint i32 %9, %10
  %12 = icmp samesign ult i32 %11, 310
  br i1 %12, label %13, label %._crit_edge

._crit_edge:                                      ; preds = %2
  %.pre = udiv i32 %6, 5120
  %.pre80 = urem i32 %.pre, 5
  %.pre82 = lshr i32 %5, 5
  br label %52

13:                                               ; preds = %2
  %14 = and i32 %7, 511
  %15 = mul nuw nsw i32 %14, 310
  %16 = udiv i32 %6, 5120
  %17 = urem i32 %16, 5
  %18 = mul nuw nsw i32 %17, 5079040
  %19 = udiv i32 %6, 25600
  %20 = mul nuw nsw i32 %19, 24601600
  %21 = lshr i32 %5, 5
  %22 = mul nuw nsw i32 %21, 158720
  %23 = or disjoint i32 %10, %20
  %24 = or disjoint i32 %23, %9
  %25 = add nuw nsw i32 %24, %22
  %26 = add nuw nsw i32 %25, %18
  %27 = add nuw nsw i32 %26, %15
  %28 = zext nneg i32 %27 to i64
  %29 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %28
  %30 = load <2 x double>, ptr addrspace(1) %29, align 16, !invariant.load !4
  %.unpack112 = extractelement <2 x double> %30, i32 0
  %.unpack2113 = extractelement <2 x double> %30, i32 1
  %31 = mul nuw nsw i32 %10, 33
  %32 = add nuw nsw i32 %31, %21
  %33 = zext nneg i32 %32 to i64
  %34 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %33
  store double %.unpack112, ptr addrspace(3) %34, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 8
  store double %.unpack2113, ptr addrspace(3) %.repack3, align 8
  %35 = sext i32 %27 to i64
  %36 = getelementptr { double, double }, ptr addrspace(1) %3, i64 %35
  %37 = getelementptr i8, ptr addrspace(1) %36, i64 10158080
  %38 = load <2 x double>, ptr addrspace(1) %37, align 16, !invariant.load !4
  %.unpack5102 = extractelement <2 x double> %38, i32 0
  %.unpack7103 = extractelement <2 x double> %38, i32 1
  %39 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 64
  store double %.unpack5102, ptr addrspace(3) %39, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 72
  store double %.unpack7103, ptr addrspace(3) %.repack8, align 8
  %40 = getelementptr i8, ptr addrspace(1) %36, i64 20316160
  %41 = load <2 x double>, ptr addrspace(1) %40, align 16, !invariant.load !4
  %.unpack10104 = extractelement <2 x double> %41, i32 0
  %.unpack12105 = extractelement <2 x double> %41, i32 1
  %42 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 128
  store double %.unpack10104, ptr addrspace(3) %42, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 136
  store double %.unpack12105, ptr addrspace(3) %.repack13, align 8
  %43 = getelementptr i8, ptr addrspace(1) %36, i64 30474240
  %44 = load <2 x double>, ptr addrspace(1) %43, align 16, !invariant.load !4
  %.unpack15106 = extractelement <2 x double> %44, i32 0
  %.unpack17107 = extractelement <2 x double> %44, i32 1
  %45 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 192
  store double %.unpack15106, ptr addrspace(3) %45, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 200
  store double %.unpack17107, ptr addrspace(3) %.repack18, align 8
  %46 = getelementptr i8, ptr addrspace(1) %36, i64 40632320
  %47 = load <2 x double>, ptr addrspace(1) %46, align 16, !invariant.load !4
  %.unpack20108 = extractelement <2 x double> %47, i32 0
  %.unpack22109 = extractelement <2 x double> %47, i32 1
  %48 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 256
  store double %.unpack20108, ptr addrspace(3) %48, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 264
  store double %.unpack22109, ptr addrspace(3) %.repack23, align 8
  %49 = getelementptr i8, ptr addrspace(1) %36, i64 50790400
  %50 = load <2 x double>, ptr addrspace(1) %49, align 16, !invariant.load !4
  %.unpack25110 = extractelement <2 x double> %50, i32 0
  %.unpack27111 = extractelement <2 x double> %50, i32 1
  %51 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 320
  store double %.unpack25110, ptr addrspace(3) %51, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %34, i64 328
  store double %.unpack27111, ptr addrspace(3) %.repack28, align 8
  br label %52

52:                                               ; preds = %._crit_edge, %13
  %.pre-phi83 = phi i32 [ %.pre82, %._crit_edge ], [ %21, %13 ]
  %.pre-phi81 = phi i32 [ %.pre80, %._crit_edge ], [ %17, %13 ]
  %53 = icmp ult i32 %11, 310
  %54 = shl nuw nsw i32 %.pre-phi81, 5
  %55 = or disjoint i32 %54, %.pre-phi83
  %56 = icmp samesign ult i32 %55, 131
  %57 = and i1 %53, %56
  br i1 %57, label %58, label %82

58:                                               ; preds = %52
  %59 = and i32 %7, 511
  %60 = mul nuw nsw i32 %59, 310
  %61 = mul nuw nsw i32 %.pre-phi81, 5079040
  %62 = udiv i32 %6, 25600
  %63 = mul nuw nsw i32 %62, 24601600
  %64 = mul nuw nsw i32 %.pre-phi83, 158720
  %65 = zext nneg i32 %60 to i64
  %66 = zext nneg i32 %61 to i64
  %67 = zext nneg i32 %64 to i64
  %68 = zext nneg i32 %63 to i64
  %69 = zext nneg i32 %11 to i64
  %70 = add i64 %69, %68
  %71 = add i64 %70, %67
  %72 = add i64 %71, %66
  %73 = add i64 %72, %65
  %74 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %73
  %75 = getelementptr inbounds i8, ptr addrspace(1) %74, i64 60948480
  %76 = load <2 x double>, ptr addrspace(1) %75, align 16, !invariant.load !4
  %.unpack30100 = extractelement <2 x double> %76, i32 0
  %.unpack32101 = extractelement <2 x double> %76, i32 1
  %77 = mul nuw nsw i32 %10, 33
  %78 = add nuw nsw i32 %77, %.pre-phi83
  %79 = zext nneg i32 %78 to i64
  %80 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %79
  %81 = getelementptr inbounds i8, ptr addrspace(3) %80, i64 384
  store double %.unpack30100, ptr addrspace(3) %81, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %80, i64 392
  store double %.unpack32101, ptr addrspace(3) %.repack33, align 8
  br label %82

82:                                               ; preds = %58, %52
  %83 = icmp ult i32 %11, 310
  %84 = icmp samesign ult i32 %55, 127
  %85 = and i1 %83, %84
  br i1 %85, label %86, label %110

86:                                               ; preds = %82
  %87 = and i32 %7, 511
  %88 = mul nuw nsw i32 %87, 310
  %89 = mul nuw nsw i32 %.pre-phi81, 5079040
  %90 = udiv i32 %6, 25600
  %91 = mul nuw nsw i32 %90, 24601600
  %92 = mul nuw nsw i32 %.pre-phi83, 158720
  %93 = zext nneg i32 %88 to i64
  %94 = zext nneg i32 %89 to i64
  %95 = zext nneg i32 %92 to i64
  %96 = zext nneg i32 %91 to i64
  %97 = zext nneg i32 %11 to i64
  %98 = add i64 %97, %96
  %99 = add i64 %98, %95
  %100 = add i64 %99, %94
  %101 = add i64 %100, %93
  %102 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %101
  %103 = getelementptr inbounds i8, ptr addrspace(1) %102, i64 71106560
  %104 = load <2 x double>, ptr addrspace(1) %103, align 16, !invariant.load !4
  %.unpack3598 = extractelement <2 x double> %104, i32 0
  %.unpack3799 = extractelement <2 x double> %104, i32 1
  %105 = mul nuw nsw i32 %10, 33
  %106 = add nuw nsw i32 %105, %.pre-phi83
  %107 = zext nneg i32 %106 to i64
  %108 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %107
  %109 = getelementptr inbounds i8, ptr addrspace(3) %108, i64 448
  store double %.unpack3598, ptr addrspace(3) %109, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %108, i64 456
  store double %.unpack3799, ptr addrspace(3) %.repack38, align 8
  br label %110

110:                                              ; preds = %86, %82
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %111 = or disjoint i32 %54, %10
  %112 = icmp samesign ult i32 %111, 155
  br i1 %112, label %113, label %151

113:                                              ; preds = %110
  %114 = mul nuw nsw i32 %.pre-phi83, 33
  %115 = add nuw nsw i32 %114, %10
  %116 = zext nneg i32 %115 to i64
  %117 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %116
  %.unpack40 = load double, ptr addrspace(3) %117, align 8
  %.elt41 = getelementptr inbounds i8, ptr addrspace(3) %117, i64 8
  %.unpack42 = load double, ptr addrspace(3) %.elt41, align 8
  %118 = and i32 %7, 511
  %119 = mul nuw nsw i32 %118, 96100
  %120 = mul nuw nsw i32 %.decomposed, 4960
  %121 = udiv i32 %6, 25600
  %122 = mul nuw nsw i32 %121, 48050
  %123 = mul nuw nsw i32 %.pre-phi83, 155
  %124 = or disjoint i32 %120, %10
  %125 = add nuw nsw i32 %124, %122
  %126 = add nuw nsw i32 %125, %123
  %127 = add nuw nsw i32 %126, %54
  %128 = add nuw nsw i32 %127, %119
  %129 = zext nneg i32 %128 to i64
  %130 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %129
  %131 = insertelement <2 x double> poison, double %.unpack40, i32 0
  %132 = insertelement <2 x double> %131, double %.unpack42, i32 1
  store <2 x double> %132, ptr addrspace(1) %130, align 16
  %133 = getelementptr inbounds i8, ptr addrspace(3) %117, i64 2112
  %.unpack45 = load double, ptr addrspace(3) %133, align 8
  %.elt46 = getelementptr inbounds i8, ptr addrspace(3) %117, i64 2120
  %.unpack47 = load double, ptr addrspace(3) %.elt46, align 8
  %134 = sext i32 %128 to i64
  %135 = getelementptr { double, double }, ptr addrspace(1) %4, i64 %134
  %136 = getelementptr i8, ptr addrspace(1) %135, i64 9920
  %137 = insertelement <2 x double> poison, double %.unpack45, i32 0
  %138 = insertelement <2 x double> %137, double %.unpack47, i32 1
  store <2 x double> %138, ptr addrspace(1) %136, align 16
  %139 = getelementptr inbounds i8, ptr addrspace(3) %117, i64 4224
  %.unpack50 = load double, ptr addrspace(3) %139, align 8
  %.elt51 = getelementptr inbounds i8, ptr addrspace(3) %117, i64 4232
  %.unpack52 = load double, ptr addrspace(3) %.elt51, align 8
  %140 = getelementptr i8, ptr addrspace(1) %135, i64 19840
  %141 = insertelement <2 x double> poison, double %.unpack50, i32 0
  %142 = insertelement <2 x double> %141, double %.unpack52, i32 1
  store <2 x double> %142, ptr addrspace(1) %140, align 16
  %143 = getelementptr inbounds i8, ptr addrspace(3) %117, i64 6336
  %.unpack55 = load double, ptr addrspace(3) %143, align 8
  %.elt56 = getelementptr inbounds i8, ptr addrspace(3) %117, i64 6344
  %.unpack57 = load double, ptr addrspace(3) %.elt56, align 8
  %144 = getelementptr i8, ptr addrspace(1) %135, i64 29760
  %145 = insertelement <2 x double> poison, double %.unpack55, i32 0
  %146 = insertelement <2 x double> %145, double %.unpack57, i32 1
  store <2 x double> %146, ptr addrspace(1) %144, align 16
  %147 = getelementptr inbounds i8, ptr addrspace(3) %117, i64 8448
  %.unpack60 = load double, ptr addrspace(3) %147, align 8
  %.elt61 = getelementptr inbounds i8, ptr addrspace(3) %117, i64 8456
  %.unpack62 = load double, ptr addrspace(3) %.elt61, align 8
  %148 = getelementptr i8, ptr addrspace(1) %135, i64 39680
  %149 = insertelement <2 x double> poison, double %.unpack60, i32 0
  %150 = insertelement <2 x double> %149, double %.unpack62, i32 1
  store <2 x double> %150, ptr addrspace(1) %148, align 16
  br label %151

151:                                              ; preds = %113, %110
  %152 = icmp ult i32 %111, 155
  %153 = or disjoint i32 %9, %.pre-phi83
  %154 = icmp samesign ult i32 %153, 290
  %155 = and i1 %154, %152
  br i1 %155, label %156, label %183

156:                                              ; preds = %151
  %157 = mul nuw nsw i32 %.pre-phi83, 33
  %158 = add nuw nsw i32 %157, %10
  %159 = zext nneg i32 %158 to i64
  %160 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %159
  %161 = getelementptr inbounds i8, ptr addrspace(3) %160, i64 10560
  %.unpack65 = load double, ptr addrspace(3) %161, align 8
  %.elt66 = getelementptr inbounds i8, ptr addrspace(3) %160, i64 10568
  %.unpack67 = load double, ptr addrspace(3) %.elt66, align 8
  %162 = and i32 %7, 511
  %163 = mul nuw nsw i32 %162, 96100
  %164 = mul nuw nsw i32 %.decomposed, 4960
  %165 = udiv i32 %6, 25600
  %166 = mul nuw nsw i32 %165, 48050
  %167 = mul nuw nsw i32 %.pre-phi83, 155
  %168 = zext nneg i32 %163 to i64
  %169 = zext nneg i32 %54 to i64
  %170 = zext nneg i32 %167 to i64
  %171 = zext nneg i32 %166 to i64
  %172 = zext nneg i32 %164 to i64
  %173 = zext nneg i32 %10 to i64
  %174 = add i64 %173, %172
  %175 = add i64 %174, %171
  %176 = add i64 %175, %170
  %177 = add i64 %176, %169
  %178 = add i64 %177, %168
  %179 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %178
  %180 = getelementptr inbounds i8, ptr addrspace(1) %179, i64 49600
  %181 = insertelement <2 x double> poison, double %.unpack65, i32 0
  %182 = insertelement <2 x double> %181, double %.unpack67, i32 1
  store <2 x double> %182, ptr addrspace(1) %180, align 16
  br label %183

183:                                              ; preds = %156, %151
  %184 = icmp ult i32 %111, 155
  %185 = icmp samesign ult i32 %153, 286
  %186 = and i1 %185, %184
  br i1 %186, label %187, label %214

187:                                              ; preds = %183
  %188 = mul nuw nsw i32 %.pre-phi83, 33
  %189 = add nuw nsw i32 %188, %10
  %190 = zext nneg i32 %189 to i64
  %191 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %190
  %192 = getelementptr inbounds i8, ptr addrspace(3) %191, i64 12672
  %.unpack70 = load double, ptr addrspace(3) %192, align 8
  %.elt71 = getelementptr inbounds i8, ptr addrspace(3) %191, i64 12680
  %.unpack72 = load double, ptr addrspace(3) %.elt71, align 8
  %193 = and i32 %7, 511
  %194 = mul nuw nsw i32 %193, 96100
  %195 = mul nuw nsw i32 %.decomposed, 4960
  %196 = udiv i32 %6, 25600
  %197 = mul nuw nsw i32 %196, 48050
  %198 = mul nuw nsw i32 %.pre-phi83, 155
  %199 = zext nneg i32 %194 to i64
  %200 = zext nneg i32 %54 to i64
  %201 = zext nneg i32 %198 to i64
  %202 = zext nneg i32 %197 to i64
  %203 = zext nneg i32 %195 to i64
  %204 = zext nneg i32 %10 to i64
  %205 = add i64 %204, %203
  %206 = add i64 %205, %202
  %207 = add i64 %206, %201
  %208 = add i64 %207, %200
  %209 = add i64 %208, %199
  %210 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %209
  %211 = getelementptr inbounds i8, ptr addrspace(1) %210, i64 59520
  %212 = insertelement <2 x double> poison, double %.unpack70, i32 0
  %213 = insertelement <2 x double> %212, double %.unpack72, i32 1
  store <2 x double> %213, ptr addrspace(1) %211, align 16
  br label %214

214:                                              ; preds = %187, %183
  %215 = icmp ult i32 %111, 155
  %216 = icmp samesign ult i32 %153, 282
  %217 = and i1 %216, %215
  br i1 %217, label %218, label %245

218:                                              ; preds = %214
  %219 = mul nuw nsw i32 %.pre-phi83, 33
  %220 = add nuw nsw i32 %219, %10
  %221 = zext nneg i32 %220 to i64
  %222 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %221
  %223 = getelementptr inbounds i8, ptr addrspace(3) %222, i64 14784
  %.unpack75 = load double, ptr addrspace(3) %223, align 8
  %.elt76 = getelementptr inbounds i8, ptr addrspace(3) %222, i64 14792
  %.unpack77 = load double, ptr addrspace(3) %.elt76, align 8
  %224 = and i32 %7, 511
  %225 = mul nuw nsw i32 %224, 96100
  %226 = mul nuw nsw i32 %.decomposed, 4960
  %227 = udiv i32 %6, 25600
  %228 = mul nuw nsw i32 %227, 48050
  %229 = mul nuw nsw i32 %.pre-phi83, 155
  %230 = zext nneg i32 %225 to i64
  %231 = zext nneg i32 %54 to i64
  %232 = zext nneg i32 %229 to i64
  %233 = zext nneg i32 %228 to i64
  %234 = zext nneg i32 %226 to i64
  %235 = zext nneg i32 %10 to i64
  %236 = add i64 %235, %234
  %237 = add i64 %236, %233
  %238 = add i64 %237, %232
  %239 = add i64 %238, %231
  %240 = add i64 %239, %230
  %241 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %240
  %242 = getelementptr inbounds i8, ptr addrspace(1) %241, i64 69440
  %243 = insertelement <2 x double> poison, double %.unpack75, i32 0
  %244 = insertelement <2 x double> %243, double %.unpack77, i32 1
  store <2 x double> %244, ptr addrspace(1) %242, align 16
  br label %245

245:                                              ; preds = %218, %214
  ret void
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_transpose_fusion_2(ptr noalias readonly align 256 captures(none) dereferenceable(5079040) %0, ptr noalias readonly align 256 captures(none) dereferenceable(787251200) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(787251200) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = udiv i32 %7, 155
  %10 = shl nuw nsw i32 %8, 2
  %11 = shl nuw nsw i32 %7, 9
  %12 = or disjoint i32 %11, %10
  %13 = zext nneg i32 %12 to i64
  %14 = udiv i32 %12, 155
  %15 = and i32 %14, 511
  %16 = shl nuw nsw i32 %9, 1
  %17 = zext nneg i32 %16 to i64
  %narrow = mul nuw nsw i32 %15, 9920
  %.idx27 = zext nneg i32 %narrow to i64
  %18 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %.idx27
  %19 = getelementptr inbounds i64, ptr addrspace(1) %18, i64 %17
  %20 = load <2 x i64>, ptr addrspace(1) %19, align 16, !invariant.load !4
  %21 = extractelement <2 x i64> %20, i32 0
  %22 = extractelement <2 x i64> %20, i32 1
  %23 = tail call i64 @llvm.smax.i64(i64 %21, i64 0)
  %24 = tail call i64 @llvm.umin.i64(i64 %23, i64 511)
  %25 = tail call i64 @llvm.smax.i64(i64 %22, i64 0)
  %26 = tail call i64 @llvm.umin.i64(i64 %25, i64 619)
  %27 = mul i32 %14, 155
  %.decomposed = sub i32 %12, %27
  %.zext32 = zext nneg i32 %.decomposed to i64
  %.idx = mul nuw nsw i64 %24, 1537600
  %28 = getelementptr inbounds i8, ptr addrspace(1) %5, i64 %.idx
  %.idx1 = mul nuw nsw i64 %26, 2480
  %29 = getelementptr inbounds i8, ptr addrspace(1) %28, i64 %.idx1
  %30 = getelementptr inbounds { double, double }, ptr addrspace(1) %29, i64 %.zext32
  %31 = load <2 x double>, ptr addrspace(1) %30, align 16, !invariant.load !4
  %.unpack57 = extractelement <2 x double> %31, i32 0
  %.unpack358 = extractelement <2 x double> %31, i32 1
  %32 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %13
  %33 = insertelement <2 x double> poison, double %.unpack57, i32 0
  %34 = insertelement <2 x double> %33, double %.unpack358, i32 1
  store <2 x double> %34, ptr addrspace(1) %32, align 64
  %35 = or disjoint i64 %13, 1
  %.lhs.trunc33 = trunc nuw nsw i64 %35 to i32
  %36 = udiv i32 %.lhs.trunc33, 155
  %37 = and i32 %36, 511
  %narrow45 = mul nuw nsw i32 %37, 9920
  %.idx28 = zext nneg i32 %narrow45 to i64
  %38 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %.idx28
  %39 = getelementptr inbounds i64, ptr addrspace(1) %38, i64 %17
  %40 = load <2 x i64>, ptr addrspace(1) %39, align 16, !invariant.load !4
  %41 = extractelement <2 x i64> %40, i32 0
  %42 = extractelement <2 x i64> %40, i32 1
  %43 = tail call i64 @llvm.smax.i64(i64 %41, i64 0)
  %44 = tail call i64 @llvm.umin.i64(i64 %43, i64 511)
  %45 = tail call i64 @llvm.smax.i64(i64 %42, i64 0)
  %46 = tail call i64 @llvm.umin.i64(i64 %45, i64 619)
  %47 = urem i32 %.lhs.trunc33, 155
  %.zext36 = zext nneg i32 %47 to i64
  %.idx6 = mul nuw nsw i64 %44, 1537600
  %48 = getelementptr inbounds i8, ptr addrspace(1) %5, i64 %.idx6
  %.idx7 = mul nuw nsw i64 %46, 2480
  %49 = getelementptr inbounds i8, ptr addrspace(1) %48, i64 %.idx7
  %50 = getelementptr inbounds { double, double }, ptr addrspace(1) %49, i64 %.zext36
  %51 = load <2 x double>, ptr addrspace(1) %50, align 16, !invariant.load !4
  %.unpack855 = extractelement <2 x double> %51, i32 0
  %.unpack1056 = extractelement <2 x double> %51, i32 1
  %52 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 16
  %53 = insertelement <2 x double> poison, double %.unpack855, i32 0
  %54 = insertelement <2 x double> %53, double %.unpack1056, i32 1
  store <2 x double> %54, ptr addrspace(1) %52, align 16
  %55 = or disjoint i64 %13, 2
  %.lhs.trunc37 = trunc nuw nsw i64 %55 to i32
  %56 = udiv i32 %.lhs.trunc37, 155
  %57 = and i32 %56, 511
  %narrow46 = mul nuw nsw i32 %57, 9920
  %.idx29 = zext nneg i32 %narrow46 to i64
  %58 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %.idx29
  %59 = getelementptr inbounds i64, ptr addrspace(1) %58, i64 %17
  %60 = load <2 x i64>, ptr addrspace(1) %59, align 16, !invariant.load !4
  %61 = extractelement <2 x i64> %60, i32 0
  %62 = extractelement <2 x i64> %60, i32 1
  %63 = tail call i64 @llvm.smax.i64(i64 %61, i64 0)
  %64 = tail call i64 @llvm.umin.i64(i64 %63, i64 511)
  %65 = tail call i64 @llvm.smax.i64(i64 %62, i64 0)
  %66 = tail call i64 @llvm.umin.i64(i64 %65, i64 619)
  %67 = urem i32 %.lhs.trunc37, 155
  %.zext40 = zext nneg i32 %67 to i64
  %.idx13 = mul nuw nsw i64 %64, 1537600
  %68 = getelementptr inbounds i8, ptr addrspace(1) %5, i64 %.idx13
  %.idx14 = mul nuw nsw i64 %66, 2480
  %69 = getelementptr inbounds i8, ptr addrspace(1) %68, i64 %.idx14
  %70 = getelementptr inbounds { double, double }, ptr addrspace(1) %69, i64 %.zext40
  %71 = load <2 x double>, ptr addrspace(1) %70, align 16, !invariant.load !4
  %.unpack1553 = extractelement <2 x double> %71, i32 0
  %.unpack1754 = extractelement <2 x double> %71, i32 1
  %72 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 32
  %73 = insertelement <2 x double> poison, double %.unpack1553, i32 0
  %74 = insertelement <2 x double> %73, double %.unpack1754, i32 1
  store <2 x double> %74, ptr addrspace(1) %72, align 32
  %75 = or disjoint i64 %13, 3
  %.lhs.trunc41 = trunc nuw nsw i64 %75 to i32
  %76 = udiv i32 %.lhs.trunc41, 155
  %77 = and i32 %76, 511
  %narrow47 = mul nuw nsw i32 %77, 9920
  %.idx30 = zext nneg i32 %narrow47 to i64
  %78 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %.idx30
  %79 = getelementptr inbounds i64, ptr addrspace(1) %78, i64 %17
  %80 = load <2 x i64>, ptr addrspace(1) %79, align 16, !invariant.load !4
  %81 = extractelement <2 x i64> %80, i32 0
  %82 = extractelement <2 x i64> %80, i32 1
  %83 = tail call i64 @llvm.smax.i64(i64 %81, i64 0)
  %84 = tail call i64 @llvm.umin.i64(i64 %83, i64 511)
  %85 = tail call i64 @llvm.smax.i64(i64 %82, i64 0)
  %86 = tail call i64 @llvm.umin.i64(i64 %85, i64 619)
  %87 = urem i32 %.lhs.trunc41, 155
  %.zext44 = zext nneg i32 %87 to i64
  %.idx20 = mul nuw nsw i64 %84, 1537600
  %88 = getelementptr inbounds i8, ptr addrspace(1) %5, i64 %.idx20
  %.idx21 = mul nuw nsw i64 %86, 2480
  %89 = getelementptr inbounds i8, ptr addrspace(1) %88, i64 %.idx21
  %90 = getelementptr inbounds { double, double }, ptr addrspace(1) %89, i64 %.zext44
  %91 = load <2 x double>, ptr addrspace(1) %90, align 16, !invariant.load !4
  %.unpack2251 = extractelement <2 x double> %91, i32 0
  %.unpack2452 = extractelement <2 x double> %91, i32 1
  %92 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 48
  %93 = insertelement <2 x double> poison, double %.unpack2251, i32 0
  %94 = insertelement <2 x double> %93, double %.unpack2452, i32 1
  store <2 x double> %94, ptr addrspace(1) %92, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_transpose_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(787251200) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(787251200) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = udiv i32 %9, 155
  %11 = trunc i32 %10 to i1
  %12 = select i1 %11, i32 24601600, i32 0
  %13 = shl nuw nsw i32 %6, 1
  %14 = shl nuw nsw i32 %5, 8
  %15 = or disjoint i32 %13, %14
  %16 = urem i32 %15, 155
  %17 = sub nuw nsw i32 %15, %16
  %18 = mul i32 %10, 155
  %.decomposed = sub i32 %9, %18
  %19 = add nuw nsw i32 %17, %.decomposed
  %20 = add nuw nsw i32 %19, %12
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %21
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !4
  %.unpack32 = extractelement <2 x double> %23, i32 0
  %.unpack233 = extractelement <2 x double> %23, i32 1
  %24 = zext nneg i32 %9 to i64
  %25 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %24
  %26 = insertelement <2 x double> poison, double %.unpack32, i32 0
  %27 = insertelement <2 x double> %26, double %.unpack233, i32 1
  store <2 x double> %27, ptr addrspace(1) %25, align 64
  %28 = or disjoint i32 %9, 1
  %29 = udiv i32 %28, 155
  %30 = trunc i32 %29 to i1
  %31 = select i1 %30, i32 24601600, i32 0
  %32 = mul i32 %29, 155
  %.decomposed20 = sub i32 %28, %32
  %33 = add nuw nsw i32 %.decomposed20, %17
  %34 = add nuw nsw i32 %33, %31
  %35 = zext nneg i32 %34 to i64
  %36 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %35
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !4
  %.unpack530 = extractelement <2 x double> %37, i32 0
  %.unpack731 = extractelement <2 x double> %37, i32 1
  %38 = getelementptr inbounds i8, ptr addrspace(1) %25, i64 16
  %39 = insertelement <2 x double> poison, double %.unpack530, i32 0
  %40 = insertelement <2 x double> %39, double %.unpack731, i32 1
  store <2 x double> %40, ptr addrspace(1) %38, align 16
  %41 = or disjoint i32 %9, 2
  %42 = udiv i32 %41, 155
  %43 = trunc i32 %42 to i1
  %44 = select i1 %43, i32 24601600, i32 0
  %45 = or disjoint i32 %15, 1
  %46 = urem i32 %45, 155
  %47 = sub nuw nsw i32 %45, %46
  %48 = mul i32 %42, 155
  %.decomposed21 = sub i32 %41, %48
  %49 = add nuw nsw i32 %47, %.decomposed21
  %50 = add nuw nsw i32 %49, %44
  %51 = zext nneg i32 %50 to i64
  %52 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %51
  %53 = load <2 x double>, ptr addrspace(1) %52, align 16, !invariant.load !4
  %.unpack1028 = extractelement <2 x double> %53, i32 0
  %.unpack1229 = extractelement <2 x double> %53, i32 1
  %54 = getelementptr inbounds i8, ptr addrspace(1) %25, i64 32
  %55 = insertelement <2 x double> poison, double %.unpack1028, i32 0
  %56 = insertelement <2 x double> %55, double %.unpack1229, i32 1
  store <2 x double> %56, ptr addrspace(1) %54, align 32
  %57 = or disjoint i32 %9, 3
  %58 = udiv i32 %57, 155
  %59 = trunc i32 %58 to i1
  %60 = select i1 %59, i32 24601600, i32 0
  %61 = udiv i32 %57, 310
  %62 = mul nuw nsw i32 %61, 155
  %63 = mul i32 %58, 155
  %.decomposed22 = sub i32 %57, %63
  %64 = add nuw nsw i32 %62, %.decomposed22
  %65 = add nuw nsw i32 %64, %60
  %66 = zext nneg i32 %65 to i64
  %67 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %66
  %68 = load <2 x double>, ptr addrspace(1) %67, align 16, !invariant.load !4
  %.unpack1526 = extractelement <2 x double> %68, i32 0
  %.unpack1727 = extractelement <2 x double> %68, i32 1
  %69 = getelementptr inbounds i8, ptr addrspace(1) %25, i64 48
  %70 = insertelement <2 x double> poison, double %.unpack1526, i32 0
  %71 = insertelement <2 x double> %70, double %.unpack1727, i32 1
  store <2 x double> %71, ptr addrspace(1) %69, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_transpose_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(5079040) %0, ptr noalias readonly align 256 captures(none) dereferenceable(787251200) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(787251200) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = udiv i32 %7, 155
  %10 = shl nuw nsw i32 %8, 2
  %11 = shl nuw nsw i32 %7, 9
  %12 = or disjoint i32 %11, %10
  %13 = zext nneg i32 %12 to i64
  %14 = udiv i32 %12, 155
  %15 = and i32 %14, 511
  %16 = shl nuw nsw i32 %9, 1
  %17 = zext nneg i32 %16 to i64
  %narrow = mul nuw nsw i32 %15, 9920
  %.idx44 = zext nneg i32 %narrow to i64
  %18 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %.idx44
  %19 = getelementptr inbounds i64, ptr addrspace(1) %18, i64 %17
  %20 = load <2 x i64>, ptr addrspace(1) %19, align 16, !invariant.load !4
  %21 = extractelement <2 x i64> %20, i32 0
  %22 = extractelement <2 x i64> %20, i32 1
  %23 = tail call i64 @llvm.smax.i64(i64 %21, i64 0)
  %24 = tail call i64 @llvm.umin.i64(i64 %23, i64 511)
  %.fr = freeze i64 %22
  %25 = tail call i64 @llvm.smax.i64(i64 %.fr, i64 0)
  %26 = tail call i64 @llvm.umin.i64(i64 %25, i64 619)
  %27 = mul i32 %14, 155
  %.decomposed = sub i32 %12, %27
  %.cmp = icmp sgt i64 %.fr, 309
  %.urem = add nsw i64 %26, -310
  %.cmp31 = icmp slt i64 %.fr, 310
  %28 = select i1 %.cmp31, i64 %26, i64 %.urem
  %narrow65 = mul nuw nsw i32 %.decomposed, 2539520
  %.idx = zext nneg i32 %narrow65 to i64
  %29 = getelementptr inbounds i8, ptr addrspace(1) %5, i64 %.idx
  %.idx1 = select i1 %.cmp, i64 393625600, i64 0
  %30 = getelementptr inbounds i8, ptr addrspace(1) %29, i64 %.idx1
  %.idx2 = mul nuw nsw i64 %24, 4960
  %31 = getelementptr inbounds i8, ptr addrspace(1) %30, i64 %.idx2
  %32 = getelementptr inbounds { double, double }, ptr addrspace(1) %31, i64 %28
  %33 = load <2 x double>, ptr addrspace(1) %32, align 16, !invariant.load !4
  %.unpack81 = extractelement <2 x double> %33, i32 0
  %.unpack482 = extractelement <2 x double> %33, i32 1
  %34 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %13
  %35 = insertelement <2 x double> poison, double %.unpack81, i32 0
  %36 = insertelement <2 x double> %35, double %.unpack482, i32 1
  store <2 x double> %36, ptr addrspace(1) %34, align 64
  %37 = or disjoint i64 %13, 1
  %.lhs.trunc53 = trunc nuw nsw i64 %37 to i32
  %38 = udiv i32 %.lhs.trunc53, 155
  %39 = and i32 %38, 511
  %narrow66 = mul nuw nsw i32 %39, 9920
  %.idx45 = zext nneg i32 %narrow66 to i64
  %40 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %.idx45
  %41 = getelementptr inbounds i64, ptr addrspace(1) %40, i64 %17
  %42 = load <2 x i64>, ptr addrspace(1) %41, align 16, !invariant.load !4
  %43 = extractelement <2 x i64> %42, i32 0
  %44 = extractelement <2 x i64> %42, i32 1
  %45 = tail call i64 @llvm.smax.i64(i64 %43, i64 0)
  %46 = tail call i64 @llvm.umin.i64(i64 %45, i64 511)
  %.fr46 = freeze i64 %44
  %47 = tail call i64 @llvm.smax.i64(i64 %.fr46, i64 0)
  %48 = tail call i64 @llvm.umin.i64(i64 %47, i64 619)
  %49 = urem i32 %.lhs.trunc53, 155
  %.cmp32 = icmp sgt i64 %.fr46, 309
  %.urem34 = add nsw i64 %48, -310
  %.cmp35 = icmp slt i64 %.fr46, 310
  %50 = select i1 %.cmp35, i64 %48, i64 %.urem34
  %narrow67 = mul nuw nsw i32 %49, 2539520
  %.idx7 = zext nneg i32 %narrow67 to i64
  %51 = getelementptr inbounds i8, ptr addrspace(1) %5, i64 %.idx7
  %.idx8 = select i1 %.cmp32, i64 393625600, i64 0
  %52 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 %.idx8
  %.idx9 = mul nuw nsw i64 %46, 4960
  %53 = getelementptr inbounds i8, ptr addrspace(1) %52, i64 %.idx9
  %54 = getelementptr inbounds { double, double }, ptr addrspace(1) %53, i64 %50
  %55 = load <2 x double>, ptr addrspace(1) %54, align 16, !invariant.load !4
  %.unpack1079 = extractelement <2 x double> %55, i32 0
  %.unpack1280 = extractelement <2 x double> %55, i32 1
  %56 = getelementptr inbounds i8, ptr addrspace(1) %34, i64 16
  %57 = insertelement <2 x double> poison, double %.unpack1079, i32 0
  %58 = insertelement <2 x double> %57, double %.unpack1280, i32 1
  store <2 x double> %58, ptr addrspace(1) %56, align 16
  %59 = or disjoint i64 %13, 2
  %.lhs.trunc57 = trunc nuw nsw i64 %59 to i32
  %60 = udiv i32 %.lhs.trunc57, 155
  %61 = and i32 %60, 511
  %narrow68 = mul nuw nsw i32 %61, 9920
  %.idx47 = zext nneg i32 %narrow68 to i64
  %62 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %.idx47
  %63 = getelementptr inbounds i64, ptr addrspace(1) %62, i64 %17
  %64 = load <2 x i64>, ptr addrspace(1) %63, align 16, !invariant.load !4
  %65 = extractelement <2 x i64> %64, i32 0
  %66 = extractelement <2 x i64> %64, i32 1
  %67 = tail call i64 @llvm.smax.i64(i64 %65, i64 0)
  %68 = tail call i64 @llvm.umin.i64(i64 %67, i64 511)
  %.fr48 = freeze i64 %66
  %69 = tail call i64 @llvm.smax.i64(i64 %.fr48, i64 0)
  %70 = tail call i64 @llvm.umin.i64(i64 %69, i64 619)
  %71 = urem i32 %.lhs.trunc57, 155
  %.cmp36 = icmp sgt i64 %.fr48, 309
  %.urem38 = add nsw i64 %70, -310
  %.cmp39 = icmp slt i64 %.fr48, 310
  %72 = select i1 %.cmp39, i64 %70, i64 %.urem38
  %narrow69 = mul nuw nsw i32 %71, 2539520
  %.idx15 = zext nneg i32 %narrow69 to i64
  %73 = getelementptr inbounds i8, ptr addrspace(1) %5, i64 %.idx15
  %.idx16 = select i1 %.cmp36, i64 393625600, i64 0
  %74 = getelementptr inbounds i8, ptr addrspace(1) %73, i64 %.idx16
  %.idx17 = mul nuw nsw i64 %68, 4960
  %75 = getelementptr inbounds i8, ptr addrspace(1) %74, i64 %.idx17
  %76 = getelementptr inbounds { double, double }, ptr addrspace(1) %75, i64 %72
  %77 = load <2 x double>, ptr addrspace(1) %76, align 16, !invariant.load !4
  %.unpack1877 = extractelement <2 x double> %77, i32 0
  %.unpack2078 = extractelement <2 x double> %77, i32 1
  %78 = getelementptr inbounds i8, ptr addrspace(1) %34, i64 32
  %79 = insertelement <2 x double> poison, double %.unpack1877, i32 0
  %80 = insertelement <2 x double> %79, double %.unpack2078, i32 1
  store <2 x double> %80, ptr addrspace(1) %78, align 32
  %81 = or disjoint i64 %13, 3
  %.lhs.trunc61 = trunc nuw nsw i64 %81 to i32
  %82 = udiv i32 %.lhs.trunc61, 155
  %83 = and i32 %82, 511
  %narrow70 = mul nuw nsw i32 %83, 9920
  %.idx49 = zext nneg i32 %narrow70 to i64
  %84 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %.idx49
  %85 = getelementptr inbounds i64, ptr addrspace(1) %84, i64 %17
  %86 = load <2 x i64>, ptr addrspace(1) %85, align 16, !invariant.load !4
  %87 = extractelement <2 x i64> %86, i32 0
  %88 = extractelement <2 x i64> %86, i32 1
  %89 = tail call i64 @llvm.smax.i64(i64 %87, i64 0)
  %90 = tail call i64 @llvm.umin.i64(i64 %89, i64 511)
  %.fr50 = freeze i64 %88
  %91 = tail call i64 @llvm.smax.i64(i64 %.fr50, i64 0)
  %92 = tail call i64 @llvm.umin.i64(i64 %91, i64 619)
  %93 = urem i32 %.lhs.trunc61, 155
  %.cmp40 = icmp sgt i64 %.fr50, 309
  %.urem42 = add nsw i64 %92, -310
  %.cmp43 = icmp slt i64 %.fr50, 310
  %94 = select i1 %.cmp43, i64 %92, i64 %.urem42
  %narrow71 = mul nuw nsw i32 %93, 2539520
  %.idx23 = zext nneg i32 %narrow71 to i64
  %95 = getelementptr inbounds i8, ptr addrspace(1) %5, i64 %.idx23
  %.idx24 = select i1 %.cmp40, i64 393625600, i64 0
  %96 = getelementptr inbounds i8, ptr addrspace(1) %95, i64 %.idx24
  %.idx25 = mul nuw nsw i64 %90, 4960
  %97 = getelementptr inbounds i8, ptr addrspace(1) %96, i64 %.idx25
  %98 = getelementptr inbounds { double, double }, ptr addrspace(1) %97, i64 %94
  %99 = load <2 x double>, ptr addrspace(1) %98, align 16, !invariant.load !4
  %.unpack2675 = extractelement <2 x double> %99, i32 0
  %.unpack2876 = extractelement <2 x double> %99, i32 1
  %100 = getelementptr inbounds i8, ptr addrspace(1) %34, i64 48
  %101 = insertelement <2 x double> poison, double %.unpack2675, i32 0
  %102 = insertelement <2 x double> %101, double %.unpack2876, i32 1
  store <2 x double> %102, ptr addrspace(1) %100, align 16
  ret void
}

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(787251200) %0, ptr noalias readonly align 256 captures(none) dereferenceable(1269760) %1, ptr noalias readonly align 256 captures(none) dereferenceable(1269760) %2, ptr noalias readonly align 256 captures(none) dereferenceable(5079040) %3, ptr noalias readonly align 256 captures(none) dereferenceable(4) %4, ptr noalias writeonly align 256 captures(none) dereferenceable(787251200) %5) local_unnamed_addr #3 {
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = addrspacecast ptr %4 to ptr addrspace(1)
  %9 = addrspacecast ptr %3 to ptr addrspace(1)
  %10 = addrspacecast ptr %1 to ptr addrspace(1)
  %11 = addrspacecast ptr %2 to ptr addrspace(1)
  %12 = addrspacecast ptr %5 to ptr addrspace(1)
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %14 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %15 = udiv i32 %14, 5
  %16 = mul i32 %15, 5
  %.decomposed = sub i32 %14, %16
  %17 = shl nuw nsw i32 %.decomposed, 5
  %18 = and i32 %13, 31
  %19 = or disjoint i32 %17, %18
  %20 = icmp samesign ult i32 %19, 155
  br i1 %20, label %21, label %._crit_edge

._crit_edge:                                      ; preds = %6
  %.pre = udiv i32 %14, 2560
  %.pre106 = urem i32 %.pre, 10
  %.pre108 = lshr i32 %13, 5
  br label %57

21:                                               ; preds = %6
  %22 = and i32 %15, 511
  %23 = mul nuw nsw i32 %22, 155
  %24 = udiv i32 %14, 2560
  %25 = urem i32 %24, 10
  %26 = mul nuw nsw i32 %25, 2539520
  %27 = udiv i32 %14, 25600
  %28 = mul nuw nsw i32 %27, 24601600
  %29 = lshr i32 %13, 5
  %30 = mul nuw nsw i32 %29, 79360
  %31 = or disjoint i32 %18, %28
  %32 = or disjoint i32 %31, %17
  %33 = add nuw nsw i32 %32, %30
  %34 = add nuw nsw i32 %33, %26
  %35 = add nuw nsw i32 %34, %23
  %36 = zext nneg i32 %35 to i64
  %37 = getelementptr inbounds { double, double }, ptr addrspace(1) %7, i64 %36
  %38 = load <2 x double>, ptr addrspace(1) %37, align 16, !invariant.load !4
  %.unpack164 = extractelement <2 x double> %38, i32 0
  %.unpack2165 = extractelement <2 x double> %38, i32 1
  %39 = mul nuw nsw i32 %18, 33
  %40 = add nuw nsw i32 %39, %29
  %41 = zext nneg i32 %40 to i64
  %42 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_01, i64 %41
  store double %.unpack164, ptr addrspace(3) %42, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %42, i64 8
  store double %.unpack2165, ptr addrspace(3) %.repack3, align 8
  %43 = sext i32 %35 to i64
  %44 = getelementptr { double, double }, ptr addrspace(1) %7, i64 %43
  %45 = getelementptr i8, ptr addrspace(1) %44, i64 5079040
  %46 = load <2 x double>, ptr addrspace(1) %45, align 16, !invariant.load !4
  %.unpack5156 = extractelement <2 x double> %46, i32 0
  %.unpack7157 = extractelement <2 x double> %46, i32 1
  %47 = getelementptr inbounds i8, ptr addrspace(3) %42, i64 64
  store double %.unpack5156, ptr addrspace(3) %47, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %42, i64 72
  store double %.unpack7157, ptr addrspace(3) %.repack8, align 8
  %48 = getelementptr i8, ptr addrspace(1) %44, i64 10158080
  %49 = load <2 x double>, ptr addrspace(1) %48, align 16, !invariant.load !4
  %.unpack10158 = extractelement <2 x double> %49, i32 0
  %.unpack12159 = extractelement <2 x double> %49, i32 1
  %50 = getelementptr inbounds i8, ptr addrspace(3) %42, i64 128
  store double %.unpack10158, ptr addrspace(3) %50, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %42, i64 136
  store double %.unpack12159, ptr addrspace(3) %.repack13, align 8
  %51 = getelementptr i8, ptr addrspace(1) %44, i64 15237120
  %52 = load <2 x double>, ptr addrspace(1) %51, align 16, !invariant.load !4
  %.unpack15160 = extractelement <2 x double> %52, i32 0
  %.unpack17161 = extractelement <2 x double> %52, i32 1
  %53 = getelementptr inbounds i8, ptr addrspace(3) %42, i64 192
  store double %.unpack15160, ptr addrspace(3) %53, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %42, i64 200
  store double %.unpack17161, ptr addrspace(3) %.repack18, align 8
  %54 = getelementptr i8, ptr addrspace(1) %44, i64 20316160
  %55 = load <2 x double>, ptr addrspace(1) %54, align 16, !invariant.load !4
  %.unpack20162 = extractelement <2 x double> %55, i32 0
  %.unpack22163 = extractelement <2 x double> %55, i32 1
  %56 = getelementptr inbounds i8, ptr addrspace(3) %42, i64 256
  store double %.unpack20162, ptr addrspace(3) %56, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %42, i64 264
  store double %.unpack22163, ptr addrspace(3) %.repack23, align 8
  br label %57

57:                                               ; preds = %._crit_edge, %21
  %.pre-phi109 = phi i32 [ %.pre108, %._crit_edge ], [ %29, %21 ]
  %.pre-phi107 = phi i32 [ %.pre106, %._crit_edge ], [ %25, %21 ]
  %58 = icmp ult i32 %19, 155
  %59 = shl nuw nsw i32 %.pre-phi107, 5
  %60 = or disjoint i32 %59, %.pre-phi109
  %61 = icmp samesign ult i32 %60, 290
  %62 = and i1 %58, %61
  br i1 %62, label %63, label %87

63:                                               ; preds = %57
  %64 = and i32 %15, 511
  %65 = mul nuw nsw i32 %64, 155
  %66 = mul nuw nsw i32 %.pre-phi107, 2539520
  %67 = udiv i32 %14, 25600
  %68 = mul nuw nsw i32 %67, 24601600
  %69 = mul nuw nsw i32 %.pre-phi109, 79360
  %70 = zext nneg i32 %65 to i64
  %71 = zext nneg i32 %66 to i64
  %72 = zext nneg i32 %69 to i64
  %73 = zext nneg i32 %68 to i64
  %74 = zext nneg i32 %19 to i64
  %75 = add i64 %74, %73
  %76 = add i64 %75, %72
  %77 = add i64 %76, %71
  %78 = add i64 %77, %70
  %79 = getelementptr inbounds { double, double }, ptr addrspace(1) %7, i64 %78
  %80 = getelementptr inbounds i8, ptr addrspace(1) %79, i64 25395200
  %81 = load <2 x double>, ptr addrspace(1) %80, align 16, !invariant.load !4
  %.unpack25154 = extractelement <2 x double> %81, i32 0
  %.unpack27155 = extractelement <2 x double> %81, i32 1
  %82 = mul nuw nsw i32 %18, 33
  %83 = add nuw nsw i32 %82, %.pre-phi109
  %84 = zext nneg i32 %83 to i64
  %85 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_01, i64 %84
  %86 = getelementptr inbounds i8, ptr addrspace(3) %85, i64 320
  store double %.unpack25154, ptr addrspace(3) %86, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %85, i64 328
  store double %.unpack27155, ptr addrspace(3) %.repack28, align 8
  br label %87

87:                                               ; preds = %63, %57
  %88 = icmp ult i32 %19, 155
  %89 = icmp samesign ult i32 %60, 286
  %90 = and i1 %88, %89
  br i1 %90, label %91, label %115

91:                                               ; preds = %87
  %92 = and i32 %15, 511
  %93 = mul nuw nsw i32 %92, 155
  %94 = mul nuw nsw i32 %.pre-phi107, 2539520
  %95 = udiv i32 %14, 25600
  %96 = mul nuw nsw i32 %95, 24601600
  %97 = mul nuw nsw i32 %.pre-phi109, 79360
  %98 = zext nneg i32 %93 to i64
  %99 = zext nneg i32 %94 to i64
  %100 = zext nneg i32 %97 to i64
  %101 = zext nneg i32 %96 to i64
  %102 = zext nneg i32 %19 to i64
  %103 = add i64 %102, %101
  %104 = add i64 %103, %100
  %105 = add i64 %104, %99
  %106 = add i64 %105, %98
  %107 = getelementptr inbounds { double, double }, ptr addrspace(1) %7, i64 %106
  %108 = getelementptr inbounds i8, ptr addrspace(1) %107, i64 30474240
  %109 = load <2 x double>, ptr addrspace(1) %108, align 16, !invariant.load !4
  %.unpack30152 = extractelement <2 x double> %109, i32 0
  %.unpack32153 = extractelement <2 x double> %109, i32 1
  %110 = mul nuw nsw i32 %18, 33
  %111 = add nuw nsw i32 %110, %.pre-phi109
  %112 = zext nneg i32 %111 to i64
  %113 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_01, i64 %112
  %114 = getelementptr inbounds i8, ptr addrspace(3) %113, i64 384
  store double %.unpack30152, ptr addrspace(3) %114, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %113, i64 392
  store double %.unpack32153, ptr addrspace(3) %.repack33, align 8
  br label %115

115:                                              ; preds = %91, %87
  %116 = icmp ult i32 %19, 155
  %117 = icmp samesign ult i32 %60, 282
  %118 = and i1 %116, %117
  br i1 %118, label %119, label %143

119:                                              ; preds = %115
  %120 = and i32 %15, 511
  %121 = mul nuw nsw i32 %120, 155
  %122 = mul nuw nsw i32 %.pre-phi107, 2539520
  %123 = udiv i32 %14, 25600
  %124 = mul nuw nsw i32 %123, 24601600
  %125 = mul nuw nsw i32 %.pre-phi109, 79360
  %126 = zext nneg i32 %121 to i64
  %127 = zext nneg i32 %122 to i64
  %128 = zext nneg i32 %125 to i64
  %129 = zext nneg i32 %124 to i64
  %130 = zext nneg i32 %19 to i64
  %131 = add i64 %130, %129
  %132 = add i64 %131, %128
  %133 = add i64 %132, %127
  %134 = add i64 %133, %126
  %135 = getelementptr inbounds { double, double }, ptr addrspace(1) %7, i64 %134
  %136 = getelementptr inbounds i8, ptr addrspace(1) %135, i64 35553280
  %137 = load <2 x double>, ptr addrspace(1) %136, align 16, !invariant.load !4
  %.unpack35150 = extractelement <2 x double> %137, i32 0
  %.unpack37151 = extractelement <2 x double> %137, i32 1
  %138 = mul nuw nsw i32 %18, 33
  %139 = add nuw nsw i32 %138, %.pre-phi109
  %140 = zext nneg i32 %139 to i64
  %141 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_01, i64 %140
  %142 = getelementptr inbounds i8, ptr addrspace(3) %141, i64 448
  store double %.unpack35150, ptr addrspace(3) %142, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %141, i64 456
  store double %.unpack37151, ptr addrspace(3) %.repack38, align 8
  br label %143

143:                                              ; preds = %119, %115
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %144 = or disjoint i32 %59, %18
  %145 = icmp samesign ult i32 %144, 310
  br i1 %145, label %146, label %293

146:                                              ; preds = %143
  %147 = mul nuw nsw i32 %.pre-phi109, 33
  %148 = add nuw nsw i32 %147, %18
  %149 = zext nneg i32 %148 to i64
  %150 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_01, i64 %149
  %.unpack40 = load double, ptr addrspace(3) %150, align 8
  %.elt41 = getelementptr inbounds i8, ptr addrspace(3) %150, i64 8
  %.unpack42 = load double, ptr addrspace(3) %.elt41, align 8
  %151 = udiv i32 %14, 25600
  %152 = mul nuw nsw i32 %151, 155
  %153 = add nuw nsw i32 %152, %.pre-phi109
  %154 = add nuw nsw i32 %153, %17
  %155 = load i32, ptr addrspace(1) %8, align 256, !invariant.load !4
  %156 = and i32 %155, 2
  %.not = icmp eq i32 %156, 0
  %157 = select i1 %.not, i32 0, i32 310
  %158 = and i32 %15, 511
  %159 = mul nuw nsw i32 %158, 620
  %160 = add nuw nsw i32 %154, %159
  %161 = add nuw nsw i32 %160, %157
  %162 = zext nneg i32 %161 to i64
  %163 = getelementptr inbounds { double, double }, ptr addrspace(1) %9, i64 %162
  %164 = load <2 x double>, ptr addrspace(1) %163, align 16, !invariant.load !4
  %.unpack43148 = extractelement <2 x double> %164, i32 0
  %.unpack45149 = extractelement <2 x double> %164, i32 1
  %165 = fmul double %.unpack40, %.unpack43148
  %166 = fmul double %.unpack42, %.unpack45149
  %167 = fsub double %165, %166
  %168 = fmul double %.unpack40, %.unpack45149
  %169 = fmul double %.unpack42, %.unpack43148
  %170 = fadd double %169, %168
  %171 = mul nuw nsw i32 %158, 310
  %172 = add nuw nsw i32 %144, %171
  %173 = zext nneg i32 %172 to i64
  %174 = getelementptr inbounds double, ptr addrspace(1) %10, i64 %173
  %175 = load double, ptr addrspace(1) %174, align 8, !invariant.load !4
  %176 = getelementptr inbounds double, ptr addrspace(1) %11, i64 %173
  %177 = load double, ptr addrspace(1) %176, align 8, !invariant.load !4
  %178 = fmul double %175, %167
  %179 = fmul double %170, %177
  %180 = fsub double %178, %179
  %181 = fmul double %175, %170
  %182 = fmul double %167, %177
  %183 = fadd double %181, %182
  %184 = mul nuw nsw i32 %158, 96100
  %185 = mul nuw nsw i32 %.decomposed, 9920
  %186 = mul nuw nsw i32 %151, 48050
  %187 = mul nuw nsw i32 %.pre-phi109, 310
  %188 = or disjoint i32 %185, %18
  %189 = add nuw nsw i32 %188, %186
  %190 = add nuw nsw i32 %189, %187
  %191 = add nuw nsw i32 %190, %59
  %192 = add nuw nsw i32 %191, %184
  %193 = zext nneg i32 %192 to i64
  %194 = getelementptr inbounds { double, double }, ptr addrspace(1) %12, i64 %193
  %195 = insertelement <2 x double> poison, double %180, i32 0
  %196 = insertelement <2 x double> %195, double %183, i32 1
  store <2 x double> %196, ptr addrspace(1) %194, align 16
  %197 = getelementptr inbounds i8, ptr addrspace(3) %150, i64 2112
  %.unpack48 = load double, ptr addrspace(3) %197, align 8
  %.elt49 = getelementptr inbounds i8, ptr addrspace(3) %150, i64 2120
  %.unpack50 = load double, ptr addrspace(3) %.elt49, align 8
  %198 = zext nneg i32 %157 to i64
  %199 = zext nneg i32 %160 to i64
  %200 = add i64 %199, %198
  %201 = getelementptr inbounds { double, double }, ptr addrspace(1) %9, i64 %200
  %202 = getelementptr inbounds i8, ptr addrspace(1) %201, i64 64
  %203 = load <2 x double>, ptr addrspace(1) %202, align 16, !invariant.load !4
  %.unpack51138 = extractelement <2 x double> %203, i32 0
  %.unpack53139 = extractelement <2 x double> %203, i32 1
  %204 = fmul double %.unpack48, %.unpack51138
  %205 = fmul double %.unpack50, %.unpack53139
  %206 = fsub double %204, %205
  %207 = fmul double %.unpack48, %.unpack53139
  %208 = fmul double %.unpack50, %.unpack51138
  %209 = fadd double %208, %207
  %210 = fmul double %175, %206
  %211 = fmul double %177, %209
  %212 = fsub double %210, %211
  %213 = fmul double %175, %209
  %214 = fmul double %177, %206
  %215 = fadd double %213, %214
  %216 = sext i32 %192 to i64
  %217 = getelementptr { double, double }, ptr addrspace(1) %12, i64 %216
  %218 = getelementptr i8, ptr addrspace(1) %217, i64 19840
  %219 = insertelement <2 x double> poison, double %212, i32 0
  %220 = insertelement <2 x double> %219, double %215, i32 1
  store <2 x double> %220, ptr addrspace(1) %218, align 16
  %221 = getelementptr inbounds i8, ptr addrspace(3) %150, i64 4224
  %.unpack56 = load double, ptr addrspace(3) %221, align 8
  %.elt57 = getelementptr inbounds i8, ptr addrspace(3) %150, i64 4232
  %.unpack58 = load double, ptr addrspace(3) %.elt57, align 8
  %222 = getelementptr inbounds i8, ptr addrspace(1) %201, i64 128
  %223 = load <2 x double>, ptr addrspace(1) %222, align 16, !invariant.load !4
  %.unpack59140 = extractelement <2 x double> %223, i32 0
  %.unpack61141 = extractelement <2 x double> %223, i32 1
  %224 = fmul double %.unpack56, %.unpack59140
  %225 = fmul double %.unpack58, %.unpack61141
  %226 = fsub double %224, %225
  %227 = fmul double %.unpack56, %.unpack61141
  %228 = fmul double %.unpack58, %.unpack59140
  %229 = fadd double %228, %227
  %230 = fmul double %175, %226
  %231 = fmul double %177, %229
  %232 = fsub double %230, %231
  %233 = fmul double %175, %229
  %234 = fmul double %177, %226
  %235 = fadd double %233, %234
  %236 = getelementptr i8, ptr addrspace(1) %217, i64 39680
  %237 = insertelement <2 x double> poison, double %232, i32 0
  %238 = insertelement <2 x double> %237, double %235, i32 1
  store <2 x double> %238, ptr addrspace(1) %236, align 16
  %239 = getelementptr inbounds i8, ptr addrspace(3) %150, i64 6336
  %.unpack64 = load double, ptr addrspace(3) %239, align 8
  %.elt65 = getelementptr inbounds i8, ptr addrspace(3) %150, i64 6344
  %.unpack66 = load double, ptr addrspace(3) %.elt65, align 8
  %240 = getelementptr inbounds i8, ptr addrspace(1) %201, i64 192
  %241 = load <2 x double>, ptr addrspace(1) %240, align 16, !invariant.load !4
  %.unpack67142 = extractelement <2 x double> %241, i32 0
  %.unpack69143 = extractelement <2 x double> %241, i32 1
  %242 = fmul double %.unpack64, %.unpack67142
  %243 = fmul double %.unpack66, %.unpack69143
  %244 = fsub double %242, %243
  %245 = fmul double %.unpack64, %.unpack69143
  %246 = fmul double %.unpack66, %.unpack67142
  %247 = fadd double %246, %245
  %248 = fmul double %175, %244
  %249 = fmul double %177, %247
  %250 = fsub double %248, %249
  %251 = fmul double %175, %247
  %252 = fmul double %177, %244
  %253 = fadd double %251, %252
  %254 = getelementptr i8, ptr addrspace(1) %217, i64 59520
  %255 = insertelement <2 x double> poison, double %250, i32 0
  %256 = insertelement <2 x double> %255, double %253, i32 1
  store <2 x double> %256, ptr addrspace(1) %254, align 16
  %257 = getelementptr inbounds i8, ptr addrspace(3) %150, i64 8448
  %.unpack72 = load double, ptr addrspace(3) %257, align 8
  %.elt73 = getelementptr inbounds i8, ptr addrspace(3) %150, i64 8456
  %.unpack74 = load double, ptr addrspace(3) %.elt73, align 8
  %258 = getelementptr inbounds i8, ptr addrspace(1) %201, i64 256
  %259 = load <2 x double>, ptr addrspace(1) %258, align 16, !invariant.load !4
  %.unpack75144 = extractelement <2 x double> %259, i32 0
  %.unpack77145 = extractelement <2 x double> %259, i32 1
  %260 = fmul double %.unpack72, %.unpack75144
  %261 = fmul double %.unpack74, %.unpack77145
  %262 = fsub double %260, %261
  %263 = fmul double %.unpack72, %.unpack77145
  %264 = fmul double %.unpack74, %.unpack75144
  %265 = fadd double %264, %263
  %266 = fmul double %175, %262
  %267 = fmul double %177, %265
  %268 = fsub double %266, %267
  %269 = fmul double %175, %265
  %270 = fmul double %177, %262
  %271 = fadd double %269, %270
  %272 = getelementptr i8, ptr addrspace(1) %217, i64 79360
  %273 = insertelement <2 x double> poison, double %268, i32 0
  %274 = insertelement <2 x double> %273, double %271, i32 1
  store <2 x double> %274, ptr addrspace(1) %272, align 16
  %275 = getelementptr inbounds i8, ptr addrspace(3) %150, i64 10560
  %.unpack80 = load double, ptr addrspace(3) %275, align 8
  %.elt81 = getelementptr inbounds i8, ptr addrspace(3) %150, i64 10568
  %.unpack82 = load double, ptr addrspace(3) %.elt81, align 8
  %276 = getelementptr inbounds i8, ptr addrspace(1) %201, i64 320
  %277 = load <2 x double>, ptr addrspace(1) %276, align 16, !invariant.load !4
  %.unpack83146 = extractelement <2 x double> %277, i32 0
  %.unpack85147 = extractelement <2 x double> %277, i32 1
  %278 = fmul double %.unpack80, %.unpack83146
  %279 = fmul double %.unpack82, %.unpack85147
  %280 = fsub double %278, %279
  %281 = fmul double %.unpack80, %.unpack85147
  %282 = fmul double %.unpack82, %.unpack83146
  %283 = fadd double %282, %281
  %284 = fmul double %175, %280
  %285 = fmul double %177, %283
  %286 = fsub double %284, %285
  %287 = fmul double %175, %283
  %288 = fmul double %177, %280
  %289 = fadd double %287, %288
  %290 = getelementptr i8, ptr addrspace(1) %217, i64 99200
  %291 = insertelement <2 x double> poison, double %286, i32 0
  %292 = insertelement <2 x double> %291, double %289, i32 1
  store <2 x double> %292, ptr addrspace(1) %290, align 16
  br label %293

293:                                              ; preds = %146, %143
  %294 = icmp ult i32 %144, 310
  %295 = or disjoint i32 %17, %.pre-phi109
  %296 = icmp samesign ult i32 %295, 131
  %297 = and i1 %296, %294
  br i1 %297, label %298, label %361

298:                                              ; preds = %293
  %299 = mul nuw nsw i32 %.pre-phi109, 33
  %300 = add nuw nsw i32 %299, %18
  %301 = zext nneg i32 %300 to i64
  %302 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_01, i64 %301
  %303 = getelementptr inbounds i8, ptr addrspace(3) %302, i64 12672
  %.unpack88 = load double, ptr addrspace(3) %303, align 8
  %.elt89 = getelementptr inbounds i8, ptr addrspace(3) %302, i64 12680
  %.unpack90 = load double, ptr addrspace(3) %.elt89, align 8
  %304 = udiv i32 %14, 25600
  %305 = mul nuw nsw i32 %304, 155
  %306 = load i32, ptr addrspace(1) %8, align 256, !invariant.load !4
  %307 = and i32 %306, 2
  %.not91 = icmp eq i32 %307, 0
  %308 = select i1 %.not91, i32 0, i32 310
  %309 = and i32 %15, 511
  %310 = mul nuw nsw i32 %309, 620
  %311 = zext nneg i32 %308 to i64
  %312 = zext nneg i32 %310 to i64
  %313 = zext nneg i32 %17 to i64
  %314 = zext nneg i32 %305 to i64
  %315 = zext nneg i32 %.pre-phi109 to i64
  %316 = add i64 %315, %314
  %317 = add i64 %316, %313
  %318 = add i64 %317, %312
  %319 = add i64 %318, %311
  %320 = getelementptr inbounds { double, double }, ptr addrspace(1) %9, i64 %319
  %321 = getelementptr inbounds i8, ptr addrspace(1) %320, i64 384
  %322 = load <2 x double>, ptr addrspace(1) %321, align 16, !invariant.load !4
  %.unpack92136 = extractelement <2 x double> %322, i32 0
  %.unpack94137 = extractelement <2 x double> %322, i32 1
  %323 = fmul double %.unpack88, %.unpack92136
  %324 = fmul double %.unpack90, %.unpack94137
  %325 = fsub double %323, %324
  %326 = fmul double %.unpack88, %.unpack94137
  %327 = fmul double %.unpack90, %.unpack92136
  %328 = fadd double %327, %326
  %329 = mul nuw nsw i32 %309, 310
  %330 = add nuw nsw i32 %144, %329
  %331 = zext nneg i32 %330 to i64
  %332 = getelementptr inbounds double, ptr addrspace(1) %10, i64 %331
  %333 = load double, ptr addrspace(1) %332, align 8, !invariant.load !4
  %334 = getelementptr inbounds double, ptr addrspace(1) %11, i64 %331
  %335 = load double, ptr addrspace(1) %334, align 8, !invariant.load !4
  %336 = fmul double %333, %325
  %337 = fmul double %328, %335
  %338 = fsub double %336, %337
  %339 = fmul double %333, %328
  %340 = fmul double %325, %335
  %341 = fadd double %339, %340
  %342 = mul nuw nsw i32 %309, 96100
  %343 = mul nuw nsw i32 %.decomposed, 9920
  %344 = mul nuw nsw i32 %304, 48050
  %345 = mul nuw nsw i32 %.pre-phi109, 310
  %346 = zext nneg i32 %342 to i64
  %347 = zext nneg i32 %59 to i64
  %348 = zext nneg i32 %345 to i64
  %349 = zext nneg i32 %344 to i64
  %350 = zext nneg i32 %343 to i64
  %351 = zext nneg i32 %18 to i64
  %352 = add i64 %351, %350
  %353 = add i64 %352, %349
  %354 = add i64 %353, %348
  %355 = add i64 %354, %347
  %356 = add i64 %355, %346
  %357 = getelementptr inbounds { double, double }, ptr addrspace(1) %12, i64 %356
  %358 = getelementptr inbounds i8, ptr addrspace(1) %357, i64 119040
  %359 = insertelement <2 x double> poison, double %338, i32 0
  %360 = insertelement <2 x double> %359, double %341, i32 1
  store <2 x double> %360, ptr addrspace(1) %358, align 16
  br label %361

361:                                              ; preds = %298, %293
  %362 = icmp ult i32 %144, 310
  %363 = icmp samesign ult i32 %295, 127
  %364 = and i1 %363, %362
  br i1 %364, label %365, label %428

365:                                              ; preds = %361
  %366 = mul nuw nsw i32 %.pre-phi109, 33
  %367 = add nuw nsw i32 %366, %18
  %368 = zext nneg i32 %367 to i64
  %369 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_01, i64 %368
  %370 = getelementptr inbounds i8, ptr addrspace(3) %369, i64 14784
  %.unpack97 = load double, ptr addrspace(3) %370, align 8
  %.elt98 = getelementptr inbounds i8, ptr addrspace(3) %369, i64 14792
  %.unpack99 = load double, ptr addrspace(3) %.elt98, align 8
  %371 = udiv i32 %14, 25600
  %372 = mul nuw nsw i32 %371, 155
  %373 = load i32, ptr addrspace(1) %8, align 256, !invariant.load !4
  %374 = and i32 %373, 2
  %.not100 = icmp eq i32 %374, 0
  %375 = select i1 %.not100, i32 0, i32 310
  %376 = and i32 %15, 511
  %377 = mul nuw nsw i32 %376, 620
  %378 = zext nneg i32 %375 to i64
  %379 = zext nneg i32 %377 to i64
  %380 = zext nneg i32 %17 to i64
  %381 = zext nneg i32 %372 to i64
  %382 = zext nneg i32 %.pre-phi109 to i64
  %383 = add i64 %382, %381
  %384 = add i64 %383, %380
  %385 = add i64 %384, %379
  %386 = add i64 %385, %378
  %387 = getelementptr inbounds { double, double }, ptr addrspace(1) %9, i64 %386
  %388 = getelementptr inbounds i8, ptr addrspace(1) %387, i64 448
  %389 = load <2 x double>, ptr addrspace(1) %388, align 16, !invariant.load !4
  %.unpack101134 = extractelement <2 x double> %389, i32 0
  %.unpack103135 = extractelement <2 x double> %389, i32 1
  %390 = fmul double %.unpack97, %.unpack101134
  %391 = fmul double %.unpack99, %.unpack103135
  %392 = fsub double %390, %391
  %393 = fmul double %.unpack97, %.unpack103135
  %394 = fmul double %.unpack99, %.unpack101134
  %395 = fadd double %394, %393
  %396 = mul nuw nsw i32 %376, 310
  %397 = add nuw nsw i32 %144, %396
  %398 = zext nneg i32 %397 to i64
  %399 = getelementptr inbounds double, ptr addrspace(1) %10, i64 %398
  %400 = load double, ptr addrspace(1) %399, align 8, !invariant.load !4
  %401 = getelementptr inbounds double, ptr addrspace(1) %11, i64 %398
  %402 = load double, ptr addrspace(1) %401, align 8, !invariant.load !4
  %403 = fmul double %400, %392
  %404 = fmul double %395, %402
  %405 = fsub double %403, %404
  %406 = fmul double %400, %395
  %407 = fmul double %392, %402
  %408 = fadd double %406, %407
  %409 = mul nuw nsw i32 %376, 96100
  %410 = mul nuw nsw i32 %.decomposed, 9920
  %411 = mul nuw nsw i32 %371, 48050
  %412 = mul nuw nsw i32 %.pre-phi109, 310
  %413 = zext nneg i32 %409 to i64
  %414 = zext nneg i32 %59 to i64
  %415 = zext nneg i32 %412 to i64
  %416 = zext nneg i32 %411 to i64
  %417 = zext nneg i32 %410 to i64
  %418 = zext nneg i32 %18 to i64
  %419 = add i64 %418, %417
  %420 = add i64 %419, %416
  %421 = add i64 %420, %415
  %422 = add i64 %421, %414
  %423 = add i64 %422, %413
  %424 = getelementptr inbounds { double, double }, ptr addrspace(1) %12, i64 %423
  %425 = getelementptr inbounds i8, ptr addrspace(1) %424, i64 138880
  %426 = insertelement <2 x double> poison, double %405, i32 0
  %427 = insertelement <2 x double> %426, double %408, i32 1
  store <2 x double> %427, ptr addrspace(1) %425, align 16
  br label %428

428:                                              ; preds = %365, %361
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #5

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { norecurse nounwind "nvvm.reqntid"="128,1,1" }
attributes #4 = { convergent nocallback nounwind }
attributes #5 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 1240}
!3 = !{i32 0, i32 128}
!4 = !{}
!5 = !{i32 0, i32 96100}
!6 = !{i32 0, i32 51200}
