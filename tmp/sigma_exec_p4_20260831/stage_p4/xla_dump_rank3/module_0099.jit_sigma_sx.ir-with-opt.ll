; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(243793920) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(121896960) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %7 = and i32 %5, 31
  %8 = icmp samesign ult i32 %7, 24
  br i1 %8, label %9, label %._crit_edge

._crit_edge:                                      ; preds = %2
  %.pre = udiv i32 %6, 20
  br label %31

9:                                                ; preds = %2
  %10 = udiv i32 %6, 20
  %11 = mul nuw nsw i32 %6, 1536
  %12 = mul nsw i32 %10, -960
  %13 = lshr i32 %5, 5
  %14 = mul nuw nsw i32 %13, 48
  %15 = or disjoint i32 %7, %11
  %16 = add nsw i32 %15, %12
  %17 = add nsw i32 %16, %14
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %18
  %20 = load <2 x double>, ptr addrspace(1) %19, align 16, !invariant.load !4
  %.unpack84 = extractelement <2 x double> %20, i32 0
  %.unpack285 = extractelement <2 x double> %20, i32 1
  %21 = mul nuw nsw i32 %7, 33
  %22 = add nuw nsw i32 %21, %13
  %23 = zext nneg i32 %22 to i64
  %24 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %23
  store double %.unpack84, ptr addrspace(3) %24, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %24, i64 8
  store double %.unpack285, ptr addrspace(3) %.repack3, align 8
  %25 = getelementptr i8, ptr addrspace(1) %19, i64 3072
  %26 = load <2 x double>, ptr addrspace(1) %25, align 16, !invariant.load !4
  %.unpack586 = extractelement <2 x double> %26, i32 0
  %.unpack787 = extractelement <2 x double> %26, i32 1
  %27 = getelementptr inbounds i8, ptr addrspace(3) %24, i64 64
  store double %.unpack586, ptr addrspace(3) %27, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %24, i64 72
  store double %.unpack787, ptr addrspace(3) %.repack8, align 8
  %28 = getelementptr i8, ptr addrspace(1) %19, i64 6144
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack1088 = extractelement <2 x double> %29, i32 0
  %.unpack1289 = extractelement <2 x double> %29, i32 1
  %30 = getelementptr inbounds i8, ptr addrspace(3) %24, i64 128
  store double %.unpack1088, ptr addrspace(3) %30, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %24, i64 136
  store double %.unpack1289, ptr addrspace(3) %.repack13, align 8
  br label %31

31:                                               ; preds = %._crit_edge, %9
  %.pre-phi = phi i32 [ %.pre, %._crit_edge ], [ %10, %9 ]
  %32 = icmp ult i32 %7, 24
  %33 = mul i32 %.pre-phi, 20
  %.decomposed = sub i32 %6, %33
  %34 = icmp samesign ult i32 %.decomposed, 19
  %35 = and i1 %32, %34
  br i1 %35, label %.critedge, label %.critedge72

.critedge:                                        ; preds = %31
  %36 = mul nuw nsw i32 %.decomposed, 1536
  %37 = mul nuw nsw i32 %.pre-phi, 29760
  %38 = lshr i32 %5, 5
  %39 = mul nuw nsw i32 %38, 48
  %40 = or disjoint i32 %37, %7
  %41 = add nuw nsw i32 %40, %36
  %42 = add nuw nsw i32 %41, %39
  %43 = zext nneg i32 %42 to i64
  %44 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %43
  %45 = getelementptr inbounds i8, ptr addrspace(1) %44, i64 9216
  %46 = load <2 x double>, ptr addrspace(1) %45, align 16, !invariant.load !4
  %.unpack1574 = extractelement <2 x double> %46, i32 0
  %.unpack1775 = extractelement <2 x double> %46, i32 1
  %47 = mul nuw nsw i32 %7, 33
  %48 = add nuw nsw i32 %47, %38
  %49 = zext nneg i32 %48 to i64
  %50 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %49
  %51 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 192
  store double %.unpack1574, ptr addrspace(3) %51, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 200
  store double %.unpack1775, ptr addrspace(3) %.repack18, align 8
  %52 = getelementptr inbounds i8, ptr addrspace(1) %44, i64 12288
  %53 = load <2 x double>, ptr addrspace(1) %52, align 16, !invariant.load !4
  %.unpack2076 = extractelement <2 x double> %53, i32 0
  %.unpack2277 = extractelement <2 x double> %53, i32 1
  %54 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 256
  store double %.unpack2076, ptr addrspace(3) %54, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 264
  store double %.unpack2277, ptr addrspace(3) %.repack23, align 8
  %55 = getelementptr inbounds i8, ptr addrspace(1) %44, i64 15360
  %56 = load <2 x double>, ptr addrspace(1) %55, align 16, !invariant.load !4
  %.unpack2578 = extractelement <2 x double> %56, i32 0
  %.unpack2779 = extractelement <2 x double> %56, i32 1
  %57 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 320
  store double %.unpack2578, ptr addrspace(3) %57, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 328
  store double %.unpack2779, ptr addrspace(3) %.repack28, align 8
  %58 = getelementptr inbounds i8, ptr addrspace(1) %44, i64 18432
  %59 = load <2 x double>, ptr addrspace(1) %58, align 16, !invariant.load !4
  %.unpack3080 = extractelement <2 x double> %59, i32 0
  %.unpack3281 = extractelement <2 x double> %59, i32 1
  %60 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 384
  store double %.unpack3080, ptr addrspace(3) %60, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 392
  store double %.unpack3281, ptr addrspace(3) %.repack33, align 8
  %61 = getelementptr inbounds i8, ptr addrspace(1) %44, i64 21504
  %62 = load <2 x double>, ptr addrspace(1) %61, align 16, !invariant.load !4
  %.unpack3582 = extractelement <2 x double> %62, i32 0
  %.unpack3783 = extractelement <2 x double> %62, i32 1
  %63 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 448
  store double %.unpack3582, ptr addrspace(3) %63, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %50, i64 456
  store double %.unpack3783, ptr addrspace(3) %.repack38, align 8
  br label %.critedge72

.critedge72:                                      ; preds = %31, %.critedge
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %64 = shl nuw nsw i32 %.decomposed, 5
  %65 = or disjoint i32 %64, %7
  %66 = icmp samesign ult i32 %65, 620
  br i1 %66, label %67, label %102

67:                                               ; preds = %.critedge72
  %68 = lshr i32 %5, 5
  %69 = mul nuw nsw i32 %68, 33
  %70 = add nuw nsw i32 %69, %7
  %71 = zext nneg i32 %70 to i64
  %72 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %71
  %.unpack40 = load double, ptr addrspace(3) %72, align 8
  %.elt41 = getelementptr inbounds i8, ptr addrspace(3) %72, i64 8
  %.unpack42 = load double, ptr addrspace(3) %.elt41, align 8
  %73 = mul nuw nsw i32 %.pre-phi, 14880
  %74 = mul nuw nsw i32 %68, 620
  %75 = or disjoint i32 %73, %7
  %76 = add nuw nsw i32 %75, %64
  %77 = add nuw nsw i32 %76, %74
  %78 = zext nneg i32 %77 to i64
  %79 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %78
  %80 = insertelement <2 x double> poison, double %.unpack40, i32 0
  %81 = insertelement <2 x double> %80, double %.unpack42, i32 1
  store <2 x double> %81, ptr addrspace(1) %79, align 16
  %82 = getelementptr inbounds i8, ptr addrspace(3) %72, i64 2112
  %.unpack45 = load double, ptr addrspace(3) %82, align 8
  %.elt46 = getelementptr inbounds i8, ptr addrspace(3) %72, i64 2120
  %.unpack47 = load double, ptr addrspace(3) %.elt46, align 8
  %83 = getelementptr inbounds i8, ptr addrspace(1) %79, i64 39680
  %84 = insertelement <2 x double> poison, double %.unpack45, i32 0
  %85 = insertelement <2 x double> %84, double %.unpack47, i32 1
  store <2 x double> %85, ptr addrspace(1) %83, align 16
  %86 = getelementptr inbounds i8, ptr addrspace(3) %72, i64 4224
  %.unpack50 = load double, ptr addrspace(3) %86, align 8
  %.elt51 = getelementptr inbounds i8, ptr addrspace(3) %72, i64 4232
  %.unpack52 = load double, ptr addrspace(3) %.elt51, align 8
  %87 = getelementptr inbounds i8, ptr addrspace(1) %79, i64 79360
  %88 = insertelement <2 x double> poison, double %.unpack50, i32 0
  %89 = insertelement <2 x double> %88, double %.unpack52, i32 1
  store <2 x double> %89, ptr addrspace(1) %87, align 16
  %90 = getelementptr inbounds i8, ptr addrspace(3) %72, i64 6336
  %.unpack55 = load double, ptr addrspace(3) %90, align 8
  %.elt56 = getelementptr inbounds i8, ptr addrspace(3) %72, i64 6344
  %.unpack57 = load double, ptr addrspace(3) %.elt56, align 8
  %91 = getelementptr inbounds i8, ptr addrspace(1) %79, i64 119040
  %92 = insertelement <2 x double> poison, double %.unpack55, i32 0
  %93 = insertelement <2 x double> %92, double %.unpack57, i32 1
  store <2 x double> %93, ptr addrspace(1) %91, align 16
  %94 = getelementptr inbounds i8, ptr addrspace(3) %72, i64 8448
  %.unpack60 = load double, ptr addrspace(3) %94, align 8
  %.elt61 = getelementptr inbounds i8, ptr addrspace(3) %72, i64 8456
  %.unpack62 = load double, ptr addrspace(3) %.elt61, align 8
  %95 = getelementptr inbounds i8, ptr addrspace(1) %79, i64 158720
  %96 = insertelement <2 x double> poison, double %.unpack60, i32 0
  %97 = insertelement <2 x double> %96, double %.unpack62, i32 1
  store <2 x double> %97, ptr addrspace(1) %95, align 16
  %98 = getelementptr inbounds i8, ptr addrspace(3) %72, i64 10560
  %.unpack65 = load double, ptr addrspace(3) %98, align 8
  %.elt66 = getelementptr inbounds i8, ptr addrspace(3) %72, i64 10568
  %.unpack67 = load double, ptr addrspace(3) %.elt66, align 8
  %99 = getelementptr inbounds i8, ptr addrspace(1) %79, i64 198400
  %100 = insertelement <2 x double> poison, double %.unpack65, i32 0
  %101 = insertelement <2 x double> %100, double %.unpack67, i32 1
  store <2 x double> %101, ptr addrspace(1) %99, align 16
  br label %102

102:                                              ; preds = %67, %.critedge72
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_complex_fusion_1(ptr noalias readonly align 16 captures(none) dereferenceable(243793920) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(121896960) %1) local_unnamed_addr #3 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = shl nuw nsw i32 %6, 1
  %8 = shl nuw nsw i32 %5, 8
  %9 = or disjoint i32 %7, %8
  %10 = udiv i32 %9, 155
  %11 = trunc i32 %10 to i1
  %12 = select i1 %11, i32 310, i32 0
  %13 = shl nuw nsw i32 %5, 7
  %14 = or disjoint i32 %13, %6
  %15 = udiv i32 %14, 155
  %16 = urem i32 %15, 24
  %17 = mul nuw nsw i32 %16, 620
  %18 = udiv i32 %14, 3720
  %19 = mul nuw nsw i32 %18, 29760
  %20 = add nuw nsw i32 %17, %19
  %21 = add nuw nsw i32 %20, %12
  %22 = shl nuw nsw i32 %6, 2
  %23 = shl nuw nsw i32 %5, 9
  %24 = or disjoint i32 %22, %23
  %25 = urem i32 %24, 310
  %26 = add nuw nsw i32 %21, %25
  %27 = zext nneg i32 %26 to i64
  %28 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %27
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack29 = extractelement <2 x double> %29, i32 0
  %.unpack230 = extractelement <2 x double> %29, i32 1
  %30 = fneg double %.unpack230
  %31 = zext nneg i32 %24 to i64
  %32 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %31
  %33 = insertelement <2 x double> poison, double %.unpack29, i32 0
  %34 = insertelement <2 x double> %33, double %30, i32 1
  store <2 x double> %34, ptr addrspace(1) %32, align 64
  %35 = or disjoint i32 %24, 1
  %36 = urem i32 %35, 310
  %37 = add nuw nsw i32 %21, %36
  %38 = zext nneg i32 %37 to i64
  %39 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %38
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !4
  %.unpack527 = extractelement <2 x double> %40, i32 0
  %.unpack728 = extractelement <2 x double> %40, i32 1
  %41 = fneg double %.unpack728
  %42 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 16
  %43 = insertelement <2 x double> poison, double %.unpack527, i32 0
  %44 = insertelement <2 x double> %43, double %41, i32 1
  store <2 x double> %44, ptr addrspace(1) %42, align 16
  %45 = or disjoint i32 %9, 1
  %46 = udiv i32 %45, 155
  %47 = trunc i32 %46 to i1
  %48 = select i1 %47, i32 310, i32 0
  %49 = or disjoint i32 %24, 2
  %50 = urem i32 %49, 310
  %51 = add nuw nsw i32 %20, %50
  %52 = add nuw nsw i32 %51, %48
  %53 = zext nneg i32 %52 to i64
  %54 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %53
  %55 = load <2 x double>, ptr addrspace(1) %54, align 16, !invariant.load !4
  %.unpack1025 = extractelement <2 x double> %55, i32 0
  %.unpack1226 = extractelement <2 x double> %55, i32 1
  %56 = fneg double %.unpack1226
  %57 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 32
  %58 = insertelement <2 x double> poison, double %.unpack1025, i32 0
  %59 = insertelement <2 x double> %58, double %56, i32 1
  store <2 x double> %59, ptr addrspace(1) %57, align 32
  %60 = or disjoint i32 %24, 3
  %61 = udiv i32 %60, 310
  %62 = trunc i32 %61 to i1
  %63 = select i1 %62, i32 310, i32 0
  %64 = mul i32 %61, 310
  %.decomposed = sub i32 %60, %64
  %65 = add nuw nsw i32 %20, %.decomposed
  %66 = add nuw nsw i32 %65, %63
  %67 = zext nneg i32 %66 to i64
  %68 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %67
  %69 = load <2 x double>, ptr addrspace(1) %68, align 16, !invariant.load !4
  %.unpack1523 = extractelement <2 x double> %69, i32 0
  %.unpack1724 = extractelement <2 x double> %69, i32 1
  %70 = fneg double %.unpack1724
  %71 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 48
  %72 = insertelement <2 x double> poison, double %.unpack1523, i32 0
  %73 = insertelement <2 x double> %72, double %70, i32 1
  store <2 x double> %73, ptr addrspace(1) %71, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(3149004800) %0, ptr noalias readonly align 256 captures(none) dereferenceable(787251200) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(3149004800) %2) local_unnamed_addr #3 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %9 = shl nuw nsw i32 %8, 2
  %10 = shl nuw nsw i32 %7, 9
  %11 = or disjoint i32 %9, %10
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %12
  %14 = load <2 x double>, ptr addrspace(1) %13, align 64, !invariant.load !4
  %.unpack38 = extractelement <2 x double> %14, i32 0
  %.unpack239 = extractelement <2 x double> %14, i32 1
  %15 = shl nuw nsw i32 %7, 7
  %16 = or disjoint i32 %15, %8
  %17 = udiv i32 %16, 155
  %18 = urem i32 %17, 310
  %19 = mul nuw nsw i32 %18, 310
  %20 = urem i32 %16, 96100
  %21 = sub nuw nsw i32 %16, %20
  %22 = add nuw nsw i32 %19, %21
  %23 = urem i32 %11, 310
  %24 = add nuw nsw i32 %22, %23
  %25 = zext nneg i32 %24 to i64
  %26 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %25
  %27 = load <2 x double>, ptr addrspace(1) %26, align 16, !invariant.load !4
  %.unpack352 = extractelement <2 x double> %27, i32 0
  %.unpack553 = extractelement <2 x double> %27, i32 1
  %28 = fmul double %.unpack38, %.unpack352
  %29 = fmul double %.unpack239, %.unpack553
  %30 = fsub double %28, %29
  %31 = fmul double %.unpack239, %.unpack352
  %32 = fmul double %.unpack38, %.unpack553
  %33 = fadd double %31, %32
  %34 = fmul double %30, 0xBFA6A09E667F3BCC
  %35 = fmul double %33, 0.000000e+00
  %36 = fsub double %34, %35
  %37 = fmul double %33, 0x3FA6A09E667F3BCC
  %38 = fmul double %30, 0.000000e+00
  %39 = fsub double %38, %37
  %40 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %12
  %41 = insertelement <2 x double> poison, double %36, i32 0
  %42 = insertelement <2 x double> %41, double %39, i32 1
  store <2 x double> %42, ptr addrspace(1) %40, align 64
  %43 = or disjoint i32 %11, 1
  %44 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 16
  %45 = load <2 x double>, ptr addrspace(1) %44, align 16, !invariant.load !4
  %.unpack840 = extractelement <2 x double> %45, i32 0
  %.unpack1041 = extractelement <2 x double> %45, i32 1
  %46 = urem i32 %43, 310
  %47 = add nuw nsw i32 %22, %46
  %48 = zext nneg i32 %47 to i64
  %49 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %48
  %50 = load <2 x double>, ptr addrspace(1) %49, align 16, !invariant.load !4
  %.unpack1150 = extractelement <2 x double> %50, i32 0
  %.unpack1351 = extractelement <2 x double> %50, i32 1
  %51 = fmul double %.unpack840, %.unpack1150
  %52 = fmul double %.unpack1041, %.unpack1351
  %53 = fsub double %51, %52
  %54 = fmul double %.unpack1041, %.unpack1150
  %55 = fmul double %.unpack840, %.unpack1351
  %56 = fadd double %54, %55
  %57 = fmul double %53, 0xBFA6A09E667F3BCC
  %58 = fmul double %56, 0.000000e+00
  %59 = fsub double %57, %58
  %60 = fmul double %56, 0x3FA6A09E667F3BCC
  %61 = fmul double %53, 0.000000e+00
  %62 = fsub double %61, %60
  %63 = getelementptr inbounds i8, ptr addrspace(1) %40, i64 16
  %64 = insertelement <2 x double> poison, double %59, i32 0
  %65 = insertelement <2 x double> %64, double %62, i32 1
  store <2 x double> %65, ptr addrspace(1) %63, align 16
  %66 = or disjoint i32 %11, 2
  %67 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 32
  %68 = load <2 x double>, ptr addrspace(1) %67, align 32, !invariant.load !4
  %.unpack1642 = extractelement <2 x double> %68, i32 0
  %.unpack1843 = extractelement <2 x double> %68, i32 1
  %69 = urem i32 %66, 310
  %70 = add nuw nsw i32 %22, %69
  %71 = zext nneg i32 %70 to i64
  %72 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %71
  %73 = load <2 x double>, ptr addrspace(1) %72, align 16, !invariant.load !4
  %.unpack1948 = extractelement <2 x double> %73, i32 0
  %.unpack2149 = extractelement <2 x double> %73, i32 1
  %74 = fmul double %.unpack1642, %.unpack1948
  %75 = fmul double %.unpack1843, %.unpack2149
  %76 = fsub double %74, %75
  %77 = fmul double %.unpack1843, %.unpack1948
  %78 = fmul double %.unpack1642, %.unpack2149
  %79 = fadd double %77, %78
  %80 = fmul double %76, 0xBFA6A09E667F3BCC
  %81 = fmul double %79, 0.000000e+00
  %82 = fsub double %80, %81
  %83 = fmul double %79, 0x3FA6A09E667F3BCC
  %84 = fmul double %76, 0.000000e+00
  %85 = fsub double %84, %83
  %86 = getelementptr inbounds i8, ptr addrspace(1) %40, i64 32
  %87 = insertelement <2 x double> poison, double %82, i32 0
  %88 = insertelement <2 x double> %87, double %85, i32 1
  store <2 x double> %88, ptr addrspace(1) %86, align 32
  %89 = or disjoint i32 %11, 3
  %90 = getelementptr inbounds i8, ptr addrspace(1) %13, i64 48
  %91 = load <2 x double>, ptr addrspace(1) %90, align 16, !invariant.load !4
  %.unpack2444 = extractelement <2 x double> %91, i32 0
  %.unpack2645 = extractelement <2 x double> %91, i32 1
  %92 = urem i32 %89, 310
  %93 = add nuw nsw i32 %22, %92
  %94 = zext nneg i32 %93 to i64
  %95 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %94
  %96 = load <2 x double>, ptr addrspace(1) %95, align 16, !invariant.load !4
  %.unpack2746 = extractelement <2 x double> %96, i32 0
  %.unpack2947 = extractelement <2 x double> %96, i32 1
  %97 = fmul double %.unpack2444, %.unpack2746
  %98 = fmul double %.unpack2645, %.unpack2947
  %99 = fsub double %97, %98
  %100 = fmul double %.unpack2645, %.unpack2746
  %101 = fmul double %.unpack2444, %.unpack2947
  %102 = fadd double %100, %101
  %103 = fmul double %99, 0xBFA6A09E667F3BCC
  %104 = fmul double %102, 0.000000e+00
  %105 = fsub double %103, %104
  %106 = fmul double %102, 0x3FA6A09E667F3BCC
  %107 = fmul double %99, 0.000000e+00
  %108 = fsub double %107, %106
  %109 = getelementptr inbounds i8, ptr addrspace(1) %40, i64 48
  %110 = insertelement <2 x double> poison, double %105, i32 0
  %111 = insertelement <2 x double> %110, double %108, i32 1
  store <2 x double> %111, ptr addrspace(1) %109, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_slice(ptr noalias readonly align 16 captures(none) dereferenceable(243793920) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(121896960) %1) local_unnamed_addr #3 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = shl nuw nsw i32 %5, 7
  %8 = or disjoint i32 %7, %6
  %9 = udiv i32 %8, 6
  %10 = shl nuw nsw i32 %8, 2
  %11 = mul nuw nsw i32 %9, 24
  %12 = add nuw nsw i32 %11, %10
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %13
  %15 = load <2 x double>, ptr addrspace(1) %14, align 16, !invariant.load !4
  %.unpack20 = extractelement <2 x double> %15, i32 0
  %.unpack221 = extractelement <2 x double> %15, i32 1
  %16 = shl nuw nsw i32 %6, 2
  %17 = shl nuw nsw i32 %5, 9
  %18 = or disjoint i32 %16, %17
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %19
  %21 = insertelement <2 x double> poison, double %.unpack20, i32 0
  %22 = insertelement <2 x double> %21, double %.unpack221, i32 1
  store <2 x double> %22, ptr addrspace(1) %20, align 64
  %23 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 16
  %24 = load <2 x double>, ptr addrspace(1) %23, align 16, !invariant.load !4
  %.unpack522 = extractelement <2 x double> %24, i32 0
  %.unpack723 = extractelement <2 x double> %24, i32 1
  %25 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 16
  %26 = insertelement <2 x double> poison, double %.unpack522, i32 0
  %27 = insertelement <2 x double> %26, double %.unpack723, i32 1
  store <2 x double> %27, ptr addrspace(1) %25, align 16
  %28 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 32
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack1024 = extractelement <2 x double> %29, i32 0
  %.unpack1225 = extractelement <2 x double> %29, i32 1
  %30 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 32
  %31 = insertelement <2 x double> poison, double %.unpack1024, i32 0
  %32 = insertelement <2 x double> %31, double %.unpack1225, i32 1
  store <2 x double> %32, ptr addrspace(1) %30, align 32
  %33 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 48
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !4
  %.unpack1526 = extractelement <2 x double> %34, i32 0
  %.unpack1727 = extractelement <2 x double> %34, i32 1
  %35 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 48
  %36 = insertelement <2 x double> poison, double %.unpack1526, i32 0
  %37 = insertelement <2 x double> %36, double %.unpack1727, i32 1
  store <2 x double> %37, ptr addrspace(1) %35, align 16
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
!3 = !{i32 0, i32 10240}
!4 = !{}
!5 = !{i32 0, i32 14880}
!6 = !{i32 0, i32 384400}
