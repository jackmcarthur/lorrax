; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @wrapped_transpose(ptr noalias readonly align 16 captures(none) dereferenceable(243793920) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(243793920) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %7 = udiv i32 %6, 10
  %8 = mul i32 %7, 10
  %.decomposed = sub i32 %6, %8
  %9 = shl nuw nsw i32 %.decomposed, 5
  %10 = and i32 %5, 31
  %11 = or disjoint i32 %9, %10
  %12 = icmp samesign ult i32 %11, 310
  br i1 %12, label %13, label %._crit_edge

._crit_edge:                                      ; preds = %2
  %.pre = lshr i32 %5, 5
  br label %48

13:                                               ; preds = %2
  %14 = mul nuw nsw i32 %7, 9920
  %15 = lshr i32 %5, 5
  %16 = mul nuw nsw i32 %15, 310
  %17 = or disjoint i32 %14, %10
  %18 = add nuw nsw i32 %17, %9
  %19 = add nuw nsw i32 %18, %16
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %20
  %22 = load <2 x double>, ptr addrspace(1) %21, align 16, !invariant.load !4
  %.unpack82 = extractelement <2 x double> %22, i32 0
  %.unpack283 = extractelement <2 x double> %22, i32 1
  %23 = mul nuw nsw i32 %10, 33
  %24 = add nuw nsw i32 %23, %15
  %25 = zext nneg i32 %24 to i64
  %26 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %25
  store double %.unpack82, ptr addrspace(3) %26, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 8
  store double %.unpack283, ptr addrspace(3) %.repack3, align 8
  %27 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 19840
  %28 = load <2 x double>, ptr addrspace(1) %27, align 16, !invariant.load !4
  %.unpack584 = extractelement <2 x double> %28, i32 0
  %.unpack785 = extractelement <2 x double> %28, i32 1
  %29 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 64
  store double %.unpack584, ptr addrspace(3) %29, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 72
  store double %.unpack785, ptr addrspace(3) %.repack8, align 8
  %30 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 39680
  %31 = load <2 x double>, ptr addrspace(1) %30, align 16, !invariant.load !4
  %.unpack1086 = extractelement <2 x double> %31, i32 0
  %.unpack1287 = extractelement <2 x double> %31, i32 1
  %32 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 128
  store double %.unpack1086, ptr addrspace(3) %32, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 136
  store double %.unpack1287, ptr addrspace(3) %.repack13, align 8
  %33 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 59520
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !4
  %.unpack1588 = extractelement <2 x double> %34, i32 0
  %.unpack1789 = extractelement <2 x double> %34, i32 1
  %35 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 192
  store double %.unpack1588, ptr addrspace(3) %35, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 200
  store double %.unpack1789, ptr addrspace(3) %.repack18, align 8
  %36 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 79360
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !4
  %.unpack2090 = extractelement <2 x double> %37, i32 0
  %.unpack2291 = extractelement <2 x double> %37, i32 1
  %38 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 256
  store double %.unpack2090, ptr addrspace(3) %38, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 264
  store double %.unpack2291, ptr addrspace(3) %.repack23, align 8
  %39 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 99200
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !4
  %.unpack2592 = extractelement <2 x double> %40, i32 0
  %.unpack2793 = extractelement <2 x double> %40, i32 1
  %41 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 320
  store double %.unpack2592, ptr addrspace(3) %41, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 328
  store double %.unpack2793, ptr addrspace(3) %.repack28, align 8
  %42 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 119040
  %43 = load <2 x double>, ptr addrspace(1) %42, align 16, !invariant.load !4
  %.unpack3094 = extractelement <2 x double> %43, i32 0
  %.unpack3295 = extractelement <2 x double> %43, i32 1
  %44 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 384
  store double %.unpack3094, ptr addrspace(3) %44, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 392
  store double %.unpack3295, ptr addrspace(3) %.repack33, align 8
  %45 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 138880
  %46 = load <2 x double>, ptr addrspace(1) %45, align 16, !invariant.load !4
  %.unpack3596 = extractelement <2 x double> %46, i32 0
  %.unpack3797 = extractelement <2 x double> %46, i32 1
  %47 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 448
  store double %.unpack3596, ptr addrspace(3) %47, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %26, i64 456
  store double %.unpack3797, ptr addrspace(3) %.repack38, align 8
  br label %48

48:                                               ; preds = %._crit_edge, %13
  %.pre-phi = phi i32 [ %.pre, %._crit_edge ], [ %15, %13 ]
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %49 = mul nuw nsw i32 %.pre-phi, 33
  %50 = add nuw nsw i32 %49, %10
  %51 = zext nneg i32 %50 to i64
  %52 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %51
  %.unpack40 = load double, ptr addrspace(3) %52, align 8
  %.elt41 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 8
  %.unpack42 = load double, ptr addrspace(3) %.elt41, align 8
  %53 = urem i32 %7, 3
  %54 = shl nuw nsw i32 %53, 5
  %55 = mul nuw nsw i32 %.decomposed, 3072
  %56 = udiv i32 %6, 30
  %57 = mul nuw nsw i32 %56, 29760
  %58 = mul nuw nsw i32 %.pre-phi, 96
  %59 = or disjoint i32 %10, %55
  %60 = add nuw nsw i32 %59, %57
  %61 = add nuw nsw i32 %60, %58
  %62 = add nuw nsw i32 %61, %54
  %63 = zext nneg i32 %62 to i64
  %64 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %63
  %65 = insertelement <2 x double> poison, double %.unpack40, i32 0
  %66 = insertelement <2 x double> %65, double %.unpack42, i32 1
  store <2 x double> %66, ptr addrspace(1) %64, align 16
  %67 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 2112
  %.unpack45 = load double, ptr addrspace(3) %67, align 8
  %.elt46 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 2120
  %.unpack47 = load double, ptr addrspace(3) %.elt46, align 8
  %68 = getelementptr inbounds i8, ptr addrspace(1) %64, i64 6144
  %69 = insertelement <2 x double> poison, double %.unpack45, i32 0
  %70 = insertelement <2 x double> %69, double %.unpack47, i32 1
  store <2 x double> %70, ptr addrspace(1) %68, align 16
  %71 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 4224
  %.unpack50 = load double, ptr addrspace(3) %71, align 8
  %.elt51 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 4232
  %.unpack52 = load double, ptr addrspace(3) %.elt51, align 8
  %72 = getelementptr inbounds i8, ptr addrspace(1) %64, i64 12288
  %73 = insertelement <2 x double> poison, double %.unpack50, i32 0
  %74 = insertelement <2 x double> %73, double %.unpack52, i32 1
  store <2 x double> %74, ptr addrspace(1) %72, align 16
  %75 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 6336
  %.unpack55 = load double, ptr addrspace(3) %75, align 8
  %.elt56 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 6344
  %.unpack57 = load double, ptr addrspace(3) %.elt56, align 8
  %76 = getelementptr inbounds i8, ptr addrspace(1) %64, i64 18432
  %77 = insertelement <2 x double> poison, double %.unpack55, i32 0
  %78 = insertelement <2 x double> %77, double %.unpack57, i32 1
  store <2 x double> %78, ptr addrspace(1) %76, align 16
  %79 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 8448
  %.unpack60 = load double, ptr addrspace(3) %79, align 8
  %.elt61 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 8456
  %.unpack62 = load double, ptr addrspace(3) %.elt61, align 8
  %80 = getelementptr inbounds i8, ptr addrspace(1) %64, i64 24576
  %81 = insertelement <2 x double> poison, double %.unpack60, i32 0
  %82 = insertelement <2 x double> %81, double %.unpack62, i32 1
  store <2 x double> %82, ptr addrspace(1) %80, align 16
  %83 = or disjoint i32 %9, %.pre-phi
  %84 = icmp samesign ult i32 %83, 290
  br i1 %84, label %85, label %88

85:                                               ; preds = %48
  %sunkaddr = getelementptr inbounds i8, ptr addrspace(3) %52, i64 10560
  %.unpack65 = load double, ptr addrspace(3) %sunkaddr, align 8
  %sunkaddr98 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 10568
  %.unpack67 = load double, ptr addrspace(3) %sunkaddr98, align 8
  %86 = insertelement <2 x double> poison, double %.unpack65, i32 0
  %87 = insertelement <2 x double> %86, double %.unpack67, i32 1
  %sunkaddr99 = getelementptr inbounds i8, ptr addrspace(1) %64, i64 30720
  store <2 x double> %87, ptr addrspace(1) %sunkaddr99, align 16
  br label %88

88:                                               ; preds = %85, %48
  %89 = icmp samesign ult i32 %83, 286
  br i1 %89, label %90, label %93

90:                                               ; preds = %88
  %sunkaddr100 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 12672
  %.unpack70 = load double, ptr addrspace(3) %sunkaddr100, align 8
  %sunkaddr101 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 12680
  %.unpack72 = load double, ptr addrspace(3) %sunkaddr101, align 8
  %91 = insertelement <2 x double> poison, double %.unpack70, i32 0
  %92 = insertelement <2 x double> %91, double %.unpack72, i32 1
  %sunkaddr102 = getelementptr inbounds i8, ptr addrspace(1) %64, i64 36864
  store <2 x double> %92, ptr addrspace(1) %sunkaddr102, align 16
  br label %93

93:                                               ; preds = %90, %88
  %94 = icmp samesign ult i32 %83, 282
  br i1 %94, label %95, label %98

95:                                               ; preds = %93
  %sunkaddr103 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 14784
  %.unpack75 = load double, ptr addrspace(3) %sunkaddr103, align 8
  %sunkaddr104 = getelementptr inbounds i8, ptr addrspace(3) %52, i64 14792
  %.unpack77 = load double, ptr addrspace(3) %sunkaddr104, align 8
  %96 = insertelement <2 x double> poison, double %.unpack75, i32 0
  %97 = insertelement <2 x double> %96, double %.unpack77, i32 1
  %sunkaddr105 = getelementptr inbounds i8, ptr addrspace(1) %64, i64 43008
  store <2 x double> %97, ptr addrspace(1) %sunkaddr105, align 16
  br label %98

98:                                               ; preds = %95, %93
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

attributes #0 = { norecurse nounwind "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 128}
!3 = !{i32 0, i32 15360}
!4 = !{}
