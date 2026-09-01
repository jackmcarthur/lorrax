; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @wrapped_transpose(ptr noalias readonly align 16 captures(none) dereferenceable(1179648) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(1179648) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %7 = and i32 %5, 31
  %8 = icmp samesign ult i32 %7, 12
  %9 = lshr i32 %5, 5
  br i1 %8, label %10, label %._crit_edge

10:                                               ; preds = %2
  %11 = mul nuw nsw i32 %9, 12
  %12 = mul nuw nsw i32 %6, 384
  %13 = or disjoint i32 %7, %12
  %14 = add nuw nsw i32 %13, %11
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %15
  %17 = load <2 x double>, ptr addrspace(1) %16, align 16, !invariant.load !4
  %.unpack55 = extractelement <2 x double> %17, i32 0
  %.unpack256 = extractelement <2 x double> %17, i32 1
  %18 = mul nuw nsw i32 %7, 33
  %19 = add nuw nsw i32 %18, %9
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %20
  store double %.unpack55, ptr addrspace(3) %21, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 8
  store double %.unpack256, ptr addrspace(3) %.repack3, align 8
  %22 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 768
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !4
  %.unpack557 = extractelement <2 x double> %23, i32 0
  %.unpack758 = extractelement <2 x double> %23, i32 1
  %24 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 64
  store double %.unpack557, ptr addrspace(3) %24, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 72
  store double %.unpack758, ptr addrspace(3) %.repack8, align 8
  %25 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 1536
  %26 = load <2 x double>, ptr addrspace(1) %25, align 16, !invariant.load !4
  %.unpack1059 = extractelement <2 x double> %26, i32 0
  %.unpack1260 = extractelement <2 x double> %26, i32 1
  %27 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 128
  store double %.unpack1059, ptr addrspace(3) %27, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 136
  store double %.unpack1260, ptr addrspace(3) %.repack13, align 8
  %28 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 2304
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack1561 = extractelement <2 x double> %29, i32 0
  %.unpack1762 = extractelement <2 x double> %29, i32 1
  %30 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 192
  store double %.unpack1561, ptr addrspace(3) %30, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 200
  store double %.unpack1762, ptr addrspace(3) %.repack18, align 8
  %31 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 3072
  %32 = load <2 x double>, ptr addrspace(1) %31, align 16, !invariant.load !4
  %.unpack2063 = extractelement <2 x double> %32, i32 0
  %.unpack2264 = extractelement <2 x double> %32, i32 1
  %33 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 256
  store double %.unpack2063, ptr addrspace(3) %33, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 264
  store double %.unpack2264, ptr addrspace(3) %.repack23, align 8
  %34 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 3840
  %35 = load <2 x double>, ptr addrspace(1) %34, align 16, !invariant.load !4
  %.unpack2565 = extractelement <2 x double> %35, i32 0
  %.unpack2766 = extractelement <2 x double> %35, i32 1
  %36 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 320
  store double %.unpack2565, ptr addrspace(3) %36, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 328
  store double %.unpack2766, ptr addrspace(3) %.repack28, align 8
  %37 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 4608
  %38 = load <2 x double>, ptr addrspace(1) %37, align 16, !invariant.load !4
  %.unpack3067 = extractelement <2 x double> %38, i32 0
  %.unpack3268 = extractelement <2 x double> %38, i32 1
  %39 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 384
  store double %.unpack3067, ptr addrspace(3) %39, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 392
  store double %.unpack3268, ptr addrspace(3) %.repack33, align 8
  %40 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 5376
  %41 = load <2 x double>, ptr addrspace(1) %40, align 16, !invariant.load !4
  %.unpack3569 = extractelement <2 x double> %41, i32 0
  %.unpack3770 = extractelement <2 x double> %41, i32 1
  %42 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 448
  store double %.unpack3569, ptr addrspace(3) %42, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 456
  store double %.unpack3770, ptr addrspace(3) %.repack38, align 8
  br label %._crit_edge

._crit_edge:                                      ; preds = %2, %10
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %43 = mul nuw nsw i32 %9, 33
  %44 = add nuw nsw i32 %43, %7
  %45 = zext nneg i32 %44 to i64
  %46 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %45
  %.unpack40 = load double, ptr addrspace(3) %46, align 8
  %.elt41 = getelementptr inbounds i8, ptr addrspace(3) %46, i64 8
  %.unpack42 = load double, ptr addrspace(3) %.elt41, align 8
  %47 = mul nuw nsw i32 %9, 6144
  %48 = shl nuw nsw i32 %6, 5
  %49 = add nuw nsw i32 %47, %48
  %50 = or disjoint i32 %49, %7
  %51 = zext nneg i32 %50 to i64
  %52 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %51
  %53 = insertelement <2 x double> poison, double %.unpack40, i32 0
  %54 = insertelement <2 x double> %53, double %.unpack42, i32 1
  store <2 x double> %54, ptr addrspace(1) %52, align 16
  %55 = getelementptr inbounds i8, ptr addrspace(3) %46, i64 2112
  %.unpack45 = load double, ptr addrspace(3) %55, align 8
  %.elt46 = getelementptr inbounds i8, ptr addrspace(3) %46, i64 2120
  %.unpack47 = load double, ptr addrspace(3) %.elt46, align 8
  %56 = getelementptr inbounds i8, ptr addrspace(1) %52, i64 393216
  %57 = insertelement <2 x double> poison, double %.unpack45, i32 0
  %58 = insertelement <2 x double> %57, double %.unpack47, i32 1
  store <2 x double> %58, ptr addrspace(1) %56, align 16
  %59 = getelementptr inbounds i8, ptr addrspace(3) %46, i64 4224
  %.unpack50 = load double, ptr addrspace(3) %59, align 8
  %.elt51 = getelementptr inbounds i8, ptr addrspace(3) %46, i64 4232
  %.unpack52 = load double, ptr addrspace(3) %.elt51, align 8
  %60 = getelementptr inbounds i8, ptr addrspace(1) %52, i64 786432
  %61 = insertelement <2 x double> poison, double %.unpack50, i32 0
  %62 = insertelement <2 x double> %61, double %.unpack52, i32 1
  store <2 x double> %62, ptr addrspace(1) %60, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_transpose_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(2359296) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(2359296) %1) local_unnamed_addr #3 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = shl nuw nsw i32 %5, 7
  %8 = or disjoint i32 %7, %6
  %9 = udiv i32 %8, 24
  %10 = mul i32 %9, 24
  %.decomposed = sub i32 %8, %10
  %11 = mul nuw nsw i32 %.decomposed, 6144
  %12 = and i32 %9, 511
  %13 = mul nuw nsw i32 %12, 12
  %14 = udiv i32 %5, 96
  %15 = or disjoint i32 %11, %14
  %16 = add nuw nsw i32 %15, %13
  %17 = zext nneg i32 %16 to i64
  %18 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %17
  %19 = load <2 x double>, ptr addrspace(1) %18, align 16, !invariant.load !4
  %.unpack5 = extractelement <2 x double> %19, i32 0
  %.unpack26 = extractelement <2 x double> %19, i32 1
  %20 = zext nneg i32 %8 to i64
  %21 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %20
  %22 = insertelement <2 x double> poison, double %.unpack5, i32 0
  %23 = insertelement <2 x double> %22, double %.unpack26, i32 1
  store <2 x double> %23, ptr addrspace(1) %21, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_transpose_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(4718592) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4718592) %1) local_unnamed_addr #3 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = shl nuw nsw i32 %5, 7
  %8 = or disjoint i32 %7, %6
  %9 = udiv i32 %8, 6
  %10 = mul i32 %9, 6
  %.decomposed = sub i32 %8, %10
  %11 = shl nuw nsw i32 %.decomposed, 2
  %12 = urem i32 %9, 24
  %13 = mul nuw nsw i32 %12, 12288
  %14 = udiv i32 %8, 144
  %15 = mul nuw nsw i32 %14, 24
  %16 = add nuw nsw i32 %15, %11
  %17 = add nuw nsw i32 %16, %13
  %18 = zext nneg i32 %17 to i64
  %19 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %18
  %20 = load <2 x double>, ptr addrspace(1) %19, align 64, !invariant.load !4
  %.unpack20 = extractelement <2 x double> %20, i32 0
  %.unpack221 = extractelement <2 x double> %20, i32 1
  %21 = shl nuw nsw i32 %6, 2
  %22 = shl nuw nsw i32 %5, 9
  %23 = or disjoint i32 %21, %22
  %24 = zext nneg i32 %23 to i64
  %25 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %24
  %26 = insertelement <2 x double> poison, double %.unpack20, i32 0
  %27 = insertelement <2 x double> %26, double %.unpack221, i32 1
  store <2 x double> %27, ptr addrspace(1) %25, align 64
  %28 = getelementptr inbounds i8, ptr addrspace(1) %19, i64 16
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack522 = extractelement <2 x double> %29, i32 0
  %.unpack723 = extractelement <2 x double> %29, i32 1
  %30 = getelementptr inbounds i8, ptr addrspace(1) %25, i64 16
  %31 = insertelement <2 x double> poison, double %.unpack522, i32 0
  %32 = insertelement <2 x double> %31, double %.unpack723, i32 1
  store <2 x double> %32, ptr addrspace(1) %30, align 16
  %33 = getelementptr inbounds i8, ptr addrspace(1) %19, i64 32
  %34 = load <2 x double>, ptr addrspace(1) %33, align 32, !invariant.load !4
  %.unpack1024 = extractelement <2 x double> %34, i32 0
  %.unpack1225 = extractelement <2 x double> %34, i32 1
  %35 = getelementptr inbounds i8, ptr addrspace(1) %25, i64 32
  %36 = insertelement <2 x double> poison, double %.unpack1024, i32 0
  %37 = insertelement <2 x double> %36, double %.unpack1225, i32 1
  store <2 x double> %37, ptr addrspace(1) %35, align 32
  %38 = getelementptr inbounds i8, ptr addrspace(1) %19, i64 48
  %39 = load <2 x double>, ptr addrspace(1) %38, align 16, !invariant.load !4
  %.unpack1526 = extractelement <2 x double> %39, i32 0
  %.unpack1727 = extractelement <2 x double> %39, i32 1
  %40 = getelementptr inbounds i8, ptr addrspace(1) %25, i64 48
  %41 = insertelement <2 x double> poison, double %.unpack1526, i32 0
  %42 = insertelement <2 x double> %41, double %.unpack1727, i32 1
  store <2 x double> %42, ptr addrspace(1) %40, align 16
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
!3 = !{i32 0, i32 192}
!4 = !{}
!5 = !{i32 0, i32 1152}
!6 = !{i32 0, i32 576}
