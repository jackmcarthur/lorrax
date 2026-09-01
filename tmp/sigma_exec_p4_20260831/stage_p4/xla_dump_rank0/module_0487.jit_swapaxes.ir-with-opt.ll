; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @wrapped_transpose(ptr noalias readonly align 16 captures(none) dereferenceable(4718592) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4718592) %1) local_unnamed_addr #0 {
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
  %.unpack60 = extractelement <2 x double> %17, i32 0
  %.unpack261 = extractelement <2 x double> %17, i32 1
  %18 = mul nuw nsw i32 %7, 33
  %19 = add nuw nsw i32 %18, %10
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %20
  store double %.unpack60, ptr addrspace(3) %21, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 8
  store double %.unpack261, ptr addrspace(3) %.repack3, align 8
  %22 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 1536
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !4
  %.unpack562 = extractelement <2 x double> %23, i32 0
  %.unpack763 = extractelement <2 x double> %23, i32 1
  %24 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 64
  store double %.unpack562, ptr addrspace(3) %24, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 72
  store double %.unpack763, ptr addrspace(3) %.repack8, align 8
  %25 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 3072
  %26 = load <2 x double>, ptr addrspace(1) %25, align 16, !invariant.load !4
  %.unpack1064 = extractelement <2 x double> %26, i32 0
  %.unpack1265 = extractelement <2 x double> %26, i32 1
  %27 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 128
  store double %.unpack1064, ptr addrspace(3) %27, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 136
  store double %.unpack1265, ptr addrspace(3) %.repack13, align 8
  %28 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 4608
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack1566 = extractelement <2 x double> %29, i32 0
  %.unpack1767 = extractelement <2 x double> %29, i32 1
  %30 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 192
  store double %.unpack1566, ptr addrspace(3) %30, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 200
  store double %.unpack1767, ptr addrspace(3) %.repack18, align 8
  %31 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 6144
  %32 = load <2 x double>, ptr addrspace(1) %31, align 16, !invariant.load !4
  %.unpack2068 = extractelement <2 x double> %32, i32 0
  %.unpack2269 = extractelement <2 x double> %32, i32 1
  %33 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 256
  store double %.unpack2068, ptr addrspace(3) %33, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 264
  store double %.unpack2269, ptr addrspace(3) %.repack23, align 8
  %34 = getelementptr inbounds i8, ptr addrspace(1) %16, i64 7680
  %35 = load <2 x double>, ptr addrspace(1) %34, align 16, !invariant.load !4
  %.unpack2570 = extractelement <2 x double> %35, i32 0
  %.unpack2771 = extractelement <2 x double> %35, i32 1
  %36 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 320
  store double %.unpack2570, ptr addrspace(3) %36, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %21, i64 328
  store double %.unpack2771, ptr addrspace(3) %.repack28, align 8
  br label %37

37:                                               ; preds = %9, %2
  %38 = icmp ult i32 %7, 24
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br i1 %38, label %39, label %73

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
  %50 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %49
  %51 = insertelement <2 x double> poison, double %.unpack30, i32 0
  %52 = insertelement <2 x double> %51, double %.unpack32, i32 1
  store <2 x double> %52, ptr addrspace(1) %50, align 16
  %53 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 2112
  %.unpack35 = load double, ptr addrspace(3) %53, align 8
  %.elt36 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 2120
  %.unpack37 = load double, ptr addrspace(3) %.elt36, align 8
  %54 = getelementptr inbounds i8, ptr addrspace(1) %50, i64 1536
  %55 = insertelement <2 x double> poison, double %.unpack35, i32 0
  %56 = insertelement <2 x double> %55, double %.unpack37, i32 1
  store <2 x double> %56, ptr addrspace(1) %54, align 16
  %57 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 4224
  %.unpack40 = load double, ptr addrspace(3) %57, align 8
  %.elt41 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 4232
  %.unpack42 = load double, ptr addrspace(3) %.elt41, align 8
  %58 = getelementptr inbounds i8, ptr addrspace(1) %50, i64 3072
  %59 = insertelement <2 x double> poison, double %.unpack40, i32 0
  %60 = insertelement <2 x double> %59, double %.unpack42, i32 1
  store <2 x double> %60, ptr addrspace(1) %58, align 16
  %61 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 6336
  %.unpack45 = load double, ptr addrspace(3) %61, align 8
  %.elt46 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 6344
  %.unpack47 = load double, ptr addrspace(3) %.elt46, align 8
  %62 = getelementptr inbounds i8, ptr addrspace(1) %50, i64 4608
  %63 = insertelement <2 x double> poison, double %.unpack45, i32 0
  %64 = insertelement <2 x double> %63, double %.unpack47, i32 1
  store <2 x double> %64, ptr addrspace(1) %62, align 16
  %65 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 8448
  %.unpack50 = load double, ptr addrspace(3) %65, align 8
  %.elt51 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 8456
  %.unpack52 = load double, ptr addrspace(3) %.elt51, align 8
  %66 = getelementptr inbounds i8, ptr addrspace(1) %50, i64 6144
  %67 = insertelement <2 x double> poison, double %.unpack50, i32 0
  %68 = insertelement <2 x double> %67, double %.unpack52, i32 1
  store <2 x double> %68, ptr addrspace(1) %66, align 16
  %69 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 10560
  %.unpack55 = load double, ptr addrspace(3) %69, align 8
  %.elt56 = getelementptr inbounds i8, ptr addrspace(3) %44, i64 10568
  %.unpack57 = load double, ptr addrspace(3) %.elt56, align 8
  %70 = getelementptr inbounds i8, ptr addrspace(1) %50, i64 7680
  %71 = insertelement <2 x double> poison, double %.unpack55, i32 0
  %72 = insertelement <2 x double> %71, double %.unpack57, i32 1
  store <2 x double> %72, ptr addrspace(1) %70, align 16
  br label %73

73:                                               ; preds = %39, %37
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
!3 = !{i32 0, i32 512}
!4 = !{}
