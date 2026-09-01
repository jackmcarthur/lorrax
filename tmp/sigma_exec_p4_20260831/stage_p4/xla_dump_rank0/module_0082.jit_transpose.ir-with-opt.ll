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
  %7 = udiv i32 %6, 3
  %8 = urem i32 %7, 10
  %9 = mul nuw nsw i32 %8, 3072
  %10 = mul i32 %7, 3
  %.decomposed = sub i32 %6, %10
  %11 = shl nuw nsw i32 %.decomposed, 5
  %12 = udiv i32 %6, 30
  %13 = mul nuw nsw i32 %12, 29760
  %14 = lshr i32 %5, 5
  %15 = mul nuw nsw i32 %14, 96
  %16 = and i32 %5, 31
  %17 = or disjoint i32 %16, %11
  %18 = add nuw nsw i32 %17, %13
  %19 = add nuw nsw i32 %18, %15
  %20 = add nuw nsw i32 %19, %9
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %21
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !4
  %.unpack86 = extractelement <2 x double> %23, i32 0
  %.unpack287 = extractelement <2 x double> %23, i32 1
  %24 = mul nuw nsw i32 %16, 33
  %25 = add nuw nsw i32 %24, %14
  %26 = zext nneg i32 %25 to i64
  %27 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %26
  store double %.unpack86, ptr addrspace(3) %27, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 8
  store double %.unpack287, ptr addrspace(3) %.repack3, align 8
  %28 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 6144
  %29 = load <2 x double>, ptr addrspace(1) %28, align 16, !invariant.load !4
  %.unpack588 = extractelement <2 x double> %29, i32 0
  %.unpack789 = extractelement <2 x double> %29, i32 1
  %30 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 64
  store double %.unpack588, ptr addrspace(3) %30, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 72
  store double %.unpack789, ptr addrspace(3) %.repack8, align 8
  %31 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 12288
  %32 = load <2 x double>, ptr addrspace(1) %31, align 16, !invariant.load !4
  %.unpack1090 = extractelement <2 x double> %32, i32 0
  %.unpack1291 = extractelement <2 x double> %32, i32 1
  %33 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 128
  store double %.unpack1090, ptr addrspace(3) %33, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 136
  store double %.unpack1291, ptr addrspace(3) %.repack13, align 8
  %34 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 18432
  %35 = load <2 x double>, ptr addrspace(1) %34, align 16, !invariant.load !4
  %.unpack1592 = extractelement <2 x double> %35, i32 0
  %.unpack1793 = extractelement <2 x double> %35, i32 1
  %36 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 192
  store double %.unpack1592, ptr addrspace(3) %36, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 200
  store double %.unpack1793, ptr addrspace(3) %.repack18, align 8
  %37 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 24576
  %38 = load <2 x double>, ptr addrspace(1) %37, align 16, !invariant.load !4
  %.unpack2094 = extractelement <2 x double> %38, i32 0
  %.unpack2295 = extractelement <2 x double> %38, i32 1
  %39 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 256
  store double %.unpack2094, ptr addrspace(3) %39, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 264
  store double %.unpack2295, ptr addrspace(3) %.repack23, align 8
  %40 = shl nuw nsw i32 %8, 5
  %41 = or disjoint i32 %40, %14
  %42 = icmp samesign ult i32 %41, 290
  br i1 %42, label %43, label %45

43:                                               ; preds = %2
  %sunkaddr = getelementptr inbounds i8, ptr addrspace(1) %22, i64 30720
  %44 = load <2 x double>, ptr addrspace(1) %sunkaddr, align 16, !invariant.load !4
  %.unpack2584 = extractelement <2 x double> %44, i32 0
  %.unpack2785 = extractelement <2 x double> %44, i32 1
  %sunkaddr96 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 320
  store double %.unpack2584, ptr addrspace(3) %sunkaddr96, align 8
  %sunkaddr97 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 328
  store double %.unpack2785, ptr addrspace(3) %sunkaddr97, align 8
  br label %45

45:                                               ; preds = %43, %2
  %46 = icmp samesign ult i32 %41, 286
  br i1 %46, label %47, label %49

47:                                               ; preds = %45
  %sunkaddr98 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 36864
  %48 = load <2 x double>, ptr addrspace(1) %sunkaddr98, align 16, !invariant.load !4
  %.unpack3082 = extractelement <2 x double> %48, i32 0
  %.unpack3283 = extractelement <2 x double> %48, i32 1
  %sunkaddr99 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 384
  store double %.unpack3082, ptr addrspace(3) %sunkaddr99, align 8
  %sunkaddr100 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 392
  store double %.unpack3283, ptr addrspace(3) %sunkaddr100, align 8
  br label %49

49:                                               ; preds = %47, %45
  %50 = icmp samesign ult i32 %41, 282
  br i1 %50, label %51, label %53

51:                                               ; preds = %49
  %sunkaddr101 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 43008
  %52 = load <2 x double>, ptr addrspace(1) %sunkaddr101, align 16, !invariant.load !4
  %.unpack3580 = extractelement <2 x double> %52, i32 0
  %.unpack3781 = extractelement <2 x double> %52, i32 1
  %sunkaddr102 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 448
  store double %.unpack3580, ptr addrspace(3) %sunkaddr102, align 8
  %sunkaddr103 = getelementptr inbounds i8, ptr addrspace(3) %27, i64 456
  store double %.unpack3781, ptr addrspace(3) %sunkaddr103, align 8
  br label %53

53:                                               ; preds = %51, %49
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %54 = or disjoint i32 %40, %16
  %55 = icmp samesign ult i32 %54, 310
  br i1 %55, label %56, label %99

56:                                               ; preds = %53
  %57 = mul nuw nsw i32 %14, 33
  %58 = add nuw nsw i32 %57, %16
  %59 = zext nneg i32 %58 to i64
  %60 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %59
  %.unpack40 = load double, ptr addrspace(3) %60, align 8
  %.elt41 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 8
  %.unpack42 = load double, ptr addrspace(3) %.elt41, align 8
  %61 = mul nuw nsw i32 %.decomposed, 9920
  %62 = mul nuw nsw i32 %14, 310
  %63 = or disjoint i32 %61, %16
  %64 = add nuw nsw i32 %63, %13
  %65 = add nuw nsw i32 %64, %62
  %66 = add nuw nsw i32 %65, %40
  %67 = zext nneg i32 %66 to i64
  %68 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %67
  %69 = insertelement <2 x double> poison, double %.unpack40, i32 0
  %70 = insertelement <2 x double> %69, double %.unpack42, i32 1
  store <2 x double> %70, ptr addrspace(1) %68, align 16
  %71 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 2112
  %.unpack45 = load double, ptr addrspace(3) %71, align 8
  %.elt46 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 2120
  %.unpack47 = load double, ptr addrspace(3) %.elt46, align 8
  %72 = getelementptr inbounds i8, ptr addrspace(1) %68, i64 19840
  %73 = insertelement <2 x double> poison, double %.unpack45, i32 0
  %74 = insertelement <2 x double> %73, double %.unpack47, i32 1
  store <2 x double> %74, ptr addrspace(1) %72, align 16
  %75 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 4224
  %.unpack50 = load double, ptr addrspace(3) %75, align 8
  %.elt51 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 4232
  %.unpack52 = load double, ptr addrspace(3) %.elt51, align 8
  %76 = getelementptr inbounds i8, ptr addrspace(1) %68, i64 39680
  %77 = insertelement <2 x double> poison, double %.unpack50, i32 0
  %78 = insertelement <2 x double> %77, double %.unpack52, i32 1
  store <2 x double> %78, ptr addrspace(1) %76, align 16
  %79 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 6336
  %.unpack55 = load double, ptr addrspace(3) %79, align 8
  %.elt56 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 6344
  %.unpack57 = load double, ptr addrspace(3) %.elt56, align 8
  %80 = getelementptr inbounds i8, ptr addrspace(1) %68, i64 59520
  %81 = insertelement <2 x double> poison, double %.unpack55, i32 0
  %82 = insertelement <2 x double> %81, double %.unpack57, i32 1
  store <2 x double> %82, ptr addrspace(1) %80, align 16
  %83 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 8448
  %.unpack60 = load double, ptr addrspace(3) %83, align 8
  %.elt61 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 8456
  %.unpack62 = load double, ptr addrspace(3) %.elt61, align 8
  %84 = getelementptr inbounds i8, ptr addrspace(1) %68, i64 79360
  %85 = insertelement <2 x double> poison, double %.unpack60, i32 0
  %86 = insertelement <2 x double> %85, double %.unpack62, i32 1
  store <2 x double> %86, ptr addrspace(1) %84, align 16
  %87 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 10560
  %.unpack65 = load double, ptr addrspace(3) %87, align 8
  %.elt66 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 10568
  %.unpack67 = load double, ptr addrspace(3) %.elt66, align 8
  %88 = getelementptr inbounds i8, ptr addrspace(1) %68, i64 99200
  %89 = insertelement <2 x double> poison, double %.unpack65, i32 0
  %90 = insertelement <2 x double> %89, double %.unpack67, i32 1
  store <2 x double> %90, ptr addrspace(1) %88, align 16
  %91 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 12672
  %.unpack70 = load double, ptr addrspace(3) %91, align 8
  %.elt71 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 12680
  %.unpack72 = load double, ptr addrspace(3) %.elt71, align 8
  %92 = getelementptr inbounds i8, ptr addrspace(1) %68, i64 119040
  %93 = insertelement <2 x double> poison, double %.unpack70, i32 0
  %94 = insertelement <2 x double> %93, double %.unpack72, i32 1
  store <2 x double> %94, ptr addrspace(1) %92, align 16
  %95 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 14784
  %.unpack75 = load double, ptr addrspace(3) %95, align 8
  %.elt76 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 14792
  %.unpack77 = load double, ptr addrspace(3) %.elt76, align 8
  %96 = getelementptr inbounds i8, ptr addrspace(1) %68, i64 138880
  %97 = insertelement <2 x double> poison, double %.unpack75, i32 0
  %98 = insertelement <2 x double> %97, double %.unpack77, i32 1
  store <2 x double> %98, ptr addrspace(1) %96, align 16
  br label %99

99:                                               ; preds = %56, %53
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
