; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_144_0 = local_unnamed_addr addrspace(1) constant [64 x i8] zeroinitializer, align 256

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define ptx_kernel void @loop_broadcast_fusion(ptr noalias writeonly align 256 captures(none) dereferenceable(66816) %0) local_unnamed_addr #0 {
  %2 = addrspacecast ptr %0 to ptr addrspace(1)
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %5 = shl nuw nsw i32 %3, 7
  %6 = or disjoint i32 %5, %4
  %7 = icmp samesign ult i32 %6, 4176
  %8 = zext nneg i32 %6 to i64
  %9 = getelementptr inbounds { double, double }, ptr addrspace(1) %2, i64 %8
  br i1 %7, label %10, label %11

10:                                               ; preds = %1
  store <2 x double> zeroinitializer, ptr addrspace(1) %9, align 16
  br label %11

11:                                               ; preds = %10, %1
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_compare_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(1) initializes((0, 1)) %1) local_unnamed_addr #2 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = load i64, ptr addrspace(1) %3, align 256, !invariant.load !4
  %6 = icmp slt i64 %5, 29
  %7 = zext i1 %6 to i8
  store i8 %7, ptr addrspace(1) %4, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_gather_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(294144) %0, ptr noalias readonly align 16 captures(none) dereferenceable(3801088) %1, ptr noalias readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(6291456) %3) local_unnamed_addr #3 {
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %10 = zext nneg i32 %9 to i64
  %11 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %12 = load i64, ptr addrspace(1) %5, align 256, !invariant.load !4
  %13 = tail call i64 @llvm.smax.i64(i64 %12, i64 0)
  %14 = tail call i64 @llvm.umin.i64(i64 %13, i64 28)
  %15 = shl nuw nsw i64 %10, 9
  %16 = and i64 %15, 32256
  %17 = shl nuw nsw i32 %11, 2
  %18 = zext nneg i32 %17 to i64
  %19 = getelementptr inbounds i32, ptr addrspace(1) %6, i64 %16
  %.idx = shl nuw nsw i64 %14, 17
  %20 = getelementptr inbounds i8, ptr addrspace(1) %19, i64 %.idx
  %21 = getelementptr inbounds i32, ptr addrspace(1) %20, i64 %18
  %22 = load <4 x i32>, ptr addrspace(1) %21, align 16, !invariant.load !4
  %23 = extractelement <4 x i32> %22, i64 0
  %24 = icmp slt i32 %23, 1532
  %25 = lshr i64 %10, 6
  %.idx4 = mul i64 %25, 24512
  %26 = getelementptr i8, ptr addrspace(1) %7, i64 %.idx4
  %27 = tail call i32 @llvm.smax.i32(i32 %23, i32 0)
  %28 = zext nneg i32 %27 to i64
  %29 = getelementptr inbounds { double, double }, ptr addrspace(1) %26, i64 %28
  br i1 %24, label %30, label %34

30:                                               ; preds = %4
  %31 = load <2 x double>, ptr addrspace(1) %29, align 16, !invariant.load !4
  %.unpack42 = extractelement <2 x double> %31, i32 0
  %.unpack643 = extractelement <2 x double> %31, i32 1
  %32 = insertvalue { double, double } poison, double %.unpack42, 0
  %33 = insertvalue { double, double } %32, double %.unpack643, 1
  br label %34

34:                                               ; preds = %30, %4
  %35 = phi { double, double } [ %33, %30 ], [ zeroinitializer, %4 ]
  %36 = getelementptr inbounds { double, double }, ptr addrspace(1) %8, i64 %15
  %37 = getelementptr inbounds { double, double }, ptr addrspace(1) %36, i64 %18
  %.elt = extractvalue { double, double } %35, 0
  %.elt8 = extractvalue { double, double } %35, 1
  %38 = insertelement <2 x double> poison, double %.elt, i32 0
  %39 = insertelement <2 x double> %38, double %.elt8, i32 1
  store <2 x double> %39, ptr addrspace(1) %37, align 64
  %40 = extractelement <4 x i32> %22, i64 1
  %41 = icmp slt i32 %40, 1532
  %42 = tail call i32 @llvm.smax.i32(i32 %40, i32 0)
  %43 = zext nneg i32 %42 to i64
  %44 = getelementptr inbounds { double, double }, ptr addrspace(1) %26, i64 %43
  br i1 %41, label %45, label %49

45:                                               ; preds = %34
  %46 = load <2 x double>, ptr addrspace(1) %44, align 16, !invariant.load !4
  %.unpack1140 = extractelement <2 x double> %46, i32 0
  %.unpack1341 = extractelement <2 x double> %46, i32 1
  %47 = insertvalue { double, double } poison, double %.unpack1140, 0
  %48 = insertvalue { double, double } %47, double %.unpack1341, 1
  br label %49

49:                                               ; preds = %45, %34
  %50 = phi { double, double } [ %48, %45 ], [ zeroinitializer, %34 ]
  %51 = getelementptr inbounds i8, ptr addrspace(1) %37, i64 16
  %.elt14 = extractvalue { double, double } %50, 0
  %.elt16 = extractvalue { double, double } %50, 1
  %52 = insertelement <2 x double> poison, double %.elt14, i32 0
  %53 = insertelement <2 x double> %52, double %.elt16, i32 1
  store <2 x double> %53, ptr addrspace(1) %51, align 16
  %54 = extractelement <4 x i32> %22, i64 2
  %55 = icmp slt i32 %54, 1532
  %56 = tail call i32 @llvm.smax.i32(i32 %54, i32 0)
  %57 = zext nneg i32 %56 to i64
  %58 = getelementptr inbounds { double, double }, ptr addrspace(1) %26, i64 %57
  br i1 %55, label %59, label %63

59:                                               ; preds = %49
  %60 = load <2 x double>, ptr addrspace(1) %58, align 16, !invariant.load !4
  %.unpack1938 = extractelement <2 x double> %60, i32 0
  %.unpack2139 = extractelement <2 x double> %60, i32 1
  %61 = insertvalue { double, double } poison, double %.unpack1938, 0
  %62 = insertvalue { double, double } %61, double %.unpack2139, 1
  br label %63

63:                                               ; preds = %59, %49
  %64 = phi { double, double } [ %62, %59 ], [ zeroinitializer, %49 ]
  %65 = getelementptr inbounds i8, ptr addrspace(1) %37, i64 32
  %.elt22 = extractvalue { double, double } %64, 0
  %.elt24 = extractvalue { double, double } %64, 1
  %66 = insertelement <2 x double> poison, double %.elt22, i32 0
  %67 = insertelement <2 x double> %66, double %.elt24, i32 1
  store <2 x double> %67, ptr addrspace(1) %65, align 32
  %68 = extractelement <4 x i32> %22, i64 3
  %69 = icmp slt i32 %68, 1532
  %70 = tail call i32 @llvm.smax.i32(i32 %68, i32 0)
  %71 = zext nneg i32 %70 to i64
  %72 = getelementptr inbounds { double, double }, ptr addrspace(1) %26, i64 %71
  br i1 %69, label %73, label %77

73:                                               ; preds = %63
  %74 = load <2 x double>, ptr addrspace(1) %72, align 16, !invariant.load !4
  %.unpack2736 = extractelement <2 x double> %74, i32 0
  %.unpack2937 = extractelement <2 x double> %74, i32 1
  %75 = insertvalue { double, double } poison, double %.unpack2736, 0
  %76 = insertvalue { double, double } %75, double %.unpack2937, 1
  br label %77

77:                                               ; preds = %73, %63
  %78 = phi { double, double } [ %76, %73 ], [ zeroinitializer, %63 ]
  %79 = getelementptr inbounds i8, ptr addrspace(1) %37, i64 48
  %.elt30 = extractvalue { double, double } %78, 0
  %.elt32 = extractvalue { double, double } %78, 1
  %80 = insertelement <2 x double> poison, double %.elt30, i32 0
  %81 = insertelement <2 x double> %80, double %.elt32, i32 1
  store <2 x double> %81, ptr addrspace(1) %79, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 captures(none) dereferenceable(6291456) %0, ptr noalias readonly align 16 captures(none) dereferenceable(524288) %1, ptr noalias readnone align 256 captures(none) dereferenceable(6291456) %2) local_unnamed_addr #3 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %8 = shl nuw nsw i32 %7, 2
  %9 = shl nuw nsw i32 %6, 9
  %10 = or disjoint i32 %8, %9
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %11
  %13 = load <2 x double>, ptr addrspace(1) %12, align 64
  %.unpack32 = extractelement <2 x double> %13, i32 0
  %.unpack233 = extractelement <2 x double> %13, i32 1
  %14 = and i32 %9, 32256
  %15 = or disjoint i32 %14, %8
  %16 = zext nneg i32 %15 to i64
  %17 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %16
  %18 = load <2 x double>, ptr addrspace(1) %17, align 16, !invariant.load !4
  %.unpack340 = extractelement <2 x double> %18, i32 0
  %.unpack541 = extractelement <2 x double> %18, i32 1
  %19 = fmul double %.unpack340, 0x40A00C3EA4553987
  %20 = fmul double %.unpack541, 0.000000e+00
  %21 = fsub double %19, %20
  %22 = fmul double %.unpack541, 0x40A00C3EA4553987
  %23 = fmul double %.unpack340, 0.000000e+00
  %24 = fadd double %23, %22
  %25 = fmul double %.unpack32, %21
  %26 = fmul double %.unpack233, %24
  %27 = fsub double %25, %26
  %28 = fmul double %.unpack233, %21
  %29 = fmul double %.unpack32, %24
  %30 = fadd double %28, %29
  %31 = insertelement <2 x double> poison, double %27, i32 0
  %32 = insertelement <2 x double> %31, double %30, i32 1
  store <2 x double> %32, ptr addrspace(1) %12, align 64
  %33 = getelementptr inbounds i8, ptr addrspace(1) %12, i64 16
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16
  %.unpack834 = extractelement <2 x double> %34, i32 0
  %.unpack1035 = extractelement <2 x double> %34, i32 1
  %35 = getelementptr inbounds i8, ptr addrspace(1) %17, i64 16
  %36 = load <2 x double>, ptr addrspace(1) %35, align 16, !invariant.load !4
  %.unpack1142 = extractelement <2 x double> %36, i32 0
  %.unpack1343 = extractelement <2 x double> %36, i32 1
  %37 = fmul double %.unpack1142, 0x40A00C3EA4553987
  %38 = fmul double %.unpack1343, 0.000000e+00
  %39 = fsub double %37, %38
  %40 = fmul double %.unpack1343, 0x40A00C3EA4553987
  %41 = fmul double %.unpack1142, 0.000000e+00
  %42 = fadd double %41, %40
  %43 = fmul double %.unpack834, %39
  %44 = fmul double %.unpack1035, %42
  %45 = fsub double %43, %44
  %46 = fmul double %.unpack1035, %39
  %47 = fmul double %.unpack834, %42
  %48 = fadd double %46, %47
  %49 = insertelement <2 x double> poison, double %45, i32 0
  %50 = insertelement <2 x double> %49, double %48, i32 1
  store <2 x double> %50, ptr addrspace(1) %33, align 16
  %51 = getelementptr inbounds i8, ptr addrspace(1) %12, i64 32
  %52 = load <2 x double>, ptr addrspace(1) %51, align 32
  %.unpack1636 = extractelement <2 x double> %52, i32 0
  %.unpack1837 = extractelement <2 x double> %52, i32 1
  %53 = getelementptr inbounds i8, ptr addrspace(1) %17, i64 32
  %54 = load <2 x double>, ptr addrspace(1) %53, align 16, !invariant.load !4
  %.unpack1944 = extractelement <2 x double> %54, i32 0
  %.unpack2145 = extractelement <2 x double> %54, i32 1
  %55 = fmul double %.unpack1944, 0x40A00C3EA4553987
  %56 = fmul double %.unpack2145, 0.000000e+00
  %57 = fsub double %55, %56
  %58 = fmul double %.unpack2145, 0x40A00C3EA4553987
  %59 = fmul double %.unpack1944, 0.000000e+00
  %60 = fadd double %59, %58
  %61 = fmul double %.unpack1636, %57
  %62 = fmul double %.unpack1837, %60
  %63 = fsub double %61, %62
  %64 = fmul double %.unpack1837, %57
  %65 = fmul double %.unpack1636, %60
  %66 = fadd double %64, %65
  %67 = insertelement <2 x double> poison, double %63, i32 0
  %68 = insertelement <2 x double> %67, double %66, i32 1
  store <2 x double> %68, ptr addrspace(1) %51, align 32
  %69 = getelementptr inbounds i8, ptr addrspace(1) %12, i64 48
  %70 = load <2 x double>, ptr addrspace(1) %69, align 16
  %.unpack2438 = extractelement <2 x double> %70, i32 0
  %.unpack2639 = extractelement <2 x double> %70, i32 1
  %71 = getelementptr inbounds i8, ptr addrspace(1) %17, i64 48
  %72 = load <2 x double>, ptr addrspace(1) %71, align 16, !invariant.load !4
  %.unpack2746 = extractelement <2 x double> %72, i32 0
  %.unpack2947 = extractelement <2 x double> %72, i32 1
  %73 = fmul double %.unpack2746, 0x40A00C3EA4553987
  %74 = fmul double %.unpack2947, 0.000000e+00
  %75 = fsub double %73, %74
  %76 = fmul double %.unpack2947, 0x40A00C3EA4553987
  %77 = fmul double %.unpack2746, 0.000000e+00
  %78 = fadd double %77, %76
  %79 = fmul double %.unpack2438, %75
  %80 = fmul double %.unpack2639, %78
  %81 = fsub double %79, %80
  %82 = fmul double %.unpack2639, %75
  %83 = fmul double %.unpack2438, %78
  %84 = fadd double %82, %83
  %85 = insertelement <2 x double> poison, double %81, i32 0
  %86 = insertelement <2 x double> %85, double %84, i32 1
  store <2 x double> %86, ptr addrspace(1) %69, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_gather_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(6291456) %0, ptr noalias readonly align 16 captures(none) dereferenceable(533136) %1, ptr noalias readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(294144) %3) local_unnamed_addr #3 {
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %10 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %11 = shl nuw nsw i32 %9, 7
  %12 = or disjoint i32 %11, %10
  %13 = icmp samesign ult i32 %12, 18384
  br i1 %13, label %14, label %61

14:                                               ; preds = %4
  %15 = zext nneg i32 %12 to i64
  %.lhs.trunc = trunc nuw nsw i32 %12 to i16
  %16 = udiv i16 %.lhs.trunc, 1532
  %17 = mul i16 %16, 1532
  %.decomposed = sub i16 %.lhs.trunc, %17
  %.val6 = load i64, ptr addrspace(1) %5, align 256, !invariant.load !4
  %18 = tail call i64 @llvm.smax.i64(i64 %.val6, i64 0)
  %19 = tail call i64 @llvm.umin.i64(i64 %18, i64 28)
  %.idx.i = mul nuw nsw i64 %19, 18384
  %20 = getelementptr inbounds i8, ptr addrspace(1) %6, i64 %.idx.i
  %narrow = mul nuw nsw i16 %.decomposed, 12
  %.idx1.i = zext nneg i16 %narrow to i64
  %21 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 %.idx1.i
  %22 = load i32, ptr addrspace(1) %21, align 4, !invariant.load !4, !alias.scope !7
  %23 = lshr i32 %22, 26
  %24 = and i32 %23, 32
  %25 = add i32 %24, %22
  %26 = tail call i32 @llvm.smax.i32(i32 %25, i32 0)
  %27 = tail call i32 @llvm.umin.i32(i32 %26, i32 31)
  %28 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 4
  %29 = load i32, ptr addrspace(1) %28, align 4, !invariant.load !4, !alias.scope !10
  %30 = lshr i32 %29, 26
  %31 = and i32 %30, 32
  %32 = add i32 %31, %29
  %33 = tail call i32 @llvm.smax.i32(i32 %32, i32 0)
  %34 = tail call i32 @llvm.umin.i32(i32 %33, i32 31)
  %35 = getelementptr inbounds i8, ptr addrspace(1) %21, i64 8
  %36 = load i32, ptr addrspace(1) %35, align 4, !invariant.load !4, !alias.scope !13
  %37 = lshr i32 %36, 26
  %38 = and i32 %37, 32
  %39 = add i32 %38, %36
  %40 = tail call i32 @llvm.smax.i32(i32 %39, i32 0)
  %41 = tail call i32 @llvm.umin.i32(i32 %40, i32 31)
  %42 = zext nneg i32 %41 to i64
  %.zext12 = zext nneg i16 %16 to i64
  %43 = shl nuw nsw i32 %27, 10
  %44 = zext nneg i32 %43 to i64
  %45 = shl nuw nsw i32 %34, 5
  %46 = zext nneg i32 %45 to i64
  %.idx = shl nuw nsw i64 %.zext12, 19
  %47 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 %.idx
  %48 = getelementptr inbounds { double, double }, ptr addrspace(1) %47, i64 %44
  %49 = getelementptr inbounds { double, double }, ptr addrspace(1) %48, i64 %46
  %50 = getelementptr inbounds { double, double }, ptr addrspace(1) %49, i64 %42
  %51 = load <2 x double>, ptr addrspace(1) %50, align 16, !invariant.load !4
  %.unpack13 = extractelement <2 x double> %51, i32 0
  %.unpack214 = extractelement <2 x double> %51, i32 1
  %52 = fmul double %.unpack13, 0x3F7FCF3D6F094292
  %53 = fmul double %.unpack214, 0.000000e+00
  %54 = fsub double %52, %53
  %55 = fmul double %.unpack214, 0x3F7FCF3D6F094292
  %56 = fmul double %.unpack13, 0.000000e+00
  %57 = fadd double %56, %55
  %58 = getelementptr inbounds { double, double }, ptr addrspace(1) %8, i64 %15
  %59 = insertelement <2 x double> poison, double %54, i32 0
  %60 = insertelement <2 x double> %59, double %57, i32 1
  store <2 x double> %60, ptr addrspace(1) %58, align 16
  br label %61

61:                                               ; preds = %14, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_complex_multiply_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(588288) %0, ptr noalias readonly align 16 captures(none) dereferenceable(355424) %1, ptr noalias readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias readonly align 256 captures(none) dereferenceable(588288) %3, ptr noalias writeonly align 256 captures(none) dereferenceable(588288) %4, ptr noalias writeonly align 256 captures(none) dereferenceable(588288) %5) local_unnamed_addr #3 {
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = addrspacecast ptr %2 to ptr addrspace(1)
  %9 = addrspacecast ptr %1 to ptr addrspace(1)
  %10 = addrspacecast ptr %3 to ptr addrspace(1)
  %11 = addrspacecast ptr %4 to ptr addrspace(1)
  %12 = addrspacecast ptr %5 to ptr addrspace(1)
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !16
  %14 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %15 = shl nuw nsw i32 %13, 7
  %16 = or disjoint i32 %15, %14
  %17 = icmp samesign ult i32 %16, 36768
  br i1 %17, label %18, label %50

18:                                               ; preds = %6
  %19 = zext nneg i32 %16 to i64
  %20 = getelementptr inbounds { double, double }, ptr addrspace(1) %7, i64 %19
  %21 = load <2 x double>, ptr addrspace(1) %20, align 16, !invariant.load !4
  %.unpack10 = extractelement <2 x double> %21, i32 0
  %.unpack211 = extractelement <2 x double> %21, i32 1
  %22 = load i64, ptr addrspace(1) %8, align 256, !invariant.load !4
  %23 = tail call i64 @llvm.smax.i64(i64 %22, i64 0)
  %24 = tail call i64 @llvm.umin.i64(i64 %23, i64 28)
  %.lhs.trunc = trunc nuw i32 %16 to i16
  %25 = urem i16 %.lhs.trunc, 1532
  %.zext = zext nneg i16 %25 to i64
  %26 = getelementptr inbounds double, ptr addrspace(1) %9, i64 %.zext
  %.idx = mul nuw nsw i64 %24, 12256
  %27 = getelementptr inbounds i8, ptr addrspace(1) %26, i64 %.idx
  %28 = load double, ptr addrspace(1) %27, align 8, !invariant.load !4
  %29 = fmul double %.unpack10, %28
  %30 = fmul double %.unpack211, 0.000000e+00
  %31 = fsub double %29, %30
  %32 = fmul double %.unpack211, %28
  %33 = fmul double %.unpack10, 0.000000e+00
  %34 = fadd double %33, %32
  %35 = getelementptr inbounds { double, double }, ptr addrspace(1) %10, i64 %19
  %36 = load <2 x double>, ptr addrspace(1) %35, align 16, !invariant.load !4
  %.unpack312 = extractelement <2 x double> %36, i32 0
  %.unpack513 = extractelement <2 x double> %36, i32 1
  %37 = fmul double %28, %.unpack312
  %38 = fmul double %.unpack513, 0.000000e+00
  %39 = fsub double %37, %38
  %40 = fmul double %28, %.unpack513
  %41 = fmul double %.unpack312, 0.000000e+00
  %42 = fadd double %41, %40
  %43 = fneg double %42
  %44 = getelementptr inbounds { double, double }, ptr addrspace(1) %11, i64 %19
  %45 = insertelement <2 x double> poison, double %31, i32 0
  %46 = insertelement <2 x double> %45, double %34, i32 1
  store <2 x double> %46, ptr addrspace(1) %44, align 16
  %47 = getelementptr inbounds { double, double }, ptr addrspace(1) %12, i64 %19
  %48 = insertelement <2 x double> poison, double %39, i32 0
  %49 = insertelement <2 x double> %48, double %43, i32 1
  store <2 x double> %49, ptr addrspace(1) %47, align 16
  br label %50

50:                                               ; preds = %18, %6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_add_fusion(ptr noalias align 256 captures(none) dereferenceable(8) %0, ptr noalias readnone align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #2 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = load i64, ptr addrspace(1) %3, align 256
  %5 = add i64 %4, 1
  store i64 %5, ptr addrspace(1) %3, align 256
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #5

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #4 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 33}
!3 = !{i32 0, i32 128}
!4 = !{}
!5 = !{i32 0, i32 768}
!6 = !{i32 0, i32 144}
!7 = !{!8}
!8 = distinct !{!8, !9, !"fused_gather_1_bitcast_7_9: argument 0"}
!9 = distinct !{!9, !"fused_gather_1_bitcast_7_9"}
!10 = !{!11}
!11 = distinct !{!11, !12, !"fused_gather_1_bitcast_7_9: argument 0"}
!12 = distinct !{!12, !"fused_gather_1_bitcast_7_9"}
!13 = !{!14}
!14 = distinct !{!14, !15, !"fused_gather_1_bitcast_7_9: argument 0"}
!15 = distinct !{!15, !"fused_gather_1_bitcast_7_9"}
!16 = !{i32 0, i32 288}
