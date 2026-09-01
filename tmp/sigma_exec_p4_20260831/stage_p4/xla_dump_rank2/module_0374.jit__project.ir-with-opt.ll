; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(121896960) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(121896960) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %7 = and i32 %5, 31
  %8 = icmp samesign ult i32 %7, 24
  br i1 %8, label %9, label %._crit_edge

._crit_edge:                                      ; preds = %2
  %.pre = lshr i32 %5, 5
  %.pre91 = shl nuw nsw i32 %6, 5
  br label %159

9:                                                ; preds = %2
  %10 = shl nuw nsw i32 %6, 5
  %11 = lshr i32 %5, 5
  %12 = or disjoint i32 %10, %11
  %13 = urem i32 %12, 310
  %14 = mul nuw nsw i32 %13, 24
  %15 = shl nuw nsw i32 %6, 3
  %16 = udiv i32 %15, 155
  %17 = mul i32 %16, 155
  %.decomposed = sub i32 %15, %17
  %18 = shl nuw nsw i32 %.decomposed, 7
  %19 = or disjoint i32 %18, %5
  %.lhs.trunc = trunc nuw nsw i32 %19 to i16
  %20 = udiv i16 %.lhs.trunc, 9920
  %narrow = mul nuw nsw i16 %20, 7440
  %21 = zext nneg i16 %narrow to i32
  %22 = mul nuw nsw i32 %16, 14880
  %23 = or disjoint i32 %22, %7
  %24 = add nuw nsw i32 %23, %14
  %25 = add nuw nsw i32 %24, %21
  %26 = zext nneg i32 %25 to i64
  %27 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %26
  %28 = load <2 x double>, ptr addrspace(1) %27, align 16, !invariant.load !4
  %.unpack114 = extractelement <2 x double> %28, i32 0
  %.unpack2115 = extractelement <2 x double> %28, i32 1
  %29 = mul nuw nsw i32 %7, 33
  %30 = add nuw nsw i32 %29, %11
  %31 = zext nneg i32 %30 to i64
  %32 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %31
  store double %.unpack114, ptr addrspace(3) %32, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 8
  store double %.unpack2115, ptr addrspace(3) %.repack3, align 8
  %33 = or disjoint i32 %12, 4
  %34 = urem i32 %33, 310
  %35 = mul nuw nsw i32 %34, 24
  %36 = or disjoint i32 %15, 1
  %37 = udiv i32 %36, 155
  %38 = mul i32 %37, 155
  %.decomposed93 = sub i32 %36, %38
  %39 = shl nuw nsw i32 %.decomposed93, 7
  %40 = or disjoint i32 %39, %5
  %.lhs.trunc70 = trunc nuw nsw i32 %40 to i16
  %41 = udiv i16 %.lhs.trunc70, 9920
  %narrow84 = mul nuw nsw i16 %41, 7440
  %42 = zext nneg i16 %narrow84 to i32
  %43 = mul nuw nsw i32 %37, 14880
  %44 = or disjoint i32 %43, %7
  %45 = add nuw nsw i32 %44, %35
  %46 = add nuw nsw i32 %45, %42
  %47 = zext nneg i32 %46 to i64
  %48 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %47
  %49 = load <2 x double>, ptr addrspace(1) %48, align 16, !invariant.load !4
  %.unpack5112 = extractelement <2 x double> %49, i32 0
  %.unpack7113 = extractelement <2 x double> %49, i32 1
  %50 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 64
  store double %.unpack5112, ptr addrspace(3) %50, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 72
  store double %.unpack7113, ptr addrspace(3) %.repack8, align 8
  %51 = or disjoint i32 %12, 8
  %52 = urem i32 %51, 310
  %53 = mul nuw nsw i32 %52, 24
  %54 = or disjoint i32 %15, 2
  %55 = udiv i32 %54, 155
  %56 = mul i32 %55, 155
  %.decomposed94 = sub i32 %54, %56
  %57 = shl nuw nsw i32 %.decomposed94, 7
  %58 = or disjoint i32 %57, %5
  %.lhs.trunc72 = trunc nuw nsw i32 %58 to i16
  %59 = udiv i16 %.lhs.trunc72, 9920
  %narrow85 = mul nuw nsw i16 %59, 7440
  %60 = zext nneg i16 %narrow85 to i32
  %61 = mul nuw nsw i32 %55, 14880
  %62 = or disjoint i32 %61, %7
  %63 = add nuw nsw i32 %62, %53
  %64 = add nuw nsw i32 %63, %60
  %65 = zext nneg i32 %64 to i64
  %66 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %65
  %67 = load <2 x double>, ptr addrspace(1) %66, align 16, !invariant.load !4
  %.unpack10110 = extractelement <2 x double> %67, i32 0
  %.unpack12111 = extractelement <2 x double> %67, i32 1
  %68 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 128
  store double %.unpack10110, ptr addrspace(3) %68, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 136
  store double %.unpack12111, ptr addrspace(3) %.repack13, align 8
  %69 = or disjoint i32 %12, 12
  %70 = urem i32 %69, 310
  %71 = mul nuw nsw i32 %70, 24
  %72 = or disjoint i32 %15, 3
  %73 = udiv i32 %72, 155
  %74 = mul i32 %73, 155
  %.decomposed95 = sub i32 %72, %74
  %75 = shl nuw nsw i32 %.decomposed95, 7
  %76 = or disjoint i32 %75, %5
  %.lhs.trunc74 = trunc nuw nsw i32 %76 to i16
  %77 = udiv i16 %.lhs.trunc74, 9920
  %narrow86 = mul nuw nsw i16 %77, 7440
  %78 = zext nneg i16 %narrow86 to i32
  %79 = mul nuw nsw i32 %73, 14880
  %80 = or disjoint i32 %79, %7
  %81 = add nuw nsw i32 %80, %71
  %82 = add nuw nsw i32 %81, %78
  %83 = zext nneg i32 %82 to i64
  %84 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %83
  %85 = load <2 x double>, ptr addrspace(1) %84, align 16, !invariant.load !4
  %.unpack15108 = extractelement <2 x double> %85, i32 0
  %.unpack17109 = extractelement <2 x double> %85, i32 1
  %86 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 192
  store double %.unpack15108, ptr addrspace(3) %86, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 200
  store double %.unpack17109, ptr addrspace(3) %.repack18, align 8
  %87 = or disjoint i32 %12, 16
  %88 = urem i32 %87, 310
  %89 = mul nuw nsw i32 %88, 24
  %90 = or disjoint i32 %15, 4
  %91 = udiv i32 %90, 155
  %92 = mul i32 %91, 155
  %.decomposed96 = sub i32 %90, %92
  %93 = shl nuw nsw i32 %.decomposed96, 7
  %94 = or disjoint i32 %93, %5
  %.lhs.trunc76 = trunc nuw nsw i32 %94 to i16
  %95 = udiv i16 %.lhs.trunc76, 9920
  %narrow87 = mul nuw nsw i16 %95, 7440
  %96 = zext nneg i16 %narrow87 to i32
  %97 = mul nuw nsw i32 %91, 14880
  %98 = or disjoint i32 %97, %7
  %99 = add nuw nsw i32 %98, %89
  %100 = add nuw nsw i32 %99, %96
  %101 = zext nneg i32 %100 to i64
  %102 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %101
  %103 = load <2 x double>, ptr addrspace(1) %102, align 16, !invariant.load !4
  %.unpack20106 = extractelement <2 x double> %103, i32 0
  %.unpack22107 = extractelement <2 x double> %103, i32 1
  %104 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 256
  store double %.unpack20106, ptr addrspace(3) %104, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 264
  store double %.unpack22107, ptr addrspace(3) %.repack23, align 8
  %105 = or disjoint i32 %12, 20
  %106 = urem i32 %105, 310
  %107 = mul nuw nsw i32 %106, 24
  %108 = or disjoint i32 %15, 5
  %109 = udiv i32 %108, 155
  %110 = mul i32 %109, 155
  %.decomposed97 = sub i32 %108, %110
  %111 = shl nuw nsw i32 %.decomposed97, 7
  %112 = or disjoint i32 %111, %5
  %.lhs.trunc78 = trunc nuw nsw i32 %112 to i16
  %113 = udiv i16 %.lhs.trunc78, 9920
  %narrow88 = mul nuw nsw i16 %113, 7440
  %114 = zext nneg i16 %narrow88 to i32
  %115 = mul nuw nsw i32 %109, 14880
  %116 = or disjoint i32 %115, %7
  %117 = add nuw nsw i32 %116, %107
  %118 = add nuw nsw i32 %117, %114
  %119 = zext nneg i32 %118 to i64
  %120 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %119
  %121 = load <2 x double>, ptr addrspace(1) %120, align 16, !invariant.load !4
  %.unpack25104 = extractelement <2 x double> %121, i32 0
  %.unpack27105 = extractelement <2 x double> %121, i32 1
  %122 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 320
  store double %.unpack25104, ptr addrspace(3) %122, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 328
  store double %.unpack27105, ptr addrspace(3) %.repack28, align 8
  %123 = or disjoint i32 %12, 24
  %124 = urem i32 %123, 310
  %125 = mul nuw nsw i32 %124, 24
  %126 = or disjoint i32 %15, 6
  %127 = udiv i32 %126, 155
  %128 = mul i32 %127, 155
  %.decomposed98 = sub i32 %126, %128
  %129 = shl nuw nsw i32 %.decomposed98, 7
  %130 = or disjoint i32 %129, %5
  %.lhs.trunc80 = trunc nuw nsw i32 %130 to i16
  %131 = udiv i16 %.lhs.trunc80, 9920
  %narrow89 = mul nuw nsw i16 %131, 7440
  %132 = zext nneg i16 %narrow89 to i32
  %133 = mul nuw nsw i32 %127, 14880
  %134 = or disjoint i32 %133, %7
  %135 = add nuw nsw i32 %134, %125
  %136 = add nuw nsw i32 %135, %132
  %137 = zext nneg i32 %136 to i64
  %138 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %137
  %139 = load <2 x double>, ptr addrspace(1) %138, align 16, !invariant.load !4
  %.unpack30102 = extractelement <2 x double> %139, i32 0
  %.unpack32103 = extractelement <2 x double> %139, i32 1
  %140 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 384
  store double %.unpack30102, ptr addrspace(3) %140, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 392
  store double %.unpack32103, ptr addrspace(3) %.repack33, align 8
  %141 = or disjoint i32 %12, 28
  %142 = urem i32 %141, 310
  %143 = mul nuw nsw i32 %142, 24
  %144 = or disjoint i32 %15, 7
  %145 = udiv i32 %144, 155
  %146 = mul i32 %145, 155
  %.decomposed99 = sub i32 %144, %146
  %147 = shl nuw nsw i32 %.decomposed99, 7
  %148 = or disjoint i32 %147, %5
  %.lhs.trunc82 = trunc nuw nsw i32 %148 to i16
  %149 = udiv i16 %.lhs.trunc82, 9920
  %narrow90 = mul nuw nsw i16 %149, 7440
  %150 = zext nneg i16 %narrow90 to i32
  %151 = mul nuw nsw i32 %145, 14880
  %152 = or disjoint i32 %151, %7
  %153 = add nuw nsw i32 %152, %143
  %154 = add nuw nsw i32 %153, %150
  %155 = zext nneg i32 %154 to i64
  %156 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %155
  %157 = load <2 x double>, ptr addrspace(1) %156, align 16, !invariant.load !4
  %.unpack35100 = extractelement <2 x double> %157, i32 0
  %.unpack37101 = extractelement <2 x double> %157, i32 1
  %158 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 448
  store double %.unpack35100, ptr addrspace(3) %158, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %32, i64 456
  store double %.unpack37101, ptr addrspace(3) %.repack38, align 8
  br label %159

159:                                              ; preds = %._crit_edge, %9
  %.pre-phi92 = phi i32 [ %.pre91, %._crit_edge ], [ %10, %9 ]
  %.pre-phi = phi i32 [ %.pre, %._crit_edge ], [ %11, %9 ]
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %160 = mul nuw nsw i32 %.pre-phi, 33
  %161 = add nuw nsw i32 %160, %7
  %162 = zext nneg i32 %161 to i64
  %163 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %162
  %.unpack40 = load double, ptr addrspace(3) %163, align 8
  %.elt41 = getelementptr inbounds i8, ptr addrspace(3) %163, i64 8
  %.unpack42 = load double, ptr addrspace(3) %.elt41, align 8
  %164 = mul nuw nsw i32 %.pre-phi, 317440
  %165 = add nuw nsw i32 %164, %.pre-phi92
  %166 = or disjoint i32 %165, %7
  %167 = zext nneg i32 %166 to i64
  %168 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %167
  %169 = insertelement <2 x double> poison, double %.unpack40, i32 0
  %170 = insertelement <2 x double> %169, double %.unpack42, i32 1
  store <2 x double> %170, ptr addrspace(1) %168, align 16
  %171 = getelementptr inbounds i8, ptr addrspace(3) %163, i64 2112
  %.unpack45 = load double, ptr addrspace(3) %171, align 8
  %.elt46 = getelementptr inbounds i8, ptr addrspace(3) %163, i64 2120
  %.unpack47 = load double, ptr addrspace(3) %.elt46, align 8
  %172 = getelementptr inbounds i8, ptr addrspace(1) %168, i64 20316160
  %173 = insertelement <2 x double> poison, double %.unpack45, i32 0
  %174 = insertelement <2 x double> %173, double %.unpack47, i32 1
  store <2 x double> %174, ptr addrspace(1) %172, align 16
  %175 = getelementptr inbounds i8, ptr addrspace(3) %163, i64 4224
  %.unpack50 = load double, ptr addrspace(3) %175, align 8
  %.elt51 = getelementptr inbounds i8, ptr addrspace(3) %163, i64 4232
  %.unpack52 = load double, ptr addrspace(3) %.elt51, align 8
  %176 = getelementptr inbounds i8, ptr addrspace(1) %168, i64 40632320
  %177 = insertelement <2 x double> poison, double %.unpack50, i32 0
  %178 = insertelement <2 x double> %177, double %.unpack52, i32 1
  store <2 x double> %178, ptr addrspace(1) %176, align 16
  %179 = getelementptr inbounds i8, ptr addrspace(3) %163, i64 6336
  %.unpack55 = load double, ptr addrspace(3) %179, align 8
  %.elt56 = getelementptr inbounds i8, ptr addrspace(3) %163, i64 6344
  %.unpack57 = load double, ptr addrspace(3) %.elt56, align 8
  %180 = getelementptr inbounds i8, ptr addrspace(1) %168, i64 60948480
  %181 = insertelement <2 x double> poison, double %.unpack55, i32 0
  %182 = insertelement <2 x double> %181, double %.unpack57, i32 1
  store <2 x double> %182, ptr addrspace(1) %180, align 16
  %183 = getelementptr inbounds i8, ptr addrspace(3) %163, i64 8448
  %.unpack60 = load double, ptr addrspace(3) %183, align 8
  %.elt61 = getelementptr inbounds i8, ptr addrspace(3) %163, i64 8456
  %.unpack62 = load double, ptr addrspace(3) %.elt61, align 8
  %184 = getelementptr inbounds i8, ptr addrspace(1) %168, i64 81264640
  %185 = insertelement <2 x double> poison, double %.unpack60, i32 0
  %186 = insertelement <2 x double> %185, double %.unpack62, i32 1
  store <2 x double> %186, ptr addrspace(1) %184, align 16
  %187 = getelementptr inbounds i8, ptr addrspace(3) %163, i64 10560
  %.unpack65 = load double, ptr addrspace(3) %187, align 8
  %.elt66 = getelementptr inbounds i8, ptr addrspace(3) %163, i64 10568
  %.unpack67 = load double, ptr addrspace(3) %.elt66, align 8
  %188 = getelementptr inbounds i8, ptr addrspace(1) %168, i64 101580800
  %189 = insertelement <2 x double> poison, double %.unpack65, i32 0
  %190 = insertelement <2 x double> %189, double %.unpack67, i32 1
  store <2 x double> %190, ptr addrspace(1) %188, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_complex_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(121896960) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(121896960) %1) local_unnamed_addr #3 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = shl nuw nsw i32 %6, 2
  %8 = shl nuw nsw i32 %5, 9
  %9 = or disjoint i32 %7, %8
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %10
  %12 = load <2 x double>, ptr addrspace(1) %11, align 16, !invariant.load !4
  %.unpack26 = extractelement <2 x double> %12, i32 0
  %.unpack227 = extractelement <2 x double> %12, i32 1
  %13 = fneg double %.unpack227
  %14 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %10
  %15 = insertelement <2 x double> poison, double %.unpack26, i32 0
  %16 = insertelement <2 x double> %15, double %13, i32 1
  store <2 x double> %16, ptr addrspace(1) %14, align 64
  %17 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 16
  %18 = load <2 x double>, ptr addrspace(1) %17, align 16, !invariant.load !4
  %.unpack528 = extractelement <2 x double> %18, i32 0
  %.unpack729 = extractelement <2 x double> %18, i32 1
  %19 = fneg double %.unpack729
  %20 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 16
  %21 = insertelement <2 x double> poison, double %.unpack528, i32 0
  %22 = insertelement <2 x double> %21, double %19, i32 1
  store <2 x double> %22, ptr addrspace(1) %20, align 16
  %23 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 32
  %24 = load <2 x double>, ptr addrspace(1) %23, align 16, !invariant.load !4
  %.unpack1030 = extractelement <2 x double> %24, i32 0
  %.unpack1231 = extractelement <2 x double> %24, i32 1
  %25 = fneg double %.unpack1231
  %26 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 32
  %27 = insertelement <2 x double> poison, double %.unpack1030, i32 0
  %28 = insertelement <2 x double> %27, double %25, i32 1
  store <2 x double> %28, ptr addrspace(1) %26, align 32
  %29 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 48
  %30 = load <2 x double>, ptr addrspace(1) %29, align 16, !invariant.load !4
  %.unpack1532 = extractelement <2 x double> %30, i32 0
  %.unpack1733 = extractelement <2 x double> %30, i32 1
  %31 = fneg double %.unpack1733
  %32 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 48
  %33 = insertelement <2 x double> poison, double %.unpack1532, i32 0
  %34 = insertelement <2 x double> %33, double %31, i32 1
  store <2 x double> %34, ptr addrspace(1) %32, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_transpose_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(1179648) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(1179648) %1) local_unnamed_addr #3 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = shl nuw nsw i32 %5, 7
  %8 = or disjoint i32 %7, %6
  %9 = udiv i32 %8, 12
  %10 = urem i32 %9, 12
  %11 = mul nuw nsw i32 %10, 6144
  %12 = udiv i32 %8, 144
  %13 = mul nuw nsw i32 %12, 12
  %14 = mul i32 %9, 12
  %.decomposed = sub i32 %8, %14
  %15 = add nuw nsw i32 %13, %.decomposed
  %16 = add nuw nsw i32 %15, %11
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

attributes #0 = { norecurse nounwind "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 128}
!3 = !{i32 0, i32 9920}
!4 = !{}
!5 = !{i32 0, i32 14880}
!6 = !{i32 0, i32 576}
