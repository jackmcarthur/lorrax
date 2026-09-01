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
  %7 = udiv i32 %6, 20
  %8 = mul i32 %7, 20
  %.decomposed = sub i32 %6, %8
  %9 = shl nuw nsw i32 %.decomposed, 5
  %10 = and i32 %5, 31
  %11 = or disjoint i32 %9, %10
  %12 = icmp samesign ult i32 %11, 620
  br i1 %12, label %13, label %._crit_edge

13:                                               ; preds = %2
  %14 = trunc i32 %7 to i1
  %15 = select i1 %14, i32 19840, i32 0
  %16 = udiv i32 %6, 40
  %17 = mul nuw nsw i32 %16, 29760
  %18 = lshr i32 %5, 5
  %19 = mul nuw nsw i32 %18, 620
  %20 = add nuw nsw i32 %11, %17
  %21 = add nuw nsw i32 %20, %19
  %22 = add nuw nsw i32 %21, %15
  %23 = zext nneg i32 %22 to i64
  %24 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %23
  %25 = load <2 x double>, ptr addrspace(1) %24, align 16, !invariant.load !4
  %.unpack96 = extractelement <2 x double> %25, i32 0
  %.unpack297 = extractelement <2 x double> %25, i32 1
  %26 = mul nuw nsw i32 %10, 33
  %27 = add nuw nsw i32 %26, %18
  %28 = zext nneg i32 %27 to i64
  %29 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %28
  store double %.unpack96, ptr addrspace(3) %29, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 8
  store double %.unpack297, ptr addrspace(3) %.repack3, align 8
  %30 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 39680
  %31 = load <2 x double>, ptr addrspace(1) %30, align 16, !invariant.load !4
  %.unpack598 = extractelement <2 x double> %31, i32 0
  %.unpack799 = extractelement <2 x double> %31, i32 1
  %32 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 64
  store double %.unpack598, ptr addrspace(3) %32, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 72
  store double %.unpack799, ptr addrspace(3) %.repack8, align 8
  %33 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 79360
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !4
  %.unpack10100 = extractelement <2 x double> %34, i32 0
  %.unpack12101 = extractelement <2 x double> %34, i32 1
  %35 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 128
  store double %.unpack10100, ptr addrspace(3) %35, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 136
  store double %.unpack12101, ptr addrspace(3) %.repack13, align 8
  %36 = getelementptr inbounds i8, ptr addrspace(1) %24, i64 119040
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !4
  %.unpack15102 = extractelement <2 x double> %37, i32 0
  %.unpack17103 = extractelement <2 x double> %37, i32 1
  %38 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 192
  store double %.unpack15102, ptr addrspace(3) %38, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %29, i64 200
  store double %.unpack17103, ptr addrspace(3) %.repack18, align 8
  br label %._crit_edge

._crit_edge:                                      ; preds = %2, %13
  %39 = icmp ult i32 %11, 620
  %40 = and i32 %7, 1
  %41 = icmp eq i32 %40, 0
  %42 = and i1 %41, %39
  br i1 %42, label %.critedge, label %.critedge81

.critedge:                                        ; preds = %._crit_edge
  %43 = udiv i32 %6, 40
  %44 = mul nuw nsw i32 %43, 29760
  %45 = lshr i32 %5, 5
  %46 = mul nuw nsw i32 %45, 620
  %47 = or disjoint i32 %44, %10
  %48 = add nuw nsw i32 %47, %9
  %49 = add nuw nsw i32 %48, %46
  %50 = zext nneg i32 %49 to i64
  %51 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %50
  %52 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 158720
  %53 = load <2 x double>, ptr addrspace(1) %52, align 16, !invariant.load !4
  %.unpack2088 = extractelement <2 x double> %53, i32 0
  %.unpack2289 = extractelement <2 x double> %53, i32 1
  %54 = mul nuw nsw i32 %10, 33
  %55 = add nuw nsw i32 %54, %45
  %56 = zext nneg i32 %55 to i64
  %57 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %56
  %58 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 256
  store double %.unpack2088, ptr addrspace(3) %58, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 264
  store double %.unpack2289, ptr addrspace(3) %.repack23, align 8
  %59 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 198400
  %60 = load <2 x double>, ptr addrspace(1) %59, align 16, !invariant.load !4
  %.unpack2590 = extractelement <2 x double> %60, i32 0
  %.unpack2791 = extractelement <2 x double> %60, i32 1
  %61 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 320
  store double %.unpack2590, ptr addrspace(3) %61, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 328
  store double %.unpack2791, ptr addrspace(3) %.repack28, align 8
  %62 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 238080
  %63 = load <2 x double>, ptr addrspace(1) %62, align 16, !invariant.load !4
  %.unpack3092 = extractelement <2 x double> %63, i32 0
  %.unpack3293 = extractelement <2 x double> %63, i32 1
  %64 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 384
  store double %.unpack3092, ptr addrspace(3) %64, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 392
  store double %.unpack3293, ptr addrspace(3) %.repack33, align 8
  %65 = getelementptr inbounds i8, ptr addrspace(1) %51, i64 277760
  %66 = load <2 x double>, ptr addrspace(1) %65, align 16, !invariant.load !4
  %.unpack3594 = extractelement <2 x double> %66, i32 0
  %.unpack3795 = extractelement <2 x double> %66, i32 1
  %67 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 448
  store double %.unpack3594, ptr addrspace(3) %67, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 456
  store double %.unpack3795, ptr addrspace(3) %.repack38, align 8
  br label %.critedge81

.critedge81:                                      ; preds = %._crit_edge, %.critedge
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %68 = shl nuw nsw i32 %40, 5
  %69 = or disjoint i32 %68, %10
  %70 = icmp samesign ult i32 %69, 48
  br i1 %70, label %71, label %97

71:                                               ; preds = %.critedge81
  %72 = lshr i32 %5, 5
  %73 = mul nuw nsw i32 %72, 33
  %74 = add nuw nsw i32 %73, %10
  %75 = zext nneg i32 %74 to i64
  %76 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %75
  %.unpack40 = load double, ptr addrspace(3) %76, align 8
  %.elt41 = getelementptr inbounds i8, ptr addrspace(3) %76, i64 8
  %.unpack42 = load double, ptr addrspace(3) %.elt41, align 8
  %77 = mul nuw nsw i32 %.decomposed, 1536
  %78 = udiv i32 %6, 40
  %79 = mul nuw nsw i32 %78, 29760
  %80 = mul nuw nsw i32 %72, 48
  %81 = or disjoint i32 %77, %10
  %82 = add nuw nsw i32 %81, %79
  %83 = add nuw nsw i32 %82, %80
  %84 = add nuw nsw i32 %83, %68
  %85 = zext nneg i32 %84 to i64
  %86 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %85
  %87 = insertelement <2 x double> poison, double %.unpack40, i32 0
  %88 = insertelement <2 x double> %87, double %.unpack42, i32 1
  store <2 x double> %88, ptr addrspace(1) %86, align 16
  %89 = getelementptr inbounds i8, ptr addrspace(3) %76, i64 2112
  %.unpack45 = load double, ptr addrspace(3) %89, align 8
  %.elt46 = getelementptr inbounds i8, ptr addrspace(3) %76, i64 2120
  %.unpack47 = load double, ptr addrspace(3) %.elt46, align 8
  %90 = getelementptr inbounds i8, ptr addrspace(1) %86, i64 3072
  %91 = insertelement <2 x double> poison, double %.unpack45, i32 0
  %92 = insertelement <2 x double> %91, double %.unpack47, i32 1
  store <2 x double> %92, ptr addrspace(1) %90, align 16
  %93 = getelementptr inbounds i8, ptr addrspace(3) %76, i64 4224
  %.unpack50 = load double, ptr addrspace(3) %93, align 8
  %.elt51 = getelementptr inbounds i8, ptr addrspace(3) %76, i64 4232
  %.unpack52 = load double, ptr addrspace(3) %.elt51, align 8
  %94 = getelementptr inbounds i8, ptr addrspace(1) %86, i64 6144
  %95 = insertelement <2 x double> poison, double %.unpack50, i32 0
  %96 = insertelement <2 x double> %95, double %.unpack52, i32 1
  store <2 x double> %96, ptr addrspace(1) %94, align 16
  br label %97

97:                                               ; preds = %71, %.critedge81
  %98 = icmp ult i32 %69, 48
  %99 = icmp samesign ult i32 %.decomposed, 19
  %100 = and i1 %99, %98
  br i1 %100, label %.critedge83, label %.critedge86

.critedge83:                                      ; preds = %97
  %101 = lshr i32 %5, 5
  %102 = mul nuw nsw i32 %101, 33
  %103 = add nuw nsw i32 %102, %10
  %104 = zext nneg i32 %103 to i64
  %105 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %104
  %106 = getelementptr inbounds i8, ptr addrspace(3) %105, i64 6336
  %.unpack55 = load double, ptr addrspace(3) %106, align 8
  %.elt56 = getelementptr inbounds i8, ptr addrspace(3) %105, i64 6344
  %.unpack57 = load double, ptr addrspace(3) %.elt56, align 8
  %107 = mul nuw nsw i32 %.decomposed, 1536
  %108 = udiv i32 %6, 40
  %109 = mul nuw nsw i32 %108, 29760
  %110 = mul nuw nsw i32 %101, 48
  %111 = or disjoint i32 %107, %10
  %112 = add nuw nsw i32 %111, %109
  %113 = add nuw nsw i32 %112, %110
  %114 = add nuw nsw i32 %113, %68
  %115 = zext nneg i32 %114 to i64
  %116 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %115
  %117 = getelementptr inbounds i8, ptr addrspace(1) %116, i64 9216
  %118 = insertelement <2 x double> poison, double %.unpack55, i32 0
  %119 = insertelement <2 x double> %118, double %.unpack57, i32 1
  store <2 x double> %119, ptr addrspace(1) %117, align 16
  %120 = getelementptr inbounds i8, ptr addrspace(3) %105, i64 8448
  %.unpack60 = load double, ptr addrspace(3) %120, align 8
  %.elt61 = getelementptr inbounds i8, ptr addrspace(3) %105, i64 8456
  %.unpack62 = load double, ptr addrspace(3) %.elt61, align 8
  %121 = getelementptr inbounds i8, ptr addrspace(1) %116, i64 12288
  %122 = insertelement <2 x double> poison, double %.unpack60, i32 0
  %123 = insertelement <2 x double> %122, double %.unpack62, i32 1
  store <2 x double> %123, ptr addrspace(1) %121, align 16
  %124 = getelementptr inbounds i8, ptr addrspace(3) %105, i64 10560
  %.unpack65 = load double, ptr addrspace(3) %124, align 8
  %.elt66 = getelementptr inbounds i8, ptr addrspace(3) %105, i64 10568
  %.unpack67 = load double, ptr addrspace(3) %.elt66, align 8
  %125 = getelementptr inbounds i8, ptr addrspace(1) %116, i64 15360
  %126 = insertelement <2 x double> poison, double %.unpack65, i32 0
  %127 = insertelement <2 x double> %126, double %.unpack67, i32 1
  store <2 x double> %127, ptr addrspace(1) %125, align 16
  %128 = getelementptr inbounds i8, ptr addrspace(3) %105, i64 12672
  %.unpack70 = load double, ptr addrspace(3) %128, align 8
  %.elt71 = getelementptr inbounds i8, ptr addrspace(3) %105, i64 12680
  %.unpack72 = load double, ptr addrspace(3) %.elt71, align 8
  %129 = getelementptr inbounds i8, ptr addrspace(1) %116, i64 18432
  %130 = insertelement <2 x double> poison, double %.unpack70, i32 0
  %131 = insertelement <2 x double> %130, double %.unpack72, i32 1
  store <2 x double> %131, ptr addrspace(1) %129, align 16
  %132 = getelementptr inbounds i8, ptr addrspace(3) %105, i64 14784
  %.unpack75 = load double, ptr addrspace(3) %132, align 8
  %.elt76 = getelementptr inbounds i8, ptr addrspace(3) %105, i64 14792
  %.unpack77 = load double, ptr addrspace(3) %.elt76, align 8
  %133 = getelementptr inbounds i8, ptr addrspace(1) %116, i64 21504
  %134 = insertelement <2 x double> poison, double %.unpack75, i32 0
  %135 = insertelement <2 x double> %134, double %.unpack77, i32 1
  store <2 x double> %135, ptr addrspace(1) %133, align 16
  br label %.critedge86

.critedge86:                                      ; preds = %97, %.critedge83
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
!3 = !{i32 0, i32 20480}
!4 = !{}
