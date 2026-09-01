; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion_1(ptr noalias readonly align 16 captures(none) dereferenceable(196608) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(4096) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %7 = lshr i32 %5, 5
  %8 = mul nuw nsw i32 %7, 48
  %9 = mul nuw nsw i32 %6, 384
  %10 = and i32 %5, 31
  %11 = or disjoint i32 %10, %9
  %12 = add nuw nsw i32 %11, %8
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds double, ptr addrspace(1) %3, i64 %13
  %15 = load double, ptr addrspace(1) %14, align 8, !invariant.load !4
  %16 = icmp samesign ult i32 %10, 16
  br i1 %16, label %17, label %20

17:                                               ; preds = %2
  %sunkaddr = getelementptr inbounds i8, ptr addrspace(1) %14, i64 256
  %18 = load double, ptr addrspace(1) %sunkaddr, align 8, !invariant.load !4
  %19 = fadd nsz double %15, %18
  br label %20

20:                                               ; preds = %17, %2
  %21 = phi double [ %19, %17 ], [ %15, %2 ]
  %22 = bitcast double %21 to <2 x i32>
  %23 = extractelement <2 x i32> %22, i64 0
  %24 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %23, i32 16, i32 31)
  %25 = insertelement <2 x i32> poison, i32 %24, i64 0
  %26 = extractelement <2 x i32> %22, i64 1
  %27 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %26, i32 16, i32 31)
  %28 = insertelement <2 x i32> %25, i32 %27, i64 1
  %29 = bitcast <2 x i32> %28 to double
  %30 = fadd nsz double %21, %29
  %31 = bitcast double %30 to <2 x i32>
  %32 = extractelement <2 x i32> %31, i64 0
  %33 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %32, i32 8, i32 31)
  %34 = insertelement <2 x i32> poison, i32 %33, i64 0
  %35 = extractelement <2 x i32> %31, i64 1
  %36 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %35, i32 8, i32 31)
  %37 = insertelement <2 x i32> %34, i32 %36, i64 1
  %38 = bitcast <2 x i32> %37 to double
  %39 = fadd nsz double %30, %38
  %40 = bitcast double %39 to <2 x i32>
  %41 = extractelement <2 x i32> %40, i64 0
  %42 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %41, i32 4, i32 31)
  %43 = insertelement <2 x i32> poison, i32 %42, i64 0
  %44 = extractelement <2 x i32> %40, i64 1
  %45 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %44, i32 4, i32 31)
  %46 = insertelement <2 x i32> %43, i32 %45, i64 1
  %47 = bitcast <2 x i32> %46 to double
  %48 = fadd nsz double %39, %47
  %49 = bitcast double %48 to <2 x i32>
  %50 = extractelement <2 x i32> %49, i64 0
  %51 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %50, i32 2, i32 31)
  %52 = insertelement <2 x i32> poison, i32 %51, i64 0
  %53 = extractelement <2 x i32> %49, i64 1
  %54 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %53, i32 2, i32 31)
  %55 = insertelement <2 x i32> %52, i32 %54, i64 1
  %56 = bitcast <2 x i32> %55 to double
  %57 = fadd nsz double %48, %56
  %58 = bitcast double %57 to <2 x i32>
  %59 = extractelement <2 x i32> %58, i64 0
  %60 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %59, i32 1, i32 31)
  %61 = extractelement <2 x i32> %58, i64 1
  %62 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %61, i32 1, i32 31)
  %63 = icmp eq i32 %10, 0
  %64 = icmp samesign ult i32 %5, 225
  %65 = and i1 %64, %63
  br i1 %65, label %66, label %75

66:                                               ; preds = %20
  %67 = shl nuw nsw i32 %6, 3
  %68 = or disjoint i32 %67, %7
  %69 = zext nneg i32 %68 to i64
  %70 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %69
  %71 = insertelement <2 x i32> poison, i32 %60, i64 0
  %72 = insertelement <2 x i32> %71, i32 %62, i64 1
  %73 = bitcast <2 x i32> %72 to double
  %74 = fadd nsz double %57, %73
  store double %74, ptr addrspace(1) %70, align 8
  br label %75

75:                                               ; preds = %66, %20
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(4096) %0, ptr noalias readonly align 256 captures(none) dereferenceable(4096) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(8) %2) local_unnamed_addr #3 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %8 = zext nneg i32 %7 to i64
  %9 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %8
  %10 = load double, ptr addrspace(1) %9, align 8, !invariant.load !4
  %11 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %8
  %12 = load double, ptr addrspace(1) %11, align 8, !invariant.load !4
  %13 = fmul double %10, %12
  %14 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 256
  %15 = load double, ptr addrspace(1) %14, align 8, !invariant.load !4
  %16 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 256
  %17 = load double, ptr addrspace(1) %16, align 8, !invariant.load !4
  %18 = fmul double %15, %17
  %19 = fadd nsz double %13, %18
  %20 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 512
  %21 = load double, ptr addrspace(1) %20, align 8, !invariant.load !4
  %22 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 512
  %23 = load double, ptr addrspace(1) %22, align 8, !invariant.load !4
  %24 = fmul double %21, %23
  %25 = fadd nsz double %19, %24
  %26 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 768
  %27 = load double, ptr addrspace(1) %26, align 8, !invariant.load !4
  %28 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 768
  %29 = load double, ptr addrspace(1) %28, align 8, !invariant.load !4
  %30 = fmul double %27, %29
  %31 = fadd nsz double %25, %30
  %32 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 1024
  %33 = load double, ptr addrspace(1) %32, align 8, !invariant.load !4
  %34 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 1024
  %35 = load double, ptr addrspace(1) %34, align 8, !invariant.load !4
  %36 = fmul double %33, %35
  %37 = fadd nsz double %31, %36
  %38 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 1280
  %39 = load double, ptr addrspace(1) %38, align 8, !invariant.load !4
  %40 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 1280
  %41 = load double, ptr addrspace(1) %40, align 8, !invariant.load !4
  %42 = fmul double %39, %41
  %43 = fadd nsz double %37, %42
  %44 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 1536
  %45 = load double, ptr addrspace(1) %44, align 8, !invariant.load !4
  %46 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 1536
  %47 = load double, ptr addrspace(1) %46, align 8, !invariant.load !4
  %48 = fmul double %45, %47
  %49 = fadd nsz double %43, %48
  %50 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 1792
  %51 = load double, ptr addrspace(1) %50, align 8, !invariant.load !4
  %52 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 1792
  %53 = load double, ptr addrspace(1) %52, align 8, !invariant.load !4
  %54 = fmul double %51, %53
  %55 = fadd nsz double %49, %54
  %56 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 2048
  %57 = load double, ptr addrspace(1) %56, align 8, !invariant.load !4
  %58 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 2048
  %59 = load double, ptr addrspace(1) %58, align 8, !invariant.load !4
  %60 = fmul double %57, %59
  %61 = fadd nsz double %55, %60
  %62 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 2304
  %63 = load double, ptr addrspace(1) %62, align 8, !invariant.load !4
  %64 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 2304
  %65 = load double, ptr addrspace(1) %64, align 8, !invariant.load !4
  %66 = fmul double %63, %65
  %67 = fadd nsz double %61, %66
  %68 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 2560
  %69 = load double, ptr addrspace(1) %68, align 8, !invariant.load !4
  %70 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 2560
  %71 = load double, ptr addrspace(1) %70, align 8, !invariant.load !4
  %72 = fmul double %69, %71
  %73 = fadd nsz double %67, %72
  %74 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 2816
  %75 = load double, ptr addrspace(1) %74, align 8, !invariant.load !4
  %76 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 2816
  %77 = load double, ptr addrspace(1) %76, align 8, !invariant.load !4
  %78 = fmul double %75, %77
  %79 = fadd nsz double %73, %78
  %80 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 3072
  %81 = load double, ptr addrspace(1) %80, align 8, !invariant.load !4
  %82 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 3072
  %83 = load double, ptr addrspace(1) %82, align 8, !invariant.load !4
  %84 = fmul double %81, %83
  %85 = fadd nsz double %79, %84
  %86 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 3328
  %87 = load double, ptr addrspace(1) %86, align 8, !invariant.load !4
  %88 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 3328
  %89 = load double, ptr addrspace(1) %88, align 8, !invariant.load !4
  %90 = fmul double %87, %89
  %91 = fadd nsz double %85, %90
  %92 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 3584
  %93 = load double, ptr addrspace(1) %92, align 8, !invariant.load !4
  %94 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 3584
  %95 = load double, ptr addrspace(1) %94, align 8, !invariant.load !4
  %96 = fmul double %93, %95
  %97 = fadd nsz double %91, %96
  %98 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 3840
  %99 = load double, ptr addrspace(1) %98, align 8, !invariant.load !4
  %100 = getelementptr inbounds i8, ptr addrspace(1) %11, i64 3840
  %101 = load double, ptr addrspace(1) %100, align 8, !invariant.load !4
  %102 = fmul double %99, %101
  %103 = fadd nsz double %97, %102
  %104 = bitcast double %103 to <2 x i32>
  %105 = extractelement <2 x i32> %104, i64 0
  %106 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %105, i32 16, i32 31)
  %107 = insertelement <2 x i32> poison, i32 %106, i64 0
  %108 = extractelement <2 x i32> %104, i64 1
  %109 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %108, i32 16, i32 31)
  %110 = insertelement <2 x i32> %107, i32 %109, i64 1
  %111 = bitcast <2 x i32> %110 to double
  %112 = fadd nsz double %103, %111
  %113 = bitcast double %112 to <2 x i32>
  %114 = extractelement <2 x i32> %113, i64 0
  %115 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %114, i32 8, i32 31)
  %116 = insertelement <2 x i32> poison, i32 %115, i64 0
  %117 = extractelement <2 x i32> %113, i64 1
  %118 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %117, i32 8, i32 31)
  %119 = insertelement <2 x i32> %116, i32 %118, i64 1
  %120 = bitcast <2 x i32> %119 to double
  %121 = fadd nsz double %112, %120
  %122 = bitcast double %121 to <2 x i32>
  %123 = extractelement <2 x i32> %122, i64 0
  %124 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %123, i32 4, i32 31)
  %125 = insertelement <2 x i32> poison, i32 %124, i64 0
  %126 = extractelement <2 x i32> %122, i64 1
  %127 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %126, i32 4, i32 31)
  %128 = insertelement <2 x i32> %125, i32 %127, i64 1
  %129 = bitcast <2 x i32> %128 to double
  %130 = fadd nsz double %121, %129
  %131 = bitcast double %130 to <2 x i32>
  %132 = extractelement <2 x i32> %131, i64 0
  %133 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %132, i32 2, i32 31)
  %134 = insertelement <2 x i32> poison, i32 %133, i64 0
  %135 = extractelement <2 x i32> %131, i64 1
  %136 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %135, i32 2, i32 31)
  %137 = insertelement <2 x i32> %134, i32 %136, i64 1
  %138 = bitcast <2 x i32> %137 to double
  %139 = fadd nsz double %130, %138
  %140 = bitcast double %139 to <2 x i32>
  %141 = extractelement <2 x i32> %140, i64 0
  %142 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %141, i32 1, i32 31)
  %143 = extractelement <2 x i32> %140, i64 1
  %144 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %143, i32 1, i32 31)
  %145 = icmp eq i32 %7, 0
  %146 = insertelement <2 x i32> poison, i32 %142, i64 0
  %147 = insertelement <2 x i32> %146, i32 %144, i64 1
  %148 = bitcast <2 x i32> %147 to double
  %149 = fadd nsz double %139, %148
  br i1 %145, label %150, label %151

150:                                              ; preds = %3
  store double %149, ptr addrspace(1) %6, align 256
  br label %151

151:                                              ; preds = %150, %3
  ret void
}

attributes #0 = { norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "nvvm.reqntid"="256,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "nvvm.reqntid"="32,1,1" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 256}
!3 = !{i32 0, i32 64}
!4 = !{}
!5 = !{i32 0, i32 32}
