; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(262144) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(1024) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %7 = lshr i32 %5, 5
  %8 = shl nuw nsw i32 %7, 8
  %9 = shl nuw nsw i32 %6, 11
  %10 = or disjoint i32 %8, %9
  %11 = and i32 %5, 31
  %12 = or disjoint i32 %10, %11
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds double, ptr addrspace(1) %3, i64 %13
  %15 = load double, ptr addrspace(1) %14, align 8, !invariant.load !4
  %16 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 256
  %17 = load double, ptr addrspace(1) %16, align 8, !invariant.load !4
  %18 = fadd nsz double %15, %17
  %19 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 512
  %20 = load double, ptr addrspace(1) %19, align 8, !invariant.load !4
  %21 = fadd nsz double %18, %20
  %22 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 768
  %23 = load double, ptr addrspace(1) %22, align 8, !invariant.load !4
  %24 = fadd nsz double %21, %23
  %25 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 1024
  %26 = load double, ptr addrspace(1) %25, align 8, !invariant.load !4
  %27 = fadd nsz double %24, %26
  %28 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 1280
  %29 = load double, ptr addrspace(1) %28, align 8, !invariant.load !4
  %30 = fadd nsz double %27, %29
  %31 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 1536
  %32 = load double, ptr addrspace(1) %31, align 8, !invariant.load !4
  %33 = fadd nsz double %30, %32
  %34 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 1792
  %35 = load double, ptr addrspace(1) %34, align 8, !invariant.load !4
  %36 = fadd nsz double %33, %35
  %37 = bitcast double %36 to <2 x i32>
  %38 = extractelement <2 x i32> %37, i64 0
  %39 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %38, i32 16, i32 31)
  %40 = insertelement <2 x i32> poison, i32 %39, i64 0
  %41 = extractelement <2 x i32> %37, i64 1
  %42 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %41, i32 16, i32 31)
  %43 = insertelement <2 x i32> %40, i32 %42, i64 1
  %44 = bitcast <2 x i32> %43 to double
  %45 = fadd nsz double %36, %44
  %46 = bitcast double %45 to <2 x i32>
  %47 = extractelement <2 x i32> %46, i64 0
  %48 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %47, i32 8, i32 31)
  %49 = insertelement <2 x i32> poison, i32 %48, i64 0
  %50 = extractelement <2 x i32> %46, i64 1
  %51 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %50, i32 8, i32 31)
  %52 = insertelement <2 x i32> %49, i32 %51, i64 1
  %53 = bitcast <2 x i32> %52 to double
  %54 = fadd nsz double %45, %53
  %55 = bitcast double %54 to <2 x i32>
  %56 = extractelement <2 x i32> %55, i64 0
  %57 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %56, i32 4, i32 31)
  %58 = insertelement <2 x i32> poison, i32 %57, i64 0
  %59 = extractelement <2 x i32> %55, i64 1
  %60 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %59, i32 4, i32 31)
  %61 = insertelement <2 x i32> %58, i32 %60, i64 1
  %62 = bitcast <2 x i32> %61 to double
  %63 = fadd nsz double %54, %62
  %64 = bitcast double %63 to <2 x i32>
  %65 = extractelement <2 x i32> %64, i64 0
  %66 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %65, i32 2, i32 31)
  %67 = insertelement <2 x i32> poison, i32 %66, i64 0
  %68 = extractelement <2 x i32> %64, i64 1
  %69 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %68, i32 2, i32 31)
  %70 = insertelement <2 x i32> %67, i32 %69, i64 1
  %71 = bitcast <2 x i32> %70 to double
  %72 = fadd nsz double %63, %71
  %73 = bitcast double %72 to <2 x i32>
  %74 = extractelement <2 x i32> %73, i64 0
  %75 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %74, i32 1, i32 31)
  %76 = extractelement <2 x i32> %73, i64 1
  %77 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %76, i32 1, i32 31)
  %78 = icmp eq i32 %11, 0
  %79 = icmp samesign ult i32 %5, 225
  %80 = and i1 %79, %78
  br i1 %80, label %81, label %90

81:                                               ; preds = %2
  %82 = shl nuw nsw i32 %6, 3
  %83 = or disjoint i32 %82, %7
  %84 = zext nneg i32 %83 to i64
  %85 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %84
  %86 = insertelement <2 x i32> poison, i32 %75, i64 0
  %87 = insertelement <2 x i32> %86, i32 %77, i64 1
  %88 = bitcast <2 x i32> %87 to double
  %89 = fadd nsz double %72, %88
  store double %89, ptr addrspace(1) %85, align 8
  br label %90

90:                                               ; preds = %81, %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #2

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(1024) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #3 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds double, ptr addrspace(1) %3, i64 %6
  %8 = load double, ptr addrspace(1) %7, align 8, !invariant.load !4
  %9 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 256
  %10 = load double, ptr addrspace(1) %9, align 8, !invariant.load !4
  %11 = fadd nsz double %8, %10
  %12 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 512
  %13 = load double, ptr addrspace(1) %12, align 8, !invariant.load !4
  %14 = fadd nsz double %11, %13
  %15 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 768
  %16 = load double, ptr addrspace(1) %15, align 8, !invariant.load !4
  %17 = fadd nsz double %14, %16
  %18 = bitcast double %17 to <2 x i32>
  %19 = extractelement <2 x i32> %18, i64 0
  %20 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %19, i32 16, i32 31)
  %21 = insertelement <2 x i32> poison, i32 %20, i64 0
  %22 = extractelement <2 x i32> %18, i64 1
  %23 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %22, i32 16, i32 31)
  %24 = insertelement <2 x i32> %21, i32 %23, i64 1
  %25 = bitcast <2 x i32> %24 to double
  %26 = fadd nsz double %17, %25
  %27 = bitcast double %26 to <2 x i32>
  %28 = extractelement <2 x i32> %27, i64 0
  %29 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %28, i32 8, i32 31)
  %30 = insertelement <2 x i32> poison, i32 %29, i64 0
  %31 = extractelement <2 x i32> %27, i64 1
  %32 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %31, i32 8, i32 31)
  %33 = insertelement <2 x i32> %30, i32 %32, i64 1
  %34 = bitcast <2 x i32> %33 to double
  %35 = fadd nsz double %26, %34
  %36 = bitcast double %35 to <2 x i32>
  %37 = extractelement <2 x i32> %36, i64 0
  %38 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %37, i32 4, i32 31)
  %39 = insertelement <2 x i32> poison, i32 %38, i64 0
  %40 = extractelement <2 x i32> %36, i64 1
  %41 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %40, i32 4, i32 31)
  %42 = insertelement <2 x i32> %39, i32 %41, i64 1
  %43 = bitcast <2 x i32> %42 to double
  %44 = fadd nsz double %35, %43
  %45 = bitcast double %44 to <2 x i32>
  %46 = extractelement <2 x i32> %45, i64 0
  %47 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %46, i32 2, i32 31)
  %48 = insertelement <2 x i32> poison, i32 %47, i64 0
  %49 = extractelement <2 x i32> %45, i64 1
  %50 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %49, i32 2, i32 31)
  %51 = insertelement <2 x i32> %48, i32 %50, i64 1
  %52 = bitcast <2 x i32> %51 to double
  %53 = fadd nsz double %44, %52
  %54 = bitcast double %53 to <2 x i32>
  %55 = extractelement <2 x i32> %54, i64 0
  %56 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %55, i32 1, i32 31)
  %57 = extractelement <2 x i32> %54, i64 1
  %58 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %57, i32 1, i32 31)
  %59 = icmp eq i32 %5, 0
  %60 = insertelement <2 x i32> poison, i32 %56, i64 0
  %61 = insertelement <2 x i32> %60, i32 %58, i64 1
  %62 = bitcast <2 x i32> %61 to double
  %63 = fadd nsz double %53, %62
  br i1 %59, label %64, label %65

64:                                               ; preds = %2
  store double %63, ptr addrspace(1) %4, align 256
  br label %65

65:                                               ; preds = %64, %2
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
!3 = !{i32 0, i32 16}
!4 = !{}
!5 = !{i32 0, i32 32}
