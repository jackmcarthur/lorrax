; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x double] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_reduce_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(12582912) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(262144) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %7 = lshr i32 %5, 5
  %8 = shl nuw nsw i32 %7, 15
  %9 = shl nuw nsw i32 %6, 5
  %10 = or disjoint i32 %8, %9
  %11 = and i32 %5, 31
  %12 = or disjoint i32 %10, %11
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds double, ptr addrspace(1) %3, i64 %13
  %15 = load double, ptr addrspace(1) %14, align 8, !invariant.load !3
  %16 = icmp samesign ult i32 %5, 512
  br i1 %16, label %17, label %20

17:                                               ; preds = %2
  %sunkaddr = getelementptr inbounds i8, ptr addrspace(1) %14, i64 8388608
  %18 = load double, ptr addrspace(1) %sunkaddr, align 8, !invariant.load !3
  %19 = fadd nsz double %15, %18
  br label %20

20:                                               ; preds = %17, %2
  %21 = phi double [ %19, %17 ], [ %15, %2 ]
  %22 = mul nuw nsw i32 %11, 33
  %23 = add nuw nsw i32 %22, %7
  %24 = zext nneg i32 %23 to i64
  %25 = getelementptr inbounds double, ptr addrspace(3) @shared_0, i64 %24
  store double %21, ptr addrspace(3) %25, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %26 = mul nuw nsw i32 %7, 33
  %27 = add nuw nsw i32 %26, %11
  %28 = zext nneg i32 %27 to i64
  %29 = getelementptr inbounds double, ptr addrspace(3) @shared_0, i64 %28
  %30 = load double, ptr addrspace(3) %29, align 8
  %31 = bitcast double %30 to <2 x i32>
  %32 = extractelement <2 x i32> %31, i64 0
  %33 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %32, i32 16, i32 31)
  %34 = insertelement <2 x i32> poison, i32 %33, i64 0
  %35 = extractelement <2 x i32> %31, i64 1
  %36 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %35, i32 16, i32 31)
  %37 = insertelement <2 x i32> %34, i32 %36, i64 1
  %38 = bitcast <2 x i32> %37 to double
  %39 = fadd nsz double %30, %38
  %40 = bitcast double %39 to <2 x i32>
  %41 = extractelement <2 x i32> %40, i64 0
  %42 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %41, i32 8, i32 31)
  %43 = insertelement <2 x i32> poison, i32 %42, i64 0
  %44 = extractelement <2 x i32> %40, i64 1
  %45 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %44, i32 8, i32 31)
  %46 = insertelement <2 x i32> %43, i32 %45, i64 1
  %47 = bitcast <2 x i32> %46 to double
  %48 = fadd nsz double %39, %47
  %49 = bitcast double %48 to <2 x i32>
  %50 = extractelement <2 x i32> %49, i64 0
  %51 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %50, i32 4, i32 31)
  %52 = insertelement <2 x i32> poison, i32 %51, i64 0
  %53 = extractelement <2 x i32> %49, i64 1
  %54 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %53, i32 4, i32 31)
  %55 = insertelement <2 x i32> %52, i32 %54, i64 1
  %56 = bitcast <2 x i32> %55 to double
  %57 = fadd nsz double %48, %56
  %58 = bitcast double %57 to <2 x i32>
  %59 = extractelement <2 x i32> %58, i64 0
  %60 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %59, i32 2, i32 31)
  %61 = insertelement <2 x i32> poison, i32 %60, i64 0
  %62 = extractelement <2 x i32> %58, i64 1
  %63 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %62, i32 2, i32 31)
  %64 = insertelement <2 x i32> %61, i32 %63, i64 1
  %65 = bitcast <2 x i32> %64 to double
  %66 = fadd nsz double %57, %65
  %67 = bitcast double %66 to <2 x i32>
  %68 = extractelement <2 x i32> %67, i64 0
  %69 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %68, i32 1, i32 31)
  %70 = extractelement <2 x i32> %67, i64 1
  %71 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %70, i32 1, i32 31)
  %72 = icmp eq i32 %11, 0
  %73 = icmp samesign ult i32 %5, 993
  %74 = and i1 %73, %72
  br i1 %74, label %75, label %83

75:                                               ; preds = %20
  %76 = or disjoint i32 %9, %7
  %77 = zext nneg i32 %76 to i64
  %78 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %77
  %79 = insertelement <2 x i32> poison, i32 %69, i64 0
  %80 = insertelement <2 x i32> %79, i32 %71, i64 1
  %81 = bitcast <2 x i32> %80 to double
  %82 = fadd nsz double %66, %81
  store double %82, ptr addrspace(1) %78, align 8
  br label %83

83:                                               ; preds = %75, %20
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #3

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 captures(none) dereferenceable(262144) %0, ptr noalias readnone align 256 captures(none) dereferenceable(262144) %1) local_unnamed_addr #4 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %6 = shl nuw nsw i32 %4, 7
  %7 = or disjoint i32 %6, %5
  %8 = zext nneg i32 %7 to i64
  %9 = getelementptr inbounds double, ptr addrspace(1) %3, i64 %8
  %10 = load double, ptr addrspace(1) %9, align 8
  %11 = fmul double %10, 0x3F95555555555555
  store double %11, ptr addrspace(1) %9, align 8
  ret void
}

attributes #0 = { norecurse nounwind "nvvm.reqntid"="1024,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #4 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 1024}
!3 = !{}
!4 = !{i32 0, i32 256}
!5 = !{i32 0, i32 128}
