; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x double] undef

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion_1(ptr noalias readonly align 16 captures(none) dereferenceable(8) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(16) initializes((0, 16)) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = load double, ptr addrspace(1) %3, align 16, !invariant.load !4
  %6 = fdiv double 3.276800e+04, %5
  %7 = tail call double @llvm.sqrt.f64(double %6)
  %8 = fmul double %7, 0x4066A09E667F3BCD
  %9 = fmul double %7, 0.000000e+00
  %10 = fadd double %9, 0.000000e+00
  %11 = insertelement <2 x double> poison, double %8, i32 0
  %12 = insertelement <2 x double> %11, double %10, i32 1
  store <2 x double> %12, ptr addrspace(1) %4, align 256
  ret void
}

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_reduce_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(16777216) %0, ptr noalias readonly align 256 captures(none) dereferenceable(16) %1, ptr noalias readonly align 16 captures(none) dereferenceable(128) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(262144) %3) local_unnamed_addr #1 {
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %10 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %11 = lshr i32 %9, 6
  %12 = lshr i32 %9, 5
  %13 = shl nuw nsw i32 %12, 15
  %14 = shl nuw nsw i32 %10, 5
  %15 = or disjoint i32 %13, %14
  %16 = and i32 %9, 31
  %17 = or disjoint i32 %15, %16
  %18 = zext nneg i32 %17 to i64
  %19 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %18
  %20 = load <2 x double>, ptr addrspace(1) %19, align 16, !invariant.load !4
  %.unpack6 = extractelement <2 x double> %20, i32 0
  %.unpack27 = extractelement <2 x double> %20, i32 1
  %21 = load <2 x double>, ptr addrspace(1) %6, align 256, !invariant.load !4
  %.unpack38 = extractelement <2 x double> %21, i32 0
  %.unpack59 = extractelement <2 x double> %21, i32 1
  %22 = fmul double %.unpack6, %.unpack38
  %23 = fmul double %.unpack27, %.unpack59
  %24 = fsub double %22, %23
  %25 = fmul double %.unpack27, %.unpack38
  %26 = fmul double %.unpack6, %.unpack59
  %27 = fadd double %25, %26
  %28 = fmul double %24, %24
  %29 = fmul double %27, %27
  %30 = fadd double %28, %29
  %31 = zext nneg i32 %11 to i64
  %32 = getelementptr inbounds double, ptr addrspace(1) %7, i64 %31
  %33 = load double, ptr addrspace(1) %32, align 8, !invariant.load !4
  %34 = fmul double %33, %30
  %35 = mul nuw nsw i32 %16, 33
  %36 = add nuw nsw i32 %35, %12
  %37 = zext nneg i32 %36 to i64
  %38 = getelementptr inbounds double, ptr addrspace(3) @shared_0, i64 %37
  store double %34, ptr addrspace(3) %38, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %39 = mul nuw nsw i32 %12, 33
  %40 = add nuw nsw i32 %39, %16
  %41 = zext nneg i32 %40 to i64
  %42 = getelementptr inbounds double, ptr addrspace(3) @shared_0, i64 %41
  %43 = load double, ptr addrspace(3) %42, align 8
  %44 = bitcast double %43 to <2 x i32>
  %45 = extractelement <2 x i32> %44, i64 0
  %46 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %45, i32 16, i32 31)
  %47 = insertelement <2 x i32> poison, i32 %46, i64 0
  %48 = extractelement <2 x i32> %44, i64 1
  %49 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %48, i32 16, i32 31)
  %50 = insertelement <2 x i32> %47, i32 %49, i64 1
  %51 = bitcast <2 x i32> %50 to double
  %52 = fadd nsz double %43, %51
  %53 = bitcast double %52 to <2 x i32>
  %54 = extractelement <2 x i32> %53, i64 0
  %55 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %54, i32 8, i32 31)
  %56 = insertelement <2 x i32> poison, i32 %55, i64 0
  %57 = extractelement <2 x i32> %53, i64 1
  %58 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %57, i32 8, i32 31)
  %59 = insertelement <2 x i32> %56, i32 %58, i64 1
  %60 = bitcast <2 x i32> %59 to double
  %61 = fadd nsz double %52, %60
  %62 = bitcast double %61 to <2 x i32>
  %63 = extractelement <2 x i32> %62, i64 0
  %64 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %63, i32 4, i32 31)
  %65 = insertelement <2 x i32> poison, i32 %64, i64 0
  %66 = extractelement <2 x i32> %62, i64 1
  %67 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %66, i32 4, i32 31)
  %68 = insertelement <2 x i32> %65, i32 %67, i64 1
  %69 = bitcast <2 x i32> %68 to double
  %70 = fadd nsz double %61, %69
  %71 = bitcast double %70 to <2 x i32>
  %72 = extractelement <2 x i32> %71, i64 0
  %73 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %72, i32 2, i32 31)
  %74 = insertelement <2 x i32> poison, i32 %73, i64 0
  %75 = extractelement <2 x i32> %71, i64 1
  %76 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %75, i32 2, i32 31)
  %77 = insertelement <2 x i32> %74, i32 %76, i64 1
  %78 = bitcast <2 x i32> %77 to double
  %79 = fadd nsz double %70, %78
  %80 = bitcast double %79 to <2 x i32>
  %81 = extractelement <2 x i32> %80, i64 0
  %82 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %81, i32 1, i32 31)
  %83 = extractelement <2 x i32> %80, i64 1
  %84 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %83, i32 1, i32 31)
  %85 = icmp eq i32 %16, 0
  %86 = icmp samesign ult i32 %9, 993
  %87 = and i1 %86, %85
  br i1 %87, label %88, label %96

88:                                               ; preds = %4
  %89 = or disjoint i32 %14, %12
  %90 = zext nneg i32 %89 to i64
  %91 = getelementptr inbounds double, ptr addrspace(1) %8, i64 %90
  %92 = insertelement <2 x i32> poison, i32 %82, i64 0
  %93 = insertelement <2 x i32> %92, i32 %84, i64 1
  %94 = bitcast <2 x i32> %93 to double
  %95 = fadd nsz double %79, %94
  store double %95, ptr addrspace(1) %91, align 8
  br label %96

96:                                               ; preds = %88, %4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 captures(none) dereferenceable(262144) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias readonly align 16 captures(none) dereferenceable(8) %2, ptr noalias readnone align 256 captures(none) dereferenceable(262144) %3) local_unnamed_addr #5 {
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !7
  %10 = load double, ptr addrspace(1) %5, align 16, !invariant.load !4
  %11 = load double, ptr addrspace(1) %6, align 16, !invariant.load !4
  %12 = fmul double %10, %11
  %13 = shl nuw nsw i32 %8, 7
  %14 = or disjoint i32 %13, %9
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds double, ptr addrspace(1) %7, i64 %15
  %17 = load double, ptr addrspace(1) %16, align 8
  %18 = fmul double %12, %17
  store double %18, ptr addrspace(1) %16, align 8
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #1 = { norecurse nounwind "nvvm.reqntid"="1024,1,1" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nounwind }
attributes #4 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #5 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{}
!5 = !{i32 0, i32 1024}
!6 = !{i32 0, i32 256}
!7 = !{i32 0, i32 128}
