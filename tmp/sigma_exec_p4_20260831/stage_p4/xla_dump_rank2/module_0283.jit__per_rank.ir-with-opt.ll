; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(8530176) %0, ptr noalias readonly align 16 captures(none) dereferenceable(116) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(8530176) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = icmp samesign ult i32 %10, 133284
  br i1 %11, label %12, label %80

12:                                               ; preds = %3
  %13 = udiv i32 %10, 4596
  %14 = udiv i32 %10, 766
  %.lhs.trunc = trunc nuw i32 %14 to i8
  %15 = urem i8 %.lhs.trunc, 6
  %.zext = zext nneg i8 %15 to i32
  %16 = udiv i32 %10, 383
  %17 = zext nneg i32 %13 to i64
  %18 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %17
  %19 = load i32, ptr addrspace(1) %18, align 4, !invariant.load !4
  %20 = icmp slt i32 %19, 0
  %21 = add i32 %19, 29
  %22 = select i1 %20, i32 %21, i32 %19
  %23 = icmp ult i32 %22, 29
  %24 = tail call i32 @llvm.smax.i32(i32 %22, i32 0)
  %25 = tail call i32 @llvm.umin.i32(i32 %24, i32 28)
  %26 = mul i32 %16, 383
  %.decomposed = sub i32 %10, %26
  %27 = shl nuw nsw i32 %.decomposed, 3
  %28 = mul nuw nsw i32 %.zext, 177712
  %29 = trunc i32 %16 to i1
  %30 = select i1 %29, i32 88856, i32 0
  %31 = add nuw nsw i32 %28, %30
  %32 = mul nuw nsw i32 %25, 3064
  %33 = add nuw nsw i32 %31, %32
  %34 = add nuw nsw i32 %33, %27
  %35 = zext nneg i32 %34 to i64
  %36 = getelementptr inbounds double, ptr addrspace(1) %5, i64 %35
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !4
  %38 = extractelement <2 x double> %37, i32 0
  %39 = extractelement <2 x double> %37, i32 1
  %40 = fmul double %39, 0.000000e+00
  %41 = fadd double %39, 0.000000e+00
  %42 = fadd double %38, %40
  %43 = shl nuw nsw i32 %8, 2
  %44 = shl nuw nsw i32 %7, 9
  %45 = or disjoint i32 %43, %44
  %46 = zext nneg i32 %45 to i64
  %47 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %46
  %.elt = select i1 %23, double %42, double 0x7FF8000000000000
  %.elt2 = select i1 %23, double %41, double 0.000000e+00
  %48 = insertelement <2 x double> poison, double %.elt, i32 0
  %49 = insertelement <2 x double> %48, double %.elt2, i32 1
  store <2 x double> %49, ptr addrspace(1) %47, align 64
  %50 = getelementptr inbounds i8, ptr addrspace(1) %36, i64 16
  %51 = load <2 x double>, ptr addrspace(1) %50, align 16, !invariant.load !4
  %52 = extractelement <2 x double> %51, i32 0
  %53 = extractelement <2 x double> %51, i32 1
  %54 = fmul double %53, 0.000000e+00
  %55 = fadd double %53, 0.000000e+00
  %56 = fadd double %52, %54
  %57 = getelementptr inbounds i8, ptr addrspace(1) %47, i64 16
  %.elt3 = select i1 %23, double %56, double 0x7FF8000000000000
  %.elt5 = select i1 %23, double %55, double 0.000000e+00
  %58 = insertelement <2 x double> poison, double %.elt3, i32 0
  %59 = insertelement <2 x double> %58, double %.elt5, i32 1
  store <2 x double> %59, ptr addrspace(1) %57, align 16
  %60 = getelementptr inbounds i8, ptr addrspace(1) %36, i64 32
  %61 = load <2 x double>, ptr addrspace(1) %60, align 16, !invariant.load !4
  %62 = extractelement <2 x double> %61, i32 0
  %63 = extractelement <2 x double> %61, i32 1
  %64 = fmul double %63, 0.000000e+00
  %65 = fadd double %63, 0.000000e+00
  %66 = fadd double %62, %64
  %67 = getelementptr inbounds i8, ptr addrspace(1) %47, i64 32
  %.elt6 = select i1 %23, double %66, double 0x7FF8000000000000
  %.elt8 = select i1 %23, double %65, double 0.000000e+00
  %68 = insertelement <2 x double> poison, double %.elt6, i32 0
  %69 = insertelement <2 x double> %68, double %.elt8, i32 1
  store <2 x double> %69, ptr addrspace(1) %67, align 32
  %70 = getelementptr inbounds i8, ptr addrspace(1) %36, i64 48
  %71 = load <2 x double>, ptr addrspace(1) %70, align 16, !invariant.load !4
  %72 = extractelement <2 x double> %71, i32 0
  %73 = extractelement <2 x double> %71, i32 1
  %74 = fmul double %73, 0.000000e+00
  %75 = fadd double %73, 0.000000e+00
  %76 = fadd double %72, %74
  %77 = getelementptr inbounds i8, ptr addrspace(1) %47, i64 48
  %.elt9 = select i1 %23, double %76, double 0x7FF8000000000000
  %.elt11 = select i1 %23, double %75, double 0.000000e+00
  %78 = insertelement <2 x double> poison, double %.elt9, i32 0
  %79 = insertelement <2 x double> %78, double %.elt11, i32 1
  store <2 x double> %79, ptr addrspace(1) %77, align 16
  br label %80

80:                                               ; preds = %12, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 1042}
!3 = !{i32 0, i32 128}
!4 = !{}
