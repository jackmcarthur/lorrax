; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(24772608) %0, ptr noalias readonly align 256 captures(none) dereferenceable(4) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(4128768) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = load i32, ptr addrspace(1) %4, align 256, !invariant.load !4
  %10 = and i32 %9, 2
  %.not = icmp eq i32 %10, 0
  %11 = select i1 %.not, i64 0, i64 12
  %12 = trunc i32 %9 to i1
  %13 = select i1 %12, i64 12, i64 0
  %14 = shl nuw nsw i32 %7, 7
  %15 = or disjoint i32 %14, %8
  %16 = udiv i32 %15, 6
  %17 = mul i32 %16, 6
  %.decomposed = sub i32 %15, %17
  %18 = shl nuw nsw i32 %.decomposed, 2
  %19 = zext nneg i32 %18 to i64
  %20 = sub nsw i64 %19, %11
  %21 = icmp ult i64 %20, 12
  %22 = sub nsw i64 %19, %13
  %23 = icmp ult i64 %22, 12
  %24 = and i1 %23, %21
  %25 = tail call i64 @llvm.smax.i64(i64 %20, i64 0)
  %26 = tail call i64 @llvm.umin.i64(i64 %25, i64 11)
  %27 = trunc nuw nsw i64 %26 to i32
  %28 = tail call i64 @llvm.smax.i64(i64 %22, i64 0)
  %29 = tail call i64 @llvm.umin.i64(i64 %28, i64 11)
  %30 = trunc nuw nsw i64 %29 to i32
  %31 = mul nuw nsw i32 %16, 144
  %32 = mul nuw nsw i32 %27, 12
  %33 = or disjoint i32 %31, %30
  %34 = add nuw nsw i32 %33, %32
  %35 = zext nneg i32 %34 to i64
  %36 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %35
  %37 = load <2 x double>, ptr addrspace(1) %36, align 16, !invariant.load !4
  %.unpack32 = extractelement <2 x double> %37, i32 0
  %.unpack233 = extractelement <2 x double> %37, i32 1
  %38 = shl nuw nsw i32 %8, 2
  %39 = shl nuw nsw i32 %7, 9
  %40 = or disjoint i32 %38, %39
  %41 = zext nneg i32 %40 to i64
  %42 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %41
  %.elt = select i1 %24, double %.unpack32, double 0.000000e+00
  %.elt4 = select i1 %24, double %.unpack233, double 0.000000e+00
  %43 = insertelement <2 x double> poison, double %.elt, i32 0
  %44 = insertelement <2 x double> %43, double %.elt4, i32 1
  store <2 x double> %44, ptr addrspace(1) %42, align 64
  %45 = or disjoint i32 %18, 1
  %46 = zext nneg i32 %45 to i64
  %47 = sub nsw i64 %46, %11
  %48 = icmp ult i64 %47, 12
  %49 = sub nsw i64 %46, %13
  %50 = icmp ult i64 %49, 12
  %51 = and i1 %50, %48
  %52 = tail call i64 @llvm.smax.i64(i64 %47, i64 0)
  %53 = tail call i64 @llvm.umin.i64(i64 %52, i64 11)
  %54 = trunc nuw nsw i64 %53 to i32
  %55 = tail call i64 @llvm.smax.i64(i64 %49, i64 0)
  %56 = tail call i64 @llvm.umin.i64(i64 %55, i64 11)
  %57 = trunc nuw nsw i64 %56 to i32
  %58 = mul nuw nsw i32 %54, 12
  %59 = or disjoint i32 %31, %57
  %60 = add nuw nsw i32 %59, %58
  %61 = zext nneg i32 %60 to i64
  %62 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %61
  %63 = load <2 x double>, ptr addrspace(1) %62, align 16, !invariant.load !4
  %.unpack630 = extractelement <2 x double> %63, i32 0
  %.unpack831 = extractelement <2 x double> %63, i32 1
  %64 = getelementptr inbounds i8, ptr addrspace(1) %42, i64 16
  %.elt9 = select i1 %51, double %.unpack630, double 0.000000e+00
  %.elt11 = select i1 %51, double %.unpack831, double 0.000000e+00
  %65 = insertelement <2 x double> poison, double %.elt9, i32 0
  %66 = insertelement <2 x double> %65, double %.elt11, i32 1
  store <2 x double> %66, ptr addrspace(1) %64, align 16
  %67 = or disjoint i32 %18, 2
  %68 = zext nneg i32 %67 to i64
  %69 = sub nsw i64 %68, %11
  %70 = icmp ult i64 %69, 12
  %71 = sub nsw i64 %68, %13
  %72 = icmp ult i64 %71, 12
  %73 = and i1 %72, %70
  %74 = tail call i64 @llvm.smax.i64(i64 %69, i64 0)
  %75 = tail call i64 @llvm.umin.i64(i64 %74, i64 11)
  %76 = trunc nuw nsw i64 %75 to i32
  %77 = tail call i64 @llvm.smax.i64(i64 %71, i64 0)
  %78 = tail call i64 @llvm.umin.i64(i64 %77, i64 11)
  %79 = trunc nuw nsw i64 %78 to i32
  %80 = mul nuw nsw i32 %76, 12
  %81 = or disjoint i32 %31, %79
  %82 = add nuw nsw i32 %81, %80
  %83 = zext nneg i32 %82 to i64
  %84 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %83
  %85 = load <2 x double>, ptr addrspace(1) %84, align 16, !invariant.load !4
  %.unpack1328 = extractelement <2 x double> %85, i32 0
  %.unpack1529 = extractelement <2 x double> %85, i32 1
  %86 = getelementptr inbounds i8, ptr addrspace(1) %42, i64 32
  %.elt16 = select i1 %73, double %.unpack1328, double 0.000000e+00
  %.elt18 = select i1 %73, double %.unpack1529, double 0.000000e+00
  %87 = insertelement <2 x double> poison, double %.elt16, i32 0
  %88 = insertelement <2 x double> %87, double %.elt18, i32 1
  store <2 x double> %88, ptr addrspace(1) %86, align 32
  %89 = or disjoint i32 %18, 3
  %90 = zext nneg i32 %89 to i64
  %91 = sub nsw i64 %90, %11
  %92 = icmp ult i64 %91, 12
  %93 = sub nsw i64 %90, %13
  %94 = icmp ult i64 %93, 12
  %95 = and i1 %94, %92
  %96 = tail call i64 @llvm.smax.i64(i64 %91, i64 0)
  %97 = tail call i64 @llvm.umin.i64(i64 %96, i64 11)
  %98 = trunc nuw nsw i64 %97 to i32
  %99 = tail call i64 @llvm.smax.i64(i64 %93, i64 0)
  %100 = tail call i64 @llvm.umin.i64(i64 %99, i64 11)
  %101 = trunc nuw nsw i64 %100 to i32
  %102 = mul nuw nsw i32 %98, 12
  %103 = or disjoint i32 %31, %101
  %104 = add nuw nsw i32 %103, %102
  %105 = zext nneg i32 %104 to i64
  %106 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %105
  %107 = load <2 x double>, ptr addrspace(1) %106, align 16, !invariant.load !4
  %.unpack2026 = extractelement <2 x double> %107, i32 0
  %.unpack2227 = extractelement <2 x double> %107, i32 1
  %108 = getelementptr inbounds i8, ptr addrspace(1) %42, i64 48
  %.elt23 = select i1 %95, double %.unpack2026, double 0.000000e+00
  %.elt25 = select i1 %95, double %.unpack2227, double 0.000000e+00
  %109 = insertelement <2 x double> poison, double %.elt23, i32 0
  %110 = insertelement <2 x double> %109, double %.elt25, i32 1
  store <2 x double> %110, ptr addrspace(1) %108, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 504}
!3 = !{i32 0, i32 128}
!4 = !{}
