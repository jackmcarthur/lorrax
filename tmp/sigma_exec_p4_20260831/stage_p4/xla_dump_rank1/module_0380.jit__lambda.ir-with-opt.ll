; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_add_fusion(ptr noalias align 16 captures(none) dereferenceable(24772608) %0, ptr noalias readonly align 16 captures(none) dereferenceable(1179648) %1, ptr noalias readonly align 16 captures(none) dereferenceable(336) %2, ptr noalias readnone align 16 captures(none) dereferenceable(24772608) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %10 = udiv i32 %8, 144
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %11
  %13 = load <2 x double>, ptr addrspace(1) %12, align 16, !invariant.load !4
  %.unpack35 = extractelement <2 x double> %13, i32 0
  %.unpack236 = extractelement <2 x double> %13, i32 1
  %14 = shl nuw nsw i32 %8, 7
  %15 = or disjoint i32 %14, %9
  %16 = udiv i32 %15, 3
  %17 = mul i32 %16, 3
  %.decomposed = sub i32 %15, %17
  %18 = shl nuw nsw i32 %.decomposed, 2
  %19 = urem i32 %16, 12
  %20 = mul nuw nsw i32 %19, 12
  %21 = add nuw nsw i32 %20, %18
  %22 = udiv i32 %15, 36
  %23 = and i32 %22, 511
  %24 = mul nuw nsw i32 %23, 144
  %25 = add nuw nsw i32 %21, %24
  %26 = zext nneg i32 %25 to i64
  %27 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %26
  %28 = load <2 x double>, ptr addrspace(1) %27, align 16, !invariant.load !4
  %.unpack337 = extractelement <2 x double> %28, i32 0
  %.unpack538 = extractelement <2 x double> %28, i32 1
  %29 = shl nuw nsw i32 %9, 2
  %30 = shl nuw nsw i32 %8, 9
  %31 = or disjoint i32 %29, %30
  %32 = zext nneg i32 %31 to i64
  %33 = getelementptr inbounds { double, double }, ptr addrspace(1) %7, i64 %32
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16
  %.unpack645 = extractelement <2 x double> %34, i32 0
  %.unpack846 = extractelement <2 x double> %34, i32 1
  %35 = fmul double %.unpack35, %.unpack337
  %36 = fmul double %.unpack236, %.unpack538
  %37 = fsub double %35, %36
  %38 = fmul double %.unpack236, %.unpack337
  %39 = fmul double %.unpack35, %.unpack538
  %40 = fadd double %38, %39
  %41 = fadd double %.unpack645, %37
  %42 = fadd double %.unpack846, %40
  %43 = insertelement <2 x double> poison, double %41, i32 0
  %44 = insertelement <2 x double> %43, double %42, i32 1
  store <2 x double> %44, ptr addrspace(1) %33, align 16
  %45 = getelementptr inbounds i8, ptr addrspace(1) %27, i64 16
  %46 = load <2 x double>, ptr addrspace(1) %45, align 16, !invariant.load !4
  %.unpack1139 = extractelement <2 x double> %46, i32 0
  %.unpack1340 = extractelement <2 x double> %46, i32 1
  %47 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 16
  %48 = load <2 x double>, ptr addrspace(1) %47, align 16
  %.unpack1447 = extractelement <2 x double> %48, i32 0
  %.unpack1648 = extractelement <2 x double> %48, i32 1
  %49 = fmul double %.unpack35, %.unpack1139
  %50 = fmul double %.unpack236, %.unpack1340
  %51 = fsub double %49, %50
  %52 = fmul double %.unpack236, %.unpack1139
  %53 = fmul double %.unpack35, %.unpack1340
  %54 = fadd double %52, %53
  %55 = fadd double %.unpack1447, %51
  %56 = fadd double %.unpack1648, %54
  %57 = insertelement <2 x double> poison, double %55, i32 0
  %58 = insertelement <2 x double> %57, double %56, i32 1
  store <2 x double> %58, ptr addrspace(1) %47, align 16
  %59 = getelementptr inbounds i8, ptr addrspace(1) %27, i64 32
  %60 = load <2 x double>, ptr addrspace(1) %59, align 16, !invariant.load !4
  %.unpack1941 = extractelement <2 x double> %60, i32 0
  %.unpack2142 = extractelement <2 x double> %60, i32 1
  %61 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 32
  %62 = load <2 x double>, ptr addrspace(1) %61, align 16
  %.unpack2249 = extractelement <2 x double> %62, i32 0
  %.unpack2450 = extractelement <2 x double> %62, i32 1
  %63 = fmul double %.unpack35, %.unpack1941
  %64 = fmul double %.unpack236, %.unpack2142
  %65 = fsub double %63, %64
  %66 = fmul double %.unpack236, %.unpack1941
  %67 = fmul double %.unpack35, %.unpack2142
  %68 = fadd double %66, %67
  %69 = fadd double %.unpack2249, %65
  %70 = fadd double %.unpack2450, %68
  %71 = insertelement <2 x double> poison, double %69, i32 0
  %72 = insertelement <2 x double> %71, double %70, i32 1
  store <2 x double> %72, ptr addrspace(1) %61, align 16
  %73 = getelementptr inbounds i8, ptr addrspace(1) %27, i64 48
  %74 = load <2 x double>, ptr addrspace(1) %73, align 16, !invariant.load !4
  %.unpack2743 = extractelement <2 x double> %74, i32 0
  %.unpack2944 = extractelement <2 x double> %74, i32 1
  %75 = getelementptr inbounds i8, ptr addrspace(1) %33, i64 48
  %76 = load <2 x double>, ptr addrspace(1) %75, align 16
  %.unpack3051 = extractelement <2 x double> %76, i32 0
  %.unpack3252 = extractelement <2 x double> %76, i32 1
  %77 = fmul double %.unpack35, %.unpack2743
  %78 = fmul double %.unpack236, %.unpack2944
  %79 = fsub double %77, %78
  %80 = fmul double %.unpack236, %.unpack2743
  %81 = fmul double %.unpack35, %.unpack2944
  %82 = fadd double %80, %81
  %83 = fadd double %.unpack3051, %79
  %84 = fadd double %.unpack3252, %82
  %85 = insertelement <2 x double> poison, double %83, i32 0
  %86 = insertelement <2 x double> %85, double %84, i32 1
  store <2 x double> %86, ptr addrspace(1) %75, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 3024}
!3 = !{i32 0, i32 128}
!4 = !{}
