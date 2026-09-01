; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_divide_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(524288) %0, ptr noalias readonly align 16 captures(none) dereferenceable(262144) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(524288) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %11
  %13 = load double, ptr addrspace(1) %12, align 8, !invariant.load !6
  %14 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %11
  %15 = load <2 x double>, ptr addrspace(1) %14, align 16, !invariant.load !6
  %.unpack5 = extractelement <2 x double> %15, i32 0
  %.unpack26 = extractelement <2 x double> %15, i32 1
  %16 = fdiv double %13, 0.000000e+00
  %17 = fmul double %13, %16
  %18 = fmul double %.unpack5, %16
  %19 = fadd double %18, %.unpack26
  %20 = fdiv double %19, %17
  %21 = fmul double %16, %.unpack26
  %22 = fsub double %21, %.unpack5
  %23 = fdiv double %22, %17
  %24 = fdiv double 0.000000e+00, %13
  %25 = fmul double %24, 0.000000e+00
  %26 = fadd double %13, %25
  %27 = fmul double %24, %.unpack26
  %28 = fadd double %.unpack5, %27
  %29 = fdiv double %28, %26
  %30 = fmul double %.unpack5, %24
  %31 = fsub double %.unpack26, %30
  %32 = fdiv double %31, %26
  %33 = tail call double @llvm.nvvm.fabs.f64(double %13)
  %34 = fcmp oeq double %33, 0.000000e+00
  %35 = fcmp ord double %.unpack5, 0.000000e+00
  %36 = fcmp ord double %.unpack26, 0.000000e+00
  %37 = or i1 %35, %36
  %38 = and i1 %34, %37
  %39 = tail call double @llvm.copysign.f64(double 0x7FF0000000000000, double %13)
  %40 = fmul double %39, %.unpack5
  %41 = fmul double %39, %.unpack26
  %42 = fcmp one double %33, 0x7FF0000000000000
  %43 = tail call double @llvm.nvvm.fabs.f64(double %.unpack5)
  %44 = fcmp oeq double %43, 0x7FF0000000000000
  %45 = tail call double @llvm.nvvm.fabs.f64(double %.unpack26)
  %46 = fcmp oeq double %45, 0x7FF0000000000000
  %47 = or i1 %44, %46
  %48 = and i1 %42, %47
  %49 = select i1 %44, double 1.000000e+00, double 0.000000e+00
  %50 = tail call double @llvm.copysign.f64(double %49, double %.unpack5)
  %51 = select i1 %46, double 1.000000e+00, double 0.000000e+00
  %52 = tail call double @llvm.copysign.f64(double %51, double %.unpack26)
  %53 = fmul double %13, %50
  %54 = tail call double @llvm.copysign.f64(double 0.000000e+00, double %.unpack26)
  %55 = fadd double %54, %53
  %56 = fmul double %55, 0x7FF0000000000000
  %57 = tail call double @llvm.copysign.f64(double 0.000000e+00, double %.unpack5)
  %58 = fmul double %13, %52
  %59 = fsub double %58, %57
  %60 = fmul double %59, 0x7FF0000000000000
  %61 = fcmp one double %43, 0x7FF0000000000000
  %62 = fcmp one double %45, 0x7FF0000000000000
  %63 = and i1 %61, %62
  %64 = fcmp oeq double %33, 0x7FF0000000000000
  %65 = and i1 %64, %63
  %66 = select i1 %64, double 1.000000e+00, double 0.000000e+00
  %67 = tail call double @llvm.copysign.f64(double %66, double %13)
  %68 = fmul double %.unpack5, %67
  %69 = fmul double %.unpack26, 0.000000e+00
  %70 = fadd double %69, %68
  %71 = fmul double %70, 0.000000e+00
  %72 = fmul double %.unpack26, %67
  %73 = fmul double %.unpack5, 0.000000e+00
  %74 = fsub double %72, %73
  %75 = fmul double %74, 0.000000e+00
  %76 = fcmp olt double %33, 0.000000e+00
  %77 = select i1 %76, double %20, double %29
  %78 = select i1 %76, double %23, double %32
  %79 = select i1 %65, double %71, double %77
  %80 = select i1 %65, double %75, double %78
  %81 = select i1 %48, double %56, double %79
  %82 = select i1 %48, double %60, double %80
  %83 = select i1 %38, double %40, double %81
  %84 = select i1 %38, double %41, double %82
  %85 = fcmp uno double %77, 0.000000e+00
  %86 = fcmp uno double %78, 0.000000e+00
  %87 = and i1 %85, %86
  %88 = select i1 %87, double %83, double %77
  %89 = select i1 %87, double %84, double %78
  %90 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %11
  %91 = insertelement <2 x double> poison, double %88, i32 0
  %92 = insertelement <2 x double> %91, double %89, i32 1
  store <2 x double> %92, ptr addrspace(1) %90, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.copysign.f64(double, double) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 256}
!5 = !{i32 0, i32 128}
!6 = !{}
