; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_complex_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(16) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(16) initializes((0, 16)) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = load <2 x double>, ptr addrspace(1) %3, align 16, !invariant.load !4
  %.unpack5 = extractelement <2 x double> %5, i32 0
  %.unpack26 = extractelement <2 x double> %5, i32 1
  %6 = tail call double @llvm.nvvm.fabs.f64(double %.unpack5)
  %7 = tail call double @llvm.nvvm.fabs.f64(double %.unpack26)
  %8 = fcmp olt double %6, %7
  %9 = fdiv double %6, %7
  %10 = fdiv double %7, %6
  %11 = fcmp oeq double %6, %7
  %12 = select i1 %8, double %9, double %10
  %13 = select i1 %11, double 1.000000e+00, double %12
  %14 = tail call double @llvm.nvvm.fabs.f64(double %13)
  %15 = tail call double @llvm.minimum.f64(double %14, double 1.000000e+00)
  %16 = tail call double @llvm.maximum.f64(double %14, double 1.000000e+00)
  %17 = fdiv double %15, %16
  %18 = fmul double %17, %17
  %19 = fmul double %16, %18
  %20 = fadd double %18, 1.000000e+00
  %21 = tail call double @llvm.minimum.f64(double %6, double %7)
  %22 = tail call double @llvm.maximum.f64(double %6, double %7)
  %23 = fdiv double %21, %22
  %24 = tail call double @llvm.sqrt.f64(double %20)
  %25 = fcmp oeq double %24, 1.000000e+00
  %26 = fcmp ogt double %18, 0.000000e+00
  %27 = fmul double %19, 5.000000e-01
  %28 = fmul double %23, %23
  %29 = fmul double %22, %28
  %30 = fadd double %28, 1.000000e+00
  %31 = and i1 %26, %25
  %32 = fadd double %16, %27
  %33 = fmul double %16, %24
  %34 = tail call double @llvm.sqrt.f64(double %30)
  %35 = fcmp oeq double %34, 1.000000e+00
  %36 = fcmp ogt double %28, 0.000000e+00
  %37 = fmul double %29, 5.000000e-01
  %38 = fcmp oeq double %16, %15
  %39 = fmul double %16, 0x3FF6A09E667F3BCD
  %40 = select i1 %31, double %32, double %33
  %41 = and i1 %36, %35
  %42 = fadd double %22, %37
  %43 = fmul double %22, %34
  %44 = tail call double @llvm.sqrt.f64(double %6)
  %45 = tail call double @llvm.sqrt.f64(double %7)
  %46 = fdiv double %44, %45
  %47 = fdiv double %45, %44
  %48 = select i1 %38, double %39, double %40
  %49 = fcmp oeq double %22, %21
  %50 = fmul double %22, 0x3FF6A09E667F3BCD
  %51 = select i1 %41, double %42, double %43
  %52 = select i1 %8, double %46, double %47
  %53 = fadd double %48, 1.000000e+00
  %54 = fadd double %13, %48
  %55 = select i1 %49, double %50, double %51
  %56 = select i1 %11, double 1.000000e+00, double %52
  %57 = tail call double @llvm.sqrt.f64(double %53)
  %58 = tail call double @llvm.sqrt.f64(double %54)
  %59 = fmul double %57, 0x3FE6A09E667F3BCC
  %60 = fmul double %58, 0x3FE6A09E667F3BCC
  %61 = fadd double %6, %55
  %62 = fmul double %45, %56
  %63 = fmul double %57, 0x3FF6A09E667F3BCD
  %64 = fmul double %58, 0x3FF6A09E667F3BCD
  %65 = fmul double %44, %59
  %66 = fmul double %45, %60
  %67 = fmul double %61, 5.000000e-01
  %68 = tail call double @llvm.sqrt.f64(double %67)
  %69 = fcmp ogt double %6, %7
  %70 = fdiv double %62, %63
  %71 = fdiv double %45, %64
  %72 = fmul double %68, 2.000000e+00
  %73 = select i1 %69, double %65, double %66
  %74 = tail call i1 @llvm.is.fpclass.f64(double %68, i32 608)
  %75 = select i1 %69, double %70, double %71
  %76 = fdiv double %7, %72
  %77 = fmul double %44, 0x3FF19435CAFFA9F8
  %78 = select i1 %74, double %73, double %68
  %79 = fmul double %45, 0x3FDD203138F6C828
  %80 = select i1 %74, double %75, double %76
  %81 = select i1 %11, double %77, double %78
  %82 = fneg double %81
  %83 = fcmp olt double %.unpack26, 0.000000e+00
  %84 = select i1 %11, double %79, double %80
  %85 = fneg double %84
  %86 = fcmp oge double %.unpack5, 0.000000e+00
  %87 = fcmp olt double %.unpack5, 0.000000e+00
  %88 = select i1 %83, double %82, double %81
  %89 = select i1 %83, double %85, double %84
  %90 = select i1 %86, double %81, double %84
  %91 = select i1 %87, double %88, double %89
  %92 = insertelement <2 x double> poison, double %90, i32 0
  %93 = insertelement <2 x double> %92, double %91, i32 1
  store <2 x double> %93, ptr addrspace(1) %4, align 256
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f64(double, i32 immarg) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{}
