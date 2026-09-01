; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_divide_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(16) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(16) initializes((0, 16)) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = load i64, ptr addrspace(1) %4, align 16, !invariant.load !4
  %8 = sitofp i64 %7 to double
  %9 = load <2 x double>, ptr addrspace(1) %5, align 16, !invariant.load !4
  %.unpack5 = extractelement <2 x double> %9, i32 0
  %.unpack26 = extractelement <2 x double> %9, i32 1
  %10 = fdiv double %.unpack5, %.unpack26
  %11 = fmul double %.unpack5, %10
  %12 = fadd double %.unpack26, %11
  %13 = fmul double %10, %8
  %14 = fadd double %13, 0.000000e+00
  %15 = fdiv double %14, %12
  %16 = fmul double %10, 0.000000e+00
  %17 = fsub double %16, %8
  %18 = fdiv double %17, %12
  %19 = fdiv double %.unpack26, %.unpack5
  %20 = fmul double %.unpack26, %19
  %21 = fadd double %.unpack5, %20
  %22 = fmul double %19, 0.000000e+00
  %23 = fadd double %22, %8
  %24 = fdiv double %23, %21
  %25 = fmul double %19, %8
  %26 = fsub double 0.000000e+00, %25
  %27 = fdiv double %26, %21
  %28 = tail call double @llvm.nvvm.fabs.f64(double %.unpack5)
  %29 = fcmp oeq double %28, 0.000000e+00
  %30 = tail call double @llvm.nvvm.fabs.f64(double %.unpack26)
  %31 = fcmp oeq double %30, 0.000000e+00
  %32 = and i1 %29, %31
  %33 = tail call double @llvm.copysign.f64(double 0x7FF0000000000000, double %.unpack5)
  %34 = fmul double %33, %8
  %35 = fcmp one double %28, 0x7FF0000000000000
  %36 = fcmp one double %30, 0x7FF0000000000000
  %37 = and i1 %35, %36
  %38 = tail call double @llvm.nvvm.fabs.f64(double %8)
  %39 = fcmp oeq double %38, 0x7FF0000000000000
  %40 = and i1 %39, %37
  %41 = select i1 %39, double 1.000000e+00, double 0.000000e+00
  %42 = tail call double @llvm.copysign.f64(double %41, double %8)
  %43 = fmul double %.unpack5, %42
  %44 = fmul double %.unpack26, 0.000000e+00
  %45 = fadd double %44, %43
  %46 = fmul double %45, 0x7FF0000000000000
  %47 = fmul double %.unpack26, %42
  %48 = fmul double %.unpack5, 0.000000e+00
  %49 = fsub double %48, %47
  %50 = fmul double %49, 0x7FF0000000000000
  %51 = fcmp one double %38, 0x7FF0000000000000
  %52 = fcmp oeq double %28, 0x7FF0000000000000
  %53 = fcmp oeq double %30, 0x7FF0000000000000
  %54 = or i1 %52, %53
  %55 = and i1 %51, %54
  %56 = select i1 %52, double 1.000000e+00, double 0.000000e+00
  %57 = tail call double @llvm.copysign.f64(double %56, double %.unpack5)
  %58 = select i1 %53, double 1.000000e+00, double 0.000000e+00
  %59 = tail call double @llvm.copysign.f64(double %58, double %.unpack26)
  %60 = fmul nnan double %57, %8
  %61 = tail call double @llvm.copysign.f64(double 0.000000e+00, double %.unpack26)
  %62 = fadd nnan double %61, %60
  %63 = fmul double %62, 0.000000e+00
  %64 = tail call double @llvm.copysign.f64(double 0.000000e+00, double %.unpack5)
  %65 = fmul nnan double %59, %8
  %66 = fsub nnan double %64, %65
  %67 = fmul double %66, 0.000000e+00
  %68 = fcmp olt double %28, %30
  %69 = select i1 %68, double %15, double %24
  %70 = select i1 %68, double %18, double %27
  %71 = select i1 %55, double %63, double %69
  %72 = select i1 %55, double %67, double %70
  %73 = select i1 %40, double %46, double %71
  %74 = select i1 %40, double %50, double %72
  %75 = select i1 %32, double %34, double %73
  %76 = select i1 %32, double 0x7FF8000000000000, double %74
  %77 = fcmp uno double %69, 0.000000e+00
  %78 = fcmp uno double %70, 0.000000e+00
  %79 = and i1 %77, %78
  %80 = select i1 %79, double %75, double %69
  %81 = select i1 %79, double %76, double %70
  %82 = insertelement <2 x double> poison, double %80, i32 0
  %83 = insertelement <2 x double> %82, double %81, i32 1
  store <2 x double> %83, ptr addrspace(1) %6, align 256
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.copysign.f64(double, double) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{}
