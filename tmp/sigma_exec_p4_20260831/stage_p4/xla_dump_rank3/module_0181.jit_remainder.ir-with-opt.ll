; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(256) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(256) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %8 = zext nneg i32 %7 to i64
  %9 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %8
  %10 = load double, ptr addrspace(1) %9, align 8, !invariant.load !5
  %11 = load i64, ptr addrspace(1) %5, align 16, !invariant.load !5
  %12 = sitofp i64 %11 to double
  %13 = tail call i32 @llvm.nvvm.d2i.hi(double %10) #3
  %14 = and i32 %13, 2147483647
  %15 = tail call i32 @llvm.nvvm.d2i.lo(double %10) #3
  %16 = tail call i32 @llvm.nvvm.d2i.hi(double %12) #3
  %17 = and i32 %16, 2147483647
  %18 = tail call i32 @llvm.nvvm.d2i.lo(double %12) #3
  %19 = tail call double @llvm.nvvm.lohi.i2d(i32 %15, i32 %14) #3
  %20 = tail call double @llvm.nvvm.lohi.i2d(i32 %18, i32 %17) #3
  %21 = icmp samesign ugt i32 %14, 2146435071
  %22 = icmp samesign ugt i32 %17, 2146435071
  %or.cond.i = select i1 %21, i1 true, i1 %22
  br i1 %or.cond.i, label %23, label %30

23:                                               ; preds = %3
  %24 = fcmp ord double %19, 0.000000e+00
  %25 = fcmp ord double %20, 0.000000e+00
  %or.cond2.i = select i1 %24, i1 %25, i1 false
  br i1 %or.cond2.i, label %28, label %26

26:                                               ; preds = %23
  %27 = tail call double @llvm.nvvm.add.rn.d(double %10, double %12) #3
  br label %__nv_fmod.exit

28:                                               ; preds = %23
  %29 = fcmp oeq double %19, 0x7FF0000000000000
  %.7.i = select i1 %29, double 0xFFF8000000000000, double %10
  br label %__nv_fmod.exit

30:                                               ; preds = %3
  %31 = fcmp oeq double %20, 0.000000e+00
  br i1 %31, label %__nv_fmod.exit, label %32

32:                                               ; preds = %30
  %33 = fcmp ult double %19, %20
  br i1 %33, label %__nv_fmod.exit, label %34

34:                                               ; preds = %32
  %35 = tail call i32 @llvm.nvvm.d2i.hi(double %19) #3
  %36 = lshr i32 %35, 20
  %37 = tail call i32 @llvm.nvvm.d2i.hi(double %20) #3
  %38 = lshr i32 %37, 20
  %39 = icmp eq i32 %36, 0
  %40 = fmul double %19, 0x4350000000000000
  %41 = tail call i32 @llvm.nvvm.d2i.hi(double %40) #3
  %42 = lshr i32 %41, 20
  %43 = add nsw i32 %42, -54
  %.9.i = select i1 %39, double %40, double %19
  %expoa.0.i = select i1 %39, i32 %43, i32 %36
  %44 = icmp eq i32 %38, 0
  %45 = fmul double %20, 0x4350000000000000
  %46 = tail call i32 @llvm.nvvm.d2i.hi(double %45) #3
  %47 = lshr i32 %46, 20
  %48 = add nsw i32 %47, -54
  %.0.i = select i1 %44, double %45, double %20
  %expob.0.i = select i1 %44, i32 %48, i32 %38
  %49 = bitcast double %.9.i to i64
  %50 = bitcast double %.0.i to i64
  %51 = and i64 %49, 4503599627370495
  %52 = or disjoint i64 %51, 4503599627370496
  %53 = and i64 %50, 4503599627370495
  %54 = or disjoint i64 %53, 4503599627370496
  %55 = add i32 %expoa.0.i, 1
  %56 = sub i32 %55, %expob.0.i
  br label %57

57:                                               ; preds = %57, %34
  %lsr.iv = phi i32 [ %lsr.iv.next, %57 ], [ %56, %34 ]
  %ia.0.i = phi i64 [ %52, %34 ], [ %62, %57 ]
  %58 = sub i64 %ia.0.i, %54
  %59 = bitcast i64 %58 to double
  %60 = tail call i32 @llvm.nvvm.d2i.hi(double %59) #3
  %61 = icmp slt i32 %60, 0
  %spec.select.i = select i1 %61, i64 %ia.0.i, i64 %58
  %62 = shl i64 %spec.select.i, 1
  %lsr.iv.next = add i32 %lsr.iv, -1
  %63 = icmp sgt i32 %lsr.iv.next, 0
  br i1 %63, label %57, label %64

64:                                               ; preds = %57
  %65 = and i64 %spec.select.i, 9223372036854775807
  %.not.i = icmp eq i64 %65, 0
  br i1 %.not.i, label %85, label %66

66:                                               ; preds = %64
  %67 = bitcast i64 %65 to double
  %68 = fmul double %67, 0x4350000000000000
  %69 = tail call i32 @llvm.nvvm.d2i.hi(double %68) #3
  %70 = lshr i32 %69, 20
  %71 = sub nsw i32 55, %70
  %72 = sub nsw i32 %expob.0.i, %71
  %73 = zext nneg i32 %71 to i64
  %74 = shl i64 %65, %73
  %75 = icmp slt i32 %72, 1
  br i1 %75, label %76, label %80

76:                                               ; preds = %66
  %77 = sub nsw i32 1, %72
  %78 = zext nneg i32 %77 to i64
  %79 = lshr i64 %74, %78
  br label %85

80:                                               ; preds = %66
  %81 = add nuw nsw i32 %72, 4095
  %82 = zext nneg i32 %81 to i64
  %83 = shl i64 %82, 52
  %84 = add i64 %83, %74
  br label %85

85:                                               ; preds = %80, %76, %64
  %ia.3.i = phi i64 [ 0, %64 ], [ %79, %76 ], [ %84, %80 ]
  %86 = bitcast i64 %ia.3.i to double
  %87 = tail call double @llvm.copysign.f64(double %86, double %10) #3
  br label %__nv_fmod.exit

__nv_fmod.exit:                                   ; preds = %26, %28, %30, %32, %85
  %.12.i = phi double [ %27, %26 ], [ %.7.i, %28 ], [ %10, %32 ], [ %87, %85 ], [ 0xFFF8000000000000, %30 ]
  %88 = fcmp olt double %.12.i, 0.000000e+00
  %89 = icmp slt i64 %11, 0
  %90 = xor i1 %89, %88
  %91 = fcmp une double %.12.i, 0.000000e+00
  %92 = and i1 %91, %90
  %93 = fadd double %.12.i, %12
  %94 = select i1 %92, double %93, double %.12.i
  %95 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %8
  store double %94, ptr addrspace(1) %95, align 8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.hi(double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.lo(double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.lohi.i2d(i32, i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.copysign.f64(double, double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.add.rn.d(double, double) #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "nvvm.reqntid"="32,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 32}
!5 = !{}
