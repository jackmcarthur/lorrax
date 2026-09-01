; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_gather_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(131072) %0, ptr noalias readonly align 16 captures(none) dereferenceable(784384) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(16777216) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %9 = shl nuw nsw i32 %7, 9
  %10 = and i32 %9, 32256
  %11 = shl nuw nsw i32 %8, 2
  %12 = or disjoint i32 %10, %11
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %13
  %15 = load <4 x i32>, ptr addrspace(1) %14, align 16, !invariant.load !4
  %16 = extractelement <4 x i32> %15, i64 0
  %17 = icmp slt i32 %16, 1532
  %18 = lshr i32 %7, 6
  %19 = mul nuw nsw i32 %18, 1532
  %20 = tail call i32 @llvm.smax.i32(i32 %16, i32 0)
  %21 = add nuw i32 %20, %19
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %22
  br i1 %17, label %24, label %28

24:                                               ; preds = %3
  %25 = load <2 x double>, ptr addrspace(1) %23, align 16, !invariant.load !4
  %.unpack38 = extractelement <2 x double> %25, i32 0
  %.unpack539 = extractelement <2 x double> %25, i32 1
  %26 = insertvalue { double, double } poison, double %.unpack38, 0
  %27 = insertvalue { double, double } %26, double %.unpack539, 1
  br label %28

28:                                               ; preds = %24, %3
  %29 = phi { double, double } [ %27, %24 ], [ zeroinitializer, %3 ]
  %30 = or disjoint i32 %11, %9
  %31 = zext nneg i32 %30 to i64
  %32 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %31
  %.elt = extractvalue { double, double } %29, 0
  %.elt7 = extractvalue { double, double } %29, 1
  %33 = insertelement <2 x double> poison, double %.elt, i32 0
  %34 = insertelement <2 x double> %33, double %.elt7, i32 1
  store <2 x double> %34, ptr addrspace(1) %32, align 64
  %35 = extractelement <4 x i32> %15, i64 1
  %36 = icmp slt i32 %35, 1532
  %37 = tail call i32 @llvm.smax.i32(i32 %35, i32 0)
  %38 = add nuw i32 %37, %19
  %39 = sext i32 %38 to i64
  %40 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %39
  br i1 %36, label %41, label %45

41:                                               ; preds = %28
  %42 = load <2 x double>, ptr addrspace(1) %40, align 16, !invariant.load !4
  %.unpack936 = extractelement <2 x double> %42, i32 0
  %.unpack1137 = extractelement <2 x double> %42, i32 1
  %43 = insertvalue { double, double } poison, double %.unpack936, 0
  %44 = insertvalue { double, double } %43, double %.unpack1137, 1
  br label %45

45:                                               ; preds = %41, %28
  %46 = phi { double, double } [ %44, %41 ], [ zeroinitializer, %28 ]
  %47 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 16
  %.elt12 = extractvalue { double, double } %46, 0
  %.elt14 = extractvalue { double, double } %46, 1
  %48 = insertelement <2 x double> poison, double %.elt12, i32 0
  %49 = insertelement <2 x double> %48, double %.elt14, i32 1
  store <2 x double> %49, ptr addrspace(1) %47, align 16
  %50 = extractelement <4 x i32> %15, i64 2
  %51 = icmp slt i32 %50, 1532
  %52 = tail call i32 @llvm.smax.i32(i32 %50, i32 0)
  %53 = add nuw i32 %52, %19
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %54
  br i1 %51, label %56, label %60

56:                                               ; preds = %45
  %57 = load <2 x double>, ptr addrspace(1) %55, align 16, !invariant.load !4
  %.unpack1634 = extractelement <2 x double> %57, i32 0
  %.unpack1835 = extractelement <2 x double> %57, i32 1
  %58 = insertvalue { double, double } poison, double %.unpack1634, 0
  %59 = insertvalue { double, double } %58, double %.unpack1835, 1
  br label %60

60:                                               ; preds = %56, %45
  %61 = phi { double, double } [ %59, %56 ], [ zeroinitializer, %45 ]
  %62 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 32
  %.elt19 = extractvalue { double, double } %61, 0
  %.elt21 = extractvalue { double, double } %61, 1
  %63 = insertelement <2 x double> poison, double %.elt19, i32 0
  %64 = insertelement <2 x double> %63, double %.elt21, i32 1
  store <2 x double> %64, ptr addrspace(1) %62, align 32
  %65 = extractelement <4 x i32> %15, i64 3
  %66 = icmp slt i32 %65, 1532
  %67 = tail call i32 @llvm.smax.i32(i32 %65, i32 0)
  %68 = add nuw i32 %67, %19
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %69
  br i1 %66, label %71, label %75

71:                                               ; preds = %60
  %72 = load <2 x double>, ptr addrspace(1) %70, align 16, !invariant.load !4
  %.unpack2332 = extractelement <2 x double> %72, i32 0
  %.unpack2533 = extractelement <2 x double> %72, i32 1
  %73 = insertvalue { double, double } poison, double %.unpack2332, 0
  %74 = insertvalue { double, double } %73, double %.unpack2533, 1
  br label %75

75:                                               ; preds = %71, %60
  %76 = phi { double, double } [ %74, %71 ], [ zeroinitializer, %60 ]
  %77 = getelementptr inbounds i8, ptr addrspace(1) %32, i64 48
  %.elt26 = extractvalue { double, double } %76, 0
  %.elt28 = extractvalue { double, double } %76, 1
  %78 = insertelement <2 x double> poison, double %.elt26, i32 0
  %79 = insertelement <2 x double> %78, double %.elt28, i32 1
  store <2 x double> %79, ptr addrspace(1) %77, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 2048}
!3 = !{i32 0, i32 128}
!4 = !{}
