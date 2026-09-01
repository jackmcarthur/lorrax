; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_add(ptr noalias align 16 dereferenceable(4718592) %0, ptr noalias align 16 dereferenceable(4718592) %1, ptr noalias align 256 dereferenceable(4718592) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %5, 4
  %7 = mul i32 %4, 512
  %8 = add i32 %6, %7
  %9 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %8
  %10 = load { double, double }, ptr %9, align 8, !invariant.load !3
  %11 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %8
  %12 = load { double, double }, ptr %11, align 8, !invariant.load !3
  %13 = extractvalue { double, double } %10, 0
  %14 = extractvalue { double, double } %12, 0
  %15 = fadd double %13, %14
  %16 = extractvalue { double, double } %10, 1
  %17 = extractvalue { double, double } %12, 1
  %18 = fadd double %16, %17
  %19 = insertvalue { double, double } poison, double %15, 0
  %20 = insertvalue { double, double } %19, double %18, 1
  %21 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %8
  store { double, double } %20, ptr %21, align 8
  %22 = add i32 %8, 1
  %23 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %22
  %24 = load { double, double }, ptr %23, align 8, !invariant.load !3
  %25 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %22
  %26 = load { double, double }, ptr %25, align 8, !invariant.load !3
  %27 = extractvalue { double, double } %24, 0
  %28 = extractvalue { double, double } %26, 0
  %29 = fadd double %27, %28
  %30 = extractvalue { double, double } %24, 1
  %31 = extractvalue { double, double } %26, 1
  %32 = fadd double %30, %31
  %33 = insertvalue { double, double } poison, double %29, 0
  %34 = insertvalue { double, double } %33, double %32, 1
  %35 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %22
  store { double, double } %34, ptr %35, align 8
  %36 = add i32 %8, 2
  %37 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %36
  %38 = load { double, double }, ptr %37, align 8, !invariant.load !3
  %39 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %36
  %40 = load { double, double }, ptr %39, align 8, !invariant.load !3
  %41 = extractvalue { double, double } %38, 0
  %42 = extractvalue { double, double } %40, 0
  %43 = fadd double %41, %42
  %44 = extractvalue { double, double } %38, 1
  %45 = extractvalue { double, double } %40, 1
  %46 = fadd double %44, %45
  %47 = insertvalue { double, double } poison, double %43, 0
  %48 = insertvalue { double, double } %47, double %46, 1
  %49 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %36
  store { double, double } %48, ptr %49, align 8
  %50 = add i32 %8, 3
  %51 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %50
  %52 = load { double, double }, ptr %51, align 8, !invariant.load !3
  %53 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %50
  %54 = load { double, double }, ptr %53, align 8, !invariant.load !3
  %55 = extractvalue { double, double } %52, 0
  %56 = extractvalue { double, double } %54, 0
  %57 = fadd double %55, %56
  %58 = extractvalue { double, double } %52, 1
  %59 = extractvalue { double, double } %54, 1
  %60 = fadd double %58, %59
  %61 = insertvalue { double, double } poison, double %57, 0
  %62 = insertvalue { double, double } %61, double %60, 1
  %63 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %50
  store { double, double } %62, ptr %63, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 576}
!2 = !{i32 0, i32 128}
!3 = !{}
