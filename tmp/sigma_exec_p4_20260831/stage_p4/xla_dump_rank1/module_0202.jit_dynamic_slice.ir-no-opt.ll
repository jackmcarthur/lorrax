; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_dynamic_slice(ptr noalias align 16 dereferenceable(72) %0, ptr noalias align 16 dereferenceable(8) %1, ptr noalias align 16 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3) #0 {
  %5 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %6 = load i64, ptr %5, align 4, !invariant.load !1
  %7 = call i64 @llvm.smin.i64(i64 %6, i64 2)
  %8 = call i64 @llvm.smax.i64(i64 %7, i64 0)
  %9 = getelementptr inbounds [1 x i64], ptr %2, i32 0, i32 0
  %10 = load i64, ptr %9, align 4, !invariant.load !1
  %11 = call i64 @llvm.smin.i64(i64 %10, i64 2)
  %12 = call i64 @llvm.smax.i64(i64 %11, i64 0)
  %13 = mul i64 %8, 3
  %14 = add i64 %13, %12
  %15 = getelementptr inbounds [9 x double], ptr %0, i32 0, i64 %14
  %16 = load double, ptr %15, align 8, !invariant.load !1
  %17 = getelementptr inbounds [1 x double], ptr %3, i32 0, i32 0
  store double %16, ptr %17, align 8
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #1

attributes #0 = { "nvvm.reqntid"="1,1,1" }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
