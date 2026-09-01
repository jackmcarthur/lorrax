; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_dynamic_slice(ptr noalias readonly align 16 captures(none) dereferenceable(72) %0, ptr noalias readonly align 16 captures(none) dereferenceable(8) %1, ptr noalias readonly align 16 captures(none) dereferenceable(8) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = addrspacecast ptr %0 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = load i64, ptr addrspace(1) %5, align 16, !invariant.load !2
  %10 = tail call i64 @llvm.smax.i64(i64 %9, i64 0)
  %11 = tail call i64 @llvm.umin.i64(i64 %10, i64 2)
  %12 = load i64, ptr addrspace(1) %6, align 16, !invariant.load !2
  %13 = tail call i64 @llvm.smax.i64(i64 %12, i64 0)
  %14 = tail call i64 @llvm.umin.i64(i64 %13, i64 2)
  %.idx = mul nuw nsw i64 %11, 24
  %15 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 %.idx
  %16 = getelementptr inbounds double, ptr addrspace(1) %15, i64 %14
  %17 = load double, ptr addrspace(1) %16, align 8, !invariant.load !2
  store double %17, ptr addrspace(1) %8, align 256
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
