; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_add_fusion(ptr noalias align 16 dereferenceable(8192) %0, ptr noalias align 16 dereferenceable(256) %1, ptr noalias align 256 dereferenceable(262144) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = urem i32 %5, 32
  %7 = mul i32 %4, 4
  %8 = udiv i32 %5, 32
  %9 = add i32 %7, %8
  %10 = getelementptr inbounds [1024 x double], ptr %0, i32 0, i32 %9
  %11 = load double, ptr %10, align 8, !invariant.load !3
  %12 = getelementptr inbounds [32 x double], ptr %1, i32 0, i32 %6
  %13 = load double, ptr %12, align 8, !invariant.load !3
  %14 = fadd double %11, %13
  %15 = mul i32 %4, 128
  %16 = add i32 %15, %5
  %17 = getelementptr inbounds [32768 x double], ptr %2, i32 0, i32 %16
  store double %14, ptr %17, align 8
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
!1 = !{i32 0, i32 256}
!2 = !{i32 0, i32 128}
!3 = !{}
