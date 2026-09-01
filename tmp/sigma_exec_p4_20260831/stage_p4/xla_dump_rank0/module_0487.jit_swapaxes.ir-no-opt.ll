; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @wrapped_transpose(ptr noalias align 16 dereferenceable(4718592) %0, ptr noalias align 256 dereferenceable(4718592) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = urem i32 %3, 32
  %6 = icmp sle i32 %5, 23
  br i1 %6, label %7, label %43

7:                                                ; preds = %2
  %8 = udiv i32 %3, 32
  %9 = mul i32 %8, 24
  %10 = mul i32 %4, 576
  %11 = add i32 %9, %10
  %12 = add i32 %11, %5
  %13 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %12
  %14 = load { double, double }, ptr %13, align 8, !invariant.load !3
  %15 = mul i32 %5, 33
  %16 = add i32 %15, %8
  %17 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %16
  store { double, double } %14, ptr %17, align 8
  %18 = add i32 %12, 96
  %19 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %18
  %20 = load { double, double }, ptr %19, align 8, !invariant.load !3
  %21 = add i32 %16, 4
  %22 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %21
  store { double, double } %20, ptr %22, align 8
  %23 = add i32 %12, 192
  %24 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %23
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !3
  %26 = add i32 %16, 8
  %27 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %26
  store { double, double } %25, ptr %27, align 8
  %28 = add i32 %12, 288
  %29 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %28
  %30 = load { double, double }, ptr %29, align 8, !invariant.load !3
  %31 = add i32 %16, 12
  %32 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %31
  store { double, double } %30, ptr %32, align 8
  %33 = add i32 %12, 384
  %34 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %33
  %35 = load { double, double }, ptr %34, align 8, !invariant.load !3
  %36 = add i32 %16, 16
  %37 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %36
  store { double, double } %35, ptr %37, align 8
  %38 = add i32 %12, 480
  %39 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %38
  %40 = load { double, double }, ptr %39, align 8, !invariant.load !3
  %41 = add i32 %16, 20
  %42 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %41
  store { double, double } %40, ptr %42, align 8
  br label %43

43:                                               ; preds = %7, %2
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br i1 %6, label %44, label %80

44:                                               ; preds = %43
  %45 = udiv i32 %3, 32
  %46 = mul i32 %45, 33
  %47 = add i32 %46, %5
  %48 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %47
  %49 = load { double, double }, ptr %48, align 8
  %50 = mul i32 %45, 24
  %51 = mul i32 %4, 576
  %52 = add i32 %50, %51
  %53 = add i32 %52, %5
  %54 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %53
  store { double, double } %49, ptr %54, align 8
  %55 = add i32 %47, 132
  %56 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %55
  %57 = load { double, double }, ptr %56, align 8
  %58 = add i32 %53, 96
  %59 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %58
  store { double, double } %57, ptr %59, align 8
  %60 = add i32 %47, 264
  %61 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %60
  %62 = load { double, double }, ptr %61, align 8
  %63 = add i32 %53, 192
  %64 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %63
  store { double, double } %62, ptr %64, align 8
  %65 = add i32 %47, 396
  %66 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %65
  %67 = load { double, double }, ptr %66, align 8
  %68 = add i32 %53, 288
  %69 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %68
  store { double, double } %67, ptr %69, align 8
  %70 = add i32 %47, 528
  %71 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %70
  %72 = load { double, double }, ptr %71, align 8
  %73 = add i32 %53, 384
  %74 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %73
  store { double, double } %72, ptr %74, align 8
  %75 = add i32 %47, 660
  %76 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %75
  %77 = load { double, double }, ptr %76, align 8
  %78 = add i32 %53, 480
  %79 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %78
  store { double, double } %77, ptr %79, align 8
  br label %80

80:                                               ; preds = %44, %43
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 128}
!2 = !{i32 0, i32 512}
!3 = !{}
