; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @wrapped_transpose(ptr noalias align 16 dereferenceable(243793920) %0, ptr noalias align 256 dereferenceable(243793920) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = urem i32 %4, 10
  %6 = mul i32 %5, 32
  %7 = urem i32 %3, 32
  %8 = add i32 %6, %7
  %9 = icmp sle i32 %8, 309
  br i1 %9, label %10, label %58

10:                                               ; preds = %2
  %11 = udiv i32 %4, 10
  %12 = mul i32 %11, 9920
  %13 = add i32 %6, %12
  %14 = udiv i32 %3, 32
  %15 = mul i32 %14, 310
  %16 = add i32 %13, %15
  %17 = add i32 %16, %7
  %18 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %17
  %19 = load { double, double }, ptr %18, align 8, !invariant.load !3
  %20 = mul i32 %7, 33
  %21 = add i32 %20, %14
  %22 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %21
  store { double, double } %19, ptr %22, align 8
  %23 = add i32 %17, 1240
  %24 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %23
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !3
  %26 = add i32 %21, 4
  %27 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %26
  store { double, double } %25, ptr %27, align 8
  %28 = add i32 %17, 2480
  %29 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %28
  %30 = load { double, double }, ptr %29, align 8, !invariant.load !3
  %31 = add i32 %21, 8
  %32 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %31
  store { double, double } %30, ptr %32, align 8
  %33 = add i32 %17, 3720
  %34 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %33
  %35 = load { double, double }, ptr %34, align 8, !invariant.load !3
  %36 = add i32 %21, 12
  %37 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %36
  store { double, double } %35, ptr %37, align 8
  %38 = add i32 %17, 4960
  %39 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %38
  %40 = load { double, double }, ptr %39, align 8, !invariant.load !3
  %41 = add i32 %21, 16
  %42 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %41
  store { double, double } %40, ptr %42, align 8
  %43 = add i32 %17, 6200
  %44 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %43
  %45 = load { double, double }, ptr %44, align 8, !invariant.load !3
  %46 = add i32 %21, 20
  %47 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %46
  store { double, double } %45, ptr %47, align 8
  %48 = add i32 %17, 7440
  %49 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %48
  %50 = load { double, double }, ptr %49, align 8, !invariant.load !3
  %51 = add i32 %21, 24
  %52 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %51
  store { double, double } %50, ptr %52, align 8
  %53 = add i32 %17, 8680
  %54 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %53
  %55 = load { double, double }, ptr %54, align 8, !invariant.load !3
  %56 = add i32 %21, 28
  %57 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %56
  store { double, double } %55, ptr %57, align 8
  br label %58

58:                                               ; preds = %10, %2
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %59 = udiv i32 %3, 32
  %60 = mul i32 %59, 33
  %61 = add i32 %60, %7
  %62 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %61
  %63 = load { double, double }, ptr %62, align 8
  %64 = udiv i32 %4, 10
  %65 = urem i32 %64, 3
  %66 = mul i32 %65, 32
  %67 = udiv i32 %3, 2
  %68 = urem i32 %67, 16
  %69 = mul i32 %68, 2
  %70 = add i32 %66, %69
  %71 = mul i32 %5, 3072
  %72 = add i32 %70, %71
  %73 = udiv i32 %4, 30
  %74 = mul i32 %73, 29760
  %75 = add i32 %72, %74
  %76 = mul i32 %59, 96
  %77 = add i32 %75, %76
  %78 = urem i32 %3, 2
  %79 = add i32 %77, %78
  %80 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %79
  store { double, double } %63, ptr %80, align 8
  %81 = add i32 %61, 132
  %82 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %81
  %83 = load { double, double }, ptr %82, align 8
  %84 = add i32 %79, 384
  %85 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %84
  store { double, double } %83, ptr %85, align 8
  %86 = add i32 %61, 264
  %87 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %86
  %88 = load { double, double }, ptr %87, align 8
  %89 = add i32 %79, 768
  %90 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %89
  store { double, double } %88, ptr %90, align 8
  %91 = add i32 %61, 396
  %92 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %91
  %93 = load { double, double }, ptr %92, align 8
  %94 = add i32 %79, 1152
  %95 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %94
  store { double, double } %93, ptr %95, align 8
  %96 = add i32 %61, 528
  %97 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %96
  %98 = load { double, double }, ptr %97, align 8
  %99 = add i32 %79, 1536
  %100 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %99
  store { double, double } %98, ptr %100, align 8
  %101 = add i32 %6, %59
  %102 = add i32 %101, 20
  %103 = icmp sle i32 %102, 309
  br i1 %103, label %104, label %110

104:                                              ; preds = %58
  %105 = add i32 %61, 660
  %106 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %105
  %107 = load { double, double }, ptr %106, align 8
  %108 = add i32 %79, 1920
  %109 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %108
  store { double, double } %107, ptr %109, align 8
  br label %110

110:                                              ; preds = %104, %58
  %111 = add i32 %101, 24
  %112 = icmp sle i32 %111, 309
  br i1 %112, label %113, label %119

113:                                              ; preds = %110
  %114 = add i32 %61, 792
  %115 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %114
  %116 = load { double, double }, ptr %115, align 8
  %117 = add i32 %79, 2304
  %118 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %117
  store { double, double } %116, ptr %118, align 8
  br label %119

119:                                              ; preds = %113, %110
  %120 = add i32 %101, 28
  %121 = icmp sle i32 %120, 309
  br i1 %121, label %122, label %128

122:                                              ; preds = %119
  %123 = add i32 %61, 924
  %124 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %123
  %125 = load { double, double }, ptr %124, align 8
  %126 = add i32 %79, 2688
  %127 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %126
  store { double, double } %125, ptr %127, align 8
  br label %128

128:                                              ; preds = %122, %119
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
!2 = !{i32 0, i32 15360}
!3 = !{}
