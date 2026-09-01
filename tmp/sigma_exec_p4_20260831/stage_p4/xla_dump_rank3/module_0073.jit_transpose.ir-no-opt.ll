; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @wrapped_transpose(ptr noalias align 16 dereferenceable(243793920) %0, ptr noalias align 256 dereferenceable(243793920) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = udiv i32 %4, 3
  %6 = urem i32 %5, 10
  %7 = mul i32 %6, 3072
  %8 = udiv i32 %3, 2
  %9 = urem i32 %8, 16
  %10 = mul i32 %9, 2
  %11 = add i32 %7, %10
  %12 = urem i32 %4, 3
  %13 = mul i32 %12, 32
  %14 = add i32 %11, %13
  %15 = udiv i32 %4, 30
  %16 = mul i32 %15, 29760
  %17 = add i32 %14, %16
  %18 = udiv i32 %3, 32
  %19 = mul i32 %18, 96
  %20 = add i32 %17, %19
  %21 = urem i32 %3, 2
  %22 = add i32 %20, %21
  %23 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %22
  %24 = load { double, double }, ptr %23, align 8, !invariant.load !3
  %25 = urem i32 %3, 32
  %26 = mul i32 %25, 33
  %27 = add i32 %26, %18
  %28 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %27
  store { double, double } %24, ptr %28, align 8
  %29 = add i32 %22, 384
  %30 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %29
  %31 = load { double, double }, ptr %30, align 8, !invariant.load !3
  %32 = add i32 %27, 4
  %33 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %32
  store { double, double } %31, ptr %33, align 8
  %34 = add i32 %22, 768
  %35 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %34
  %36 = load { double, double }, ptr %35, align 8, !invariant.load !3
  %37 = add i32 %27, 8
  %38 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %37
  store { double, double } %36, ptr %38, align 8
  %39 = add i32 %22, 1152
  %40 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %39
  %41 = load { double, double }, ptr %40, align 8, !invariant.load !3
  %42 = add i32 %27, 12
  %43 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %42
  store { double, double } %41, ptr %43, align 8
  %44 = add i32 %22, 1536
  %45 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %44
  %46 = load { double, double }, ptr %45, align 8, !invariant.load !3
  %47 = add i32 %27, 16
  %48 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %47
  store { double, double } %46, ptr %48, align 8
  %49 = mul i32 %6, 32
  %50 = add i32 %49, %18
  %51 = add i32 %50, 20
  %52 = icmp sle i32 %51, 309
  br i1 %52, label %53, label %59

53:                                               ; preds = %2
  %54 = add i32 %22, 1920
  %55 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %54
  %56 = load { double, double }, ptr %55, align 8, !invariant.load !3
  %57 = add i32 %27, 20
  %58 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %57
  store { double, double } %56, ptr %58, align 8
  br label %59

59:                                               ; preds = %53, %2
  %60 = add i32 %50, 24
  %61 = icmp sle i32 %60, 309
  br i1 %61, label %62, label %68

62:                                               ; preds = %59
  %63 = add i32 %22, 2304
  %64 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %63
  %65 = load { double, double }, ptr %64, align 8, !invariant.load !3
  %66 = add i32 %27, 24
  %67 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %66
  store { double, double } %65, ptr %67, align 8
  br label %68

68:                                               ; preds = %62, %59
  %69 = add i32 %50, 28
  %70 = icmp sle i32 %69, 309
  br i1 %70, label %71, label %77

71:                                               ; preds = %68
  %72 = add i32 %22, 2688
  %73 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %72
  %74 = load { double, double }, ptr %73, align 8, !invariant.load !3
  %75 = add i32 %27, 28
  %76 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %75
  store { double, double } %74, ptr %76, align 8
  br label %77

77:                                               ; preds = %71, %68
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %78 = add i32 %49, %25
  %79 = icmp sle i32 %78, 309
  br i1 %79, label %80, label %127

80:                                               ; preds = %77
  %81 = mul i32 %18, 33
  %82 = add i32 %81, %25
  %83 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %82
  %84 = load { double, double }, ptr %83, align 8
  %85 = mul i32 %12, 9920
  %86 = add i32 %49, %85
  %87 = add i32 %86, %16
  %88 = mul i32 %18, 310
  %89 = add i32 %87, %88
  %90 = add i32 %89, %25
  %91 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %90
  store { double, double } %84, ptr %91, align 8
  %92 = add i32 %82, 132
  %93 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %92
  %94 = load { double, double }, ptr %93, align 8
  %95 = add i32 %90, 1240
  %96 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %95
  store { double, double } %94, ptr %96, align 8
  %97 = add i32 %82, 264
  %98 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %97
  %99 = load { double, double }, ptr %98, align 8
  %100 = add i32 %90, 2480
  %101 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %100
  store { double, double } %99, ptr %101, align 8
  %102 = add i32 %82, 396
  %103 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %102
  %104 = load { double, double }, ptr %103, align 8
  %105 = add i32 %90, 3720
  %106 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %105
  store { double, double } %104, ptr %106, align 8
  %107 = add i32 %82, 528
  %108 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %107
  %109 = load { double, double }, ptr %108, align 8
  %110 = add i32 %90, 4960
  %111 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %110
  store { double, double } %109, ptr %111, align 8
  %112 = add i32 %82, 660
  %113 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %112
  %114 = load { double, double }, ptr %113, align 8
  %115 = add i32 %90, 6200
  %116 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %115
  store { double, double } %114, ptr %116, align 8
  %117 = add i32 %82, 792
  %118 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %117
  %119 = load { double, double }, ptr %118, align 8
  %120 = add i32 %90, 7440
  %121 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %120
  store { double, double } %119, ptr %121, align 8
  %122 = add i32 %82, 924
  %123 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %122
  %124 = load { double, double }, ptr %123, align 8
  %125 = add i32 %90, 8680
  %126 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %125
  store { double, double } %124, ptr %126, align 8
  br label %127

127:                                              ; preds = %80, %77
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
