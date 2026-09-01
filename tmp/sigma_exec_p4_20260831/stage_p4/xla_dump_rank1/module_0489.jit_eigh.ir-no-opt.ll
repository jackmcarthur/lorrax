; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef
@shared_01 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @input_transpose_fusion_1(ptr noalias align 16 dereferenceable(4718592) %0, ptr noalias align 256 dereferenceable(4718592) %1) #0 {
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
  br i1 %6, label %44, label %182

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
  %54 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %53
  %55 = load { double, double }, ptr %54, align 8, !invariant.load !3
  %56 = extractvalue { double, double } %55, 1
  %57 = extractvalue { double, double } %55, 0
  %58 = fneg double %56
  %59 = extractvalue { double, double } %49, 0
  %60 = fadd double %59, %57
  %61 = extractvalue { double, double } %49, 1
  %62 = fadd double %61, %58
  %63 = fmul double %60, 5.000000e-01
  %64 = fmul double %62, 0.000000e+00
  %65 = fsub double %63, %64
  %66 = fmul double %62, 5.000000e-01
  %67 = fmul double %60, 0.000000e+00
  %68 = fadd double %66, %67
  %69 = insertvalue { double, double } poison, double %65, 0
  %70 = insertvalue { double, double } %69, double %68, 1
  %71 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %53
  store { double, double } %70, ptr %71, align 8
  %72 = add i32 %47, 132
  %73 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %72
  %74 = load { double, double }, ptr %73, align 8
  %75 = add i32 %53, 96
  %76 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %75
  %77 = load { double, double }, ptr %76, align 8, !invariant.load !3
  %78 = extractvalue { double, double } %77, 1
  %79 = extractvalue { double, double } %77, 0
  %80 = fneg double %78
  %81 = extractvalue { double, double } %74, 0
  %82 = fadd double %81, %79
  %83 = extractvalue { double, double } %74, 1
  %84 = fadd double %83, %80
  %85 = fmul double %82, 5.000000e-01
  %86 = fmul double %84, 0.000000e+00
  %87 = fsub double %85, %86
  %88 = fmul double %84, 5.000000e-01
  %89 = fmul double %82, 0.000000e+00
  %90 = fadd double %88, %89
  %91 = insertvalue { double, double } poison, double %87, 0
  %92 = insertvalue { double, double } %91, double %90, 1
  %93 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %75
  store { double, double } %92, ptr %93, align 8
  %94 = add i32 %47, 264
  %95 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %94
  %96 = load { double, double }, ptr %95, align 8
  %97 = add i32 %53, 192
  %98 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %97
  %99 = load { double, double }, ptr %98, align 8, !invariant.load !3
  %100 = extractvalue { double, double } %99, 1
  %101 = extractvalue { double, double } %99, 0
  %102 = fneg double %100
  %103 = extractvalue { double, double } %96, 0
  %104 = fadd double %103, %101
  %105 = extractvalue { double, double } %96, 1
  %106 = fadd double %105, %102
  %107 = fmul double %104, 5.000000e-01
  %108 = fmul double %106, 0.000000e+00
  %109 = fsub double %107, %108
  %110 = fmul double %106, 5.000000e-01
  %111 = fmul double %104, 0.000000e+00
  %112 = fadd double %110, %111
  %113 = insertvalue { double, double } poison, double %109, 0
  %114 = insertvalue { double, double } %113, double %112, 1
  %115 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %97
  store { double, double } %114, ptr %115, align 8
  %116 = add i32 %47, 396
  %117 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %116
  %118 = load { double, double }, ptr %117, align 8
  %119 = add i32 %53, 288
  %120 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %119
  %121 = load { double, double }, ptr %120, align 8, !invariant.load !3
  %122 = extractvalue { double, double } %121, 1
  %123 = extractvalue { double, double } %121, 0
  %124 = fneg double %122
  %125 = extractvalue { double, double } %118, 0
  %126 = fadd double %125, %123
  %127 = extractvalue { double, double } %118, 1
  %128 = fadd double %127, %124
  %129 = fmul double %126, 5.000000e-01
  %130 = fmul double %128, 0.000000e+00
  %131 = fsub double %129, %130
  %132 = fmul double %128, 5.000000e-01
  %133 = fmul double %126, 0.000000e+00
  %134 = fadd double %132, %133
  %135 = insertvalue { double, double } poison, double %131, 0
  %136 = insertvalue { double, double } %135, double %134, 1
  %137 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %119
  store { double, double } %136, ptr %137, align 8
  %138 = add i32 %47, 528
  %139 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %138
  %140 = load { double, double }, ptr %139, align 8
  %141 = add i32 %53, 384
  %142 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %141
  %143 = load { double, double }, ptr %142, align 8, !invariant.load !3
  %144 = extractvalue { double, double } %143, 1
  %145 = extractvalue { double, double } %143, 0
  %146 = fneg double %144
  %147 = extractvalue { double, double } %140, 0
  %148 = fadd double %147, %145
  %149 = extractvalue { double, double } %140, 1
  %150 = fadd double %149, %146
  %151 = fmul double %148, 5.000000e-01
  %152 = fmul double %150, 0.000000e+00
  %153 = fsub double %151, %152
  %154 = fmul double %150, 5.000000e-01
  %155 = fmul double %148, 0.000000e+00
  %156 = fadd double %154, %155
  %157 = insertvalue { double, double } poison, double %153, 0
  %158 = insertvalue { double, double } %157, double %156, 1
  %159 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %141
  store { double, double } %158, ptr %159, align 8
  %160 = add i32 %47, 660
  %161 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %160
  %162 = load { double, double }, ptr %161, align 8
  %163 = add i32 %53, 480
  %164 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %163
  %165 = load { double, double }, ptr %164, align 8, !invariant.load !3
  %166 = extractvalue { double, double } %165, 1
  %167 = extractvalue { double, double } %165, 0
  %168 = fneg double %166
  %169 = extractvalue { double, double } %162, 0
  %170 = fadd double %169, %167
  %171 = extractvalue { double, double } %162, 1
  %172 = fadd double %171, %168
  %173 = fmul double %170, 5.000000e-01
  %174 = fmul double %172, 0.000000e+00
  %175 = fsub double %173, %174
  %176 = fmul double %172, 5.000000e-01
  %177 = fmul double %170, 0.000000e+00
  %178 = fadd double %176, %177
  %179 = insertvalue { double, double } poison, double %175, 0
  %180 = insertvalue { double, double } %179, double %178, 1
  %181 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %163
  store { double, double } %180, ptr %181, align 8
  br label %182

182:                                              ; preds = %44, %43
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

define ptx_kernel void @loop_select_fusion(ptr noalias align 256 dereferenceable(98304) %0, ptr noalias align 256 dereferenceable(2048) %1, ptr noalias align 256 dereferenceable(98304) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = udiv i32 %7, 24
  %9 = getelementptr inbounds [512 x i32], ptr %1, i32 0, i32 %8
  %10 = load i32, ptr %9, align 4, !invariant.load !3
  %11 = icmp eq i32 %10, 0
  %12 = getelementptr inbounds [12288 x double], ptr %0, i32 0, i32 %7
  %13 = load double, ptr %12, align 8
  %14 = select i1 %11, double %13, double 0x7FF8000000000000
  store double %14, ptr %12, align 8
  ret void
}

define ptx_kernel void @input_transpose_fusion(ptr noalias align 256 dereferenceable(4718592) %0, ptr noalias align 256 dereferenceable(2048) %1, ptr noalias align 256 dereferenceable(4718592) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %6 = urem i32 %4, 32
  %7 = icmp sle i32 %6, 23
  br i1 %7, label %8, label %53

8:                                                ; preds = %3
  %9 = getelementptr inbounds [512 x i32], ptr %1, i32 0, i32 %5
  %10 = load i32, ptr %9, align 4, !invariant.load !3
  %11 = icmp eq i32 %10, 0
  %12 = udiv i32 %4, 32
  %13 = mul i32 %12, 24
  %14 = mul i32 %5, 576
  %15 = add i32 %13, %14
  %16 = add i32 %15, %6
  %17 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %16
  %18 = load { double, double }, ptr %17, align 8, !invariant.load !3
  %19 = select i1 %11, { double, double } %18, { double, double } { double 0x7FF8000000000000, double 0x7FF8000000000000 }
  %20 = mul i32 %6, 33
  %21 = add i32 %20, %12
  %22 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %21
  store { double, double } %19, ptr %22, align 8
  %23 = add i32 %16, 96
  %24 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %23
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !3
  %26 = select i1 %11, { double, double } %25, { double, double } { double 0x7FF8000000000000, double 0x7FF8000000000000 }
  %27 = add i32 %21, 4
  %28 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %27
  store { double, double } %26, ptr %28, align 8
  %29 = add i32 %16, 192
  %30 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %29
  %31 = load { double, double }, ptr %30, align 8, !invariant.load !3
  %32 = select i1 %11, { double, double } %31, { double, double } { double 0x7FF8000000000000, double 0x7FF8000000000000 }
  %33 = add i32 %21, 8
  %34 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %33
  store { double, double } %32, ptr %34, align 8
  %35 = add i32 %16, 288
  %36 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %35
  %37 = load { double, double }, ptr %36, align 8, !invariant.load !3
  %38 = select i1 %11, { double, double } %37, { double, double } { double 0x7FF8000000000000, double 0x7FF8000000000000 }
  %39 = add i32 %21, 12
  %40 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %39
  store { double, double } %38, ptr %40, align 8
  %41 = add i32 %16, 384
  %42 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %41
  %43 = load { double, double }, ptr %42, align 8, !invariant.load !3
  %44 = select i1 %11, { double, double } %43, { double, double } { double 0x7FF8000000000000, double 0x7FF8000000000000 }
  %45 = add i32 %21, 16
  %46 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %45
  store { double, double } %44, ptr %46, align 8
  %47 = add i32 %16, 480
  %48 = getelementptr inbounds [294912 x { double, double }], ptr %0, i32 0, i32 %47
  %49 = load { double, double }, ptr %48, align 8, !invariant.load !3
  %50 = select i1 %11, { double, double } %49, { double, double } { double 0x7FF8000000000000, double 0x7FF8000000000000 }
  %51 = add i32 %21, 20
  %52 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %51
  store { double, double } %50, ptr %52, align 8
  br label %53

53:                                               ; preds = %8, %3
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br i1 %7, label %54, label %90

54:                                               ; preds = %53
  %55 = udiv i32 %4, 32
  %56 = mul i32 %55, 33
  %57 = add i32 %56, %6
  %58 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %57
  %59 = load { double, double }, ptr %58, align 8
  %60 = mul i32 %55, 24
  %61 = mul i32 %5, 576
  %62 = add i32 %60, %61
  %63 = add i32 %62, %6
  %64 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %63
  store { double, double } %59, ptr %64, align 8
  %65 = add i32 %57, 132
  %66 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %65
  %67 = load { double, double }, ptr %66, align 8
  %68 = add i32 %63, 96
  %69 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %68
  store { double, double } %67, ptr %69, align 8
  %70 = add i32 %57, 264
  %71 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %70
  %72 = load { double, double }, ptr %71, align 8
  %73 = add i32 %63, 192
  %74 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %73
  store { double, double } %72, ptr %74, align 8
  %75 = add i32 %57, 396
  %76 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %75
  %77 = load { double, double }, ptr %76, align 8
  %78 = add i32 %63, 288
  %79 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %78
  store { double, double } %77, ptr %79, align 8
  %80 = add i32 %57, 528
  %81 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %80
  %82 = load { double, double }, ptr %81, align 8
  %83 = add i32 %63, 384
  %84 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %83
  store { double, double } %82, ptr %84, align 8
  %85 = add i32 %57, 660
  %86 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %85
  %87 = load { double, double }, ptr %86, align 8
  %88 = add i32 %63, 480
  %89 = getelementptr inbounds [294912 x { double, double }], ptr %2, i32 0, i32 %88
  store { double, double } %87, ptr %89, align 8
  br label %90

90:                                               ; preds = %54, %53
  ret void
}

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 128}
!2 = !{i32 0, i32 512}
!3 = !{}
!4 = !{i32 0, i32 96}
