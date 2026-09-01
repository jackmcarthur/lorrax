; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @input_transpose_fusion(ptr noalias align 256 dereferenceable(121896960) %0, ptr noalias align 256 dereferenceable(121896960) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = urem i32 %3, 32
  %6 = icmp sle i32 %5, 23
  br i1 %6, label %7, label %155

7:                                                ; preds = %2
  %8 = mul i32 %4, 32
  %9 = udiv i32 %3, 32
  %10 = add i32 %8, %9
  %11 = urem i32 %10, 310
  %12 = mul i32 %11, 24
  %13 = mul i32 %4, 8
  %14 = urem i32 %13, 155
  %15 = mul i32 %14, 128
  %16 = add i32 %15, %3
  %17 = udiv i32 %16, 9920
  %18 = mul i32 %17, 7440
  %19 = add i32 %12, %18
  %20 = udiv i32 %13, 155
  %21 = mul i32 %20, 14880
  %22 = add i32 %19, %21
  %23 = add i32 %22, %5
  %24 = getelementptr inbounds [7618560 x { double, double }], ptr %0, i32 0, i32 %23
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !3
  %26 = mul i32 %5, 33
  %27 = add i32 %26, %9
  %28 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %27
  store { double, double } %25, ptr %28, align 8
  %29 = add i32 %10, 4
  %30 = urem i32 %29, 310
  %31 = mul i32 %30, 24
  %32 = add i32 %13, 1
  %33 = urem i32 %32, 155
  %34 = mul i32 %33, 128
  %35 = add i32 %34, %3
  %36 = udiv i32 %35, 9920
  %37 = mul i32 %36, 7440
  %38 = add i32 %31, %37
  %39 = udiv i32 %32, 155
  %40 = mul i32 %39, 14880
  %41 = add i32 %38, %40
  %42 = add i32 %41, %5
  %43 = getelementptr inbounds [7618560 x { double, double }], ptr %0, i32 0, i32 %42
  %44 = load { double, double }, ptr %43, align 8, !invariant.load !3
  %45 = add i32 %27, 4
  %46 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %45
  store { double, double } %44, ptr %46, align 8
  %47 = add i32 %10, 8
  %48 = urem i32 %47, 310
  %49 = mul i32 %48, 24
  %50 = add i32 %13, 2
  %51 = urem i32 %50, 155
  %52 = mul i32 %51, 128
  %53 = add i32 %52, %3
  %54 = udiv i32 %53, 9920
  %55 = mul i32 %54, 7440
  %56 = add i32 %49, %55
  %57 = udiv i32 %50, 155
  %58 = mul i32 %57, 14880
  %59 = add i32 %56, %58
  %60 = add i32 %59, %5
  %61 = getelementptr inbounds [7618560 x { double, double }], ptr %0, i32 0, i32 %60
  %62 = load { double, double }, ptr %61, align 8, !invariant.load !3
  %63 = add i32 %27, 8
  %64 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %63
  store { double, double } %62, ptr %64, align 8
  %65 = add i32 %10, 12
  %66 = urem i32 %65, 310
  %67 = mul i32 %66, 24
  %68 = add i32 %13, 3
  %69 = urem i32 %68, 155
  %70 = mul i32 %69, 128
  %71 = add i32 %70, %3
  %72 = udiv i32 %71, 9920
  %73 = mul i32 %72, 7440
  %74 = add i32 %67, %73
  %75 = udiv i32 %68, 155
  %76 = mul i32 %75, 14880
  %77 = add i32 %74, %76
  %78 = add i32 %77, %5
  %79 = getelementptr inbounds [7618560 x { double, double }], ptr %0, i32 0, i32 %78
  %80 = load { double, double }, ptr %79, align 8, !invariant.load !3
  %81 = add i32 %27, 12
  %82 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %81
  store { double, double } %80, ptr %82, align 8
  %83 = add i32 %10, 16
  %84 = urem i32 %83, 310
  %85 = mul i32 %84, 24
  %86 = add i32 %13, 4
  %87 = urem i32 %86, 155
  %88 = mul i32 %87, 128
  %89 = add i32 %88, %3
  %90 = udiv i32 %89, 9920
  %91 = mul i32 %90, 7440
  %92 = add i32 %85, %91
  %93 = udiv i32 %86, 155
  %94 = mul i32 %93, 14880
  %95 = add i32 %92, %94
  %96 = add i32 %95, %5
  %97 = getelementptr inbounds [7618560 x { double, double }], ptr %0, i32 0, i32 %96
  %98 = load { double, double }, ptr %97, align 8, !invariant.load !3
  %99 = add i32 %27, 16
  %100 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %99
  store { double, double } %98, ptr %100, align 8
  %101 = add i32 %10, 20
  %102 = urem i32 %101, 310
  %103 = mul i32 %102, 24
  %104 = add i32 %13, 5
  %105 = urem i32 %104, 155
  %106 = mul i32 %105, 128
  %107 = add i32 %106, %3
  %108 = udiv i32 %107, 9920
  %109 = mul i32 %108, 7440
  %110 = add i32 %103, %109
  %111 = udiv i32 %104, 155
  %112 = mul i32 %111, 14880
  %113 = add i32 %110, %112
  %114 = add i32 %113, %5
  %115 = getelementptr inbounds [7618560 x { double, double }], ptr %0, i32 0, i32 %114
  %116 = load { double, double }, ptr %115, align 8, !invariant.load !3
  %117 = add i32 %27, 20
  %118 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %117
  store { double, double } %116, ptr %118, align 8
  %119 = add i32 %10, 24
  %120 = urem i32 %119, 310
  %121 = mul i32 %120, 24
  %122 = add i32 %13, 6
  %123 = urem i32 %122, 155
  %124 = mul i32 %123, 128
  %125 = add i32 %124, %3
  %126 = udiv i32 %125, 9920
  %127 = mul i32 %126, 7440
  %128 = add i32 %121, %127
  %129 = udiv i32 %122, 155
  %130 = mul i32 %129, 14880
  %131 = add i32 %128, %130
  %132 = add i32 %131, %5
  %133 = getelementptr inbounds [7618560 x { double, double }], ptr %0, i32 0, i32 %132
  %134 = load { double, double }, ptr %133, align 8, !invariant.load !3
  %135 = add i32 %27, 24
  %136 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %135
  store { double, double } %134, ptr %136, align 8
  %137 = add i32 %10, 28
  %138 = urem i32 %137, 310
  %139 = mul i32 %138, 24
  %140 = add i32 %13, 7
  %141 = urem i32 %140, 155
  %142 = mul i32 %141, 128
  %143 = add i32 %142, %3
  %144 = udiv i32 %143, 9920
  %145 = mul i32 %144, 7440
  %146 = add i32 %139, %145
  %147 = udiv i32 %140, 155
  %148 = mul i32 %147, 14880
  %149 = add i32 %146, %148
  %150 = add i32 %149, %5
  %151 = getelementptr inbounds [7618560 x { double, double }], ptr %0, i32 0, i32 %150
  %152 = load { double, double }, ptr %151, align 8, !invariant.load !3
  %153 = add i32 %27, 28
  %154 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %153
  store { double, double } %152, ptr %154, align 8
  br label %155

155:                                              ; preds = %7, %2
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %156 = udiv i32 %3, 32
  %157 = mul i32 %156, 33
  %158 = add i32 %157, %5
  %159 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %158
  %160 = load { double, double }, ptr %159, align 8
  %161 = mul i32 %156, 317440
  %162 = mul i32 %4, 32
  %163 = add i32 %161, %162
  %164 = add i32 %163, %5
  %165 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %164
  store { double, double } %160, ptr %165, align 8
  %166 = add i32 %158, 132
  %167 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %166
  %168 = load { double, double }, ptr %167, align 8
  %169 = add i32 %164, 1269760
  %170 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %169
  store { double, double } %168, ptr %170, align 8
  %171 = add i32 %158, 264
  %172 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %171
  %173 = load { double, double }, ptr %172, align 8
  %174 = add i32 %164, 2539520
  %175 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %174
  store { double, double } %173, ptr %175, align 8
  %176 = add i32 %158, 396
  %177 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %176
  %178 = load { double, double }, ptr %177, align 8
  %179 = add i32 %164, 3809280
  %180 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %179
  store { double, double } %178, ptr %180, align 8
  %181 = add i32 %158, 528
  %182 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %181
  %183 = load { double, double }, ptr %182, align 8
  %184 = add i32 %164, 5079040
  %185 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %184
  store { double, double } %183, ptr %185, align 8
  %186 = add i32 %158, 660
  %187 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %186
  %188 = load { double, double }, ptr %187, align 8
  %189 = add i32 %164, 6348800
  %190 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %189
  store { double, double } %188, ptr %190, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

define ptx_kernel void @loop_complex_fusion(ptr noalias align 16 dereferenceable(121896960) %0, ptr noalias align 256 dereferenceable(121896960) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = mul i32 %4, 4
  %6 = mul i32 %3, 512
  %7 = add i32 %5, %6
  %8 = getelementptr inbounds [7618560 x { double, double }], ptr %0, i32 0, i32 %7
  %9 = load { double, double }, ptr %8, align 8, !invariant.load !3
  %10 = extractvalue { double, double } %9, 1
  %11 = extractvalue { double, double } %9, 0
  %12 = fneg double %10
  %13 = insertvalue { double, double } poison, double %11, 0
  %14 = insertvalue { double, double } %13, double %12, 1
  %15 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %7
  store { double, double } %14, ptr %15, align 8
  %16 = add i32 %7, 1
  %17 = getelementptr inbounds [7618560 x { double, double }], ptr %0, i32 0, i32 %16
  %18 = load { double, double }, ptr %17, align 8, !invariant.load !3
  %19 = extractvalue { double, double } %18, 1
  %20 = extractvalue { double, double } %18, 0
  %21 = fneg double %19
  %22 = insertvalue { double, double } poison, double %20, 0
  %23 = insertvalue { double, double } %22, double %21, 1
  %24 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %16
  store { double, double } %23, ptr %24, align 8
  %25 = add i32 %7, 2
  %26 = getelementptr inbounds [7618560 x { double, double }], ptr %0, i32 0, i32 %25
  %27 = load { double, double }, ptr %26, align 8, !invariant.load !3
  %28 = extractvalue { double, double } %27, 1
  %29 = extractvalue { double, double } %27, 0
  %30 = fneg double %28
  %31 = insertvalue { double, double } poison, double %29, 0
  %32 = insertvalue { double, double } %31, double %30, 1
  %33 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %25
  store { double, double } %32, ptr %33, align 8
  %34 = add i32 %7, 3
  %35 = getelementptr inbounds [7618560 x { double, double }], ptr %0, i32 0, i32 %34
  %36 = load { double, double }, ptr %35, align 8, !invariant.load !3
  %37 = extractvalue { double, double } %36, 1
  %38 = extractvalue { double, double } %36, 0
  %39 = fneg double %37
  %40 = insertvalue { double, double } poison, double %38, 0
  %41 = insertvalue { double, double } %40, double %39, 1
  %42 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %34
  store { double, double } %41, ptr %42, align 8
  ret void
}

define ptx_kernel void @loop_transpose_fusion(ptr noalias align 256 dereferenceable(1179648) %0, ptr noalias align 256 dereferenceable(1179648) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = mul i32 %3, 128
  %6 = add i32 %5, %4
  %7 = udiv i32 %6, 12
  %8 = urem i32 %7, 12
  %9 = mul i32 %8, 6144
  %10 = udiv i32 %6, 144
  %11 = mul i32 %10, 12
  %12 = add i32 %9, %11
  %13 = urem i32 %6, 12
  %14 = add i32 %12, %13
  %15 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %14
  %16 = load { double, double }, ptr %15, align 8, !invariant.load !3
  %17 = getelementptr inbounds [73728 x { double, double }], ptr %1, i32 0, i32 %6
  store { double, double } %16, ptr %17, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 128}
!2 = !{i32 0, i32 9920}
!3 = !{}
!4 = !{i32 0, i32 14880}
!5 = !{i32 0, i32 576}
