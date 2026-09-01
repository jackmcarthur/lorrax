; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef
@shared_01 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @loop_real_fusion(ptr noalias align 256 dereferenceable(5079040) %0, ptr noalias align 256 dereferenceable(4) %1, ptr noalias align 256 dereferenceable(1269760) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = urem i32 %7, 310
  %9 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  %10 = load i32, ptr %9, align 4, !invariant.load !3
  %11 = and i32 %10, 1
  %12 = mul i32 %11, 310
  %13 = icmp slt i32 %12, 0
  %14 = add i32 %12, 620
  %15 = select i1 %13, i32 %14, i32 %12
  %16 = call i32 @llvm.smin.i32(i32 %15, i32 310)
  %17 = call i32 @llvm.smax.i32(i32 %16, i32 0)
  %18 = add i32 %8, %17
  %19 = udiv i32 %7, 310
  %20 = mul i32 %19, 620
  %21 = add i32 %20, %18
  %22 = getelementptr inbounds [317440 x { double, double }], ptr %0, i32 0, i32 %21
  %23 = load { double, double }, ptr %22, align 8, !invariant.load !3
  %24 = extractvalue { double, double } %23, 0
  %25 = getelementptr inbounds [158720 x double], ptr %2, i32 0, i32 %7
  store double %24, ptr %25, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

define ptx_kernel void @loop_negate_fusion(ptr noalias align 256 dereferenceable(5079040) %0, ptr noalias align 256 dereferenceable(4) %1, ptr noalias align 256 dereferenceable(1269760) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %4, 128
  %7 = add i32 %6, %5
  %8 = urem i32 %7, 310
  %9 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  %10 = load i32, ptr %9, align 4, !invariant.load !3
  %11 = and i32 %10, 1
  %12 = mul i32 %11, 310
  %13 = icmp slt i32 %12, 0
  %14 = add i32 %12, 620
  %15 = select i1 %13, i32 %14, i32 %12
  %16 = call i32 @llvm.smin.i32(i32 %15, i32 310)
  %17 = call i32 @llvm.smax.i32(i32 %16, i32 0)
  %18 = add i32 %8, %17
  %19 = udiv i32 %7, 310
  %20 = mul i32 %19, 620
  %21 = add i32 %20, %18
  %22 = getelementptr inbounds [317440 x { double, double }], ptr %0, i32 0, i32 %21
  %23 = load { double, double }, ptr %22, align 8, !invariant.load !3
  %24 = extractvalue { double, double } %23, 1
  %25 = fneg double %24
  %26 = getelementptr inbounds [158720 x double], ptr %2, i32 0, i32 %7
  store double %25, ptr %26, align 8
  ret void
}

define ptx_kernel void @wrapped_gather(ptr noalias align 16 dereferenceable(44590400) %0, ptr noalias align 256 dereferenceable(2048) %1, ptr noalias align 256 dereferenceable(787251200) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = mul i32 %5, 2
  %7 = mul i32 %4, 256
  %8 = add i32 %6, %7
  %9 = udiv i32 %8, 155
  %10 = urem i32 %9, 512
  %11 = getelementptr inbounds [512 x i32], ptr %1, i32 0, i32 %10
  %12 = load i32, ptr %11, align 4, !invariant.load !3
  %13 = call i32 @llvm.smin.i32(i32 %12, i32 28)
  %14 = call i32 @llvm.smax.i32(i32 %13, i32 0)
  %15 = mul i32 %5, 4
  %16 = mul i32 %4, 512
  %17 = add i32 %15, %16
  %18 = urem i32 %17, 310
  %19 = mul i32 %18, 310
  %20 = mul i32 %14, 96100
  %21 = add i32 %19, %20
  %22 = udiv i32 %4, 310
  %23 = add i32 %21, %22
  %24 = getelementptr inbounds [2786900 x { double, double }], ptr %0, i32 0, i32 %23
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !3
  %26 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i32 %17
  store { double, double } %25, ptr %26, align 8
  %27 = add i32 %17, 1
  %28 = urem i32 %27, 310
  %29 = mul i32 %28, 310
  %30 = add i32 %29, %20
  %31 = add i32 %30, %22
  %32 = getelementptr inbounds [2786900 x { double, double }], ptr %0, i32 0, i32 %31
  %33 = load { double, double }, ptr %32, align 8, !invariant.load !3
  %34 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i32 %27
  store { double, double } %33, ptr %34, align 8
  %35 = add i32 %8, 1
  %36 = udiv i32 %35, 155
  %37 = urem i32 %36, 512
  %38 = getelementptr inbounds [512 x i32], ptr %1, i32 0, i32 %37
  %39 = load i32, ptr %38, align 4, !invariant.load !3
  %40 = call i32 @llvm.smin.i32(i32 %39, i32 28)
  %41 = call i32 @llvm.smax.i32(i32 %40, i32 0)
  %42 = add i32 %17, 2
  %43 = urem i32 %42, 310
  %44 = mul i32 %43, 310
  %45 = mul i32 %41, 96100
  %46 = add i32 %44, %45
  %47 = add i32 %46, %22
  %48 = getelementptr inbounds [2786900 x { double, double }], ptr %0, i32 0, i32 %47
  %49 = load { double, double }, ptr %48, align 8, !invariant.load !3
  %50 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i32 %42
  store { double, double } %49, ptr %50, align 8
  %51 = add i32 %17, 3
  %52 = udiv i32 %51, 310
  %53 = urem i32 %52, 512
  %54 = getelementptr inbounds [512 x i32], ptr %1, i32 0, i32 %53
  %55 = load i32, ptr %54, align 4, !invariant.load !3
  %56 = call i32 @llvm.smin.i32(i32 %55, i32 28)
  %57 = call i32 @llvm.smax.i32(i32 %56, i32 0)
  %58 = urem i32 %51, 310
  %59 = mul i32 %58, 310
  %60 = mul i32 %57, 96100
  %61 = add i32 %59, %60
  %62 = add i32 %61, %22
  %63 = getelementptr inbounds [2786900 x { double, double }], ptr %0, i32 0, i32 %62
  %64 = load { double, double }, ptr %63, align 8, !invariant.load !3
  %65 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i32 %51
  store { double, double } %64, ptr %65, align 8
  ret void
}

define ptx_kernel void @input_transpose_fusion(ptr noalias align 256 dereferenceable(787251200) %0, ptr noalias align 256 dereferenceable(787251200) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %5 = urem i32 %4, 10
  %6 = mul i32 %5, 32
  %7 = urem i32 %3, 32
  %8 = add i32 %6, %7
  %9 = icmp sle i32 %8, 309
  br i1 %9, label %10, label %56

10:                                               ; preds = %2
  %11 = udiv i32 %4, 10
  %12 = urem i32 %11, 512
  %13 = mul i32 %12, 310
  %14 = udiv i32 %4, 5120
  %15 = urem i32 %14, 5
  %16 = mul i32 %15, 5079040
  %17 = add i32 %13, %16
  %18 = add i32 %17, %6
  %19 = udiv i32 %4, 25600
  %20 = mul i32 %19, 24601600
  %21 = add i32 %18, %20
  %22 = udiv i32 %3, 32
  %23 = mul i32 %22, 158720
  %24 = add i32 %21, %23
  %25 = add i32 %24, %7
  %26 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %25
  %27 = load { double, double }, ptr %26, align 8, !invariant.load !3
  %28 = mul i32 %7, 33
  %29 = add i32 %28, %22
  %30 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %29
  store { double, double } %27, ptr %30, align 8
  %31 = add i32 %25, 634880
  %32 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %31
  %33 = load { double, double }, ptr %32, align 8, !invariant.load !3
  %34 = add i32 %29, 4
  %35 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %34
  store { double, double } %33, ptr %35, align 8
  %36 = add i32 %25, 1269760
  %37 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %36
  %38 = load { double, double }, ptr %37, align 8, !invariant.load !3
  %39 = add i32 %29, 8
  %40 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %39
  store { double, double } %38, ptr %40, align 8
  %41 = add i32 %25, 1904640
  %42 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %41
  %43 = load { double, double }, ptr %42, align 8, !invariant.load !3
  %44 = add i32 %29, 12
  %45 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %44
  store { double, double } %43, ptr %45, align 8
  %46 = add i32 %25, 2539520
  %47 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %46
  %48 = load { double, double }, ptr %47, align 8, !invariant.load !3
  %49 = add i32 %29, 16
  %50 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %49
  store { double, double } %48, ptr %50, align 8
  %51 = add i32 %25, 3174400
  %52 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %51
  %53 = load { double, double }, ptr %52, align 8, !invariant.load !3
  %54 = add i32 %29, 20
  %55 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %54
  store { double, double } %53, ptr %55, align 8
  br label %56

56:                                               ; preds = %10, %2
  %57 = udiv i32 %4, 5120
  %58 = urem i32 %57, 5
  %59 = mul i32 %58, 32
  %60 = udiv i32 %3, 32
  %61 = add i32 %59, %60
  %62 = add i32 %61, 24
  %63 = icmp sle i32 %62, 154
  %64 = and i1 %63, %9
  br i1 %64, label %65, label %85

65:                                               ; preds = %56
  %66 = udiv i32 %4, 10
  %67 = urem i32 %66, 512
  %68 = mul i32 %67, 310
  %69 = mul i32 %58, 5079040
  %70 = add i32 %68, %69
  %71 = add i32 %70, %6
  %72 = udiv i32 %4, 25600
  %73 = mul i32 %72, 24601600
  %74 = add i32 %71, %73
  %75 = mul i32 %60, 158720
  %76 = add i32 %74, %75
  %77 = add i32 %76, %7
  %78 = add i32 %77, 3809280
  %79 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %78
  %80 = load { double, double }, ptr %79, align 8, !invariant.load !3
  %81 = mul i32 %7, 33
  %82 = add i32 %81, %60
  %83 = add i32 %82, 24
  %84 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %83
  store { double, double } %80, ptr %84, align 8
  br label %85

85:                                               ; preds = %65, %56
  %86 = add i32 %61, 28
  %87 = icmp sle i32 %86, 154
  %88 = and i1 %87, %9
  br i1 %88, label %89, label %109

89:                                               ; preds = %85
  %90 = udiv i32 %4, 10
  %91 = urem i32 %90, 512
  %92 = mul i32 %91, 310
  %93 = mul i32 %58, 5079040
  %94 = add i32 %92, %93
  %95 = add i32 %94, %6
  %96 = udiv i32 %4, 25600
  %97 = mul i32 %96, 24601600
  %98 = add i32 %95, %97
  %99 = mul i32 %60, 158720
  %100 = add i32 %98, %99
  %101 = add i32 %100, %7
  %102 = add i32 %101, 4444160
  %103 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %102
  %104 = load { double, double }, ptr %103, align 8, !invariant.load !3
  %105 = mul i32 %7, 33
  %106 = add i32 %105, %60
  %107 = add i32 %106, 28
  %108 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %107
  store { double, double } %104, ptr %108, align 8
  br label %109

109:                                              ; preds = %89, %85
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %110 = add i32 %59, %7
  %111 = icmp sle i32 %110, 154
  br i1 %111, label %112, label %150

112:                                              ; preds = %109
  %113 = mul i32 %60, 33
  %114 = add i32 %113, %7
  %115 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %114
  %116 = load { double, double }, ptr %115, align 8
  %117 = udiv i32 %4, 10
  %118 = urem i32 %117, 512
  %119 = mul i32 %118, 96100
  %120 = add i32 %119, %59
  %121 = mul i32 %5, 4960
  %122 = add i32 %120, %121
  %123 = udiv i32 %4, 25600
  %124 = mul i32 %123, 48050
  %125 = add i32 %122, %124
  %126 = mul i32 %60, 155
  %127 = add i32 %125, %126
  %128 = add i32 %127, %7
  %129 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %128
  store { double, double } %116, ptr %129, align 8
  %130 = add i32 %114, 132
  %131 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %130
  %132 = load { double, double }, ptr %131, align 8
  %133 = add i32 %128, 620
  %134 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %133
  store { double, double } %132, ptr %134, align 8
  %135 = add i32 %114, 264
  %136 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %135
  %137 = load { double, double }, ptr %136, align 8
  %138 = add i32 %128, 1240
  %139 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %138
  store { double, double } %137, ptr %139, align 8
  %140 = add i32 %114, 396
  %141 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %140
  %142 = load { double, double }, ptr %141, align 8
  %143 = add i32 %128, 1860
  %144 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %143
  store { double, double } %142, ptr %144, align 8
  %145 = add i32 %114, 528
  %146 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %145
  %147 = load { double, double }, ptr %146, align 8
  %148 = add i32 %128, 2480
  %149 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %148
  store { double, double } %147, ptr %149, align 8
  br label %150

150:                                              ; preds = %112, %109
  %151 = add i32 %6, %60
  %152 = add i32 %151, 20
  %153 = icmp sle i32 %152, 309
  %154 = and i1 %153, %111
  br i1 %154, label %155, label %175

155:                                              ; preds = %150
  %156 = mul i32 %60, 33
  %157 = add i32 %156, %7
  %158 = add i32 %157, 660
  %159 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %158
  %160 = load { double, double }, ptr %159, align 8
  %161 = udiv i32 %4, 10
  %162 = urem i32 %161, 512
  %163 = mul i32 %162, 96100
  %164 = add i32 %163, %59
  %165 = mul i32 %5, 4960
  %166 = add i32 %164, %165
  %167 = udiv i32 %4, 25600
  %168 = mul i32 %167, 48050
  %169 = add i32 %166, %168
  %170 = mul i32 %60, 155
  %171 = add i32 %169, %170
  %172 = add i32 %171, %7
  %173 = add i32 %172, 3100
  %174 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %173
  store { double, double } %160, ptr %174, align 8
  br label %175

175:                                              ; preds = %155, %150
  %176 = add i32 %151, 24
  %177 = icmp sle i32 %176, 309
  %178 = and i1 %177, %111
  br i1 %178, label %179, label %199

179:                                              ; preds = %175
  %180 = mul i32 %60, 33
  %181 = add i32 %180, %7
  %182 = add i32 %181, 792
  %183 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %182
  %184 = load { double, double }, ptr %183, align 8
  %185 = udiv i32 %4, 10
  %186 = urem i32 %185, 512
  %187 = mul i32 %186, 96100
  %188 = add i32 %187, %59
  %189 = mul i32 %5, 4960
  %190 = add i32 %188, %189
  %191 = udiv i32 %4, 25600
  %192 = mul i32 %191, 48050
  %193 = add i32 %190, %192
  %194 = mul i32 %60, 155
  %195 = add i32 %193, %194
  %196 = add i32 %195, %7
  %197 = add i32 %196, 3720
  %198 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %197
  store { double, double } %184, ptr %198, align 8
  br label %199

199:                                              ; preds = %179, %175
  %200 = add i32 %151, 28
  %201 = icmp sle i32 %200, 309
  %202 = and i1 %201, %111
  br i1 %202, label %203, label %223

203:                                              ; preds = %199
  %204 = mul i32 %60, 33
  %205 = add i32 %204, %7
  %206 = add i32 %205, 924
  %207 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %206
  %208 = load { double, double }, ptr %207, align 8
  %209 = udiv i32 %4, 10
  %210 = urem i32 %209, 512
  %211 = mul i32 %210, 96100
  %212 = add i32 %211, %59
  %213 = mul i32 %5, 4960
  %214 = add i32 %212, %213
  %215 = udiv i32 %4, 25600
  %216 = mul i32 %215, 48050
  %217 = add i32 %214, %216
  %218 = mul i32 %60, 155
  %219 = add i32 %217, %218
  %220 = add i32 %219, %7
  %221 = add i32 %220, 4340
  %222 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %221
  store { double, double } %208, ptr %222, align 8
  br label %223

223:                                              ; preds = %203, %199
  ret void
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

define ptx_kernel void @loop_transpose_fusion_2(ptr noalias align 256 dereferenceable(5079040) %0, ptr noalias align 256 dereferenceable(787251200) %1, ptr noalias align 256 dereferenceable(787251200) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %5 = sext i32 %4 to i64
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = sext i32 %6 to i64
  %8 = udiv i64 %5, 155
  %9 = mul i64 %7, 4
  %10 = mul i64 %5, 512
  %11 = add i64 %9, %10
  %12 = udiv i64 %11, 155
  %13 = urem i64 %12, 512
  %14 = mul i64 %13, 1240
  %15 = mul i64 %8, 2
  %16 = add i64 %14, %15
  %17 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %16
  %18 = load i64, ptr %17, align 4, !invariant.load !3
  %19 = call i64 @llvm.smin.i64(i64 %18, i64 511)
  %20 = call i64 @llvm.smax.i64(i64 %19, i64 0)
  %21 = add i64 %16, 1
  %22 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %21
  %23 = load i64, ptr %22, align 4, !invariant.load !3
  %24 = call i64 @llvm.smin.i64(i64 %23, i64 619)
  %25 = call i64 @llvm.smax.i64(i64 %24, i64 0)
  %26 = mul i64 %20, 96100
  %27 = mul i64 %25, 155
  %28 = add i64 %26, %27
  %29 = urem i64 %11, 155
  %30 = add i64 %28, %29
  %31 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i64 %30
  %32 = load { double, double }, ptr %31, align 8, !invariant.load !3
  %33 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i64 %11
  store { double, double } %32, ptr %33, align 8
  %34 = add i64 %11, 1
  %35 = udiv i64 %34, 155
  %36 = urem i64 %35, 512
  %37 = mul i64 %36, 1240
  %38 = add i64 %37, %15
  %39 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %38
  %40 = load i64, ptr %39, align 4, !invariant.load !3
  %41 = call i64 @llvm.smin.i64(i64 %40, i64 511)
  %42 = call i64 @llvm.smax.i64(i64 %41, i64 0)
  %43 = add i64 %38, 1
  %44 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %43
  %45 = load i64, ptr %44, align 4, !invariant.load !3
  %46 = call i64 @llvm.smin.i64(i64 %45, i64 619)
  %47 = call i64 @llvm.smax.i64(i64 %46, i64 0)
  %48 = mul i64 %42, 96100
  %49 = mul i64 %47, 155
  %50 = add i64 %48, %49
  %51 = urem i64 %34, 155
  %52 = add i64 %50, %51
  %53 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i64 %52
  %54 = load { double, double }, ptr %53, align 8, !invariant.load !3
  %55 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i64 %34
  store { double, double } %54, ptr %55, align 8
  %56 = add i64 %11, 2
  %57 = udiv i64 %56, 155
  %58 = urem i64 %57, 512
  %59 = mul i64 %58, 1240
  %60 = add i64 %59, %15
  %61 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %60
  %62 = load i64, ptr %61, align 4, !invariant.load !3
  %63 = call i64 @llvm.smin.i64(i64 %62, i64 511)
  %64 = call i64 @llvm.smax.i64(i64 %63, i64 0)
  %65 = add i64 %60, 1
  %66 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %65
  %67 = load i64, ptr %66, align 4, !invariant.load !3
  %68 = call i64 @llvm.smin.i64(i64 %67, i64 619)
  %69 = call i64 @llvm.smax.i64(i64 %68, i64 0)
  %70 = mul i64 %64, 96100
  %71 = mul i64 %69, 155
  %72 = add i64 %70, %71
  %73 = urem i64 %56, 155
  %74 = add i64 %72, %73
  %75 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i64 %74
  %76 = load { double, double }, ptr %75, align 8, !invariant.load !3
  %77 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i64 %56
  store { double, double } %76, ptr %77, align 8
  %78 = add i64 %11, 3
  %79 = udiv i64 %78, 155
  %80 = urem i64 %79, 512
  %81 = mul i64 %80, 1240
  %82 = add i64 %81, %15
  %83 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %82
  %84 = load i64, ptr %83, align 4, !invariant.load !3
  %85 = call i64 @llvm.smin.i64(i64 %84, i64 511)
  %86 = call i64 @llvm.smax.i64(i64 %85, i64 0)
  %87 = add i64 %82, 1
  %88 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %87
  %89 = load i64, ptr %88, align 4, !invariant.load !3
  %90 = call i64 @llvm.smin.i64(i64 %89, i64 619)
  %91 = call i64 @llvm.smax.i64(i64 %90, i64 0)
  %92 = mul i64 %86, 96100
  %93 = mul i64 %91, 155
  %94 = add i64 %92, %93
  %95 = urem i64 %78, 155
  %96 = add i64 %94, %95
  %97 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i64 %96
  %98 = load { double, double }, ptr %97, align 8, !invariant.load !3
  %99 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i64 %78
  store { double, double } %98, ptr %99, align 8
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #2

define ptx_kernel void @loop_transpose_fusion(ptr noalias align 256 dereferenceable(787251200) %0, ptr noalias align 256 dereferenceable(787251200) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %5 = mul i32 %4, 4
  %6 = mul i32 %3, 512
  %7 = add i32 %5, %6
  %8 = udiv i32 %7, 155
  %9 = urem i32 %8, 2
  %10 = mul i32 %9, 24601600
  %11 = mul i32 %4, 2
  %12 = mul i32 %3, 256
  %13 = add i32 %11, %12
  %14 = udiv i32 %13, 155
  %15 = mul i32 %14, 155
  %16 = add i32 %10, %15
  %17 = urem i32 %7, 155
  %18 = add i32 %16, %17
  %19 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %18
  %20 = load { double, double }, ptr %19, align 8, !invariant.load !3
  %21 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %7
  store { double, double } %20, ptr %21, align 8
  %22 = add i32 %7, 1
  %23 = udiv i32 %22, 155
  %24 = urem i32 %23, 2
  %25 = mul i32 %24, 24601600
  %26 = add i32 %25, %15
  %27 = urem i32 %22, 155
  %28 = add i32 %26, %27
  %29 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %28
  %30 = load { double, double }, ptr %29, align 8, !invariant.load !3
  %31 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %22
  store { double, double } %30, ptr %31, align 8
  %32 = add i32 %7, 2
  %33 = udiv i32 %32, 155
  %34 = urem i32 %33, 2
  %35 = mul i32 %34, 24601600
  %36 = add i32 %13, 1
  %37 = udiv i32 %36, 155
  %38 = mul i32 %37, 155
  %39 = add i32 %35, %38
  %40 = urem i32 %32, 155
  %41 = add i32 %39, %40
  %42 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %41
  %43 = load { double, double }, ptr %42, align 8, !invariant.load !3
  %44 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %32
  store { double, double } %43, ptr %44, align 8
  %45 = add i32 %7, 3
  %46 = udiv i32 %45, 155
  %47 = urem i32 %46, 2
  %48 = mul i32 %47, 24601600
  %49 = udiv i32 %45, 310
  %50 = mul i32 %49, 155
  %51 = add i32 %48, %50
  %52 = urem i32 %45, 155
  %53 = add i32 %51, %52
  %54 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %53
  %55 = load { double, double }, ptr %54, align 8, !invariant.load !3
  %56 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %45
  store { double, double } %55, ptr %56, align 8
  ret void
}

define ptx_kernel void @loop_transpose_fusion_1(ptr noalias align 256 dereferenceable(5079040) %0, ptr noalias align 256 dereferenceable(787251200) %1, ptr noalias align 256 dereferenceable(787251200) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %5 = sext i32 %4 to i64
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = sext i32 %6 to i64
  %8 = udiv i64 %5, 155
  %9 = mul i64 %7, 4
  %10 = mul i64 %5, 512
  %11 = add i64 %9, %10
  %12 = udiv i64 %11, 155
  %13 = urem i64 %12, 512
  %14 = mul i64 %13, 1240
  %15 = mul i64 %8, 2
  %16 = add i64 %14, %15
  %17 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %16
  %18 = load i64, ptr %17, align 4, !invariant.load !3
  %19 = call i64 @llvm.smin.i64(i64 %18, i64 511)
  %20 = call i64 @llvm.smax.i64(i64 %19, i64 0)
  %21 = add i64 %16, 1
  %22 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %21
  %23 = load i64, ptr %22, align 4, !invariant.load !3
  %24 = call i64 @llvm.smin.i64(i64 %23, i64 619)
  %25 = call i64 @llvm.smax.i64(i64 %24, i64 0)
  %26 = urem i64 %11, 155
  %27 = mul i64 %26, 158720
  %28 = udiv i64 %25, 310
  %29 = mul i64 %28, 24601600
  %30 = add i64 %27, %29
  %31 = mul i64 %20, 310
  %32 = add i64 %30, %31
  %33 = urem i64 %25, 310
  %34 = add i64 %32, %33
  %35 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i64 %34
  %36 = load { double, double }, ptr %35, align 8, !invariant.load !3
  %37 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i64 %11
  store { double, double } %36, ptr %37, align 8
  %38 = add i64 %11, 1
  %39 = udiv i64 %38, 155
  %40 = urem i64 %39, 512
  %41 = mul i64 %40, 1240
  %42 = add i64 %41, %15
  %43 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %42
  %44 = load i64, ptr %43, align 4, !invariant.load !3
  %45 = call i64 @llvm.smin.i64(i64 %44, i64 511)
  %46 = call i64 @llvm.smax.i64(i64 %45, i64 0)
  %47 = add i64 %42, 1
  %48 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %47
  %49 = load i64, ptr %48, align 4, !invariant.load !3
  %50 = call i64 @llvm.smin.i64(i64 %49, i64 619)
  %51 = call i64 @llvm.smax.i64(i64 %50, i64 0)
  %52 = urem i64 %38, 155
  %53 = mul i64 %52, 158720
  %54 = udiv i64 %51, 310
  %55 = mul i64 %54, 24601600
  %56 = add i64 %53, %55
  %57 = mul i64 %46, 310
  %58 = add i64 %56, %57
  %59 = urem i64 %51, 310
  %60 = add i64 %58, %59
  %61 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i64 %60
  %62 = load { double, double }, ptr %61, align 8, !invariant.load !3
  %63 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i64 %38
  store { double, double } %62, ptr %63, align 8
  %64 = add i64 %11, 2
  %65 = udiv i64 %64, 155
  %66 = urem i64 %65, 512
  %67 = mul i64 %66, 1240
  %68 = add i64 %67, %15
  %69 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %68
  %70 = load i64, ptr %69, align 4, !invariant.load !3
  %71 = call i64 @llvm.smin.i64(i64 %70, i64 511)
  %72 = call i64 @llvm.smax.i64(i64 %71, i64 0)
  %73 = add i64 %68, 1
  %74 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %73
  %75 = load i64, ptr %74, align 4, !invariant.load !3
  %76 = call i64 @llvm.smin.i64(i64 %75, i64 619)
  %77 = call i64 @llvm.smax.i64(i64 %76, i64 0)
  %78 = urem i64 %64, 155
  %79 = mul i64 %78, 158720
  %80 = udiv i64 %77, 310
  %81 = mul i64 %80, 24601600
  %82 = add i64 %79, %81
  %83 = mul i64 %72, 310
  %84 = add i64 %82, %83
  %85 = urem i64 %77, 310
  %86 = add i64 %84, %85
  %87 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i64 %86
  %88 = load { double, double }, ptr %87, align 8, !invariant.load !3
  %89 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i64 %64
  store { double, double } %88, ptr %89, align 8
  %90 = add i64 %11, 3
  %91 = udiv i64 %90, 155
  %92 = urem i64 %91, 512
  %93 = mul i64 %92, 1240
  %94 = add i64 %93, %15
  %95 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %94
  %96 = load i64, ptr %95, align 4, !invariant.load !3
  %97 = call i64 @llvm.smin.i64(i64 %96, i64 511)
  %98 = call i64 @llvm.smax.i64(i64 %97, i64 0)
  %99 = add i64 %94, 1
  %100 = getelementptr inbounds [634880 x i64], ptr %0, i32 0, i64 %99
  %101 = load i64, ptr %100, align 4, !invariant.load !3
  %102 = call i64 @llvm.smin.i64(i64 %101, i64 619)
  %103 = call i64 @llvm.smax.i64(i64 %102, i64 0)
  %104 = urem i64 %90, 155
  %105 = mul i64 %104, 158720
  %106 = udiv i64 %103, 310
  %107 = mul i64 %106, 24601600
  %108 = add i64 %105, %107
  %109 = mul i64 %98, 310
  %110 = add i64 %108, %109
  %111 = urem i64 %103, 310
  %112 = add i64 %110, %111
  %113 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i64 %112
  %114 = load { double, double }, ptr %113, align 8, !invariant.load !3
  %115 = getelementptr inbounds [49203200 x { double, double }], ptr %2, i32 0, i64 %90
  store { double, double } %114, ptr %115, align 8
  ret void
}

define ptx_kernel void @input_transpose_fusion_1(ptr noalias align 256 dereferenceable(787251200) %0, ptr noalias align 256 dereferenceable(1269760) %1, ptr noalias align 256 dereferenceable(1269760) %2, ptr noalias align 256 dereferenceable(5079040) %3, ptr noalias align 256 dereferenceable(4) %4, ptr noalias align 256 dereferenceable(787251200) %5) #0 {
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %9 = urem i32 %8, 5
  %10 = mul i32 %9, 32
  %11 = urem i32 %7, 32
  %12 = add i32 %10, %11
  %13 = icmp sle i32 %12, 154
  br i1 %13, label %14, label %55

14:                                               ; preds = %6
  %15 = udiv i32 %8, 5
  %16 = urem i32 %15, 512
  %17 = mul i32 %16, 155
  %18 = udiv i32 %8, 2560
  %19 = urem i32 %18, 10
  %20 = mul i32 %19, 2539520
  %21 = add i32 %17, %20
  %22 = add i32 %21, %10
  %23 = udiv i32 %8, 25600
  %24 = mul i32 %23, 24601600
  %25 = add i32 %22, %24
  %26 = udiv i32 %7, 32
  %27 = mul i32 %26, 79360
  %28 = add i32 %25, %27
  %29 = add i32 %28, %11
  %30 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %29
  %31 = load { double, double }, ptr %30, align 8, !invariant.load !3
  %32 = mul i32 %11, 33
  %33 = add i32 %32, %26
  %34 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %33
  store { double, double } %31, ptr %34, align 8
  %35 = add i32 %29, 317440
  %36 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %35
  %37 = load { double, double }, ptr %36, align 8, !invariant.load !3
  %38 = add i32 %33, 4
  %39 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %38
  store { double, double } %37, ptr %39, align 8
  %40 = add i32 %29, 634880
  %41 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %40
  %42 = load { double, double }, ptr %41, align 8, !invariant.load !3
  %43 = add i32 %33, 8
  %44 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %43
  store { double, double } %42, ptr %44, align 8
  %45 = add i32 %29, 952320
  %46 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %45
  %47 = load { double, double }, ptr %46, align 8, !invariant.load !3
  %48 = add i32 %33, 12
  %49 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %48
  store { double, double } %47, ptr %49, align 8
  %50 = add i32 %29, 1269760
  %51 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %50
  %52 = load { double, double }, ptr %51, align 8, !invariant.load !3
  %53 = add i32 %33, 16
  %54 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %53
  store { double, double } %52, ptr %54, align 8
  br label %55

55:                                               ; preds = %14, %6
  %56 = udiv i32 %8, 2560
  %57 = urem i32 %56, 10
  %58 = mul i32 %57, 32
  %59 = udiv i32 %7, 32
  %60 = add i32 %58, %59
  %61 = add i32 %60, 20
  %62 = icmp sle i32 %61, 309
  %63 = and i1 %62, %13
  br i1 %63, label %64, label %84

64:                                               ; preds = %55
  %65 = udiv i32 %8, 5
  %66 = urem i32 %65, 512
  %67 = mul i32 %66, 155
  %68 = mul i32 %57, 2539520
  %69 = add i32 %67, %68
  %70 = add i32 %69, %10
  %71 = udiv i32 %8, 25600
  %72 = mul i32 %71, 24601600
  %73 = add i32 %70, %72
  %74 = mul i32 %59, 79360
  %75 = add i32 %73, %74
  %76 = add i32 %75, %11
  %77 = add i32 %76, 1587200
  %78 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %77
  %79 = load { double, double }, ptr %78, align 8, !invariant.load !3
  %80 = mul i32 %11, 33
  %81 = add i32 %80, %59
  %82 = add i32 %81, 20
  %83 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %82
  store { double, double } %79, ptr %83, align 8
  br label %84

84:                                               ; preds = %64, %55
  %85 = add i32 %60, 24
  %86 = icmp sle i32 %85, 309
  %87 = and i1 %86, %13
  br i1 %87, label %88, label %108

88:                                               ; preds = %84
  %89 = udiv i32 %8, 5
  %90 = urem i32 %89, 512
  %91 = mul i32 %90, 155
  %92 = mul i32 %57, 2539520
  %93 = add i32 %91, %92
  %94 = add i32 %93, %10
  %95 = udiv i32 %8, 25600
  %96 = mul i32 %95, 24601600
  %97 = add i32 %94, %96
  %98 = mul i32 %59, 79360
  %99 = add i32 %97, %98
  %100 = add i32 %99, %11
  %101 = add i32 %100, 1904640
  %102 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %101
  %103 = load { double, double }, ptr %102, align 8, !invariant.load !3
  %104 = mul i32 %11, 33
  %105 = add i32 %104, %59
  %106 = add i32 %105, 24
  %107 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %106
  store { double, double } %103, ptr %107, align 8
  br label %108

108:                                              ; preds = %88, %84
  %109 = add i32 %60, 28
  %110 = icmp sle i32 %109, 309
  %111 = and i1 %110, %13
  br i1 %111, label %112, label %132

112:                                              ; preds = %108
  %113 = udiv i32 %8, 5
  %114 = urem i32 %113, 512
  %115 = mul i32 %114, 155
  %116 = mul i32 %57, 2539520
  %117 = add i32 %115, %116
  %118 = add i32 %117, %10
  %119 = udiv i32 %8, 25600
  %120 = mul i32 %119, 24601600
  %121 = add i32 %118, %120
  %122 = mul i32 %59, 79360
  %123 = add i32 %121, %122
  %124 = add i32 %123, %11
  %125 = add i32 %124, 2222080
  %126 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %125
  %127 = load { double, double }, ptr %126, align 8, !invariant.load !3
  %128 = mul i32 %11, 33
  %129 = add i32 %128, %59
  %130 = add i32 %129, 28
  %131 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %130
  store { double, double } %127, ptr %131, align 8
  br label %132

132:                                              ; preds = %112, %108
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %133 = add i32 %58, %11
  %134 = icmp sle i32 %133, 309
  br i1 %134, label %135, label %391

135:                                              ; preds = %132
  %136 = mul i32 %59, 33
  %137 = add i32 %136, %11
  %138 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %137
  %139 = load { double, double }, ptr %138, align 8
  %140 = udiv i32 %8, 25600
  %141 = mul i32 %140, 155
  %142 = add i32 %10, %141
  %143 = add i32 %142, %59
  %144 = getelementptr inbounds [1 x i32], ptr %4, i32 0, i32 0
  %145 = load i32, ptr %144, align 4, !invariant.load !3
  %146 = lshr i32 %145, 1
  %147 = and i32 %146, 1
  %148 = mul i32 %147, 310
  %149 = icmp slt i32 %148, 0
  %150 = add i32 %148, 620
  %151 = select i1 %149, i32 %150, i32 %148
  %152 = call i32 @llvm.smin.i32(i32 %151, i32 310)
  %153 = call i32 @llvm.smax.i32(i32 %152, i32 0)
  %154 = add i32 %143, %153
  %155 = udiv i32 %8, 5
  %156 = urem i32 %155, 512
  %157 = mul i32 %156, 620
  %158 = add i32 %157, %154
  %159 = getelementptr inbounds [317440 x { double, double }], ptr %3, i32 0, i32 %158
  %160 = load { double, double }, ptr %159, align 8, !invariant.load !3
  %161 = extractvalue { double, double } %160, 0
  %162 = extractvalue { double, double } %160, 1
  %163 = extractvalue { double, double } %139, 0
  %164 = extractvalue { double, double } %139, 1
  %165 = fmul double %161, %163
  %166 = fmul double %162, %164
  %167 = fsub double %165, %166
  %168 = fmul double %162, %163
  %169 = fmul double %161, %164
  %170 = fadd double %168, %169
  %171 = mul i32 %156, 310
  %172 = add i32 %171, %58
  %173 = add i32 %172, %11
  %174 = getelementptr inbounds [158720 x double], ptr %1, i32 0, i32 %173
  %175 = load double, ptr %174, align 8, !invariant.load !3
  %176 = getelementptr inbounds [158720 x double], ptr %2, i32 0, i32 %173
  %177 = load double, ptr %176, align 8, !invariant.load !3
  %178 = fmul double %167, %175
  %179 = fmul double %170, %177
  %180 = fsub double %178, %179
  %181 = fmul double %170, %175
  %182 = fmul double %167, %177
  %183 = fadd double %181, %182
  %184 = insertvalue { double, double } poison, double %180, 0
  %185 = insertvalue { double, double } %184, double %183, 1
  %186 = mul i32 %156, 96100
  %187 = add i32 %186, %58
  %188 = mul i32 %9, 9920
  %189 = add i32 %187, %188
  %190 = mul i32 %140, 48050
  %191 = add i32 %189, %190
  %192 = mul i32 %59, 310
  %193 = add i32 %191, %192
  %194 = add i32 %193, %11
  %195 = getelementptr inbounds [49203200 x { double, double }], ptr %5, i32 0, i32 %194
  store { double, double } %185, ptr %195, align 8
  %196 = add i32 %137, 132
  %197 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %196
  %198 = load { double, double }, ptr %197, align 8
  %199 = add i32 %143, 4
  %200 = load i32, ptr %144, align 4, !invariant.load !3
  %201 = lshr i32 %200, 1
  %202 = and i32 %201, 1
  %203 = mul i32 %202, 310
  %204 = icmp slt i32 %203, 0
  %205 = add i32 %203, 620
  %206 = select i1 %204, i32 %205, i32 %203
  %207 = call i32 @llvm.smin.i32(i32 %206, i32 310)
  %208 = call i32 @llvm.smax.i32(i32 %207, i32 0)
  %209 = add i32 %199, %208
  %210 = add i32 %157, %209
  %211 = getelementptr inbounds [317440 x { double, double }], ptr %3, i32 0, i32 %210
  %212 = load { double, double }, ptr %211, align 8, !invariant.load !3
  %213 = extractvalue { double, double } %212, 0
  %214 = extractvalue { double, double } %212, 1
  %215 = extractvalue { double, double } %198, 0
  %216 = extractvalue { double, double } %198, 1
  %217 = fmul double %213, %215
  %218 = fmul double %214, %216
  %219 = fsub double %217, %218
  %220 = fmul double %214, %215
  %221 = fmul double %213, %216
  %222 = fadd double %220, %221
  %223 = load double, ptr %174, align 8, !invariant.load !3
  %224 = load double, ptr %176, align 8, !invariant.load !3
  %225 = fmul double %219, %223
  %226 = fmul double %222, %224
  %227 = fsub double %225, %226
  %228 = fmul double %222, %223
  %229 = fmul double %219, %224
  %230 = fadd double %228, %229
  %231 = insertvalue { double, double } poison, double %227, 0
  %232 = insertvalue { double, double } %231, double %230, 1
  %233 = add i32 %194, 1240
  %234 = getelementptr inbounds [49203200 x { double, double }], ptr %5, i32 0, i32 %233
  store { double, double } %232, ptr %234, align 8
  %235 = add i32 %137, 264
  %236 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %235
  %237 = load { double, double }, ptr %236, align 8
  %238 = add i32 %143, 8
  %239 = load i32, ptr %144, align 4, !invariant.load !3
  %240 = lshr i32 %239, 1
  %241 = and i32 %240, 1
  %242 = mul i32 %241, 310
  %243 = icmp slt i32 %242, 0
  %244 = add i32 %242, 620
  %245 = select i1 %243, i32 %244, i32 %242
  %246 = call i32 @llvm.smin.i32(i32 %245, i32 310)
  %247 = call i32 @llvm.smax.i32(i32 %246, i32 0)
  %248 = add i32 %238, %247
  %249 = add i32 %157, %248
  %250 = getelementptr inbounds [317440 x { double, double }], ptr %3, i32 0, i32 %249
  %251 = load { double, double }, ptr %250, align 8, !invariant.load !3
  %252 = extractvalue { double, double } %251, 0
  %253 = extractvalue { double, double } %251, 1
  %254 = extractvalue { double, double } %237, 0
  %255 = extractvalue { double, double } %237, 1
  %256 = fmul double %252, %254
  %257 = fmul double %253, %255
  %258 = fsub double %256, %257
  %259 = fmul double %253, %254
  %260 = fmul double %252, %255
  %261 = fadd double %259, %260
  %262 = load double, ptr %174, align 8, !invariant.load !3
  %263 = load double, ptr %176, align 8, !invariant.load !3
  %264 = fmul double %258, %262
  %265 = fmul double %261, %263
  %266 = fsub double %264, %265
  %267 = fmul double %261, %262
  %268 = fmul double %258, %263
  %269 = fadd double %267, %268
  %270 = insertvalue { double, double } poison, double %266, 0
  %271 = insertvalue { double, double } %270, double %269, 1
  %272 = add i32 %194, 2480
  %273 = getelementptr inbounds [49203200 x { double, double }], ptr %5, i32 0, i32 %272
  store { double, double } %271, ptr %273, align 8
  %274 = add i32 %137, 396
  %275 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %274
  %276 = load { double, double }, ptr %275, align 8
  %277 = add i32 %143, 12
  %278 = load i32, ptr %144, align 4, !invariant.load !3
  %279 = lshr i32 %278, 1
  %280 = and i32 %279, 1
  %281 = mul i32 %280, 310
  %282 = icmp slt i32 %281, 0
  %283 = add i32 %281, 620
  %284 = select i1 %282, i32 %283, i32 %281
  %285 = call i32 @llvm.smin.i32(i32 %284, i32 310)
  %286 = call i32 @llvm.smax.i32(i32 %285, i32 0)
  %287 = add i32 %277, %286
  %288 = add i32 %157, %287
  %289 = getelementptr inbounds [317440 x { double, double }], ptr %3, i32 0, i32 %288
  %290 = load { double, double }, ptr %289, align 8, !invariant.load !3
  %291 = extractvalue { double, double } %290, 0
  %292 = extractvalue { double, double } %290, 1
  %293 = extractvalue { double, double } %276, 0
  %294 = extractvalue { double, double } %276, 1
  %295 = fmul double %291, %293
  %296 = fmul double %292, %294
  %297 = fsub double %295, %296
  %298 = fmul double %292, %293
  %299 = fmul double %291, %294
  %300 = fadd double %298, %299
  %301 = load double, ptr %174, align 8, !invariant.load !3
  %302 = load double, ptr %176, align 8, !invariant.load !3
  %303 = fmul double %297, %301
  %304 = fmul double %300, %302
  %305 = fsub double %303, %304
  %306 = fmul double %300, %301
  %307 = fmul double %297, %302
  %308 = fadd double %306, %307
  %309 = insertvalue { double, double } poison, double %305, 0
  %310 = insertvalue { double, double } %309, double %308, 1
  %311 = add i32 %194, 3720
  %312 = getelementptr inbounds [49203200 x { double, double }], ptr %5, i32 0, i32 %311
  store { double, double } %310, ptr %312, align 8
  %313 = add i32 %137, 528
  %314 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %313
  %315 = load { double, double }, ptr %314, align 8
  %316 = add i32 %143, 16
  %317 = load i32, ptr %144, align 4, !invariant.load !3
  %318 = lshr i32 %317, 1
  %319 = and i32 %318, 1
  %320 = mul i32 %319, 310
  %321 = icmp slt i32 %320, 0
  %322 = add i32 %320, 620
  %323 = select i1 %321, i32 %322, i32 %320
  %324 = call i32 @llvm.smin.i32(i32 %323, i32 310)
  %325 = call i32 @llvm.smax.i32(i32 %324, i32 0)
  %326 = add i32 %316, %325
  %327 = add i32 %157, %326
  %328 = getelementptr inbounds [317440 x { double, double }], ptr %3, i32 0, i32 %327
  %329 = load { double, double }, ptr %328, align 8, !invariant.load !3
  %330 = extractvalue { double, double } %329, 0
  %331 = extractvalue { double, double } %329, 1
  %332 = extractvalue { double, double } %315, 0
  %333 = extractvalue { double, double } %315, 1
  %334 = fmul double %330, %332
  %335 = fmul double %331, %333
  %336 = fsub double %334, %335
  %337 = fmul double %331, %332
  %338 = fmul double %330, %333
  %339 = fadd double %337, %338
  %340 = load double, ptr %174, align 8, !invariant.load !3
  %341 = load double, ptr %176, align 8, !invariant.load !3
  %342 = fmul double %336, %340
  %343 = fmul double %339, %341
  %344 = fsub double %342, %343
  %345 = fmul double %339, %340
  %346 = fmul double %336, %341
  %347 = fadd double %345, %346
  %348 = insertvalue { double, double } poison, double %344, 0
  %349 = insertvalue { double, double } %348, double %347, 1
  %350 = add i32 %194, 4960
  %351 = getelementptr inbounds [49203200 x { double, double }], ptr %5, i32 0, i32 %350
  store { double, double } %349, ptr %351, align 8
  %352 = add i32 %137, 660
  %353 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %352
  %354 = load { double, double }, ptr %353, align 8
  %355 = add i32 %143, 20
  %356 = load i32, ptr %144, align 4, !invariant.load !3
  %357 = lshr i32 %356, 1
  %358 = and i32 %357, 1
  %359 = mul i32 %358, 310
  %360 = icmp slt i32 %359, 0
  %361 = add i32 %359, 620
  %362 = select i1 %360, i32 %361, i32 %359
  %363 = call i32 @llvm.smin.i32(i32 %362, i32 310)
  %364 = call i32 @llvm.smax.i32(i32 %363, i32 0)
  %365 = add i32 %355, %364
  %366 = add i32 %157, %365
  %367 = getelementptr inbounds [317440 x { double, double }], ptr %3, i32 0, i32 %366
  %368 = load { double, double }, ptr %367, align 8, !invariant.load !3
  %369 = extractvalue { double, double } %368, 0
  %370 = extractvalue { double, double } %368, 1
  %371 = extractvalue { double, double } %354, 0
  %372 = extractvalue { double, double } %354, 1
  %373 = fmul double %369, %371
  %374 = fmul double %370, %372
  %375 = fsub double %373, %374
  %376 = fmul double %370, %371
  %377 = fmul double %369, %372
  %378 = fadd double %376, %377
  %379 = load double, ptr %174, align 8, !invariant.load !3
  %380 = load double, ptr %176, align 8, !invariant.load !3
  %381 = fmul double %375, %379
  %382 = fmul double %378, %380
  %383 = fsub double %381, %382
  %384 = fmul double %378, %379
  %385 = fmul double %375, %380
  %386 = fadd double %384, %385
  %387 = insertvalue { double, double } poison, double %383, 0
  %388 = insertvalue { double, double } %387, double %386, 1
  %389 = add i32 %194, 6200
  %390 = getelementptr inbounds [49203200 x { double, double }], ptr %5, i32 0, i32 %389
  store { double, double } %388, ptr %390, align 8
  br label %391

391:                                              ; preds = %135, %132
  %392 = add i32 %10, %59
  %393 = add i32 %392, 24
  %394 = icmp sle i32 %393, 154
  %395 = and i1 %394, %134
  br i1 %395, label %396, label %460

396:                                              ; preds = %391
  %397 = mul i32 %59, 33
  %398 = add i32 %397, %11
  %399 = add i32 %398, 792
  %400 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %399
  %401 = load { double, double }, ptr %400, align 8
  %402 = udiv i32 %8, 25600
  %403 = mul i32 %402, 155
  %404 = add i32 %10, %403
  %405 = add i32 %404, %59
  %406 = add i32 %405, 24
  %407 = getelementptr inbounds [1 x i32], ptr %4, i32 0, i32 0
  %408 = load i32, ptr %407, align 4, !invariant.load !3
  %409 = lshr i32 %408, 1
  %410 = and i32 %409, 1
  %411 = mul i32 %410, 310
  %412 = icmp slt i32 %411, 0
  %413 = add i32 %411, 620
  %414 = select i1 %412, i32 %413, i32 %411
  %415 = call i32 @llvm.smin.i32(i32 %414, i32 310)
  %416 = call i32 @llvm.smax.i32(i32 %415, i32 0)
  %417 = add i32 %406, %416
  %418 = udiv i32 %8, 5
  %419 = urem i32 %418, 512
  %420 = mul i32 %419, 620
  %421 = add i32 %420, %417
  %422 = getelementptr inbounds [317440 x { double, double }], ptr %3, i32 0, i32 %421
  %423 = load { double, double }, ptr %422, align 8, !invariant.load !3
  %424 = extractvalue { double, double } %423, 0
  %425 = extractvalue { double, double } %423, 1
  %426 = extractvalue { double, double } %401, 0
  %427 = extractvalue { double, double } %401, 1
  %428 = fmul double %424, %426
  %429 = fmul double %425, %427
  %430 = fsub double %428, %429
  %431 = fmul double %425, %426
  %432 = fmul double %424, %427
  %433 = fadd double %431, %432
  %434 = mul i32 %419, 310
  %435 = add i32 %434, %58
  %436 = add i32 %435, %11
  %437 = getelementptr inbounds [158720 x double], ptr %1, i32 0, i32 %436
  %438 = load double, ptr %437, align 8, !invariant.load !3
  %439 = getelementptr inbounds [158720 x double], ptr %2, i32 0, i32 %436
  %440 = load double, ptr %439, align 8, !invariant.load !3
  %441 = fmul double %430, %438
  %442 = fmul double %433, %440
  %443 = fsub double %441, %442
  %444 = fmul double %433, %438
  %445 = fmul double %430, %440
  %446 = fadd double %444, %445
  %447 = insertvalue { double, double } poison, double %443, 0
  %448 = insertvalue { double, double } %447, double %446, 1
  %449 = mul i32 %419, 96100
  %450 = add i32 %449, %58
  %451 = mul i32 %9, 9920
  %452 = add i32 %450, %451
  %453 = mul i32 %402, 48050
  %454 = add i32 %452, %453
  %455 = mul i32 %59, 310
  %456 = add i32 %454, %455
  %457 = add i32 %456, %11
  %458 = add i32 %457, 7440
  %459 = getelementptr inbounds [49203200 x { double, double }], ptr %5, i32 0, i32 %458
  store { double, double } %448, ptr %459, align 8
  br label %460

460:                                              ; preds = %396, %391
  %461 = add i32 %392, 28
  %462 = icmp sle i32 %461, 154
  %463 = and i1 %462, %134
  br i1 %463, label %464, label %528

464:                                              ; preds = %460
  %465 = mul i32 %59, 33
  %466 = add i32 %465, %11
  %467 = add i32 %466, 924
  %468 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %467
  %469 = load { double, double }, ptr %468, align 8
  %470 = udiv i32 %8, 25600
  %471 = mul i32 %470, 155
  %472 = add i32 %10, %471
  %473 = add i32 %472, %59
  %474 = add i32 %473, 28
  %475 = getelementptr inbounds [1 x i32], ptr %4, i32 0, i32 0
  %476 = load i32, ptr %475, align 4, !invariant.load !3
  %477 = lshr i32 %476, 1
  %478 = and i32 %477, 1
  %479 = mul i32 %478, 310
  %480 = icmp slt i32 %479, 0
  %481 = add i32 %479, 620
  %482 = select i1 %480, i32 %481, i32 %479
  %483 = call i32 @llvm.smin.i32(i32 %482, i32 310)
  %484 = call i32 @llvm.smax.i32(i32 %483, i32 0)
  %485 = add i32 %474, %484
  %486 = udiv i32 %8, 5
  %487 = urem i32 %486, 512
  %488 = mul i32 %487, 620
  %489 = add i32 %488, %485
  %490 = getelementptr inbounds [317440 x { double, double }], ptr %3, i32 0, i32 %489
  %491 = load { double, double }, ptr %490, align 8, !invariant.load !3
  %492 = extractvalue { double, double } %491, 0
  %493 = extractvalue { double, double } %491, 1
  %494 = extractvalue { double, double } %469, 0
  %495 = extractvalue { double, double } %469, 1
  %496 = fmul double %492, %494
  %497 = fmul double %493, %495
  %498 = fsub double %496, %497
  %499 = fmul double %493, %494
  %500 = fmul double %492, %495
  %501 = fadd double %499, %500
  %502 = mul i32 %487, 310
  %503 = add i32 %502, %58
  %504 = add i32 %503, %11
  %505 = getelementptr inbounds [158720 x double], ptr %1, i32 0, i32 %504
  %506 = load double, ptr %505, align 8, !invariant.load !3
  %507 = getelementptr inbounds [158720 x double], ptr %2, i32 0, i32 %504
  %508 = load double, ptr %507, align 8, !invariant.load !3
  %509 = fmul double %498, %506
  %510 = fmul double %501, %508
  %511 = fsub double %509, %510
  %512 = fmul double %501, %506
  %513 = fmul double %498, %508
  %514 = fadd double %512, %513
  %515 = insertvalue { double, double } poison, double %511, 0
  %516 = insertvalue { double, double } %515, double %514, 1
  %517 = mul i32 %487, 96100
  %518 = add i32 %517, %58
  %519 = mul i32 %9, 9920
  %520 = add i32 %518, %519
  %521 = mul i32 %470, 48050
  %522 = add i32 %520, %521
  %523 = mul i32 %59, 310
  %524 = add i32 %522, %523
  %525 = add i32 %524, %11
  %526 = add i32 %525, 8680
  %527 = getelementptr inbounds [49203200 x { double, double }], ptr %5, i32 0, i32 %526
  store { double, double } %516, ptr %527, align 8
  br label %528

528:                                              ; preds = %464, %460
  ret void
}

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 1240}
!2 = !{i32 0, i32 128}
!3 = !{}
!4 = !{i32 0, i32 96100}
!5 = !{i32 0, i32 51200}
