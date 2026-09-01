; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef
@shared_01 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @wrapped_gather(ptr noalias align 16 dereferenceable(44590400) %0, ptr noalias align 256 dereferenceable(2048) %1, ptr noalias align 256 dereferenceable(787251200) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
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

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

define ptx_kernel void @input_transpose_fusion_1(ptr noalias align 256 dereferenceable(787251200) %0, ptr noalias align 256 dereferenceable(787251200) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
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
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
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
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
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
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
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

define ptx_kernel void @input_transpose_fusion(ptr noalias align 256 dereferenceable(787251200) %0, ptr noalias align 256 dereferenceable(787251200) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %5 = urem i32 %4, 5
  %6 = mul i32 %5, 32
  %7 = urem i32 %3, 32
  %8 = add i32 %6, %7
  %9 = icmp sle i32 %8, 154
  br i1 %9, label %10, label %51

10:                                               ; preds = %2
  %11 = udiv i32 %4, 5
  %12 = urem i32 %11, 512
  %13 = mul i32 %12, 155
  %14 = udiv i32 %4, 2560
  %15 = urem i32 %14, 10
  %16 = mul i32 %15, 2539520
  %17 = add i32 %13, %16
  %18 = add i32 %17, %6
  %19 = udiv i32 %4, 25600
  %20 = mul i32 %19, 24601600
  %21 = add i32 %18, %20
  %22 = udiv i32 %3, 32
  %23 = mul i32 %22, 79360
  %24 = add i32 %21, %23
  %25 = add i32 %24, %7
  %26 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %25
  %27 = load { double, double }, ptr %26, align 8, !invariant.load !3
  %28 = mul i32 %7, 33
  %29 = add i32 %28, %22
  %30 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %29
  store { double, double } %27, ptr %30, align 8
  %31 = add i32 %25, 317440
  %32 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %31
  %33 = load { double, double }, ptr %32, align 8, !invariant.load !3
  %34 = add i32 %29, 4
  %35 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %34
  store { double, double } %33, ptr %35, align 8
  %36 = add i32 %25, 634880
  %37 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %36
  %38 = load { double, double }, ptr %37, align 8, !invariant.load !3
  %39 = add i32 %29, 8
  %40 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %39
  store { double, double } %38, ptr %40, align 8
  %41 = add i32 %25, 952320
  %42 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %41
  %43 = load { double, double }, ptr %42, align 8, !invariant.load !3
  %44 = add i32 %29, 12
  %45 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %44
  store { double, double } %43, ptr %45, align 8
  %46 = add i32 %25, 1269760
  %47 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %46
  %48 = load { double, double }, ptr %47, align 8, !invariant.load !3
  %49 = add i32 %29, 16
  %50 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %49
  store { double, double } %48, ptr %50, align 8
  br label %51

51:                                               ; preds = %10, %2
  %52 = udiv i32 %4, 2560
  %53 = urem i32 %52, 10
  %54 = mul i32 %53, 32
  %55 = udiv i32 %3, 32
  %56 = add i32 %54, %55
  %57 = add i32 %56, 20
  %58 = icmp sle i32 %57, 309
  %59 = and i1 %58, %9
  br i1 %59, label %60, label %80

60:                                               ; preds = %51
  %61 = udiv i32 %4, 5
  %62 = urem i32 %61, 512
  %63 = mul i32 %62, 155
  %64 = mul i32 %53, 2539520
  %65 = add i32 %63, %64
  %66 = add i32 %65, %6
  %67 = udiv i32 %4, 25600
  %68 = mul i32 %67, 24601600
  %69 = add i32 %66, %68
  %70 = mul i32 %55, 79360
  %71 = add i32 %69, %70
  %72 = add i32 %71, %7
  %73 = add i32 %72, 1587200
  %74 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %73
  %75 = load { double, double }, ptr %74, align 8, !invariant.load !3
  %76 = mul i32 %7, 33
  %77 = add i32 %76, %55
  %78 = add i32 %77, 20
  %79 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %78
  store { double, double } %75, ptr %79, align 8
  br label %80

80:                                               ; preds = %60, %51
  %81 = add i32 %56, 24
  %82 = icmp sle i32 %81, 309
  %83 = and i1 %82, %9
  br i1 %83, label %84, label %104

84:                                               ; preds = %80
  %85 = udiv i32 %4, 5
  %86 = urem i32 %85, 512
  %87 = mul i32 %86, 155
  %88 = mul i32 %53, 2539520
  %89 = add i32 %87, %88
  %90 = add i32 %89, %6
  %91 = udiv i32 %4, 25600
  %92 = mul i32 %91, 24601600
  %93 = add i32 %90, %92
  %94 = mul i32 %55, 79360
  %95 = add i32 %93, %94
  %96 = add i32 %95, %7
  %97 = add i32 %96, 1904640
  %98 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %97
  %99 = load { double, double }, ptr %98, align 8, !invariant.load !3
  %100 = mul i32 %7, 33
  %101 = add i32 %100, %55
  %102 = add i32 %101, 24
  %103 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %102
  store { double, double } %99, ptr %103, align 8
  br label %104

104:                                              ; preds = %84, %80
  %105 = add i32 %56, 28
  %106 = icmp sle i32 %105, 309
  %107 = and i1 %106, %9
  br i1 %107, label %108, label %128

108:                                              ; preds = %104
  %109 = udiv i32 %4, 5
  %110 = urem i32 %109, 512
  %111 = mul i32 %110, 155
  %112 = mul i32 %53, 2539520
  %113 = add i32 %111, %112
  %114 = add i32 %113, %6
  %115 = udiv i32 %4, 25600
  %116 = mul i32 %115, 24601600
  %117 = add i32 %114, %116
  %118 = mul i32 %55, 79360
  %119 = add i32 %117, %118
  %120 = add i32 %119, %7
  %121 = add i32 %120, 2222080
  %122 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %121
  %123 = load { double, double }, ptr %122, align 8, !invariant.load !3
  %124 = mul i32 %7, 33
  %125 = add i32 %124, %55
  %126 = add i32 %125, 28
  %127 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %126
  store { double, double } %123, ptr %127, align 8
  br label %128

128:                                              ; preds = %108, %104
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %129 = add i32 %54, %7
  %130 = icmp sle i32 %129, 309
  br i1 %130, label %131, label %174

131:                                              ; preds = %128
  %132 = mul i32 %55, 33
  %133 = add i32 %132, %7
  %134 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %133
  %135 = load { double, double }, ptr %134, align 8
  %136 = udiv i32 %4, 5
  %137 = urem i32 %136, 512
  %138 = mul i32 %137, 96100
  %139 = add i32 %138, %54
  %140 = mul i32 %5, 9920
  %141 = add i32 %139, %140
  %142 = udiv i32 %4, 25600
  %143 = mul i32 %142, 48050
  %144 = add i32 %141, %143
  %145 = mul i32 %55, 310
  %146 = add i32 %144, %145
  %147 = add i32 %146, %7
  %148 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %147
  store { double, double } %135, ptr %148, align 8
  %149 = add i32 %133, 132
  %150 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %149
  %151 = load { double, double }, ptr %150, align 8
  %152 = add i32 %147, 1240
  %153 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %152
  store { double, double } %151, ptr %153, align 8
  %154 = add i32 %133, 264
  %155 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %154
  %156 = load { double, double }, ptr %155, align 8
  %157 = add i32 %147, 2480
  %158 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %157
  store { double, double } %156, ptr %158, align 8
  %159 = add i32 %133, 396
  %160 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %159
  %161 = load { double, double }, ptr %160, align 8
  %162 = add i32 %147, 3720
  %163 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %162
  store { double, double } %161, ptr %163, align 8
  %164 = add i32 %133, 528
  %165 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %164
  %166 = load { double, double }, ptr %165, align 8
  %167 = add i32 %147, 4960
  %168 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %167
  store { double, double } %166, ptr %168, align 8
  %169 = add i32 %133, 660
  %170 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %169
  %171 = load { double, double }, ptr %170, align 8
  %172 = add i32 %147, 6200
  %173 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %172
  store { double, double } %171, ptr %173, align 8
  br label %174

174:                                              ; preds = %131, %128
  %175 = add i32 %6, %55
  %176 = add i32 %175, 24
  %177 = icmp sle i32 %176, 154
  %178 = and i1 %177, %130
  br i1 %178, label %179, label %199

179:                                              ; preds = %174
  %180 = mul i32 %55, 33
  %181 = add i32 %180, %7
  %182 = add i32 %181, 792
  %183 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %182
  %184 = load { double, double }, ptr %183, align 8
  %185 = udiv i32 %4, 5
  %186 = urem i32 %185, 512
  %187 = mul i32 %186, 96100
  %188 = add i32 %187, %54
  %189 = mul i32 %5, 9920
  %190 = add i32 %188, %189
  %191 = udiv i32 %4, 25600
  %192 = mul i32 %191, 48050
  %193 = add i32 %190, %192
  %194 = mul i32 %55, 310
  %195 = add i32 %193, %194
  %196 = add i32 %195, %7
  %197 = add i32 %196, 7440
  %198 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %197
  store { double, double } %184, ptr %198, align 8
  br label %199

199:                                              ; preds = %179, %174
  %200 = add i32 %175, 28
  %201 = icmp sle i32 %200, 154
  %202 = and i1 %201, %130
  br i1 %202, label %203, label %223

203:                                              ; preds = %199
  %204 = mul i32 %55, 33
  %205 = add i32 %204, %7
  %206 = add i32 %205, 924
  %207 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %206
  %208 = load { double, double }, ptr %207, align 8
  %209 = udiv i32 %4, 5
  %210 = urem i32 %209, 512
  %211 = mul i32 %210, 96100
  %212 = add i32 %211, %54
  %213 = mul i32 %5, 9920
  %214 = add i32 %212, %213
  %215 = udiv i32 %4, 25600
  %216 = mul i32 %215, 48050
  %217 = add i32 %214, %216
  %218 = mul i32 %55, 310
  %219 = add i32 %217, %218
  %220 = add i32 %219, %7
  %221 = add i32 %220, 8680
  %222 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %221
  store { double, double } %208, ptr %222, align 8
  br label %223

223:                                              ; preds = %203, %199
  ret void
}

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 96100}
!2 = !{i32 0, i32 128}
!3 = !{}
!4 = !{i32 0, i32 51200}
