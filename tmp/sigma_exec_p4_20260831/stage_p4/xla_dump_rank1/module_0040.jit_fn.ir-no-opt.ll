; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_2 = private addrspace(3) global [16 x i64] undef
@shared_1 = private addrspace(3) global [16 x i64] undef
@shared_0 = private addrspace(3) global [16 x double] undef
@global_smem = external addrspace(3) global [0 x i8], align 16
@shared_01 = private addrspace(3) global [13 x i64] undef

declare double @__nv_fabs(double)

declare double @__nv_sqrt(double)

define ptx_kernel void @input_reduce_fusion(ptr noalias align 16 dereferenceable(787251200) %0, ptr noalias align 256 dereferenceable(51200) %1, ptr noalias align 256 dereferenceable(51200) %2, ptr noalias align 256 dereferenceable(51200) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %7 = mul i32 %6, 7688
  %8 = add i32 %7, %5
  %9 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %8
  %10 = load { double, double }, ptr %9, align 8, !invariant.load !3
  %11 = extractvalue { double, double } %10, 0
  %12 = fcmp une double %11, %11
  %13 = extractvalue { double, double } %10, 1
  %14 = fcmp une double %13, %13
  %15 = or i1 %12, %14
  %16 = zext i1 %15 to i64
  %17 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 0, i64 %16)
  %18 = call double @__nv_fabs(double %11)
  %19 = fcmp one double %18, 0x7FF0000000000000
  %20 = call double @__nv_fabs(double %13)
  %21 = fcmp one double %20, 0x7FF0000000000000
  %22 = and i1 %19, %21
  %23 = zext i1 %22 to i8
  %24 = icmp eq i8 %23, 0
  %25 = zext i1 %24 to i64
  %26 = call i64 @region_0_1_reduce_sum_2_0(i64 0, i64 %25)
  %27 = call double @llvm.maximum.f64(double %18, double %20)
  %28 = call double @llvm.minimum.f64(double %18, double %20)
  %29 = fdiv double %28, %27
  %30 = fmul double %29, %29
  %31 = fadd double %30, 1.000000e+00
  %32 = call double @__nv_sqrt(double %31)
  %33 = fmul double %27, %32
  %34 = fcmp uno double %33, %33
  %35 = select i1 %34, double %28, double %33
  %36 = select i1 %22, double %35, double 0.000000e+00
  %37 = call double @region_2_3_reduce_max_2_0(double 0xFFF0000000000000, double %36)
  %38 = add i32 %8, 512
  %39 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %38
  %40 = load { double, double }, ptr %39, align 8, !invariant.load !3
  %41 = extractvalue { double, double } %40, 0
  %42 = fcmp une double %41, %41
  %43 = extractvalue { double, double } %40, 1
  %44 = fcmp une double %43, %43
  %45 = or i1 %42, %44
  %46 = zext i1 %45 to i64
  %47 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %17, i64 %46)
  %48 = call double @__nv_fabs(double %41)
  %49 = fcmp one double %48, 0x7FF0000000000000
  %50 = call double @__nv_fabs(double %43)
  %51 = fcmp one double %50, 0x7FF0000000000000
  %52 = and i1 %49, %51
  %53 = zext i1 %52 to i8
  %54 = icmp eq i8 %53, 0
  %55 = zext i1 %54 to i64
  %56 = call i64 @region_0_1_reduce_sum_2_0(i64 %26, i64 %55)
  %57 = call double @llvm.maximum.f64(double %48, double %50)
  %58 = call double @llvm.minimum.f64(double %48, double %50)
  %59 = fdiv double %58, %57
  %60 = fmul double %59, %59
  %61 = fadd double %60, 1.000000e+00
  %62 = call double @__nv_sqrt(double %61)
  %63 = fmul double %57, %62
  %64 = fcmp uno double %63, %63
  %65 = select i1 %64, double %58, double %63
  %66 = select i1 %52, double %65, double 0.000000e+00
  %67 = call double @region_2_3_reduce_max_2_0(double %37, double %66)
  %68 = add i32 %8, 1024
  %69 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %68
  %70 = load { double, double }, ptr %69, align 8, !invariant.load !3
  %71 = extractvalue { double, double } %70, 0
  %72 = fcmp une double %71, %71
  %73 = extractvalue { double, double } %70, 1
  %74 = fcmp une double %73, %73
  %75 = or i1 %72, %74
  %76 = zext i1 %75 to i64
  %77 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %47, i64 %76)
  %78 = call double @__nv_fabs(double %71)
  %79 = fcmp one double %78, 0x7FF0000000000000
  %80 = call double @__nv_fabs(double %73)
  %81 = fcmp one double %80, 0x7FF0000000000000
  %82 = and i1 %79, %81
  %83 = zext i1 %82 to i8
  %84 = icmp eq i8 %83, 0
  %85 = zext i1 %84 to i64
  %86 = call i64 @region_0_1_reduce_sum_2_0(i64 %56, i64 %85)
  %87 = call double @llvm.maximum.f64(double %78, double %80)
  %88 = call double @llvm.minimum.f64(double %78, double %80)
  %89 = fdiv double %88, %87
  %90 = fmul double %89, %89
  %91 = fadd double %90, 1.000000e+00
  %92 = call double @__nv_sqrt(double %91)
  %93 = fmul double %87, %92
  %94 = fcmp uno double %93, %93
  %95 = select i1 %94, double %88, double %93
  %96 = select i1 %82, double %95, double 0.000000e+00
  %97 = call double @region_2_3_reduce_max_2_0(double %67, double %96)
  %98 = add i32 %8, 1536
  %99 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %98
  %100 = load { double, double }, ptr %99, align 8, !invariant.load !3
  %101 = extractvalue { double, double } %100, 0
  %102 = fcmp une double %101, %101
  %103 = extractvalue { double, double } %100, 1
  %104 = fcmp une double %103, %103
  %105 = or i1 %102, %104
  %106 = zext i1 %105 to i64
  %107 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %77, i64 %106)
  %108 = call double @__nv_fabs(double %101)
  %109 = fcmp one double %108, 0x7FF0000000000000
  %110 = call double @__nv_fabs(double %103)
  %111 = fcmp one double %110, 0x7FF0000000000000
  %112 = and i1 %109, %111
  %113 = zext i1 %112 to i8
  %114 = icmp eq i8 %113, 0
  %115 = zext i1 %114 to i64
  %116 = call i64 @region_0_1_reduce_sum_2_0(i64 %86, i64 %115)
  %117 = call double @llvm.maximum.f64(double %108, double %110)
  %118 = call double @llvm.minimum.f64(double %108, double %110)
  %119 = fdiv double %118, %117
  %120 = fmul double %119, %119
  %121 = fadd double %120, 1.000000e+00
  %122 = call double @__nv_sqrt(double %121)
  %123 = fmul double %117, %122
  %124 = fcmp uno double %123, %123
  %125 = select i1 %124, double %118, double %123
  %126 = select i1 %112, double %125, double 0.000000e+00
  %127 = call double @region_2_3_reduce_max_2_0(double %97, double %126)
  %128 = add i32 %8, 2048
  %129 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %128
  %130 = load { double, double }, ptr %129, align 8, !invariant.load !3
  %131 = extractvalue { double, double } %130, 0
  %132 = fcmp une double %131, %131
  %133 = extractvalue { double, double } %130, 1
  %134 = fcmp une double %133, %133
  %135 = or i1 %132, %134
  %136 = zext i1 %135 to i64
  %137 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %107, i64 %136)
  %138 = call double @__nv_fabs(double %131)
  %139 = fcmp one double %138, 0x7FF0000000000000
  %140 = call double @__nv_fabs(double %133)
  %141 = fcmp one double %140, 0x7FF0000000000000
  %142 = and i1 %139, %141
  %143 = zext i1 %142 to i8
  %144 = icmp eq i8 %143, 0
  %145 = zext i1 %144 to i64
  %146 = call i64 @region_0_1_reduce_sum_2_0(i64 %116, i64 %145)
  %147 = call double @llvm.maximum.f64(double %138, double %140)
  %148 = call double @llvm.minimum.f64(double %138, double %140)
  %149 = fdiv double %148, %147
  %150 = fmul double %149, %149
  %151 = fadd double %150, 1.000000e+00
  %152 = call double @__nv_sqrt(double %151)
  %153 = fmul double %147, %152
  %154 = fcmp uno double %153, %153
  %155 = select i1 %154, double %148, double %153
  %156 = select i1 %142, double %155, double 0.000000e+00
  %157 = call double @region_2_3_reduce_max_2_0(double %127, double %156)
  %158 = add i32 %8, 2560
  %159 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %158
  %160 = load { double, double }, ptr %159, align 8, !invariant.load !3
  %161 = extractvalue { double, double } %160, 0
  %162 = fcmp une double %161, %161
  %163 = extractvalue { double, double } %160, 1
  %164 = fcmp une double %163, %163
  %165 = or i1 %162, %164
  %166 = zext i1 %165 to i64
  %167 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %137, i64 %166)
  %168 = call double @__nv_fabs(double %161)
  %169 = fcmp one double %168, 0x7FF0000000000000
  %170 = call double @__nv_fabs(double %163)
  %171 = fcmp one double %170, 0x7FF0000000000000
  %172 = and i1 %169, %171
  %173 = zext i1 %172 to i8
  %174 = icmp eq i8 %173, 0
  %175 = zext i1 %174 to i64
  %176 = call i64 @region_0_1_reduce_sum_2_0(i64 %146, i64 %175)
  %177 = call double @llvm.maximum.f64(double %168, double %170)
  %178 = call double @llvm.minimum.f64(double %168, double %170)
  %179 = fdiv double %178, %177
  %180 = fmul double %179, %179
  %181 = fadd double %180, 1.000000e+00
  %182 = call double @__nv_sqrt(double %181)
  %183 = fmul double %177, %182
  %184 = fcmp uno double %183, %183
  %185 = select i1 %184, double %178, double %183
  %186 = select i1 %172, double %185, double 0.000000e+00
  %187 = call double @region_2_3_reduce_max_2_0(double %157, double %186)
  %188 = add i32 %8, 3072
  %189 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %188
  %190 = load { double, double }, ptr %189, align 8, !invariant.load !3
  %191 = extractvalue { double, double } %190, 0
  %192 = fcmp une double %191, %191
  %193 = extractvalue { double, double } %190, 1
  %194 = fcmp une double %193, %193
  %195 = or i1 %192, %194
  %196 = zext i1 %195 to i64
  %197 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %167, i64 %196)
  %198 = call double @__nv_fabs(double %191)
  %199 = fcmp one double %198, 0x7FF0000000000000
  %200 = call double @__nv_fabs(double %193)
  %201 = fcmp one double %200, 0x7FF0000000000000
  %202 = and i1 %199, %201
  %203 = zext i1 %202 to i8
  %204 = icmp eq i8 %203, 0
  %205 = zext i1 %204 to i64
  %206 = call i64 @region_0_1_reduce_sum_2_0(i64 %176, i64 %205)
  %207 = call double @llvm.maximum.f64(double %198, double %200)
  %208 = call double @llvm.minimum.f64(double %198, double %200)
  %209 = fdiv double %208, %207
  %210 = fmul double %209, %209
  %211 = fadd double %210, 1.000000e+00
  %212 = call double @__nv_sqrt(double %211)
  %213 = fmul double %207, %212
  %214 = fcmp uno double %213, %213
  %215 = select i1 %214, double %208, double %213
  %216 = select i1 %202, double %215, double 0.000000e+00
  %217 = call double @region_2_3_reduce_max_2_0(double %187, double %216)
  %218 = add i32 %8, 3584
  %219 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %218
  %220 = load { double, double }, ptr %219, align 8, !invariant.load !3
  %221 = extractvalue { double, double } %220, 0
  %222 = fcmp une double %221, %221
  %223 = extractvalue { double, double } %220, 1
  %224 = fcmp une double %223, %223
  %225 = or i1 %222, %224
  %226 = zext i1 %225 to i64
  %227 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %197, i64 %226)
  %228 = call double @__nv_fabs(double %221)
  %229 = fcmp one double %228, 0x7FF0000000000000
  %230 = call double @__nv_fabs(double %223)
  %231 = fcmp one double %230, 0x7FF0000000000000
  %232 = and i1 %229, %231
  %233 = zext i1 %232 to i8
  %234 = icmp eq i8 %233, 0
  %235 = zext i1 %234 to i64
  %236 = call i64 @region_0_1_reduce_sum_2_0(i64 %206, i64 %235)
  %237 = call double @llvm.maximum.f64(double %228, double %230)
  %238 = call double @llvm.minimum.f64(double %228, double %230)
  %239 = fdiv double %238, %237
  %240 = fmul double %239, %239
  %241 = fadd double %240, 1.000000e+00
  %242 = call double @__nv_sqrt(double %241)
  %243 = fmul double %237, %242
  %244 = fcmp uno double %243, %243
  %245 = select i1 %244, double %238, double %243
  %246 = select i1 %232, double %245, double 0.000000e+00
  %247 = call double @region_2_3_reduce_max_2_0(double %217, double %246)
  %248 = add i32 %8, 4096
  %249 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %248
  %250 = load { double, double }, ptr %249, align 8, !invariant.load !3
  %251 = extractvalue { double, double } %250, 0
  %252 = fcmp une double %251, %251
  %253 = extractvalue { double, double } %250, 1
  %254 = fcmp une double %253, %253
  %255 = or i1 %252, %254
  %256 = zext i1 %255 to i64
  %257 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %227, i64 %256)
  %258 = call double @__nv_fabs(double %251)
  %259 = fcmp one double %258, 0x7FF0000000000000
  %260 = call double @__nv_fabs(double %253)
  %261 = fcmp one double %260, 0x7FF0000000000000
  %262 = and i1 %259, %261
  %263 = zext i1 %262 to i8
  %264 = icmp eq i8 %263, 0
  %265 = zext i1 %264 to i64
  %266 = call i64 @region_0_1_reduce_sum_2_0(i64 %236, i64 %265)
  %267 = call double @llvm.maximum.f64(double %258, double %260)
  %268 = call double @llvm.minimum.f64(double %258, double %260)
  %269 = fdiv double %268, %267
  %270 = fmul double %269, %269
  %271 = fadd double %270, 1.000000e+00
  %272 = call double @__nv_sqrt(double %271)
  %273 = fmul double %267, %272
  %274 = fcmp uno double %273, %273
  %275 = select i1 %274, double %268, double %273
  %276 = select i1 %262, double %275, double 0.000000e+00
  %277 = call double @region_2_3_reduce_max_2_0(double %247, double %276)
  %278 = add i32 %8, 4608
  %279 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %278
  %280 = load { double, double }, ptr %279, align 8, !invariant.load !3
  %281 = extractvalue { double, double } %280, 0
  %282 = fcmp une double %281, %281
  %283 = extractvalue { double, double } %280, 1
  %284 = fcmp une double %283, %283
  %285 = or i1 %282, %284
  %286 = zext i1 %285 to i64
  %287 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %257, i64 %286)
  %288 = call double @__nv_fabs(double %281)
  %289 = fcmp one double %288, 0x7FF0000000000000
  %290 = call double @__nv_fabs(double %283)
  %291 = fcmp one double %290, 0x7FF0000000000000
  %292 = and i1 %289, %291
  %293 = zext i1 %292 to i8
  %294 = icmp eq i8 %293, 0
  %295 = zext i1 %294 to i64
  %296 = call i64 @region_0_1_reduce_sum_2_0(i64 %266, i64 %295)
  %297 = call double @llvm.maximum.f64(double %288, double %290)
  %298 = call double @llvm.minimum.f64(double %288, double %290)
  %299 = fdiv double %298, %297
  %300 = fmul double %299, %299
  %301 = fadd double %300, 1.000000e+00
  %302 = call double @__nv_sqrt(double %301)
  %303 = fmul double %297, %302
  %304 = fcmp uno double %303, %303
  %305 = select i1 %304, double %298, double %303
  %306 = select i1 %292, double %305, double 0.000000e+00
  %307 = call double @region_2_3_reduce_max_2_0(double %277, double %306)
  %308 = add i32 %8, 5120
  %309 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %308
  %310 = load { double, double }, ptr %309, align 8, !invariant.load !3
  %311 = extractvalue { double, double } %310, 0
  %312 = fcmp une double %311, %311
  %313 = extractvalue { double, double } %310, 1
  %314 = fcmp une double %313, %313
  %315 = or i1 %312, %314
  %316 = zext i1 %315 to i64
  %317 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %287, i64 %316)
  %318 = call double @__nv_fabs(double %311)
  %319 = fcmp one double %318, 0x7FF0000000000000
  %320 = call double @__nv_fabs(double %313)
  %321 = fcmp one double %320, 0x7FF0000000000000
  %322 = and i1 %319, %321
  %323 = zext i1 %322 to i8
  %324 = icmp eq i8 %323, 0
  %325 = zext i1 %324 to i64
  %326 = call i64 @region_0_1_reduce_sum_2_0(i64 %296, i64 %325)
  %327 = call double @llvm.maximum.f64(double %318, double %320)
  %328 = call double @llvm.minimum.f64(double %318, double %320)
  %329 = fdiv double %328, %327
  %330 = fmul double %329, %329
  %331 = fadd double %330, 1.000000e+00
  %332 = call double @__nv_sqrt(double %331)
  %333 = fmul double %327, %332
  %334 = fcmp uno double %333, %333
  %335 = select i1 %334, double %328, double %333
  %336 = select i1 %322, double %335, double 0.000000e+00
  %337 = call double @region_2_3_reduce_max_2_0(double %307, double %336)
  %338 = add i32 %8, 5632
  %339 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %338
  %340 = load { double, double }, ptr %339, align 8, !invariant.load !3
  %341 = extractvalue { double, double } %340, 0
  %342 = fcmp une double %341, %341
  %343 = extractvalue { double, double } %340, 1
  %344 = fcmp une double %343, %343
  %345 = or i1 %342, %344
  %346 = zext i1 %345 to i64
  %347 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %317, i64 %346)
  %348 = call double @__nv_fabs(double %341)
  %349 = fcmp one double %348, 0x7FF0000000000000
  %350 = call double @__nv_fabs(double %343)
  %351 = fcmp one double %350, 0x7FF0000000000000
  %352 = and i1 %349, %351
  %353 = zext i1 %352 to i8
  %354 = icmp eq i8 %353, 0
  %355 = zext i1 %354 to i64
  %356 = call i64 @region_0_1_reduce_sum_2_0(i64 %326, i64 %355)
  %357 = call double @llvm.maximum.f64(double %348, double %350)
  %358 = call double @llvm.minimum.f64(double %348, double %350)
  %359 = fdiv double %358, %357
  %360 = fmul double %359, %359
  %361 = fadd double %360, 1.000000e+00
  %362 = call double @__nv_sqrt(double %361)
  %363 = fmul double %357, %362
  %364 = fcmp uno double %363, %363
  %365 = select i1 %364, double %358, double %363
  %366 = select i1 %352, double %365, double 0.000000e+00
  %367 = call double @region_2_3_reduce_max_2_0(double %337, double %366)
  %368 = add i32 %8, 6144
  %369 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %368
  %370 = load { double, double }, ptr %369, align 8, !invariant.load !3
  %371 = extractvalue { double, double } %370, 0
  %372 = fcmp une double %371, %371
  %373 = extractvalue { double, double } %370, 1
  %374 = fcmp une double %373, %373
  %375 = or i1 %372, %374
  %376 = zext i1 %375 to i64
  %377 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %347, i64 %376)
  %378 = call double @__nv_fabs(double %371)
  %379 = fcmp one double %378, 0x7FF0000000000000
  %380 = call double @__nv_fabs(double %373)
  %381 = fcmp one double %380, 0x7FF0000000000000
  %382 = and i1 %379, %381
  %383 = zext i1 %382 to i8
  %384 = icmp eq i8 %383, 0
  %385 = zext i1 %384 to i64
  %386 = call i64 @region_0_1_reduce_sum_2_0(i64 %356, i64 %385)
  %387 = call double @llvm.maximum.f64(double %378, double %380)
  %388 = call double @llvm.minimum.f64(double %378, double %380)
  %389 = fdiv double %388, %387
  %390 = fmul double %389, %389
  %391 = fadd double %390, 1.000000e+00
  %392 = call double @__nv_sqrt(double %391)
  %393 = fmul double %387, %392
  %394 = fcmp uno double %393, %393
  %395 = select i1 %394, double %388, double %393
  %396 = select i1 %382, double %395, double 0.000000e+00
  %397 = call double @region_2_3_reduce_max_2_0(double %367, double %396)
  %398 = add i32 %8, 6656
  %399 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %398
  %400 = load { double, double }, ptr %399, align 8, !invariant.load !3
  %401 = extractvalue { double, double } %400, 0
  %402 = fcmp une double %401, %401
  %403 = extractvalue { double, double } %400, 1
  %404 = fcmp une double %403, %403
  %405 = or i1 %402, %404
  %406 = zext i1 %405 to i64
  %407 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %377, i64 %406)
  %408 = call double @__nv_fabs(double %401)
  %409 = fcmp one double %408, 0x7FF0000000000000
  %410 = call double @__nv_fabs(double %403)
  %411 = fcmp one double %410, 0x7FF0000000000000
  %412 = and i1 %409, %411
  %413 = zext i1 %412 to i8
  %414 = icmp eq i8 %413, 0
  %415 = zext i1 %414 to i64
  %416 = call i64 @region_0_1_reduce_sum_2_0(i64 %386, i64 %415)
  %417 = call double @llvm.maximum.f64(double %408, double %410)
  %418 = call double @llvm.minimum.f64(double %408, double %410)
  %419 = fdiv double %418, %417
  %420 = fmul double %419, %419
  %421 = fadd double %420, 1.000000e+00
  %422 = call double @__nv_sqrt(double %421)
  %423 = fmul double %417, %422
  %424 = fcmp uno double %423, %423
  %425 = select i1 %424, double %418, double %423
  %426 = select i1 %412, double %425, double 0.000000e+00
  %427 = call double @region_2_3_reduce_max_2_0(double %397, double %426)
  %428 = add i32 %8, 7168
  %429 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %428
  %430 = load { double, double }, ptr %429, align 8, !invariant.load !3
  %431 = extractvalue { double, double } %430, 0
  %432 = fcmp une double %431, %431
  %433 = extractvalue { double, double } %430, 1
  %434 = fcmp une double %433, %433
  %435 = or i1 %432, %434
  %436 = zext i1 %435 to i64
  %437 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %407, i64 %436)
  %438 = call double @__nv_fabs(double %431)
  %439 = fcmp one double %438, 0x7FF0000000000000
  %440 = call double @__nv_fabs(double %433)
  %441 = fcmp one double %440, 0x7FF0000000000000
  %442 = and i1 %439, %441
  %443 = zext i1 %442 to i8
  %444 = icmp eq i8 %443, 0
  %445 = zext i1 %444 to i64
  %446 = call i64 @region_0_1_reduce_sum_2_0(i64 %416, i64 %445)
  %447 = call double @llvm.maximum.f64(double %438, double %440)
  %448 = call double @llvm.minimum.f64(double %438, double %440)
  %449 = fdiv double %448, %447
  %450 = fmul double %449, %449
  %451 = fadd double %450, 1.000000e+00
  %452 = call double @__nv_sqrt(double %451)
  %453 = fmul double %447, %452
  %454 = fcmp uno double %453, %453
  %455 = select i1 %454, double %448, double %453
  %456 = select i1 %442, double %455, double 0.000000e+00
  %457 = call double @region_2_3_reduce_max_2_0(double %427, double %456)
  %458 = icmp sle i32 %5, 7
  br i1 %458, label %459, label %490

459:                                              ; preds = %4
  %460 = add i32 %8, 7680
  %461 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %460
  %462 = load { double, double }, ptr %461, align 8, !invariant.load !3
  %463 = extractvalue { double, double } %462, 0
  %464 = fcmp une double %463, %463
  %465 = extractvalue { double, double } %462, 1
  %466 = fcmp une double %465, %465
  %467 = or i1 %464, %466
  %468 = zext i1 %467 to i64
  %469 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %437, i64 %468)
  %470 = call double @__nv_fabs(double %463)
  %471 = fcmp one double %470, 0x7FF0000000000000
  %472 = call double @__nv_fabs(double %465)
  %473 = fcmp one double %472, 0x7FF0000000000000
  %474 = and i1 %471, %473
  %475 = zext i1 %474 to i8
  %476 = icmp eq i8 %475, 0
  %477 = zext i1 %476 to i64
  %478 = call i64 @region_0_1_reduce_sum_2_0(i64 %446, i64 %477)
  %479 = call double @llvm.maximum.f64(double %470, double %472)
  %480 = call double @llvm.minimum.f64(double %470, double %472)
  %481 = fdiv double %480, %479
  %482 = fmul double %481, %481
  %483 = fadd double %482, 1.000000e+00
  %484 = call double @__nv_sqrt(double %483)
  %485 = fmul double %479, %484
  %486 = fcmp uno double %485, %485
  %487 = select i1 %486, double %480, double %485
  %488 = select i1 %474, double %487, double 0.000000e+00
  %489 = call double @region_2_3_reduce_max_2_0(double %457, double %488)
  br label %491

490:                                              ; preds = %4
  br label %491

491:                                              ; preds = %459, %490
  %492 = phi i64 [ %437, %490 ], [ %469, %459 ]
  %493 = phi i64 [ %446, %490 ], [ %478, %459 ]
  %494 = phi double [ %457, %490 ], [ %489, %459 ]
  br label %495

495:                                              ; preds = %491
  %496 = bitcast i64 %492 to <2 x i32>
  %497 = extractelement <2 x i32> %496, i32 0
  %498 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %497, i32 16, i32 31)
  %499 = insertelement <2 x i32> undef, i32 %498, i32 0
  %500 = extractelement <2 x i32> %496, i32 1
  %501 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %500, i32 16, i32 31)
  %502 = insertelement <2 x i32> %499, i32 %501, i32 1
  %503 = bitcast <2 x i32> %502 to i64
  %504 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %492, i64 %503)
  %505 = bitcast i64 %504 to <2 x i32>
  %506 = extractelement <2 x i32> %505, i32 0
  %507 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %506, i32 8, i32 31)
  %508 = insertelement <2 x i32> undef, i32 %507, i32 0
  %509 = extractelement <2 x i32> %505, i32 1
  %510 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %509, i32 8, i32 31)
  %511 = insertelement <2 x i32> %508, i32 %510, i32 1
  %512 = bitcast <2 x i32> %511 to i64
  %513 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %504, i64 %512)
  %514 = bitcast i64 %513 to <2 x i32>
  %515 = extractelement <2 x i32> %514, i32 0
  %516 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %515, i32 4, i32 31)
  %517 = insertelement <2 x i32> undef, i32 %516, i32 0
  %518 = extractelement <2 x i32> %514, i32 1
  %519 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %518, i32 4, i32 31)
  %520 = insertelement <2 x i32> %517, i32 %519, i32 1
  %521 = bitcast <2 x i32> %520 to i64
  %522 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %513, i64 %521)
  %523 = bitcast i64 %522 to <2 x i32>
  %524 = extractelement <2 x i32> %523, i32 0
  %525 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %524, i32 2, i32 31)
  %526 = insertelement <2 x i32> undef, i32 %525, i32 0
  %527 = extractelement <2 x i32> %523, i32 1
  %528 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %527, i32 2, i32 31)
  %529 = insertelement <2 x i32> %526, i32 %528, i32 1
  %530 = bitcast <2 x i32> %529 to i64
  %531 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %522, i64 %530)
  %532 = bitcast i64 %531 to <2 x i32>
  %533 = extractelement <2 x i32> %532, i32 0
  %534 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %533, i32 1, i32 31)
  %535 = insertelement <2 x i32> undef, i32 %534, i32 0
  %536 = extractelement <2 x i32> %532, i32 1
  %537 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %536, i32 1, i32 31)
  %538 = insertelement <2 x i32> %535, i32 %537, i32 1
  %539 = bitcast <2 x i32> %538 to i64
  %540 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %531, i64 %539)
  %541 = bitcast i64 %493 to <2 x i32>
  %542 = extractelement <2 x i32> %541, i32 0
  %543 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %542, i32 16, i32 31)
  %544 = insertelement <2 x i32> undef, i32 %543, i32 0
  %545 = extractelement <2 x i32> %541, i32 1
  %546 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %545, i32 16, i32 31)
  %547 = insertelement <2 x i32> %544, i32 %546, i32 1
  %548 = bitcast <2 x i32> %547 to i64
  %549 = call i64 @region_0_1_reduce_sum_2_0(i64 %493, i64 %548)
  %550 = bitcast i64 %549 to <2 x i32>
  %551 = extractelement <2 x i32> %550, i32 0
  %552 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %551, i32 8, i32 31)
  %553 = insertelement <2 x i32> undef, i32 %552, i32 0
  %554 = extractelement <2 x i32> %550, i32 1
  %555 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %554, i32 8, i32 31)
  %556 = insertelement <2 x i32> %553, i32 %555, i32 1
  %557 = bitcast <2 x i32> %556 to i64
  %558 = call i64 @region_0_1_reduce_sum_2_0(i64 %549, i64 %557)
  %559 = bitcast i64 %558 to <2 x i32>
  %560 = extractelement <2 x i32> %559, i32 0
  %561 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %560, i32 4, i32 31)
  %562 = insertelement <2 x i32> undef, i32 %561, i32 0
  %563 = extractelement <2 x i32> %559, i32 1
  %564 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %563, i32 4, i32 31)
  %565 = insertelement <2 x i32> %562, i32 %564, i32 1
  %566 = bitcast <2 x i32> %565 to i64
  %567 = call i64 @region_0_1_reduce_sum_2_0(i64 %558, i64 %566)
  %568 = bitcast i64 %567 to <2 x i32>
  %569 = extractelement <2 x i32> %568, i32 0
  %570 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %569, i32 2, i32 31)
  %571 = insertelement <2 x i32> undef, i32 %570, i32 0
  %572 = extractelement <2 x i32> %568, i32 1
  %573 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %572, i32 2, i32 31)
  %574 = insertelement <2 x i32> %571, i32 %573, i32 1
  %575 = bitcast <2 x i32> %574 to i64
  %576 = call i64 @region_0_1_reduce_sum_2_0(i64 %567, i64 %575)
  %577 = bitcast i64 %576 to <2 x i32>
  %578 = extractelement <2 x i32> %577, i32 0
  %579 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %578, i32 1, i32 31)
  %580 = insertelement <2 x i32> undef, i32 %579, i32 0
  %581 = extractelement <2 x i32> %577, i32 1
  %582 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %581, i32 1, i32 31)
  %583 = insertelement <2 x i32> %580, i32 %582, i32 1
  %584 = bitcast <2 x i32> %583 to i64
  %585 = call i64 @region_0_1_reduce_sum_2_0(i64 %576, i64 %584)
  %586 = bitcast double %494 to i64
  %587 = bitcast i64 %586 to <2 x i32>
  %588 = extractelement <2 x i32> %587, i32 0
  %589 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %588, i32 16, i32 31)
  %590 = insertelement <2 x i32> undef, i32 %589, i32 0
  %591 = extractelement <2 x i32> %587, i32 1
  %592 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %591, i32 16, i32 31)
  %593 = insertelement <2 x i32> %590, i32 %592, i32 1
  %594 = bitcast <2 x i32> %593 to double
  %595 = call double @region_2_3_reduce_max_2_0(double %494, double %594)
  %596 = bitcast double %595 to i64
  %597 = bitcast i64 %596 to <2 x i32>
  %598 = extractelement <2 x i32> %597, i32 0
  %599 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %598, i32 8, i32 31)
  %600 = insertelement <2 x i32> undef, i32 %599, i32 0
  %601 = extractelement <2 x i32> %597, i32 1
  %602 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %601, i32 8, i32 31)
  %603 = insertelement <2 x i32> %600, i32 %602, i32 1
  %604 = bitcast <2 x i32> %603 to double
  %605 = call double @region_2_3_reduce_max_2_0(double %595, double %604)
  %606 = bitcast double %605 to i64
  %607 = bitcast i64 %606 to <2 x i32>
  %608 = extractelement <2 x i32> %607, i32 0
  %609 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %608, i32 4, i32 31)
  %610 = insertelement <2 x i32> undef, i32 %609, i32 0
  %611 = extractelement <2 x i32> %607, i32 1
  %612 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %611, i32 4, i32 31)
  %613 = insertelement <2 x i32> %610, i32 %612, i32 1
  %614 = bitcast <2 x i32> %613 to double
  %615 = call double @region_2_3_reduce_max_2_0(double %605, double %614)
  %616 = bitcast double %615 to i64
  %617 = bitcast i64 %616 to <2 x i32>
  %618 = extractelement <2 x i32> %617, i32 0
  %619 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %618, i32 2, i32 31)
  %620 = insertelement <2 x i32> undef, i32 %619, i32 0
  %621 = extractelement <2 x i32> %617, i32 1
  %622 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %621, i32 2, i32 31)
  %623 = insertelement <2 x i32> %620, i32 %622, i32 1
  %624 = bitcast <2 x i32> %623 to double
  %625 = call double @region_2_3_reduce_max_2_0(double %615, double %624)
  %626 = bitcast double %625 to i64
  %627 = bitcast i64 %626 to <2 x i32>
  %628 = extractelement <2 x i32> %627, i32 0
  %629 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %628, i32 1, i32 31)
  %630 = insertelement <2 x i32> undef, i32 %629, i32 0
  %631 = extractelement <2 x i32> %627, i32 1
  %632 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %631, i32 1, i32 31)
  %633 = insertelement <2 x i32> %630, i32 %632, i32 1
  %634 = bitcast <2 x i32> %633 to double
  %635 = call double @region_2_3_reduce_max_2_0(double %625, double %634)
  %636 = urem i32 %5, 32
  %637 = icmp eq i32 %636, 0
  br i1 %637, label %638, label %643

638:                                              ; preds = %495
  %639 = udiv i32 %5, 32
  %640 = getelementptr inbounds [16 x i64], ptr addrspacecast (ptr addrspace(3) @shared_2 to ptr), i32 0, i32 %639
  store i64 %540, ptr %640, align 4
  %641 = getelementptr inbounds [16 x i64], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %639
  store i64 %585, ptr %641, align 4
  %642 = getelementptr inbounds [16 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %639
  store double %635, ptr %642, align 8
  br label %643

643:                                              ; preds = %638, %495
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %644 = icmp sle i32 %5, 31
  br i1 %644, label %645, label %806

645:                                              ; preds = %643
  %646 = icmp sle i32 %5, 15
  br i1 %646, label %647, label %654

647:                                              ; preds = %645
  %648 = getelementptr inbounds [16 x i64], ptr addrspacecast (ptr addrspace(3) @shared_2 to ptr), i32 0, i32 %5
  %649 = load i64, ptr %648, align 4
  %650 = getelementptr inbounds [16 x i64], ptr addrspacecast (ptr addrspace(3) @shared_1 to ptr), i32 0, i32 %5
  %651 = load i64, ptr %650, align 4
  %652 = getelementptr inbounds [16 x double], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %5
  %653 = load double, ptr %652, align 8
  br label %655

654:                                              ; preds = %645
  br label %655

655:                                              ; preds = %647, %654
  %656 = phi i64 [ 0, %654 ], [ %649, %647 ]
  %657 = phi i64 [ 0, %654 ], [ %651, %647 ]
  %658 = phi double [ 0xFFF0000000000000, %654 ], [ %653, %647 ]
  br label %659

659:                                              ; preds = %655
  %660 = bitcast i64 %656 to <2 x i32>
  %661 = extractelement <2 x i32> %660, i32 0
  %662 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %661, i32 16, i32 31)
  %663 = insertelement <2 x i32> undef, i32 %662, i32 0
  %664 = extractelement <2 x i32> %660, i32 1
  %665 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %664, i32 16, i32 31)
  %666 = insertelement <2 x i32> %663, i32 %665, i32 1
  %667 = bitcast <2 x i32> %666 to i64
  %668 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %656, i64 %667)
  %669 = bitcast i64 %668 to <2 x i32>
  %670 = extractelement <2 x i32> %669, i32 0
  %671 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %670, i32 8, i32 31)
  %672 = insertelement <2 x i32> undef, i32 %671, i32 0
  %673 = extractelement <2 x i32> %669, i32 1
  %674 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %673, i32 8, i32 31)
  %675 = insertelement <2 x i32> %672, i32 %674, i32 1
  %676 = bitcast <2 x i32> %675 to i64
  %677 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %668, i64 %676)
  %678 = bitcast i64 %677 to <2 x i32>
  %679 = extractelement <2 x i32> %678, i32 0
  %680 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %679, i32 4, i32 31)
  %681 = insertelement <2 x i32> undef, i32 %680, i32 0
  %682 = extractelement <2 x i32> %678, i32 1
  %683 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %682, i32 4, i32 31)
  %684 = insertelement <2 x i32> %681, i32 %683, i32 1
  %685 = bitcast <2 x i32> %684 to i64
  %686 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %677, i64 %685)
  %687 = bitcast i64 %686 to <2 x i32>
  %688 = extractelement <2 x i32> %687, i32 0
  %689 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %688, i32 2, i32 31)
  %690 = insertelement <2 x i32> undef, i32 %689, i32 0
  %691 = extractelement <2 x i32> %687, i32 1
  %692 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %691, i32 2, i32 31)
  %693 = insertelement <2 x i32> %690, i32 %692, i32 1
  %694 = bitcast <2 x i32> %693 to i64
  %695 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %686, i64 %694)
  %696 = bitcast i64 %695 to <2 x i32>
  %697 = extractelement <2 x i32> %696, i32 0
  %698 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %697, i32 1, i32 31)
  %699 = insertelement <2 x i32> undef, i32 %698, i32 0
  %700 = extractelement <2 x i32> %696, i32 1
  %701 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %700, i32 1, i32 31)
  %702 = insertelement <2 x i32> %699, i32 %701, i32 1
  %703 = bitcast <2 x i32> %702 to i64
  %704 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %695, i64 %703)
  %705 = bitcast i64 %657 to <2 x i32>
  %706 = extractelement <2 x i32> %705, i32 0
  %707 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %706, i32 16, i32 31)
  %708 = insertelement <2 x i32> undef, i32 %707, i32 0
  %709 = extractelement <2 x i32> %705, i32 1
  %710 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %709, i32 16, i32 31)
  %711 = insertelement <2 x i32> %708, i32 %710, i32 1
  %712 = bitcast <2 x i32> %711 to i64
  %713 = call i64 @region_0_1_reduce_sum_2_0(i64 %657, i64 %712)
  %714 = bitcast i64 %713 to <2 x i32>
  %715 = extractelement <2 x i32> %714, i32 0
  %716 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %715, i32 8, i32 31)
  %717 = insertelement <2 x i32> undef, i32 %716, i32 0
  %718 = extractelement <2 x i32> %714, i32 1
  %719 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %718, i32 8, i32 31)
  %720 = insertelement <2 x i32> %717, i32 %719, i32 1
  %721 = bitcast <2 x i32> %720 to i64
  %722 = call i64 @region_0_1_reduce_sum_2_0(i64 %713, i64 %721)
  %723 = bitcast i64 %722 to <2 x i32>
  %724 = extractelement <2 x i32> %723, i32 0
  %725 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %724, i32 4, i32 31)
  %726 = insertelement <2 x i32> undef, i32 %725, i32 0
  %727 = extractelement <2 x i32> %723, i32 1
  %728 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %727, i32 4, i32 31)
  %729 = insertelement <2 x i32> %726, i32 %728, i32 1
  %730 = bitcast <2 x i32> %729 to i64
  %731 = call i64 @region_0_1_reduce_sum_2_0(i64 %722, i64 %730)
  %732 = bitcast i64 %731 to <2 x i32>
  %733 = extractelement <2 x i32> %732, i32 0
  %734 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %733, i32 2, i32 31)
  %735 = insertelement <2 x i32> undef, i32 %734, i32 0
  %736 = extractelement <2 x i32> %732, i32 1
  %737 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %736, i32 2, i32 31)
  %738 = insertelement <2 x i32> %735, i32 %737, i32 1
  %739 = bitcast <2 x i32> %738 to i64
  %740 = call i64 @region_0_1_reduce_sum_2_0(i64 %731, i64 %739)
  %741 = bitcast i64 %740 to <2 x i32>
  %742 = extractelement <2 x i32> %741, i32 0
  %743 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %742, i32 1, i32 31)
  %744 = insertelement <2 x i32> undef, i32 %743, i32 0
  %745 = extractelement <2 x i32> %741, i32 1
  %746 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %745, i32 1, i32 31)
  %747 = insertelement <2 x i32> %744, i32 %746, i32 1
  %748 = bitcast <2 x i32> %747 to i64
  %749 = call i64 @region_0_1_reduce_sum_2_0(i64 %740, i64 %748)
  %750 = bitcast double %658 to i64
  %751 = bitcast i64 %750 to <2 x i32>
  %752 = extractelement <2 x i32> %751, i32 0
  %753 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %752, i32 16, i32 31)
  %754 = insertelement <2 x i32> undef, i32 %753, i32 0
  %755 = extractelement <2 x i32> %751, i32 1
  %756 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %755, i32 16, i32 31)
  %757 = insertelement <2 x i32> %754, i32 %756, i32 1
  %758 = bitcast <2 x i32> %757 to double
  %759 = call double @region_2_3_reduce_max_2_0(double %658, double %758)
  %760 = bitcast double %759 to i64
  %761 = bitcast i64 %760 to <2 x i32>
  %762 = extractelement <2 x i32> %761, i32 0
  %763 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %762, i32 8, i32 31)
  %764 = insertelement <2 x i32> undef, i32 %763, i32 0
  %765 = extractelement <2 x i32> %761, i32 1
  %766 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %765, i32 8, i32 31)
  %767 = insertelement <2 x i32> %764, i32 %766, i32 1
  %768 = bitcast <2 x i32> %767 to double
  %769 = call double @region_2_3_reduce_max_2_0(double %759, double %768)
  %770 = bitcast double %769 to i64
  %771 = bitcast i64 %770 to <2 x i32>
  %772 = extractelement <2 x i32> %771, i32 0
  %773 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %772, i32 4, i32 31)
  %774 = insertelement <2 x i32> undef, i32 %773, i32 0
  %775 = extractelement <2 x i32> %771, i32 1
  %776 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %775, i32 4, i32 31)
  %777 = insertelement <2 x i32> %774, i32 %776, i32 1
  %778 = bitcast <2 x i32> %777 to double
  %779 = call double @region_2_3_reduce_max_2_0(double %769, double %778)
  %780 = bitcast double %779 to i64
  %781 = bitcast i64 %780 to <2 x i32>
  %782 = extractelement <2 x i32> %781, i32 0
  %783 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %782, i32 2, i32 31)
  %784 = insertelement <2 x i32> undef, i32 %783, i32 0
  %785 = extractelement <2 x i32> %781, i32 1
  %786 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %785, i32 2, i32 31)
  %787 = insertelement <2 x i32> %784, i32 %786, i32 1
  %788 = bitcast <2 x i32> %787 to double
  %789 = call double @region_2_3_reduce_max_2_0(double %779, double %788)
  %790 = bitcast double %789 to i64
  %791 = bitcast i64 %790 to <2 x i32>
  %792 = extractelement <2 x i32> %791, i32 0
  %793 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %792, i32 1, i32 31)
  %794 = insertelement <2 x i32> undef, i32 %793, i32 0
  %795 = extractelement <2 x i32> %791, i32 1
  %796 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %795, i32 1, i32 31)
  %797 = insertelement <2 x i32> %794, i32 %796, i32 1
  %798 = bitcast <2 x i32> %797 to double
  %799 = call double @region_2_3_reduce_max_2_0(double %789, double %798)
  %800 = icmp eq i32 %5, 0
  br i1 %800, label %801, label %805

801:                                              ; preds = %659
  %802 = getelementptr inbounds [6400 x i64], ptr %1, i32 0, i32 %6
  store i64 %704, ptr %802, align 4
  %803 = getelementptr inbounds [6400 x i64], ptr %2, i32 0, i32 %6
  store i64 %749, ptr %803, align 4
  %804 = getelementptr inbounds [6400 x double], ptr %3, i32 0, i32 %6
  store double %799, ptr %804, align 8
  br label %805

805:                                              ; preds = %801, %659
  br label %806

806:                                              ; preds = %805, %643
  ret void
}

define internal i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %0, i64 %1) {
  %3 = add i64 %0, %1
  ret i64 %3
}

define internal i64 @region_0_1_reduce_sum_2_0(i64 %0, i64 %1) {
  %3 = add i64 %0, %1
  ret i64 %3
}

define internal double @region_2_3_reduce_max_2_0(double %0, double %1) {
  %3 = call double @llvm.maximum.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #2

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #3

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #4

define ptx_kernel void @input_reduce_fusion_3(ptr noalias align 256 dereferenceable(51200) %arg0, ptr noalias align 256 dereferenceable(8) %arg1) #5 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = addrspacecast ptr null to ptr addrspace(1)
  %4 = addrspacecast ptr null to ptr addrspace(1)
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = and i32 %5, 255
  %7 = urem i32 %6, 32
  %8 = udiv i32 %5, 32
  %9 = shl i32 %7, 0
  %10 = or i32 0, %9
  %11 = shl i32 %8, 5
  %12 = or i32 %10, %11
  %13 = and i32 %12, 255
  %14 = shl i32 %13, 1
  %15 = or disjoint i32 %14, 0
  %16 = xor i32 0, %15
  %17 = xor i32 %16, 0
  %18 = xor i32 %16, 512
  %19 = xor i32 %16, 1024
  %20 = xor i32 %16, 1536
  %21 = xor i32 %16, 2048
  %22 = xor i32 %16, 2560
  %23 = xor i32 %16, 3072
  %24 = xor i32 %16, 3584
  %25 = xor i32 %16, 4096
  %26 = xor i32 %16, 4608
  %27 = xor i32 %16, 5120
  %28 = xor i32 %16, 5632
  %29 = xor i32 %16, 6144
  %30 = xor i32 %16, 6656
  %31 = xor i32 %16, 7168
  %32 = xor i32 %16, 7680
  %33 = add i32 %17, 0
  %34 = add i32 %18, 0
  %35 = add i32 %19, 0
  %36 = add i32 %20, 0
  %37 = add i32 %21, 0
  %38 = add i32 %22, 0
  %39 = add i32 %23, 0
  %40 = add i32 %24, 0
  %41 = add i32 %25, 0
  %42 = add i32 %26, 0
  %43 = add i32 %27, 0
  %44 = add i32 %28, 0
  %45 = add i32 %29, 0
  %46 = add i32 %30, 0
  %47 = add i32 %31, 0
  %48 = add i32 %32, 0
  %49 = sext i32 %33 to i64
  %50 = sext i32 %34 to i64
  %51 = sext i32 %35 to i64
  %52 = sext i32 %36 to i64
  %53 = sext i32 %37 to i64
  %54 = sext i32 %38 to i64
  %55 = sext i32 %39 to i64
  %56 = sext i32 %40 to i64
  %57 = sext i32 %41 to i64
  %58 = sext i32 %42 to i64
  %59 = sext i32 %43 to i64
  %60 = sext i32 %44 to i64
  %61 = sext i32 %45 to i64
  %62 = sext i32 %46 to i64
  %63 = sext i32 %47 to i64
  %64 = sext i32 %48 to i64
  %65 = icmp slt i64 %49, 6400
  %66 = icmp slt i64 %50, 6400
  %67 = icmp slt i64 %51, 6400
  %68 = icmp slt i64 %52, 6400
  %69 = icmp slt i64 %53, 6400
  %70 = icmp slt i64 %54, 6400
  %71 = icmp slt i64 %55, 6400
  %72 = icmp slt i64 %56, 6400
  %73 = icmp slt i64 %57, 6400
  %74 = icmp slt i64 %58, 6400
  %75 = icmp slt i64 %59, 6400
  %76 = icmp slt i64 %60, 6400
  %77 = icmp slt i64 %61, 6400
  %78 = icmp slt i64 %62, 6400
  %79 = icmp slt i64 %63, 6400
  %80 = icmp slt i64 %64, 6400
  %81 = getelementptr double, ptr addrspace(1) %1, i64 %49
  %82 = getelementptr double, ptr addrspace(1) %1, i64 %50
  %83 = getelementptr double, ptr addrspace(1) %1, i64 %51
  %84 = getelementptr double, ptr addrspace(1) %1, i64 %52
  %85 = getelementptr double, ptr addrspace(1) %1, i64 %53
  %86 = getelementptr double, ptr addrspace(1) %1, i64 %54
  %87 = getelementptr double, ptr addrspace(1) %1, i64 %55
  %88 = getelementptr double, ptr addrspace(1) %1, i64 %56
  %89 = getelementptr double, ptr addrspace(1) %1, i64 %57
  %90 = getelementptr double, ptr addrspace(1) %1, i64 %58
  %91 = getelementptr double, ptr addrspace(1) %1, i64 %59
  %92 = getelementptr double, ptr addrspace(1) %1, i64 %60
  %93 = getelementptr double, ptr addrspace(1) %1, i64 %61
  %94 = getelementptr double, ptr addrspace(1) %1, i64 %62
  %95 = getelementptr double, ptr addrspace(1) %1, i64 %63
  %96 = getelementptr double, ptr addrspace(1) %1, i64 %64
  %97 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %81, i1 %65)
  %98 = extractvalue { i64, i64 } %97, 0
  %99 = bitcast i64 %98 to <1 x double>
  %100 = extractvalue { i64, i64 } %97, 1
  %101 = bitcast i64 %100 to <1 x double>
  %102 = extractelement <1 x double> %99, i32 0
  %103 = extractelement <1 x double> %101, i32 0
  %104 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %82, i1 %66)
  %105 = extractvalue { i64, i64 } %104, 0
  %106 = bitcast i64 %105 to <1 x double>
  %107 = extractvalue { i64, i64 } %104, 1
  %108 = bitcast i64 %107 to <1 x double>
  %109 = extractelement <1 x double> %106, i32 0
  %110 = extractelement <1 x double> %108, i32 0
  %111 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %83, i1 %67)
  %112 = extractvalue { i64, i64 } %111, 0
  %113 = bitcast i64 %112 to <1 x double>
  %114 = extractvalue { i64, i64 } %111, 1
  %115 = bitcast i64 %114 to <1 x double>
  %116 = extractelement <1 x double> %113, i32 0
  %117 = extractelement <1 x double> %115, i32 0
  %118 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %84, i1 %68)
  %119 = extractvalue { i64, i64 } %118, 0
  %120 = bitcast i64 %119 to <1 x double>
  %121 = extractvalue { i64, i64 } %118, 1
  %122 = bitcast i64 %121 to <1 x double>
  %123 = extractelement <1 x double> %120, i32 0
  %124 = extractelement <1 x double> %122, i32 0
  %125 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %85, i1 %69)
  %126 = extractvalue { i64, i64 } %125, 0
  %127 = bitcast i64 %126 to <1 x double>
  %128 = extractvalue { i64, i64 } %125, 1
  %129 = bitcast i64 %128 to <1 x double>
  %130 = extractelement <1 x double> %127, i32 0
  %131 = extractelement <1 x double> %129, i32 0
  %132 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %86, i1 %70)
  %133 = extractvalue { i64, i64 } %132, 0
  %134 = bitcast i64 %133 to <1 x double>
  %135 = extractvalue { i64, i64 } %132, 1
  %136 = bitcast i64 %135 to <1 x double>
  %137 = extractelement <1 x double> %134, i32 0
  %138 = extractelement <1 x double> %136, i32 0
  %139 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %87, i1 %71)
  %140 = extractvalue { i64, i64 } %139, 0
  %141 = bitcast i64 %140 to <1 x double>
  %142 = extractvalue { i64, i64 } %139, 1
  %143 = bitcast i64 %142 to <1 x double>
  %144 = extractelement <1 x double> %141, i32 0
  %145 = extractelement <1 x double> %143, i32 0
  %146 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %88, i1 %72)
  %147 = extractvalue { i64, i64 } %146, 0
  %148 = bitcast i64 %147 to <1 x double>
  %149 = extractvalue { i64, i64 } %146, 1
  %150 = bitcast i64 %149 to <1 x double>
  %151 = extractelement <1 x double> %148, i32 0
  %152 = extractelement <1 x double> %150, i32 0
  %153 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %89, i1 %73)
  %154 = extractvalue { i64, i64 } %153, 0
  %155 = bitcast i64 %154 to <1 x double>
  %156 = extractvalue { i64, i64 } %153, 1
  %157 = bitcast i64 %156 to <1 x double>
  %158 = extractelement <1 x double> %155, i32 0
  %159 = extractelement <1 x double> %157, i32 0
  %160 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %90, i1 %74)
  %161 = extractvalue { i64, i64 } %160, 0
  %162 = bitcast i64 %161 to <1 x double>
  %163 = extractvalue { i64, i64 } %160, 1
  %164 = bitcast i64 %163 to <1 x double>
  %165 = extractelement <1 x double> %162, i32 0
  %166 = extractelement <1 x double> %164, i32 0
  %167 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %91, i1 %75)
  %168 = extractvalue { i64, i64 } %167, 0
  %169 = bitcast i64 %168 to <1 x double>
  %170 = extractvalue { i64, i64 } %167, 1
  %171 = bitcast i64 %170 to <1 x double>
  %172 = extractelement <1 x double> %169, i32 0
  %173 = extractelement <1 x double> %171, i32 0
  %174 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %92, i1 %76)
  %175 = extractvalue { i64, i64 } %174, 0
  %176 = bitcast i64 %175 to <1 x double>
  %177 = extractvalue { i64, i64 } %174, 1
  %178 = bitcast i64 %177 to <1 x double>
  %179 = extractelement <1 x double> %176, i32 0
  %180 = extractelement <1 x double> %178, i32 0
  %181 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %93, i1 %77)
  %182 = extractvalue { i64, i64 } %181, 0
  %183 = bitcast i64 %182 to <1 x double>
  %184 = extractvalue { i64, i64 } %181, 1
  %185 = bitcast i64 %184 to <1 x double>
  %186 = extractelement <1 x double> %183, i32 0
  %187 = extractelement <1 x double> %185, i32 0
  %188 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %94, i1 %78)
  %189 = extractvalue { i64, i64 } %188, 0
  %190 = bitcast i64 %189 to <1 x double>
  %191 = extractvalue { i64, i64 } %188, 1
  %192 = bitcast i64 %191 to <1 x double>
  %193 = extractelement <1 x double> %190, i32 0
  %194 = extractelement <1 x double> %192, i32 0
  %195 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %95, i1 %79)
  %196 = extractvalue { i64, i64 } %195, 0
  %197 = bitcast i64 %196 to <1 x double>
  %198 = extractvalue { i64, i64 } %195, 1
  %199 = bitcast i64 %198 to <1 x double>
  %200 = extractelement <1 x double> %197, i32 0
  %201 = extractelement <1 x double> %199, i32 0
  %202 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %96, i1 %80)
  %203 = extractvalue { i64, i64 } %202, 0
  %204 = bitcast i64 %203 to <1 x double>
  %205 = extractvalue { i64, i64 } %202, 1
  %206 = bitcast i64 %205 to <1 x double>
  %207 = extractelement <1 x double> %204, i32 0
  %208 = extractelement <1 x double> %206, i32 0
  %209 = icmp slt i32 %33, 6400
  %210 = icmp slt i32 %34, 6400
  %211 = icmp slt i32 %35, 6400
  %212 = icmp slt i32 %36, 6400
  %213 = icmp slt i32 %37, 6400
  %214 = icmp slt i32 %38, 6400
  %215 = icmp slt i32 %39, 6400
  %216 = icmp slt i32 %40, 6400
  %217 = icmp slt i32 %41, 6400
  %218 = icmp slt i32 %42, 6400
  %219 = icmp slt i32 %43, 6400
  %220 = icmp slt i32 %44, 6400
  %221 = icmp slt i32 %45, 6400
  %222 = icmp slt i32 %46, 6400
  %223 = icmp slt i32 %47, 6400
  %224 = icmp slt i32 %48, 6400
  %225 = select i1 %209, double %102, double 0xFFF0000000000000
  %226 = select i1 %209, double %103, double 0xFFF0000000000000
  %227 = select i1 %210, double %109, double 0xFFF0000000000000
  %228 = select i1 %210, double %110, double 0xFFF0000000000000
  %229 = select i1 %211, double %116, double 0xFFF0000000000000
  %230 = select i1 %211, double %117, double 0xFFF0000000000000
  %231 = select i1 %212, double %123, double 0xFFF0000000000000
  %232 = select i1 %212, double %124, double 0xFFF0000000000000
  %233 = select i1 %213, double %130, double 0xFFF0000000000000
  %234 = select i1 %213, double %131, double 0xFFF0000000000000
  %235 = select i1 %214, double %137, double 0xFFF0000000000000
  %236 = select i1 %214, double %138, double 0xFFF0000000000000
  %237 = select i1 %215, double %144, double 0xFFF0000000000000
  %238 = select i1 %215, double %145, double 0xFFF0000000000000
  %239 = select i1 %216, double %151, double 0xFFF0000000000000
  %240 = select i1 %216, double %152, double 0xFFF0000000000000
  %241 = select i1 %217, double %158, double 0xFFF0000000000000
  %242 = select i1 %217, double %159, double 0xFFF0000000000000
  %243 = select i1 %218, double %165, double 0xFFF0000000000000
  %244 = select i1 %218, double %166, double 0xFFF0000000000000
  %245 = select i1 %219, double %172, double 0xFFF0000000000000
  %246 = select i1 %219, double %173, double 0xFFF0000000000000
  %247 = select i1 %220, double %179, double 0xFFF0000000000000
  %248 = select i1 %220, double %180, double 0xFFF0000000000000
  %249 = select i1 %221, double %186, double 0xFFF0000000000000
  %250 = select i1 %221, double %187, double 0xFFF0000000000000
  %251 = select i1 %222, double %193, double 0xFFF0000000000000
  %252 = select i1 %222, double %194, double 0xFFF0000000000000
  %253 = select i1 %223, double %200, double 0xFFF0000000000000
  %254 = select i1 %223, double %201, double 0xFFF0000000000000
  %255 = select i1 %224, double %207, double 0xFFF0000000000000
  %256 = select i1 %224, double %208, double 0xFFF0000000000000
  %257 = call double @llvm.maximum.f64(double %225, double %226)
  %258 = call double @llvm.maximum.f64(double %227, double %228)
  %259 = call double @llvm.maximum.f64(double %229, double %230)
  %260 = call double @llvm.maximum.f64(double %231, double %232)
  %261 = call double @llvm.maximum.f64(double %233, double %234)
  %262 = call double @llvm.maximum.f64(double %235, double %236)
  %263 = call double @llvm.maximum.f64(double %237, double %238)
  %264 = call double @llvm.maximum.f64(double %239, double %240)
  %265 = call double @llvm.maximum.f64(double %241, double %242)
  %266 = call double @llvm.maximum.f64(double %243, double %244)
  %267 = call double @llvm.maximum.f64(double %245, double %246)
  %268 = call double @llvm.maximum.f64(double %247, double %248)
  %269 = call double @llvm.maximum.f64(double %249, double %250)
  %270 = call double @llvm.maximum.f64(double %251, double %252)
  %271 = call double @llvm.maximum.f64(double %253, double %254)
  %272 = call double @llvm.maximum.f64(double %255, double %256)
  %273 = call double @llvm.maximum.f64(double %257, double %258)
  %274 = call double @llvm.maximum.f64(double %259, double %260)
  %275 = call double @llvm.maximum.f64(double %261, double %262)
  %276 = call double @llvm.maximum.f64(double %263, double %264)
  %277 = call double @llvm.maximum.f64(double %265, double %266)
  %278 = call double @llvm.maximum.f64(double %267, double %268)
  %279 = call double @llvm.maximum.f64(double %269, double %270)
  %280 = call double @llvm.maximum.f64(double %271, double %272)
  %281 = call double @llvm.maximum.f64(double %273, double %274)
  %282 = call double @llvm.maximum.f64(double %275, double %276)
  %283 = call double @llvm.maximum.f64(double %277, double %278)
  %284 = call double @llvm.maximum.f64(double %279, double %280)
  %285 = call double @llvm.maximum.f64(double %281, double %282)
  %286 = call double @llvm.maximum.f64(double %283, double %284)
  %287 = call double @llvm.maximum.f64(double %285, double %286)
  %288 = bitcast double %287 to <2 x float>
  %289 = extractelement <2 x float> %288, i32 0
  %290 = extractelement <2 x float> %288, i32 1
  %291 = bitcast float %289 to i32
  %292 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %291, i32 16, i32 31)
  %293 = bitcast i32 %292 to float
  %294 = bitcast float %290 to i32
  %295 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %294, i32 16, i32 31)
  %296 = bitcast i32 %295 to float
  %297 = insertelement <2 x float> undef, float %293, i32 0
  %298 = insertelement <2 x float> %297, float %296, i32 1
  %299 = bitcast <2 x float> %298 to double
  %300 = call double @llvm.maximum.f64(double %287, double %299)
  %301 = bitcast double %300 to <2 x float>
  %302 = extractelement <2 x float> %301, i32 0
  %303 = extractelement <2 x float> %301, i32 1
  %304 = bitcast float %302 to i32
  %305 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %304, i32 8, i32 31)
  %306 = bitcast i32 %305 to float
  %307 = bitcast float %303 to i32
  %308 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %307, i32 8, i32 31)
  %309 = bitcast i32 %308 to float
  %310 = insertelement <2 x float> undef, float %306, i32 0
  %311 = insertelement <2 x float> %310, float %309, i32 1
  %312 = bitcast <2 x float> %311 to double
  %313 = call double @llvm.maximum.f64(double %300, double %312)
  %314 = bitcast double %313 to <2 x float>
  %315 = extractelement <2 x float> %314, i32 0
  %316 = extractelement <2 x float> %314, i32 1
  %317 = bitcast float %315 to i32
  %318 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %317, i32 4, i32 31)
  %319 = bitcast i32 %318 to float
  %320 = bitcast float %316 to i32
  %321 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %320, i32 4, i32 31)
  %322 = bitcast i32 %321 to float
  %323 = insertelement <2 x float> undef, float %319, i32 0
  %324 = insertelement <2 x float> %323, float %322, i32 1
  %325 = bitcast <2 x float> %324 to double
  %326 = call double @llvm.maximum.f64(double %313, double %325)
  %327 = bitcast double %326 to <2 x float>
  %328 = extractelement <2 x float> %327, i32 0
  %329 = extractelement <2 x float> %327, i32 1
  %330 = bitcast float %328 to i32
  %331 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %330, i32 2, i32 31)
  %332 = bitcast i32 %331 to float
  %333 = bitcast float %329 to i32
  %334 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %333, i32 2, i32 31)
  %335 = bitcast i32 %334 to float
  %336 = insertelement <2 x float> undef, float %332, i32 0
  %337 = insertelement <2 x float> %336, float %335, i32 1
  %338 = bitcast <2 x float> %337 to double
  %339 = call double @llvm.maximum.f64(double %326, double %338)
  %340 = bitcast double %339 to <2 x float>
  %341 = extractelement <2 x float> %340, i32 0
  %342 = extractelement <2 x float> %340, i32 1
  %343 = bitcast float %341 to i32
  %344 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %343, i32 1, i32 31)
  %345 = bitcast i32 %344 to float
  %346 = bitcast float %342 to i32
  %347 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %346, i32 1, i32 31)
  %348 = bitcast i32 %347 to float
  %349 = insertelement <2 x float> undef, float %345, i32 0
  %350 = insertelement <2 x float> %349, float %348, i32 1
  %351 = bitcast <2 x float> %350 to double
  %352 = call double @llvm.maximum.f64(double %339, double %351)
  %353 = and i32 %12, 224
  %354 = lshr i32 %353, 2
  %355 = or disjoint i32 0, %354
  %356 = xor i32 0, %355
  %357 = xor i32 %356, 0
  %358 = xor i32 %357, 0
  %359 = add i32 %358, 0
  %360 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %359
  %361 = insertelement <1 x double> undef, double %352, i32 0
  %362 = extractelement <1 x double> %361, i32 0
  %363 = bitcast double %362 to i64
  %364 = insertelement <1 x i64> undef, i64 %363, i32 0
  store <1 x i64> %364, ptr addrspace(3) %360, align 8
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %365 = and i32 %12, 7
  %366 = shl i32 %365, 3
  %367 = or disjoint i32 %366, 0
  %368 = xor i32 0, %367
  %369 = xor i32 %368, 0
  %370 = xor i32 %369, 0
  %371 = add i32 %370, 0
  %372 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %371
  %373 = load i64, ptr addrspace(3) %372, align 8
  %374 = insertelement <1 x i64> undef, i64 %373, i32 0
  %375 = extractelement <1 x i64> %374, i32 0
  %376 = bitcast i64 %375 to double
  %377 = insertelement <1 x double> undef, double %376, i32 0
  %378 = extractelement <1 x double> %377, i32 0
  %379 = bitcast double %378 to <2 x float>
  %380 = extractelement <2 x float> %379, i32 0
  %381 = extractelement <2 x float> %379, i32 1
  %382 = bitcast float %380 to i32
  %383 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %382, i32 4, i32 31)
  %384 = bitcast i32 %383 to float
  %385 = bitcast float %381 to i32
  %386 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %385, i32 4, i32 31)
  %387 = bitcast i32 %386 to float
  %388 = insertelement <2 x float> undef, float %384, i32 0
  %389 = insertelement <2 x float> %388, float %387, i32 1
  %390 = bitcast <2 x float> %389 to double
  %391 = call double @llvm.maximum.f64(double %378, double %390)
  %392 = bitcast double %391 to <2 x float>
  %393 = extractelement <2 x float> %392, i32 0
  %394 = extractelement <2 x float> %392, i32 1
  %395 = bitcast float %393 to i32
  %396 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %395, i32 2, i32 31)
  %397 = bitcast i32 %396 to float
  %398 = bitcast float %394 to i32
  %399 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %398, i32 2, i32 31)
  %400 = bitcast i32 %399 to float
  %401 = insertelement <2 x float> undef, float %397, i32 0
  %402 = insertelement <2 x float> %401, float %400, i32 1
  %403 = bitcast <2 x float> %402 to double
  %404 = call double @llvm.maximum.f64(double %391, double %403)
  %405 = bitcast double %404 to <2 x float>
  %406 = extractelement <2 x float> %405, i32 0
  %407 = extractelement <2 x float> %405, i32 1
  %408 = bitcast float %406 to i32
  %409 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %408, i32 1, i32 31)
  %410 = bitcast i32 %409 to float
  %411 = bitcast float %407 to i32
  %412 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %411, i32 1, i32 31)
  %413 = bitcast i32 %412 to float
  %414 = insertelement <2 x float> undef, float %410, i32 0
  %415 = insertelement <2 x float> %414, float %413, i32 1
  %416 = bitcast <2 x float> %415 to double
  %417 = call double @llvm.maximum.f64(double %404, double %416)
  %418 = and i32 %7, -1
  %419 = icmp eq i32 %418, 0
  %420 = and i32 %8, -1
  %421 = icmp eq i32 %420, 0
  %422 = and i1 %419, %421
  %423 = and i1 %422, true
  %424 = insertelement <1 x double> undef, double %417, i32 0
  %425 = bitcast <1 x double> %424 to i64
  call void asm sideeffect "@$2 st.global.b64 [ $1 + 0 ], { $0 };", "l,l,b"(i64 %425, ptr addrspace(1) %2, i1 %423)
  ret void
}

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #3

define ptx_kernel void @input_reduce_fusion_1(ptr noalias align 256 dereferenceable(51200) %0, ptr noalias align 256 dereferenceable(8) %1) #6 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %4 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %3
  %5 = load i64, ptr %4, align 4, !invariant.load !3
  %6 = call i64 @region_0_1_reduce_sum_2_02(i64 0, i64 %5)
  %7 = add i32 %3, 416
  %8 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %7
  %9 = load i64, ptr %8, align 4, !invariant.load !3
  %10 = call i64 @region_0_1_reduce_sum_2_02(i64 %6, i64 %9)
  %11 = add i32 %3, 832
  %12 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %11
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = call i64 @region_0_1_reduce_sum_2_02(i64 %10, i64 %13)
  %15 = add i32 %3, 1248
  %16 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %15
  %17 = load i64, ptr %16, align 4, !invariant.load !3
  %18 = call i64 @region_0_1_reduce_sum_2_02(i64 %14, i64 %17)
  %19 = add i32 %3, 1664
  %20 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %19
  %21 = load i64, ptr %20, align 4, !invariant.load !3
  %22 = call i64 @region_0_1_reduce_sum_2_02(i64 %18, i64 %21)
  %23 = add i32 %3, 2080
  %24 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %23
  %25 = load i64, ptr %24, align 4, !invariant.load !3
  %26 = call i64 @region_0_1_reduce_sum_2_02(i64 %22, i64 %25)
  %27 = add i32 %3, 2496
  %28 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %27
  %29 = load i64, ptr %28, align 4, !invariant.load !3
  %30 = call i64 @region_0_1_reduce_sum_2_02(i64 %26, i64 %29)
  %31 = add i32 %3, 2912
  %32 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %31
  %33 = load i64, ptr %32, align 4, !invariant.load !3
  %34 = call i64 @region_0_1_reduce_sum_2_02(i64 %30, i64 %33)
  %35 = add i32 %3, 3328
  %36 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %35
  %37 = load i64, ptr %36, align 4, !invariant.load !3
  %38 = call i64 @region_0_1_reduce_sum_2_02(i64 %34, i64 %37)
  %39 = add i32 %3, 3744
  %40 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %39
  %41 = load i64, ptr %40, align 4, !invariant.load !3
  %42 = call i64 @region_0_1_reduce_sum_2_02(i64 %38, i64 %41)
  %43 = add i32 %3, 4160
  %44 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %43
  %45 = load i64, ptr %44, align 4, !invariant.load !3
  %46 = call i64 @region_0_1_reduce_sum_2_02(i64 %42, i64 %45)
  %47 = add i32 %3, 4576
  %48 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %47
  %49 = load i64, ptr %48, align 4, !invariant.load !3
  %50 = call i64 @region_0_1_reduce_sum_2_02(i64 %46, i64 %49)
  %51 = add i32 %3, 4992
  %52 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %51
  %53 = load i64, ptr %52, align 4, !invariant.load !3
  %54 = call i64 @region_0_1_reduce_sum_2_02(i64 %50, i64 %53)
  %55 = add i32 %3, 5408
  %56 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %55
  %57 = load i64, ptr %56, align 4, !invariant.load !3
  %58 = call i64 @region_0_1_reduce_sum_2_02(i64 %54, i64 %57)
  %59 = add i32 %3, 5824
  %60 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %59
  %61 = load i64, ptr %60, align 4, !invariant.load !3
  %62 = call i64 @region_0_1_reduce_sum_2_02(i64 %58, i64 %61)
  %63 = icmp sle i32 %3, 159
  br i1 %63, label %64, label %69

64:                                               ; preds = %2
  %65 = add i32 %3, 6240
  %66 = getelementptr inbounds [6400 x i64], ptr %0, i32 0, i32 %65
  %67 = load i64, ptr %66, align 4, !invariant.load !3
  %68 = call i64 @region_0_1_reduce_sum_2_02(i64 %62, i64 %67)
  br label %70

69:                                               ; preds = %2
  br label %70

70:                                               ; preds = %64, %69
  %71 = phi i64 [ %62, %69 ], [ %68, %64 ]
  br label %72

72:                                               ; preds = %70
  %73 = bitcast i64 %71 to <2 x i32>
  %74 = extractelement <2 x i32> %73, i32 0
  %75 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %74, i32 16, i32 31)
  %76 = insertelement <2 x i32> undef, i32 %75, i32 0
  %77 = extractelement <2 x i32> %73, i32 1
  %78 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %77, i32 16, i32 31)
  %79 = insertelement <2 x i32> %76, i32 %78, i32 1
  %80 = bitcast <2 x i32> %79 to i64
  %81 = call i64 @region_0_1_reduce_sum_2_02(i64 %71, i64 %80)
  %82 = bitcast i64 %81 to <2 x i32>
  %83 = extractelement <2 x i32> %82, i32 0
  %84 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %83, i32 8, i32 31)
  %85 = insertelement <2 x i32> undef, i32 %84, i32 0
  %86 = extractelement <2 x i32> %82, i32 1
  %87 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %86, i32 8, i32 31)
  %88 = insertelement <2 x i32> %85, i32 %87, i32 1
  %89 = bitcast <2 x i32> %88 to i64
  %90 = call i64 @region_0_1_reduce_sum_2_02(i64 %81, i64 %89)
  %91 = bitcast i64 %90 to <2 x i32>
  %92 = extractelement <2 x i32> %91, i32 0
  %93 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %92, i32 4, i32 31)
  %94 = insertelement <2 x i32> undef, i32 %93, i32 0
  %95 = extractelement <2 x i32> %91, i32 1
  %96 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %95, i32 4, i32 31)
  %97 = insertelement <2 x i32> %94, i32 %96, i32 1
  %98 = bitcast <2 x i32> %97 to i64
  %99 = call i64 @region_0_1_reduce_sum_2_02(i64 %90, i64 %98)
  %100 = bitcast i64 %99 to <2 x i32>
  %101 = extractelement <2 x i32> %100, i32 0
  %102 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %101, i32 2, i32 31)
  %103 = insertelement <2 x i32> undef, i32 %102, i32 0
  %104 = extractelement <2 x i32> %100, i32 1
  %105 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %104, i32 2, i32 31)
  %106 = insertelement <2 x i32> %103, i32 %105, i32 1
  %107 = bitcast <2 x i32> %106 to i64
  %108 = call i64 @region_0_1_reduce_sum_2_02(i64 %99, i64 %107)
  %109 = bitcast i64 %108 to <2 x i32>
  %110 = extractelement <2 x i32> %109, i32 0
  %111 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %110, i32 1, i32 31)
  %112 = insertelement <2 x i32> undef, i32 %111, i32 0
  %113 = extractelement <2 x i32> %109, i32 1
  %114 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %113, i32 1, i32 31)
  %115 = insertelement <2 x i32> %112, i32 %114, i32 1
  %116 = bitcast <2 x i32> %115 to i64
  %117 = call i64 @region_0_1_reduce_sum_2_02(i64 %108, i64 %116)
  %118 = urem i32 %3, 32
  %119 = icmp eq i32 %118, 0
  br i1 %119, label %120, label %123

120:                                              ; preds = %72
  %121 = udiv i32 %3, 32
  %122 = getelementptr inbounds [13 x i64], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %121
  store i64 %117, ptr %122, align 4
  br label %123

123:                                              ; preds = %120, %72
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %124 = icmp sle i32 %3, 31
  br i1 %124, label %125, label %183

125:                                              ; preds = %123
  %126 = icmp sle i32 %3, 12
  br i1 %126, label %127, label %130

127:                                              ; preds = %125
  %128 = getelementptr inbounds [13 x i64], ptr addrspacecast (ptr addrspace(3) @shared_01 to ptr), i32 0, i32 %3
  %129 = load i64, ptr %128, align 4
  br label %131

130:                                              ; preds = %125
  br label %131

131:                                              ; preds = %127, %130
  %132 = phi i64 [ 0, %130 ], [ %129, %127 ]
  br label %133

133:                                              ; preds = %131
  %134 = bitcast i64 %132 to <2 x i32>
  %135 = extractelement <2 x i32> %134, i32 0
  %136 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %135, i32 16, i32 31)
  %137 = insertelement <2 x i32> undef, i32 %136, i32 0
  %138 = extractelement <2 x i32> %134, i32 1
  %139 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %138, i32 16, i32 31)
  %140 = insertelement <2 x i32> %137, i32 %139, i32 1
  %141 = bitcast <2 x i32> %140 to i64
  %142 = call i64 @region_0_1_reduce_sum_2_02(i64 %132, i64 %141)
  %143 = bitcast i64 %142 to <2 x i32>
  %144 = extractelement <2 x i32> %143, i32 0
  %145 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %144, i32 8, i32 31)
  %146 = insertelement <2 x i32> undef, i32 %145, i32 0
  %147 = extractelement <2 x i32> %143, i32 1
  %148 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %147, i32 8, i32 31)
  %149 = insertelement <2 x i32> %146, i32 %148, i32 1
  %150 = bitcast <2 x i32> %149 to i64
  %151 = call i64 @region_0_1_reduce_sum_2_02(i64 %142, i64 %150)
  %152 = bitcast i64 %151 to <2 x i32>
  %153 = extractelement <2 x i32> %152, i32 0
  %154 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %153, i32 4, i32 31)
  %155 = insertelement <2 x i32> undef, i32 %154, i32 0
  %156 = extractelement <2 x i32> %152, i32 1
  %157 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %156, i32 4, i32 31)
  %158 = insertelement <2 x i32> %155, i32 %157, i32 1
  %159 = bitcast <2 x i32> %158 to i64
  %160 = call i64 @region_0_1_reduce_sum_2_02(i64 %151, i64 %159)
  %161 = bitcast i64 %160 to <2 x i32>
  %162 = extractelement <2 x i32> %161, i32 0
  %163 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %162, i32 2, i32 31)
  %164 = insertelement <2 x i32> undef, i32 %163, i32 0
  %165 = extractelement <2 x i32> %161, i32 1
  %166 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %165, i32 2, i32 31)
  %167 = insertelement <2 x i32> %164, i32 %166, i32 1
  %168 = bitcast <2 x i32> %167 to i64
  %169 = call i64 @region_0_1_reduce_sum_2_02(i64 %160, i64 %168)
  %170 = bitcast i64 %169 to <2 x i32>
  %171 = extractelement <2 x i32> %170, i32 0
  %172 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %171, i32 1, i32 31)
  %173 = insertelement <2 x i32> undef, i32 %172, i32 0
  %174 = extractelement <2 x i32> %170, i32 1
  %175 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %174, i32 1, i32 31)
  %176 = insertelement <2 x i32> %173, i32 %175, i32 1
  %177 = bitcast <2 x i32> %176 to i64
  %178 = call i64 @region_0_1_reduce_sum_2_02(i64 %169, i64 %177)
  %179 = icmp eq i32 %3, 0
  br i1 %179, label %180, label %182

180:                                              ; preds = %133
  %181 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  store i64 %178, ptr %181, align 4
  br label %182

182:                                              ; preds = %180, %133
  br label %183

183:                                              ; preds = %182, %123
  ret void
}

define internal i64 @region_0_1_reduce_sum_2_02(i64 %0, i64 %1) {
  %3 = add i64 %0, %1
  ret i64 %3
}

define ptx_kernel void @wrapped_concatenate(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(16) %2) #7 {
  %4 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %5 = load i64, ptr %4, align 4, !invariant.load !3
  %6 = getelementptr inbounds [2 x i64], ptr %2, i32 0, i32 0
  store i64 %5, ptr %6, align 4
  %7 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %8 = load i64, ptr %7, align 4, !invariant.load !3
  %9 = getelementptr inbounds [2 x i64], ptr %2, i32 0, i32 1
  store i64 %8, ptr %9, align 4
  ret void
}

define ptx_kernel void @wrapped_slice_1(ptr noalias align 256 dereferenceable(16) %0, ptr noalias align 256 dereferenceable(8) %1) #7 {
  %3 = getelementptr inbounds [2 x i64], ptr %0, i32 0, i32 1
  %4 = load i64, ptr %3, align 4, !invariant.load !3
  %5 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  store i64 %4, ptr %5, align 4
  ret void
}

define ptx_kernel void @wrapped_slice(ptr noalias align 256 dereferenceable(16) %0, ptr noalias align 256 dereferenceable(8) %1) #7 {
  %3 = getelementptr inbounds [2 x i64], ptr %0, i32 0, i32 0
  %4 = load i64, ptr %3, align 4, !invariant.load !3
  %5 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  store i64 %4, ptr %5, align 4
  ret void
}

define ptx_kernel void @input_concatenate_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(24) %3) #7 {
  %5 = getelementptr inbounds [1 x i64], ptr %2, i32 0, i32 0
  %6 = load i64, ptr %5, align 4, !invariant.load !3
  %7 = sitofp i64 %6 to double
  %8 = getelementptr inbounds [3 x double], ptr %3, i32 0, i32 0
  store double %7, ptr %8, align 8
  %9 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %10 = load i64, ptr %9, align 4, !invariant.load !3
  %11 = sitofp i64 %10 to double
  %12 = getelementptr inbounds [3 x double], ptr %3, i32 0, i32 1
  store double %11, ptr %12, align 8
  %13 = getelementptr inbounds [1 x double], ptr %0, i32 0, i32 0
  %14 = load double, ptr %13, align 8, !invariant.load !3
  %15 = getelementptr inbounds [3 x double], ptr %3, i32 0, i32 2
  store double %14, ptr %15, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="512,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #4 = { convergent nocallback nounwind }
attributes #5 = { "nvvm.reqntid"="256,1,1" }
attributes #6 = { "nvvm.reqntid"="416,1,1" }
attributes #7 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}
!nvvm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 512}
!2 = !{i32 0, i32 6400}
!3 = !{}
!4 = !{i32 0, i32 416}
