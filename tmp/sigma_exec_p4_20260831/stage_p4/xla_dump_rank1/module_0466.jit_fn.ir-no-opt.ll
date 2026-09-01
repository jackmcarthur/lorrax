; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_fabs(double)

declare double @__nv_sqrt(double)

define ptx_kernel void @input_reduce_fusion(ptr noalias align 16 dereferenceable(1179648) %0, ptr noalias align 256 dereferenceable(2048) %1, ptr noalias align 256 dereferenceable(2048) %2, ptr noalias align 256 dereferenceable(2048) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %7 = udiv i32 %5, 32
  %8 = mul i32 %7, 288
  %9 = mul i32 %6, 2304
  %10 = add i32 %8, %9
  %11 = urem i32 %5, 32
  %12 = add i32 %10, %11
  %13 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %12
  %14 = load { double, double }, ptr %13, align 8, !invariant.load !3
  %15 = extractvalue { double, double } %14, 0
  %16 = fcmp une double %15, %15
  %17 = extractvalue { double, double } %14, 1
  %18 = fcmp une double %17, %17
  %19 = or i1 %16, %18
  %20 = zext i1 %19 to i64
  %21 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 0, i64 %20)
  %22 = call double @__nv_fabs(double %15)
  %23 = fcmp one double %22, 0x7FF0000000000000
  %24 = call double @__nv_fabs(double %17)
  %25 = fcmp one double %24, 0x7FF0000000000000
  %26 = and i1 %23, %25
  %27 = zext i1 %26 to i8
  %28 = icmp eq i8 %27, 0
  %29 = zext i1 %28 to i64
  %30 = call i64 @region_0_1_reduce_sum_2_0(i64 0, i64 %29)
  %31 = call double @llvm.maximum.f64(double %22, double %24)
  %32 = call double @llvm.minimum.f64(double %22, double %24)
  %33 = fdiv double %32, %31
  %34 = fmul double %33, %33
  %35 = fadd double %34, 1.000000e+00
  %36 = call double @__nv_sqrt(double %35)
  %37 = fmul double %31, %36
  %38 = fcmp uno double %37, %37
  %39 = select i1 %38, double %32, double %37
  %40 = select i1 %26, double %39, double 0.000000e+00
  %41 = call double @region_2_3_reduce_max_2_0(double 0xFFF0000000000000, double %40)
  %42 = add i32 %12, 32
  %43 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %42
  %44 = load { double, double }, ptr %43, align 8, !invariant.load !3
  %45 = extractvalue { double, double } %44, 0
  %46 = fcmp une double %45, %45
  %47 = extractvalue { double, double } %44, 1
  %48 = fcmp une double %47, %47
  %49 = or i1 %46, %48
  %50 = zext i1 %49 to i64
  %51 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %21, i64 %50)
  %52 = call double @__nv_fabs(double %45)
  %53 = fcmp one double %52, 0x7FF0000000000000
  %54 = call double @__nv_fabs(double %47)
  %55 = fcmp one double %54, 0x7FF0000000000000
  %56 = and i1 %53, %55
  %57 = zext i1 %56 to i8
  %58 = icmp eq i8 %57, 0
  %59 = zext i1 %58 to i64
  %60 = call i64 @region_0_1_reduce_sum_2_0(i64 %30, i64 %59)
  %61 = call double @llvm.maximum.f64(double %52, double %54)
  %62 = call double @llvm.minimum.f64(double %52, double %54)
  %63 = fdiv double %62, %61
  %64 = fmul double %63, %63
  %65 = fadd double %64, 1.000000e+00
  %66 = call double @__nv_sqrt(double %65)
  %67 = fmul double %61, %66
  %68 = fcmp uno double %67, %67
  %69 = select i1 %68, double %62, double %67
  %70 = select i1 %56, double %69, double 0.000000e+00
  %71 = call double @region_2_3_reduce_max_2_0(double %41, double %70)
  %72 = add i32 %12, 64
  %73 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %72
  %74 = load { double, double }, ptr %73, align 8, !invariant.load !3
  %75 = extractvalue { double, double } %74, 0
  %76 = fcmp une double %75, %75
  %77 = extractvalue { double, double } %74, 1
  %78 = fcmp une double %77, %77
  %79 = or i1 %76, %78
  %80 = zext i1 %79 to i64
  %81 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %51, i64 %80)
  %82 = call double @__nv_fabs(double %75)
  %83 = fcmp one double %82, 0x7FF0000000000000
  %84 = call double @__nv_fabs(double %77)
  %85 = fcmp one double %84, 0x7FF0000000000000
  %86 = and i1 %83, %85
  %87 = zext i1 %86 to i8
  %88 = icmp eq i8 %87, 0
  %89 = zext i1 %88 to i64
  %90 = call i64 @region_0_1_reduce_sum_2_0(i64 %60, i64 %89)
  %91 = call double @llvm.maximum.f64(double %82, double %84)
  %92 = call double @llvm.minimum.f64(double %82, double %84)
  %93 = fdiv double %92, %91
  %94 = fmul double %93, %93
  %95 = fadd double %94, 1.000000e+00
  %96 = call double @__nv_sqrt(double %95)
  %97 = fmul double %91, %96
  %98 = fcmp uno double %97, %97
  %99 = select i1 %98, double %92, double %97
  %100 = select i1 %86, double %99, double 0.000000e+00
  %101 = call double @region_2_3_reduce_max_2_0(double %71, double %100)
  %102 = add i32 %12, 96
  %103 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %102
  %104 = load { double, double }, ptr %103, align 8, !invariant.load !3
  %105 = extractvalue { double, double } %104, 0
  %106 = fcmp une double %105, %105
  %107 = extractvalue { double, double } %104, 1
  %108 = fcmp une double %107, %107
  %109 = or i1 %106, %108
  %110 = zext i1 %109 to i64
  %111 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %81, i64 %110)
  %112 = call double @__nv_fabs(double %105)
  %113 = fcmp one double %112, 0x7FF0000000000000
  %114 = call double @__nv_fabs(double %107)
  %115 = fcmp one double %114, 0x7FF0000000000000
  %116 = and i1 %113, %115
  %117 = zext i1 %116 to i8
  %118 = icmp eq i8 %117, 0
  %119 = zext i1 %118 to i64
  %120 = call i64 @region_0_1_reduce_sum_2_0(i64 %90, i64 %119)
  %121 = call double @llvm.maximum.f64(double %112, double %114)
  %122 = call double @llvm.minimum.f64(double %112, double %114)
  %123 = fdiv double %122, %121
  %124 = fmul double %123, %123
  %125 = fadd double %124, 1.000000e+00
  %126 = call double @__nv_sqrt(double %125)
  %127 = fmul double %121, %126
  %128 = fcmp uno double %127, %127
  %129 = select i1 %128, double %122, double %127
  %130 = select i1 %116, double %129, double 0.000000e+00
  %131 = call double @region_2_3_reduce_max_2_0(double %101, double %130)
  %132 = add i32 %12, 128
  %133 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %132
  %134 = load { double, double }, ptr %133, align 8, !invariant.load !3
  %135 = extractvalue { double, double } %134, 0
  %136 = fcmp une double %135, %135
  %137 = extractvalue { double, double } %134, 1
  %138 = fcmp une double %137, %137
  %139 = or i1 %136, %138
  %140 = zext i1 %139 to i64
  %141 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %111, i64 %140)
  %142 = call double @__nv_fabs(double %135)
  %143 = fcmp one double %142, 0x7FF0000000000000
  %144 = call double @__nv_fabs(double %137)
  %145 = fcmp one double %144, 0x7FF0000000000000
  %146 = and i1 %143, %145
  %147 = zext i1 %146 to i8
  %148 = icmp eq i8 %147, 0
  %149 = zext i1 %148 to i64
  %150 = call i64 @region_0_1_reduce_sum_2_0(i64 %120, i64 %149)
  %151 = call double @llvm.maximum.f64(double %142, double %144)
  %152 = call double @llvm.minimum.f64(double %142, double %144)
  %153 = fdiv double %152, %151
  %154 = fmul double %153, %153
  %155 = fadd double %154, 1.000000e+00
  %156 = call double @__nv_sqrt(double %155)
  %157 = fmul double %151, %156
  %158 = fcmp uno double %157, %157
  %159 = select i1 %158, double %152, double %157
  %160 = select i1 %146, double %159, double 0.000000e+00
  %161 = call double @region_2_3_reduce_max_2_0(double %131, double %160)
  %162 = add i32 %12, 160
  %163 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %162
  %164 = load { double, double }, ptr %163, align 8, !invariant.load !3
  %165 = extractvalue { double, double } %164, 0
  %166 = fcmp une double %165, %165
  %167 = extractvalue { double, double } %164, 1
  %168 = fcmp une double %167, %167
  %169 = or i1 %166, %168
  %170 = zext i1 %169 to i64
  %171 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %141, i64 %170)
  %172 = call double @__nv_fabs(double %165)
  %173 = fcmp one double %172, 0x7FF0000000000000
  %174 = call double @__nv_fabs(double %167)
  %175 = fcmp one double %174, 0x7FF0000000000000
  %176 = and i1 %173, %175
  %177 = zext i1 %176 to i8
  %178 = icmp eq i8 %177, 0
  %179 = zext i1 %178 to i64
  %180 = call i64 @region_0_1_reduce_sum_2_0(i64 %150, i64 %179)
  %181 = call double @llvm.maximum.f64(double %172, double %174)
  %182 = call double @llvm.minimum.f64(double %172, double %174)
  %183 = fdiv double %182, %181
  %184 = fmul double %183, %183
  %185 = fadd double %184, 1.000000e+00
  %186 = call double @__nv_sqrt(double %185)
  %187 = fmul double %181, %186
  %188 = fcmp uno double %187, %187
  %189 = select i1 %188, double %182, double %187
  %190 = select i1 %176, double %189, double 0.000000e+00
  %191 = call double @region_2_3_reduce_max_2_0(double %161, double %190)
  %192 = add i32 %12, 192
  %193 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %192
  %194 = load { double, double }, ptr %193, align 8, !invariant.load !3
  %195 = extractvalue { double, double } %194, 0
  %196 = fcmp une double %195, %195
  %197 = extractvalue { double, double } %194, 1
  %198 = fcmp une double %197, %197
  %199 = or i1 %196, %198
  %200 = zext i1 %199 to i64
  %201 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %171, i64 %200)
  %202 = call double @__nv_fabs(double %195)
  %203 = fcmp one double %202, 0x7FF0000000000000
  %204 = call double @__nv_fabs(double %197)
  %205 = fcmp one double %204, 0x7FF0000000000000
  %206 = and i1 %203, %205
  %207 = zext i1 %206 to i8
  %208 = icmp eq i8 %207, 0
  %209 = zext i1 %208 to i64
  %210 = call i64 @region_0_1_reduce_sum_2_0(i64 %180, i64 %209)
  %211 = call double @llvm.maximum.f64(double %202, double %204)
  %212 = call double @llvm.minimum.f64(double %202, double %204)
  %213 = fdiv double %212, %211
  %214 = fmul double %213, %213
  %215 = fadd double %214, 1.000000e+00
  %216 = call double @__nv_sqrt(double %215)
  %217 = fmul double %211, %216
  %218 = fcmp uno double %217, %217
  %219 = select i1 %218, double %212, double %217
  %220 = select i1 %206, double %219, double 0.000000e+00
  %221 = call double @region_2_3_reduce_max_2_0(double %191, double %220)
  %222 = add i32 %12, 224
  %223 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %222
  %224 = load { double, double }, ptr %223, align 8, !invariant.load !3
  %225 = extractvalue { double, double } %224, 0
  %226 = fcmp une double %225, %225
  %227 = extractvalue { double, double } %224, 1
  %228 = fcmp une double %227, %227
  %229 = or i1 %226, %228
  %230 = zext i1 %229 to i64
  %231 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %201, i64 %230)
  %232 = call double @__nv_fabs(double %225)
  %233 = fcmp one double %232, 0x7FF0000000000000
  %234 = call double @__nv_fabs(double %227)
  %235 = fcmp one double %234, 0x7FF0000000000000
  %236 = and i1 %233, %235
  %237 = zext i1 %236 to i8
  %238 = icmp eq i8 %237, 0
  %239 = zext i1 %238 to i64
  %240 = call i64 @region_0_1_reduce_sum_2_0(i64 %210, i64 %239)
  %241 = call double @llvm.maximum.f64(double %232, double %234)
  %242 = call double @llvm.minimum.f64(double %232, double %234)
  %243 = fdiv double %242, %241
  %244 = fmul double %243, %243
  %245 = fadd double %244, 1.000000e+00
  %246 = call double @__nv_sqrt(double %245)
  %247 = fmul double %241, %246
  %248 = fcmp uno double %247, %247
  %249 = select i1 %248, double %242, double %247
  %250 = select i1 %236, double %249, double 0.000000e+00
  %251 = call double @region_2_3_reduce_max_2_0(double %221, double %250)
  %252 = add i32 %12, 256
  %253 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %252
  %254 = load { double, double }, ptr %253, align 8, !invariant.load !3
  %255 = extractvalue { double, double } %254, 0
  %256 = fcmp une double %255, %255
  %257 = extractvalue { double, double } %254, 1
  %258 = fcmp une double %257, %257
  %259 = or i1 %256, %258
  %260 = zext i1 %259 to i64
  %261 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %231, i64 %260)
  %262 = call double @__nv_fabs(double %255)
  %263 = fcmp one double %262, 0x7FF0000000000000
  %264 = call double @__nv_fabs(double %257)
  %265 = fcmp one double %264, 0x7FF0000000000000
  %266 = and i1 %263, %265
  %267 = zext i1 %266 to i8
  %268 = icmp eq i8 %267, 0
  %269 = zext i1 %268 to i64
  %270 = call i64 @region_0_1_reduce_sum_2_0(i64 %240, i64 %269)
  %271 = call double @llvm.maximum.f64(double %262, double %264)
  %272 = call double @llvm.minimum.f64(double %262, double %264)
  %273 = fdiv double %272, %271
  %274 = fmul double %273, %273
  %275 = fadd double %274, 1.000000e+00
  %276 = call double @__nv_sqrt(double %275)
  %277 = fmul double %271, %276
  %278 = fcmp uno double %277, %277
  %279 = select i1 %278, double %272, double %277
  %280 = select i1 %266, double %279, double 0.000000e+00
  %281 = call double @region_2_3_reduce_max_2_0(double %251, double %280)
  %282 = bitcast i64 %261 to <2 x i32>
  %283 = extractelement <2 x i32> %282, i32 0
  %284 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %283, i32 16, i32 31)
  %285 = insertelement <2 x i32> undef, i32 %284, i32 0
  %286 = extractelement <2 x i32> %282, i32 1
  %287 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %286, i32 16, i32 31)
  %288 = insertelement <2 x i32> %285, i32 %287, i32 1
  %289 = bitcast <2 x i32> %288 to i64
  %290 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %261, i64 %289)
  %291 = bitcast i64 %290 to <2 x i32>
  %292 = extractelement <2 x i32> %291, i32 0
  %293 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %292, i32 8, i32 31)
  %294 = insertelement <2 x i32> undef, i32 %293, i32 0
  %295 = extractelement <2 x i32> %291, i32 1
  %296 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %295, i32 8, i32 31)
  %297 = insertelement <2 x i32> %294, i32 %296, i32 1
  %298 = bitcast <2 x i32> %297 to i64
  %299 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %290, i64 %298)
  %300 = bitcast i64 %299 to <2 x i32>
  %301 = extractelement <2 x i32> %300, i32 0
  %302 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %301, i32 4, i32 31)
  %303 = insertelement <2 x i32> undef, i32 %302, i32 0
  %304 = extractelement <2 x i32> %300, i32 1
  %305 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %304, i32 4, i32 31)
  %306 = insertelement <2 x i32> %303, i32 %305, i32 1
  %307 = bitcast <2 x i32> %306 to i64
  %308 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %299, i64 %307)
  %309 = bitcast i64 %308 to <2 x i32>
  %310 = extractelement <2 x i32> %309, i32 0
  %311 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %310, i32 2, i32 31)
  %312 = insertelement <2 x i32> undef, i32 %311, i32 0
  %313 = extractelement <2 x i32> %309, i32 1
  %314 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %313, i32 2, i32 31)
  %315 = insertelement <2 x i32> %312, i32 %314, i32 1
  %316 = bitcast <2 x i32> %315 to i64
  %317 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %308, i64 %316)
  %318 = bitcast i64 %317 to <2 x i32>
  %319 = extractelement <2 x i32> %318, i32 0
  %320 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %319, i32 1, i32 31)
  %321 = insertelement <2 x i32> undef, i32 %320, i32 0
  %322 = extractelement <2 x i32> %318, i32 1
  %323 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %322, i32 1, i32 31)
  %324 = insertelement <2 x i32> %321, i32 %323, i32 1
  %325 = bitcast <2 x i32> %324 to i64
  %326 = call i64 @region_0_1_clone_2_reduce_sum_30_0(i64 %317, i64 %325)
  %327 = bitcast i64 %270 to <2 x i32>
  %328 = extractelement <2 x i32> %327, i32 0
  %329 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %328, i32 16, i32 31)
  %330 = insertelement <2 x i32> undef, i32 %329, i32 0
  %331 = extractelement <2 x i32> %327, i32 1
  %332 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %331, i32 16, i32 31)
  %333 = insertelement <2 x i32> %330, i32 %332, i32 1
  %334 = bitcast <2 x i32> %333 to i64
  %335 = call i64 @region_0_1_reduce_sum_2_0(i64 %270, i64 %334)
  %336 = bitcast i64 %335 to <2 x i32>
  %337 = extractelement <2 x i32> %336, i32 0
  %338 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %337, i32 8, i32 31)
  %339 = insertelement <2 x i32> undef, i32 %338, i32 0
  %340 = extractelement <2 x i32> %336, i32 1
  %341 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %340, i32 8, i32 31)
  %342 = insertelement <2 x i32> %339, i32 %341, i32 1
  %343 = bitcast <2 x i32> %342 to i64
  %344 = call i64 @region_0_1_reduce_sum_2_0(i64 %335, i64 %343)
  %345 = bitcast i64 %344 to <2 x i32>
  %346 = extractelement <2 x i32> %345, i32 0
  %347 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %346, i32 4, i32 31)
  %348 = insertelement <2 x i32> undef, i32 %347, i32 0
  %349 = extractelement <2 x i32> %345, i32 1
  %350 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %349, i32 4, i32 31)
  %351 = insertelement <2 x i32> %348, i32 %350, i32 1
  %352 = bitcast <2 x i32> %351 to i64
  %353 = call i64 @region_0_1_reduce_sum_2_0(i64 %344, i64 %352)
  %354 = bitcast i64 %353 to <2 x i32>
  %355 = extractelement <2 x i32> %354, i32 0
  %356 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %355, i32 2, i32 31)
  %357 = insertelement <2 x i32> undef, i32 %356, i32 0
  %358 = extractelement <2 x i32> %354, i32 1
  %359 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %358, i32 2, i32 31)
  %360 = insertelement <2 x i32> %357, i32 %359, i32 1
  %361 = bitcast <2 x i32> %360 to i64
  %362 = call i64 @region_0_1_reduce_sum_2_0(i64 %353, i64 %361)
  %363 = bitcast i64 %362 to <2 x i32>
  %364 = extractelement <2 x i32> %363, i32 0
  %365 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %364, i32 1, i32 31)
  %366 = insertelement <2 x i32> undef, i32 %365, i32 0
  %367 = extractelement <2 x i32> %363, i32 1
  %368 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %367, i32 1, i32 31)
  %369 = insertelement <2 x i32> %366, i32 %368, i32 1
  %370 = bitcast <2 x i32> %369 to i64
  %371 = call i64 @region_0_1_reduce_sum_2_0(i64 %362, i64 %370)
  %372 = bitcast double %281 to i64
  %373 = bitcast i64 %372 to <2 x i32>
  %374 = extractelement <2 x i32> %373, i32 0
  %375 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %374, i32 16, i32 31)
  %376 = insertelement <2 x i32> undef, i32 %375, i32 0
  %377 = extractelement <2 x i32> %373, i32 1
  %378 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %377, i32 16, i32 31)
  %379 = insertelement <2 x i32> %376, i32 %378, i32 1
  %380 = bitcast <2 x i32> %379 to double
  %381 = call double @region_2_3_reduce_max_2_0(double %281, double %380)
  %382 = bitcast double %381 to i64
  %383 = bitcast i64 %382 to <2 x i32>
  %384 = extractelement <2 x i32> %383, i32 0
  %385 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %384, i32 8, i32 31)
  %386 = insertelement <2 x i32> undef, i32 %385, i32 0
  %387 = extractelement <2 x i32> %383, i32 1
  %388 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %387, i32 8, i32 31)
  %389 = insertelement <2 x i32> %386, i32 %388, i32 1
  %390 = bitcast <2 x i32> %389 to double
  %391 = call double @region_2_3_reduce_max_2_0(double %381, double %390)
  %392 = bitcast double %391 to i64
  %393 = bitcast i64 %392 to <2 x i32>
  %394 = extractelement <2 x i32> %393, i32 0
  %395 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %394, i32 4, i32 31)
  %396 = insertelement <2 x i32> undef, i32 %395, i32 0
  %397 = extractelement <2 x i32> %393, i32 1
  %398 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %397, i32 4, i32 31)
  %399 = insertelement <2 x i32> %396, i32 %398, i32 1
  %400 = bitcast <2 x i32> %399 to double
  %401 = call double @region_2_3_reduce_max_2_0(double %391, double %400)
  %402 = bitcast double %401 to i64
  %403 = bitcast i64 %402 to <2 x i32>
  %404 = extractelement <2 x i32> %403, i32 0
  %405 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %404, i32 2, i32 31)
  %406 = insertelement <2 x i32> undef, i32 %405, i32 0
  %407 = extractelement <2 x i32> %403, i32 1
  %408 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %407, i32 2, i32 31)
  %409 = insertelement <2 x i32> %406, i32 %408, i32 1
  %410 = bitcast <2 x i32> %409 to double
  %411 = call double @region_2_3_reduce_max_2_0(double %401, double %410)
  %412 = bitcast double %411 to i64
  %413 = bitcast i64 %412 to <2 x i32>
  %414 = extractelement <2 x i32> %413, i32 0
  %415 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %414, i32 1, i32 31)
  %416 = insertelement <2 x i32> undef, i32 %415, i32 0
  %417 = extractelement <2 x i32> %413, i32 1
  %418 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %417, i32 1, i32 31)
  %419 = insertelement <2 x i32> %416, i32 %418, i32 1
  %420 = bitcast <2 x i32> %419 to double
  %421 = call double @region_2_3_reduce_max_2_0(double %411, double %420)
  %422 = icmp eq i32 %11, 0
  %423 = icmp sle i32 %5, 224
  %424 = and i1 %422, %423
  %425 = mul i32 %6, 8
  %426 = add i32 %425, %7
  br i1 %424, label %427, label %431

427:                                              ; preds = %4
  %428 = getelementptr inbounds [256 x i64], ptr %1, i32 0, i32 %426
  store i64 %326, ptr %428, align 4
  %429 = getelementptr inbounds [256 x i64], ptr %2, i32 0, i32 %426
  store i64 %371, ptr %429, align 4
  %430 = getelementptr inbounds [256 x double], ptr %3, i32 0, i32 %426
  store double %421, ptr %430, align 8
  br label %431

431:                                              ; preds = %427, %4
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

define ptx_kernel void @input_reduce_fusion_3(ptr noalias align 256 dereferenceable(2048) %0, ptr noalias align 256 dereferenceable(8) %1) #4 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %4 = getelementptr inbounds [256 x double], ptr %0, i32 0, i32 %3
  %5 = load double, ptr %4, align 8, !invariant.load !3
  %6 = call double @region_2_3_reduce_max_2_01(double 0xFFF0000000000000, double %5)
  %7 = add i32 %3, 32
  %8 = getelementptr inbounds [256 x double], ptr %0, i32 0, i32 %7
  %9 = load double, ptr %8, align 8, !invariant.load !3
  %10 = call double @region_2_3_reduce_max_2_01(double %6, double %9)
  %11 = add i32 %3, 64
  %12 = getelementptr inbounds [256 x double], ptr %0, i32 0, i32 %11
  %13 = load double, ptr %12, align 8, !invariant.load !3
  %14 = call double @region_2_3_reduce_max_2_01(double %10, double %13)
  %15 = add i32 %3, 96
  %16 = getelementptr inbounds [256 x double], ptr %0, i32 0, i32 %15
  %17 = load double, ptr %16, align 8, !invariant.load !3
  %18 = call double @region_2_3_reduce_max_2_01(double %14, double %17)
  %19 = add i32 %3, 128
  %20 = getelementptr inbounds [256 x double], ptr %0, i32 0, i32 %19
  %21 = load double, ptr %20, align 8, !invariant.load !3
  %22 = call double @region_2_3_reduce_max_2_01(double %18, double %21)
  %23 = add i32 %3, 160
  %24 = getelementptr inbounds [256 x double], ptr %0, i32 0, i32 %23
  %25 = load double, ptr %24, align 8, !invariant.load !3
  %26 = call double @region_2_3_reduce_max_2_01(double %22, double %25)
  %27 = add i32 %3, 192
  %28 = getelementptr inbounds [256 x double], ptr %0, i32 0, i32 %27
  %29 = load double, ptr %28, align 8, !invariant.load !3
  %30 = call double @region_2_3_reduce_max_2_01(double %26, double %29)
  %31 = add i32 %3, 224
  %32 = getelementptr inbounds [256 x double], ptr %0, i32 0, i32 %31
  %33 = load double, ptr %32, align 8, !invariant.load !3
  %34 = call double @region_2_3_reduce_max_2_01(double %30, double %33)
  %35 = bitcast double %34 to i64
  %36 = bitcast i64 %35 to <2 x i32>
  %37 = extractelement <2 x i32> %36, i32 0
  %38 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %37, i32 16, i32 31)
  %39 = insertelement <2 x i32> undef, i32 %38, i32 0
  %40 = extractelement <2 x i32> %36, i32 1
  %41 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %40, i32 16, i32 31)
  %42 = insertelement <2 x i32> %39, i32 %41, i32 1
  %43 = bitcast <2 x i32> %42 to double
  %44 = call double @region_2_3_reduce_max_2_01(double %34, double %43)
  %45 = bitcast double %44 to i64
  %46 = bitcast i64 %45 to <2 x i32>
  %47 = extractelement <2 x i32> %46, i32 0
  %48 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %47, i32 8, i32 31)
  %49 = insertelement <2 x i32> undef, i32 %48, i32 0
  %50 = extractelement <2 x i32> %46, i32 1
  %51 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %50, i32 8, i32 31)
  %52 = insertelement <2 x i32> %49, i32 %51, i32 1
  %53 = bitcast <2 x i32> %52 to double
  %54 = call double @region_2_3_reduce_max_2_01(double %44, double %53)
  %55 = bitcast double %54 to i64
  %56 = bitcast i64 %55 to <2 x i32>
  %57 = extractelement <2 x i32> %56, i32 0
  %58 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %57, i32 4, i32 31)
  %59 = insertelement <2 x i32> undef, i32 %58, i32 0
  %60 = extractelement <2 x i32> %56, i32 1
  %61 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %60, i32 4, i32 31)
  %62 = insertelement <2 x i32> %59, i32 %61, i32 1
  %63 = bitcast <2 x i32> %62 to double
  %64 = call double @region_2_3_reduce_max_2_01(double %54, double %63)
  %65 = bitcast double %64 to i64
  %66 = bitcast i64 %65 to <2 x i32>
  %67 = extractelement <2 x i32> %66, i32 0
  %68 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %67, i32 2, i32 31)
  %69 = insertelement <2 x i32> undef, i32 %68, i32 0
  %70 = extractelement <2 x i32> %66, i32 1
  %71 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %70, i32 2, i32 31)
  %72 = insertelement <2 x i32> %69, i32 %71, i32 1
  %73 = bitcast <2 x i32> %72 to double
  %74 = call double @region_2_3_reduce_max_2_01(double %64, double %73)
  %75 = bitcast double %74 to i64
  %76 = bitcast i64 %75 to <2 x i32>
  %77 = extractelement <2 x i32> %76, i32 0
  %78 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %77, i32 1, i32 31)
  %79 = insertelement <2 x i32> undef, i32 %78, i32 0
  %80 = extractelement <2 x i32> %76, i32 1
  %81 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %80, i32 1, i32 31)
  %82 = insertelement <2 x i32> %79, i32 %81, i32 1
  %83 = bitcast <2 x i32> %82 to double
  %84 = call double @region_2_3_reduce_max_2_01(double %74, double %83)
  %85 = icmp eq i32 %3, 0
  br i1 %85, label %86, label %88

86:                                               ; preds = %2
  %87 = getelementptr inbounds [1 x double], ptr %1, i32 0, i32 0
  store double %84, ptr %87, align 8
  br label %88

88:                                               ; preds = %86, %2
  ret void
}

define internal double @region_2_3_reduce_max_2_01(double %0, double %1) {
  %3 = call double @llvm.maximum.f64(double %0, double %1)
  ret double %3
}

define ptx_kernel void @input_reduce_fusion_1(ptr noalias align 256 dereferenceable(2048) %0, ptr noalias align 256 dereferenceable(8) %1) #4 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %4 = getelementptr inbounds [256 x i64], ptr %0, i32 0, i32 %3
  %5 = load i64, ptr %4, align 4, !invariant.load !3
  %6 = call i64 @region_0_1_reduce_sum_2_02(i64 0, i64 %5)
  %7 = add i32 %3, 32
  %8 = getelementptr inbounds [256 x i64], ptr %0, i32 0, i32 %7
  %9 = load i64, ptr %8, align 4, !invariant.load !3
  %10 = call i64 @region_0_1_reduce_sum_2_02(i64 %6, i64 %9)
  %11 = add i32 %3, 64
  %12 = getelementptr inbounds [256 x i64], ptr %0, i32 0, i32 %11
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = call i64 @region_0_1_reduce_sum_2_02(i64 %10, i64 %13)
  %15 = add i32 %3, 96
  %16 = getelementptr inbounds [256 x i64], ptr %0, i32 0, i32 %15
  %17 = load i64, ptr %16, align 4, !invariant.load !3
  %18 = call i64 @region_0_1_reduce_sum_2_02(i64 %14, i64 %17)
  %19 = add i32 %3, 128
  %20 = getelementptr inbounds [256 x i64], ptr %0, i32 0, i32 %19
  %21 = load i64, ptr %20, align 4, !invariant.load !3
  %22 = call i64 @region_0_1_reduce_sum_2_02(i64 %18, i64 %21)
  %23 = add i32 %3, 160
  %24 = getelementptr inbounds [256 x i64], ptr %0, i32 0, i32 %23
  %25 = load i64, ptr %24, align 4, !invariant.load !3
  %26 = call i64 @region_0_1_reduce_sum_2_02(i64 %22, i64 %25)
  %27 = add i32 %3, 192
  %28 = getelementptr inbounds [256 x i64], ptr %0, i32 0, i32 %27
  %29 = load i64, ptr %28, align 4, !invariant.load !3
  %30 = call i64 @region_0_1_reduce_sum_2_02(i64 %26, i64 %29)
  %31 = add i32 %3, 224
  %32 = getelementptr inbounds [256 x i64], ptr %0, i32 0, i32 %31
  %33 = load i64, ptr %32, align 4, !invariant.load !3
  %34 = call i64 @region_0_1_reduce_sum_2_02(i64 %30, i64 %33)
  %35 = bitcast i64 %34 to <2 x i32>
  %36 = extractelement <2 x i32> %35, i32 0
  %37 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %36, i32 16, i32 31)
  %38 = insertelement <2 x i32> undef, i32 %37, i32 0
  %39 = extractelement <2 x i32> %35, i32 1
  %40 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %39, i32 16, i32 31)
  %41 = insertelement <2 x i32> %38, i32 %40, i32 1
  %42 = bitcast <2 x i32> %41 to i64
  %43 = call i64 @region_0_1_reduce_sum_2_02(i64 %34, i64 %42)
  %44 = bitcast i64 %43 to <2 x i32>
  %45 = extractelement <2 x i32> %44, i32 0
  %46 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %45, i32 8, i32 31)
  %47 = insertelement <2 x i32> undef, i32 %46, i32 0
  %48 = extractelement <2 x i32> %44, i32 1
  %49 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %48, i32 8, i32 31)
  %50 = insertelement <2 x i32> %47, i32 %49, i32 1
  %51 = bitcast <2 x i32> %50 to i64
  %52 = call i64 @region_0_1_reduce_sum_2_02(i64 %43, i64 %51)
  %53 = bitcast i64 %52 to <2 x i32>
  %54 = extractelement <2 x i32> %53, i32 0
  %55 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %54, i32 4, i32 31)
  %56 = insertelement <2 x i32> undef, i32 %55, i32 0
  %57 = extractelement <2 x i32> %53, i32 1
  %58 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %57, i32 4, i32 31)
  %59 = insertelement <2 x i32> %56, i32 %58, i32 1
  %60 = bitcast <2 x i32> %59 to i64
  %61 = call i64 @region_0_1_reduce_sum_2_02(i64 %52, i64 %60)
  %62 = bitcast i64 %61 to <2 x i32>
  %63 = extractelement <2 x i32> %62, i32 0
  %64 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %63, i32 2, i32 31)
  %65 = insertelement <2 x i32> undef, i32 %64, i32 0
  %66 = extractelement <2 x i32> %62, i32 1
  %67 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %66, i32 2, i32 31)
  %68 = insertelement <2 x i32> %65, i32 %67, i32 1
  %69 = bitcast <2 x i32> %68 to i64
  %70 = call i64 @region_0_1_reduce_sum_2_02(i64 %61, i64 %69)
  %71 = bitcast i64 %70 to <2 x i32>
  %72 = extractelement <2 x i32> %71, i32 0
  %73 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %72, i32 1, i32 31)
  %74 = insertelement <2 x i32> undef, i32 %73, i32 0
  %75 = extractelement <2 x i32> %71, i32 1
  %76 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %75, i32 1, i32 31)
  %77 = insertelement <2 x i32> %74, i32 %76, i32 1
  %78 = bitcast <2 x i32> %77 to i64
  %79 = call i64 @region_0_1_reduce_sum_2_02(i64 %70, i64 %78)
  %80 = icmp eq i32 %3, 0
  br i1 %80, label %81, label %83

81:                                               ; preds = %2
  %82 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  store i64 %79, ptr %82, align 4
  br label %83

83:                                               ; preds = %81, %2
  ret void
}

define internal i64 @region_0_1_reduce_sum_2_02(i64 %0, i64 %1) {
  %3 = add i64 %0, %1
  ret i64 %3
}

define ptx_kernel void @wrapped_concatenate(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(16) %2) #5 {
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

define ptx_kernel void @wrapped_slice_1(ptr noalias align 256 dereferenceable(16) %0, ptr noalias align 256 dereferenceable(8) %1) #5 {
  %3 = getelementptr inbounds [2 x i64], ptr %0, i32 0, i32 1
  %4 = load i64, ptr %3, align 4, !invariant.load !3
  %5 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  store i64 %4, ptr %5, align 4
  ret void
}

define ptx_kernel void @wrapped_slice(ptr noalias align 256 dereferenceable(16) %0, ptr noalias align 256 dereferenceable(8) %1) #5 {
  %3 = getelementptr inbounds [2 x i64], ptr %0, i32 0, i32 0
  %4 = load i64, ptr %3, align 4, !invariant.load !3
  %5 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  store i64 %4, ptr %5, align 4
  ret void
}

define ptx_kernel void @input_concatenate_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(24) %3) #5 {
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

attributes #0 = { "nvvm.reqntid"="256,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #4 = { "nvvm.reqntid"="32,1,1" }
attributes #5 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 256}
!2 = !{i32 0, i32 32}
!3 = !{}
