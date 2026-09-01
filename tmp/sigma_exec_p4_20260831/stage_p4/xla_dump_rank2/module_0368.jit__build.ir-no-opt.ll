; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_broadcast_fusion(ptr noalias align 256 dereferenceable(787251200) %0) #0 {
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %4 = mul i32 %3, 4
  %5 = mul i32 %2, 512
  %6 = add i32 %4, %5
  %7 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %6
  store { double, double } zeroinitializer, ptr %7, align 8
  %8 = add i32 %6, 1
  %9 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %8
  store { double, double } zeroinitializer, ptr %9, align 8
  %10 = add i32 %6, 2
  %11 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %10
  store { double, double } zeroinitializer, ptr %11, align 8
  %12 = add i32 %6, 3
  %13 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i32 %12
  store { double, double } zeroinitializer, ptr %13, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

define ptx_kernel void @loop_compare_fusion(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(1) %1) #2 {
  %3 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %4 = load i64, ptr %3, align 4, !invariant.load !3
  %5 = icmp slt i64 %4, 4
  %6 = zext i1 %5 to i8
  %7 = getelementptr inbounds [1 x i8], ptr %1, i32 0, i32 0
  store i8 %6, ptr %7, align 1
  ret void
}

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 16 dereferenceable(16) %0, ptr noalias align 256 dereferenceable(16) %1) #2 {
  %3 = getelementptr inbounds [1 x { double, double }], ptr %0, i32 0, i32 0
  %4 = load { double, double }, ptr %3, align 8, !invariant.load !3
  %5 = extractvalue { double, double } %4, 0
  %6 = extractvalue { double, double } %4, 1
  %7 = fmul double %5, -0.000000e+00
  %8 = fmul double %6, -1.000000e+00
  %9 = fsub double %7, %8
  %10 = fmul double %6, -0.000000e+00
  %11 = fmul double %5, -1.000000e+00
  %12 = fadd double %10, %11
  %13 = insertvalue { double, double } poison, double %9, 0
  %14 = insertvalue { double, double } %13, double %12, 1
  %15 = getelementptr inbounds [1 x { double, double }], ptr %1, i32 0, i32 0
  store { double, double } %14, ptr %15, align 8
  ret void
}

define ptx_kernel void @loop_slice_fusion(ptr noalias align 16 dereferenceable(192) %0, ptr noalias align 256 dereferenceable(8) %1, ptr noalias align 256 dereferenceable(8) %2, ptr noalias align 256 dereferenceable(8) %3, ptr noalias align 256 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(8) %5, ptr noalias align 256 dereferenceable(8) %6, ptr noalias align 256 dereferenceable(8) %7) #2 {
  %9 = call double @fused_slice_dynamic_slice_31_6(ptr %0, ptr %1, i64 0, i64 4)
  %10 = call double @fused_slice_dynamic_slice_31_6(ptr %0, ptr %1, i64 0, i64 0)
  %11 = call double @fused_slice_dynamic_slice_31_6(ptr %0, ptr %1, i64 0, i64 5)
  %12 = call double @fused_slice_dynamic_slice_31_6(ptr %0, ptr %1, i64 0, i64 1)
  %13 = call double @fused_slice_dynamic_slice_31_6(ptr %0, ptr %1, i64 0, i64 2)
  %14 = call double @fused_slice_dynamic_slice_31_6(ptr %0, ptr %1, i64 0, i64 3)
  %15 = getelementptr inbounds [1 x double], ptr %2, i32 0, i32 0
  store double %9, ptr %15, align 8
  %16 = getelementptr inbounds [1 x double], ptr %3, i32 0, i32 0
  store double %10, ptr %16, align 8
  %17 = getelementptr inbounds [1 x double], ptr %4, i32 0, i32 0
  store double %11, ptr %17, align 8
  %18 = getelementptr inbounds [1 x double], ptr %5, i32 0, i32 0
  store double %12, ptr %18, align 8
  %19 = getelementptr inbounds [1 x double], ptr %6, i32 0, i32 0
  store double %13, ptr %19, align 8
  %20 = getelementptr inbounds [1 x double], ptr %7, i32 0, i32 0
  store double %14, ptr %20, align 8
  ret void
}

define internal double @fused_slice_dynamic_slice_31_6(ptr noalias %0, ptr noalias %1, i64 %2, i64 %3) {
  %5 = getelementptr inbounds [1 x i64], ptr %1, i32 0, i32 0
  %6 = load i64, ptr %5, align 4, !invariant.load !3
  %7 = icmp slt i64 %6, 0
  %8 = add i64 %6, 4
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = call i64 @llvm.smin.i64(i64 %9, i64 3)
  %11 = call i64 @llvm.smax.i64(i64 %10, i64 0)
  %12 = add i64 %2, %11
  %13 = mul i64 %12, 6
  %14 = add i64 %13, %3
  %15 = getelementptr inbounds [24 x double], ptr %0, i32 0, i64 %14
  %16 = load double, ptr %15, align 8, !invariant.load !3
  ret double %16
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #3

define ptx_kernel void @loop_add_fusion(ptr noalias align 256 dereferenceable(787251200) %0, ptr noalias align 16 dereferenceable(3149004800) %1, ptr noalias align 256 dereferenceable(16) %2, ptr noalias align 16 dereferenceable(3149004800) %3, ptr noalias align 256 dereferenceable(8) %4, ptr noalias align 256 dereferenceable(8) %5, ptr noalias align 256 dereferenceable(8) %6, ptr noalias align 256 dereferenceable(8) %7, ptr noalias align 256 dereferenceable(8) %8, ptr noalias align 256 dereferenceable(8) %9, ptr noalias align 256 dereferenceable(1) %10, ptr noalias align 16 dereferenceable(8) %11, ptr noalias align 16 dereferenceable(16) %12, ptr noalias align 256 dereferenceable(8) %13, ptr noalias align 256 dereferenceable(787251200) %14) #0 {
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %17 = sext i32 %16 to i64
  %18 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %19 = sext i32 %18 to i64
  %20 = getelementptr inbounds [1 x i8], ptr %10, i32 0, i32 0
  %21 = load i8, ptr %20, align 1, !invariant.load !3
  %22 = getelementptr inbounds [1 x i64], ptr %13, i32 0, i32 0
  %23 = load i64, ptr %22, align 4, !invariant.load !3
  %24 = icmp slt i64 %23, 0
  %25 = add i64 %23, 4
  %26 = select i1 %24, i64 %25, i64 %23
  %27 = call i64 @llvm.smin.i64(i64 %26, i64 3)
  %28 = call i64 @llvm.smax.i64(i64 %27, i64 0)
  %29 = getelementptr inbounds [4 x i32], ptr %12, i32 0, i64 %28
  %30 = load i32, ptr %29, align 4, !invariant.load !3
  %31 = icmp slt i32 %30, 0
  %32 = add i32 %30, 4
  %33 = select i1 %31, i32 %32, i32 %30
  %34 = sext i32 %33 to i64
  %35 = call i64 @llvm.smin.i64(i64 %34, i64 3)
  %36 = call i64 @llvm.smax.i64(i64 %35, i64 0)
  %37 = trunc i8 %21 to i1
  %38 = getelementptr inbounds [1 x double], ptr %11, i32 0, i32 0
  %39 = load double, ptr %38, align 8, !invariant.load !3
  %40 = getelementptr inbounds [1 x double], ptr %9, i32 0, i32 0
  %41 = load double, ptr %40, align 8, !invariant.load !3
  %42 = getelementptr inbounds [1 x double], ptr %8, i32 0, i32 0
  %43 = load double, ptr %42, align 8, !invariant.load !3
  %44 = getelementptr inbounds [1 x { double, double }], ptr %2, i32 0, i32 0
  %45 = load { double, double }, ptr %44, align 8, !invariant.load !3
  %46 = getelementptr inbounds [1 x double], ptr %7, i32 0, i32 0
  %47 = load double, ptr %46, align 8, !invariant.load !3
  %48 = getelementptr inbounds [1 x double], ptr %6, i32 0, i32 0
  %49 = load double, ptr %48, align 8, !invariant.load !3
  %50 = getelementptr inbounds [1 x double], ptr %5, i32 0, i32 0
  %51 = load double, ptr %50, align 8, !invariant.load !3
  %52 = getelementptr inbounds [1 x double], ptr %4, i32 0, i32 0
  %53 = load double, ptr %52, align 8, !invariant.load !3
  %54 = mul i64 %36, 49203200
  %55 = mul i64 %19, 4
  %56 = add i64 %54, %55
  %57 = mul i64 %17, 512
  %58 = add i64 %56, %57
  %59 = getelementptr inbounds [196812800 x { double, double }], ptr %3, i32 0, i64 %58
  %60 = load { double, double }, ptr %59, align 8, !invariant.load !3
  %61 = extractvalue { double, double } %60, 0
  %62 = insertvalue { double, double } poison, double %61, 0
  %63 = insertvalue { double, double } %62, double 0.000000e+00, 1
  %64 = select i1 %37, { double, double } %63, { double, double } %60
  %65 = extractvalue { double, double } %64, 0
  %66 = fsub double %65, %39
  %67 = extractvalue { double, double } %64, 1
  %68 = fcmp ogt double %61, %41
  %69 = fcmp ole double %61, %43
  %70 = extractvalue { double, double } %45, 0
  %71 = extractvalue { double, double } %45, 1
  %72 = fmul double %66, %70
  %73 = fmul double %67, %71
  %74 = fsub double %72, %73
  %75 = fmul double %67, %70
  %76 = fmul double %66, %71
  %77 = fadd double %75, %76
  %78 = fmul double %74, 5.000000e-01
  %79 = call double @__nv_exp(double %78)
  %80 = call double @__nv_sin(double %77)
  %81 = fmul double %79, %80
  %82 = and i1 %68, %69
  %83 = extractvalue { double, double } %60, 1
  %84 = fneg double %83
  %85 = fcmp oge double %84, %47
  %86 = call double @__nv_cos(double %77)
  %87 = fmul double %79, %86
  %88 = call double @__nv_exp(double %74)
  %89 = fmul double %81, %79
  %90 = fmul double %88, %80
  %91 = and i1 %82, %85
  %92 = fcmp ogt double %84, %49
  %93 = fcmp oeq double %88, 0x7FF0000000000000
  %94 = fmul double %87, %79
  %95 = fmul double %88, %86
  %96 = fcmp oeq double %77, 0.000000e+00
  %97 = select i1 %93, double %89, double %90
  %98 = and i1 %91, %92
  %99 = fcmp olt double %84, %51
  %100 = select i1 %93, double %94, double %95
  %101 = select i1 %96, double 0.000000e+00, double %97
  %102 = and i1 %98, %99
  %103 = fcmp ole double %84, %53
  %104 = getelementptr inbounds [196812800 x { double, double }], ptr %1, i32 0, i64 %58
  %105 = load { double, double }, ptr %104, align 8, !invariant.load !3
  %106 = and i1 %102, %103
  %107 = extractvalue { double, double } %105, 0
  %108 = extractvalue { double, double } %105, 1
  %109 = fmul double %107, %100
  %110 = fmul double %108, %101
  %111 = fsub double %109, %110
  %112 = fmul double %108, %100
  %113 = fmul double %107, %101
  %114 = fadd double %112, %113
  %115 = insertvalue { double, double } poison, double %111, 0
  %116 = insertvalue { double, double } %115, double %114, 1
  %117 = add i64 %55, %57
  %118 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i64 %117
  %119 = load { double, double }, ptr %118, align 8
  %120 = select i1 %106, { double, double } %116, { double, double } zeroinitializer
  %121 = extractvalue { double, double } %119, 0
  %122 = extractvalue { double, double } %120, 0
  %123 = fadd double %121, %122
  %124 = extractvalue { double, double } %119, 1
  %125 = extractvalue { double, double } %120, 1
  %126 = fadd double %124, %125
  %127 = insertvalue { double, double } poison, double %123, 0
  %128 = insertvalue { double, double } %127, double %126, 1
  store { double, double } %128, ptr %118, align 8
  %129 = add i64 %58, 1
  %130 = getelementptr inbounds [196812800 x { double, double }], ptr %3, i32 0, i64 %129
  %131 = load { double, double }, ptr %130, align 8, !invariant.load !3
  %132 = extractvalue { double, double } %131, 0
  %133 = insertvalue { double, double } poison, double %132, 0
  %134 = insertvalue { double, double } %133, double 0.000000e+00, 1
  %135 = select i1 %37, { double, double } %134, { double, double } %131
  %136 = extractvalue { double, double } %135, 0
  %137 = fsub double %136, %39
  %138 = extractvalue { double, double } %135, 1
  %139 = fcmp ogt double %132, %41
  %140 = fcmp ole double %132, %43
  %141 = fmul double %137, %70
  %142 = fmul double %138, %71
  %143 = fsub double %141, %142
  %144 = fmul double %138, %70
  %145 = fmul double %137, %71
  %146 = fadd double %144, %145
  %147 = fmul double %143, 5.000000e-01
  %148 = call double @__nv_exp(double %147)
  %149 = call double @__nv_sin(double %146)
  %150 = fmul double %148, %149
  %151 = and i1 %139, %140
  %152 = extractvalue { double, double } %131, 1
  %153 = fneg double %152
  %154 = fcmp oge double %153, %47
  %155 = call double @__nv_cos(double %146)
  %156 = fmul double %148, %155
  %157 = call double @__nv_exp(double %143)
  %158 = fmul double %150, %148
  %159 = fmul double %157, %149
  %160 = and i1 %151, %154
  %161 = fcmp ogt double %153, %49
  %162 = fcmp oeq double %157, 0x7FF0000000000000
  %163 = fmul double %156, %148
  %164 = fmul double %157, %155
  %165 = fcmp oeq double %146, 0.000000e+00
  %166 = select i1 %162, double %158, double %159
  %167 = and i1 %160, %161
  %168 = fcmp olt double %153, %51
  %169 = select i1 %162, double %163, double %164
  %170 = select i1 %165, double 0.000000e+00, double %166
  %171 = and i1 %167, %168
  %172 = fcmp ole double %153, %53
  %173 = getelementptr inbounds [196812800 x { double, double }], ptr %1, i32 0, i64 %129
  %174 = load { double, double }, ptr %173, align 8, !invariant.load !3
  %175 = and i1 %171, %172
  %176 = extractvalue { double, double } %174, 0
  %177 = extractvalue { double, double } %174, 1
  %178 = fmul double %176, %169
  %179 = fmul double %177, %170
  %180 = fsub double %178, %179
  %181 = fmul double %177, %169
  %182 = fmul double %176, %170
  %183 = fadd double %181, %182
  %184 = insertvalue { double, double } poison, double %180, 0
  %185 = insertvalue { double, double } %184, double %183, 1
  %186 = add i64 %117, 1
  %187 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i64 %186
  %188 = load { double, double }, ptr %187, align 8
  %189 = select i1 %175, { double, double } %185, { double, double } zeroinitializer
  %190 = extractvalue { double, double } %188, 0
  %191 = extractvalue { double, double } %189, 0
  %192 = fadd double %190, %191
  %193 = extractvalue { double, double } %188, 1
  %194 = extractvalue { double, double } %189, 1
  %195 = fadd double %193, %194
  %196 = insertvalue { double, double } poison, double %192, 0
  %197 = insertvalue { double, double } %196, double %195, 1
  store { double, double } %197, ptr %187, align 8
  %198 = add i64 %58, 2
  %199 = getelementptr inbounds [196812800 x { double, double }], ptr %3, i32 0, i64 %198
  %200 = load { double, double }, ptr %199, align 8, !invariant.load !3
  %201 = extractvalue { double, double } %200, 0
  %202 = insertvalue { double, double } poison, double %201, 0
  %203 = insertvalue { double, double } %202, double 0.000000e+00, 1
  %204 = select i1 %37, { double, double } %203, { double, double } %200
  %205 = extractvalue { double, double } %204, 0
  %206 = fsub double %205, %39
  %207 = extractvalue { double, double } %204, 1
  %208 = fcmp ogt double %201, %41
  %209 = fcmp ole double %201, %43
  %210 = fmul double %206, %70
  %211 = fmul double %207, %71
  %212 = fsub double %210, %211
  %213 = fmul double %207, %70
  %214 = fmul double %206, %71
  %215 = fadd double %213, %214
  %216 = fmul double %212, 5.000000e-01
  %217 = call double @__nv_exp(double %216)
  %218 = call double @__nv_sin(double %215)
  %219 = fmul double %217, %218
  %220 = and i1 %208, %209
  %221 = extractvalue { double, double } %200, 1
  %222 = fneg double %221
  %223 = fcmp oge double %222, %47
  %224 = call double @__nv_cos(double %215)
  %225 = fmul double %217, %224
  %226 = call double @__nv_exp(double %212)
  %227 = fmul double %219, %217
  %228 = fmul double %226, %218
  %229 = and i1 %220, %223
  %230 = fcmp ogt double %222, %49
  %231 = fcmp oeq double %226, 0x7FF0000000000000
  %232 = fmul double %225, %217
  %233 = fmul double %226, %224
  %234 = fcmp oeq double %215, 0.000000e+00
  %235 = select i1 %231, double %227, double %228
  %236 = and i1 %229, %230
  %237 = fcmp olt double %222, %51
  %238 = select i1 %231, double %232, double %233
  %239 = select i1 %234, double 0.000000e+00, double %235
  %240 = and i1 %236, %237
  %241 = fcmp ole double %222, %53
  %242 = getelementptr inbounds [196812800 x { double, double }], ptr %1, i32 0, i64 %198
  %243 = load { double, double }, ptr %242, align 8, !invariant.load !3
  %244 = and i1 %240, %241
  %245 = extractvalue { double, double } %243, 0
  %246 = extractvalue { double, double } %243, 1
  %247 = fmul double %245, %238
  %248 = fmul double %246, %239
  %249 = fsub double %247, %248
  %250 = fmul double %246, %238
  %251 = fmul double %245, %239
  %252 = fadd double %250, %251
  %253 = insertvalue { double, double } poison, double %249, 0
  %254 = insertvalue { double, double } %253, double %252, 1
  %255 = add i64 %117, 2
  %256 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i64 %255
  %257 = load { double, double }, ptr %256, align 8
  %258 = select i1 %244, { double, double } %254, { double, double } zeroinitializer
  %259 = extractvalue { double, double } %257, 0
  %260 = extractvalue { double, double } %258, 0
  %261 = fadd double %259, %260
  %262 = extractvalue { double, double } %257, 1
  %263 = extractvalue { double, double } %258, 1
  %264 = fadd double %262, %263
  %265 = insertvalue { double, double } poison, double %261, 0
  %266 = insertvalue { double, double } %265, double %264, 1
  store { double, double } %266, ptr %256, align 8
  %267 = add i64 %58, 3
  %268 = getelementptr inbounds [196812800 x { double, double }], ptr %3, i32 0, i64 %267
  %269 = load { double, double }, ptr %268, align 8, !invariant.load !3
  %270 = extractvalue { double, double } %269, 0
  %271 = insertvalue { double, double } poison, double %270, 0
  %272 = insertvalue { double, double } %271, double 0.000000e+00, 1
  %273 = select i1 %37, { double, double } %272, { double, double } %269
  %274 = extractvalue { double, double } %273, 0
  %275 = fsub double %274, %39
  %276 = extractvalue { double, double } %273, 1
  %277 = fcmp ogt double %270, %41
  %278 = fcmp ole double %270, %43
  %279 = fmul double %275, %70
  %280 = fmul double %276, %71
  %281 = fsub double %279, %280
  %282 = fmul double %276, %70
  %283 = fmul double %275, %71
  %284 = fadd double %282, %283
  %285 = fmul double %281, 5.000000e-01
  %286 = call double @__nv_exp(double %285)
  %287 = call double @__nv_sin(double %284)
  %288 = fmul double %286, %287
  %289 = and i1 %277, %278
  %290 = extractvalue { double, double } %269, 1
  %291 = fneg double %290
  %292 = fcmp oge double %291, %47
  %293 = call double @__nv_cos(double %284)
  %294 = fmul double %286, %293
  %295 = call double @__nv_exp(double %281)
  %296 = fmul double %288, %286
  %297 = fmul double %295, %287
  %298 = and i1 %289, %292
  %299 = fcmp ogt double %291, %49
  %300 = fcmp oeq double %295, 0x7FF0000000000000
  %301 = fmul double %294, %286
  %302 = fmul double %295, %293
  %303 = fcmp oeq double %284, 0.000000e+00
  %304 = select i1 %300, double %296, double %297
  %305 = and i1 %298, %299
  %306 = fcmp olt double %291, %51
  %307 = select i1 %300, double %301, double %302
  %308 = select i1 %303, double 0.000000e+00, double %304
  %309 = and i1 %305, %306
  %310 = fcmp ole double %291, %53
  %311 = getelementptr inbounds [196812800 x { double, double }], ptr %1, i32 0, i64 %267
  %312 = load { double, double }, ptr %311, align 8, !invariant.load !3
  %313 = and i1 %309, %310
  %314 = extractvalue { double, double } %312, 0
  %315 = extractvalue { double, double } %312, 1
  %316 = fmul double %314, %307
  %317 = fmul double %315, %308
  %318 = fsub double %316, %317
  %319 = fmul double %315, %307
  %320 = fmul double %314, %308
  %321 = fadd double %319, %320
  %322 = insertvalue { double, double } poison, double %318, 0
  %323 = insertvalue { double, double } %322, double %321, 1
  %324 = add i64 %117, 3
  %325 = getelementptr inbounds [49203200 x { double, double }], ptr %0, i32 0, i64 %324
  %326 = load { double, double }, ptr %325, align 8
  %327 = select i1 %313, { double, double } %323, { double, double } zeroinitializer
  %328 = extractvalue { double, double } %326, 0
  %329 = extractvalue { double, double } %327, 0
  %330 = fadd double %328, %329
  %331 = extractvalue { double, double } %326, 1
  %332 = extractvalue { double, double } %327, 1
  %333 = fadd double %331, %332
  %334 = insertvalue { double, double } poison, double %330, 0
  %335 = insertvalue { double, double } %334, double %333, 1
  store { double, double } %335, ptr %325, align 8
  ret void
}

declare double @__nv_exp(double)

declare double @__nv_sin(double)

declare double @__nv_cos(double)

define ptx_kernel void @loop_add_fusion_1(ptr noalias align 256 dereferenceable(8) %0, ptr noalias align 256 dereferenceable(8) %1) #2 {
  %3 = getelementptr inbounds [1 x i64], ptr %0, i32 0, i32 0
  %4 = load i64, ptr %3, align 4
  %5 = add i64 %4, 1
  store i64 %5, ptr %3, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { "nvvm.reqntid"="1,1,1" }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 96100}
!2 = !{i32 0, i32 128}
!3 = !{}
