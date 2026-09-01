; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_2 = private unnamed_addr addrspace(3) global [16 x i64] undef
@shared_1 = private unnamed_addr addrspace(3) global [16 x i64] undef
@shared_0 = private unnamed_addr addrspace(3) global [16 x double] undef
@global_smem = external local_unnamed_addr addrspace(3) global [0 x i8], align 16
@shared_01 = private unnamed_addr addrspace(3) global [13 x i64] undef

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_reduce_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(787251200) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(51200) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(51200) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(51200) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %3 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %1 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %10 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %11 = mul nuw nsw i32 %10, 7688
  %12 = add nuw nsw i32 %11, %9
  %13 = zext nneg i32 %12 to i64
  %14 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %13
  %15 = load <2 x double>, ptr addrspace(1) %14, align 16, !invariant.load !6
  %.unpack82 = extractelement <2 x double> %15, i32 0
  %.unpack283 = extractelement <2 x double> %15, i32 1
  %16 = fcmp uno double %.unpack82, %.unpack283
  %17 = zext i1 %16 to i64
  %18 = tail call double @llvm.nvvm.fabs.f64(double %.unpack82)
  %19 = fcmp ueq double %18, 0x7FF0000000000000
  %20 = tail call double @llvm.nvvm.fabs.f64(double %.unpack283)
  %21 = fcmp ueq double %20, 0x7FF0000000000000
  %.not4 = or i1 %19, %21
  %22 = zext i1 %.not4 to i64
  %23 = tail call double @llvm.maximum.f64(double %18, double %20)
  %24 = tail call double @llvm.minimum.f64(double %18, double %20)
  %25 = fdiv double %24, %23
  %26 = fmul double %25, %25
  %27 = fadd double %26, 1.000000e+00
  %28 = tail call double @llvm.sqrt.f64(double %27)
  %29 = fmul double %23, %28
  %30 = fcmp uno double %29, 0.000000e+00
  %31 = select i1 %30, double %24, double %29
  %32 = select i1 %.not4, double 0.000000e+00, double %31
  %33 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 8192
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !6
  %.unpack584 = extractelement <2 x double> %34, i32 0
  %.unpack785 = extractelement <2 x double> %34, i32 1
  %35 = fcmp uno double %.unpack584, %.unpack785
  %36 = zext i1 %35 to i64
  %37 = add nuw nsw i64 %36, %17
  %38 = tail call double @llvm.nvvm.fabs.f64(double %.unpack584)
  %39 = fcmp ueq double %38, 0x7FF0000000000000
  %40 = tail call double @llvm.nvvm.fabs.f64(double %.unpack785)
  %41 = fcmp ueq double %40, 0x7FF0000000000000
  %.not9 = or i1 %39, %41
  %42 = zext i1 %.not9 to i64
  %43 = add nuw nsw i64 %42, %22
  %44 = tail call double @llvm.maximum.f64(double %38, double %40)
  %45 = tail call double @llvm.minimum.f64(double %38, double %40)
  %46 = fdiv double %45, %44
  %47 = fmul double %46, %46
  %48 = fadd double %47, 1.000000e+00
  %49 = tail call double @llvm.sqrt.f64(double %48)
  %50 = fmul double %44, %49
  %51 = fcmp uno double %50, 0.000000e+00
  %52 = select i1 %51, double %45, double %50
  %53 = select i1 %.not9, double 0.000000e+00, double %52
  %54 = tail call double @llvm.maximum.f64(double %32, double %53)
  %55 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 16384
  %56 = load <2 x double>, ptr addrspace(1) %55, align 16, !invariant.load !6
  %.unpack1086 = extractelement <2 x double> %56, i32 0
  %.unpack1287 = extractelement <2 x double> %56, i32 1
  %57 = fcmp uno double %.unpack1086, %.unpack1287
  %58 = zext i1 %57 to i64
  %59 = add nuw nsw i64 %37, %58
  %60 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1086)
  %61 = fcmp ueq double %60, 0x7FF0000000000000
  %62 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1287)
  %63 = fcmp ueq double %62, 0x7FF0000000000000
  %.not14 = or i1 %61, %63
  %64 = zext i1 %.not14 to i64
  %65 = add nuw nsw i64 %43, %64
  %66 = tail call double @llvm.maximum.f64(double %60, double %62)
  %67 = tail call double @llvm.minimum.f64(double %60, double %62)
  %68 = fdiv double %67, %66
  %69 = fmul double %68, %68
  %70 = fadd double %69, 1.000000e+00
  %71 = tail call double @llvm.sqrt.f64(double %70)
  %72 = fmul double %66, %71
  %73 = fcmp uno double %72, 0.000000e+00
  %74 = select i1 %73, double %67, double %72
  %75 = select i1 %.not14, double 0.000000e+00, double %74
  %76 = tail call double @llvm.maximum.f64(double %54, double %75)
  %77 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 24576
  %78 = load <2 x double>, ptr addrspace(1) %77, align 16, !invariant.load !6
  %.unpack1588 = extractelement <2 x double> %78, i32 0
  %.unpack1789 = extractelement <2 x double> %78, i32 1
  %79 = fcmp uno double %.unpack1588, %.unpack1789
  %80 = zext i1 %79 to i64
  %81 = add nuw nsw i64 %59, %80
  %82 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1588)
  %83 = fcmp ueq double %82, 0x7FF0000000000000
  %84 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1789)
  %85 = fcmp ueq double %84, 0x7FF0000000000000
  %.not19 = or i1 %83, %85
  %86 = zext i1 %.not19 to i64
  %87 = add nuw nsw i64 %65, %86
  %88 = tail call double @llvm.maximum.f64(double %82, double %84)
  %89 = tail call double @llvm.minimum.f64(double %82, double %84)
  %90 = fdiv double %89, %88
  %91 = fmul double %90, %90
  %92 = fadd double %91, 1.000000e+00
  %93 = tail call double @llvm.sqrt.f64(double %92)
  %94 = fmul double %88, %93
  %95 = fcmp uno double %94, 0.000000e+00
  %96 = select i1 %95, double %89, double %94
  %97 = select i1 %.not19, double 0.000000e+00, double %96
  %98 = tail call double @llvm.maximum.f64(double %76, double %97)
  %99 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 32768
  %100 = load <2 x double>, ptr addrspace(1) %99, align 16, !invariant.load !6
  %.unpack2090 = extractelement <2 x double> %100, i32 0
  %.unpack2291 = extractelement <2 x double> %100, i32 1
  %101 = fcmp uno double %.unpack2090, %.unpack2291
  %102 = zext i1 %101 to i64
  %103 = add nuw nsw i64 %81, %102
  %104 = tail call double @llvm.nvvm.fabs.f64(double %.unpack2090)
  %105 = fcmp ueq double %104, 0x7FF0000000000000
  %106 = tail call double @llvm.nvvm.fabs.f64(double %.unpack2291)
  %107 = fcmp ueq double %106, 0x7FF0000000000000
  %.not24 = or i1 %105, %107
  %108 = zext i1 %.not24 to i64
  %109 = add nuw nsw i64 %87, %108
  %110 = tail call double @llvm.maximum.f64(double %104, double %106)
  %111 = tail call double @llvm.minimum.f64(double %104, double %106)
  %112 = fdiv double %111, %110
  %113 = fmul double %112, %112
  %114 = fadd double %113, 1.000000e+00
  %115 = tail call double @llvm.sqrt.f64(double %114)
  %116 = fmul double %110, %115
  %117 = fcmp uno double %116, 0.000000e+00
  %118 = select i1 %117, double %111, double %116
  %119 = select i1 %.not24, double 0.000000e+00, double %118
  %120 = tail call double @llvm.maximum.f64(double %98, double %119)
  %121 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 40960
  %122 = load <2 x double>, ptr addrspace(1) %121, align 16, !invariant.load !6
  %.unpack2592 = extractelement <2 x double> %122, i32 0
  %.unpack2793 = extractelement <2 x double> %122, i32 1
  %123 = fcmp uno double %.unpack2592, %.unpack2793
  %124 = zext i1 %123 to i64
  %125 = add nuw nsw i64 %103, %124
  %126 = tail call double @llvm.nvvm.fabs.f64(double %.unpack2592)
  %127 = fcmp ueq double %126, 0x7FF0000000000000
  %128 = tail call double @llvm.nvvm.fabs.f64(double %.unpack2793)
  %129 = fcmp ueq double %128, 0x7FF0000000000000
  %.not29 = or i1 %127, %129
  %130 = zext i1 %.not29 to i64
  %131 = add nuw nsw i64 %109, %130
  %132 = tail call double @llvm.maximum.f64(double %126, double %128)
  %133 = tail call double @llvm.minimum.f64(double %126, double %128)
  %134 = fdiv double %133, %132
  %135 = fmul double %134, %134
  %136 = fadd double %135, 1.000000e+00
  %137 = tail call double @llvm.sqrt.f64(double %136)
  %138 = fmul double %132, %137
  %139 = fcmp uno double %138, 0.000000e+00
  %140 = select i1 %139, double %133, double %138
  %141 = select i1 %.not29, double 0.000000e+00, double %140
  %142 = tail call double @llvm.maximum.f64(double %120, double %141)
  %143 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 49152
  %144 = load <2 x double>, ptr addrspace(1) %143, align 16, !invariant.load !6
  %.unpack3094 = extractelement <2 x double> %144, i32 0
  %.unpack3295 = extractelement <2 x double> %144, i32 1
  %145 = fcmp uno double %.unpack3094, %.unpack3295
  %146 = zext i1 %145 to i64
  %147 = add nuw nsw i64 %125, %146
  %148 = tail call double @llvm.nvvm.fabs.f64(double %.unpack3094)
  %149 = fcmp ueq double %148, 0x7FF0000000000000
  %150 = tail call double @llvm.nvvm.fabs.f64(double %.unpack3295)
  %151 = fcmp ueq double %150, 0x7FF0000000000000
  %.not34 = or i1 %149, %151
  %152 = zext i1 %.not34 to i64
  %153 = add nuw nsw i64 %131, %152
  %154 = tail call double @llvm.maximum.f64(double %148, double %150)
  %155 = tail call double @llvm.minimum.f64(double %148, double %150)
  %156 = fdiv double %155, %154
  %157 = fmul double %156, %156
  %158 = fadd double %157, 1.000000e+00
  %159 = tail call double @llvm.sqrt.f64(double %158)
  %160 = fmul double %154, %159
  %161 = fcmp uno double %160, 0.000000e+00
  %162 = select i1 %161, double %155, double %160
  %163 = select i1 %.not34, double 0.000000e+00, double %162
  %164 = tail call double @llvm.maximum.f64(double %142, double %163)
  %165 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 57344
  %166 = load <2 x double>, ptr addrspace(1) %165, align 16, !invariant.load !6
  %.unpack3596 = extractelement <2 x double> %166, i32 0
  %.unpack3797 = extractelement <2 x double> %166, i32 1
  %167 = fcmp uno double %.unpack3596, %.unpack3797
  %168 = zext i1 %167 to i64
  %169 = add nuw nsw i64 %147, %168
  %170 = tail call double @llvm.nvvm.fabs.f64(double %.unpack3596)
  %171 = fcmp ueq double %170, 0x7FF0000000000000
  %172 = tail call double @llvm.nvvm.fabs.f64(double %.unpack3797)
  %173 = fcmp ueq double %172, 0x7FF0000000000000
  %.not39 = or i1 %171, %173
  %174 = zext i1 %.not39 to i64
  %175 = add nuw nsw i64 %153, %174
  %176 = tail call double @llvm.maximum.f64(double %170, double %172)
  %177 = tail call double @llvm.minimum.f64(double %170, double %172)
  %178 = fdiv double %177, %176
  %179 = fmul double %178, %178
  %180 = fadd double %179, 1.000000e+00
  %181 = tail call double @llvm.sqrt.f64(double %180)
  %182 = fmul double %176, %181
  %183 = fcmp uno double %182, 0.000000e+00
  %184 = select i1 %183, double %177, double %182
  %185 = select i1 %.not39, double 0.000000e+00, double %184
  %186 = tail call double @llvm.maximum.f64(double %164, double %185)
  %187 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 65536
  %188 = load <2 x double>, ptr addrspace(1) %187, align 16, !invariant.load !6
  %.unpack4098 = extractelement <2 x double> %188, i32 0
  %.unpack4299 = extractelement <2 x double> %188, i32 1
  %189 = fcmp uno double %.unpack4098, %.unpack4299
  %190 = zext i1 %189 to i64
  %191 = add nuw nsw i64 %169, %190
  %192 = tail call double @llvm.nvvm.fabs.f64(double %.unpack4098)
  %193 = fcmp ueq double %192, 0x7FF0000000000000
  %194 = tail call double @llvm.nvvm.fabs.f64(double %.unpack4299)
  %195 = fcmp ueq double %194, 0x7FF0000000000000
  %.not44 = or i1 %193, %195
  %196 = zext i1 %.not44 to i64
  %197 = add nuw nsw i64 %175, %196
  %198 = tail call double @llvm.maximum.f64(double %192, double %194)
  %199 = tail call double @llvm.minimum.f64(double %192, double %194)
  %200 = fdiv double %199, %198
  %201 = fmul double %200, %200
  %202 = fadd double %201, 1.000000e+00
  %203 = tail call double @llvm.sqrt.f64(double %202)
  %204 = fmul double %198, %203
  %205 = fcmp uno double %204, 0.000000e+00
  %206 = select i1 %205, double %199, double %204
  %207 = select i1 %.not44, double 0.000000e+00, double %206
  %208 = tail call double @llvm.maximum.f64(double %186, double %207)
  %209 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 73728
  %210 = load <2 x double>, ptr addrspace(1) %209, align 16, !invariant.load !6
  %.unpack45100 = extractelement <2 x double> %210, i32 0
  %.unpack47101 = extractelement <2 x double> %210, i32 1
  %211 = fcmp uno double %.unpack45100, %.unpack47101
  %212 = zext i1 %211 to i64
  %213 = add nuw nsw i64 %191, %212
  %214 = tail call double @llvm.nvvm.fabs.f64(double %.unpack45100)
  %215 = fcmp ueq double %214, 0x7FF0000000000000
  %216 = tail call double @llvm.nvvm.fabs.f64(double %.unpack47101)
  %217 = fcmp ueq double %216, 0x7FF0000000000000
  %.not49 = or i1 %215, %217
  %218 = zext i1 %.not49 to i64
  %219 = add nuw nsw i64 %197, %218
  %220 = tail call double @llvm.maximum.f64(double %214, double %216)
  %221 = tail call double @llvm.minimum.f64(double %214, double %216)
  %222 = fdiv double %221, %220
  %223 = fmul double %222, %222
  %224 = fadd double %223, 1.000000e+00
  %225 = tail call double @llvm.sqrt.f64(double %224)
  %226 = fmul double %220, %225
  %227 = fcmp uno double %226, 0.000000e+00
  %228 = select i1 %227, double %221, double %226
  %229 = select i1 %.not49, double 0.000000e+00, double %228
  %230 = tail call double @llvm.maximum.f64(double %208, double %229)
  %231 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 81920
  %232 = load <2 x double>, ptr addrspace(1) %231, align 16, !invariant.load !6
  %.unpack50102 = extractelement <2 x double> %232, i32 0
  %.unpack52103 = extractelement <2 x double> %232, i32 1
  %233 = fcmp uno double %.unpack50102, %.unpack52103
  %234 = zext i1 %233 to i64
  %235 = add nuw nsw i64 %213, %234
  %236 = tail call double @llvm.nvvm.fabs.f64(double %.unpack50102)
  %237 = fcmp ueq double %236, 0x7FF0000000000000
  %238 = tail call double @llvm.nvvm.fabs.f64(double %.unpack52103)
  %239 = fcmp ueq double %238, 0x7FF0000000000000
  %.not54 = or i1 %237, %239
  %240 = zext i1 %.not54 to i64
  %241 = add nuw nsw i64 %219, %240
  %242 = tail call double @llvm.maximum.f64(double %236, double %238)
  %243 = tail call double @llvm.minimum.f64(double %236, double %238)
  %244 = fdiv double %243, %242
  %245 = fmul double %244, %244
  %246 = fadd double %245, 1.000000e+00
  %247 = tail call double @llvm.sqrt.f64(double %246)
  %248 = fmul double %242, %247
  %249 = fcmp uno double %248, 0.000000e+00
  %250 = select i1 %249, double %243, double %248
  %251 = select i1 %.not54, double 0.000000e+00, double %250
  %252 = tail call double @llvm.maximum.f64(double %230, double %251)
  %253 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 90112
  %254 = load <2 x double>, ptr addrspace(1) %253, align 16, !invariant.load !6
  %.unpack55104 = extractelement <2 x double> %254, i32 0
  %.unpack57105 = extractelement <2 x double> %254, i32 1
  %255 = fcmp uno double %.unpack55104, %.unpack57105
  %256 = zext i1 %255 to i64
  %257 = add nuw nsw i64 %235, %256
  %258 = tail call double @llvm.nvvm.fabs.f64(double %.unpack55104)
  %259 = fcmp ueq double %258, 0x7FF0000000000000
  %260 = tail call double @llvm.nvvm.fabs.f64(double %.unpack57105)
  %261 = fcmp ueq double %260, 0x7FF0000000000000
  %.not59 = or i1 %259, %261
  %262 = zext i1 %.not59 to i64
  %263 = add nuw nsw i64 %241, %262
  %264 = tail call double @llvm.maximum.f64(double %258, double %260)
  %265 = tail call double @llvm.minimum.f64(double %258, double %260)
  %266 = fdiv double %265, %264
  %267 = fmul double %266, %266
  %268 = fadd double %267, 1.000000e+00
  %269 = tail call double @llvm.sqrt.f64(double %268)
  %270 = fmul double %264, %269
  %271 = fcmp uno double %270, 0.000000e+00
  %272 = select i1 %271, double %265, double %270
  %273 = select i1 %.not59, double 0.000000e+00, double %272
  %274 = tail call double @llvm.maximum.f64(double %252, double %273)
  %275 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 98304
  %276 = load <2 x double>, ptr addrspace(1) %275, align 16, !invariant.load !6
  %.unpack60106 = extractelement <2 x double> %276, i32 0
  %.unpack62107 = extractelement <2 x double> %276, i32 1
  %277 = fcmp uno double %.unpack60106, %.unpack62107
  %278 = zext i1 %277 to i64
  %279 = add nuw nsw i64 %257, %278
  %280 = tail call double @llvm.nvvm.fabs.f64(double %.unpack60106)
  %281 = fcmp ueq double %280, 0x7FF0000000000000
  %282 = tail call double @llvm.nvvm.fabs.f64(double %.unpack62107)
  %283 = fcmp ueq double %282, 0x7FF0000000000000
  %.not64 = or i1 %281, %283
  %284 = zext i1 %.not64 to i64
  %285 = add nuw nsw i64 %263, %284
  %286 = tail call double @llvm.maximum.f64(double %280, double %282)
  %287 = tail call double @llvm.minimum.f64(double %280, double %282)
  %288 = fdiv double %287, %286
  %289 = fmul double %288, %288
  %290 = fadd double %289, 1.000000e+00
  %291 = tail call double @llvm.sqrt.f64(double %290)
  %292 = fmul double %286, %291
  %293 = fcmp uno double %292, 0.000000e+00
  %294 = select i1 %293, double %287, double %292
  %295 = select i1 %.not64, double 0.000000e+00, double %294
  %296 = tail call double @llvm.maximum.f64(double %274, double %295)
  %297 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 106496
  %298 = load <2 x double>, ptr addrspace(1) %297, align 16, !invariant.load !6
  %.unpack65108 = extractelement <2 x double> %298, i32 0
  %.unpack67109 = extractelement <2 x double> %298, i32 1
  %299 = fcmp uno double %.unpack65108, %.unpack67109
  %300 = zext i1 %299 to i64
  %301 = add nuw nsw i64 %279, %300
  %302 = tail call double @llvm.nvvm.fabs.f64(double %.unpack65108)
  %303 = fcmp ueq double %302, 0x7FF0000000000000
  %304 = tail call double @llvm.nvvm.fabs.f64(double %.unpack67109)
  %305 = fcmp ueq double %304, 0x7FF0000000000000
  %.not69 = or i1 %303, %305
  %306 = zext i1 %.not69 to i64
  %307 = add nuw nsw i64 %285, %306
  %308 = tail call double @llvm.maximum.f64(double %302, double %304)
  %309 = tail call double @llvm.minimum.f64(double %302, double %304)
  %310 = fdiv double %309, %308
  %311 = fmul double %310, %310
  %312 = fadd double %311, 1.000000e+00
  %313 = tail call double @llvm.sqrt.f64(double %312)
  %314 = fmul double %308, %313
  %315 = fcmp uno double %314, 0.000000e+00
  %316 = select i1 %315, double %309, double %314
  %317 = select i1 %.not69, double 0.000000e+00, double %316
  %318 = tail call double @llvm.maximum.f64(double %296, double %317)
  %319 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 114688
  %320 = load <2 x double>, ptr addrspace(1) %319, align 16, !invariant.load !6
  %.unpack70110 = extractelement <2 x double> %320, i32 0
  %.unpack72111 = extractelement <2 x double> %320, i32 1
  %321 = fcmp uno double %.unpack70110, %.unpack72111
  %322 = zext i1 %321 to i64
  %323 = add nuw nsw i64 %301, %322
  %324 = tail call double @llvm.nvvm.fabs.f64(double %.unpack70110)
  %325 = fcmp ueq double %324, 0x7FF0000000000000
  %326 = tail call double @llvm.nvvm.fabs.f64(double %.unpack72111)
  %327 = fcmp ueq double %326, 0x7FF0000000000000
  %.not74 = or i1 %325, %327
  %328 = zext i1 %.not74 to i64
  %329 = add nuw nsw i64 %307, %328
  %330 = tail call double @llvm.maximum.f64(double %324, double %326)
  %331 = tail call double @llvm.minimum.f64(double %324, double %326)
  %332 = fdiv double %331, %330
  %333 = fmul double %332, %332
  %334 = fadd double %333, 1.000000e+00
  %335 = tail call double @llvm.sqrt.f64(double %334)
  %336 = fmul double %330, %335
  %337 = fcmp uno double %336, 0.000000e+00
  %338 = select i1 %337, double %331, double %336
  %339 = select i1 %.not74, double 0.000000e+00, double %338
  %340 = tail call double @llvm.maximum.f64(double %318, double %339)
  %341 = icmp samesign ult i32 %9, 8
  br i1 %341, label %342, label %365

342:                                              ; preds = %4
  %343 = getelementptr inbounds i8, ptr addrspace(1) %14, i64 122880
  %344 = load <2 x double>, ptr addrspace(1) %343, align 16, !invariant.load !6
  %.unpack7580 = extractelement <2 x double> %344, i32 0
  %.unpack7781 = extractelement <2 x double> %344, i32 1
  %345 = fcmp uno double %.unpack7580, %.unpack7781
  %346 = zext i1 %345 to i64
  %347 = add nuw nsw i64 %323, %346
  %348 = tail call double @llvm.nvvm.fabs.f64(double %.unpack7580)
  %349 = fcmp ueq double %348, 0x7FF0000000000000
  %350 = tail call double @llvm.nvvm.fabs.f64(double %.unpack7781)
  %351 = fcmp ueq double %350, 0x7FF0000000000000
  %.not79 = or i1 %349, %351
  %352 = zext i1 %.not79 to i64
  %353 = add nuw nsw i64 %329, %352
  %354 = tail call double @llvm.maximum.f64(double %348, double %350)
  %355 = tail call double @llvm.minimum.f64(double %348, double %350)
  %356 = fdiv double %355, %354
  %357 = fmul double %356, %356
  %358 = fadd double %357, 1.000000e+00
  %359 = tail call double @llvm.sqrt.f64(double %358)
  %360 = fmul double %354, %359
  %361 = fcmp uno double %360, 0.000000e+00
  %362 = select i1 %361, double %355, double %360
  %363 = select i1 %.not79, double 0.000000e+00, double %362
  %364 = tail call double @llvm.maximum.f64(double %340, double %363)
  br label %365

365:                                              ; preds = %342, %4
  %366 = phi i64 [ %347, %342 ], [ %323, %4 ]
  %367 = phi i64 [ %353, %342 ], [ %329, %4 ]
  %368 = phi double [ %364, %342 ], [ %340, %4 ]
  %369 = bitcast i64 %366 to <2 x i32>
  %370 = extractelement <2 x i32> %369, i64 0
  %371 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %370, i32 16, i32 31)
  %372 = insertelement <2 x i32> poison, i32 %371, i64 0
  %373 = extractelement <2 x i32> %369, i64 1
  %374 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %373, i32 16, i32 31)
  %375 = insertelement <2 x i32> %372, i32 %374, i64 1
  %376 = bitcast <2 x i32> %375 to i64
  %377 = add i64 %366, %376
  %378 = bitcast i64 %377 to <2 x i32>
  %379 = extractelement <2 x i32> %378, i64 0
  %380 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %379, i32 8, i32 31)
  %381 = insertelement <2 x i32> poison, i32 %380, i64 0
  %382 = extractelement <2 x i32> %378, i64 1
  %383 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %382, i32 8, i32 31)
  %384 = insertelement <2 x i32> %381, i32 %383, i64 1
  %385 = bitcast <2 x i32> %384 to i64
  %386 = add i64 %377, %385
  %387 = bitcast i64 %386 to <2 x i32>
  %388 = extractelement <2 x i32> %387, i64 0
  %389 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %388, i32 4, i32 31)
  %390 = insertelement <2 x i32> poison, i32 %389, i64 0
  %391 = extractelement <2 x i32> %387, i64 1
  %392 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %391, i32 4, i32 31)
  %393 = insertelement <2 x i32> %390, i32 %392, i64 1
  %394 = bitcast <2 x i32> %393 to i64
  %395 = add i64 %386, %394
  %396 = bitcast i64 %395 to <2 x i32>
  %397 = extractelement <2 x i32> %396, i64 0
  %398 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %397, i32 2, i32 31)
  %399 = insertelement <2 x i32> poison, i32 %398, i64 0
  %400 = extractelement <2 x i32> %396, i64 1
  %401 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %400, i32 2, i32 31)
  %402 = insertelement <2 x i32> %399, i32 %401, i64 1
  %403 = bitcast <2 x i32> %402 to i64
  %404 = add i64 %395, %403
  %405 = bitcast i64 %404 to <2 x i32>
  %406 = extractelement <2 x i32> %405, i64 0
  %407 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %406, i32 1, i32 31)
  %408 = extractelement <2 x i32> %405, i64 1
  %409 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %408, i32 1, i32 31)
  %410 = bitcast i64 %367 to <2 x i32>
  %411 = extractelement <2 x i32> %410, i64 0
  %412 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %411, i32 16, i32 31)
  %413 = insertelement <2 x i32> poison, i32 %412, i64 0
  %414 = extractelement <2 x i32> %410, i64 1
  %415 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %414, i32 16, i32 31)
  %416 = insertelement <2 x i32> %413, i32 %415, i64 1
  %417 = bitcast <2 x i32> %416 to i64
  %418 = add i64 %367, %417
  %419 = bitcast i64 %418 to <2 x i32>
  %420 = extractelement <2 x i32> %419, i64 0
  %421 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %420, i32 8, i32 31)
  %422 = insertelement <2 x i32> poison, i32 %421, i64 0
  %423 = extractelement <2 x i32> %419, i64 1
  %424 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %423, i32 8, i32 31)
  %425 = insertelement <2 x i32> %422, i32 %424, i64 1
  %426 = bitcast <2 x i32> %425 to i64
  %427 = add i64 %418, %426
  %428 = bitcast i64 %427 to <2 x i32>
  %429 = extractelement <2 x i32> %428, i64 0
  %430 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %429, i32 4, i32 31)
  %431 = insertelement <2 x i32> poison, i32 %430, i64 0
  %432 = extractelement <2 x i32> %428, i64 1
  %433 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %432, i32 4, i32 31)
  %434 = insertelement <2 x i32> %431, i32 %433, i64 1
  %435 = bitcast <2 x i32> %434 to i64
  %436 = add i64 %427, %435
  %437 = bitcast i64 %436 to <2 x i32>
  %438 = extractelement <2 x i32> %437, i64 0
  %439 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %438, i32 2, i32 31)
  %440 = insertelement <2 x i32> poison, i32 %439, i64 0
  %441 = extractelement <2 x i32> %437, i64 1
  %442 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %441, i32 2, i32 31)
  %443 = insertelement <2 x i32> %440, i32 %442, i64 1
  %444 = bitcast <2 x i32> %443 to i64
  %445 = add i64 %436, %444
  %446 = bitcast i64 %445 to <2 x i32>
  %447 = extractelement <2 x i32> %446, i64 0
  %448 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %447, i32 1, i32 31)
  %449 = extractelement <2 x i32> %446, i64 1
  %450 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %449, i32 1, i32 31)
  %451 = bitcast double %368 to <2 x i32>
  %452 = extractelement <2 x i32> %451, i64 0
  %453 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %452, i32 16, i32 31)
  %454 = insertelement <2 x i32> poison, i32 %453, i64 0
  %455 = extractelement <2 x i32> %451, i64 1
  %456 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %455, i32 16, i32 31)
  %457 = insertelement <2 x i32> %454, i32 %456, i64 1
  %458 = bitcast <2 x i32> %457 to double
  %459 = tail call double @llvm.maximum.f64(double %368, double %458)
  %460 = bitcast double %459 to <2 x i32>
  %461 = extractelement <2 x i32> %460, i64 0
  %462 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %461, i32 8, i32 31)
  %463 = insertelement <2 x i32> poison, i32 %462, i64 0
  %464 = extractelement <2 x i32> %460, i64 1
  %465 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %464, i32 8, i32 31)
  %466 = insertelement <2 x i32> %463, i32 %465, i64 1
  %467 = bitcast <2 x i32> %466 to double
  %468 = tail call double @llvm.maximum.f64(double %459, double %467)
  %469 = bitcast double %468 to <2 x i32>
  %470 = extractelement <2 x i32> %469, i64 0
  %471 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %470, i32 4, i32 31)
  %472 = insertelement <2 x i32> poison, i32 %471, i64 0
  %473 = extractelement <2 x i32> %469, i64 1
  %474 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %473, i32 4, i32 31)
  %475 = insertelement <2 x i32> %472, i32 %474, i64 1
  %476 = bitcast <2 x i32> %475 to double
  %477 = tail call double @llvm.maximum.f64(double %468, double %476)
  %478 = bitcast double %477 to <2 x i32>
  %479 = extractelement <2 x i32> %478, i64 0
  %480 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %479, i32 2, i32 31)
  %481 = insertelement <2 x i32> poison, i32 %480, i64 0
  %482 = extractelement <2 x i32> %478, i64 1
  %483 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %482, i32 2, i32 31)
  %484 = insertelement <2 x i32> %481, i32 %483, i64 1
  %485 = bitcast <2 x i32> %484 to double
  %486 = tail call double @llvm.maximum.f64(double %477, double %485)
  %487 = bitcast double %486 to <2 x i32>
  %488 = extractelement <2 x i32> %487, i64 0
  %489 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %488, i32 1, i32 31)
  %490 = extractelement <2 x i32> %487, i64 1
  %491 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %490, i32 1, i32 31)
  %492 = and i32 %9, 31
  %493 = icmp eq i32 %492, 0
  br i1 %493, label %494, label %512

494:                                              ; preds = %365
  %495 = lshr exact i32 %9, 5
  %496 = zext nneg i32 %495 to i64
  %497 = getelementptr inbounds double, ptr addrspace(3) @shared_0, i64 %496
  %498 = getelementptr inbounds i64, ptr addrspace(3) @shared_1, i64 %496
  %499 = getelementptr inbounds i64, ptr addrspace(3) @shared_2, i64 %496
  %500 = insertelement <2 x i32> poison, i32 %489, i64 0
  %501 = insertelement <2 x i32> %500, i32 %491, i64 1
  %502 = bitcast <2 x i32> %501 to double
  %503 = tail call double @llvm.maximum.f64(double %486, double %502)
  %504 = insertelement <2 x i32> poison, i32 %448, i64 0
  %505 = insertelement <2 x i32> %504, i32 %450, i64 1
  %506 = bitcast <2 x i32> %505 to i64
  %507 = add i64 %445, %506
  %508 = insertelement <2 x i32> poison, i32 %407, i64 0
  %509 = insertelement <2 x i32> %508, i32 %409, i64 1
  %510 = bitcast <2 x i32> %509 to i64
  %511 = add i64 %404, %510
  store i64 %511, ptr addrspace(3) %499, align 4
  store i64 %507, ptr addrspace(3) %498, align 4
  store double %503, ptr addrspace(3) %497, align 8
  br label %512

512:                                              ; preds = %494, %365
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %513 = icmp samesign ult i32 %9, 32
  br i1 %513, label %514, label %669

514:                                              ; preds = %512
  %515 = icmp samesign ult i32 %9, 16
  %516 = zext nneg i32 %9 to i64
  %517 = getelementptr inbounds double, ptr addrspace(3) @shared_0, i64 %516
  %518 = getelementptr inbounds i64, ptr addrspace(3) @shared_1, i64 %516
  %519 = getelementptr inbounds i64, ptr addrspace(3) @shared_2, i64 %516
  br i1 %515, label %520, label %524

520:                                              ; preds = %514
  %521 = load i64, ptr addrspace(3) %519, align 4
  %522 = load i64, ptr addrspace(3) %518, align 4
  %523 = load double, ptr addrspace(3) %517, align 8
  br label %524

524:                                              ; preds = %520, %514
  %525 = phi i64 [ %521, %520 ], [ 0, %514 ]
  %526 = phi i64 [ %522, %520 ], [ 0, %514 ]
  %527 = phi double [ %523, %520 ], [ 0xFFF0000000000000, %514 ]
  %528 = bitcast i64 %525 to <2 x i32>
  %529 = extractelement <2 x i32> %528, i64 0
  %530 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %529, i32 16, i32 31)
  %531 = insertelement <2 x i32> poison, i32 %530, i64 0
  %532 = extractelement <2 x i32> %528, i64 1
  %533 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %532, i32 16, i32 31)
  %534 = insertelement <2 x i32> %531, i32 %533, i64 1
  %535 = bitcast <2 x i32> %534 to i64
  %536 = add i64 %525, %535
  %537 = bitcast i64 %536 to <2 x i32>
  %538 = extractelement <2 x i32> %537, i64 0
  %539 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %538, i32 8, i32 31)
  %540 = insertelement <2 x i32> poison, i32 %539, i64 0
  %541 = extractelement <2 x i32> %537, i64 1
  %542 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %541, i32 8, i32 31)
  %543 = insertelement <2 x i32> %540, i32 %542, i64 1
  %544 = bitcast <2 x i32> %543 to i64
  %545 = add i64 %536, %544
  %546 = bitcast i64 %545 to <2 x i32>
  %547 = extractelement <2 x i32> %546, i64 0
  %548 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %547, i32 4, i32 31)
  %549 = insertelement <2 x i32> poison, i32 %548, i64 0
  %550 = extractelement <2 x i32> %546, i64 1
  %551 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %550, i32 4, i32 31)
  %552 = insertelement <2 x i32> %549, i32 %551, i64 1
  %553 = bitcast <2 x i32> %552 to i64
  %554 = add i64 %545, %553
  %555 = bitcast i64 %554 to <2 x i32>
  %556 = extractelement <2 x i32> %555, i64 0
  %557 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %556, i32 2, i32 31)
  %558 = insertelement <2 x i32> poison, i32 %557, i64 0
  %559 = extractelement <2 x i32> %555, i64 1
  %560 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %559, i32 2, i32 31)
  %561 = insertelement <2 x i32> %558, i32 %560, i64 1
  %562 = bitcast <2 x i32> %561 to i64
  %563 = add i64 %554, %562
  %564 = bitcast i64 %563 to <2 x i32>
  %565 = extractelement <2 x i32> %564, i64 0
  %566 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %565, i32 1, i32 31)
  %567 = extractelement <2 x i32> %564, i64 1
  %568 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %567, i32 1, i32 31)
  %569 = bitcast i64 %526 to <2 x i32>
  %570 = extractelement <2 x i32> %569, i64 0
  %571 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %570, i32 16, i32 31)
  %572 = insertelement <2 x i32> poison, i32 %571, i64 0
  %573 = extractelement <2 x i32> %569, i64 1
  %574 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %573, i32 16, i32 31)
  %575 = insertelement <2 x i32> %572, i32 %574, i64 1
  %576 = bitcast <2 x i32> %575 to i64
  %577 = add i64 %526, %576
  %578 = bitcast i64 %577 to <2 x i32>
  %579 = extractelement <2 x i32> %578, i64 0
  %580 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %579, i32 8, i32 31)
  %581 = insertelement <2 x i32> poison, i32 %580, i64 0
  %582 = extractelement <2 x i32> %578, i64 1
  %583 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %582, i32 8, i32 31)
  %584 = insertelement <2 x i32> %581, i32 %583, i64 1
  %585 = bitcast <2 x i32> %584 to i64
  %586 = add i64 %577, %585
  %587 = bitcast i64 %586 to <2 x i32>
  %588 = extractelement <2 x i32> %587, i64 0
  %589 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %588, i32 4, i32 31)
  %590 = insertelement <2 x i32> poison, i32 %589, i64 0
  %591 = extractelement <2 x i32> %587, i64 1
  %592 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %591, i32 4, i32 31)
  %593 = insertelement <2 x i32> %590, i32 %592, i64 1
  %594 = bitcast <2 x i32> %593 to i64
  %595 = add i64 %586, %594
  %596 = bitcast i64 %595 to <2 x i32>
  %597 = extractelement <2 x i32> %596, i64 0
  %598 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %597, i32 2, i32 31)
  %599 = insertelement <2 x i32> poison, i32 %598, i64 0
  %600 = extractelement <2 x i32> %596, i64 1
  %601 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %600, i32 2, i32 31)
  %602 = insertelement <2 x i32> %599, i32 %601, i64 1
  %603 = bitcast <2 x i32> %602 to i64
  %604 = add i64 %595, %603
  %605 = bitcast i64 %604 to <2 x i32>
  %606 = extractelement <2 x i32> %605, i64 0
  %607 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %606, i32 1, i32 31)
  %608 = extractelement <2 x i32> %605, i64 1
  %609 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %608, i32 1, i32 31)
  %610 = bitcast double %527 to <2 x i32>
  %611 = extractelement <2 x i32> %610, i64 0
  %612 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %611, i32 16, i32 31)
  %613 = insertelement <2 x i32> poison, i32 %612, i64 0
  %614 = extractelement <2 x i32> %610, i64 1
  %615 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %614, i32 16, i32 31)
  %616 = insertelement <2 x i32> %613, i32 %615, i64 1
  %617 = bitcast <2 x i32> %616 to double
  %618 = tail call double @llvm.maximum.f64(double %527, double %617)
  %619 = bitcast double %618 to <2 x i32>
  %620 = extractelement <2 x i32> %619, i64 0
  %621 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %620, i32 8, i32 31)
  %622 = insertelement <2 x i32> poison, i32 %621, i64 0
  %623 = extractelement <2 x i32> %619, i64 1
  %624 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %623, i32 8, i32 31)
  %625 = insertelement <2 x i32> %622, i32 %624, i64 1
  %626 = bitcast <2 x i32> %625 to double
  %627 = tail call double @llvm.maximum.f64(double %618, double %626)
  %628 = bitcast double %627 to <2 x i32>
  %629 = extractelement <2 x i32> %628, i64 0
  %630 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %629, i32 4, i32 31)
  %631 = insertelement <2 x i32> poison, i32 %630, i64 0
  %632 = extractelement <2 x i32> %628, i64 1
  %633 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %632, i32 4, i32 31)
  %634 = insertelement <2 x i32> %631, i32 %633, i64 1
  %635 = bitcast <2 x i32> %634 to double
  %636 = tail call double @llvm.maximum.f64(double %627, double %635)
  %637 = bitcast double %636 to <2 x i32>
  %638 = extractelement <2 x i32> %637, i64 0
  %639 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %638, i32 2, i32 31)
  %640 = insertelement <2 x i32> poison, i32 %639, i64 0
  %641 = extractelement <2 x i32> %637, i64 1
  %642 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %641, i32 2, i32 31)
  %643 = insertelement <2 x i32> %640, i32 %642, i64 1
  %644 = bitcast <2 x i32> %643 to double
  %645 = tail call double @llvm.maximum.f64(double %636, double %644)
  %646 = bitcast double %645 to <2 x i32>
  %647 = extractelement <2 x i32> %646, i64 0
  %648 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %647, i32 1, i32 31)
  %649 = extractelement <2 x i32> %646, i64 1
  %650 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %649, i32 1, i32 31)
  %651 = icmp eq i32 %9, 0
  br i1 %651, label %652, label %669

652:                                              ; preds = %524
  %653 = zext nneg i32 %10 to i64
  %654 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %653
  %655 = getelementptr inbounds i64, ptr addrspace(1) %7, i64 %653
  %656 = getelementptr inbounds i64, ptr addrspace(1) %8, i64 %653
  %657 = insertelement <2 x i32> poison, i32 %648, i64 0
  %658 = insertelement <2 x i32> %657, i32 %650, i64 1
  %659 = bitcast <2 x i32> %658 to double
  %660 = tail call double @llvm.maximum.f64(double %645, double %659)
  %661 = insertelement <2 x i32> poison, i32 %607, i64 0
  %662 = insertelement <2 x i32> %661, i32 %609, i64 1
  %663 = bitcast <2 x i32> %662 to i64
  %664 = add i64 %604, %663
  %665 = insertelement <2 x i32> poison, i32 %566, i64 0
  %666 = insertelement <2 x i32> %665, i32 %568, i64 1
  %667 = bitcast <2 x i32> %666 to i64
  %668 = add i64 %563, %667
  store i64 %668, ptr addrspace(1) %656, align 8
  store i64 %664, ptr addrspace(1) %655, align 8
  store double %660, ptr addrspace(1) %654, align 8
  br label %669

669:                                              ; preds = %524, %652, %512
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maximum.f64(double, double) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minimum.f64(double, double) #2

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #3

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #4

; Function Attrs: nounwind
define ptx_kernel void @input_reduce_fusion_3(ptr noalias align 256 dereferenceable(51200) %arg0, ptr noalias align 256 dereferenceable(8) %arg1) local_unnamed_addr #5 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = tail call range(i32 0, 256) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = shl i32 %3, 1
  %5 = or disjoint i32 %4, 6144
  %6 = zext i32 %4 to i64
  %7 = icmp samesign ult i32 %5, 6400
  %8 = getelementptr double, ptr addrspace(1) %1, i64 %6
  %9 = getelementptr i8, ptr addrspace(1) %8, i64 4096
  %10 = getelementptr i8, ptr addrspace(1) %8, i64 8192
  %11 = getelementptr i8, ptr addrspace(1) %8, i64 12288
  %12 = getelementptr i8, ptr addrspace(1) %8, i64 16384
  %13 = getelementptr i8, ptr addrspace(1) %8, i64 20480
  %14 = getelementptr i8, ptr addrspace(1) %8, i64 24576
  %15 = getelementptr i8, ptr addrspace(1) %8, i64 28672
  %16 = getelementptr i8, ptr addrspace(1) %8, i64 32768
  %17 = getelementptr i8, ptr addrspace(1) %8, i64 36864
  %18 = getelementptr i8, ptr addrspace(1) %8, i64 40960
  %19 = getelementptr i8, ptr addrspace(1) %8, i64 45056
  %20 = getelementptr i8, ptr addrspace(1) %8, i64 49152
  %21 = getelementptr i8, ptr addrspace(1) %8, i64 53248
  %22 = getelementptr i8, ptr addrspace(1) %8, i64 57344
  %23 = getelementptr i8, ptr addrspace(1) %8, i64 61440
  %24 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %8, i1 true) #9
  %25 = extractvalue { i64, i64 } %24, 0
  %26 = extractvalue { i64, i64 } %24, 1
  %27 = bitcast i64 %25 to double
  %28 = bitcast i64 %26 to double
  %29 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %9, i1 true) #9
  %30 = extractvalue { i64, i64 } %29, 0
  %31 = extractvalue { i64, i64 } %29, 1
  %32 = bitcast i64 %30 to double
  %33 = bitcast i64 %31 to double
  %34 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %10, i1 true) #9
  %35 = extractvalue { i64, i64 } %34, 0
  %36 = extractvalue { i64, i64 } %34, 1
  %37 = bitcast i64 %35 to double
  %38 = bitcast i64 %36 to double
  %39 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %11, i1 true) #9
  %40 = extractvalue { i64, i64 } %39, 0
  %41 = extractvalue { i64, i64 } %39, 1
  %42 = bitcast i64 %40 to double
  %43 = bitcast i64 %41 to double
  %44 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %12, i1 true) #9
  %45 = extractvalue { i64, i64 } %44, 0
  %46 = extractvalue { i64, i64 } %44, 1
  %47 = bitcast i64 %45 to double
  %48 = bitcast i64 %46 to double
  %49 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %13, i1 true) #9
  %50 = extractvalue { i64, i64 } %49, 0
  %51 = extractvalue { i64, i64 } %49, 1
  %52 = bitcast i64 %50 to double
  %53 = bitcast i64 %51 to double
  %54 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %14, i1 true) #9
  %55 = extractvalue { i64, i64 } %54, 0
  %56 = extractvalue { i64, i64 } %54, 1
  %57 = bitcast i64 %55 to double
  %58 = bitcast i64 %56 to double
  %59 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %15, i1 true) #9
  %60 = extractvalue { i64, i64 } %59, 0
  %61 = extractvalue { i64, i64 } %59, 1
  %62 = bitcast i64 %60 to double
  %63 = bitcast i64 %61 to double
  %64 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %16, i1 true) #9
  %65 = extractvalue { i64, i64 } %64, 0
  %66 = extractvalue { i64, i64 } %64, 1
  %67 = bitcast i64 %65 to double
  %68 = bitcast i64 %66 to double
  %69 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %17, i1 true) #9
  %70 = extractvalue { i64, i64 } %69, 0
  %71 = extractvalue { i64, i64 } %69, 1
  %72 = bitcast i64 %70 to double
  %73 = bitcast i64 %71 to double
  %74 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %18, i1 true) #9
  %75 = extractvalue { i64, i64 } %74, 0
  %76 = extractvalue { i64, i64 } %74, 1
  %77 = bitcast i64 %75 to double
  %78 = bitcast i64 %76 to double
  %79 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %19, i1 true) #9
  %80 = extractvalue { i64, i64 } %79, 0
  %81 = extractvalue { i64, i64 } %79, 1
  %82 = bitcast i64 %80 to double
  %83 = bitcast i64 %81 to double
  %84 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %20, i1 %7) #9
  %85 = extractvalue { i64, i64 } %84, 0
  %86 = extractvalue { i64, i64 } %84, 1
  %87 = bitcast i64 %85 to double
  %88 = bitcast i64 %86 to double
  %89 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %21, i1 false) #9
  %90 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %22, i1 false) #9
  %91 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %23, i1 false) #9
  %92 = tail call double @llvm.maximum.f64(double %27, double %28)
  %93 = tail call double @llvm.maximum.f64(double %32, double %33)
  %94 = tail call double @llvm.maximum.f64(double %37, double %38)
  %95 = tail call double @llvm.maximum.f64(double %42, double %43)
  %96 = tail call double @llvm.maximum.f64(double %47, double %48)
  %97 = tail call double @llvm.maximum.f64(double %52, double %53)
  %98 = tail call double @llvm.maximum.f64(double %57, double %58)
  %99 = tail call double @llvm.maximum.f64(double %62, double %63)
  %100 = tail call double @llvm.maximum.f64(double %67, double %68)
  %101 = tail call double @llvm.maximum.f64(double %72, double %73)
  %102 = tail call double @llvm.maximum.f64(double %77, double %78)
  %103 = tail call double @llvm.maximum.f64(double %82, double %83)
  %104 = tail call double @llvm.maximum.f64(double %87, double %88)
  %105 = tail call double @llvm.maximum.f64(double %92, double %93)
  %106 = tail call double @llvm.maximum.f64(double %94, double %95)
  %107 = tail call double @llvm.maximum.f64(double %96, double %97)
  %108 = tail call double @llvm.maximum.f64(double %98, double %99)
  %109 = tail call double @llvm.maximum.f64(double %100, double %101)
  %110 = tail call double @llvm.maximum.f64(double %102, double %103)
  %111 = tail call double @llvm.maximum.f64(double %105, double %106)
  %112 = tail call double @llvm.maximum.f64(double %107, double %108)
  %113 = tail call double @llvm.maximum.f64(double %109, double %110)
  %114 = tail call double @llvm.maximum.f64(double %111, double %112)
  %115 = tail call double @llvm.maximum.f64(double %113, double %104)
  %116 = select i1 %7, double %115, double %113
  %117 = tail call double @llvm.maximum.f64(double %114, double %116)
  %bc = bitcast double %117 to <2 x i32>
  %118 = extractelement <2 x i32> %bc, i64 0
  %119 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %118, i32 16, i32 31)
  %120 = extractelement <2 x i32> %bc, i64 1
  %121 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %120, i32 16, i32 31)
  %122 = insertelement <2 x i32> poison, i32 %119, i64 0
  %123 = insertelement <2 x i32> %122, i32 %121, i64 1
  %124 = bitcast <2 x i32> %123 to double
  %125 = tail call double @llvm.maximum.f64(double %117, double %124)
  %bc2 = bitcast double %125 to <2 x i32>
  %126 = extractelement <2 x i32> %bc2, i64 0
  %127 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %126, i32 8, i32 31)
  %128 = extractelement <2 x i32> %bc2, i64 1
  %129 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %128, i32 8, i32 31)
  %130 = insertelement <2 x i32> poison, i32 %127, i64 0
  %131 = insertelement <2 x i32> %130, i32 %129, i64 1
  %132 = bitcast <2 x i32> %131 to double
  %133 = tail call double @llvm.maximum.f64(double %125, double %132)
  %bc4 = bitcast double %133 to <2 x i32>
  %134 = extractelement <2 x i32> %bc4, i64 0
  %135 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %134, i32 4, i32 31)
  %136 = extractelement <2 x i32> %bc4, i64 1
  %137 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %136, i32 4, i32 31)
  %138 = insertelement <2 x i32> poison, i32 %135, i64 0
  %139 = insertelement <2 x i32> %138, i32 %137, i64 1
  %140 = bitcast <2 x i32> %139 to double
  %141 = tail call double @llvm.maximum.f64(double %133, double %140)
  %bc6 = bitcast double %141 to <2 x i32>
  %142 = extractelement <2 x i32> %bc6, i64 0
  %143 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %142, i32 2, i32 31)
  %144 = extractelement <2 x i32> %bc6, i64 1
  %145 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %144, i32 2, i32 31)
  %146 = insertelement <2 x i32> poison, i32 %143, i64 0
  %147 = insertelement <2 x i32> %146, i32 %145, i64 1
  %148 = bitcast <2 x i32> %147 to double
  %149 = tail call double @llvm.maximum.f64(double %141, double %148)
  %bc8 = bitcast double %149 to <2 x i32>
  %150 = extractelement <2 x i32> %bc8, i64 0
  %151 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %150, i32 1, i32 31)
  %152 = extractelement <2 x i32> %bc8, i64 1
  %153 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %152, i32 1, i32 31)
  %154 = insertelement <2 x i32> poison, i32 %151, i64 0
  %155 = insertelement <2 x i32> %154, i32 %153, i64 1
  %156 = bitcast <2 x i32> %155 to double
  %157 = tail call double @llvm.maximum.f64(double %149, double %156)
  %158 = lshr i32 %3, 2
  %159 = and i32 %158, 56
  %160 = zext nneg i32 %159 to i64
  %161 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %160
  store double %157, ptr addrspace(3) %161, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %162 = shl nuw nsw i32 %3, 3
  %163 = and i32 %162, 56
  %164 = zext nneg i32 %163 to i64
  %165 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %164
  %166 = load double, ptr addrspace(3) %165, align 8
  %bc10 = bitcast double %166 to <2 x i32>
  %167 = extractelement <2 x i32> %bc10, i64 0
  %168 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %167, i32 4, i32 31)
  %169 = extractelement <2 x i32> %bc10, i64 1
  %170 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %169, i32 4, i32 31)
  %171 = insertelement <2 x i32> poison, i32 %168, i64 0
  %172 = insertelement <2 x i32> %171, i32 %170, i64 1
  %173 = bitcast <2 x i32> %172 to double
  %174 = tail call double @llvm.maximum.f64(double %166, double %173)
  %bc12 = bitcast double %174 to <2 x i32>
  %175 = extractelement <2 x i32> %bc12, i64 0
  %176 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %175, i32 2, i32 31)
  %177 = extractelement <2 x i32> %bc12, i64 1
  %178 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %177, i32 2, i32 31)
  %179 = insertelement <2 x i32> poison, i32 %176, i64 0
  %180 = insertelement <2 x i32> %179, i32 %178, i64 1
  %181 = bitcast <2 x i32> %180 to double
  %182 = tail call double @llvm.maximum.f64(double %174, double %181)
  %bc14 = bitcast double %182 to <2 x i32>
  %183 = extractelement <2 x i32> %bc14, i64 0
  %184 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %183, i32 1, i32 31)
  %185 = extractelement <2 x i32> %bc14, i64 1
  %186 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %185, i32 1, i32 31)
  %187 = insertelement <2 x i32> poison, i32 %184, i64 0
  %188 = insertelement <2 x i32> %187, i32 %186, i64 1
  %189 = bitcast <2 x i32> %188 to double
  %190 = tail call double @llvm.maximum.f64(double %182, double %189)
  %191 = icmp eq i32 %3, 0
  %192 = bitcast double %190 to i64
  tail call void asm sideeffect "@$2 st.global.b64 [ $1 + 0 ], { $0 };", "l,l,b"(i64 %192, ptr addrspace(1) %2, i1 %191) #9
  ret void
}

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #3

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_reduce_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(51200) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #6 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !7
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds i64, ptr addrspace(1) %3, i64 %6
  %8 = load i64, ptr addrspace(1) %7, align 8, !invariant.load !6
  %9 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 3328
  %10 = load i64, ptr addrspace(1) %9, align 8, !invariant.load !6
  %11 = add i64 %10, %8
  %12 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 6656
  %13 = load i64, ptr addrspace(1) %12, align 8, !invariant.load !6
  %14 = add i64 %11, %13
  %15 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 9984
  %16 = load i64, ptr addrspace(1) %15, align 8, !invariant.load !6
  %17 = add i64 %14, %16
  %18 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 13312
  %19 = load i64, ptr addrspace(1) %18, align 8, !invariant.load !6
  %20 = add i64 %17, %19
  %21 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 16640
  %22 = load i64, ptr addrspace(1) %21, align 8, !invariant.load !6
  %23 = add i64 %20, %22
  %24 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 19968
  %25 = load i64, ptr addrspace(1) %24, align 8, !invariant.load !6
  %26 = add i64 %23, %25
  %27 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 23296
  %28 = load i64, ptr addrspace(1) %27, align 8, !invariant.load !6
  %29 = add i64 %26, %28
  %30 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 26624
  %31 = load i64, ptr addrspace(1) %30, align 8, !invariant.load !6
  %32 = add i64 %29, %31
  %33 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 29952
  %34 = load i64, ptr addrspace(1) %33, align 8, !invariant.load !6
  %35 = add i64 %32, %34
  %36 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 33280
  %37 = load i64, ptr addrspace(1) %36, align 8, !invariant.load !6
  %38 = add i64 %35, %37
  %39 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 36608
  %40 = load i64, ptr addrspace(1) %39, align 8, !invariant.load !6
  %41 = add i64 %38, %40
  %42 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 39936
  %43 = load i64, ptr addrspace(1) %42, align 8, !invariant.load !6
  %44 = add i64 %41, %43
  %45 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 43264
  %46 = load i64, ptr addrspace(1) %45, align 8, !invariant.load !6
  %47 = add i64 %44, %46
  %48 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 46592
  %49 = load i64, ptr addrspace(1) %48, align 8, !invariant.load !6
  %50 = add i64 %47, %49
  %51 = icmp samesign ult i32 %5, 160
  br i1 %51, label %52, label %55

52:                                               ; preds = %2
  %sunkaddr = getelementptr inbounds i8, ptr addrspace(1) %7, i64 49920
  %53 = load i64, ptr addrspace(1) %sunkaddr, align 8, !invariant.load !6
  %54 = add i64 %53, %50
  br label %55

55:                                               ; preds = %52, %2
  %56 = phi i64 [ %54, %52 ], [ %50, %2 ]
  %57 = trunc i64 %6 to i32
  %58 = bitcast i64 %56 to <2 x i32>
  %59 = extractelement <2 x i32> %58, i64 0
  %60 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %59, i32 16, i32 31)
  %61 = insertelement <2 x i32> poison, i32 %60, i64 0
  %62 = extractelement <2 x i32> %58, i64 1
  %63 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %62, i32 16, i32 31)
  %64 = insertelement <2 x i32> %61, i32 %63, i64 1
  %65 = bitcast <2 x i32> %64 to i64
  %66 = add i64 %56, %65
  %67 = bitcast i64 %66 to <2 x i32>
  %68 = extractelement <2 x i32> %67, i64 0
  %69 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %68, i32 8, i32 31)
  %70 = insertelement <2 x i32> poison, i32 %69, i64 0
  %71 = extractelement <2 x i32> %67, i64 1
  %72 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %71, i32 8, i32 31)
  %73 = insertelement <2 x i32> %70, i32 %72, i64 1
  %74 = bitcast <2 x i32> %73 to i64
  %75 = add i64 %66, %74
  %76 = bitcast i64 %75 to <2 x i32>
  %77 = extractelement <2 x i32> %76, i64 0
  %78 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %77, i32 4, i32 31)
  %79 = insertelement <2 x i32> poison, i32 %78, i64 0
  %80 = extractelement <2 x i32> %76, i64 1
  %81 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %80, i32 4, i32 31)
  %82 = insertelement <2 x i32> %79, i32 %81, i64 1
  %83 = bitcast <2 x i32> %82 to i64
  %84 = add i64 %75, %83
  %85 = bitcast i64 %84 to <2 x i32>
  %86 = extractelement <2 x i32> %85, i64 0
  %87 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %86, i32 2, i32 31)
  %88 = insertelement <2 x i32> poison, i32 %87, i64 0
  %89 = extractelement <2 x i32> %85, i64 1
  %90 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %89, i32 2, i32 31)
  %91 = insertelement <2 x i32> %88, i32 %90, i64 1
  %92 = bitcast <2 x i32> %91 to i64
  %93 = add i64 %84, %92
  %94 = bitcast i64 %93 to <2 x i32>
  %95 = extractelement <2 x i32> %94, i64 0
  %96 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %95, i32 1, i32 31)
  %97 = extractelement <2 x i32> %94, i64 1
  %98 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %97, i32 1, i32 31)
  %99 = and i32 %57, 31
  %100 = icmp eq i32 %99, 0
  br i1 %100, label %101, label %110

101:                                              ; preds = %55
  %102 = trunc i64 %6 to i32
  %103 = lshr exact i32 %102, 5
  %104 = zext nneg i32 %103 to i64
  %105 = getelementptr inbounds i64, ptr addrspace(3) @shared_01, i64 %104
  %106 = insertelement <2 x i32> poison, i32 %96, i64 0
  %107 = insertelement <2 x i32> %106, i32 %98, i64 1
  %108 = bitcast <2 x i32> %107 to i64
  %109 = add i64 %93, %108
  store i64 %109, ptr addrspace(3) %105, align 4
  br label %110

110:                                              ; preds = %101, %55
  %111 = trunc i64 %6 to i32
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %112 = icmp samesign ult i32 %111, 32
  br i1 %112, label %113, label %169

113:                                              ; preds = %110
  %114 = trunc i64 %6 to i32
  %115 = icmp samesign ult i32 %114, 13
  %116 = getelementptr inbounds i64, ptr addrspace(3) @shared_01, i64 %6
  br i1 %115, label %117, label %119

117:                                              ; preds = %113
  %118 = load i64, ptr addrspace(3) %116, align 4
  br label %119

119:                                              ; preds = %117, %113
  %120 = phi i64 [ %118, %117 ], [ 0, %113 ]
  %121 = trunc i64 %6 to i32
  %122 = bitcast i64 %120 to <2 x i32>
  %123 = extractelement <2 x i32> %122, i64 0
  %124 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %123, i32 16, i32 31)
  %125 = insertelement <2 x i32> poison, i32 %124, i64 0
  %126 = extractelement <2 x i32> %122, i64 1
  %127 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %126, i32 16, i32 31)
  %128 = insertelement <2 x i32> %125, i32 %127, i64 1
  %129 = bitcast <2 x i32> %128 to i64
  %130 = add i64 %120, %129
  %131 = bitcast i64 %130 to <2 x i32>
  %132 = extractelement <2 x i32> %131, i64 0
  %133 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %132, i32 8, i32 31)
  %134 = insertelement <2 x i32> poison, i32 %133, i64 0
  %135 = extractelement <2 x i32> %131, i64 1
  %136 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %135, i32 8, i32 31)
  %137 = insertelement <2 x i32> %134, i32 %136, i64 1
  %138 = bitcast <2 x i32> %137 to i64
  %139 = add i64 %130, %138
  %140 = bitcast i64 %139 to <2 x i32>
  %141 = extractelement <2 x i32> %140, i64 0
  %142 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %141, i32 4, i32 31)
  %143 = insertelement <2 x i32> poison, i32 %142, i64 0
  %144 = extractelement <2 x i32> %140, i64 1
  %145 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %144, i32 4, i32 31)
  %146 = insertelement <2 x i32> %143, i32 %145, i64 1
  %147 = bitcast <2 x i32> %146 to i64
  %148 = add i64 %139, %147
  %149 = bitcast i64 %148 to <2 x i32>
  %150 = extractelement <2 x i32> %149, i64 0
  %151 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %150, i32 2, i32 31)
  %152 = insertelement <2 x i32> poison, i32 %151, i64 0
  %153 = extractelement <2 x i32> %149, i64 1
  %154 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %153, i32 2, i32 31)
  %155 = insertelement <2 x i32> %152, i32 %154, i64 1
  %156 = bitcast <2 x i32> %155 to i64
  %157 = add i64 %148, %156
  %158 = bitcast i64 %157 to <2 x i32>
  %159 = extractelement <2 x i32> %158, i64 0
  %160 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %159, i32 1, i32 31)
  %161 = extractelement <2 x i32> %158, i64 1
  %162 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %161, i32 1, i32 31)
  %163 = icmp eq i32 %121, 0
  %164 = insertelement <2 x i32> poison, i32 %160, i64 0
  %165 = insertelement <2 x i32> %164, i32 %162, i64 1
  %166 = bitcast <2 x i32> %165 to i64
  %167 = add i64 %157, %166
  br i1 %163, label %168, label %169

168:                                              ; preds = %119
  store i64 %167, ptr addrspace(1) %4, align 256
  br label %169

169:                                              ; preds = %119, %168, %110
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_concatenate(ptr noalias readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(16) initializes((0, 16)) %2) local_unnamed_addr #7 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = load i64, ptr addrspace(1) %4, align 256, !invariant.load !6
  %8 = load i64, ptr addrspace(1) %6, align 256, !invariant.load !6
  %9 = insertelement <2 x i64> poison, i64 %7, i32 0
  %10 = insertelement <2 x i64> %9, i64 %8, i32 1
  store <2 x i64> %10, ptr addrspace(1) %5, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_slice_1(ptr noalias readonly align 256 captures(none) dereferenceable(16) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %1) local_unnamed_addr #7 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = getelementptr inbounds i8, ptr addrspace(1) %3, i64 8
  %6 = load i64, ptr addrspace(1) %5, align 8, !invariant.load !6
  store i64 %6, ptr addrspace(1) %4, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_slice(ptr noalias readonly align 256 captures(none) dereferenceable(16) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %1) local_unnamed_addr #7 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = load i64, ptr addrspace(1) %3, align 256, !invariant.load !6
  store i64 %5, ptr addrspace(1) %4, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @input_concatenate_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(24) initializes((0, 24)) %3) local_unnamed_addr #7 {
  %5 = addrspacecast ptr %2 to ptr addrspace(1)
  %6 = addrspacecast ptr %3 to ptr addrspace(1)
  %7 = addrspacecast ptr %1 to ptr addrspace(1)
  %8 = addrspacecast ptr %0 to ptr addrspace(1)
  %9 = load i64, ptr addrspace(1) %5, align 256, !invariant.load !6
  %10 = sitofp i64 %9 to double
  %11 = load i64, ptr addrspace(1) %7, align 256, !invariant.load !6
  %12 = sitofp i64 %11 to double
  %13 = insertelement <2 x double> poison, double %10, i32 0
  %14 = insertelement <2 x double> %13, double %12, i32 1
  store <2 x double> %14, ptr addrspace(1) %6, align 256
  %15 = load double, ptr addrspace(1) %8, align 256, !invariant.load !6
  %16 = getelementptr inbounds i8, ptr addrspace(1) %6, i64 16
  store double %15, ptr addrspace(1) %16, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #8

attributes #0 = { norecurse nounwind "nvvm.reqntid"="512,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #4 = { convergent nocallback nounwind }
attributes #5 = { nounwind "nvvm.reqntid"="256,1,1" }
attributes #6 = { norecurse nounwind "nvvm.reqntid"="416,1,1" }
attributes #7 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { nounwind }

!llvm.module.flags = !{!0, !1}
!nvvm.annotations = !{}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 512}
!5 = !{i32 0, i32 6400}
!6 = !{}
!7 = !{i32 0, i32 416}
