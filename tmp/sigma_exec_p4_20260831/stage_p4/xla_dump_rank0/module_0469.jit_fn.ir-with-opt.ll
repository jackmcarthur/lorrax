; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(1179648) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(2048) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(2048) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(2048) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %3 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %1 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %10 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %11 = lshr i32 %9, 5
  %12 = mul nuw nsw i32 %11, 288
  %13 = mul nuw nsw i32 %10, 2304
  %14 = add nuw nsw i32 %12, %13
  %15 = and i32 %9, 31
  %16 = or disjoint i32 %14, %15
  %17 = zext nneg i32 %16 to i64
  %18 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %17
  %19 = load <2 x double>, ptr addrspace(1) %18, align 16, !invariant.load !6
  %.unpack45 = extractelement <2 x double> %19, i32 0
  %.unpack246 = extractelement <2 x double> %19, i32 1
  %20 = fcmp uno double %.unpack45, %.unpack246
  %21 = zext i1 %20 to i64
  %22 = tail call double @llvm.nvvm.fabs.f64(double %.unpack45)
  %23 = fcmp ueq double %22, 0x7FF0000000000000
  %24 = tail call double @llvm.nvvm.fabs.f64(double %.unpack246)
  %25 = fcmp ueq double %24, 0x7FF0000000000000
  %.not4 = or i1 %23, %25
  %26 = zext i1 %.not4 to i64
  %27 = tail call double @llvm.maximum.f64(double %22, double %24)
  %28 = tail call double @llvm.minimum.f64(double %22, double %24)
  %29 = fdiv double %28, %27
  %30 = fmul double %29, %29
  %31 = fadd double %30, 1.000000e+00
  %32 = tail call double @llvm.sqrt.f64(double %31)
  %33 = fmul double %27, %32
  %34 = fcmp uno double %33, 0.000000e+00
  %35 = select i1 %34, double %28, double %33
  %36 = select i1 %.not4, double 0.000000e+00, double %35
  %37 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 512
  %38 = load <2 x double>, ptr addrspace(1) %37, align 16, !invariant.load !6
  %.unpack547 = extractelement <2 x double> %38, i32 0
  %.unpack748 = extractelement <2 x double> %38, i32 1
  %39 = fcmp uno double %.unpack547, %.unpack748
  %40 = zext i1 %39 to i64
  %41 = add nuw nsw i64 %40, %21
  %42 = tail call double @llvm.nvvm.fabs.f64(double %.unpack547)
  %43 = fcmp ueq double %42, 0x7FF0000000000000
  %44 = tail call double @llvm.nvvm.fabs.f64(double %.unpack748)
  %45 = fcmp ueq double %44, 0x7FF0000000000000
  %.not9 = or i1 %43, %45
  %46 = zext i1 %.not9 to i64
  %47 = add nuw nsw i64 %46, %26
  %48 = tail call double @llvm.maximum.f64(double %42, double %44)
  %49 = tail call double @llvm.minimum.f64(double %42, double %44)
  %50 = fdiv double %49, %48
  %51 = fmul double %50, %50
  %52 = fadd double %51, 1.000000e+00
  %53 = tail call double @llvm.sqrt.f64(double %52)
  %54 = fmul double %48, %53
  %55 = fcmp uno double %54, 0.000000e+00
  %56 = select i1 %55, double %49, double %54
  %57 = select i1 %.not9, double 0.000000e+00, double %56
  %58 = tail call double @llvm.maximum.f64(double %36, double %57)
  %59 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 1024
  %60 = load <2 x double>, ptr addrspace(1) %59, align 16, !invariant.load !6
  %.unpack1049 = extractelement <2 x double> %60, i32 0
  %.unpack1250 = extractelement <2 x double> %60, i32 1
  %61 = fcmp uno double %.unpack1049, %.unpack1250
  %62 = zext i1 %61 to i64
  %63 = add nuw nsw i64 %41, %62
  %64 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1049)
  %65 = fcmp ueq double %64, 0x7FF0000000000000
  %66 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1250)
  %67 = fcmp ueq double %66, 0x7FF0000000000000
  %.not14 = or i1 %65, %67
  %68 = zext i1 %.not14 to i64
  %69 = add nuw nsw i64 %47, %68
  %70 = tail call double @llvm.maximum.f64(double %64, double %66)
  %71 = tail call double @llvm.minimum.f64(double %64, double %66)
  %72 = fdiv double %71, %70
  %73 = fmul double %72, %72
  %74 = fadd double %73, 1.000000e+00
  %75 = tail call double @llvm.sqrt.f64(double %74)
  %76 = fmul double %70, %75
  %77 = fcmp uno double %76, 0.000000e+00
  %78 = select i1 %77, double %71, double %76
  %79 = select i1 %.not14, double 0.000000e+00, double %78
  %80 = tail call double @llvm.maximum.f64(double %58, double %79)
  %81 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 1536
  %82 = load <2 x double>, ptr addrspace(1) %81, align 16, !invariant.load !6
  %.unpack1551 = extractelement <2 x double> %82, i32 0
  %.unpack1752 = extractelement <2 x double> %82, i32 1
  %83 = fcmp uno double %.unpack1551, %.unpack1752
  %84 = zext i1 %83 to i64
  %85 = add nuw nsw i64 %63, %84
  %86 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1551)
  %87 = fcmp ueq double %86, 0x7FF0000000000000
  %88 = tail call double @llvm.nvvm.fabs.f64(double %.unpack1752)
  %89 = fcmp ueq double %88, 0x7FF0000000000000
  %.not19 = or i1 %87, %89
  %90 = zext i1 %.not19 to i64
  %91 = add nuw nsw i64 %69, %90
  %92 = tail call double @llvm.maximum.f64(double %86, double %88)
  %93 = tail call double @llvm.minimum.f64(double %86, double %88)
  %94 = fdiv double %93, %92
  %95 = fmul double %94, %94
  %96 = fadd double %95, 1.000000e+00
  %97 = tail call double @llvm.sqrt.f64(double %96)
  %98 = fmul double %92, %97
  %99 = fcmp uno double %98, 0.000000e+00
  %100 = select i1 %99, double %93, double %98
  %101 = select i1 %.not19, double 0.000000e+00, double %100
  %102 = tail call double @llvm.maximum.f64(double %80, double %101)
  %103 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 2048
  %104 = load <2 x double>, ptr addrspace(1) %103, align 16, !invariant.load !6
  %.unpack2053 = extractelement <2 x double> %104, i32 0
  %.unpack2254 = extractelement <2 x double> %104, i32 1
  %105 = fcmp uno double %.unpack2053, %.unpack2254
  %106 = zext i1 %105 to i64
  %107 = add nuw nsw i64 %85, %106
  %108 = tail call double @llvm.nvvm.fabs.f64(double %.unpack2053)
  %109 = fcmp ueq double %108, 0x7FF0000000000000
  %110 = tail call double @llvm.nvvm.fabs.f64(double %.unpack2254)
  %111 = fcmp ueq double %110, 0x7FF0000000000000
  %.not24 = or i1 %109, %111
  %112 = zext i1 %.not24 to i64
  %113 = add nuw nsw i64 %91, %112
  %114 = tail call double @llvm.maximum.f64(double %108, double %110)
  %115 = tail call double @llvm.minimum.f64(double %108, double %110)
  %116 = fdiv double %115, %114
  %117 = fmul double %116, %116
  %118 = fadd double %117, 1.000000e+00
  %119 = tail call double @llvm.sqrt.f64(double %118)
  %120 = fmul double %114, %119
  %121 = fcmp uno double %120, 0.000000e+00
  %122 = select i1 %121, double %115, double %120
  %123 = select i1 %.not24, double 0.000000e+00, double %122
  %124 = tail call double @llvm.maximum.f64(double %102, double %123)
  %125 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 2560
  %126 = load <2 x double>, ptr addrspace(1) %125, align 16, !invariant.load !6
  %.unpack2555 = extractelement <2 x double> %126, i32 0
  %.unpack2756 = extractelement <2 x double> %126, i32 1
  %127 = fcmp uno double %.unpack2555, %.unpack2756
  %128 = zext i1 %127 to i64
  %129 = add nuw nsw i64 %107, %128
  %130 = tail call double @llvm.nvvm.fabs.f64(double %.unpack2555)
  %131 = fcmp ueq double %130, 0x7FF0000000000000
  %132 = tail call double @llvm.nvvm.fabs.f64(double %.unpack2756)
  %133 = fcmp ueq double %132, 0x7FF0000000000000
  %.not29 = or i1 %131, %133
  %134 = zext i1 %.not29 to i64
  %135 = add nuw nsw i64 %113, %134
  %136 = tail call double @llvm.maximum.f64(double %130, double %132)
  %137 = tail call double @llvm.minimum.f64(double %130, double %132)
  %138 = fdiv double %137, %136
  %139 = fmul double %138, %138
  %140 = fadd double %139, 1.000000e+00
  %141 = tail call double @llvm.sqrt.f64(double %140)
  %142 = fmul double %136, %141
  %143 = fcmp uno double %142, 0.000000e+00
  %144 = select i1 %143, double %137, double %142
  %145 = select i1 %.not29, double 0.000000e+00, double %144
  %146 = tail call double @llvm.maximum.f64(double %124, double %145)
  %147 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 3072
  %148 = load <2 x double>, ptr addrspace(1) %147, align 16, !invariant.load !6
  %.unpack3057 = extractelement <2 x double> %148, i32 0
  %.unpack3258 = extractelement <2 x double> %148, i32 1
  %149 = fcmp uno double %.unpack3057, %.unpack3258
  %150 = zext i1 %149 to i64
  %151 = add nuw nsw i64 %129, %150
  %152 = tail call double @llvm.nvvm.fabs.f64(double %.unpack3057)
  %153 = fcmp ueq double %152, 0x7FF0000000000000
  %154 = tail call double @llvm.nvvm.fabs.f64(double %.unpack3258)
  %155 = fcmp ueq double %154, 0x7FF0000000000000
  %.not34 = or i1 %153, %155
  %156 = zext i1 %.not34 to i64
  %157 = add nuw nsw i64 %135, %156
  %158 = tail call double @llvm.maximum.f64(double %152, double %154)
  %159 = tail call double @llvm.minimum.f64(double %152, double %154)
  %160 = fdiv double %159, %158
  %161 = fmul double %160, %160
  %162 = fadd double %161, 1.000000e+00
  %163 = tail call double @llvm.sqrt.f64(double %162)
  %164 = fmul double %158, %163
  %165 = fcmp uno double %164, 0.000000e+00
  %166 = select i1 %165, double %159, double %164
  %167 = select i1 %.not34, double 0.000000e+00, double %166
  %168 = tail call double @llvm.maximum.f64(double %146, double %167)
  %169 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 3584
  %170 = load <2 x double>, ptr addrspace(1) %169, align 16, !invariant.load !6
  %.unpack3559 = extractelement <2 x double> %170, i32 0
  %.unpack3760 = extractelement <2 x double> %170, i32 1
  %171 = fcmp uno double %.unpack3559, %.unpack3760
  %172 = zext i1 %171 to i64
  %173 = add nuw nsw i64 %151, %172
  %174 = tail call double @llvm.nvvm.fabs.f64(double %.unpack3559)
  %175 = fcmp ueq double %174, 0x7FF0000000000000
  %176 = tail call double @llvm.nvvm.fabs.f64(double %.unpack3760)
  %177 = fcmp ueq double %176, 0x7FF0000000000000
  %.not39 = or i1 %175, %177
  %178 = zext i1 %.not39 to i64
  %179 = add nuw nsw i64 %157, %178
  %180 = tail call double @llvm.maximum.f64(double %174, double %176)
  %181 = tail call double @llvm.minimum.f64(double %174, double %176)
  %182 = fdiv double %181, %180
  %183 = fmul double %182, %182
  %184 = fadd double %183, 1.000000e+00
  %185 = tail call double @llvm.sqrt.f64(double %184)
  %186 = fmul double %180, %185
  %187 = fcmp uno double %186, 0.000000e+00
  %188 = select i1 %187, double %181, double %186
  %189 = select i1 %.not39, double 0.000000e+00, double %188
  %190 = tail call double @llvm.maximum.f64(double %168, double %189)
  %191 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 4096
  %192 = load <2 x double>, ptr addrspace(1) %191, align 16, !invariant.load !6
  %.unpack4061 = extractelement <2 x double> %192, i32 0
  %.unpack4262 = extractelement <2 x double> %192, i32 1
  %193 = fcmp uno double %.unpack4061, %.unpack4262
  %194 = zext i1 %193 to i64
  %195 = add nuw nsw i64 %173, %194
  %196 = tail call double @llvm.nvvm.fabs.f64(double %.unpack4061)
  %197 = fcmp ueq double %196, 0x7FF0000000000000
  %198 = tail call double @llvm.nvvm.fabs.f64(double %.unpack4262)
  %199 = fcmp ueq double %198, 0x7FF0000000000000
  %.not44 = or i1 %197, %199
  %200 = zext i1 %.not44 to i64
  %201 = add nuw nsw i64 %179, %200
  %202 = tail call double @llvm.maximum.f64(double %196, double %198)
  %203 = tail call double @llvm.minimum.f64(double %196, double %198)
  %204 = fdiv double %203, %202
  %205 = fmul double %204, %204
  %206 = fadd double %205, 1.000000e+00
  %207 = tail call double @llvm.sqrt.f64(double %206)
  %208 = fmul double %202, %207
  %209 = fcmp uno double %208, 0.000000e+00
  %210 = select i1 %209, double %203, double %208
  %211 = select i1 %.not44, double 0.000000e+00, double %210
  %212 = tail call double @llvm.maximum.f64(double %190, double %211)
  %213 = bitcast i64 %195 to <2 x i32>
  %214 = extractelement <2 x i32> %213, i64 0
  %215 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %214, i32 16, i32 31)
  %216 = insertelement <2 x i32> poison, i32 %215, i64 0
  %217 = extractelement <2 x i32> %213, i64 1
  %218 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %217, i32 16, i32 31)
  %219 = insertelement <2 x i32> %216, i32 %218, i64 1
  %220 = bitcast <2 x i32> %219 to i64
  %221 = add i64 %195, %220
  %222 = bitcast i64 %221 to <2 x i32>
  %223 = extractelement <2 x i32> %222, i64 0
  %224 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %223, i32 8, i32 31)
  %225 = insertelement <2 x i32> poison, i32 %224, i64 0
  %226 = extractelement <2 x i32> %222, i64 1
  %227 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %226, i32 8, i32 31)
  %228 = insertelement <2 x i32> %225, i32 %227, i64 1
  %229 = bitcast <2 x i32> %228 to i64
  %230 = add i64 %221, %229
  %231 = bitcast i64 %230 to <2 x i32>
  %232 = extractelement <2 x i32> %231, i64 0
  %233 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %232, i32 4, i32 31)
  %234 = insertelement <2 x i32> poison, i32 %233, i64 0
  %235 = extractelement <2 x i32> %231, i64 1
  %236 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %235, i32 4, i32 31)
  %237 = insertelement <2 x i32> %234, i32 %236, i64 1
  %238 = bitcast <2 x i32> %237 to i64
  %239 = add i64 %230, %238
  %240 = bitcast i64 %239 to <2 x i32>
  %241 = extractelement <2 x i32> %240, i64 0
  %242 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %241, i32 2, i32 31)
  %243 = insertelement <2 x i32> poison, i32 %242, i64 0
  %244 = extractelement <2 x i32> %240, i64 1
  %245 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %244, i32 2, i32 31)
  %246 = insertelement <2 x i32> %243, i32 %245, i64 1
  %247 = bitcast <2 x i32> %246 to i64
  %248 = add i64 %239, %247
  %249 = bitcast i64 %248 to <2 x i32>
  %250 = extractelement <2 x i32> %249, i64 0
  %251 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %250, i32 1, i32 31)
  %252 = extractelement <2 x i32> %249, i64 1
  %253 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %252, i32 1, i32 31)
  %254 = bitcast i64 %201 to <2 x i32>
  %255 = extractelement <2 x i32> %254, i64 0
  %256 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %255, i32 16, i32 31)
  %257 = insertelement <2 x i32> poison, i32 %256, i64 0
  %258 = extractelement <2 x i32> %254, i64 1
  %259 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %258, i32 16, i32 31)
  %260 = insertelement <2 x i32> %257, i32 %259, i64 1
  %261 = bitcast <2 x i32> %260 to i64
  %262 = add i64 %201, %261
  %263 = bitcast i64 %262 to <2 x i32>
  %264 = extractelement <2 x i32> %263, i64 0
  %265 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %264, i32 8, i32 31)
  %266 = insertelement <2 x i32> poison, i32 %265, i64 0
  %267 = extractelement <2 x i32> %263, i64 1
  %268 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %267, i32 8, i32 31)
  %269 = insertelement <2 x i32> %266, i32 %268, i64 1
  %270 = bitcast <2 x i32> %269 to i64
  %271 = add i64 %262, %270
  %272 = bitcast i64 %271 to <2 x i32>
  %273 = extractelement <2 x i32> %272, i64 0
  %274 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %273, i32 4, i32 31)
  %275 = insertelement <2 x i32> poison, i32 %274, i64 0
  %276 = extractelement <2 x i32> %272, i64 1
  %277 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %276, i32 4, i32 31)
  %278 = insertelement <2 x i32> %275, i32 %277, i64 1
  %279 = bitcast <2 x i32> %278 to i64
  %280 = add i64 %271, %279
  %281 = bitcast i64 %280 to <2 x i32>
  %282 = extractelement <2 x i32> %281, i64 0
  %283 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %282, i32 2, i32 31)
  %284 = insertelement <2 x i32> poison, i32 %283, i64 0
  %285 = extractelement <2 x i32> %281, i64 1
  %286 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %285, i32 2, i32 31)
  %287 = insertelement <2 x i32> %284, i32 %286, i64 1
  %288 = bitcast <2 x i32> %287 to i64
  %289 = add i64 %280, %288
  %290 = bitcast i64 %289 to <2 x i32>
  %291 = extractelement <2 x i32> %290, i64 0
  %292 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %291, i32 1, i32 31)
  %293 = extractelement <2 x i32> %290, i64 1
  %294 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %293, i32 1, i32 31)
  %295 = bitcast double %212 to <2 x i32>
  %296 = extractelement <2 x i32> %295, i64 0
  %297 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %296, i32 16, i32 31)
  %298 = insertelement <2 x i32> poison, i32 %297, i64 0
  %299 = extractelement <2 x i32> %295, i64 1
  %300 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %299, i32 16, i32 31)
  %301 = insertelement <2 x i32> %298, i32 %300, i64 1
  %302 = bitcast <2 x i32> %301 to double
  %303 = tail call double @llvm.maximum.f64(double %212, double %302)
  %304 = bitcast double %303 to <2 x i32>
  %305 = extractelement <2 x i32> %304, i64 0
  %306 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %305, i32 8, i32 31)
  %307 = insertelement <2 x i32> poison, i32 %306, i64 0
  %308 = extractelement <2 x i32> %304, i64 1
  %309 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %308, i32 8, i32 31)
  %310 = insertelement <2 x i32> %307, i32 %309, i64 1
  %311 = bitcast <2 x i32> %310 to double
  %312 = tail call double @llvm.maximum.f64(double %303, double %311)
  %313 = bitcast double %312 to <2 x i32>
  %314 = extractelement <2 x i32> %313, i64 0
  %315 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %314, i32 4, i32 31)
  %316 = insertelement <2 x i32> poison, i32 %315, i64 0
  %317 = extractelement <2 x i32> %313, i64 1
  %318 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %317, i32 4, i32 31)
  %319 = insertelement <2 x i32> %316, i32 %318, i64 1
  %320 = bitcast <2 x i32> %319 to double
  %321 = tail call double @llvm.maximum.f64(double %312, double %320)
  %322 = bitcast double %321 to <2 x i32>
  %323 = extractelement <2 x i32> %322, i64 0
  %324 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %323, i32 2, i32 31)
  %325 = insertelement <2 x i32> poison, i32 %324, i64 0
  %326 = extractelement <2 x i32> %322, i64 1
  %327 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %326, i32 2, i32 31)
  %328 = insertelement <2 x i32> %325, i32 %327, i64 1
  %329 = bitcast <2 x i32> %328 to double
  %330 = tail call double @llvm.maximum.f64(double %321, double %329)
  %331 = bitcast double %330 to <2 x i32>
  %332 = extractelement <2 x i32> %331, i64 0
  %333 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %332, i32 1, i32 31)
  %334 = extractelement <2 x i32> %331, i64 1
  %335 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %334, i32 1, i32 31)
  %336 = icmp eq i32 %15, 0
  %337 = icmp samesign ult i32 %9, 225
  %338 = and i1 %337, %336
  br i1 %338, label %339, label %358

339:                                              ; preds = %4
  %340 = shl nuw nsw i32 %10, 3
  %341 = or disjoint i32 %340, %11
  %342 = zext nneg i32 %341 to i64
  %343 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %342
  %344 = getelementptr inbounds i64, ptr addrspace(1) %7, i64 %342
  %345 = getelementptr inbounds i64, ptr addrspace(1) %8, i64 %342
  %346 = insertelement <2 x i32> poison, i32 %333, i64 0
  %347 = insertelement <2 x i32> %346, i32 %335, i64 1
  %348 = bitcast <2 x i32> %347 to double
  %349 = tail call double @llvm.maximum.f64(double %330, double %348)
  %350 = insertelement <2 x i32> poison, i32 %292, i64 0
  %351 = insertelement <2 x i32> %350, i32 %294, i64 1
  %352 = bitcast <2 x i32> %351 to i64
  %353 = add i64 %289, %352
  %354 = insertelement <2 x i32> poison, i32 %251, i64 0
  %355 = insertelement <2 x i32> %354, i32 %253, i64 1
  %356 = bitcast <2 x i32> %355 to i64
  %357 = add i64 %248, %356
  store i64 %357, ptr addrspace(1) %345, align 8
  store i64 %353, ptr addrspace(1) %344, align 8
  store double %349, ptr addrspace(1) %343, align 8
  br label %358

358:                                              ; preds = %339, %4
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

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion_3(ptr noalias readonly align 256 captures(none) dereferenceable(2048) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #4 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds double, ptr addrspace(1) %3, i64 %6
  %8 = load double, ptr addrspace(1) %7, align 8, !invariant.load !6
  %9 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 256
  %10 = load double, ptr addrspace(1) %9, align 8, !invariant.load !6
  %11 = tail call double @llvm.maximum.f64(double %8, double %10)
  %12 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 512
  %13 = load double, ptr addrspace(1) %12, align 8, !invariant.load !6
  %14 = tail call double @llvm.maximum.f64(double %11, double %13)
  %15 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 768
  %16 = load double, ptr addrspace(1) %15, align 8, !invariant.load !6
  %17 = tail call double @llvm.maximum.f64(double %14, double %16)
  %18 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 1024
  %19 = load double, ptr addrspace(1) %18, align 8, !invariant.load !6
  %20 = tail call double @llvm.maximum.f64(double %17, double %19)
  %21 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 1280
  %22 = load double, ptr addrspace(1) %21, align 8, !invariant.load !6
  %23 = tail call double @llvm.maximum.f64(double %20, double %22)
  %24 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 1536
  %25 = load double, ptr addrspace(1) %24, align 8, !invariant.load !6
  %26 = tail call double @llvm.maximum.f64(double %23, double %25)
  %27 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 1792
  %28 = load double, ptr addrspace(1) %27, align 8, !invariant.load !6
  %29 = tail call double @llvm.maximum.f64(double %26, double %28)
  %30 = bitcast double %29 to <2 x i32>
  %31 = extractelement <2 x i32> %30, i64 0
  %32 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %31, i32 16, i32 31)
  %33 = insertelement <2 x i32> poison, i32 %32, i64 0
  %34 = extractelement <2 x i32> %30, i64 1
  %35 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %34, i32 16, i32 31)
  %36 = insertelement <2 x i32> %33, i32 %35, i64 1
  %37 = bitcast <2 x i32> %36 to double
  %38 = tail call double @llvm.maximum.f64(double %29, double %37)
  %39 = bitcast double %38 to <2 x i32>
  %40 = extractelement <2 x i32> %39, i64 0
  %41 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %40, i32 8, i32 31)
  %42 = insertelement <2 x i32> poison, i32 %41, i64 0
  %43 = extractelement <2 x i32> %39, i64 1
  %44 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %43, i32 8, i32 31)
  %45 = insertelement <2 x i32> %42, i32 %44, i64 1
  %46 = bitcast <2 x i32> %45 to double
  %47 = tail call double @llvm.maximum.f64(double %38, double %46)
  %48 = bitcast double %47 to <2 x i32>
  %49 = extractelement <2 x i32> %48, i64 0
  %50 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %49, i32 4, i32 31)
  %51 = insertelement <2 x i32> poison, i32 %50, i64 0
  %52 = extractelement <2 x i32> %48, i64 1
  %53 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %52, i32 4, i32 31)
  %54 = insertelement <2 x i32> %51, i32 %53, i64 1
  %55 = bitcast <2 x i32> %54 to double
  %56 = tail call double @llvm.maximum.f64(double %47, double %55)
  %57 = bitcast double %56 to <2 x i32>
  %58 = extractelement <2 x i32> %57, i64 0
  %59 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %58, i32 2, i32 31)
  %60 = insertelement <2 x i32> poison, i32 %59, i64 0
  %61 = extractelement <2 x i32> %57, i64 1
  %62 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %61, i32 2, i32 31)
  %63 = insertelement <2 x i32> %60, i32 %62, i64 1
  %64 = bitcast <2 x i32> %63 to double
  %65 = tail call double @llvm.maximum.f64(double %56, double %64)
  %66 = bitcast double %65 to <2 x i32>
  %67 = extractelement <2 x i32> %66, i64 0
  %68 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %67, i32 1, i32 31)
  %69 = extractelement <2 x i32> %66, i64 1
  %70 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %69, i32 1, i32 31)
  %71 = icmp eq i32 %5, 0
  br i1 %71, label %72, label %77

72:                                               ; preds = %2
  %73 = insertelement <2 x i32> poison, i32 %68, i64 0
  %74 = insertelement <2 x i32> %73, i32 %70, i64 1
  %75 = bitcast <2 x i32> %74 to double
  %76 = tail call double @llvm.maximum.f64(double %65, double %75)
  store double %76, ptr addrspace(1) %4, align 256
  br label %77

77:                                               ; preds = %72, %2
  ret void
}

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(2048) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #4 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds i64, ptr addrspace(1) %3, i64 %6
  %8 = load i64, ptr addrspace(1) %7, align 8, !invariant.load !6
  %9 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 256
  %10 = load i64, ptr addrspace(1) %9, align 8, !invariant.load !6
  %11 = add i64 %10, %8
  %12 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 512
  %13 = load i64, ptr addrspace(1) %12, align 8, !invariant.load !6
  %14 = add i64 %11, %13
  %15 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 768
  %16 = load i64, ptr addrspace(1) %15, align 8, !invariant.load !6
  %17 = add i64 %14, %16
  %18 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 1024
  %19 = load i64, ptr addrspace(1) %18, align 8, !invariant.load !6
  %20 = add i64 %17, %19
  %21 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 1280
  %22 = load i64, ptr addrspace(1) %21, align 8, !invariant.load !6
  %23 = add i64 %20, %22
  %24 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 1536
  %25 = load i64, ptr addrspace(1) %24, align 8, !invariant.load !6
  %26 = add i64 %23, %25
  %27 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 1792
  %28 = load i64, ptr addrspace(1) %27, align 8, !invariant.load !6
  %29 = add i64 %26, %28
  %30 = bitcast i64 %29 to <2 x i32>
  %31 = extractelement <2 x i32> %30, i64 0
  %32 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %31, i32 16, i32 31)
  %33 = insertelement <2 x i32> poison, i32 %32, i64 0
  %34 = extractelement <2 x i32> %30, i64 1
  %35 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %34, i32 16, i32 31)
  %36 = insertelement <2 x i32> %33, i32 %35, i64 1
  %37 = bitcast <2 x i32> %36 to i64
  %38 = add i64 %29, %37
  %39 = bitcast i64 %38 to <2 x i32>
  %40 = extractelement <2 x i32> %39, i64 0
  %41 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %40, i32 8, i32 31)
  %42 = insertelement <2 x i32> poison, i32 %41, i64 0
  %43 = extractelement <2 x i32> %39, i64 1
  %44 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %43, i32 8, i32 31)
  %45 = insertelement <2 x i32> %42, i32 %44, i64 1
  %46 = bitcast <2 x i32> %45 to i64
  %47 = add i64 %38, %46
  %48 = bitcast i64 %47 to <2 x i32>
  %49 = extractelement <2 x i32> %48, i64 0
  %50 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %49, i32 4, i32 31)
  %51 = insertelement <2 x i32> poison, i32 %50, i64 0
  %52 = extractelement <2 x i32> %48, i64 1
  %53 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %52, i32 4, i32 31)
  %54 = insertelement <2 x i32> %51, i32 %53, i64 1
  %55 = bitcast <2 x i32> %54 to i64
  %56 = add i64 %47, %55
  %57 = bitcast i64 %56 to <2 x i32>
  %58 = extractelement <2 x i32> %57, i64 0
  %59 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %58, i32 2, i32 31)
  %60 = insertelement <2 x i32> poison, i32 %59, i64 0
  %61 = extractelement <2 x i32> %57, i64 1
  %62 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %61, i32 2, i32 31)
  %63 = insertelement <2 x i32> %60, i32 %62, i64 1
  %64 = bitcast <2 x i32> %63 to i64
  %65 = add i64 %56, %64
  %66 = bitcast i64 %65 to <2 x i32>
  %67 = extractelement <2 x i32> %66, i64 0
  %68 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %67, i32 1, i32 31)
  %69 = extractelement <2 x i32> %66, i64 1
  %70 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %69, i32 1, i32 31)
  %71 = icmp eq i32 %5, 0
  %72 = insertelement <2 x i32> poison, i32 %68, i64 0
  %73 = insertelement <2 x i32> %72, i32 %70, i64 1
  %74 = bitcast <2 x i32> %73 to i64
  %75 = add i64 %65, %74
  br i1 %71, label %76, label %77

76:                                               ; preds = %2
  store i64 %75, ptr addrspace(1) %4, align 256
  br label %77

77:                                               ; preds = %76, %2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_concatenate(ptr noalias readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(16) initializes((0, 16)) %2) local_unnamed_addr #5 {
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
define ptx_kernel void @wrapped_slice_1(ptr noalias readonly align 256 captures(none) dereferenceable(16) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = getelementptr inbounds i8, ptr addrspace(1) %3, i64 8
  %6 = load i64, ptr addrspace(1) %5, align 8, !invariant.load !6
  store i64 %6, ptr addrspace(1) %4, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_slice(ptr noalias readonly align 256 captures(none) dereferenceable(16) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = load i64, ptr addrspace(1) %3, align 256, !invariant.load !6
  store i64 %5, ptr addrspace(1) %4, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @input_concatenate_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias readonly align 256 captures(none) dereferenceable(8) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(24) initializes((0, 24)) %3) local_unnamed_addr #5 {
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
declare double @llvm.sqrt.f64(double) #6

attributes #0 = { norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "nvvm.reqntid"="256,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #4 = { norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "nvvm.reqntid"="32,1,1" }
attributes #5 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 256}
!5 = !{i32 0, i32 32}
!6 = !{}
