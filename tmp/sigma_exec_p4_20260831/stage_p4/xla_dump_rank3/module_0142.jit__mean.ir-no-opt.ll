; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@global_smem = external addrspace(3) global [0 x i8], align 16

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32) #1

define ptx_kernel void @input_reduce_fusion(ptr noalias align 16 dereferenceable(12582912) %arg0, ptr noalias align 256 dereferenceable(262144) %arg1) #3 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = addrspacecast ptr null to ptr addrspace(1)
  %4 = addrspacecast ptr null to ptr addrspace(1)
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = sext i32 %5 to i64
  %7 = mul i64 %6, 8
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %9 = and i32 %8, 63
  %10 = urem i32 %9, 32
  %11 = udiv i32 %8, 32
  %12 = shl i32 %10, 0
  %13 = or i32 0, %12
  %14 = shl i32 %11, 5
  %15 = or i32 %13, %14
  %16 = and i32 %15, 60
  %17 = lshr i32 %16, 2
  %18 = or disjoint i32 %17, 0
  %19 = xor i32 0, %18
  %20 = xor i32 %19, 0
  %21 = xor i32 %19, 16
  %22 = xor i32 %19, 32
  %23 = xor i32 %19, 48
  %24 = add i32 %20, 0
  %25 = add i32 %21, 0
  %26 = add i32 %22, 0
  %27 = add i32 %23, 0
  %28 = sext i32 %24 to i64
  %29 = sext i32 %25 to i64
  %30 = sext i32 %26 to i64
  %31 = sext i32 %27 to i64
  %32 = icmp slt i64 %28, 48
  %33 = icmp slt i64 %29, 48
  %34 = icmp slt i64 %30, 48
  %35 = icmp slt i64 %31, 48
  %36 = mul i64 %28, 32768
  %37 = mul i64 %29, 32768
  %38 = mul i64 %30, 32768
  %39 = mul i64 %31, 32768
  %40 = and i32 %15, 3
  %41 = shl i32 %40, 1
  %42 = or disjoint i32 %41, 0
  %43 = xor i32 0, %42
  %44 = xor i32 %43, 0
  %45 = add i32 %44, 0
  %46 = and i32 %15, 7
  %47 = lshr i32 %46, 0
  %48 = or disjoint i32 %47, 0
  %49 = xor i32 0, %48
  %50 = xor i32 %49, 0
  %51 = add i32 %50, 0
  %52 = sext i32 %45 to i64
  %53 = sext i32 %51 to i64
  %54 = add i64 %52, %36
  %55 = add i64 %52, %37
  %56 = add i64 %52, %38
  %57 = add i64 %52, %39
  %58 = getelementptr double, ptr addrspace(1) %1, i64 %7
  %59 = getelementptr double, ptr addrspace(1) %58, i64 %54
  %60 = getelementptr double, ptr addrspace(1) %58, i64 %55
  %61 = getelementptr double, ptr addrspace(1) %58, i64 %56
  %62 = getelementptr double, ptr addrspace(1) %58, i64 %57
  %63 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %59, i1 %32)
  %64 = extractvalue { i64, i64 } %63, 0
  %65 = bitcast i64 %64 to <1 x double>
  %66 = extractvalue { i64, i64 } %63, 1
  %67 = bitcast i64 %66 to <1 x double>
  %68 = extractelement <1 x double> %65, i32 0
  %69 = extractelement <1 x double> %67, i32 0
  %70 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %60, i1 %33)
  %71 = extractvalue { i64, i64 } %70, 0
  %72 = bitcast i64 %71 to <1 x double>
  %73 = extractvalue { i64, i64 } %70, 1
  %74 = bitcast i64 %73 to <1 x double>
  %75 = extractelement <1 x double> %72, i32 0
  %76 = extractelement <1 x double> %74, i32 0
  %77 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %61, i1 %34)
  %78 = extractvalue { i64, i64 } %77, 0
  %79 = bitcast i64 %78 to <1 x double>
  %80 = extractvalue { i64, i64 } %77, 1
  %81 = bitcast i64 %80 to <1 x double>
  %82 = extractelement <1 x double> %79, i32 0
  %83 = extractelement <1 x double> %81, i32 0
  %84 = call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %62, i1 %35)
  %85 = extractvalue { i64, i64 } %84, 0
  %86 = bitcast i64 %85 to <1 x double>
  %87 = extractvalue { i64, i64 } %84, 1
  %88 = bitcast i64 %87 to <1 x double>
  %89 = extractelement <1 x double> %86, i32 0
  %90 = extractelement <1 x double> %88, i32 0
  %91 = icmp slt i32 %24, 48
  %92 = icmp slt i32 %25, 48
  %93 = icmp slt i32 %26, 48
  %94 = icmp slt i32 %27, 48
  %95 = select i1 %91, double %68, double 0.000000e+00
  %96 = select i1 %91, double %69, double 0.000000e+00
  %97 = select i1 %92, double %75, double 0.000000e+00
  %98 = select i1 %92, double %76, double 0.000000e+00
  %99 = select i1 %93, double %82, double 0.000000e+00
  %100 = select i1 %93, double %83, double 0.000000e+00
  %101 = select i1 %94, double %89, double 0.000000e+00
  %102 = select i1 %94, double %90, double 0.000000e+00
  %103 = fadd double %95, %97
  %104 = fadd double %99, %101
  %105 = fadd double %103, %104
  %106 = fadd double %96, %98
  %107 = fadd double %100, %102
  %108 = fadd double %106, %107
  %109 = bitcast double %105 to <2 x float>
  %110 = extractelement <2 x float> %109, i32 0
  %111 = extractelement <2 x float> %109, i32 1
  %112 = bitcast float %110 to i32
  %113 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %112, i32 16, i32 31)
  %114 = bitcast i32 %113 to float
  %115 = bitcast float %111 to i32
  %116 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %115, i32 16, i32 31)
  %117 = bitcast i32 %116 to float
  %118 = insertelement <2 x float> undef, float %114, i32 0
  %119 = insertelement <2 x float> %118, float %117, i32 1
  %120 = bitcast <2 x float> %119 to double
  %121 = fadd double %105, %120
  %122 = bitcast double %121 to <2 x float>
  %123 = extractelement <2 x float> %122, i32 0
  %124 = extractelement <2 x float> %122, i32 1
  %125 = bitcast float %123 to i32
  %126 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %125, i32 8, i32 31)
  %127 = bitcast i32 %126 to float
  %128 = bitcast float %124 to i32
  %129 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %128, i32 8, i32 31)
  %130 = bitcast i32 %129 to float
  %131 = insertelement <2 x float> undef, float %127, i32 0
  %132 = insertelement <2 x float> %131, float %130, i32 1
  %133 = bitcast <2 x float> %132 to double
  %134 = fadd double %121, %133
  %135 = bitcast double %134 to <2 x float>
  %136 = extractelement <2 x float> %135, i32 0
  %137 = extractelement <2 x float> %135, i32 1
  %138 = bitcast float %136 to i32
  %139 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %138, i32 4, i32 31)
  %140 = bitcast i32 %139 to float
  %141 = bitcast float %137 to i32
  %142 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %141, i32 4, i32 31)
  %143 = bitcast i32 %142 to float
  %144 = insertelement <2 x float> undef, float %140, i32 0
  %145 = insertelement <2 x float> %144, float %143, i32 1
  %146 = bitcast <2 x float> %145 to double
  %147 = fadd double %134, %146
  %148 = bitcast double %108 to <2 x float>
  %149 = extractelement <2 x float> %148, i32 0
  %150 = extractelement <2 x float> %148, i32 1
  %151 = bitcast float %149 to i32
  %152 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %151, i32 16, i32 31)
  %153 = bitcast i32 %152 to float
  %154 = bitcast float %150 to i32
  %155 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %154, i32 16, i32 31)
  %156 = bitcast i32 %155 to float
  %157 = insertelement <2 x float> undef, float %153, i32 0
  %158 = insertelement <2 x float> %157, float %156, i32 1
  %159 = bitcast <2 x float> %158 to double
  %160 = fadd double %108, %159
  %161 = bitcast double %160 to <2 x float>
  %162 = extractelement <2 x float> %161, i32 0
  %163 = extractelement <2 x float> %161, i32 1
  %164 = bitcast float %162 to i32
  %165 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %164, i32 8, i32 31)
  %166 = bitcast i32 %165 to float
  %167 = bitcast float %163 to i32
  %168 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %167, i32 8, i32 31)
  %169 = bitcast i32 %168 to float
  %170 = insertelement <2 x float> undef, float %166, i32 0
  %171 = insertelement <2 x float> %170, float %169, i32 1
  %172 = bitcast <2 x float> %171 to double
  %173 = fadd double %160, %172
  %174 = bitcast double %173 to <2 x float>
  %175 = extractelement <2 x float> %174, i32 0
  %176 = extractelement <2 x float> %174, i32 1
  %177 = bitcast float %175 to i32
  %178 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %177, i32 4, i32 31)
  %179 = bitcast i32 %178 to float
  %180 = bitcast float %176 to i32
  %181 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %180, i32 4, i32 31)
  %182 = bitcast i32 %181 to float
  %183 = insertelement <2 x float> undef, float %179, i32 0
  %184 = insertelement <2 x float> %183, float %182, i32 1
  %185 = bitcast <2 x float> %184 to double
  %186 = fadd double %173, %185
  %187 = shl i32 %40, 5
  %188 = and i32 %15, 32
  %189 = icmp eq i32 %188, 0
  %190 = select i1 %189, i32 0, i32 16
  %191 = or disjoint i32 %187, %190
  %192 = or disjoint i32 %191, 0
  %193 = xor i32 0, %192
  %194 = xor i32 %193, 0
  %195 = xor i32 %194, 0
  %196 = add i32 %195, 0
  %197 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %196
  %198 = insertelement <2 x double> undef, double %147, i32 0
  %199 = insertelement <2 x double> %198, double %186, i32 1
  %200 = extractelement <2 x double> %199, i32 0
  %201 = extractelement <2 x double> %199, i32 1
  %202 = bitcast double %200 to i64
  %203 = bitcast double %201 to i64
  %204 = insertelement <2 x i64> undef, i64 %202, i32 0
  %205 = insertelement <2 x i64> %204, i64 %203, i32 1
  store <2 x i64> %205, ptr addrspace(3) %197, align 16
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %206 = and i32 %15, 4
  %207 = icmp eq i32 %206, 0
  %208 = select i1 %207, i32 0, i32 16
  %209 = or disjoint i32 %187, %208
  %210 = or disjoint i32 %209, 0
  %211 = xor i32 0, %210
  %212 = xor i32 %211, 0
  %213 = xor i32 %212, 0
  %214 = add i32 %213, 0
  %215 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 %214
  %216 = load <2 x i64>, ptr addrspace(3) %215, align 16
  %217 = extractelement <2 x i64> %216, i32 0
  %218 = extractelement <2 x i64> %216, i32 1
  %219 = insertelement <2 x i64> undef, i64 %217, i32 0
  %220 = insertelement <2 x i64> %219, i64 %218, i32 1
  %221 = extractelement <2 x i64> %220, i32 0
  %222 = extractelement <2 x i64> %220, i32 1
  %223 = bitcast i64 %221 to double
  %224 = bitcast i64 %222 to double
  %225 = insertelement <2 x double> undef, double %223, i32 0
  %226 = insertelement <2 x double> %225, double %224, i32 1
  %227 = extractelement <2 x double> %226, i32 0
  %228 = extractelement <2 x double> %226, i32 1
  %229 = bitcast double %227 to <2 x float>
  %230 = extractelement <2 x float> %229, i32 0
  %231 = extractelement <2 x float> %229, i32 1
  %232 = bitcast float %230 to i32
  %233 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %232, i32 4, i32 31)
  %234 = bitcast i32 %233 to float
  %235 = bitcast float %231 to i32
  %236 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %235, i32 4, i32 31)
  %237 = bitcast i32 %236 to float
  %238 = insertelement <2 x float> undef, float %234, i32 0
  %239 = insertelement <2 x float> %238, float %237, i32 1
  %240 = bitcast <2 x float> %239 to double
  %241 = fadd double %227, %240
  %242 = bitcast double %228 to <2 x float>
  %243 = extractelement <2 x float> %242, i32 0
  %244 = extractelement <2 x float> %242, i32 1
  %245 = bitcast float %243 to i32
  %246 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %245, i32 4, i32 31)
  %247 = bitcast i32 %246 to float
  %248 = bitcast float %244 to i32
  %249 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %248, i32 4, i32 31)
  %250 = bitcast i32 %249 to float
  %251 = insertelement <2 x float> undef, float %247, i32 0
  %252 = insertelement <2 x float> %251, float %250, i32 1
  %253 = bitcast <2 x float> %252 to double
  %254 = fadd double %228, %253
  %255 = getelementptr double, ptr addrspace(1) %2, i64 %7
  %256 = getelementptr double, ptr addrspace(1) %255, i64 %53
  %257 = and i32 %10, 4
  %258 = icmp eq i32 %257, 0
  %259 = select i1 %258, double %241, double %254
  %260 = select i1 %258, double %254, double %241
  %261 = and i32 %10, 24
  %262 = lshr i32 %261, 0
  %263 = and i32 %10, 6
  %264 = lshr i32 %263, 1
  %265 = and i32 %10, 1
  %266 = icmp eq i32 %265, 0
  %267 = select i1 %266, i32 0, i32 4
  %268 = or disjoint i32 %262, %264
  %269 = or disjoint i32 %268, %267
  %270 = or disjoint i32 %269, 0
  %271 = xor i32 %270, 0
  %272 = bitcast double %259 to <2 x float>
  %273 = extractelement <2 x float> %272, i32 0
  %274 = extractelement <2 x float> %272, i32 1
  %275 = bitcast float %273 to i32
  %276 = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %275, i32 %271, i32 31)
  %277 = bitcast i32 %276 to float
  %278 = bitcast float %274 to i32
  %279 = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %278, i32 %271, i32 31)
  %280 = bitcast i32 %279 to float
  %281 = insertelement <2 x float> undef, float %277, i32 0
  %282 = insertelement <2 x float> %281, float %280, i32 1
  %283 = bitcast <2 x float> %282 to double
  %284 = xor i32 %270, 4
  %285 = bitcast double %260 to <2 x float>
  %286 = extractelement <2 x float> %285, i32 0
  %287 = extractelement <2 x float> %285, i32 1
  %288 = bitcast float %286 to i32
  %289 = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %288, i32 %284, i32 31)
  %290 = bitcast i32 %289 to float
  %291 = bitcast float %287 to i32
  %292 = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %291, i32 %284, i32 31)
  %293 = bitcast i32 %292 to float
  %294 = insertelement <2 x float> undef, float %290, i32 0
  %295 = insertelement <2 x float> %294, float %293, i32 1
  %296 = bitcast <2 x float> %295 to double
  %297 = select i1 %266, double %283, double %296
  %298 = icmp eq i32 %261, 0
  %299 = and i32 %11, 1
  %300 = icmp eq i32 %299, 0
  %301 = and i1 %298, %300
  %302 = insertelement <1 x double> undef, double %297, i32 0
  %303 = bitcast <1 x double> %302 to i64
  call void asm sideeffect "@$2 st.global.b64 [ $1 + 0 ], { $0 };", "l,l,b"(i64 %303, ptr addrspace(1) %256, i1 %301)
  ret void
}

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 dereferenceable(262144) %0, ptr noalias align 256 dereferenceable(262144) %1) #4 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %5 = mul i32 %3, 128
  %6 = add i32 %5, %4
  %7 = getelementptr inbounds [32768 x double], ptr %0, i32 0, i32 %6
  %8 = load double, ptr %7, align 8
  %9 = fmul double %8, 0x3F95555555555555
  store double %9, ptr %7, align 8
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { "nvvm.reqntid"="64,1,1" }
attributes #4 = { "nvvm.reqntid"="128,1,1" }

!llvm.module.flags = !{!0}
!nvvm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 256}
!2 = !{i32 0, i32 128}
