; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef
@shared_1 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef
@shared_01 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef
@shared_02 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef
@shared_03 = private unnamed_addr addrspace(3) global [1056 x { double, double }] undef
@buffer_for_constant_76_0 = local_unnamed_addr addrspace(1) global [64 x i8] zeroinitializer, align 256
@buffer_for_constant_55_0 = local_unnamed_addr addrspace(1) global [64 x i8] zeroinitializer, align 256

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion_3(ptr noalias readonly align 16 captures(none) dereferenceable(24772608) %0, ptr noalias readonly align 16 captures(none) dereferenceable(196608) %1, ptr noalias readonly align 16 captures(none) dereferenceable(49152) %2, ptr noalias readonly align 256 captures(none) dereferenceable(16) %3, ptr noalias readonly align 256 captures(none) dereferenceable(4) %4, ptr noalias writeonly align 256 captures(none) dereferenceable(1179648) %5) local_unnamed_addr #0 {
  %7 = addrspacecast ptr %4 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %2 to ptr addrspace(1)
  %10 = addrspacecast ptr %0 to ptr addrspace(1)
  %11 = addrspacecast ptr %1 to ptr addrspace(1)
  %12 = addrspacecast ptr %5 to ptr addrspace(1)
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %14 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %15 = and i32 %13, 31
  %16 = icmp samesign ult i32 %15, 12
  br i1 %16, label %17, label %._crit_edge

._crit_edge:                                      ; preds = %6
  %.pre = lshr i32 %13, 5
  br label %208

17:                                               ; preds = %6
  %18 = load i32, ptr addrspace(1) %7, align 256, !invariant.load !4
  %19 = tail call i32 @llvm.umin.i32(i32 %18, i32 3)
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds i32, ptr addrspace(1) %8, i64 %20
  %22 = load i32, ptr addrspace(1) %21, align 4, !invariant.load !4
  %.not = icmp eq i32 %22, 0
  %23 = select i1 %.not, i32 0, i32 12
  %24 = add nuw nsw i32 %23, %15
  %25 = shl nuw nsw i32 %14, 3
  %26 = udiv i32 %25, 3
  %27 = mul i32 %26, 3
  %.decomposed = sub i32 %25, %27
  %28 = shl nuw nsw i32 %.decomposed, 2
  %29 = lshr i32 %13, 5
  %30 = or disjoint i32 %28, %29
  %31 = tail call i32 @llvm.umin.i32(i32 %26, i32 511)
  %32 = tail call i32 @llvm.umin.i32(i32 %30, i32 11)
  %33 = mul nuw nsw i32 %26, 24
  %34 = add nuw nsw i32 %24, %33
  %35 = zext nneg i32 %34 to i64
  %36 = getelementptr inbounds i32, ptr addrspace(1) %9, i64 %35
  %37 = load i32, ptr addrspace(1) %36, align 4, !invariant.load !4
  %38 = icmp slt i32 %37, 0
  %39 = add i32 %37, 21
  %40 = select i1 %38, i32 %39, i32 %37
  %41 = tail call i32 @llvm.smax.i32(i32 %40, i32 0)
  %42 = tail call i32 @llvm.umin.i32(i32 %41, i32 20)
  %43 = mul nuw nsw i32 %42, 73728
  %44 = mul nuw nsw i32 %31, 144
  %45 = mul nuw nsw i32 %32, 12
  %46 = add nuw nsw i32 %45, %15
  %47 = add nuw nsw i32 %46, %44
  %48 = add nuw nsw i32 %47, %43
  %49 = zext nneg i32 %48 to i64
  %50 = getelementptr inbounds { double, double }, ptr addrspace(1) %10, i64 %49
  %51 = load <2 x double>, ptr addrspace(1) %50, align 16, !invariant.load !4
  %.unpack91 = extractelement <2 x double> %51, i32 0
  %.unpack292 = extractelement <2 x double> %51, i32 1
  %52 = mul nuw nsw i32 %15, 33
  %53 = add nuw nsw i32 %52, %29
  %54 = zext nneg i32 %53 to i64
  %55 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %54
  store double %.unpack91, ptr addrspace(3) %55, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 8
  store double %.unpack292, ptr addrspace(3) %.repack3, align 8
  %56 = or disjoint i32 %25, 1
  %57 = udiv i32 %56, 3
  %58 = mul i32 %57, 3
  %.decomposed65 = sub i32 %56, %58
  %59 = shl nuw nsw i32 %.decomposed65, 2
  %60 = or disjoint i32 %59, %29
  %61 = tail call i32 @llvm.umin.i32(i32 %57, i32 511)
  %62 = tail call i32 @llvm.umin.i32(i32 %60, i32 11)
  %63 = mul nuw nsw i32 %57, 24
  %64 = add nuw nsw i32 %24, %63
  %65 = zext nneg i32 %64 to i64
  %66 = getelementptr inbounds i32, ptr addrspace(1) %9, i64 %65
  %67 = load i32, ptr addrspace(1) %66, align 4, !invariant.load !4
  %68 = icmp slt i32 %67, 0
  %69 = add i32 %67, 21
  %70 = select i1 %68, i32 %69, i32 %67
  %71 = tail call i32 @llvm.smax.i32(i32 %70, i32 0)
  %72 = tail call i32 @llvm.umin.i32(i32 %71, i32 20)
  %73 = mul nuw nsw i32 %72, 73728
  %74 = mul nuw nsw i32 %61, 144
  %75 = mul nuw nsw i32 %62, 12
  %76 = add nuw nsw i32 %75, %15
  %77 = add nuw nsw i32 %76, %74
  %78 = add nuw nsw i32 %77, %73
  %79 = zext nneg i32 %78 to i64
  %80 = getelementptr inbounds { double, double }, ptr addrspace(1) %10, i64 %79
  %81 = load <2 x double>, ptr addrspace(1) %80, align 16, !invariant.load !4
  %.unpack589 = extractelement <2 x double> %81, i32 0
  %.unpack790 = extractelement <2 x double> %81, i32 1
  %82 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 64
  store double %.unpack589, ptr addrspace(3) %82, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 72
  store double %.unpack790, ptr addrspace(3) %.repack8, align 8
  %83 = or disjoint i32 %25, 2
  %84 = udiv i32 %83, 3
  %85 = mul i32 %84, 3
  %.decomposed66 = sub i32 %83, %85
  %86 = shl nuw nsw i32 %.decomposed66, 2
  %87 = or disjoint i32 %86, %29
  %88 = tail call i32 @llvm.umin.i32(i32 %84, i32 511)
  %89 = tail call i32 @llvm.umin.i32(i32 %87, i32 11)
  %90 = mul nuw nsw i32 %84, 24
  %91 = add nuw nsw i32 %24, %90
  %92 = zext nneg i32 %91 to i64
  %93 = getelementptr inbounds i32, ptr addrspace(1) %9, i64 %92
  %94 = load i32, ptr addrspace(1) %93, align 4, !invariant.load !4
  %95 = icmp slt i32 %94, 0
  %96 = add i32 %94, 21
  %97 = select i1 %95, i32 %96, i32 %94
  %98 = tail call i32 @llvm.smax.i32(i32 %97, i32 0)
  %99 = tail call i32 @llvm.umin.i32(i32 %98, i32 20)
  %100 = mul nuw nsw i32 %99, 73728
  %101 = mul nuw nsw i32 %88, 144
  %102 = mul nuw nsw i32 %89, 12
  %103 = add nuw nsw i32 %102, %15
  %104 = add nuw nsw i32 %103, %101
  %105 = add nuw nsw i32 %104, %100
  %106 = zext nneg i32 %105 to i64
  %107 = getelementptr inbounds { double, double }, ptr addrspace(1) %10, i64 %106
  %108 = load <2 x double>, ptr addrspace(1) %107, align 16, !invariant.load !4
  %.unpack1087 = extractelement <2 x double> %108, i32 0
  %.unpack1288 = extractelement <2 x double> %108, i32 1
  %109 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 128
  store double %.unpack1087, ptr addrspace(3) %109, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 136
  store double %.unpack1288, ptr addrspace(3) %.repack13, align 8
  %110 = tail call i32 @llvm.umin.i32(i32 %26, i32 510)
  %111 = getelementptr inbounds i8, ptr addrspace(1) %36, i64 96
  %112 = load i32, ptr addrspace(1) %111, align 4, !invariant.load !4
  %113 = icmp slt i32 %112, 0
  %114 = add i32 %112, 21
  %115 = select i1 %113, i32 %114, i32 %112
  %116 = tail call i32 @llvm.smax.i32(i32 %115, i32 0)
  %117 = tail call i32 @llvm.umin.i32(i32 %116, i32 20)
  %118 = mul nuw nsw i32 %117, 73728
  %119 = mul nuw nsw i32 %110, 144
  %120 = zext nneg i32 %118 to i64
  %121 = zext nneg i32 %45 to i64
  %122 = zext nneg i32 %119 to i64
  %123 = zext nneg i32 %15 to i64
  %124 = add i64 %123, %122
  %125 = add i64 %124, %121
  %126 = add i64 %125, %120
  %127 = getelementptr inbounds { double, double }, ptr addrspace(1) %10, i64 %126
  %128 = getelementptr inbounds i8, ptr addrspace(1) %127, i64 2304
  %129 = load <2 x double>, ptr addrspace(1) %128, align 16, !invariant.load !4
  %.unpack1585 = extractelement <2 x double> %129, i32 0
  %.unpack1786 = extractelement <2 x double> %129, i32 1
  %130 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 192
  store double %.unpack1585, ptr addrspace(3) %130, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 200
  store double %.unpack1786, ptr addrspace(3) %.repack18, align 8
  %131 = or disjoint i32 %25, 4
  %132 = udiv i32 %131, 3
  %133 = tail call i32 @llvm.umin.i32(i32 %132, i32 511)
  %134 = mul nuw nsw i32 %132, 24
  %135 = add nuw nsw i32 %24, %134
  %136 = zext nneg i32 %135 to i64
  %137 = getelementptr inbounds i32, ptr addrspace(1) %9, i64 %136
  %138 = load i32, ptr addrspace(1) %137, align 4, !invariant.load !4
  %139 = icmp slt i32 %138, 0
  %140 = add i32 %138, 21
  %141 = select i1 %139, i32 %140, i32 %138
  %142 = tail call i32 @llvm.smax.i32(i32 %141, i32 0)
  %143 = tail call i32 @llvm.umin.i32(i32 %142, i32 20)
  %144 = mul nuw nsw i32 %143, 73728
  %145 = mul nuw nsw i32 %133, 144
  %146 = add nuw nsw i32 %76, %145
  %147 = add nuw nsw i32 %146, %144
  %148 = zext nneg i32 %147 to i64
  %149 = getelementptr inbounds { double, double }, ptr addrspace(1) %10, i64 %148
  %150 = load <2 x double>, ptr addrspace(1) %149, align 16, !invariant.load !4
  %.unpack2083 = extractelement <2 x double> %150, i32 0
  %.unpack2284 = extractelement <2 x double> %150, i32 1
  %151 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 256
  store double %.unpack2083, ptr addrspace(3) %151, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 264
  store double %.unpack2284, ptr addrspace(3) %.repack23, align 8
  %152 = or disjoint i32 %25, 5
  %153 = udiv i32 %152, 3
  %154 = tail call i32 @llvm.umin.i32(i32 %153, i32 511)
  %155 = mul nuw nsw i32 %153, 24
  %156 = add nuw nsw i32 %24, %155
  %157 = zext nneg i32 %156 to i64
  %158 = getelementptr inbounds i32, ptr addrspace(1) %9, i64 %157
  %159 = load i32, ptr addrspace(1) %158, align 4, !invariant.load !4
  %160 = icmp slt i32 %159, 0
  %161 = add i32 %159, 21
  %162 = select i1 %160, i32 %161, i32 %159
  %163 = tail call i32 @llvm.smax.i32(i32 %162, i32 0)
  %164 = tail call i32 @llvm.umin.i32(i32 %163, i32 20)
  %165 = mul nuw nsw i32 %164, 73728
  %166 = mul nuw nsw i32 %154, 144
  %167 = add nuw nsw i32 %103, %166
  %168 = add nuw nsw i32 %167, %165
  %169 = zext nneg i32 %168 to i64
  %170 = getelementptr inbounds { double, double }, ptr addrspace(1) %10, i64 %169
  %171 = load <2 x double>, ptr addrspace(1) %170, align 16, !invariant.load !4
  %.unpack2581 = extractelement <2 x double> %171, i32 0
  %.unpack2782 = extractelement <2 x double> %171, i32 1
  %172 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 320
  store double %.unpack2581, ptr addrspace(3) %172, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 328
  store double %.unpack2782, ptr addrspace(3) %.repack28, align 8
  %173 = getelementptr inbounds i8, ptr addrspace(1) %36, i64 192
  %174 = load i32, ptr addrspace(1) %173, align 4, !invariant.load !4
  %175 = icmp slt i32 %174, 0
  %176 = add i32 %174, 21
  %177 = select i1 %175, i32 %176, i32 %174
  %178 = tail call i32 @llvm.smax.i32(i32 %177, i32 0)
  %179 = tail call i32 @llvm.umin.i32(i32 %178, i32 20)
  %180 = mul nuw nsw i32 %179, 73728
  %181 = zext nneg i32 %180 to i64
  %182 = add i64 %125, %181
  %183 = getelementptr inbounds { double, double }, ptr addrspace(1) %10, i64 %182
  %184 = getelementptr inbounds i8, ptr addrspace(1) %183, i64 4608
  %185 = load <2 x double>, ptr addrspace(1) %184, align 16, !invariant.load !4
  %.unpack3079 = extractelement <2 x double> %185, i32 0
  %.unpack3280 = extractelement <2 x double> %185, i32 1
  %186 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 384
  store double %.unpack3079, ptr addrspace(3) %186, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 392
  store double %.unpack3280, ptr addrspace(3) %.repack33, align 8
  %187 = or disjoint i32 %25, 7
  %188 = udiv i32 %187, 3
  %189 = tail call i32 @llvm.umin.i32(i32 %188, i32 511)
  %190 = mul nuw nsw i32 %188, 24
  %191 = add nuw nsw i32 %24, %190
  %192 = zext nneg i32 %191 to i64
  %193 = getelementptr inbounds i32, ptr addrspace(1) %9, i64 %192
  %194 = load i32, ptr addrspace(1) %193, align 4, !invariant.load !4
  %195 = icmp slt i32 %194, 0
  %196 = add i32 %194, 21
  %197 = select i1 %195, i32 %196, i32 %194
  %198 = tail call i32 @llvm.smax.i32(i32 %197, i32 0)
  %199 = tail call i32 @llvm.umin.i32(i32 %198, i32 20)
  %200 = mul nuw nsw i32 %199, 73728
  %201 = mul nuw nsw i32 %189, 144
  %202 = add nuw nsw i32 %76, %201
  %203 = add nuw nsw i32 %202, %200
  %204 = zext nneg i32 %203 to i64
  %205 = getelementptr inbounds { double, double }, ptr addrspace(1) %10, i64 %204
  %206 = load <2 x double>, ptr addrspace(1) %205, align 16, !invariant.load !4
  %.unpack3577 = extractelement <2 x double> %206, i32 0
  %.unpack3778 = extractelement <2 x double> %206, i32 1
  %207 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 448
  store double %.unpack3577, ptr addrspace(3) %207, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %55, i64 456
  store double %.unpack3778, ptr addrspace(3) %.repack38, align 8
  br label %208

208:                                              ; preds = %._crit_edge, %17
  %.pre-phi = phi i32 [ %.pre, %._crit_edge ], [ %29, %17 ]
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %209 = shl nuw nsw i32 %14, 5
  %210 = or disjoint i32 %209, %15
  %211 = udiv i32 %210, 12
  %212 = load i32, ptr addrspace(1) %7, align 256, !invariant.load !4
  %213 = tail call i32 @llvm.umin.i32(i32 %212, i32 3)
  %214 = zext nneg i32 %213 to i64
  %215 = getelementptr inbounds i32, ptr addrspace(1) %8, i64 %214
  %216 = load i32, ptr addrspace(1) %215, align 4, !invariant.load !4
  %.not40 = icmp eq i32 %216, 0
  %217 = select i1 %.not40, i32 0, i32 12
  %218 = mul nuw nsw i32 %.pre-phi, 33
  %219 = add nuw nsw i32 %218, %15
  %220 = zext nneg i32 %219 to i64
  %221 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_0, i64 %220
  %.unpack41 = load double, ptr addrspace(3) %221, align 8
  %.elt42 = getelementptr inbounds i8, ptr addrspace(3) %221, i64 8
  %.unpack43 = load double, ptr addrspace(3) %.elt42, align 8
  %222 = mul nuw nsw i32 %211, 24
  %223 = add nuw nsw i32 %217, %222
  %224 = or disjoint i32 %223, %.pre-phi
  %225 = zext nneg i32 %224 to i64
  %226 = getelementptr inbounds i32, ptr addrspace(1) %9, i64 %225
  %227 = load i32, ptr addrspace(1) %226, align 4, !invariant.load !4
  %228 = icmp slt i32 %227, 0
  %229 = add i32 %227, 21
  %230 = select i1 %228, i32 %229, i32 %227
  %231 = icmp ult i32 %230, 21
  %232 = getelementptr inbounds { double, double }, ptr addrspace(1) %11, i64 %225
  %233 = load <2 x double>, ptr addrspace(1) %232, align 16, !invariant.load !4
  %.unpack4475 = extractelement <2 x double> %233, i32 0
  %.unpack4676 = extractelement <2 x double> %233, i32 1
  %234 = select i1 %231, double %.unpack41, double 0x7FF8000000000000
  %235 = select i1 %231, double %.unpack43, double 0.000000e+00
  %236 = fmul double %.unpack4475, %234
  %237 = fmul double %.unpack4676, %235
  %238 = fsub double %236, %237
  %239 = fmul double %.unpack4676, %234
  %240 = fmul double %.unpack4475, %235
  %241 = fadd double %239, %240
  %242 = mul nuw nsw i32 %.pre-phi, 6144
  %243 = add nuw nsw i32 %242, %209
  %244 = or disjoint i32 %243, %15
  %245 = zext nneg i32 %244 to i64
  %246 = getelementptr inbounds { double, double }, ptr addrspace(1) %12, i64 %245
  %247 = insertelement <2 x double> poison, double %238, i32 0
  %248 = insertelement <2 x double> %247, double %241, i32 1
  store <2 x double> %248, ptr addrspace(1) %246, align 16
  %249 = getelementptr inbounds i8, ptr addrspace(3) %221, i64 2112
  %.unpack49 = load double, ptr addrspace(3) %249, align 8
  %.elt50 = getelementptr inbounds i8, ptr addrspace(3) %221, i64 2120
  %.unpack51 = load double, ptr addrspace(3) %.elt50, align 8
  %250 = zext nneg i32 %.pre-phi to i64
  %251 = zext nneg i32 %223 to i64
  %252 = add i64 %251, %250
  %253 = getelementptr inbounds i32, ptr addrspace(1) %9, i64 %252
  %254 = getelementptr inbounds i8, ptr addrspace(1) %253, i64 16
  %255 = load i32, ptr addrspace(1) %254, align 4, !invariant.load !4
  %256 = icmp slt i32 %255, 0
  %257 = add i32 %255, 21
  %258 = select i1 %256, i32 %257, i32 %255
  %259 = icmp ult i32 %258, 21
  %260 = getelementptr inbounds { double, double }, ptr addrspace(1) %11, i64 %252
  %261 = getelementptr inbounds i8, ptr addrspace(1) %260, i64 64
  %262 = load <2 x double>, ptr addrspace(1) %261, align 16, !invariant.load !4
  %.unpack5271 = extractelement <2 x double> %262, i32 0
  %.unpack5472 = extractelement <2 x double> %262, i32 1
  %263 = select i1 %259, double %.unpack49, double 0x7FF8000000000000
  %264 = select i1 %259, double %.unpack51, double 0.000000e+00
  %265 = fmul double %.unpack5271, %263
  %266 = fmul double %.unpack5472, %264
  %267 = fsub double %265, %266
  %268 = fmul double %.unpack5472, %263
  %269 = fmul double %.unpack5271, %264
  %270 = fadd double %268, %269
  %271 = getelementptr inbounds i8, ptr addrspace(1) %246, i64 393216
  %272 = insertelement <2 x double> poison, double %267, i32 0
  %273 = insertelement <2 x double> %272, double %270, i32 1
  store <2 x double> %273, ptr addrspace(1) %271, align 16
  %274 = getelementptr inbounds i8, ptr addrspace(3) %221, i64 4224
  %.unpack57 = load double, ptr addrspace(3) %274, align 8
  %.elt58 = getelementptr inbounds i8, ptr addrspace(3) %221, i64 4232
  %.unpack59 = load double, ptr addrspace(3) %.elt58, align 8
  %275 = getelementptr inbounds i8, ptr addrspace(1) %253, i64 32
  %276 = load i32, ptr addrspace(1) %275, align 4, !invariant.load !4
  %277 = icmp slt i32 %276, 0
  %278 = add i32 %276, 21
  %279 = select i1 %277, i32 %278, i32 %276
  %280 = icmp ult i32 %279, 21
  %281 = getelementptr inbounds i8, ptr addrspace(1) %260, i64 128
  %282 = load <2 x double>, ptr addrspace(1) %281, align 16, !invariant.load !4
  %.unpack6073 = extractelement <2 x double> %282, i32 0
  %.unpack6274 = extractelement <2 x double> %282, i32 1
  %283 = select i1 %280, double %.unpack57, double 0x7FF8000000000000
  %284 = select i1 %280, double %.unpack59, double 0.000000e+00
  %285 = fmul double %.unpack6073, %283
  %286 = fmul double %.unpack6274, %284
  %287 = fsub double %285, %286
  %288 = fmul double %.unpack6274, %283
  %289 = fmul double %.unpack6073, %284
  %290 = fadd double %288, %289
  %291 = getelementptr inbounds i8, ptr addrspace(1) %246, i64 786432
  %292 = insertelement <2 x double> poison, double %287, i32 0
  %293 = insertelement <2 x double> %292, double %290, i32 1
  store <2 x double> %293, ptr addrspace(1) %291, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #2

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion_2(ptr noalias readonly align 16 captures(none) dereferenceable(24772608) %0, ptr noalias readonly align 16 captures(none) dereferenceable(196608) %1, ptr noalias readonly align 16 captures(none) dereferenceable(49152) %2, ptr noalias readonly align 256 captures(none) dereferenceable(16) %3, ptr noalias readonly align 256 captures(none) dereferenceable(4) %4, ptr noalias readonly align 256 captures(none) dereferenceable(16) %5, ptr noalias writeonly align 256 captures(none) dereferenceable(1179648) %6, ptr noalias writeonly align 256 captures(none) dereferenceable(1179648) %7) local_unnamed_addr #0 {
  %9 = addrspacecast ptr %4 to ptr addrspace(1)
  %10 = addrspacecast ptr %3 to ptr addrspace(1)
  %11 = addrspacecast ptr %5 to ptr addrspace(1)
  %12 = addrspacecast ptr %2 to ptr addrspace(1)
  %13 = addrspacecast ptr %0 to ptr addrspace(1)
  %14 = addrspacecast ptr %1 to ptr addrspace(1)
  %15 = addrspacecast ptr %6 to ptr addrspace(1)
  %16 = addrspacecast ptr %7 to ptr addrspace(1)
  %17 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %18 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %19 = and i32 %17, 31
  %20 = icmp samesign ult i32 %19, 12
  br i1 %20, label %21, label %._crit_edge

._crit_edge:                                      ; preds = %8
  %.pre = lshr i32 %17, 5
  br label %334

21:                                               ; preds = %8
  %22 = load i32, ptr addrspace(1) %9, align 256, !invariant.load !4
  %23 = tail call i32 @llvm.umin.i32(i32 %22, i32 3)
  %24 = zext nneg i32 %23 to i64
  %25 = getelementptr inbounds i32, ptr addrspace(1) %10, i64 %24
  %26 = load i32, ptr addrspace(1) %25, align 4, !invariant.load !4
  %.not = icmp eq i32 %26, 0
  %27 = select i1 %.not, i32 0, i32 12
  %28 = getelementptr inbounds i32, ptr addrspace(1) %11, i64 %24
  %29 = load i32, ptr addrspace(1) %28, align 4, !invariant.load !4
  %.not1 = icmp eq i32 %29, 0
  %30 = select i1 %.not1, i32 0, i32 12
  %31 = add nuw nsw i32 %30, %19
  %32 = shl nuw nsw i32 %18, 3
  %33 = udiv i32 %32, 3
  %34 = mul i32 %33, 3
  %.decomposed = sub i32 %32, %34
  %35 = shl nuw nsw i32 %.decomposed, 2
  %36 = lshr i32 %17, 5
  %37 = or disjoint i32 %35, %36
  %38 = tail call i32 @llvm.umin.i32(i32 %33, i32 511)
  %39 = tail call i32 @llvm.umin.i32(i32 %37, i32 11)
  %40 = mul nuw nsw i32 %33, 24
  %41 = add nuw nsw i32 %37, %40
  %42 = add nuw nsw i32 %41, %27
  %43 = zext nneg i32 %42 to i64
  %44 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %43
  %45 = load i32, ptr addrspace(1) %44, align 4, !invariant.load !4
  %46 = icmp slt i32 %45, 0
  %47 = add i32 %45, 21
  %48 = select i1 %46, i32 %47, i32 %45
  %49 = tail call i32 @llvm.smax.i32(i32 %48, i32 0)
  %50 = tail call i32 @llvm.umin.i32(i32 %49, i32 20)
  %51 = mul nuw nsw i32 %50, 73728
  %52 = mul nuw nsw i32 %38, 144
  %53 = mul nuw nsw i32 %39, 12
  %54 = add nuw nsw i32 %53, %19
  %55 = add nuw nsw i32 %54, %52
  %56 = add nuw nsw i32 %55, %51
  %57 = zext nneg i32 %56 to i64
  %58 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %57
  %59 = load <2 x double>, ptr addrspace(1) %58, align 16, !invariant.load !4
  %.unpack185 = extractelement <2 x double> %59, i32 0
  %.unpack3186 = extractelement <2 x double> %59, i32 1
  %60 = mul nuw nsw i32 %19, 33
  %61 = add nuw nsw i32 %60, %36
  %62 = zext nneg i32 %61 to i64
  %63 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_1, i64 %62
  store double %.unpack185, ptr addrspace(3) %63, align 8
  %.repack4 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 8
  store double %.unpack3186, ptr addrspace(3) %.repack4, align 8
  %64 = add nuw nsw i32 %31, %40
  %65 = zext nneg i32 %64 to i64
  %66 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %65
  %67 = load i32, ptr addrspace(1) %66, align 4, !invariant.load !4
  %68 = icmp slt i32 %67, 0
  %69 = add i32 %67, 21
  %70 = select i1 %68, i32 %69, i32 %67
  %71 = tail call i32 @llvm.smax.i32(i32 %70, i32 0)
  %72 = tail call i32 @llvm.umin.i32(i32 %71, i32 20)
  %73 = mul nuw nsw i32 %72, 73728
  %74 = add nuw nsw i32 %55, %73
  %75 = zext nneg i32 %74 to i64
  %76 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %75
  %77 = load <2 x double>, ptr addrspace(1) %76, align 16, !invariant.load !4
  %.unpack6183 = extractelement <2 x double> %77, i32 0
  %.unpack8184 = extractelement <2 x double> %77, i32 1
  %78 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_01, i64 %62
  store double %.unpack6183, ptr addrspace(3) %78, align 8
  %.repack9 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 8
  store double %.unpack8184, ptr addrspace(3) %.repack9, align 8
  %79 = or disjoint i32 %32, 1
  %80 = udiv i32 %79, 3
  %81 = mul i32 %80, 3
  %.decomposed124 = sub i32 %79, %81
  %82 = shl nuw nsw i32 %.decomposed124, 2
  %83 = or disjoint i32 %82, %36
  %84 = tail call i32 @llvm.umin.i32(i32 %80, i32 511)
  %85 = tail call i32 @llvm.umin.i32(i32 %83, i32 11)
  %86 = add nuw nsw i32 %27, %83
  %87 = mul nuw nsw i32 %80, 24
  %88 = add nuw nsw i32 %86, %87
  %89 = zext nneg i32 %88 to i64
  %90 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %89
  %91 = load i32, ptr addrspace(1) %90, align 4, !invariant.load !4
  %92 = icmp slt i32 %91, 0
  %93 = add i32 %91, 21
  %94 = select i1 %92, i32 %93, i32 %91
  %95 = tail call i32 @llvm.smax.i32(i32 %94, i32 0)
  %96 = tail call i32 @llvm.umin.i32(i32 %95, i32 20)
  %97 = mul nuw nsw i32 %96, 73728
  %98 = mul nuw nsw i32 %84, 144
  %99 = mul nuw nsw i32 %85, 12
  %100 = add nuw nsw i32 %99, %19
  %101 = add nuw nsw i32 %100, %98
  %102 = add nuw nsw i32 %101, %97
  %103 = zext nneg i32 %102 to i64
  %104 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %103
  %105 = load <2 x double>, ptr addrspace(1) %104, align 16, !invariant.load !4
  %.unpack11181 = extractelement <2 x double> %105, i32 0
  %.unpack13182 = extractelement <2 x double> %105, i32 1
  %106 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 64
  store double %.unpack11181, ptr addrspace(3) %106, align 8
  %.repack14127 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 72
  store double %.unpack13182, ptr addrspace(3) %.repack14127, align 8
  %107 = add nuw nsw i32 %31, %87
  %108 = zext nneg i32 %107 to i64
  %109 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %108
  %110 = load i32, ptr addrspace(1) %109, align 4, !invariant.load !4
  %111 = icmp slt i32 %110, 0
  %112 = add i32 %110, 21
  %113 = select i1 %111, i32 %112, i32 %110
  %114 = tail call i32 @llvm.smax.i32(i32 %113, i32 0)
  %115 = tail call i32 @llvm.umin.i32(i32 %114, i32 20)
  %116 = mul nuw nsw i32 %115, 73728
  %117 = add nuw nsw i32 %101, %116
  %118 = zext nneg i32 %117 to i64
  %119 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %118
  %120 = load <2 x double>, ptr addrspace(1) %119, align 16, !invariant.load !4
  %.unpack16179 = extractelement <2 x double> %120, i32 0
  %.unpack18180 = extractelement <2 x double> %120, i32 1
  %121 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 64
  store double %.unpack16179, ptr addrspace(3) %121, align 8
  %.repack19128 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 72
  store double %.unpack18180, ptr addrspace(3) %.repack19128, align 8
  %122 = or disjoint i32 %32, 2
  %123 = udiv i32 %122, 3
  %124 = mul i32 %123, 3
  %.decomposed125 = sub i32 %122, %124
  %125 = shl nuw nsw i32 %.decomposed125, 2
  %126 = or disjoint i32 %125, %36
  %127 = tail call i32 @llvm.umin.i32(i32 %123, i32 511)
  %128 = tail call i32 @llvm.umin.i32(i32 %126, i32 11)
  %129 = add nuw nsw i32 %27, %126
  %130 = mul nuw nsw i32 %123, 24
  %131 = add nuw nsw i32 %129, %130
  %132 = zext nneg i32 %131 to i64
  %133 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %132
  %134 = load i32, ptr addrspace(1) %133, align 4, !invariant.load !4
  %135 = icmp slt i32 %134, 0
  %136 = add i32 %134, 21
  %137 = select i1 %135, i32 %136, i32 %134
  %138 = tail call i32 @llvm.smax.i32(i32 %137, i32 0)
  %139 = tail call i32 @llvm.umin.i32(i32 %138, i32 20)
  %140 = mul nuw nsw i32 %139, 73728
  %141 = mul nuw nsw i32 %127, 144
  %142 = mul nuw nsw i32 %128, 12
  %143 = add nuw nsw i32 %142, %19
  %144 = add nuw nsw i32 %143, %141
  %145 = add nuw nsw i32 %144, %140
  %146 = zext nneg i32 %145 to i64
  %147 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %146
  %148 = load <2 x double>, ptr addrspace(1) %147, align 16, !invariant.load !4
  %.unpack21177 = extractelement <2 x double> %148, i32 0
  %.unpack23178 = extractelement <2 x double> %148, i32 1
  %149 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 128
  store double %.unpack21177, ptr addrspace(3) %149, align 8
  %.repack24129 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 136
  store double %.unpack23178, ptr addrspace(3) %.repack24129, align 8
  %150 = add nuw nsw i32 %31, %130
  %151 = zext nneg i32 %150 to i64
  %152 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %151
  %153 = load i32, ptr addrspace(1) %152, align 4, !invariant.load !4
  %154 = icmp slt i32 %153, 0
  %155 = add i32 %153, 21
  %156 = select i1 %154, i32 %155, i32 %153
  %157 = tail call i32 @llvm.smax.i32(i32 %156, i32 0)
  %158 = tail call i32 @llvm.umin.i32(i32 %157, i32 20)
  %159 = mul nuw nsw i32 %158, 73728
  %160 = add nuw nsw i32 %144, %159
  %161 = zext nneg i32 %160 to i64
  %162 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %161
  %163 = load <2 x double>, ptr addrspace(1) %162, align 16, !invariant.load !4
  %.unpack26175 = extractelement <2 x double> %163, i32 0
  %.unpack28176 = extractelement <2 x double> %163, i32 1
  %164 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 128
  store double %.unpack26175, ptr addrspace(3) %164, align 8
  %.repack29130 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 136
  store double %.unpack28176, ptr addrspace(3) %.repack29130, align 8
  %165 = tail call i32 @llvm.umin.i32(i32 %33, i32 510)
  %166 = getelementptr inbounds i8, ptr addrspace(1) %44, i64 96
  %167 = load i32, ptr addrspace(1) %166, align 4, !invariant.load !4
  %168 = icmp slt i32 %167, 0
  %169 = add i32 %167, 21
  %170 = select i1 %168, i32 %169, i32 %167
  %171 = tail call i32 @llvm.smax.i32(i32 %170, i32 0)
  %172 = tail call i32 @llvm.umin.i32(i32 %171, i32 20)
  %173 = mul nuw nsw i32 %172, 73728
  %174 = mul nuw nsw i32 %165, 144
  %175 = zext nneg i32 %173 to i64
  %176 = zext nneg i32 %54 to i64
  %177 = zext nneg i32 %174 to i64
  %178 = add i64 %176, %177
  %179 = add i64 %178, %175
  %180 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %179
  %181 = getelementptr inbounds i8, ptr addrspace(1) %180, i64 2304
  %182 = load <2 x double>, ptr addrspace(1) %181, align 16, !invariant.load !4
  %.unpack31173 = extractelement <2 x double> %182, i32 0
  %.unpack33174 = extractelement <2 x double> %182, i32 1
  %183 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 192
  store double %.unpack31173, ptr addrspace(3) %183, align 8
  %.repack34132 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 200
  store double %.unpack33174, ptr addrspace(3) %.repack34132, align 8
  %184 = getelementptr inbounds i8, ptr addrspace(1) %66, i64 96
  %185 = load i32, ptr addrspace(1) %184, align 4, !invariant.load !4
  %186 = icmp slt i32 %185, 0
  %187 = add i32 %185, 21
  %188 = select i1 %186, i32 %187, i32 %185
  %189 = tail call i32 @llvm.smax.i32(i32 %188, i32 0)
  %190 = tail call i32 @llvm.umin.i32(i32 %189, i32 20)
  %191 = mul nuw nsw i32 %190, 73728
  %192 = zext nneg i32 %191 to i64
  %193 = add i64 %178, %192
  %194 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %193
  %195 = getelementptr inbounds i8, ptr addrspace(1) %194, i64 2304
  %196 = load <2 x double>, ptr addrspace(1) %195, align 16, !invariant.load !4
  %.unpack36171 = extractelement <2 x double> %196, i32 0
  %.unpack38172 = extractelement <2 x double> %196, i32 1
  %197 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 192
  store double %.unpack36171, ptr addrspace(3) %197, align 8
  %.repack39134 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 200
  store double %.unpack38172, ptr addrspace(3) %.repack39134, align 8
  %198 = or disjoint i32 %32, 4
  %199 = udiv i32 %198, 3
  %200 = tail call i32 @llvm.umin.i32(i32 %199, i32 511)
  %201 = mul nuw nsw i32 %199, 24
  %202 = add nuw nsw i32 %86, %201
  %203 = zext nneg i32 %202 to i64
  %204 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %203
  %205 = load i32, ptr addrspace(1) %204, align 4, !invariant.load !4
  %206 = icmp slt i32 %205, 0
  %207 = add i32 %205, 21
  %208 = select i1 %206, i32 %207, i32 %205
  %209 = tail call i32 @llvm.smax.i32(i32 %208, i32 0)
  %210 = tail call i32 @llvm.umin.i32(i32 %209, i32 20)
  %211 = mul nuw nsw i32 %210, 73728
  %212 = mul nuw nsw i32 %200, 144
  %213 = add nuw nsw i32 %100, %212
  %214 = add nuw nsw i32 %213, %211
  %215 = zext nneg i32 %214 to i64
  %216 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %215
  %217 = load <2 x double>, ptr addrspace(1) %216, align 16, !invariant.load !4
  %.unpack41169 = extractelement <2 x double> %217, i32 0
  %.unpack43170 = extractelement <2 x double> %217, i32 1
  %218 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 256
  store double %.unpack41169, ptr addrspace(3) %218, align 8
  %.repack44135 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 264
  store double %.unpack43170, ptr addrspace(3) %.repack44135, align 8
  %219 = add nuw nsw i32 %31, %201
  %220 = zext nneg i32 %219 to i64
  %221 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %220
  %222 = load i32, ptr addrspace(1) %221, align 4, !invariant.load !4
  %223 = icmp slt i32 %222, 0
  %224 = add i32 %222, 21
  %225 = select i1 %223, i32 %224, i32 %222
  %226 = tail call i32 @llvm.smax.i32(i32 %225, i32 0)
  %227 = tail call i32 @llvm.umin.i32(i32 %226, i32 20)
  %228 = mul nuw nsw i32 %227, 73728
  %229 = add nuw nsw i32 %213, %228
  %230 = zext nneg i32 %229 to i64
  %231 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %230
  %232 = load <2 x double>, ptr addrspace(1) %231, align 16, !invariant.load !4
  %.unpack46167 = extractelement <2 x double> %232, i32 0
  %.unpack48168 = extractelement <2 x double> %232, i32 1
  %233 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 256
  store double %.unpack46167, ptr addrspace(3) %233, align 8
  %.repack49136 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 264
  store double %.unpack48168, ptr addrspace(3) %.repack49136, align 8
  %234 = or disjoint i32 %32, 5
  %235 = udiv i32 %234, 3
  %236 = tail call i32 @llvm.umin.i32(i32 %235, i32 511)
  %237 = mul nuw nsw i32 %235, 24
  %238 = add nuw nsw i32 %129, %237
  %239 = zext nneg i32 %238 to i64
  %240 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %239
  %241 = load i32, ptr addrspace(1) %240, align 4, !invariant.load !4
  %242 = icmp slt i32 %241, 0
  %243 = add i32 %241, 21
  %244 = select i1 %242, i32 %243, i32 %241
  %245 = tail call i32 @llvm.smax.i32(i32 %244, i32 0)
  %246 = tail call i32 @llvm.umin.i32(i32 %245, i32 20)
  %247 = mul nuw nsw i32 %246, 73728
  %248 = mul nuw nsw i32 %236, 144
  %249 = add nuw nsw i32 %143, %248
  %250 = add nuw nsw i32 %249, %247
  %251 = zext nneg i32 %250 to i64
  %252 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %251
  %253 = load <2 x double>, ptr addrspace(1) %252, align 16, !invariant.load !4
  %.unpack51165 = extractelement <2 x double> %253, i32 0
  %.unpack53166 = extractelement <2 x double> %253, i32 1
  %254 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 320
  store double %.unpack51165, ptr addrspace(3) %254, align 8
  %.repack54137 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 328
  store double %.unpack53166, ptr addrspace(3) %.repack54137, align 8
  %255 = add nuw nsw i32 %31, %237
  %256 = zext nneg i32 %255 to i64
  %257 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %256
  %258 = load i32, ptr addrspace(1) %257, align 4, !invariant.load !4
  %259 = icmp slt i32 %258, 0
  %260 = add i32 %258, 21
  %261 = select i1 %259, i32 %260, i32 %258
  %262 = tail call i32 @llvm.smax.i32(i32 %261, i32 0)
  %263 = tail call i32 @llvm.umin.i32(i32 %262, i32 20)
  %264 = mul nuw nsw i32 %263, 73728
  %265 = add nuw nsw i32 %249, %264
  %266 = zext nneg i32 %265 to i64
  %267 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %266
  %268 = load <2 x double>, ptr addrspace(1) %267, align 16, !invariant.load !4
  %.unpack56163 = extractelement <2 x double> %268, i32 0
  %.unpack58164 = extractelement <2 x double> %268, i32 1
  %269 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 320
  store double %.unpack56163, ptr addrspace(3) %269, align 8
  %.repack59138 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 328
  store double %.unpack58164, ptr addrspace(3) %.repack59138, align 8
  %270 = getelementptr inbounds i8, ptr addrspace(1) %44, i64 192
  %271 = load i32, ptr addrspace(1) %270, align 4, !invariant.load !4
  %272 = icmp slt i32 %271, 0
  %273 = add i32 %271, 21
  %274 = select i1 %272, i32 %273, i32 %271
  %275 = tail call i32 @llvm.smax.i32(i32 %274, i32 0)
  %276 = tail call i32 @llvm.umin.i32(i32 %275, i32 20)
  %277 = mul nuw nsw i32 %276, 73728
  %278 = zext nneg i32 %277 to i64
  %279 = add i64 %178, %278
  %280 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %279
  %281 = getelementptr inbounds i8, ptr addrspace(1) %280, i64 4608
  %282 = load <2 x double>, ptr addrspace(1) %281, align 16, !invariant.load !4
  %.unpack61161 = extractelement <2 x double> %282, i32 0
  %.unpack63162 = extractelement <2 x double> %282, i32 1
  %283 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 384
  store double %.unpack61161, ptr addrspace(3) %283, align 8
  %.repack64140 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 392
  store double %.unpack63162, ptr addrspace(3) %.repack64140, align 8
  %284 = getelementptr inbounds i8, ptr addrspace(1) %66, i64 192
  %285 = load i32, ptr addrspace(1) %284, align 4, !invariant.load !4
  %286 = icmp slt i32 %285, 0
  %287 = add i32 %285, 21
  %288 = select i1 %286, i32 %287, i32 %285
  %289 = tail call i32 @llvm.smax.i32(i32 %288, i32 0)
  %290 = tail call i32 @llvm.umin.i32(i32 %289, i32 20)
  %291 = mul nuw nsw i32 %290, 73728
  %292 = zext nneg i32 %291 to i64
  %293 = add i64 %178, %292
  %294 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %293
  %295 = getelementptr inbounds i8, ptr addrspace(1) %294, i64 4608
  %296 = load <2 x double>, ptr addrspace(1) %295, align 16, !invariant.load !4
  %.unpack66159 = extractelement <2 x double> %296, i32 0
  %.unpack68160 = extractelement <2 x double> %296, i32 1
  %297 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 384
  store double %.unpack66159, ptr addrspace(3) %297, align 8
  %.repack69142 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 392
  store double %.unpack68160, ptr addrspace(3) %.repack69142, align 8
  %298 = or disjoint i32 %32, 7
  %299 = udiv i32 %298, 3
  %300 = tail call i32 @llvm.umin.i32(i32 %299, i32 511)
  %301 = mul nuw nsw i32 %299, 24
  %302 = add nuw nsw i32 %86, %301
  %303 = zext nneg i32 %302 to i64
  %304 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %303
  %305 = load i32, ptr addrspace(1) %304, align 4, !invariant.load !4
  %306 = icmp slt i32 %305, 0
  %307 = add i32 %305, 21
  %308 = select i1 %306, i32 %307, i32 %305
  %309 = tail call i32 @llvm.smax.i32(i32 %308, i32 0)
  %310 = tail call i32 @llvm.umin.i32(i32 %309, i32 20)
  %311 = mul nuw nsw i32 %310, 73728
  %312 = mul nuw nsw i32 %300, 144
  %313 = add nuw nsw i32 %100, %312
  %314 = add nuw nsw i32 %313, %311
  %315 = zext nneg i32 %314 to i64
  %316 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %315
  %317 = load <2 x double>, ptr addrspace(1) %316, align 16, !invariant.load !4
  %.unpack71157 = extractelement <2 x double> %317, i32 0
  %.unpack73158 = extractelement <2 x double> %317, i32 1
  %318 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 448
  store double %.unpack71157, ptr addrspace(3) %318, align 8
  %.repack74143 = getelementptr inbounds i8, ptr addrspace(3) %63, i64 456
  store double %.unpack73158, ptr addrspace(3) %.repack74143, align 8
  %319 = add nuw nsw i32 %31, %301
  %320 = zext nneg i32 %319 to i64
  %321 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %320
  %322 = load i32, ptr addrspace(1) %321, align 4, !invariant.load !4
  %323 = icmp slt i32 %322, 0
  %324 = add i32 %322, 21
  %325 = select i1 %323, i32 %324, i32 %322
  %326 = tail call i32 @llvm.smax.i32(i32 %325, i32 0)
  %327 = tail call i32 @llvm.umin.i32(i32 %326, i32 20)
  %328 = mul nuw nsw i32 %327, 73728
  %329 = add nuw nsw i32 %313, %328
  %330 = zext nneg i32 %329 to i64
  %331 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %330
  %332 = load <2 x double>, ptr addrspace(1) %331, align 16, !invariant.load !4
  %.unpack76155 = extractelement <2 x double> %332, i32 0
  %.unpack78156 = extractelement <2 x double> %332, i32 1
  %333 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 448
  store double %.unpack76155, ptr addrspace(3) %333, align 8
  %.repack79144 = getelementptr inbounds i8, ptr addrspace(3) %78, i64 456
  store double %.unpack78156, ptr addrspace(3) %.repack79144, align 8
  br label %334

334:                                              ; preds = %._crit_edge, %21
  %.pre-phi = phi i32 [ %.pre, %._crit_edge ], [ %36, %21 ]
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %335 = shl nuw nsw i32 %18, 5
  %336 = or disjoint i32 %335, %19
  %337 = udiv i32 %336, 12
  %338 = mul i32 %337, 12
  %.decomposed126 = sub i32 %336, %338
  %339 = load i32, ptr addrspace(1) %9, align 256, !invariant.load !4
  %340 = tail call i32 @llvm.umin.i32(i32 %339, i32 3)
  %341 = zext nneg i32 %340 to i64
  %342 = getelementptr inbounds i32, ptr addrspace(1) %10, i64 %341
  %343 = load i32, ptr addrspace(1) %342, align 4, !invariant.load !4
  %.not81 = icmp eq i32 %343, 0
  %344 = select i1 %.not81, i32 0, i32 12
  %345 = mul nuw nsw i32 %337, 24
  %346 = add nuw nsw i32 %345, %.decomposed126
  %347 = add nuw nsw i32 %346, %344
  %348 = zext nneg i32 %347 to i64
  %349 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %348
  %350 = load i32, ptr addrspace(1) %349, align 4, !invariant.load !4
  %351 = icmp slt i32 %350, 0
  %352 = add i32 %350, 21
  %353 = select i1 %351, i32 %352, i32 %350
  %354 = icmp ult i32 %353, 21
  %355 = getelementptr inbounds { double, double }, ptr addrspace(1) %14, i64 %348
  %356 = load <2 x double>, ptr addrspace(1) %355, align 16, !invariant.load !4
  %.unpack82153 = extractelement <2 x double> %356, i32 0
  %.unpack84154 = extractelement <2 x double> %356, i32 1
  %357 = getelementptr inbounds i32, ptr addrspace(1) %11, i64 %341
  %358 = load i32, ptr addrspace(1) %357, align 4, !invariant.load !4
  %.not85 = icmp eq i32 %358, 0
  %359 = select i1 %.not85, i32 0, i32 12
  %360 = mul nuw nsw i32 %.pre-phi, 33
  %361 = add nuw nsw i32 %360, %19
  %362 = zext nneg i32 %361 to i64
  %363 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_1, i64 %362
  %.unpack86 = load double, ptr addrspace(3) %363, align 8
  %.elt87 = getelementptr inbounds i8, ptr addrspace(3) %363, i64 8
  %.unpack88 = load double, ptr addrspace(3) %.elt87, align 8
  %364 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_01, i64 %362
  %.unpack89 = load double, ptr addrspace(3) %364, align 8
  %.elt90 = getelementptr inbounds i8, ptr addrspace(3) %364, i64 8
  %.unpack91 = load double, ptr addrspace(3) %.elt90, align 8
  %365 = select i1 %354, double %.unpack86, double 0x7FF8000000000000
  %366 = select i1 %354, double %.unpack88, double 0.000000e+00
  %367 = fmul double %.unpack82153, %365
  %368 = fmul double %.unpack84154, %366
  %369 = fsub double %367, %368
  %370 = fmul double %.unpack84154, %365
  %371 = fmul double %.unpack82153, %366
  %372 = fadd double %370, %371
  %373 = add nuw nsw i32 %359, %345
  %374 = or disjoint i32 %373, %.pre-phi
  %375 = zext nneg i32 %374 to i64
  %376 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %375
  %377 = load i32, ptr addrspace(1) %376, align 4, !invariant.load !4
  %378 = icmp slt i32 %377, 0
  %379 = add i32 %377, 21
  %380 = select i1 %378, i32 %379, i32 %377
  %381 = icmp ult i32 %380, 21
  %382 = mul nuw nsw i32 %.pre-phi, 6144
  %383 = add nuw nsw i32 %382, %335
  %384 = or disjoint i32 %383, %19
  %385 = zext nneg i32 %384 to i64
  %386 = getelementptr inbounds { double, double }, ptr addrspace(1) %15, i64 %385
  %387 = insertelement <2 x double> poison, double %369, i32 0
  %388 = insertelement <2 x double> %387, double %372, i32 1
  store <2 x double> %388, ptr addrspace(1) %386, align 16
  %389 = getelementptr inbounds { double, double }, ptr addrspace(1) %16, i64 %385
  %.elt = select i1 %381, double %.unpack89, double 0x7FF8000000000000
  %.elt95 = select i1 %381, double %.unpack91, double 0.000000e+00
  %390 = insertelement <2 x double> poison, double %.elt, i32 0
  %391 = insertelement <2 x double> %390, double %.elt95, i32 1
  store <2 x double> %391, ptr addrspace(1) %389, align 16
  %392 = getelementptr inbounds i8, ptr addrspace(3) %363, i64 2112
  %.unpack97 = load double, ptr addrspace(3) %392, align 8
  %.elt98145 = getelementptr inbounds i8, ptr addrspace(3) %363, i64 2120
  %.unpack99 = load double, ptr addrspace(3) %.elt98145, align 8
  %393 = getelementptr inbounds i8, ptr addrspace(3) %364, i64 2112
  %.unpack101 = load double, ptr addrspace(3) %393, align 8
  %.elt102146 = getelementptr inbounds i8, ptr addrspace(3) %364, i64 2120
  %.unpack103 = load double, ptr addrspace(3) %.elt102146, align 8
  %394 = select i1 %354, double %.unpack97, double 0x7FF8000000000000
  %395 = select i1 %354, double %.unpack99, double 0.000000e+00
  %396 = fmul double %.unpack82153, %394
  %397 = fmul double %.unpack84154, %395
  %398 = fsub double %396, %397
  %399 = fmul double %.unpack84154, %394
  %400 = fmul double %.unpack82153, %395
  %401 = fadd double %399, %400
  %402 = zext nneg i32 %.pre-phi to i64
  %403 = zext nneg i32 %373 to i64
  %404 = add i64 %403, %402
  %405 = getelementptr inbounds i32, ptr addrspace(1) %12, i64 %404
  %406 = getelementptr inbounds i8, ptr addrspace(1) %405, i64 16
  %407 = load i32, ptr addrspace(1) %406, align 4, !invariant.load !4
  %408 = icmp slt i32 %407, 0
  %409 = add i32 %407, 21
  %410 = select i1 %408, i32 %409, i32 %407
  %411 = icmp ult i32 %410, 21
  %412 = getelementptr inbounds i8, ptr addrspace(1) %386, i64 393216
  %413 = insertelement <2 x double> poison, double %398, i32 0
  %414 = insertelement <2 x double> %413, double %401, i32 1
  store <2 x double> %414, ptr addrspace(1) %412, align 16
  %415 = getelementptr inbounds i8, ptr addrspace(1) %389, i64 393216
  %.elt107 = select i1 %411, double %.unpack101, double 0x7FF8000000000000
  %.elt109 = select i1 %411, double %.unpack103, double 0.000000e+00
  %416 = insertelement <2 x double> poison, double %.elt107, i32 0
  %417 = insertelement <2 x double> %416, double %.elt109, i32 1
  store <2 x double> %417, ptr addrspace(1) %415, align 16
  %418 = getelementptr inbounds i8, ptr addrspace(3) %363, i64 4224
  %.unpack111 = load double, ptr addrspace(3) %418, align 8
  %.elt112149 = getelementptr inbounds i8, ptr addrspace(3) %363, i64 4232
  %.unpack113 = load double, ptr addrspace(3) %.elt112149, align 8
  %419 = getelementptr inbounds i8, ptr addrspace(3) %364, i64 4224
  %.unpack115 = load double, ptr addrspace(3) %419, align 8
  %.elt116150 = getelementptr inbounds i8, ptr addrspace(3) %364, i64 4232
  %.unpack117 = load double, ptr addrspace(3) %.elt116150, align 8
  %420 = select i1 %354, double %.unpack111, double 0x7FF8000000000000
  %421 = select i1 %354, double %.unpack113, double 0.000000e+00
  %422 = fmul double %.unpack82153, %420
  %423 = fmul double %.unpack84154, %421
  %424 = fsub double %422, %423
  %425 = fmul double %.unpack84154, %420
  %426 = fmul double %.unpack82153, %421
  %427 = fadd double %425, %426
  %428 = getelementptr inbounds i8, ptr addrspace(1) %405, i64 32
  %429 = load i32, ptr addrspace(1) %428, align 4, !invariant.load !4
  %430 = icmp slt i32 %429, 0
  %431 = add i32 %429, 21
  %432 = select i1 %430, i32 %431, i32 %429
  %433 = icmp ult i32 %432, 21
  %434 = getelementptr inbounds i8, ptr addrspace(1) %386, i64 786432
  %435 = insertelement <2 x double> poison, double %424, i32 0
  %436 = insertelement <2 x double> %435, double %427, i32 1
  store <2 x double> %436, ptr addrspace(1) %434, align 16
  %437 = getelementptr inbounds i8, ptr addrspace(1) %389, i64 786432
  %.elt121 = select i1 %433, double %.unpack115, double 0x7FF8000000000000
  %.elt123 = select i1 %433, double %.unpack117, double 0.000000e+00
  %438 = insertelement <2 x double> poison, double %.elt121, i32 0
  %439 = insertelement <2 x double> %438, double %.elt123, i32 1
  store <2 x double> %439, ptr addrspace(1) %437, align 16
  ret void
}

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(1179648) %0, ptr noalias readonly align 16 captures(none) dereferenceable(24772608) %1, ptr noalias align 256 captures(none) dereferenceable(1179648) %2, ptr noalias readonly align 256 captures(none) dereferenceable(1179648) %3, ptr noalias readonly align 16 captures(none) dereferenceable(196608) %4, ptr noalias readonly align 16 captures(none) dereferenceable(196608) %5, ptr noalias readonly align 16 captures(none) dereferenceable(49152) %6, ptr noalias readonly align 256 captures(none) dereferenceable(16) %7, ptr noalias readonly align 256 captures(none) dereferenceable(4) %8, ptr noalias readonly align 256 captures(none) dereferenceable(16) %9, ptr noalias readnone align 256 captures(none) dereferenceable(1179648) %10) local_unnamed_addr #0 {
  %12 = addrspacecast ptr %8 to ptr addrspace(1)
  %13 = addrspacecast ptr %9 to ptr addrspace(1)
  %14 = addrspacecast ptr %6 to ptr addrspace(1)
  %15 = addrspacecast ptr %1 to ptr addrspace(1)
  %16 = addrspacecast ptr %4 to ptr addrspace(1)
  %17 = addrspacecast ptr %7 to ptr addrspace(1)
  %18 = addrspacecast ptr %5 to ptr addrspace(1)
  %19 = addrspacecast ptr %3 to ptr addrspace(1)
  %20 = addrspacecast ptr %0 to ptr addrspace(1)
  %21 = addrspacecast ptr %2 to ptr addrspace(1)
  %22 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %23 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !3
  %24 = and i32 %22, 31
  %25 = icmp samesign ult i32 %24, 12
  br i1 %25, label %26, label %._crit_edge

._crit_edge:                                      ; preds = %11
  %.pre = lshr i32 %22, 5
  br label %219

26:                                               ; preds = %11
  %27 = load i32, ptr addrspace(1) %12, align 256, !invariant.load !4
  %28 = tail call i32 @llvm.umin.i32(i32 %27, i32 3)
  %29 = zext nneg i32 %28 to i64
  %30 = getelementptr inbounds i32, ptr addrspace(1) %13, i64 %29
  %31 = load i32, ptr addrspace(1) %30, align 4, !invariant.load !4
  %.not = icmp eq i32 %31, 0
  %32 = select i1 %.not, i32 0, i32 12
  %33 = shl nuw nsw i32 %23, 3
  %34 = udiv i32 %33, 3
  %35 = mul i32 %34, 3
  %.decomposed = sub i32 %33, %35
  %36 = shl nuw nsw i32 %.decomposed, 2
  %37 = lshr i32 %22, 5
  %38 = or disjoint i32 %36, %37
  %39 = tail call i32 @llvm.umin.i32(i32 %34, i32 511)
  %40 = tail call i32 @llvm.umin.i32(i32 %38, i32 11)
  %41 = mul nuw nsw i32 %34, 24
  %42 = add nuw nsw i32 %38, %41
  %43 = add nuw nsw i32 %42, %32
  %44 = zext nneg i32 %43 to i64
  %45 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 %44
  %46 = load i32, ptr addrspace(1) %45, align 4, !invariant.load !4
  %47 = icmp slt i32 %46, 0
  %48 = add i32 %46, 21
  %49 = select i1 %47, i32 %48, i32 %46
  %50 = tail call i32 @llvm.smax.i32(i32 %49, i32 0)
  %51 = tail call i32 @llvm.umin.i32(i32 %50, i32 20)
  %52 = mul nuw nsw i32 %51, 73728
  %53 = mul nuw nsw i32 %39, 144
  %54 = mul nuw nsw i32 %40, 12
  %55 = add nuw nsw i32 %54, %24
  %56 = add nuw nsw i32 %55, %53
  %57 = add nuw nsw i32 %56, %52
  %58 = zext nneg i32 %57 to i64
  %59 = getelementptr inbounds { double, double }, ptr addrspace(1) %15, i64 %58
  %60 = load <2 x double>, ptr addrspace(1) %59, align 16, !invariant.load !4
  %.unpack149 = extractelement <2 x double> %60, i32 0
  %.unpack2150 = extractelement <2 x double> %60, i32 1
  %61 = mul nuw nsw i32 %24, 33
  %62 = add nuw nsw i32 %61, %37
  %63 = zext nneg i32 %62 to i64
  %64 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_02, i64 %63
  store double %.unpack149, ptr addrspace(3) %64, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 8
  store double %.unpack2150, ptr addrspace(3) %.repack3, align 8
  %65 = or disjoint i32 %33, 1
  %66 = udiv i32 %65, 3
  %67 = mul i32 %66, 3
  %.decomposed96 = sub i32 %65, %67
  %68 = shl nuw nsw i32 %.decomposed96, 2
  %69 = or disjoint i32 %68, %37
  %70 = tail call i32 @llvm.umin.i32(i32 %66, i32 511)
  %71 = tail call i32 @llvm.umin.i32(i32 %69, i32 11)
  %72 = add nuw nsw i32 %32, %69
  %73 = mul nuw nsw i32 %66, 24
  %74 = add nuw nsw i32 %72, %73
  %75 = zext nneg i32 %74 to i64
  %76 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 %75
  %77 = load i32, ptr addrspace(1) %76, align 4, !invariant.load !4
  %78 = icmp slt i32 %77, 0
  %79 = add i32 %77, 21
  %80 = select i1 %78, i32 %79, i32 %77
  %81 = tail call i32 @llvm.smax.i32(i32 %80, i32 0)
  %82 = tail call i32 @llvm.umin.i32(i32 %81, i32 20)
  %83 = mul nuw nsw i32 %82, 73728
  %84 = mul nuw nsw i32 %70, 144
  %85 = mul nuw nsw i32 %71, 12
  %86 = add nuw nsw i32 %85, %24
  %87 = add nuw nsw i32 %86, %84
  %88 = add nuw nsw i32 %87, %83
  %89 = zext nneg i32 %88 to i64
  %90 = getelementptr inbounds { double, double }, ptr addrspace(1) %15, i64 %89
  %91 = load <2 x double>, ptr addrspace(1) %90, align 16, !invariant.load !4
  %.unpack5147 = extractelement <2 x double> %91, i32 0
  %.unpack7148 = extractelement <2 x double> %91, i32 1
  %92 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 64
  store double %.unpack5147, ptr addrspace(3) %92, align 8
  %.repack8 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 72
  store double %.unpack7148, ptr addrspace(3) %.repack8, align 8
  %93 = or disjoint i32 %33, 2
  %94 = udiv i32 %93, 3
  %95 = mul i32 %94, 3
  %.decomposed97 = sub i32 %93, %95
  %96 = shl nuw nsw i32 %.decomposed97, 2
  %97 = or disjoint i32 %96, %37
  %98 = tail call i32 @llvm.umin.i32(i32 %94, i32 511)
  %99 = tail call i32 @llvm.umin.i32(i32 %97, i32 11)
  %100 = add nuw nsw i32 %32, %97
  %101 = mul nuw nsw i32 %94, 24
  %102 = add nuw nsw i32 %100, %101
  %103 = zext nneg i32 %102 to i64
  %104 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 %103
  %105 = load i32, ptr addrspace(1) %104, align 4, !invariant.load !4
  %106 = icmp slt i32 %105, 0
  %107 = add i32 %105, 21
  %108 = select i1 %106, i32 %107, i32 %105
  %109 = tail call i32 @llvm.smax.i32(i32 %108, i32 0)
  %110 = tail call i32 @llvm.umin.i32(i32 %109, i32 20)
  %111 = mul nuw nsw i32 %110, 73728
  %112 = mul nuw nsw i32 %98, 144
  %113 = mul nuw nsw i32 %99, 12
  %114 = add nuw nsw i32 %113, %24
  %115 = add nuw nsw i32 %114, %112
  %116 = add nuw nsw i32 %115, %111
  %117 = zext nneg i32 %116 to i64
  %118 = getelementptr inbounds { double, double }, ptr addrspace(1) %15, i64 %117
  %119 = load <2 x double>, ptr addrspace(1) %118, align 16, !invariant.load !4
  %.unpack10145 = extractelement <2 x double> %119, i32 0
  %.unpack12146 = extractelement <2 x double> %119, i32 1
  %120 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 128
  store double %.unpack10145, ptr addrspace(3) %120, align 8
  %.repack13 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 136
  store double %.unpack12146, ptr addrspace(3) %.repack13, align 8
  %121 = tail call i32 @llvm.umin.i32(i32 %34, i32 510)
  %122 = getelementptr inbounds i8, ptr addrspace(1) %45, i64 96
  %123 = load i32, ptr addrspace(1) %122, align 4, !invariant.load !4
  %124 = icmp slt i32 %123, 0
  %125 = add i32 %123, 21
  %126 = select i1 %124, i32 %125, i32 %123
  %127 = tail call i32 @llvm.smax.i32(i32 %126, i32 0)
  %128 = tail call i32 @llvm.umin.i32(i32 %127, i32 20)
  %129 = mul nuw nsw i32 %128, 73728
  %130 = mul nuw nsw i32 %121, 144
  %131 = zext nneg i32 %129 to i64
  %132 = zext nneg i32 %54 to i64
  %133 = zext nneg i32 %130 to i64
  %134 = zext nneg i32 %24 to i64
  %135 = add i64 %134, %133
  %136 = add i64 %135, %132
  %137 = add i64 %136, %131
  %138 = getelementptr inbounds { double, double }, ptr addrspace(1) %15, i64 %137
  %139 = getelementptr inbounds i8, ptr addrspace(1) %138, i64 2304
  %140 = load <2 x double>, ptr addrspace(1) %139, align 16, !invariant.load !4
  %.unpack15143 = extractelement <2 x double> %140, i32 0
  %.unpack17144 = extractelement <2 x double> %140, i32 1
  %141 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 192
  store double %.unpack15143, ptr addrspace(3) %141, align 8
  %.repack18 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 200
  store double %.unpack17144, ptr addrspace(3) %.repack18, align 8
  %142 = or disjoint i32 %33, 4
  %143 = udiv i32 %142, 3
  %144 = tail call i32 @llvm.umin.i32(i32 %143, i32 511)
  %145 = mul nuw nsw i32 %143, 24
  %146 = add nuw nsw i32 %72, %145
  %147 = zext nneg i32 %146 to i64
  %148 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 %147
  %149 = load i32, ptr addrspace(1) %148, align 4, !invariant.load !4
  %150 = icmp slt i32 %149, 0
  %151 = add i32 %149, 21
  %152 = select i1 %150, i32 %151, i32 %149
  %153 = tail call i32 @llvm.smax.i32(i32 %152, i32 0)
  %154 = tail call i32 @llvm.umin.i32(i32 %153, i32 20)
  %155 = mul nuw nsw i32 %154, 73728
  %156 = mul nuw nsw i32 %144, 144
  %157 = add nuw nsw i32 %86, %156
  %158 = add nuw nsw i32 %157, %155
  %159 = zext nneg i32 %158 to i64
  %160 = getelementptr inbounds { double, double }, ptr addrspace(1) %15, i64 %159
  %161 = load <2 x double>, ptr addrspace(1) %160, align 16, !invariant.load !4
  %.unpack20141 = extractelement <2 x double> %161, i32 0
  %.unpack22142 = extractelement <2 x double> %161, i32 1
  %162 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 256
  store double %.unpack20141, ptr addrspace(3) %162, align 8
  %.repack23 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 264
  store double %.unpack22142, ptr addrspace(3) %.repack23, align 8
  %163 = or disjoint i32 %33, 5
  %164 = udiv i32 %163, 3
  %165 = tail call i32 @llvm.umin.i32(i32 %164, i32 511)
  %166 = mul nuw nsw i32 %164, 24
  %167 = add nuw nsw i32 %100, %166
  %168 = zext nneg i32 %167 to i64
  %169 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 %168
  %170 = load i32, ptr addrspace(1) %169, align 4, !invariant.load !4
  %171 = icmp slt i32 %170, 0
  %172 = add i32 %170, 21
  %173 = select i1 %171, i32 %172, i32 %170
  %174 = tail call i32 @llvm.smax.i32(i32 %173, i32 0)
  %175 = tail call i32 @llvm.umin.i32(i32 %174, i32 20)
  %176 = mul nuw nsw i32 %175, 73728
  %177 = mul nuw nsw i32 %165, 144
  %178 = add nuw nsw i32 %114, %177
  %179 = add nuw nsw i32 %178, %176
  %180 = zext nneg i32 %179 to i64
  %181 = getelementptr inbounds { double, double }, ptr addrspace(1) %15, i64 %180
  %182 = load <2 x double>, ptr addrspace(1) %181, align 16, !invariant.load !4
  %.unpack25139 = extractelement <2 x double> %182, i32 0
  %.unpack27140 = extractelement <2 x double> %182, i32 1
  %183 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 320
  store double %.unpack25139, ptr addrspace(3) %183, align 8
  %.repack28 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 328
  store double %.unpack27140, ptr addrspace(3) %.repack28, align 8
  %184 = getelementptr inbounds i8, ptr addrspace(1) %45, i64 192
  %185 = load i32, ptr addrspace(1) %184, align 4, !invariant.load !4
  %186 = icmp slt i32 %185, 0
  %187 = add i32 %185, 21
  %188 = select i1 %186, i32 %187, i32 %185
  %189 = tail call i32 @llvm.smax.i32(i32 %188, i32 0)
  %190 = tail call i32 @llvm.umin.i32(i32 %189, i32 20)
  %191 = mul nuw nsw i32 %190, 73728
  %192 = zext nneg i32 %191 to i64
  %193 = add i64 %136, %192
  %194 = getelementptr inbounds { double, double }, ptr addrspace(1) %15, i64 %193
  %195 = getelementptr inbounds i8, ptr addrspace(1) %194, i64 4608
  %196 = load <2 x double>, ptr addrspace(1) %195, align 16, !invariant.load !4
  %.unpack30137 = extractelement <2 x double> %196, i32 0
  %.unpack32138 = extractelement <2 x double> %196, i32 1
  %197 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 384
  store double %.unpack30137, ptr addrspace(3) %197, align 8
  %.repack33 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 392
  store double %.unpack32138, ptr addrspace(3) %.repack33, align 8
  %198 = or disjoint i32 %33, 7
  %199 = udiv i32 %198, 3
  %200 = tail call i32 @llvm.umin.i32(i32 %199, i32 511)
  %201 = mul nuw nsw i32 %199, 24
  %202 = add nuw nsw i32 %72, %201
  %203 = zext nneg i32 %202 to i64
  %204 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 %203
  %205 = load i32, ptr addrspace(1) %204, align 4, !invariant.load !4
  %206 = icmp slt i32 %205, 0
  %207 = add i32 %205, 21
  %208 = select i1 %206, i32 %207, i32 %205
  %209 = tail call i32 @llvm.smax.i32(i32 %208, i32 0)
  %210 = tail call i32 @llvm.umin.i32(i32 %209, i32 20)
  %211 = mul nuw nsw i32 %210, 73728
  %212 = mul nuw nsw i32 %200, 144
  %213 = add nuw nsw i32 %86, %212
  %214 = add nuw nsw i32 %213, %211
  %215 = zext nneg i32 %214 to i64
  %216 = getelementptr inbounds { double, double }, ptr addrspace(1) %15, i64 %215
  %217 = load <2 x double>, ptr addrspace(1) %216, align 16, !invariant.load !4
  %.unpack35135 = extractelement <2 x double> %217, i32 0
  %.unpack37136 = extractelement <2 x double> %217, i32 1
  %218 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 448
  store double %.unpack35135, ptr addrspace(3) %218, align 8
  %.repack38 = getelementptr inbounds i8, ptr addrspace(3) %64, i64 456
  store double %.unpack37136, ptr addrspace(3) %.repack38, align 8
  br label %219

219:                                              ; preds = %._crit_edge, %26
  %.pre-phi = phi i32 [ %.pre, %._crit_edge ], [ %37, %26 ]
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %220 = shl nuw nsw i32 %23, 5
  %221 = or disjoint i32 %220, %24
  %222 = udiv i32 %221, 12
  %223 = mul i32 %222, 12
  %.decomposed98 = sub i32 %221, %223
  %224 = load i32, ptr addrspace(1) %12, align 256, !invariant.load !4
  %225 = tail call i32 @llvm.umin.i32(i32 %224, i32 3)
  %226 = zext nneg i32 %225 to i64
  %227 = getelementptr inbounds i32, ptr addrspace(1) %13, i64 %226
  %228 = load i32, ptr addrspace(1) %227, align 4, !invariant.load !4
  %.not40 = icmp eq i32 %228, 0
  %229 = select i1 %.not40, i32 0, i32 12
  %230 = mul nuw nsw i32 %222, 24
  %231 = add nuw nsw i32 %230, %.decomposed98
  %232 = add nuw nsw i32 %231, %229
  %233 = zext nneg i32 %232 to i64
  %234 = getelementptr inbounds i32, ptr addrspace(1) %14, i64 %233
  %235 = load i32, ptr addrspace(1) %234, align 4, !invariant.load !4
  %236 = icmp slt i32 %235, 0
  %237 = add i32 %235, 21
  %238 = select i1 %236, i32 %237, i32 %235
  %239 = icmp ult i32 %238, 21
  %240 = getelementptr inbounds { double, double }, ptr addrspace(1) %16, i64 %233
  %241 = load <2 x double>, ptr addrspace(1) %240, align 16, !invariant.load !4
  %.unpack41109 = extractelement <2 x double> %241, i32 0
  %.unpack43110 = extractelement <2 x double> %241, i32 1
  %242 = getelementptr inbounds i32, ptr addrspace(1) %17, i64 %226
  %243 = load i32, ptr addrspace(1) %242, align 4, !invariant.load !4
  %.not44 = icmp eq i32 %243, 0
  %244 = select i1 %.not44, i32 0, i32 12
  %245 = mul nuw nsw i32 %.pre-phi, 33
  %246 = add nuw nsw i32 %245, %24
  %247 = zext nneg i32 %246 to i64
  %248 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_02, i64 %247
  %.unpack45 = load double, ptr addrspace(3) %248, align 8
  %.elt46 = getelementptr inbounds i8, ptr addrspace(3) %248, i64 8
  %.unpack47 = load double, ptr addrspace(3) %.elt46, align 8
  %249 = add nuw nsw i32 %244, %230
  %250 = or disjoint i32 %249, %.pre-phi
  %251 = zext nneg i32 %250 to i64
  %252 = getelementptr inbounds { double, double }, ptr addrspace(1) %18, i64 %251
  %253 = load <2 x double>, ptr addrspace(1) %252, align 16, !invariant.load !4
  %.unpack48115 = extractelement <2 x double> %253, i32 0
  %.unpack50116 = extractelement <2 x double> %253, i32 1
  %254 = mul nuw nsw i32 %.pre-phi, 6144
  %255 = add nuw nsw i32 %254, %220
  %256 = or disjoint i32 %255, %24
  %257 = zext nneg i32 %256 to i64
  %258 = getelementptr inbounds { double, double }, ptr addrspace(1) %19, i64 %257
  %259 = load <2 x double>, ptr addrspace(1) %258, align 16, !invariant.load !4
  %.unpack51117 = extractelement <2 x double> %259, i32 0
  %.unpack53118 = extractelement <2 x double> %259, i32 1
  %260 = select i1 %239, double %.unpack45, double 0x7FF8000000000000
  %261 = select i1 %239, double %.unpack47, double 0.000000e+00
  %262 = fmul double %.unpack41109, %260
  %263 = fmul double %.unpack43110, %261
  %264 = fsub double %262, %263
  %265 = fmul double %.unpack43110, %260
  %266 = fmul double %.unpack41109, %261
  %267 = fadd double %265, %266
  %268 = getelementptr inbounds { double, double }, ptr addrspace(1) %20, i64 %257
  %269 = load <2 x double>, ptr addrspace(1) %268, align 16, !invariant.load !4
  %.unpack54123 = extractelement <2 x double> %269, i32 0
  %.unpack56124 = extractelement <2 x double> %269, i32 1
  %270 = getelementptr inbounds { double, double }, ptr addrspace(1) %21, i64 %257
  %271 = load <2 x double>, ptr addrspace(1) %270, align 16
  %.unpack57129 = extractelement <2 x double> %271, i32 0
  %.unpack59130 = extractelement <2 x double> %271, i32 1
  %272 = fmul double %.unpack48115, %.unpack51117
  %273 = fmul double %.unpack50116, %.unpack53118
  %274 = fsub double %272, %273
  %275 = fmul double %.unpack50116, %.unpack51117
  %276 = fmul double %.unpack48115, %.unpack53118
  %277 = fadd double %275, %276
  %278 = fadd double %264, %.unpack54123
  %279 = fadd double %267, %.unpack56124
  %280 = fadd double %274, %.unpack57129
  %281 = fadd double %277, %.unpack59130
  %282 = fadd double %278, %280
  %283 = fadd double %279, %281
  %284 = fmul double %282, 5.000000e-01
  %285 = fmul double %283, 0.000000e+00
  %286 = fsub double %284, %285
  %287 = fmul double %283, 5.000000e-01
  %288 = fmul double %282, 0.000000e+00
  %289 = fadd double %288, %287
  %290 = insertelement <2 x double> poison, double %286, i32 0
  %291 = insertelement <2 x double> %290, double %289, i32 1
  store <2 x double> %291, ptr addrspace(1) %270, align 16
  %292 = getelementptr inbounds i8, ptr addrspace(3) %248, i64 2112
  %.unpack62 = load double, ptr addrspace(3) %292, align 8
  %.elt63 = getelementptr inbounds i8, ptr addrspace(3) %248, i64 2120
  %.unpack64 = load double, ptr addrspace(3) %.elt63, align 8
  %293 = zext nneg i32 %.pre-phi to i64
  %294 = zext nneg i32 %249 to i64
  %295 = add i64 %294, %293
  %296 = getelementptr inbounds { double, double }, ptr addrspace(1) %18, i64 %295
  %297 = getelementptr inbounds i8, ptr addrspace(1) %296, i64 64
  %298 = load <2 x double>, ptr addrspace(1) %297, align 16, !invariant.load !4
  %.unpack65111 = extractelement <2 x double> %298, i32 0
  %.unpack67112 = extractelement <2 x double> %298, i32 1
  %299 = getelementptr inbounds i8, ptr addrspace(1) %258, i64 393216
  %300 = load <2 x double>, ptr addrspace(1) %299, align 16, !invariant.load !4
  %.unpack68119 = extractelement <2 x double> %300, i32 0
  %.unpack70120 = extractelement <2 x double> %300, i32 1
  %301 = select i1 %239, double %.unpack62, double 0x7FF8000000000000
  %302 = select i1 %239, double %.unpack64, double 0.000000e+00
  %303 = fmul double %.unpack41109, %301
  %304 = fmul double %.unpack43110, %302
  %305 = fsub double %303, %304
  %306 = fmul double %.unpack43110, %301
  %307 = fmul double %.unpack41109, %302
  %308 = fadd double %306, %307
  %309 = getelementptr inbounds i8, ptr addrspace(1) %268, i64 393216
  %310 = load <2 x double>, ptr addrspace(1) %309, align 16, !invariant.load !4
  %.unpack71125 = extractelement <2 x double> %310, i32 0
  %.unpack73126 = extractelement <2 x double> %310, i32 1
  %311 = getelementptr inbounds i8, ptr addrspace(1) %270, i64 393216
  %312 = load <2 x double>, ptr addrspace(1) %311, align 16
  %.unpack74131 = extractelement <2 x double> %312, i32 0
  %.unpack76132 = extractelement <2 x double> %312, i32 1
  %313 = fmul double %.unpack65111, %.unpack68119
  %314 = fmul double %.unpack67112, %.unpack70120
  %315 = fsub double %313, %314
  %316 = fmul double %.unpack67112, %.unpack68119
  %317 = fmul double %.unpack65111, %.unpack70120
  %318 = fadd double %316, %317
  %319 = fadd double %305, %.unpack71125
  %320 = fadd double %308, %.unpack73126
  %321 = fadd double %315, %.unpack74131
  %322 = fadd double %318, %.unpack76132
  %323 = fadd double %319, %321
  %324 = fadd double %320, %322
  %325 = fmul double %323, 5.000000e-01
  %326 = fmul double %324, 0.000000e+00
  %327 = fsub double %325, %326
  %328 = fmul double %324, 5.000000e-01
  %329 = fmul double %323, 0.000000e+00
  %330 = fadd double %329, %328
  %331 = insertelement <2 x double> poison, double %327, i32 0
  %332 = insertelement <2 x double> %331, double %330, i32 1
  store <2 x double> %332, ptr addrspace(1) %311, align 16
  %333 = getelementptr inbounds i8, ptr addrspace(3) %248, i64 4224
  %.unpack79 = load double, ptr addrspace(3) %333, align 8
  %.elt80 = getelementptr inbounds i8, ptr addrspace(3) %248, i64 4232
  %.unpack81 = load double, ptr addrspace(3) %.elt80, align 8
  %334 = getelementptr inbounds i8, ptr addrspace(1) %296, i64 128
  %335 = load <2 x double>, ptr addrspace(1) %334, align 16, !invariant.load !4
  %.unpack82113 = extractelement <2 x double> %335, i32 0
  %.unpack84114 = extractelement <2 x double> %335, i32 1
  %336 = getelementptr inbounds i8, ptr addrspace(1) %258, i64 786432
  %337 = load <2 x double>, ptr addrspace(1) %336, align 16, !invariant.load !4
  %.unpack85121 = extractelement <2 x double> %337, i32 0
  %.unpack87122 = extractelement <2 x double> %337, i32 1
  %338 = select i1 %239, double %.unpack79, double 0x7FF8000000000000
  %339 = select i1 %239, double %.unpack81, double 0.000000e+00
  %340 = fmul double %.unpack41109, %338
  %341 = fmul double %.unpack43110, %339
  %342 = fsub double %340, %341
  %343 = fmul double %.unpack43110, %338
  %344 = fmul double %.unpack41109, %339
  %345 = fadd double %343, %344
  %346 = getelementptr inbounds i8, ptr addrspace(1) %268, i64 786432
  %347 = load <2 x double>, ptr addrspace(1) %346, align 16, !invariant.load !4
  %.unpack88127 = extractelement <2 x double> %347, i32 0
  %.unpack90128 = extractelement <2 x double> %347, i32 1
  %348 = getelementptr inbounds i8, ptr addrspace(1) %270, i64 786432
  %349 = load <2 x double>, ptr addrspace(1) %348, align 16
  %.unpack91133 = extractelement <2 x double> %349, i32 0
  %.unpack93134 = extractelement <2 x double> %349, i32 1
  %350 = fmul double %.unpack82113, %.unpack85121
  %351 = fmul double %.unpack84114, %.unpack87122
  %352 = fsub double %350, %351
  %353 = fmul double %.unpack84114, %.unpack85121
  %354 = fmul double %.unpack82113, %.unpack87122
  %355 = fadd double %353, %354
  %356 = fadd double %342, %.unpack88127
  %357 = fadd double %345, %.unpack90128
  %358 = fadd double %352, %.unpack91133
  %359 = fadd double %355, %.unpack93134
  %360 = fadd double %356, %358
  %361 = fadd double %357, %359
  %362 = fmul double %360, 5.000000e-01
  %363 = fmul double %361, 0.000000e+00
  %364 = fsub double %362, %363
  %365 = fmul double %361, 5.000000e-01
  %366 = fmul double %360, 0.000000e+00
  %367 = fadd double %366, %365
  %368 = insertelement <2 x double> poison, double %364, i32 0
  %369 = insertelement <2 x double> %368, double %367, i32 1
  store <2 x double> %369, ptr addrspace(1) %348, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_transpose_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(2359296) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(2359296) %1) local_unnamed_addr #4 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = shl nuw nsw i32 %5, 7
  %8 = or disjoint i32 %7, %6
  %9 = udiv i32 %8, 24
  %10 = mul i32 %9, 24
  %.decomposed = sub i32 %8, %10
  %11 = mul nuw nsw i32 %.decomposed, 6144
  %12 = and i32 %9, 511
  %13 = mul nuw nsw i32 %12, 12
  %14 = udiv i32 %5, 96
  %15 = or disjoint i32 %11, %14
  %16 = add nuw nsw i32 %15, %13
  %17 = zext nneg i32 %16 to i64
  %18 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %17
  %19 = load <2 x double>, ptr addrspace(1) %18, align 16, !invariant.load !4
  %.unpack5 = extractelement <2 x double> %19, i32 0
  %.unpack26 = extractelement <2 x double> %19, i32 1
  %20 = zext nneg i32 %8 to i64
  %21 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %20
  %22 = insertelement <2 x double> poison, double %.unpack5, i32 0
  %23 = insertelement <2 x double> %22, double %.unpack26, i32 1
  store <2 x double> %23, ptr addrspace(1) %21, align 16
  ret void
}

; Function Attrs: norecurse nounwind
define ptx_kernel void @input_transpose_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(4718592) %0, ptr noalias readonly align 256 captures(none) dereferenceable(4718592) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(4718592) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = addrspacecast ptr %0 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %9 = shl nuw nsw i32 %8, 5
  %10 = and i32 %7, 31
  %11 = or disjoint i32 %9, %10
  %12 = udiv i32 %11, 24
  %13 = mul i32 %12, 24
  %.decomposed = sub i32 %11, %13
  %14 = lshr i32 %7, 5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !7)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  %15 = mul nuw nsw i32 %14, 12288
  %16 = add nuw nsw i32 %11, %15
  %17 = zext nneg i32 %16 to i64
  %18 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %17
  %19 = load <2 x double>, ptr addrspace(1) %18, align 16, !invariant.load !4, !alias.scope !10, !noalias !7
  %.unpack.i138 = extractelement <2 x double> %19, i32 0
  %.unpack2.i139 = extractelement <2 x double> %19, i32 1
  %20 = mul nuw nsw i32 %12, 576
  %21 = mul nuw nsw i32 %14, 24
  %22 = or disjoint i32 %20, %.decomposed
  %23 = add nuw nsw i32 %22, %21
  %24 = zext nneg i32 %23 to i64
  %25 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %24
  %26 = load <2 x double>, ptr addrspace(1) %25, align 16, !invariant.load !4, !alias.scope !7, !noalias !10
  %.unpack3.i174 = extractelement <2 x double> %26, i32 0
  %.unpack5.i175 = extractelement <2 x double> %26, i32 1
  %27 = fadd double %.unpack.i138, %.unpack3.i174
  %28 = fadd double %.unpack2.i139, %.unpack5.i175
  tail call void @llvm.experimental.noalias.scope.decl(metadata !12)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !15)
  %29 = mul i32 %.decomposed, 12287
  %30 = add i32 %11, %29
  %31 = or disjoint i32 %30, %14
  %32 = zext nneg i32 %31 to i64
  %33 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %32
  %34 = load <2 x double>, ptr addrspace(1) %33, align 16, !invariant.load !4, !alias.scope !15, !noalias !12
  %.unpack.i52150 = extractelement <2 x double> %34, i32 0
  %.unpack2.i54151 = extractelement <2 x double> %34, i32 1
  %35 = mul nuw nsw i32 %.decomposed, 24
  %36 = add nuw nsw i32 %20, %35
  %37 = or disjoint i32 %36, %14
  %38 = zext nneg i32 %37 to i64
  %39 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %38
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !4, !alias.scope !12, !noalias !15
  %.unpack3.i55172 = extractelement <2 x double> %40, i32 0
  %.unpack5.i57173 = extractelement <2 x double> %40, i32 1
  %41 = fadd double %.unpack.i52150, %.unpack3.i55172
  %42 = fadd double %.unpack2.i54151, %.unpack5.i57173
  %43 = fadd double %27, %41
  %44 = fsub double %42, %28
  %45 = fmul double %43, 5.000000e-01
  %46 = fmul double %44, 0.000000e+00
  %47 = fsub double %45, %46
  %48 = fmul double %44, 5.000000e-01
  %49 = fmul double %43, 0.000000e+00
  %50 = fadd double %49, %48
  %51 = mul nuw nsw i32 %10, 33
  %52 = add nuw nsw i32 %51, %14
  %53 = zext nneg i32 %52 to i64
  %54 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_03, i64 %53
  store double %47, ptr addrspace(3) %54, align 8
  %.repack1 = getelementptr inbounds i8, ptr addrspace(3) %54, i64 8
  store double %50, ptr addrspace(3) %.repack1, align 8
  tail call void @llvm.experimental.noalias.scope.decl(metadata !17)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !20)
  %55 = getelementptr inbounds { double, double }, ptr addrspace(1) %18, i64 49152
  %56 = load <2 x double>, ptr addrspace(1) %55, align 16, !invariant.load !4, !alias.scope !20, !noalias !17
  %.unpack.i58140 = extractelement <2 x double> %56, i32 0
  %.unpack2.i60141 = extractelement <2 x double> %56, i32 1
  %57 = add i32 %23, 96
  %58 = zext nneg i32 %57 to i64
  %59 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %58
  %60 = load <2 x double>, ptr addrspace(1) %59, align 16, !invariant.load !4, !alias.scope !17, !noalias !20
  %.unpack3.i61170 = extractelement <2 x double> %60, i32 0
  %.unpack5.i63171 = extractelement <2 x double> %60, i32 1
  %61 = fadd double %.unpack.i58140, %.unpack3.i61170
  %62 = fadd double %.unpack2.i60141, %.unpack5.i63171
  tail call void @llvm.experimental.noalias.scope.decl(metadata !22)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !25)
  %63 = zext nneg i32 %30 to i64
  %64 = zext nneg i32 %14 to i64
  %65 = add i64 %63, %64
  %66 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %65
  %67 = getelementptr inbounds i8, ptr addrspace(1) %66, i64 64
  %68 = load <2 x double>, ptr addrspace(1) %67, align 16, !invariant.load !4, !alias.scope !25, !noalias !22
  %.unpack.i64128 = extractelement <2 x double> %68, i32 0
  %.unpack2.i66129 = extractelement <2 x double> %68, i32 1
  %69 = zext nneg i32 %36 to i64
  %70 = add i64 %69, %64
  %71 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %70
  %72 = getelementptr inbounds i8, ptr addrspace(1) %71, i64 64
  %73 = load <2 x double>, ptr addrspace(1) %72, align 16, !invariant.load !4, !alias.scope !22, !noalias !25
  %.unpack3.i67152 = extractelement <2 x double> %73, i32 0
  %.unpack5.i69153 = extractelement <2 x double> %73, i32 1
  %74 = fadd double %.unpack.i64128, %.unpack3.i67152
  %75 = fadd double %.unpack2.i66129, %.unpack5.i69153
  %76 = fadd double %61, %74
  %77 = fsub double %75, %62
  %78 = fmul double %76, 5.000000e-01
  %79 = fmul double %77, 0.000000e+00
  %80 = fsub double %78, %79
  %81 = fmul double %77, 5.000000e-01
  %82 = fmul double %76, 0.000000e+00
  %83 = fadd double %82, %81
  %84 = getelementptr inbounds i8, ptr addrspace(3) %54, i64 64
  store double %80, ptr addrspace(3) %84, align 8
  %.repack3 = getelementptr inbounds i8, ptr addrspace(3) %54, i64 72
  store double %83, ptr addrspace(3) %.repack3, align 8
  tail call void @llvm.experimental.noalias.scope.decl(metadata !27)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !30)
  %85 = getelementptr inbounds { double, double }, ptr addrspace(1) %18, i64 98304
  %86 = load <2 x double>, ptr addrspace(1) %85, align 16, !invariant.load !4, !alias.scope !30, !noalias !27
  %.unpack.i70142 = extractelement <2 x double> %86, i32 0
  %.unpack2.i72143 = extractelement <2 x double> %86, i32 1
  %87 = add i32 %23, 192
  %88 = zext nneg i32 %87 to i64
  %89 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %88
  %90 = load <2 x double>, ptr addrspace(1) %89, align 16, !invariant.load !4, !alias.scope !27, !noalias !30
  %.unpack3.i73168 = extractelement <2 x double> %90, i32 0
  %.unpack5.i75169 = extractelement <2 x double> %90, i32 1
  %91 = fadd double %.unpack.i70142, %.unpack3.i73168
  %92 = fadd double %.unpack2.i72143, %.unpack5.i75169
  tail call void @llvm.experimental.noalias.scope.decl(metadata !32)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !35)
  %93 = getelementptr inbounds i8, ptr addrspace(1) %66, i64 128
  %94 = load <2 x double>, ptr addrspace(1) %93, align 16, !invariant.load !4, !alias.scope !35, !noalias !32
  %.unpack.i76130 = extractelement <2 x double> %94, i32 0
  %.unpack2.i78131 = extractelement <2 x double> %94, i32 1
  %95 = getelementptr inbounds i8, ptr addrspace(1) %71, i64 128
  %96 = load <2 x double>, ptr addrspace(1) %95, align 16, !invariant.load !4, !alias.scope !32, !noalias !35
  %.unpack3.i79154 = extractelement <2 x double> %96, i32 0
  %.unpack5.i81155 = extractelement <2 x double> %96, i32 1
  %97 = fadd double %.unpack.i76130, %.unpack3.i79154
  %98 = fadd double %.unpack2.i78131, %.unpack5.i81155
  %99 = fadd double %91, %97
  %100 = fsub double %98, %92
  %101 = fmul double %99, 5.000000e-01
  %102 = fmul double %100, 0.000000e+00
  %103 = fsub double %101, %102
  %104 = fmul double %100, 5.000000e-01
  %105 = fmul double %99, 0.000000e+00
  %106 = fadd double %105, %104
  %107 = getelementptr inbounds i8, ptr addrspace(3) %54, i64 128
  store double %103, ptr addrspace(3) %107, align 8
  %.repack5 = getelementptr inbounds i8, ptr addrspace(3) %54, i64 136
  store double %106, ptr addrspace(3) %.repack5, align 8
  tail call void @llvm.experimental.noalias.scope.decl(metadata !37)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !40)
  %108 = getelementptr inbounds { double, double }, ptr addrspace(1) %18, i64 147456
  %109 = load <2 x double>, ptr addrspace(1) %108, align 16, !invariant.load !4, !alias.scope !40, !noalias !37
  %.unpack.i82144 = extractelement <2 x double> %109, i32 0
  %.unpack2.i84145 = extractelement <2 x double> %109, i32 1
  %110 = add i32 %23, 288
  %111 = zext nneg i32 %110 to i64
  %112 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %111
  %113 = load <2 x double>, ptr addrspace(1) %112, align 16, !invariant.load !4, !alias.scope !37, !noalias !40
  %.unpack3.i85166 = extractelement <2 x double> %113, i32 0
  %.unpack5.i87167 = extractelement <2 x double> %113, i32 1
  %114 = fadd double %.unpack.i82144, %.unpack3.i85166
  %115 = fadd double %.unpack2.i84145, %.unpack5.i87167
  tail call void @llvm.experimental.noalias.scope.decl(metadata !42)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !45)
  %116 = getelementptr inbounds i8, ptr addrspace(1) %66, i64 192
  %117 = load <2 x double>, ptr addrspace(1) %116, align 16, !invariant.load !4, !alias.scope !45, !noalias !42
  %.unpack.i88132 = extractelement <2 x double> %117, i32 0
  %.unpack2.i90133 = extractelement <2 x double> %117, i32 1
  %118 = getelementptr inbounds i8, ptr addrspace(1) %71, i64 192
  %119 = load <2 x double>, ptr addrspace(1) %118, align 16, !invariant.load !4, !alias.scope !42, !noalias !45
  %.unpack3.i91156 = extractelement <2 x double> %119, i32 0
  %.unpack5.i93157 = extractelement <2 x double> %119, i32 1
  %120 = fadd double %.unpack.i88132, %.unpack3.i91156
  %121 = fadd double %.unpack2.i90133, %.unpack5.i93157
  %122 = fadd double %114, %120
  %123 = fsub double %121, %115
  %124 = fmul double %122, 5.000000e-01
  %125 = fmul double %123, 0.000000e+00
  %126 = fsub double %124, %125
  %127 = fmul double %123, 5.000000e-01
  %128 = fmul double %122, 0.000000e+00
  %129 = fadd double %128, %127
  %130 = getelementptr inbounds i8, ptr addrspace(3) %54, i64 192
  store double %126, ptr addrspace(3) %130, align 8
  %.repack7 = getelementptr inbounds i8, ptr addrspace(3) %54, i64 200
  store double %129, ptr addrspace(3) %.repack7, align 8
  tail call void @llvm.experimental.noalias.scope.decl(metadata !47)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !50)
  %131 = getelementptr inbounds { double, double }, ptr addrspace(1) %18, i64 196608
  %132 = load <2 x double>, ptr addrspace(1) %131, align 16, !invariant.load !4, !alias.scope !50, !noalias !47
  %.unpack.i94146 = extractelement <2 x double> %132, i32 0
  %.unpack2.i96147 = extractelement <2 x double> %132, i32 1
  %133 = add i32 %23, 384
  %134 = zext nneg i32 %133 to i64
  %135 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %134
  %136 = load <2 x double>, ptr addrspace(1) %135, align 16, !invariant.load !4, !alias.scope !47, !noalias !50
  %.unpack3.i97164 = extractelement <2 x double> %136, i32 0
  %.unpack5.i99165 = extractelement <2 x double> %136, i32 1
  %137 = fadd double %.unpack.i94146, %.unpack3.i97164
  %138 = fadd double %.unpack2.i96147, %.unpack5.i99165
  tail call void @llvm.experimental.noalias.scope.decl(metadata !52)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !55)
  %139 = getelementptr inbounds i8, ptr addrspace(1) %66, i64 256
  %140 = load <2 x double>, ptr addrspace(1) %139, align 16, !invariant.load !4, !alias.scope !55, !noalias !52
  %.unpack.i100134 = extractelement <2 x double> %140, i32 0
  %.unpack2.i102135 = extractelement <2 x double> %140, i32 1
  %141 = getelementptr inbounds i8, ptr addrspace(1) %71, i64 256
  %142 = load <2 x double>, ptr addrspace(1) %141, align 16, !invariant.load !4, !alias.scope !52, !noalias !55
  %.unpack3.i103158 = extractelement <2 x double> %142, i32 0
  %.unpack5.i105159 = extractelement <2 x double> %142, i32 1
  %143 = fadd double %.unpack.i100134, %.unpack3.i103158
  %144 = fadd double %.unpack2.i102135, %.unpack5.i105159
  %145 = fadd double %137, %143
  %146 = fsub double %144, %138
  %147 = fmul double %145, 5.000000e-01
  %148 = fmul double %146, 0.000000e+00
  %149 = fsub double %147, %148
  %150 = fmul double %146, 5.000000e-01
  %151 = fmul double %145, 0.000000e+00
  %152 = fadd double %151, %150
  %153 = getelementptr inbounds i8, ptr addrspace(3) %54, i64 256
  store double %149, ptr addrspace(3) %153, align 8
  %.repack9 = getelementptr inbounds i8, ptr addrspace(3) %54, i64 264
  store double %152, ptr addrspace(3) %.repack9, align 8
  tail call void @llvm.experimental.noalias.scope.decl(metadata !57)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !60)
  %154 = getelementptr inbounds { double, double }, ptr addrspace(1) %18, i64 245760
  %155 = load <2 x double>, ptr addrspace(1) %154, align 16, !invariant.load !4, !alias.scope !60, !noalias !57
  %.unpack.i106148 = extractelement <2 x double> %155, i32 0
  %.unpack2.i108149 = extractelement <2 x double> %155, i32 1
  %156 = add i32 %23, 480
  %157 = zext nneg i32 %156 to i64
  %158 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %157
  %159 = load <2 x double>, ptr addrspace(1) %158, align 16, !invariant.load !4, !alias.scope !57, !noalias !60
  %.unpack3.i109162 = extractelement <2 x double> %159, i32 0
  %.unpack5.i111163 = extractelement <2 x double> %159, i32 1
  %160 = fadd double %.unpack.i106148, %.unpack3.i109162
  %161 = fadd double %.unpack2.i108149, %.unpack5.i111163
  tail call void @llvm.experimental.noalias.scope.decl(metadata !62)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !65)
  %162 = getelementptr inbounds i8, ptr addrspace(1) %66, i64 320
  %163 = load <2 x double>, ptr addrspace(1) %162, align 16, !invariant.load !4, !alias.scope !65, !noalias !62
  %.unpack.i112136 = extractelement <2 x double> %163, i32 0
  %.unpack2.i114137 = extractelement <2 x double> %163, i32 1
  %164 = getelementptr inbounds i8, ptr addrspace(1) %71, i64 320
  %165 = load <2 x double>, ptr addrspace(1) %164, align 16, !invariant.load !4, !alias.scope !62, !noalias !65
  %.unpack3.i115160 = extractelement <2 x double> %165, i32 0
  %.unpack5.i117161 = extractelement <2 x double> %165, i32 1
  %166 = fadd double %.unpack.i112136, %.unpack3.i115160
  %167 = fadd double %.unpack2.i114137, %.unpack5.i117161
  %168 = fadd double %160, %166
  %169 = fsub double %167, %161
  %170 = fmul double %168, 5.000000e-01
  %171 = fmul double %169, 0.000000e+00
  %172 = fsub double %170, %171
  %173 = fmul double %169, 5.000000e-01
  %174 = fmul double %168, 0.000000e+00
  %175 = fadd double %174, %173
  %176 = getelementptr inbounds i8, ptr addrspace(3) %54, i64 320
  store double %172, ptr addrspace(3) %176, align 8
  %.repack11 = getelementptr inbounds i8, ptr addrspace(3) %54, i64 328
  store double %175, ptr addrspace(3) %.repack11, align 8
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %177 = icmp samesign ult i32 %10, 24
  br i1 %177, label %178, label %218

178:                                              ; preds = %3
  %179 = mul nuw nsw i32 %14, 33
  %180 = add nuw nsw i32 %179, %10
  %181 = zext nneg i32 %180 to i64
  %182 = getelementptr inbounds { double, double }, ptr addrspace(3) @shared_03, i64 %181
  %.unpack = load double, ptr addrspace(3) %182, align 8
  %.elt13 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 8
  %.unpack14 = load double, ptr addrspace(3) %.elt13, align 8
  %183 = mul nuw nsw i32 %8, 768
  %184 = or disjoint i32 %10, %183
  %185 = add nuw nsw i32 %184, %21
  %186 = zext nneg i32 %185 to i64
  %187 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %186
  %188 = insertelement <2 x double> poison, double %.unpack, i32 0
  %189 = insertelement <2 x double> %188, double %.unpack14, i32 1
  store <2 x double> %189, ptr addrspace(1) %187, align 16
  %190 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 2112
  %.unpack17 = load double, ptr addrspace(3) %190, align 8
  %.elt18 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 2120
  %.unpack19 = load double, ptr addrspace(3) %.elt18, align 8
  %191 = getelementptr inbounds i8, ptr addrspace(1) %187, i64 1536
  %192 = insertelement <2 x double> poison, double %.unpack17, i32 0
  %193 = insertelement <2 x double> %192, double %.unpack19, i32 1
  store <2 x double> %193, ptr addrspace(1) %191, align 16
  %194 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 4224
  %.unpack22 = load double, ptr addrspace(3) %194, align 8
  %.elt23 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 4232
  %.unpack24 = load double, ptr addrspace(3) %.elt23, align 8
  %195 = getelementptr inbounds i8, ptr addrspace(1) %187, i64 3072
  %196 = insertelement <2 x double> poison, double %.unpack22, i32 0
  %197 = insertelement <2 x double> %196, double %.unpack24, i32 1
  store <2 x double> %197, ptr addrspace(1) %195, align 16
  %198 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 6336
  %.unpack27 = load double, ptr addrspace(3) %198, align 8
  %.elt28 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 6344
  %.unpack29 = load double, ptr addrspace(3) %.elt28, align 8
  %199 = getelementptr inbounds i8, ptr addrspace(1) %187, i64 4608
  %200 = insertelement <2 x double> poison, double %.unpack27, i32 0
  %201 = insertelement <2 x double> %200, double %.unpack29, i32 1
  store <2 x double> %201, ptr addrspace(1) %199, align 16
  %202 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 8448
  %.unpack32 = load double, ptr addrspace(3) %202, align 8
  %.elt33 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 8456
  %.unpack34 = load double, ptr addrspace(3) %.elt33, align 8
  %203 = getelementptr inbounds i8, ptr addrspace(1) %187, i64 6144
  %204 = insertelement <2 x double> poison, double %.unpack32, i32 0
  %205 = insertelement <2 x double> %204, double %.unpack34, i32 1
  store <2 x double> %205, ptr addrspace(1) %203, align 16
  %206 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 10560
  %.unpack37 = load double, ptr addrspace(3) %206, align 8
  %.elt38 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 10568
  %.unpack39 = load double, ptr addrspace(3) %.elt38, align 8
  %207 = getelementptr inbounds i8, ptr addrspace(1) %187, i64 7680
  %208 = insertelement <2 x double> poison, double %.unpack37, i32 0
  %209 = insertelement <2 x double> %208, double %.unpack39, i32 1
  store <2 x double> %209, ptr addrspace(1) %207, align 16
  %210 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 12672
  %.unpack42 = load double, ptr addrspace(3) %210, align 8
  %.elt43 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 12680
  %.unpack44 = load double, ptr addrspace(3) %.elt43, align 8
  %211 = getelementptr inbounds i8, ptr addrspace(1) %187, i64 9216
  %212 = insertelement <2 x double> poison, double %.unpack42, i32 0
  %213 = insertelement <2 x double> %212, double %.unpack44, i32 1
  store <2 x double> %213, ptr addrspace(1) %211, align 16
  %214 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 14784
  %.unpack47 = load double, ptr addrspace(3) %214, align 8
  %.elt48 = getelementptr inbounds i8, ptr addrspace(3) %182, i64 14792
  %.unpack49 = load double, ptr addrspace(3) %.elt48, align 8
  %215 = getelementptr inbounds i8, ptr addrspace(1) %187, i64 10752
  %216 = insertelement <2 x double> poison, double %.unpack47, i32 0
  %217 = insertelement <2 x double> %216, double %.unpack49, i32 1
  store <2 x double> %217, ptr addrspace(1) %215, align 16
  br label %218

218:                                              ; preds = %178, %3
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #5

attributes #0 = { norecurse nounwind "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nounwind }
attributes #4 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 128}
!3 = !{i32 0, i32 192}
!4 = !{}
!5 = !{i32 0, i32 1152}
!6 = !{i32 0, i32 384}
!7 = !{!8}
!8 = distinct !{!8, !9, !"fused_transpose_1_add_57_3: argument 0"}
!9 = distinct !{!9, !"fused_transpose_1_add_57_3"}
!10 = !{!11}
!11 = distinct !{!11, !9, !"fused_transpose_1_add_57_3: argument 1"}
!12 = !{!13}
!13 = distinct !{!13, !14, !"fused_transpose_1_add_57_3: argument 0"}
!14 = distinct !{!14, !"fused_transpose_1_add_57_3"}
!15 = !{!16}
!16 = distinct !{!16, !14, !"fused_transpose_1_add_57_3: argument 1"}
!17 = !{!18}
!18 = distinct !{!18, !19, !"fused_transpose_1_add_57_3: argument 0"}
!19 = distinct !{!19, !"fused_transpose_1_add_57_3"}
!20 = !{!21}
!21 = distinct !{!21, !19, !"fused_transpose_1_add_57_3: argument 1"}
!22 = !{!23}
!23 = distinct !{!23, !24, !"fused_transpose_1_add_57_3: argument 0"}
!24 = distinct !{!24, !"fused_transpose_1_add_57_3"}
!25 = !{!26}
!26 = distinct !{!26, !24, !"fused_transpose_1_add_57_3: argument 1"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"fused_transpose_1_add_57_3: argument 0"}
!29 = distinct !{!29, !"fused_transpose_1_add_57_3"}
!30 = !{!31}
!31 = distinct !{!31, !29, !"fused_transpose_1_add_57_3: argument 1"}
!32 = !{!33}
!33 = distinct !{!33, !34, !"fused_transpose_1_add_57_3: argument 0"}
!34 = distinct !{!34, !"fused_transpose_1_add_57_3"}
!35 = !{!36}
!36 = distinct !{!36, !34, !"fused_transpose_1_add_57_3: argument 1"}
!37 = !{!38}
!38 = distinct !{!38, !39, !"fused_transpose_1_add_57_3: argument 0"}
!39 = distinct !{!39, !"fused_transpose_1_add_57_3"}
!40 = !{!41}
!41 = distinct !{!41, !39, !"fused_transpose_1_add_57_3: argument 1"}
!42 = !{!43}
!43 = distinct !{!43, !44, !"fused_transpose_1_add_57_3: argument 0"}
!44 = distinct !{!44, !"fused_transpose_1_add_57_3"}
!45 = !{!46}
!46 = distinct !{!46, !44, !"fused_transpose_1_add_57_3: argument 1"}
!47 = !{!48}
!48 = distinct !{!48, !49, !"fused_transpose_1_add_57_3: argument 0"}
!49 = distinct !{!49, !"fused_transpose_1_add_57_3"}
!50 = !{!51}
!51 = distinct !{!51, !49, !"fused_transpose_1_add_57_3: argument 1"}
!52 = !{!53}
!53 = distinct !{!53, !54, !"fused_transpose_1_add_57_3: argument 0"}
!54 = distinct !{!54, !"fused_transpose_1_add_57_3"}
!55 = !{!56}
!56 = distinct !{!56, !54, !"fused_transpose_1_add_57_3: argument 1"}
!57 = !{!58}
!58 = distinct !{!58, !59, !"fused_transpose_1_add_57_3: argument 0"}
!59 = distinct !{!59, !"fused_transpose_1_add_57_3"}
!60 = !{!61}
!61 = distinct !{!61, !59, !"fused_transpose_1_add_57_3: argument 1"}
!62 = !{!63}
!63 = distinct !{!63, !64, !"fused_transpose_1_add_57_3: argument 0"}
!64 = distinct !{!64, !"fused_transpose_1_add_57_3"}
!65 = !{!66}
!66 = distinct !{!66, !64, !"fused_transpose_1_add_57_3: argument 1"}
