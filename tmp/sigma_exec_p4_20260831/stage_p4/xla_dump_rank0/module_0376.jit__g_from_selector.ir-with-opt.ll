; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__cudart_i2opi_d = internal unnamed_addr addrspace(1) constant [18 x i64] [i64 7780917995555872008, i64 4397547296490951402, i64 8441921394348257659, i64 5712322887342352941, i64 7869616827067468215, i64 -1211730484530615009, i64 2303758334597371919, i64 -7168499653074671557, i64 4148332274289687028, i64 -1613291254968254911, i64 -1692731182770600828, i64 -135693905287338178, i64 452944820249399836, i64 -5249950069107600672, i64 -121206125134887583, i64 -2638381946312093631, i64 -277156292786332224, i64 -6703182060581546711], align 8
@__cudart_sin_cos_coeffs = internal unnamed_addr addrspace(1) constant [16 x double] [double 0xBE5AE5F12CB0D246, double 0x3EC71DE369ACE392, double 0xBF2A01A019DB62A1, double 0x3F81111111110818, double 0xBFC5555555555554, double 0.000000e+00, double 0.000000e+00, double 0xBDA8FF8320FD8164, double 0x3E21EEA7C1EF8528, double 0xBE927E4F8E06E6D9, double 0x3EFA01A019DDBCE9, double 0xBF56C16C16C15D47, double 0x3FA5555555555551, double -5.000000e-01, double 1.000000e+00, double 0.000000e+00], align 16

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define ptx_kernel void @loop_select_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(98304) %0, ptr noalias readonly align 16 captures(none) dereferenceable(98304) %1, ptr noalias readonly align 16 captures(none) dereferenceable(8) %2, ptr noalias readonly align 16 captures(none) dereferenceable(16) %3, ptr noalias writeonly align 256 captures(none) dereferenceable(196608) %4) local_unnamed_addr #0 {
  %6 = addrspacecast ptr %1 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = addrspacecast ptr %0 to ptr addrspace(1)
  %10 = addrspacecast ptr %4 to ptr addrspace(1)
  %11 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %12 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %13 = shl nuw nsw i32 %11, 7
  %14 = or disjoint i32 %13, %12
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds double, ptr addrspace(1) %6, i64 %15
  %17 = load double, ptr addrspace(1) %16, align 8, !invariant.load !6
  %18 = load double, ptr addrspace(1) %7, align 16, !invariant.load !6
  %19 = fsub double %17, %18
  %20 = load <2 x double>, ptr addrspace(1) %8, align 16, !invariant.load !6
  %.unpack24 = extractelement <2 x double> %20, i32 0
  %.unpack225 = extractelement <2 x double> %20, i32 1
  %21 = fmul double %.unpack24, 0.000000e+00
  %22 = fsub double %21, %.unpack225
  %23 = fmul double %.unpack225, 0.000000e+00
  %24 = fadd double %.unpack24, %23
  %25 = fmul double %24, 0.000000e+00
  %26 = fmul double %22, %19
  %27 = fsub double %25, %26
  %28 = fmul double %22, -0.000000e+00
  %29 = fmul double %19, %24
  %30 = fsub double %28, %29
  %31 = fmul double %27, 5.000000e-01
  %32 = tail call double @llvm.fma.f64(double %31, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %33 = tail call i32 @llvm.nvvm.d2i.lo(double %32) #6
  %34 = tail call double @llvm.nvvm.add.rn.d(double %32, double 0xC338000000000000) #6
  %35 = tail call double @llvm.fma.f64(double %34, double 0xBFE62E42FEFA39EF, double %31)
  %36 = tail call double @llvm.fma.f64(double %34, double 0xBC7ABC9E3B39803F, double %35)
  %37 = tail call double @llvm.fma.f64(double %36, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %38 = tail call double @llvm.fma.f64(double %37, double %36, double 0x3EC71DEE62401315)
  %39 = tail call double @llvm.fma.f64(double %38, double %36, double 0x3EFA01997C89EB71)
  %40 = tail call double @llvm.fma.f64(double %39, double %36, double 0x3F2A01A014761F65)
  %41 = tail call double @llvm.fma.f64(double %40, double %36, double 0x3F56C16C1852B7AF)
  %42 = tail call double @llvm.fma.f64(double %41, double %36, double 0x3F81111111122322)
  %43 = tail call double @llvm.fma.f64(double %42, double %36, double 0x3FA55555555502A1)
  %44 = tail call double @llvm.fma.f64(double %43, double %36, double 0x3FC5555555555511)
  %45 = tail call double @llvm.fma.f64(double %44, double %36, double 0x3FE000000000000B)
  %46 = tail call double @llvm.fma.f64(double %45, double %36, double 1.000000e+00)
  %47 = tail call double @llvm.fma.f64(double %46, double %36, double 1.000000e+00)
  %48 = tail call i32 @llvm.nvvm.d2i.lo(double %47) #6
  %49 = tail call i32 @llvm.nvvm.d2i.hi(double %47) #6
  %50 = shl i32 %33, 20
  %51 = add i32 %49, %50
  %52 = tail call double @llvm.nvvm.lohi.i2d(i32 %48, i32 %51) #6
  %53 = tail call i32 @llvm.nvvm.d2i.hi(double %31) #6
  %54 = bitcast i32 %53 to float
  %55 = tail call float @llvm.nvvm.fabs.f32(float %54)
  %56 = fcmp olt float %55, 0x4010C46560000000
  br i1 %56, label %__nv_exp.exit, label %__internal_fast_icmp_abs_lt.exit.i

__internal_fast_icmp_abs_lt.exit.i:               ; preds = %5
  %57 = fcmp olt double %31, 0.000000e+00
  %58 = fadd double %31, 0x7FF0000000000000
  %z.0.i11 = select i1 %57, double 0.000000e+00, double %58
  %59 = fcmp olt float %55, 0x4010E90000000000
  br i1 %59, label %60, label %__nv_exp.exit

60:                                               ; preds = %__internal_fast_icmp_abs_lt.exit.i
  %61 = sdiv i32 %33, 2
  %62 = shl i32 %61, 20
  %63 = add i32 %49, %62
  %64 = tail call double @llvm.nvvm.lohi.i2d(i32 %48, i32 %63) #6
  %65 = sub nsw i32 %33, %61
  %66 = shl i32 %65, 20
  %67 = add nsw i32 %66, 1072693248
  %68 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %67) #6
  %69 = fmul double %68, %64
  br label %__nv_exp.exit

__nv_exp.exit:                                    ; preds = %5, %__internal_fast_icmp_abs_lt.exit.i, %60
  %z.2.i = phi double [ %52, %5 ], [ %69, %60 ], [ %z.0.i11, %__internal_fast_icmp_abs_lt.exit.i ]
  %70 = tail call double @llvm.nvvm.fabs.f64(double %30)
  %71 = fcmp oeq double %70, 0x7FF0000000000000
  br i1 %71, label %72, label %74

72:                                               ; preds = %__nv_exp.exit
  %73 = tail call double @llvm.nvvm.mul.rn.d(double %30, double 0.000000e+00) #6
  br label %__nv_sin.exit

74:                                               ; preds = %__nv_exp.exit
  %75 = fmul double %30, 0x3FE45F306DC9C883
  %76 = tail call i32 @llvm.nvvm.d2i.rn(double %75) #6
  %77 = sitofp i32 %76 to double
  %78 = fneg double %77
  %79 = tail call double @llvm.fma.f64(double %78, double 0x3FF921FB54442D18, double %30)
  %80 = tail call double @llvm.fma.f64(double %78, double 0x3C91A62633145C00, double %79)
  %81 = tail call double @llvm.fma.f64(double %78, double 0x397B839A252049C0, double %80)
  %82 = fcmp ult double %70, 0x41E0000000000000
  br i1 %82, label %__nv_sin.exit, label %83

83:                                               ; preds = %74
  %84 = tail call fastcc { double, i32 } @__internal_trig_reduction_slowpathd(double %30) #6
  %newret20 = extractvalue { double, i32 } %84, 0
  %newret22 = extractvalue { double, i32 } %84, 1
  br label %__nv_sin.exit

__nv_sin.exit:                                    ; preds = %72, %74, %83
  %z.0.i = phi double [ %73, %72 ], [ %newret20, %83 ], [ %81, %74 ]
  %i.0.i = phi i32 [ 0, %72 ], [ %newret22, %83 ], [ %76, %74 ]
  %85 = fcmp oeq double %70, 0x7FF0000000000000
  %86 = and i32 %i.0.i, 1
  %87 = shl nuw nsw i32 %86, 3
  %88 = zext nneg i32 %87 to i64
  %89 = getelementptr inbounds nuw double, ptr addrspace(1) @__cudart_sin_cos_coeffs, i64 %88
  %90 = load <2 x double>, ptr addrspace(1) %89, align 16, !invariant.load !6
  %91 = extractelement <2 x double> %90, i32 0
  %92 = extractelement <2 x double> %90, i32 1
  %93 = getelementptr inbounds nuw i8, ptr addrspace(1) %89, i64 16
  %94 = load <2 x double>, ptr addrspace(1) %93, align 16, !invariant.load !6
  %95 = extractelement <2 x double> %94, i32 0
  %96 = extractelement <2 x double> %94, i32 1
  %97 = getelementptr inbounds nuw i8, ptr addrspace(1) %89, i64 32
  %98 = load <2 x double>, ptr addrspace(1) %97, align 16, !invariant.load !6
  %99 = extractelement <2 x double> %98, i32 0
  %100 = extractelement <2 x double> %98, i32 1
  br i1 %85, label %101, label %103

101:                                              ; preds = %__nv_sin.exit
  %102 = tail call double @llvm.nvvm.mul.rn.d(double %30, double 0.000000e+00) #6
  br label %__nv_cos.exit

103:                                              ; preds = %__nv_sin.exit
  %104 = fmul double %30, 0x3FE45F306DC9C883
  %105 = tail call i32 @llvm.nvvm.d2i.rn(double %104) #6
  %106 = sitofp i32 %105 to double
  %107 = fneg double %106
  %108 = tail call double @llvm.fma.f64(double %107, double 0x3FF921FB54442D18, double %30)
  %109 = tail call double @llvm.fma.f64(double %107, double 0x3C91A62633145C00, double %108)
  %110 = tail call double @llvm.fma.f64(double %107, double 0x397B839A252049C0, double %109)
  %111 = fcmp ult double %70, 0x41E0000000000000
  br i1 %111, label %__internal_trig_reduction_kerneld.exit.i, label %112

112:                                              ; preds = %103
  %113 = tail call fastcc { double, i32 } @__internal_trig_reduction_slowpathd(double %30) #6
  %newret = extractvalue { double, i32 } %113, 0
  %newret18 = extractvalue { double, i32 } %113, 1
  br label %__internal_trig_reduction_kerneld.exit.i

__internal_trig_reduction_kerneld.exit.i:         ; preds = %112, %103
  %t.i1.0.i = phi double [ %newret, %112 ], [ %110, %103 ]
  %q.i.0.i = phi i32 [ %newret18, %112 ], [ %105, %103 ]
  %114 = add nsw i32 %q.i.0.i, 1
  br label %__nv_cos.exit

__nv_cos.exit:                                    ; preds = %101, %__internal_trig_reduction_kerneld.exit.i
  %z.0.i5 = phi double [ %102, %101 ], [ %t.i1.0.i, %__internal_trig_reduction_kerneld.exit.i ]
  %i.0.i6 = phi i32 [ 1, %101 ], [ %114, %__internal_trig_reduction_kerneld.exit.i ]
  %115 = and i32 %i.0.i6, 1
  %116 = shl nuw nsw i32 %115, 3
  %117 = zext nneg i32 %116 to i64
  %118 = getelementptr inbounds nuw double, ptr addrspace(1) @__cudart_sin_cos_coeffs, i64 %117
  %119 = load <2 x double>, ptr addrspace(1) %118, align 16, !invariant.load !6
  %120 = extractelement <2 x double> %119, i32 0
  %121 = extractelement <2 x double> %119, i32 1
  %122 = getelementptr inbounds nuw i8, ptr addrspace(1) %118, i64 16
  %123 = load <2 x double>, ptr addrspace(1) %122, align 16, !invariant.load !6
  %124 = extractelement <2 x double> %123, i32 0
  %125 = extractelement <2 x double> %123, i32 1
  %126 = getelementptr inbounds nuw i8, ptr addrspace(1) %118, i64 32
  %127 = load <2 x double>, ptr addrspace(1) %126, align 16, !invariant.load !6
  %128 = extractelement <2 x double> %127, i32 0
  %129 = extractelement <2 x double> %127, i32 1
  %130 = tail call double @llvm.fma.f64(double %27, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %131 = tail call i32 @llvm.nvvm.d2i.lo(double %130) #6
  %132 = tail call double @llvm.nvvm.add.rn.d(double %130, double 0xC338000000000000) #6
  %133 = tail call double @llvm.fma.f64(double %132, double 0xBFE62E42FEFA39EF, double %27)
  %134 = tail call double @llvm.fma.f64(double %132, double 0xBC7ABC9E3B39803F, double %133)
  %135 = tail call double @llvm.fma.f64(double %134, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %136 = tail call double @llvm.fma.f64(double %135, double %134, double 0x3EC71DEE62401315)
  %137 = tail call double @llvm.fma.f64(double %136, double %134, double 0x3EFA01997C89EB71)
  %138 = tail call double @llvm.fma.f64(double %137, double %134, double 0x3F2A01A014761F65)
  %139 = tail call double @llvm.fma.f64(double %138, double %134, double 0x3F56C16C1852B7AF)
  %140 = tail call double @llvm.fma.f64(double %139, double %134, double 0x3F81111111122322)
  %141 = tail call double @llvm.fma.f64(double %140, double %134, double 0x3FA55555555502A1)
  %142 = tail call double @llvm.fma.f64(double %141, double %134, double 0x3FC5555555555511)
  %143 = tail call double @llvm.fma.f64(double %142, double %134, double 0x3FE000000000000B)
  %144 = tail call double @llvm.fma.f64(double %143, double %134, double 1.000000e+00)
  %145 = tail call double @llvm.fma.f64(double %144, double %134, double 1.000000e+00)
  %146 = tail call i32 @llvm.nvvm.d2i.lo(double %145) #6
  %147 = tail call i32 @llvm.nvvm.d2i.hi(double %145) #6
  %148 = shl i32 %131, 20
  %149 = add i32 %147, %148
  %150 = tail call double @llvm.nvvm.lohi.i2d(i32 %146, i32 %149) #6
  %151 = tail call i32 @llvm.nvvm.d2i.hi(double %27) #6
  %152 = bitcast i32 %151 to float
  %153 = tail call float @llvm.nvvm.fabs.f32(float %152)
  %154 = fcmp olt float %153, 0x4010C46560000000
  br i1 %154, label %__nv_exp.exit15, label %__internal_fast_icmp_abs_lt.exit.i12

__internal_fast_icmp_abs_lt.exit.i12:             ; preds = %__nv_cos.exit
  %155 = fcmp olt double %27, 0.000000e+00
  %156 = fadd double %27, 0x7FF0000000000000
  %z.0.i13 = select i1 %155, double 0.000000e+00, double %156
  %157 = fcmp olt float %153, 0x4010E90000000000
  br i1 %157, label %158, label %__nv_exp.exit15

158:                                              ; preds = %__internal_fast_icmp_abs_lt.exit.i12
  %159 = sdiv i32 %131, 2
  %160 = shl i32 %159, 20
  %161 = add i32 %147, %160
  %162 = tail call double @llvm.nvvm.lohi.i2d(i32 %146, i32 %161) #6
  %163 = sub nsw i32 %131, %159
  %164 = shl i32 %163, 20
  %165 = add nsw i32 %164, 1072693248
  %166 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %165) #6
  %167 = fmul double %166, %162
  br label %__nv_exp.exit15

__nv_exp.exit15:                                  ; preds = %__nv_cos.exit, %__internal_fast_icmp_abs_lt.exit.i12, %158
  %z.2.i14 = phi double [ %150, %__nv_cos.exit ], [ %167, %158 ], [ %z.0.i13, %__internal_fast_icmp_abs_lt.exit.i12 ]
  %168 = and i32 %i.0.i6, 2
  %.not1.i9 = icmp eq i32 %168, 0
  %.not.i7 = icmp eq i32 %115, 0
  %169 = select i1 %.not.i7, double 0x3DE5DB65F9785EBA, double 0xBDA8FF8320FD8164
  %170 = tail call double @llvm.nvvm.mul.rn.d(double %z.0.i5, double %z.0.i5) #6
  %171 = tail call double @llvm.fma.f64(double %169, double %170, double %120)
  %172 = tail call double @llvm.fma.f64(double %171, double %170, double %121)
  %173 = tail call double @llvm.fma.f64(double %172, double %170, double %124)
  %174 = tail call double @llvm.fma.f64(double %173, double %170, double %125)
  %175 = tail call double @llvm.fma.f64(double %174, double %170, double %128)
  %176 = tail call double @llvm.fma.f64(double %175, double %170, double %129)
  %177 = tail call double @llvm.fma.f64(double %176, double %z.0.i5, double %z.0.i5)
  %178 = tail call double @llvm.fma.f64(double %176, double %170, double 1.000000e+00)
  %spec.select.i8 = select i1 %.not.i7, double %177, double %178
  %179 = fsub double 0.000000e+00, %spec.select.i8
  %.1.i10 = select i1 %.not1.i9, double %spec.select.i8, double %179
  %180 = fmul double %z.2.i, %.1.i10
  %181 = and i32 %i.0.i, 2
  %.not1.i = icmp eq i32 %181, 0
  %.not.i = icmp eq i32 %86, 0
  %182 = select i1 %.not.i, double 0x3DE5DB65F9785EBA, double 0xBDA8FF8320FD8164
  %183 = tail call double @llvm.nvvm.mul.rn.d(double %z.0.i, double %z.0.i) #6
  %184 = tail call double @llvm.fma.f64(double %182, double %183, double %91)
  %185 = tail call double @llvm.fma.f64(double %184, double %183, double %92)
  %186 = tail call double @llvm.fma.f64(double %185, double %183, double %95)
  %187 = tail call double @llvm.fma.f64(double %186, double %183, double %96)
  %188 = tail call double @llvm.fma.f64(double %187, double %183, double %99)
  %189 = tail call double @llvm.fma.f64(double %188, double %183, double %100)
  %190 = tail call double @llvm.fma.f64(double %189, double %z.0.i, double %z.0.i)
  %191 = tail call double @llvm.fma.f64(double %189, double %183, double 1.000000e+00)
  %spec.select.i = select i1 %.not.i, double %190, double %191
  %192 = fsub double 0.000000e+00, %spec.select.i
  %.1.i = select i1 %.not1.i, double %spec.select.i, double %192
  %193 = fmul double %z.2.i, %.1.i
  %194 = fmul double %z.2.i, %193
  %195 = fmul double %.1.i, %z.2.i14
  %196 = fcmp oeq double %z.2.i14, 0x7FF0000000000000
  %197 = fmul double %z.2.i, %180
  %198 = fmul double %.1.i10, %z.2.i14
  %199 = fcmp oeq double %30, 0.000000e+00
  %200 = select i1 %196, double %194, double %195
  %201 = select i1 %196, double %197, double %198
  %202 = select i1 %199, double 0.000000e+00, double %200
  %203 = getelementptr inbounds double, ptr addrspace(1) %9, i64 %15
  %204 = load double, ptr addrspace(1) %203, align 8, !invariant.load !6
  %205 = fcmp une double %204, 0.000000e+00
  %206 = fmul double %204, %201
  %207 = fmul double %202, 0.000000e+00
  %208 = fsub double %206, %207
  %209 = fmul double %204, %202
  %210 = fmul double %201, 0.000000e+00
  %211 = fadd double %210, %209
  %212 = getelementptr inbounds { double, double }, ptr addrspace(1) %10, i64 %15
  %.elt = select i1 %205, double %208, double 0.000000e+00
  %.elt4 = select i1 %205, double %211, double 0.000000e+00
  %213 = insertelement <2 x double> poison, double %.elt, i32 0
  %214 = insertelement <2 x double> %213, double %.elt4, i32 1
  store <2 x double> %214, ptr addrspace(1) %212, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(196608) %0, ptr noalias readonly align 16 captures(none) dereferenceable(121896960) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(121896960) %2) local_unnamed_addr #2 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !7
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %9 = shl nuw nsw i32 %7, 7
  %10 = or disjoint i32 %9, %8
  %11 = udiv i32 %10, 155
  %12 = zext nneg i32 %11 to i64
  %13 = getelementptr inbounds { double, double }, ptr addrspace(1) %4, i64 %12
  %14 = load <2 x double>, ptr addrspace(1) %13, align 16, !invariant.load !6
  %.unpack29 = extractelement <2 x double> %14, i32 0
  %.unpack230 = extractelement <2 x double> %14, i32 1
  %15 = shl nuw nsw i32 %8, 2
  %16 = shl nuw nsw i32 %7, 9
  %17 = or disjoint i32 %15, %16
  %18 = zext nneg i32 %17 to i64
  %19 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %18
  %20 = load <2 x double>, ptr addrspace(1) %19, align 16, !invariant.load !6
  %.unpack331 = extractelement <2 x double> %20, i32 0
  %.unpack532 = extractelement <2 x double> %20, i32 1
  %21 = fmul double %.unpack29, %.unpack331
  %22 = fmul double %.unpack230, %.unpack532
  %23 = fadd double %21, %22
  %24 = fmul double %.unpack230, %.unpack331
  %25 = fmul double %.unpack29, %.unpack532
  %26 = fsub double %24, %25
  %27 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %18
  %28 = insertelement <2 x double> poison, double %23, i32 0
  %29 = insertelement <2 x double> %28, double %26, i32 1
  store <2 x double> %29, ptr addrspace(1) %27, align 64
  %30 = getelementptr inbounds i8, ptr addrspace(1) %19, i64 16
  %31 = load <2 x double>, ptr addrspace(1) %30, align 16, !invariant.load !6
  %.unpack833 = extractelement <2 x double> %31, i32 0
  %.unpack1034 = extractelement <2 x double> %31, i32 1
  %32 = fmul double %.unpack29, %.unpack833
  %33 = fmul double %.unpack230, %.unpack1034
  %34 = fadd double %32, %33
  %35 = fmul double %.unpack230, %.unpack833
  %36 = fmul double %.unpack29, %.unpack1034
  %37 = fsub double %35, %36
  %38 = getelementptr inbounds i8, ptr addrspace(1) %27, i64 16
  %39 = insertelement <2 x double> poison, double %34, i32 0
  %40 = insertelement <2 x double> %39, double %37, i32 1
  store <2 x double> %40, ptr addrspace(1) %38, align 16
  %41 = getelementptr inbounds i8, ptr addrspace(1) %19, i64 32
  %42 = load <2 x double>, ptr addrspace(1) %41, align 16, !invariant.load !6
  %.unpack1335 = extractelement <2 x double> %42, i32 0
  %.unpack1536 = extractelement <2 x double> %42, i32 1
  %43 = fmul double %.unpack29, %.unpack1335
  %44 = fmul double %.unpack230, %.unpack1536
  %45 = fadd double %43, %44
  %46 = fmul double %.unpack230, %.unpack1335
  %47 = fmul double %.unpack29, %.unpack1536
  %48 = fsub double %46, %47
  %49 = getelementptr inbounds i8, ptr addrspace(1) %27, i64 32
  %50 = insertelement <2 x double> poison, double %45, i32 0
  %51 = insertelement <2 x double> %50, double %48, i32 1
  store <2 x double> %51, ptr addrspace(1) %49, align 32
  %52 = getelementptr inbounds i8, ptr addrspace(1) %19, i64 48
  %53 = load <2 x double>, ptr addrspace(1) %52, align 16, !invariant.load !6
  %.unpack1837 = extractelement <2 x double> %53, i32 0
  %.unpack2038 = extractelement <2 x double> %53, i32 1
  %54 = fmul double %.unpack29, %.unpack1837
  %55 = fmul double %.unpack230, %.unpack2038
  %56 = fadd double %54, %55
  %57 = fmul double %.unpack230, %.unpack1837
  %58 = fmul double %.unpack29, %.unpack2038
  %59 = fsub double %57, %58
  %60 = getelementptr inbounds i8, ptr addrspace(1) %27, i64 48
  %61 = insertelement <2 x double> poison, double %56, i32 0
  %62 = insertelement <2 x double> %61, double %59, i32 1
  store <2 x double> %62, ptr addrspace(1) %60, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.mul.rn.d(double, double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.rn(double) #3

; Function Attrs: nofree noinline nosync nounwind memory(none)
define internal fastcc { double, i32 } @__internal_trig_reduction_slowpathd(double %a) unnamed_addr #4 {
  %result = alloca [5 x i64], align 8
  %1 = addrspacecast ptr %result to ptr addrspace(5)
  %2 = tail call i32 @llvm.nvvm.d2i.hi(double %a) #6
  %3 = and i32 %2, -2147483648
  %4 = lshr i32 %2, 20
  %5 = and i32 %4, 2047
  %6 = icmp eq i32 %5, 2047
  br i1 %6, label %90, label %7

7:                                                ; preds = %0
  %8 = add nsw i32 %5, -1024
  %9 = bitcast double %a to i64
  %10 = shl i64 %9, 11
  %11 = or i64 %10, -9223372036854775808
  %12 = lshr i32 %8, 6
  %13 = sub nsw i32 15, %12
  %14 = sub nsw i32 19, %12
  %15 = tail call i32 @llvm.smin.i32(i32 %14, i32 18) #6
  %16 = icmp slt i32 %13, %15
  br i1 %16, label %.lr.ph.preheader, label %.._crit_edge_crit_edge

.lr.ph.preheader:                                 ; preds = %7
  %17 = sub i32 15, %15
  %18 = sub i32 %17, %12
  %19 = zext i32 %8 to i64
  %20 = lshr i64 %19, 6
  %21 = sub i32 0, %12
  %22 = sext i32 %21 to i64
  %23 = add i64 %20, %22
  %24 = shl nsw i64 %23, 3
  %scevgep = getelementptr i8, ptr addrspace(5) %1, i64 %24
  %25 = shl nsw i64 %22, 3
  %26 = add nsw i64 %25, 120
  %scevgep7 = getelementptr i8, ptr addrspace(1) @__cudart_i2opi_d, i64 %26
  br label %.lr.ph

.._crit_edge_crit_edge:                           ; preds = %7
  %.pre = sext i32 %12 to i64
  %.pre3 = sub i64 0, %.pre
  br label %._crit_edge

.lr.ph:                                           ; preds = %.lr.ph.preheader, %.lr.ph
  %lsr.iv8 = phi ptr addrspace(1) [ %scevgep7, %.lr.ph.preheader ], [ %scevgep9, %.lr.ph ]
  %lsr.iv5 = phi ptr addrspace(5) [ %scevgep, %.lr.ph.preheader ], [ %scevgep6, %.lr.ph ]
  %lsr.iv = phi i32 [ %18, %.lr.ph.preheader ], [ %math, %.lr.ph ]
  %p.129.07 = phi i64 [ %30, %.lr.ph ], [ 0, %.lr.ph.preheader ]
  %27 = load i64, ptr addrspace(1) %lsr.iv8, align 8, !invariant.load !6
  %28 = tail call { i64, i64 } asm "{\0A\09.reg .u32 r0, r1, r2, r3, alo, ahi, blo, bhi, clo, chi;\0A\09mov.b64         {alo,ahi}, $2;    \0A\09mov.b64         {blo,bhi}, $3;    \0A\09mov.b64         {clo,chi}, $4;    \0A\09mad.lo.cc.u32   r0, alo, blo, clo;\0A\09madc.hi.cc.u32  r1, alo, blo, chi;\0A\09madc.hi.u32     r2, alo, bhi,   0;\0A\09mad.lo.cc.u32   r1, alo, bhi,  r1;\0A\09madc.hi.cc.u32  r2, ahi, blo,  r2;\0A\09madc.hi.u32     r3, ahi, bhi,   0;\0A\09mad.lo.cc.u32   r1, ahi, blo,  r1;\0A\09madc.lo.cc.u32  r2, ahi, bhi,  r2;\0A\09addc.u32        r3,  r3,   0;     \0A\09mov.b64         $0, {r0,r1};      \0A\09mov.b64         $1, {r2,r3};      \0A\09}", "=l,=l,l,l,l"(i64 %27, i64 %11, i64 %p.129.07) #7, !srcloc !8
  %29 = extractvalue { i64, i64 } %28, 0
  %30 = extractvalue { i64, i64 } %28, 1
  %31 = sext i32 %12 to i64
  %32 = sub i64 0, %31
  store i64 %29, ptr addrspace(5) %lsr.iv5, align 8
  %33 = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %lsr.iv, i32 1)
  %math = extractvalue { i32, i1 } %33, 0
  %ov = extractvalue { i32, i1 } %33, 1
  %scevgep6 = getelementptr i8, ptr addrspace(5) %lsr.iv5, i64 8
  %scevgep9 = getelementptr i8, ptr addrspace(1) %lsr.iv8, i64 8
  br i1 %ov, label %._crit_edge, label %.lr.ph, !llvm.loop !9

._crit_edge:                                      ; preds = %.lr.ph, %.._crit_edge_crit_edge
  %.pre-phi4 = phi i64 [ %.pre3, %.._crit_edge_crit_edge ], [ %32, %.lr.ph ]
  %p.129.0.lcssa = phi i64 [ 0, %.._crit_edge_crit_edge ], [ %30, %.lr.ph ]
  %q.0.lcssa = phi i32 [ %13, %.._crit_edge_crit_edge ], [ %15, %.lr.ph ]
  %34 = sext i32 %q.0.lcssa to i64
  %35 = sub i64 %34, %.pre-phi4
  %36 = getelementptr i64, ptr addrspace(5) %1, i64 %35
  %37 = getelementptr i8, ptr addrspace(5) %36, i64 -120
  store i64 %p.129.0.lcssa, ptr addrspace(5) %37, align 8
  %38 = and i32 %4, 63
  %39 = getelementptr inbounds i8, ptr addrspace(5) %1, i64 16
  %40 = load i64, ptr addrspace(5) %39, align 8
  %41 = getelementptr inbounds i8, ptr addrspace(5) %1, i64 24
  %42 = load i64, ptr addrspace(5) %41, align 8
  %.not = icmp eq i32 %38, 0
  br i1 %.not, label %55, label %43

43:                                               ; preds = %._crit_edge
  %44 = sub nuw nsw i32 64, %38
  %45 = zext nneg i32 %38 to i64
  %46 = shl i64 %42, %45
  %47 = zext nneg i32 %44 to i64
  %48 = lshr i64 %40, %47
  %49 = or i64 %46, %48
  %50 = shl i64 %40, %45
  %51 = getelementptr inbounds i8, ptr addrspace(5) %1, i64 8
  %52 = load i64, ptr addrspace(5) %51, align 8
  %53 = lshr i64 %52, %47
  %54 = or i64 %53, %50
  br label %55

55:                                               ; preds = %43, %._crit_edge
  %hi.0 = phi i64 [ %49, %43 ], [ %42, %._crit_edge ]
  %lo.0 = phi i64 [ %54, %43 ], [ %40, %._crit_edge ]
  %56 = lshr i64 %hi.0, 62
  %57 = trunc nuw nsw i64 %56 to i32
  %58 = tail call i64 @llvm.fshl.i64(i64 %hi.0, i64 %lo.0, i64 2)
  %59 = shl i64 %lo.0, 2
  %60 = lshr i64 %58, 63
  %61 = trunc nuw nsw i64 %60 to i32
  %62 = add nuw nsw i32 %61, %57
  %.not3 = icmp eq i32 %3, 0
  %63 = sub nsw i32 0, %62
  %spec.select = select i1 %.not3, i32 %62, i32 %63
  %.not4 = icmp sgt i64 %58, -1
  %64 = xor i32 %3, -2147483648
  br i1 %.not4, label %69, label %65

65:                                               ; preds = %55
  %66 = tail call { i64, i64 } asm "{\0A\09.reg .u32 r0, r1, r2, r3, a0, a1, a2, a3, b0, b1, b2, b3;\0A\09mov.b64         {a0,a1}, $2;\0A\09mov.b64         {a2,a3}, $3;\0A\09mov.b64         {b0,b1}, $4;\0A\09mov.b64         {b2,b3}, $5;\0A\09sub.cc.u32      r0, a0, b0; \0A\09subc.cc.u32     r1, a1, b1; \0A\09subc.cc.u32     r2, a2, b2; \0A\09subc.u32        r3, a3, b3; \0A\09mov.b64         $0, {r0,r1};\0A\09mov.b64         $1, {r2,r3};\0A\09}", "=l,=l,l,l,l,l"(i64 0, i64 0, i64 %59, i64 %58) #7, !srcloc !11
  %67 = extractvalue { i64, i64 } %66, 0
  %68 = extractvalue { i64, i64 } %66, 1
  br label %69

69:                                               ; preds = %65, %55
  %hi.1 = phi i64 [ %68, %65 ], [ %58, %55 ]
  %lo.1 = phi i64 [ %67, %65 ], [ %59, %55 ]
  %s.0 = phi i32 [ %64, %65 ], [ %3, %55 ]
  %ctlz = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %hi.1, i1 false)
  %70 = freeze i64 %lo.1
  %spec.select6 = tail call i64 @llvm.fshl.i64(i64 %hi.1, i64 %70, i64 %ctlz)
  %71 = tail call { i64, i64 } asm "{\0A\09.reg .u32 r0, r1, r2, r3, alo, ahi, blo, bhi;\0A\09mov.b64         {alo,ahi}, $2;   \0A\09mov.b64         {blo,bhi}, $3;   \0A\09mul.lo.u32      r0, alo, blo;    \0A\09mul.hi.u32      r1, alo, blo;    \0A\09mad.lo.cc.u32   r1, alo, bhi, r1;\0A\09madc.hi.u32     r2, alo, bhi,  0;\0A\09mad.lo.cc.u32   r1, ahi, blo, r1;\0A\09madc.hi.cc.u32  r2, ahi, blo, r2;\0A\09madc.hi.u32     r3, ahi, bhi,  0;\0A\09mad.lo.cc.u32   r2, ahi, bhi, r2;\0A\09addc.u32        r3, r3,  0;      \0A\09mov.b64         $0, {r0,r1};     \0A\09mov.b64         $1, {r2,r3};     \0A\09}", "=l,=l,l,l"(i64 %spec.select6, i64 -3958705157555305931) #7, !srcloc !12
  %72 = extractvalue { i64, i64 } %71, 1
  %73 = icmp sgt i64 %72, 0
  %74 = add nuw nsw i64 %ctlz, 1
  %75 = extractvalue { i64, i64 } %71, 0
  br i1 %73, label %76, label %79

76:                                               ; preds = %69
  %77 = tail call { i64, i64 } asm "{\0A\09.reg .u32 r0, r1, r2, r3, a0, a1, a2, a3, b0, b1, b2, b3;\0A\09mov.b64         {a0,a1}, $2;\0A\09mov.b64         {a2,a3}, $3;\0A\09mov.b64         {b0,b1}, $4;\0A\09mov.b64         {b2,b3}, $5;\0A\09add.cc.u32      r0, a0, b0; \0A\09addc.cc.u32     r1, a1, b1; \0A\09addc.cc.u32     r2, a2, b2; \0A\09addc.u32        r3, a3, b3; \0A\09mov.b64         $0, {r0,r1};\0A\09mov.b64         $1, {r2,r3};\0A\09}", "=l,=l,l,l,l,l"(i64 %75, i64 %72, i64 %75, i64 %72) #7, !srcloc !13
  %78 = extractvalue { i64, i64 } %77, 1
  br label %79

79:                                               ; preds = %76, %69
  %hi.3 = phi i64 [ %78, %76 ], [ %72, %69 ]
  %e.0 = phi i64 [ %74, %76 ], [ %ctlz, %69 ]
  %80 = zext i32 %s.0 to i64
  %81 = shl nuw i64 %80, 32
  %82 = add i64 %hi.3, 1
  %83 = lshr i64 %82, 10
  %84 = add nuw nsw i64 %83, 1
  %85 = lshr i64 %84, 1
  %86 = shl nuw nsw i64 %e.0, 52
  %reass.sub14 = sub nsw i64 %85, %86
  %87 = add nsw i64 %reass.sub14, 4602678819172646912
  %88 = or i64 %87, %81
  %89 = bitcast i64 %88 to double
  br label %90

90:                                               ; preds = %0, %79
  %.030.0 = phi double [ %89, %79 ], [ %a, %0 ]
  %.131.0 = phi i32 [ %spec.select, %79 ], [ 0, %0 ]
  %newret = insertvalue { double, i32 } poison, double %.030.0, 0
  %newret2 = insertvalue { double, i32 } %newret, i32 %.131.0, 1
  ret { double, i32 } %newret2
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.hi(double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.lo(double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.lohi.i2d(i32, i32) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.ctlz.i64(i64, i1 immarg) #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.add.rn.d(double, double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fabs.f32(float) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fma.f64(double, double, double) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.fshl.i64(i64, i64, i64) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #5

attributes #0 = { nofree nosync nounwind memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nofree noinline nosync nounwind memory(none) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nounwind }
attributes #7 = { nounwind memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 96}
!5 = !{i32 0, i32 128}
!6 = !{}
!7 = !{i32 0, i32 14880}
!8 = !{i32 161521, i32 161525, i32 161594, i32 161642, i32 161690, i32 161738, i32 161786, i32 161834, i32 161882, i32 161930, i32 161978, i32 162026, i32 162074, i32 162122, i32 162170, i32 162218, i32 162266}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.count", i32 1}
!11 = !{i32 159255, i32 159259, i32 159330, i32 159372, i32 159414, i32 159456, i32 159498, i32 159540, i32 159582, i32 159624, i32 159666, i32 159708, i32 159750}
!12 = !{i32 160296, i32 160300, i32 160359, i32 160406, i32 160453, i32 160500, i32 160547, i32 160594, i32 160641, i32 160688, i32 160735, i32 160782, i32 160829, i32 160876, i32 160923, i32 160970}
!13 = !{i32 158057, i32 158061, i32 158132, i32 158174, i32 158216, i32 158258, i32 158300, i32 158342, i32 158384, i32 158426, i32 158468, i32 158510, i32 158552}
