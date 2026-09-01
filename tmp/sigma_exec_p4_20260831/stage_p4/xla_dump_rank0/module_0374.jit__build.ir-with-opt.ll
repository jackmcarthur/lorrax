; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_90_0 = local_unnamed_addr addrspace(1) constant [64 x i8] zeroinitializer, align 256
@__cudart_i2opi_d = internal unnamed_addr addrspace(1) constant [18 x i64] [i64 7780917995555872008, i64 4397547296490951402, i64 8441921394348257659, i64 5712322887342352941, i64 7869616827067468215, i64 -1211730484530615009, i64 2303758334597371919, i64 -7168499653074671557, i64 4148332274289687028, i64 -1613291254968254911, i64 -1692731182770600828, i64 -135693905287338178, i64 452944820249399836, i64 -5249950069107600672, i64 -121206125134887583, i64 -2638381946312093631, i64 -277156292786332224, i64 -6703182060581546711], align 8
@__cudart_sin_cos_coeffs = internal unnamed_addr addrspace(1) constant [16 x double] [double 0xBE5AE5F12CB0D246, double 0x3EC71DE369ACE392, double 0xBF2A01A019DB62A1, double 0x3F81111111110818, double 0xBFC5555555555554, double 0.000000e+00, double 0.000000e+00, double 0xBDA8FF8320FD8164, double 0x3E21EEA7C1EF8528, double 0xBE927E4F8E06E6D9, double 0x3EFA01A019DDBCE9, double 0xBF56C16C16C15D47, double 0x3FA5555555555551, double -5.000000e-01, double 1.000000e+00, double 0.000000e+00], align 16

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define ptx_kernel void @loop_broadcast_fusion(ptr noalias writeonly align 256 captures(none) dereferenceable(787251200) %0) local_unnamed_addr #0 {
  %2 = addrspacecast ptr %0 to ptr addrspace(1)
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %5 = shl nuw nsw i32 %4, 2
  %6 = shl nuw nsw i32 %3, 9
  %7 = or disjoint i32 %5, %6
  %8 = zext nneg i32 %7 to i64
  %9 = getelementptr inbounds { double, double }, ptr addrspace(1) %2, i64 %8
  store <2 x double> zeroinitializer, ptr addrspace(1) %9, align 64
  %10 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 16
  store <2 x double> zeroinitializer, ptr addrspace(1) %10, align 16
  %11 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 32
  store <2 x double> zeroinitializer, ptr addrspace(1) %11, align 32
  %12 = getelementptr inbounds i8, ptr addrspace(1) %9, i64 48
  store <2 x double> zeroinitializer, ptr addrspace(1) %12, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_compare_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(8) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(1) initializes((0, 1)) %1) local_unnamed_addr #2 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = load i64, ptr addrspace(1) %3, align 256, !invariant.load !6
  %6 = icmp slt i64 %5, 4
  %7 = zext i1 %6 to i8
  store i8 %7, ptr addrspace(1) %4, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(16) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(16) initializes((0, 16)) %1) local_unnamed_addr #2 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = load <2 x double>, ptr addrspace(1) %3, align 16, !invariant.load !6
  %.unpack5 = extractelement <2 x double> %5, i32 0
  %.unpack26 = extractelement <2 x double> %5, i32 1
  %6 = fmul double %.unpack5, 0.000000e+00
  %7 = fsub double %.unpack26, %6
  %8 = fmul double %.unpack26, -0.000000e+00
  %9 = fsub double %8, %.unpack5
  %10 = insertelement <2 x double> poison, double %7, i32 0
  %11 = insertelement <2 x double> %10, double %9, i32 1
  store <2 x double> %11, ptr addrspace(1) %4, align 256
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_slice_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(192) %0, ptr noalias readonly align 256 captures(none) dereferenceable(8) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %3, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %4, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %5, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %6, ptr noalias writeonly align 256 captures(none) dereferenceable(8) initializes((0, 8)) %7) local_unnamed_addr #2 {
  %9 = addrspacecast ptr %1 to ptr addrspace(1)
  %10 = addrspacecast ptr %0 to ptr addrspace(1)
  %11 = addrspacecast ptr %2 to ptr addrspace(1)
  %12 = addrspacecast ptr %3 to ptr addrspace(1)
  %13 = addrspacecast ptr %4 to ptr addrspace(1)
  %14 = addrspacecast ptr %5 to ptr addrspace(1)
  %15 = addrspacecast ptr %6 to ptr addrspace(1)
  %16 = addrspacecast ptr %7 to ptr addrspace(1)
  %.val5 = load i64, ptr addrspace(1) %9, align 256, !invariant.load !6
  %17 = lshr i64 %.val5, 61
  %18 = and i64 %17, 4
  %19 = add i64 %18, %.val5
  %20 = tail call i64 @llvm.smax.i64(i64 %19, i64 0)
  %21 = tail call i64 @llvm.umin.i64(i64 %20, i64 3)
  %.idx.i = mul nuw nsw i64 %21, 48
  %22 = getelementptr inbounds i8, ptr addrspace(1) %10, i64 %.idx.i
  %23 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 32
  %24 = load <2 x double>, ptr addrspace(1) %23, align 16, !invariant.load !6
  %25 = extractelement <2 x double> %24, i32 0
  %26 = extractelement <2 x double> %24, i32 1
  %27 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !6
  %28 = extractelement <2 x double> %27, i32 0
  %29 = extractelement <2 x double> %27, i32 1
  %30 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 16
  %31 = load <2 x double>, ptr addrspace(1) %30, align 16, !invariant.load !6
  %32 = extractelement <2 x double> %31, i32 0
  %33 = extractelement <2 x double> %31, i32 1
  store double %25, ptr addrspace(1) %11, align 256
  store double %28, ptr addrspace(1) %12, align 256
  store double %26, ptr addrspace(1) %13, align 256
  store double %29, ptr addrspace(1) %14, align 256
  store double %32, ptr addrspace(1) %15, align 256
  store double %33, ptr addrspace(1) %16, align 256
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #3

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define ptx_kernel void @loop_add_fusion(ptr noalias align 256 captures(none) dereferenceable(787251200) %0, ptr noalias readonly align 16 captures(none) dereferenceable(3149004800) %1, ptr noalias readonly align 256 captures(none) dereferenceable(16) %2, ptr noalias readonly align 16 captures(none) dereferenceable(3149004800) %3, ptr noalias readonly align 256 captures(none) dereferenceable(8) %4, ptr noalias readonly align 256 captures(none) dereferenceable(8) %5, ptr noalias readonly align 256 captures(none) dereferenceable(8) %6, ptr noalias readonly align 256 captures(none) dereferenceable(8) %7, ptr noalias readonly align 256 captures(none) dereferenceable(8) %8, ptr noalias readonly align 256 captures(none) dereferenceable(8) %9, ptr noalias readonly align 256 captures(none) dereferenceable(1) %10, ptr noalias readonly align 16 captures(none) dereferenceable(8) %11, ptr noalias readonly align 16 captures(none) dereferenceable(16) %12, ptr noalias readonly align 256 captures(none) dereferenceable(8) %13, ptr noalias readnone align 256 captures(none) dereferenceable(787251200) %14) local_unnamed_addr #4 {
  %16 = addrspacecast ptr %10 to ptr addrspace(1)
  %17 = addrspacecast ptr %13 to ptr addrspace(1)
  %18 = addrspacecast ptr %12 to ptr addrspace(1)
  %19 = addrspacecast ptr %11 to ptr addrspace(1)
  %20 = addrspacecast ptr %9 to ptr addrspace(1)
  %21 = addrspacecast ptr %8 to ptr addrspace(1)
  %22 = addrspacecast ptr %2 to ptr addrspace(1)
  %23 = addrspacecast ptr %7 to ptr addrspace(1)
  %24 = addrspacecast ptr %6 to ptr addrspace(1)
  %25 = addrspacecast ptr %5 to ptr addrspace(1)
  %26 = addrspacecast ptr %4 to ptr addrspace(1)
  %27 = addrspacecast ptr %3 to ptr addrspace(1)
  %28 = addrspacecast ptr %1 to ptr addrspace(1)
  %29 = addrspacecast ptr %0 to ptr addrspace(1)
  %30 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %31 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %32 = load i8, ptr addrspace(1) %16, align 256, !invariant.load !6
  %33 = load i64, ptr addrspace(1) %17, align 256, !invariant.load !6
  %34 = lshr i64 %33, 61
  %35 = and i64 %34, 4
  %36 = add i64 %35, %33
  %37 = tail call i64 @llvm.smax.i64(i64 %36, i64 0)
  %38 = tail call i64 @llvm.umin.i64(i64 %37, i64 3)
  %39 = getelementptr inbounds i32, ptr addrspace(1) %18, i64 %38
  %40 = load i32, ptr addrspace(1) %39, align 4, !invariant.load !6
  %41 = lshr i32 %40, 29
  %42 = and i32 %41, 4
  %43 = add i32 %42, %40
  %44 = tail call i32 @llvm.smax.i32(i32 %43, i32 0)
  %45 = tail call i32 @llvm.umin.i32(i32 %44, i32 3)
  %46 = trunc i8 %32 to i1
  %47 = load double, ptr addrspace(1) %19, align 16, !invariant.load !6
  %48 = load double, ptr addrspace(1) %20, align 256, !invariant.load !6
  %49 = load double, ptr addrspace(1) %21, align 256, !invariant.load !6
  %50 = load <2 x double>, ptr addrspace(1) %22, align 256, !invariant.load !6
  %.unpack206 = extractelement <2 x double> %50, i32 0
  %.unpack2207 = extractelement <2 x double> %50, i32 1
  %51 = load double, ptr addrspace(1) %23, align 256, !invariant.load !6
  %52 = load double, ptr addrspace(1) %24, align 256, !invariant.load !6
  %53 = load double, ptr addrspace(1) %25, align 256, !invariant.load !6
  %54 = load double, ptr addrspace(1) %26, align 256, !invariant.load !6
  %narrow = mul nuw nsw i32 %45, 49203200
  %55 = shl nuw nsw i32 %31, 2
  %56 = shl nuw nsw i32 %30, 9
  %57 = or disjoint i32 %56, %55
  %narrow3 = add nuw nsw i32 %57, %narrow
  %58 = zext nneg i32 %narrow3 to i64
  %59 = getelementptr inbounds { double, double }, ptr addrspace(1) %27, i64 %58
  %60 = load <2 x double>, ptr addrspace(1) %59, align 16, !invariant.load !6
  %.unpack4208 = extractelement <2 x double> %60, i32 0
  %.unpack6209 = extractelement <2 x double> %60, i32 1
  %61 = fsub double %.unpack4208, %47
  %62 = select i1 %46, double 0.000000e+00, double %.unpack6209
  %63 = fmul double %.unpack206, %61
  %64 = fmul double %.unpack2207, %62
  %65 = fsub double %63, %64
  %66 = fmul double %.unpack206, %62
  %67 = fmul double %.unpack2207, %61
  %68 = fadd double %67, %66
  %69 = fmul double %65, 5.000000e-01
  %70 = tail call double @llvm.fma.f64(double %69, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %71 = tail call i32 @llvm.nvvm.d2i.lo(double %70) #7
  %72 = tail call double @llvm.nvvm.add.rn.d(double %70, double 0xC338000000000000) #7
  %73 = tail call double @llvm.fma.f64(double %72, double 0xBFE62E42FEFA39EF, double %69)
  %74 = tail call double @llvm.fma.f64(double %72, double 0xBC7ABC9E3B39803F, double %73)
  %75 = tail call double @llvm.fma.f64(double %74, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %76 = tail call double @llvm.fma.f64(double %75, double %74, double 0x3EC71DEE62401315)
  %77 = tail call double @llvm.fma.f64(double %76, double %74, double 0x3EFA01997C89EB71)
  %78 = tail call double @llvm.fma.f64(double %77, double %74, double 0x3F2A01A014761F65)
  %79 = tail call double @llvm.fma.f64(double %78, double %74, double 0x3F56C16C1852B7AF)
  %80 = tail call double @llvm.fma.f64(double %79, double %74, double 0x3F81111111122322)
  %81 = tail call double @llvm.fma.f64(double %80, double %74, double 0x3FA55555555502A1)
  %82 = tail call double @llvm.fma.f64(double %81, double %74, double 0x3FC5555555555511)
  %83 = tail call double @llvm.fma.f64(double %82, double %74, double 0x3FE000000000000B)
  %84 = tail call double @llvm.fma.f64(double %83, double %74, double 1.000000e+00)
  %85 = tail call double @llvm.fma.f64(double %84, double %74, double 1.000000e+00)
  %86 = tail call i32 @llvm.nvvm.d2i.lo(double %85) #7
  %87 = tail call i32 @llvm.nvvm.d2i.hi(double %85) #7
  %88 = shl i32 %71, 20
  %89 = add i32 %87, %88
  %90 = tail call double @llvm.nvvm.lohi.i2d(i32 %86, i32 %89) #7
  %91 = tail call i32 @llvm.nvvm.d2i.hi(double %69) #7
  %92 = bitcast i32 %91 to float
  %93 = tail call float @llvm.nvvm.fabs.f32(float %92)
  %94 = fcmp olt float %93, 0x4010C46560000000
  br i1 %94, label %__nv_exp.exit, label %__internal_fast_icmp_abs_lt.exit.i

__internal_fast_icmp_abs_lt.exit.i:               ; preds = %15
  %95 = fcmp olt double %69, 0.000000e+00
  %96 = fadd double %69, 0x7FF0000000000000
  %z.0.i105 = select i1 %95, double 0.000000e+00, double %96
  %97 = fcmp olt float %93, 0x4010E90000000000
  br i1 %97, label %98, label %__nv_exp.exit

98:                                               ; preds = %__internal_fast_icmp_abs_lt.exit.i
  %99 = sdiv i32 %71, 2
  %100 = shl i32 %99, 20
  %101 = add i32 %87, %100
  %102 = tail call double @llvm.nvvm.lohi.i2d(i32 %86, i32 %101) #7
  %103 = sub nsw i32 %71, %99
  %104 = shl i32 %103, 20
  %105 = add nsw i32 %104, 1072693248
  %106 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %105) #7
  %107 = fmul double %106, %102
  br label %__nv_exp.exit

__nv_exp.exit:                                    ; preds = %15, %__internal_fast_icmp_abs_lt.exit.i, %98
  %z.2.i = phi double [ %90, %15 ], [ %107, %98 ], [ %z.0.i105, %__internal_fast_icmp_abs_lt.exit.i ]
  %108 = tail call double @llvm.nvvm.fabs.f64(double %68)
  %109 = fcmp oeq double %108, 0x7FF0000000000000
  br i1 %109, label %110, label %112

110:                                              ; preds = %__nv_exp.exit
  %111 = tail call double @llvm.nvvm.mul.rn.d(double %68, double 0.000000e+00) #7
  br label %__nv_sin.exit

112:                                              ; preds = %__nv_exp.exit
  %113 = fmul double %68, 0x3FE45F306DC9C883
  %114 = tail call i32 @llvm.nvvm.d2i.rn(double %113) #7
  %115 = sitofp i32 %114 to double
  %116 = fneg double %115
  %117 = tail call double @llvm.fma.f64(double %116, double 0x3FF921FB54442D18, double %68)
  %118 = tail call double @llvm.fma.f64(double %116, double 0x3C91A62633145C00, double %117)
  %119 = tail call double @llvm.fma.f64(double %116, double 0x397B839A252049C0, double %118)
  %120 = fcmp ult double %108, 0x41E0000000000000
  br i1 %120, label %__nv_sin.exit, label %121

121:                                              ; preds = %112
  %122 = tail call fastcc { double, i32 } @__internal_trig_reduction_slowpathd(double %68) #7
  %newret174 = extractvalue { double, i32 } %122, 0
  %newret176 = extractvalue { double, i32 } %122, 1
  br label %__nv_sin.exit

__nv_sin.exit:                                    ; preds = %110, %112, %121
  %z.0.i = phi double [ %111, %110 ], [ %newret174, %121 ], [ %119, %112 ]
  %i.0.i = phi i32 [ 0, %110 ], [ %newret176, %121 ], [ %114, %112 ]
  %123 = fcmp oeq double %108, 0x7FF0000000000000
  %124 = fcmp ole double %.unpack4208, %49
  %125 = fcmp ogt double %.unpack4208, %48
  %126 = and i32 %i.0.i, 1
  %127 = shl nuw nsw i32 %126, 3
  %128 = zext nneg i32 %127 to i64
  %129 = getelementptr inbounds nuw double, ptr addrspace(1) @__cudart_sin_cos_coeffs, i64 %128
  %130 = tail call double @llvm.nvvm.mul.rn.d(double %z.0.i, double %z.0.i) #7
  %.not.i = icmp eq i32 %126, 0
  %131 = select i1 %.not.i, double 0x3DE5DB65F9785EBA, double 0xBDA8FF8320FD8164
  %132 = load <2 x double>, ptr addrspace(1) %129, align 16, !invariant.load !6
  %133 = extractelement <2 x double> %132, i32 0
  %134 = extractelement <2 x double> %132, i32 1
  %135 = tail call double @llvm.fma.f64(double %131, double %130, double %133)
  %136 = tail call double @llvm.fma.f64(double %135, double %130, double %134)
  %137 = getelementptr inbounds nuw i8, ptr addrspace(1) %129, i64 16
  %138 = load <2 x double>, ptr addrspace(1) %137, align 16, !invariant.load !6
  %139 = extractelement <2 x double> %138, i32 0
  %140 = extractelement <2 x double> %138, i32 1
  %141 = tail call double @llvm.fma.f64(double %136, double %130, double %139)
  %142 = tail call double @llvm.fma.f64(double %141, double %130, double %140)
  %143 = getelementptr inbounds nuw i8, ptr addrspace(1) %129, i64 32
  %144 = load <2 x double>, ptr addrspace(1) %143, align 16, !invariant.load !6
  %145 = extractelement <2 x double> %144, i32 0
  %146 = extractelement <2 x double> %144, i32 1
  %147 = tail call double @llvm.fma.f64(double %142, double %130, double %145)
  %148 = tail call double @llvm.fma.f64(double %147, double %130, double %146)
  %149 = tail call double @llvm.fma.f64(double %148, double %z.0.i, double %z.0.i)
  %150 = tail call double @llvm.fma.f64(double %148, double %130, double 1.000000e+00)
  %spec.select.i = select i1 %.not.i, double %149, double %150
  %151 = and i32 %i.0.i, 2
  %.not1.i = icmp eq i32 %151, 0
  %152 = fsub double 0.000000e+00, %spec.select.i
  %.1.i = select i1 %.not1.i, double %spec.select.i, double %152
  %153 = fmul double %z.2.i, %.1.i
  %154 = and i1 %125, %124
  %155 = fneg double %.unpack6209
  br i1 %123, label %156, label %158

156:                                              ; preds = %__nv_sin.exit
  %157 = tail call double @llvm.nvvm.mul.rn.d(double %68, double 0.000000e+00) #7
  br label %__nv_cos.exit

158:                                              ; preds = %__nv_sin.exit
  %159 = fmul double %68, 0x3FE45F306DC9C883
  %160 = tail call i32 @llvm.nvvm.d2i.rn(double %159) #7
  %161 = sitofp i32 %160 to double
  %162 = fneg double %161
  %163 = tail call double @llvm.fma.f64(double %162, double 0x3FF921FB54442D18, double %68)
  %164 = tail call double @llvm.fma.f64(double %162, double 0x3C91A62633145C00, double %163)
  %165 = tail call double @llvm.fma.f64(double %162, double 0x397B839A252049C0, double %164)
  %166 = fcmp ult double %108, 0x41E0000000000000
  br i1 %166, label %__internal_trig_reduction_kerneld.exit.i, label %167

167:                                              ; preds = %158
  %168 = tail call fastcc { double, i32 } @__internal_trig_reduction_slowpathd(double %68) #7
  %newret158 = extractvalue { double, i32 } %168, 0
  %newret160 = extractvalue { double, i32 } %168, 1
  br label %__internal_trig_reduction_kerneld.exit.i

__internal_trig_reduction_kerneld.exit.i:         ; preds = %167, %158
  %t.i1.0.i = phi double [ %newret158, %167 ], [ %165, %158 ]
  %q.i.0.i = phi i32 [ %newret160, %167 ], [ %160, %158 ]
  %169 = add nsw i32 %q.i.0.i, 1
  br label %__nv_cos.exit

__nv_cos.exit:                                    ; preds = %156, %__internal_trig_reduction_kerneld.exit.i
  %z.0.i69 = phi double [ %157, %156 ], [ %t.i1.0.i, %__internal_trig_reduction_kerneld.exit.i ]
  %i.0.i70 = phi i32 [ 1, %156 ], [ %169, %__internal_trig_reduction_kerneld.exit.i ]
  %170 = and i32 %i.0.i70, 1
  %171 = shl nuw nsw i32 %170, 3
  %172 = zext nneg i32 %171 to i64
  %173 = getelementptr inbounds nuw double, ptr addrspace(1) @__cudart_sin_cos_coeffs, i64 %172
  %174 = tail call double @llvm.nvvm.mul.rn.d(double %z.0.i69, double %z.0.i69) #7
  %.not.i71 = icmp eq i32 %170, 0
  %175 = select i1 %.not.i71, double 0x3DE5DB65F9785EBA, double 0xBDA8FF8320FD8164
  %176 = load <2 x double>, ptr addrspace(1) %173, align 16, !invariant.load !6
  %177 = extractelement <2 x double> %176, i32 0
  %178 = extractelement <2 x double> %176, i32 1
  %179 = tail call double @llvm.fma.f64(double %175, double %174, double %177)
  %180 = tail call double @llvm.fma.f64(double %179, double %174, double %178)
  %181 = getelementptr inbounds nuw i8, ptr addrspace(1) %173, i64 16
  %182 = load <2 x double>, ptr addrspace(1) %181, align 16, !invariant.load !6
  %183 = extractelement <2 x double> %182, i32 0
  %184 = extractelement <2 x double> %182, i32 1
  %185 = tail call double @llvm.fma.f64(double %180, double %174, double %183)
  %186 = tail call double @llvm.fma.f64(double %185, double %174, double %184)
  %187 = getelementptr inbounds nuw i8, ptr addrspace(1) %173, i64 32
  %188 = load <2 x double>, ptr addrspace(1) %187, align 16, !invariant.load !6
  %189 = extractelement <2 x double> %188, i32 0
  %190 = extractelement <2 x double> %188, i32 1
  %191 = tail call double @llvm.fma.f64(double %186, double %174, double %189)
  %192 = tail call double @llvm.fma.f64(double %191, double %174, double %190)
  %193 = tail call double @llvm.fma.f64(double %192, double %z.0.i69, double %z.0.i69)
  %194 = tail call double @llvm.fma.f64(double %192, double %174, double 1.000000e+00)
  %spec.select.i72 = select i1 %.not.i71, double %193, double %194
  %195 = and i32 %i.0.i70, 2
  %.not1.i73 = icmp eq i32 %195, 0
  %196 = fsub double 0.000000e+00, %spec.select.i72
  %.1.i74 = select i1 %.not1.i73, double %spec.select.i72, double %196
  %197 = fmul double %z.2.i, %.1.i74
  %198 = tail call double @llvm.fma.f64(double %65, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %199 = tail call i32 @llvm.nvvm.d2i.lo(double %198) #7
  %200 = tail call double @llvm.nvvm.add.rn.d(double %198, double 0xC338000000000000) #7
  %201 = tail call double @llvm.fma.f64(double %200, double 0xBFE62E42FEFA39EF, double %65)
  %202 = tail call double @llvm.fma.f64(double %200, double 0xBC7ABC9E3B39803F, double %201)
  %203 = tail call double @llvm.fma.f64(double %202, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %204 = tail call double @llvm.fma.f64(double %203, double %202, double 0x3EC71DEE62401315)
  %205 = tail call double @llvm.fma.f64(double %204, double %202, double 0x3EFA01997C89EB71)
  %206 = tail call double @llvm.fma.f64(double %205, double %202, double 0x3F2A01A014761F65)
  %207 = tail call double @llvm.fma.f64(double %206, double %202, double 0x3F56C16C1852B7AF)
  %208 = tail call double @llvm.fma.f64(double %207, double %202, double 0x3F81111111122322)
  %209 = tail call double @llvm.fma.f64(double %208, double %202, double 0x3FA55555555502A1)
  %210 = tail call double @llvm.fma.f64(double %209, double %202, double 0x3FC5555555555511)
  %211 = tail call double @llvm.fma.f64(double %210, double %202, double 0x3FE000000000000B)
  %212 = tail call double @llvm.fma.f64(double %211, double %202, double 1.000000e+00)
  %213 = tail call double @llvm.fma.f64(double %212, double %202, double 1.000000e+00)
  %214 = tail call i32 @llvm.nvvm.d2i.lo(double %213) #7
  %215 = tail call i32 @llvm.nvvm.d2i.hi(double %213) #7
  %216 = shl i32 %199, 20
  %217 = add i32 %215, %216
  %218 = tail call double @llvm.nvvm.lohi.i2d(i32 %214, i32 %217) #7
  %219 = tail call i32 @llvm.nvvm.d2i.hi(double %65) #7
  %220 = bitcast i32 %219 to float
  %221 = tail call float @llvm.nvvm.fabs.f32(float %220)
  %222 = fcmp olt float %221, 0x4010C46560000000
  br i1 %222, label %__nv_exp.exit109, label %__internal_fast_icmp_abs_lt.exit.i106

__internal_fast_icmp_abs_lt.exit.i106:            ; preds = %__nv_cos.exit
  %223 = fcmp olt double %65, 0.000000e+00
  %224 = fadd double %65, 0x7FF0000000000000
  %z.0.i107 = select i1 %223, double 0.000000e+00, double %224
  %225 = fcmp olt float %221, 0x4010E90000000000
  br i1 %225, label %226, label %__nv_exp.exit109

226:                                              ; preds = %__internal_fast_icmp_abs_lt.exit.i106
  %227 = sdiv i32 %199, 2
  %228 = shl i32 %227, 20
  %229 = add i32 %215, %228
  %230 = tail call double @llvm.nvvm.lohi.i2d(i32 %214, i32 %229) #7
  %231 = sub nsw i32 %199, %227
  %232 = shl i32 %231, 20
  %233 = add nsw i32 %232, 1072693248
  %234 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %233) #7
  %235 = fmul double %234, %230
  br label %__nv_exp.exit109

__nv_exp.exit109:                                 ; preds = %__nv_cos.exit, %__internal_fast_icmp_abs_lt.exit.i106, %226
  %z.2.i108 = phi double [ %218, %__nv_cos.exit ], [ %235, %226 ], [ %z.0.i107, %__internal_fast_icmp_abs_lt.exit.i106 ]
  %236 = fcmp ole double %51, %155
  %237 = fmul double %z.2.i, %153
  %238 = fmul double %.1.i, %z.2.i108
  %239 = and i1 %154, %236
  %240 = fcmp olt double %52, %155
  %241 = fcmp oeq double %z.2.i108, 0x7FF0000000000000
  %242 = fmul double %z.2.i, %197
  %243 = fmul double %.1.i74, %z.2.i108
  %244 = fcmp oeq double %68, 0.000000e+00
  %245 = select i1 %241, double %237, double %238
  %246 = and i1 %240, %239
  %247 = fcmp ogt double %53, %155
  %248 = select i1 %241, double %242, double %243
  %249 = select i1 %244, double 0.000000e+00, double %245
  %250 = and i1 %247, %246
  %251 = fcmp oge double %54, %155
  %252 = getelementptr inbounds { double, double }, ptr addrspace(1) %28, i64 %58
  %253 = load <2 x double>, ptr addrspace(1) %252, align 16, !invariant.load !6
  %.unpack7200 = extractelement <2 x double> %253, i32 0
  %.unpack9201 = extractelement <2 x double> %253, i32 1
  %254 = and i1 %251, %250
  %255 = fmul double %.unpack7200, %248
  %256 = fmul double %.unpack9201, %249
  %257 = fsub double %255, %256
  %258 = fmul double %.unpack9201, %248
  %259 = fmul double %.unpack7200, %249
  %260 = fadd double %258, %259
  %261 = zext nneg i32 %57 to i64
  %262 = getelementptr inbounds { double, double }, ptr addrspace(1) %29, i64 %261
  %263 = load <2 x double>, ptr addrspace(1) %262, align 64
  %.unpack10202 = extractelement <2 x double> %263, i32 0
  %.unpack12203 = extractelement <2 x double> %263, i32 1
  %264 = select i1 %254, double %257, double 0.000000e+00
  %265 = fadd double %.unpack10202, %264
  %266 = select i1 %254, double %260, double 0.000000e+00
  %267 = fadd double %.unpack12203, %266
  %268 = insertelement <2 x double> poison, double %265, i32 0
  %269 = insertelement <2 x double> %268, double %267, i32 1
  store <2 x double> %269, ptr addrspace(1) %262, align 64
  %270 = getelementptr inbounds i8, ptr addrspace(1) %59, i64 16
  %271 = load <2 x double>, ptr addrspace(1) %270, align 16, !invariant.load !6
  %.unpack15204 = extractelement <2 x double> %271, i32 0
  %.unpack17205 = extractelement <2 x double> %271, i32 1
  %272 = fsub double %.unpack15204, %47
  %273 = select i1 %46, double 0.000000e+00, double %.unpack17205
  %274 = fmul double %.unpack206, %272
  %275 = fmul double %.unpack2207, %273
  %276 = fsub double %274, %275
  %277 = fmul double %.unpack206, %273
  %278 = fmul double %.unpack2207, %272
  %279 = fadd double %278, %277
  %280 = fmul double %276, 5.000000e-01
  %281 = tail call double @llvm.fma.f64(double %280, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %282 = tail call i32 @llvm.nvvm.d2i.lo(double %281) #7
  %283 = tail call double @llvm.nvvm.add.rn.d(double %281, double 0xC338000000000000) #7
  %284 = tail call double @llvm.fma.f64(double %283, double 0xBFE62E42FEFA39EF, double %280)
  %285 = tail call double @llvm.fma.f64(double %283, double 0xBC7ABC9E3B39803F, double %284)
  %286 = tail call double @llvm.fma.f64(double %285, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %287 = tail call double @llvm.fma.f64(double %286, double %285, double 0x3EC71DEE62401315)
  %288 = tail call double @llvm.fma.f64(double %287, double %285, double 0x3EFA01997C89EB71)
  %289 = tail call double @llvm.fma.f64(double %288, double %285, double 0x3F2A01A014761F65)
  %290 = tail call double @llvm.fma.f64(double %289, double %285, double 0x3F56C16C1852B7AF)
  %291 = tail call double @llvm.fma.f64(double %290, double %285, double 0x3F81111111122322)
  %292 = tail call double @llvm.fma.f64(double %291, double %285, double 0x3FA55555555502A1)
  %293 = tail call double @llvm.fma.f64(double %292, double %285, double 0x3FC5555555555511)
  %294 = tail call double @llvm.fma.f64(double %293, double %285, double 0x3FE000000000000B)
  %295 = tail call double @llvm.fma.f64(double %294, double %285, double 1.000000e+00)
  %296 = tail call double @llvm.fma.f64(double %295, double %285, double 1.000000e+00)
  %297 = tail call i32 @llvm.nvvm.d2i.lo(double %296) #7
  %298 = tail call i32 @llvm.nvvm.d2i.hi(double %296) #7
  %299 = shl i32 %282, 20
  %300 = add i32 %298, %299
  %301 = tail call double @llvm.nvvm.lohi.i2d(i32 %297, i32 %300) #7
  %302 = tail call i32 @llvm.nvvm.d2i.hi(double %280) #7
  %303 = bitcast i32 %302 to float
  %304 = tail call float @llvm.nvvm.fabs.f32(float %303)
  %305 = fcmp olt float %304, 0x4010C46560000000
  br i1 %305, label %__nv_exp.exit113, label %__internal_fast_icmp_abs_lt.exit.i110

__internal_fast_icmp_abs_lt.exit.i110:            ; preds = %__nv_exp.exit109
  %306 = fcmp olt double %280, 0.000000e+00
  %307 = fadd double %280, 0x7FF0000000000000
  %z.0.i111 = select i1 %306, double 0.000000e+00, double %307
  %308 = fcmp olt float %304, 0x4010E90000000000
  br i1 %308, label %309, label %__nv_exp.exit113

309:                                              ; preds = %__internal_fast_icmp_abs_lt.exit.i110
  %310 = sdiv i32 %282, 2
  %311 = shl i32 %310, 20
  %312 = add i32 %298, %311
  %313 = tail call double @llvm.nvvm.lohi.i2d(i32 %297, i32 %312) #7
  %314 = sub nsw i32 %282, %310
  %315 = shl i32 %314, 20
  %316 = add nsw i32 %315, 1072693248
  %317 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %316) #7
  %318 = fmul double %317, %313
  br label %__nv_exp.exit113

__nv_exp.exit113:                                 ; preds = %__nv_exp.exit109, %__internal_fast_icmp_abs_lt.exit.i110, %309
  %z.2.i112 = phi double [ %301, %__nv_exp.exit109 ], [ %318, %309 ], [ %z.0.i111, %__internal_fast_icmp_abs_lt.exit.i110 ]
  %319 = tail call double @llvm.nvvm.fabs.f64(double %279)
  %320 = fcmp oeq double %319, 0x7FF0000000000000
  br i1 %320, label %321, label %323

321:                                              ; preds = %__nv_exp.exit113
  %322 = tail call double @llvm.nvvm.mul.rn.d(double %279, double 0.000000e+00) #7
  br label %__nv_sin.exit54

323:                                              ; preds = %__nv_exp.exit113
  %324 = fmul double %279, 0x3FE45F306DC9C883
  %325 = tail call i32 @llvm.nvvm.d2i.rn(double %324) #7
  %326 = sitofp i32 %325 to double
  %327 = fneg double %326
  %328 = tail call double @llvm.fma.f64(double %327, double 0x3FF921FB54442D18, double %279)
  %329 = tail call double @llvm.fma.f64(double %327, double 0x3C91A62633145C00, double %328)
  %330 = tail call double @llvm.fma.f64(double %327, double 0x397B839A252049C0, double %329)
  %331 = fcmp ult double %319, 0x41E0000000000000
  br i1 %331, label %__nv_sin.exit54, label %332

332:                                              ; preds = %323
  %333 = tail call fastcc { double, i32 } @__internal_trig_reduction_slowpathd(double %279) #7
  %newret170 = extractvalue { double, i32 } %333, 0
  %newret172 = extractvalue { double, i32 } %333, 1
  br label %__nv_sin.exit54

__nv_sin.exit54:                                  ; preds = %321, %323, %332
  %z.0.i48 = phi double [ %322, %321 ], [ %newret170, %332 ], [ %330, %323 ]
  %i.0.i49 = phi i32 [ 0, %321 ], [ %newret172, %332 ], [ %325, %323 ]
  %334 = fcmp oeq double %319, 0x7FF0000000000000
  %335 = fcmp ole double %.unpack15204, %49
  %336 = fcmp ogt double %.unpack15204, %48
  %337 = and i32 %i.0.i49, 1
  %338 = shl nuw nsw i32 %337, 3
  %339 = zext nneg i32 %338 to i64
  %340 = getelementptr inbounds nuw double, ptr addrspace(1) @__cudart_sin_cos_coeffs, i64 %339
  %341 = tail call double @llvm.nvvm.mul.rn.d(double %z.0.i48, double %z.0.i48) #7
  %.not.i50 = icmp eq i32 %337, 0
  %342 = select i1 %.not.i50, double 0x3DE5DB65F9785EBA, double 0xBDA8FF8320FD8164
  %343 = load <2 x double>, ptr addrspace(1) %340, align 16, !invariant.load !6
  %344 = extractelement <2 x double> %343, i32 0
  %345 = extractelement <2 x double> %343, i32 1
  %346 = tail call double @llvm.fma.f64(double %342, double %341, double %344)
  %347 = tail call double @llvm.fma.f64(double %346, double %341, double %345)
  %348 = getelementptr inbounds nuw i8, ptr addrspace(1) %340, i64 16
  %349 = load <2 x double>, ptr addrspace(1) %348, align 16, !invariant.load !6
  %350 = extractelement <2 x double> %349, i32 0
  %351 = extractelement <2 x double> %349, i32 1
  %352 = tail call double @llvm.fma.f64(double %347, double %341, double %350)
  %353 = tail call double @llvm.fma.f64(double %352, double %341, double %351)
  %354 = getelementptr inbounds nuw i8, ptr addrspace(1) %340, i64 32
  %355 = load <2 x double>, ptr addrspace(1) %354, align 16, !invariant.load !6
  %356 = extractelement <2 x double> %355, i32 0
  %357 = extractelement <2 x double> %355, i32 1
  %358 = tail call double @llvm.fma.f64(double %353, double %341, double %356)
  %359 = tail call double @llvm.fma.f64(double %358, double %341, double %357)
  %360 = tail call double @llvm.fma.f64(double %359, double %z.0.i48, double %z.0.i48)
  %361 = tail call double @llvm.fma.f64(double %359, double %341, double 1.000000e+00)
  %spec.select.i51 = select i1 %.not.i50, double %360, double %361
  %362 = and i32 %i.0.i49, 2
  %.not1.i52 = icmp eq i32 %362, 0
  %363 = fsub double 0.000000e+00, %spec.select.i51
  %.1.i53 = select i1 %.not1.i52, double %spec.select.i51, double %363
  %364 = fmul double %z.2.i112, %.1.i53
  %365 = and i1 %336, %335
  %366 = fneg double %.unpack17205
  br i1 %334, label %367, label %369

367:                                              ; preds = %__nv_sin.exit54
  %368 = tail call double @llvm.nvvm.mul.rn.d(double %279, double 0.000000e+00) #7
  br label %__nv_cos.exit84

369:                                              ; preds = %__nv_sin.exit54
  %370 = fmul double %279, 0x3FE45F306DC9C883
  %371 = tail call i32 @llvm.nvvm.d2i.rn(double %370) #7
  %372 = sitofp i32 %371 to double
  %373 = fneg double %372
  %374 = tail call double @llvm.fma.f64(double %373, double 0x3FF921FB54442D18, double %279)
  %375 = tail call double @llvm.fma.f64(double %373, double 0x3C91A62633145C00, double %374)
  %376 = tail call double @llvm.fma.f64(double %373, double 0x397B839A252049C0, double %375)
  %377 = fcmp ult double %319, 0x41E0000000000000
  br i1 %377, label %__internal_trig_reduction_kerneld.exit.i75, label %378

378:                                              ; preds = %369
  %379 = tail call fastcc { double, i32 } @__internal_trig_reduction_slowpathd(double %279) #7
  %newret154 = extractvalue { double, i32 } %379, 0
  %newret156 = extractvalue { double, i32 } %379, 1
  br label %__internal_trig_reduction_kerneld.exit.i75

__internal_trig_reduction_kerneld.exit.i75:       ; preds = %378, %369
  %t.i1.0.i76 = phi double [ %newret154, %378 ], [ %376, %369 ]
  %q.i.0.i77 = phi i32 [ %newret156, %378 ], [ %371, %369 ]
  %380 = add nsw i32 %q.i.0.i77, 1
  br label %__nv_cos.exit84

__nv_cos.exit84:                                  ; preds = %367, %__internal_trig_reduction_kerneld.exit.i75
  %z.0.i78 = phi double [ %368, %367 ], [ %t.i1.0.i76, %__internal_trig_reduction_kerneld.exit.i75 ]
  %i.0.i79 = phi i32 [ 1, %367 ], [ %380, %__internal_trig_reduction_kerneld.exit.i75 ]
  %381 = and i32 %i.0.i79, 1
  %382 = shl nuw nsw i32 %381, 3
  %383 = zext nneg i32 %382 to i64
  %384 = getelementptr inbounds nuw double, ptr addrspace(1) @__cudart_sin_cos_coeffs, i64 %383
  %385 = tail call double @llvm.nvvm.mul.rn.d(double %z.0.i78, double %z.0.i78) #7
  %.not.i80 = icmp eq i32 %381, 0
  %386 = select i1 %.not.i80, double 0x3DE5DB65F9785EBA, double 0xBDA8FF8320FD8164
  %387 = load <2 x double>, ptr addrspace(1) %384, align 16, !invariant.load !6
  %388 = extractelement <2 x double> %387, i32 0
  %389 = extractelement <2 x double> %387, i32 1
  %390 = tail call double @llvm.fma.f64(double %386, double %385, double %388)
  %391 = tail call double @llvm.fma.f64(double %390, double %385, double %389)
  %392 = getelementptr inbounds nuw i8, ptr addrspace(1) %384, i64 16
  %393 = load <2 x double>, ptr addrspace(1) %392, align 16, !invariant.load !6
  %394 = extractelement <2 x double> %393, i32 0
  %395 = extractelement <2 x double> %393, i32 1
  %396 = tail call double @llvm.fma.f64(double %391, double %385, double %394)
  %397 = tail call double @llvm.fma.f64(double %396, double %385, double %395)
  %398 = getelementptr inbounds nuw i8, ptr addrspace(1) %384, i64 32
  %399 = load <2 x double>, ptr addrspace(1) %398, align 16, !invariant.load !6
  %400 = extractelement <2 x double> %399, i32 0
  %401 = extractelement <2 x double> %399, i32 1
  %402 = tail call double @llvm.fma.f64(double %397, double %385, double %400)
  %403 = tail call double @llvm.fma.f64(double %402, double %385, double %401)
  %404 = tail call double @llvm.fma.f64(double %403, double %z.0.i78, double %z.0.i78)
  %405 = tail call double @llvm.fma.f64(double %403, double %385, double 1.000000e+00)
  %spec.select.i81 = select i1 %.not.i80, double %404, double %405
  %406 = and i32 %i.0.i79, 2
  %.not1.i82 = icmp eq i32 %406, 0
  %407 = fsub double 0.000000e+00, %spec.select.i81
  %.1.i83 = select i1 %.not1.i82, double %spec.select.i81, double %407
  %408 = fmul double %z.2.i112, %.1.i83
  %409 = tail call double @llvm.fma.f64(double %276, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %410 = tail call i32 @llvm.nvvm.d2i.lo(double %409) #7
  %411 = tail call double @llvm.nvvm.add.rn.d(double %409, double 0xC338000000000000) #7
  %412 = tail call double @llvm.fma.f64(double %411, double 0xBFE62E42FEFA39EF, double %276)
  %413 = tail call double @llvm.fma.f64(double %411, double 0xBC7ABC9E3B39803F, double %412)
  %414 = tail call double @llvm.fma.f64(double %413, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %415 = tail call double @llvm.fma.f64(double %414, double %413, double 0x3EC71DEE62401315)
  %416 = tail call double @llvm.fma.f64(double %415, double %413, double 0x3EFA01997C89EB71)
  %417 = tail call double @llvm.fma.f64(double %416, double %413, double 0x3F2A01A014761F65)
  %418 = tail call double @llvm.fma.f64(double %417, double %413, double 0x3F56C16C1852B7AF)
  %419 = tail call double @llvm.fma.f64(double %418, double %413, double 0x3F81111111122322)
  %420 = tail call double @llvm.fma.f64(double %419, double %413, double 0x3FA55555555502A1)
  %421 = tail call double @llvm.fma.f64(double %420, double %413, double 0x3FC5555555555511)
  %422 = tail call double @llvm.fma.f64(double %421, double %413, double 0x3FE000000000000B)
  %423 = tail call double @llvm.fma.f64(double %422, double %413, double 1.000000e+00)
  %424 = tail call double @llvm.fma.f64(double %423, double %413, double 1.000000e+00)
  %425 = tail call i32 @llvm.nvvm.d2i.lo(double %424) #7
  %426 = tail call i32 @llvm.nvvm.d2i.hi(double %424) #7
  %427 = shl i32 %410, 20
  %428 = add i32 %426, %427
  %429 = tail call double @llvm.nvvm.lohi.i2d(i32 %425, i32 %428) #7
  %430 = tail call i32 @llvm.nvvm.d2i.hi(double %276) #7
  %431 = bitcast i32 %430 to float
  %432 = tail call float @llvm.nvvm.fabs.f32(float %431)
  %433 = fcmp olt float %432, 0x4010C46560000000
  br i1 %433, label %__nv_exp.exit117, label %__internal_fast_icmp_abs_lt.exit.i114

__internal_fast_icmp_abs_lt.exit.i114:            ; preds = %__nv_cos.exit84
  %434 = fcmp olt double %276, 0.000000e+00
  %435 = fadd double %276, 0x7FF0000000000000
  %z.0.i115 = select i1 %434, double 0.000000e+00, double %435
  %436 = fcmp olt float %432, 0x4010E90000000000
  br i1 %436, label %437, label %__nv_exp.exit117

437:                                              ; preds = %__internal_fast_icmp_abs_lt.exit.i114
  %438 = sdiv i32 %410, 2
  %439 = shl i32 %438, 20
  %440 = add i32 %426, %439
  %441 = tail call double @llvm.nvvm.lohi.i2d(i32 %425, i32 %440) #7
  %442 = sub nsw i32 %410, %438
  %443 = shl i32 %442, 20
  %444 = add nsw i32 %443, 1072693248
  %445 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %444) #7
  %446 = fmul double %445, %441
  br label %__nv_exp.exit117

__nv_exp.exit117:                                 ; preds = %__nv_cos.exit84, %__internal_fast_icmp_abs_lt.exit.i114, %437
  %z.2.i116 = phi double [ %429, %__nv_cos.exit84 ], [ %446, %437 ], [ %z.0.i115, %__internal_fast_icmp_abs_lt.exit.i114 ]
  %447 = fcmp ole double %51, %366
  %448 = fmul double %z.2.i112, %364
  %449 = fmul double %.1.i53, %z.2.i116
  %450 = and i1 %365, %447
  %451 = fcmp olt double %52, %366
  %452 = fcmp oeq double %z.2.i116, 0x7FF0000000000000
  %453 = fmul double %z.2.i112, %408
  %454 = fmul double %.1.i83, %z.2.i116
  %455 = fcmp oeq double %279, 0.000000e+00
  %456 = select i1 %452, double %448, double %449
  %457 = and i1 %451, %450
  %458 = fcmp ogt double %53, %366
  %459 = select i1 %452, double %453, double %454
  %460 = select i1 %455, double 0.000000e+00, double %456
  %461 = and i1 %458, %457
  %462 = fcmp oge double %54, %366
  %463 = getelementptr inbounds i8, ptr addrspace(1) %252, i64 16
  %464 = load <2 x double>, ptr addrspace(1) %463, align 16, !invariant.load !6
  %.unpack18194 = extractelement <2 x double> %464, i32 0
  %.unpack20195 = extractelement <2 x double> %464, i32 1
  %465 = and i1 %462, %461
  %466 = fmul double %.unpack18194, %459
  %467 = fmul double %.unpack20195, %460
  %468 = fsub double %466, %467
  %469 = fmul double %.unpack20195, %459
  %470 = fmul double %.unpack18194, %460
  %471 = fadd double %469, %470
  %472 = getelementptr inbounds i8, ptr addrspace(1) %262, i64 16
  %473 = load <2 x double>, ptr addrspace(1) %472, align 16
  %.unpack21196 = extractelement <2 x double> %473, i32 0
  %.unpack23197 = extractelement <2 x double> %473, i32 1
  %474 = select i1 %465, double %468, double 0.000000e+00
  %475 = fadd double %.unpack21196, %474
  %476 = select i1 %465, double %471, double 0.000000e+00
  %477 = fadd double %.unpack23197, %476
  %478 = insertelement <2 x double> poison, double %475, i32 0
  %479 = insertelement <2 x double> %478, double %477, i32 1
  store <2 x double> %479, ptr addrspace(1) %472, align 16
  %480 = getelementptr inbounds i8, ptr addrspace(1) %59, i64 32
  %481 = load <2 x double>, ptr addrspace(1) %480, align 16, !invariant.load !6
  %.unpack26198 = extractelement <2 x double> %481, i32 0
  %.unpack28199 = extractelement <2 x double> %481, i32 1
  %482 = fsub double %.unpack26198, %47
  %483 = select i1 %46, double 0.000000e+00, double %.unpack28199
  %484 = fmul double %.unpack206, %482
  %485 = fmul double %.unpack2207, %483
  %486 = fsub double %484, %485
  %487 = fmul double %.unpack206, %483
  %488 = fmul double %.unpack2207, %482
  %489 = fadd double %488, %487
  %490 = fmul double %486, 5.000000e-01
  %491 = tail call double @llvm.fma.f64(double %490, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %492 = tail call i32 @llvm.nvvm.d2i.lo(double %491) #7
  %493 = tail call double @llvm.nvvm.add.rn.d(double %491, double 0xC338000000000000) #7
  %494 = tail call double @llvm.fma.f64(double %493, double 0xBFE62E42FEFA39EF, double %490)
  %495 = tail call double @llvm.fma.f64(double %493, double 0xBC7ABC9E3B39803F, double %494)
  %496 = tail call double @llvm.fma.f64(double %495, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %497 = tail call double @llvm.fma.f64(double %496, double %495, double 0x3EC71DEE62401315)
  %498 = tail call double @llvm.fma.f64(double %497, double %495, double 0x3EFA01997C89EB71)
  %499 = tail call double @llvm.fma.f64(double %498, double %495, double 0x3F2A01A014761F65)
  %500 = tail call double @llvm.fma.f64(double %499, double %495, double 0x3F56C16C1852B7AF)
  %501 = tail call double @llvm.fma.f64(double %500, double %495, double 0x3F81111111122322)
  %502 = tail call double @llvm.fma.f64(double %501, double %495, double 0x3FA55555555502A1)
  %503 = tail call double @llvm.fma.f64(double %502, double %495, double 0x3FC5555555555511)
  %504 = tail call double @llvm.fma.f64(double %503, double %495, double 0x3FE000000000000B)
  %505 = tail call double @llvm.fma.f64(double %504, double %495, double 1.000000e+00)
  %506 = tail call double @llvm.fma.f64(double %505, double %495, double 1.000000e+00)
  %507 = tail call i32 @llvm.nvvm.d2i.lo(double %506) #7
  %508 = tail call i32 @llvm.nvvm.d2i.hi(double %506) #7
  %509 = shl i32 %492, 20
  %510 = add i32 %508, %509
  %511 = tail call double @llvm.nvvm.lohi.i2d(i32 %507, i32 %510) #7
  %512 = tail call i32 @llvm.nvvm.d2i.hi(double %490) #7
  %513 = bitcast i32 %512 to float
  %514 = tail call float @llvm.nvvm.fabs.f32(float %513)
  %515 = fcmp olt float %514, 0x4010C46560000000
  br i1 %515, label %__nv_exp.exit121, label %__internal_fast_icmp_abs_lt.exit.i118

__internal_fast_icmp_abs_lt.exit.i118:            ; preds = %__nv_exp.exit117
  %516 = fcmp olt double %490, 0.000000e+00
  %517 = fadd double %490, 0x7FF0000000000000
  %z.0.i119 = select i1 %516, double 0.000000e+00, double %517
  %518 = fcmp olt float %514, 0x4010E90000000000
  br i1 %518, label %519, label %__nv_exp.exit121

519:                                              ; preds = %__internal_fast_icmp_abs_lt.exit.i118
  %520 = sdiv i32 %492, 2
  %521 = shl i32 %520, 20
  %522 = add i32 %508, %521
  %523 = tail call double @llvm.nvvm.lohi.i2d(i32 %507, i32 %522) #7
  %524 = sub nsw i32 %492, %520
  %525 = shl i32 %524, 20
  %526 = add nsw i32 %525, 1072693248
  %527 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %526) #7
  %528 = fmul double %527, %523
  br label %__nv_exp.exit121

__nv_exp.exit121:                                 ; preds = %__nv_exp.exit117, %__internal_fast_icmp_abs_lt.exit.i118, %519
  %z.2.i120 = phi double [ %511, %__nv_exp.exit117 ], [ %528, %519 ], [ %z.0.i119, %__internal_fast_icmp_abs_lt.exit.i118 ]
  %529 = tail call double @llvm.nvvm.fabs.f64(double %489)
  %530 = fcmp oeq double %529, 0x7FF0000000000000
  br i1 %530, label %531, label %533

531:                                              ; preds = %__nv_exp.exit121
  %532 = tail call double @llvm.nvvm.mul.rn.d(double %489, double 0.000000e+00) #7
  br label %__nv_sin.exit61

533:                                              ; preds = %__nv_exp.exit121
  %534 = fmul double %489, 0x3FE45F306DC9C883
  %535 = tail call i32 @llvm.nvvm.d2i.rn(double %534) #7
  %536 = sitofp i32 %535 to double
  %537 = fneg double %536
  %538 = tail call double @llvm.fma.f64(double %537, double 0x3FF921FB54442D18, double %489)
  %539 = tail call double @llvm.fma.f64(double %537, double 0x3C91A62633145C00, double %538)
  %540 = tail call double @llvm.fma.f64(double %537, double 0x397B839A252049C0, double %539)
  %541 = fcmp ult double %529, 0x41E0000000000000
  br i1 %541, label %__nv_sin.exit61, label %542

542:                                              ; preds = %533
  %543 = tail call fastcc { double, i32 } @__internal_trig_reduction_slowpathd(double %489) #7
  %newret166 = extractvalue { double, i32 } %543, 0
  %newret168 = extractvalue { double, i32 } %543, 1
  br label %__nv_sin.exit61

__nv_sin.exit61:                                  ; preds = %531, %533, %542
  %z.0.i55 = phi double [ %532, %531 ], [ %newret166, %542 ], [ %540, %533 ]
  %i.0.i56 = phi i32 [ 0, %531 ], [ %newret168, %542 ], [ %535, %533 ]
  %544 = fcmp oeq double %529, 0x7FF0000000000000
  %545 = fcmp ole double %.unpack26198, %49
  %546 = fcmp ogt double %.unpack26198, %48
  %547 = and i32 %i.0.i56, 1
  %548 = shl nuw nsw i32 %547, 3
  %549 = zext nneg i32 %548 to i64
  %550 = getelementptr inbounds nuw double, ptr addrspace(1) @__cudart_sin_cos_coeffs, i64 %549
  %551 = tail call double @llvm.nvvm.mul.rn.d(double %z.0.i55, double %z.0.i55) #7
  %.not.i57 = icmp eq i32 %547, 0
  %552 = select i1 %.not.i57, double 0x3DE5DB65F9785EBA, double 0xBDA8FF8320FD8164
  %553 = load <2 x double>, ptr addrspace(1) %550, align 16, !invariant.load !6
  %554 = extractelement <2 x double> %553, i32 0
  %555 = extractelement <2 x double> %553, i32 1
  %556 = tail call double @llvm.fma.f64(double %552, double %551, double %554)
  %557 = tail call double @llvm.fma.f64(double %556, double %551, double %555)
  %558 = getelementptr inbounds nuw i8, ptr addrspace(1) %550, i64 16
  %559 = load <2 x double>, ptr addrspace(1) %558, align 16, !invariant.load !6
  %560 = extractelement <2 x double> %559, i32 0
  %561 = extractelement <2 x double> %559, i32 1
  %562 = tail call double @llvm.fma.f64(double %557, double %551, double %560)
  %563 = tail call double @llvm.fma.f64(double %562, double %551, double %561)
  %564 = getelementptr inbounds nuw i8, ptr addrspace(1) %550, i64 32
  %565 = load <2 x double>, ptr addrspace(1) %564, align 16, !invariant.load !6
  %566 = extractelement <2 x double> %565, i32 0
  %567 = extractelement <2 x double> %565, i32 1
  %568 = tail call double @llvm.fma.f64(double %563, double %551, double %566)
  %569 = tail call double @llvm.fma.f64(double %568, double %551, double %567)
  %570 = tail call double @llvm.fma.f64(double %569, double %z.0.i55, double %z.0.i55)
  %571 = tail call double @llvm.fma.f64(double %569, double %551, double 1.000000e+00)
  %spec.select.i58 = select i1 %.not.i57, double %570, double %571
  %572 = and i32 %i.0.i56, 2
  %.not1.i59 = icmp eq i32 %572, 0
  %573 = fsub double 0.000000e+00, %spec.select.i58
  %.1.i60 = select i1 %.not1.i59, double %spec.select.i58, double %573
  %574 = fmul double %z.2.i120, %.1.i60
  %575 = and i1 %546, %545
  %576 = fneg double %.unpack28199
  br i1 %544, label %577, label %579

577:                                              ; preds = %__nv_sin.exit61
  %578 = tail call double @llvm.nvvm.mul.rn.d(double %489, double 0.000000e+00) #7
  br label %__nv_cos.exit94

579:                                              ; preds = %__nv_sin.exit61
  %580 = fmul double %489, 0x3FE45F306DC9C883
  %581 = tail call i32 @llvm.nvvm.d2i.rn(double %580) #7
  %582 = sitofp i32 %581 to double
  %583 = fneg double %582
  %584 = tail call double @llvm.fma.f64(double %583, double 0x3FF921FB54442D18, double %489)
  %585 = tail call double @llvm.fma.f64(double %583, double 0x3C91A62633145C00, double %584)
  %586 = tail call double @llvm.fma.f64(double %583, double 0x397B839A252049C0, double %585)
  %587 = fcmp ult double %529, 0x41E0000000000000
  br i1 %587, label %__internal_trig_reduction_kerneld.exit.i85, label %588

588:                                              ; preds = %579
  %589 = tail call fastcc { double, i32 } @__internal_trig_reduction_slowpathd(double %489) #7
  %newret150 = extractvalue { double, i32 } %589, 0
  %newret152 = extractvalue { double, i32 } %589, 1
  br label %__internal_trig_reduction_kerneld.exit.i85

__internal_trig_reduction_kerneld.exit.i85:       ; preds = %588, %579
  %t.i1.0.i86 = phi double [ %newret150, %588 ], [ %586, %579 ]
  %q.i.0.i87 = phi i32 [ %newret152, %588 ], [ %581, %579 ]
  %590 = add nsw i32 %q.i.0.i87, 1
  br label %__nv_cos.exit94

__nv_cos.exit94:                                  ; preds = %577, %__internal_trig_reduction_kerneld.exit.i85
  %z.0.i88 = phi double [ %578, %577 ], [ %t.i1.0.i86, %__internal_trig_reduction_kerneld.exit.i85 ]
  %i.0.i89 = phi i32 [ 1, %577 ], [ %590, %__internal_trig_reduction_kerneld.exit.i85 ]
  %591 = and i32 %i.0.i89, 1
  %592 = shl nuw nsw i32 %591, 3
  %593 = zext nneg i32 %592 to i64
  %594 = getelementptr inbounds nuw double, ptr addrspace(1) @__cudart_sin_cos_coeffs, i64 %593
  %595 = tail call double @llvm.nvvm.mul.rn.d(double %z.0.i88, double %z.0.i88) #7
  %.not.i90 = icmp eq i32 %591, 0
  %596 = select i1 %.not.i90, double 0x3DE5DB65F9785EBA, double 0xBDA8FF8320FD8164
  %597 = load <2 x double>, ptr addrspace(1) %594, align 16, !invariant.load !6
  %598 = extractelement <2 x double> %597, i32 0
  %599 = extractelement <2 x double> %597, i32 1
  %600 = tail call double @llvm.fma.f64(double %596, double %595, double %598)
  %601 = tail call double @llvm.fma.f64(double %600, double %595, double %599)
  %602 = getelementptr inbounds nuw i8, ptr addrspace(1) %594, i64 16
  %603 = load <2 x double>, ptr addrspace(1) %602, align 16, !invariant.load !6
  %604 = extractelement <2 x double> %603, i32 0
  %605 = extractelement <2 x double> %603, i32 1
  %606 = tail call double @llvm.fma.f64(double %601, double %595, double %604)
  %607 = tail call double @llvm.fma.f64(double %606, double %595, double %605)
  %608 = getelementptr inbounds nuw i8, ptr addrspace(1) %594, i64 32
  %609 = load <2 x double>, ptr addrspace(1) %608, align 16, !invariant.load !6
  %610 = extractelement <2 x double> %609, i32 0
  %611 = extractelement <2 x double> %609, i32 1
  %612 = tail call double @llvm.fma.f64(double %607, double %595, double %610)
  %613 = tail call double @llvm.fma.f64(double %612, double %595, double %611)
  %614 = tail call double @llvm.fma.f64(double %613, double %z.0.i88, double %z.0.i88)
  %615 = tail call double @llvm.fma.f64(double %613, double %595, double 1.000000e+00)
  %spec.select.i91 = select i1 %.not.i90, double %614, double %615
  %616 = and i32 %i.0.i89, 2
  %.not1.i92 = icmp eq i32 %616, 0
  %617 = fsub double 0.000000e+00, %spec.select.i91
  %.1.i93 = select i1 %.not1.i92, double %spec.select.i91, double %617
  %618 = fmul double %z.2.i120, %.1.i93
  %619 = tail call double @llvm.fma.f64(double %486, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %620 = tail call i32 @llvm.nvvm.d2i.lo(double %619) #7
  %621 = tail call double @llvm.nvvm.add.rn.d(double %619, double 0xC338000000000000) #7
  %622 = tail call double @llvm.fma.f64(double %621, double 0xBFE62E42FEFA39EF, double %486)
  %623 = tail call double @llvm.fma.f64(double %621, double 0xBC7ABC9E3B39803F, double %622)
  %624 = tail call double @llvm.fma.f64(double %623, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %625 = tail call double @llvm.fma.f64(double %624, double %623, double 0x3EC71DEE62401315)
  %626 = tail call double @llvm.fma.f64(double %625, double %623, double 0x3EFA01997C89EB71)
  %627 = tail call double @llvm.fma.f64(double %626, double %623, double 0x3F2A01A014761F65)
  %628 = tail call double @llvm.fma.f64(double %627, double %623, double 0x3F56C16C1852B7AF)
  %629 = tail call double @llvm.fma.f64(double %628, double %623, double 0x3F81111111122322)
  %630 = tail call double @llvm.fma.f64(double %629, double %623, double 0x3FA55555555502A1)
  %631 = tail call double @llvm.fma.f64(double %630, double %623, double 0x3FC5555555555511)
  %632 = tail call double @llvm.fma.f64(double %631, double %623, double 0x3FE000000000000B)
  %633 = tail call double @llvm.fma.f64(double %632, double %623, double 1.000000e+00)
  %634 = tail call double @llvm.fma.f64(double %633, double %623, double 1.000000e+00)
  %635 = tail call i32 @llvm.nvvm.d2i.lo(double %634) #7
  %636 = tail call i32 @llvm.nvvm.d2i.hi(double %634) #7
  %637 = shl i32 %620, 20
  %638 = add i32 %636, %637
  %639 = tail call double @llvm.nvvm.lohi.i2d(i32 %635, i32 %638) #7
  %640 = tail call i32 @llvm.nvvm.d2i.hi(double %486) #7
  %641 = bitcast i32 %640 to float
  %642 = tail call float @llvm.nvvm.fabs.f32(float %641)
  %643 = fcmp olt float %642, 0x4010C46560000000
  br i1 %643, label %__nv_exp.exit125, label %__internal_fast_icmp_abs_lt.exit.i122

__internal_fast_icmp_abs_lt.exit.i122:            ; preds = %__nv_cos.exit94
  %644 = fcmp olt double %486, 0.000000e+00
  %645 = fadd double %486, 0x7FF0000000000000
  %z.0.i123 = select i1 %644, double 0.000000e+00, double %645
  %646 = fcmp olt float %642, 0x4010E90000000000
  br i1 %646, label %647, label %__nv_exp.exit125

647:                                              ; preds = %__internal_fast_icmp_abs_lt.exit.i122
  %648 = sdiv i32 %620, 2
  %649 = shl i32 %648, 20
  %650 = add i32 %636, %649
  %651 = tail call double @llvm.nvvm.lohi.i2d(i32 %635, i32 %650) #7
  %652 = sub nsw i32 %620, %648
  %653 = shl i32 %652, 20
  %654 = add nsw i32 %653, 1072693248
  %655 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %654) #7
  %656 = fmul double %655, %651
  br label %__nv_exp.exit125

__nv_exp.exit125:                                 ; preds = %__nv_cos.exit94, %__internal_fast_icmp_abs_lt.exit.i122, %647
  %z.2.i124 = phi double [ %639, %__nv_cos.exit94 ], [ %656, %647 ], [ %z.0.i123, %__internal_fast_icmp_abs_lt.exit.i122 ]
  %657 = fcmp ole double %51, %576
  %658 = fmul double %z.2.i120, %574
  %659 = fmul double %.1.i60, %z.2.i124
  %660 = and i1 %575, %657
  %661 = fcmp olt double %52, %576
  %662 = fcmp oeq double %z.2.i124, 0x7FF0000000000000
  %663 = fmul double %z.2.i120, %618
  %664 = fmul double %.1.i93, %z.2.i124
  %665 = fcmp oeq double %489, 0.000000e+00
  %666 = select i1 %662, double %658, double %659
  %667 = and i1 %661, %660
  %668 = fcmp ogt double %53, %576
  %669 = select i1 %662, double %663, double %664
  %670 = select i1 %665, double 0.000000e+00, double %666
  %671 = and i1 %668, %667
  %672 = fcmp oge double %54, %576
  %673 = getelementptr inbounds i8, ptr addrspace(1) %252, i64 32
  %674 = load <2 x double>, ptr addrspace(1) %673, align 16, !invariant.load !6
  %.unpack29188 = extractelement <2 x double> %674, i32 0
  %.unpack31189 = extractelement <2 x double> %674, i32 1
  %675 = and i1 %672, %671
  %676 = fmul double %.unpack29188, %669
  %677 = fmul double %.unpack31189, %670
  %678 = fsub double %676, %677
  %679 = fmul double %.unpack31189, %669
  %680 = fmul double %.unpack29188, %670
  %681 = fadd double %679, %680
  %682 = getelementptr inbounds i8, ptr addrspace(1) %262, i64 32
  %683 = load <2 x double>, ptr addrspace(1) %682, align 32
  %.unpack32190 = extractelement <2 x double> %683, i32 0
  %.unpack34191 = extractelement <2 x double> %683, i32 1
  %684 = select i1 %675, double %678, double 0.000000e+00
  %685 = fadd double %.unpack32190, %684
  %686 = select i1 %675, double %681, double 0.000000e+00
  %687 = fadd double %.unpack34191, %686
  %688 = insertelement <2 x double> poison, double %685, i32 0
  %689 = insertelement <2 x double> %688, double %687, i32 1
  store <2 x double> %689, ptr addrspace(1) %682, align 32
  %690 = getelementptr inbounds i8, ptr addrspace(1) %59, i64 48
  %691 = load <2 x double>, ptr addrspace(1) %690, align 16, !invariant.load !6
  %.unpack37192 = extractelement <2 x double> %691, i32 0
  %.unpack39193 = extractelement <2 x double> %691, i32 1
  %692 = fsub double %.unpack37192, %47
  %693 = select i1 %46, double 0.000000e+00, double %.unpack39193
  %694 = fmul double %.unpack206, %692
  %695 = fmul double %.unpack2207, %693
  %696 = fsub double %694, %695
  %697 = fmul double %.unpack206, %693
  %698 = fmul double %.unpack2207, %692
  %699 = fadd double %698, %697
  %700 = fmul double %696, 5.000000e-01
  %701 = tail call double @llvm.fma.f64(double %700, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %702 = tail call i32 @llvm.nvvm.d2i.lo(double %701) #7
  %703 = tail call double @llvm.nvvm.add.rn.d(double %701, double 0xC338000000000000) #7
  %704 = tail call double @llvm.fma.f64(double %703, double 0xBFE62E42FEFA39EF, double %700)
  %705 = tail call double @llvm.fma.f64(double %703, double 0xBC7ABC9E3B39803F, double %704)
  %706 = tail call double @llvm.fma.f64(double %705, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %707 = tail call double @llvm.fma.f64(double %706, double %705, double 0x3EC71DEE62401315)
  %708 = tail call double @llvm.fma.f64(double %707, double %705, double 0x3EFA01997C89EB71)
  %709 = tail call double @llvm.fma.f64(double %708, double %705, double 0x3F2A01A014761F65)
  %710 = tail call double @llvm.fma.f64(double %709, double %705, double 0x3F56C16C1852B7AF)
  %711 = tail call double @llvm.fma.f64(double %710, double %705, double 0x3F81111111122322)
  %712 = tail call double @llvm.fma.f64(double %711, double %705, double 0x3FA55555555502A1)
  %713 = tail call double @llvm.fma.f64(double %712, double %705, double 0x3FC5555555555511)
  %714 = tail call double @llvm.fma.f64(double %713, double %705, double 0x3FE000000000000B)
  %715 = tail call double @llvm.fma.f64(double %714, double %705, double 1.000000e+00)
  %716 = tail call double @llvm.fma.f64(double %715, double %705, double 1.000000e+00)
  %717 = tail call i32 @llvm.nvvm.d2i.lo(double %716) #7
  %718 = tail call i32 @llvm.nvvm.d2i.hi(double %716) #7
  %719 = shl i32 %702, 20
  %720 = add i32 %718, %719
  %721 = tail call double @llvm.nvvm.lohi.i2d(i32 %717, i32 %720) #7
  %722 = tail call i32 @llvm.nvvm.d2i.hi(double %700) #7
  %723 = bitcast i32 %722 to float
  %724 = tail call float @llvm.nvvm.fabs.f32(float %723)
  %725 = fcmp olt float %724, 0x4010C46560000000
  br i1 %725, label %__nv_exp.exit129, label %__internal_fast_icmp_abs_lt.exit.i126

__internal_fast_icmp_abs_lt.exit.i126:            ; preds = %__nv_exp.exit125
  %726 = fcmp olt double %700, 0.000000e+00
  %727 = fadd double %700, 0x7FF0000000000000
  %z.0.i127 = select i1 %726, double 0.000000e+00, double %727
  %728 = fcmp olt float %724, 0x4010E90000000000
  br i1 %728, label %729, label %__nv_exp.exit129

729:                                              ; preds = %__internal_fast_icmp_abs_lt.exit.i126
  %730 = sdiv i32 %702, 2
  %731 = shl i32 %730, 20
  %732 = add i32 %718, %731
  %733 = tail call double @llvm.nvvm.lohi.i2d(i32 %717, i32 %732) #7
  %734 = sub nsw i32 %702, %730
  %735 = shl i32 %734, 20
  %736 = add nsw i32 %735, 1072693248
  %737 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %736) #7
  %738 = fmul double %737, %733
  br label %__nv_exp.exit129

__nv_exp.exit129:                                 ; preds = %__nv_exp.exit125, %__internal_fast_icmp_abs_lt.exit.i126, %729
  %z.2.i128 = phi double [ %721, %__nv_exp.exit125 ], [ %738, %729 ], [ %z.0.i127, %__internal_fast_icmp_abs_lt.exit.i126 ]
  %739 = tail call double @llvm.nvvm.fabs.f64(double %699)
  %740 = fcmp oeq double %739, 0x7FF0000000000000
  br i1 %740, label %741, label %743

741:                                              ; preds = %__nv_exp.exit129
  %742 = tail call double @llvm.nvvm.mul.rn.d(double %699, double 0.000000e+00) #7
  br label %__nv_sin.exit68

743:                                              ; preds = %__nv_exp.exit129
  %744 = fmul double %699, 0x3FE45F306DC9C883
  %745 = tail call i32 @llvm.nvvm.d2i.rn(double %744) #7
  %746 = sitofp i32 %745 to double
  %747 = fneg double %746
  %748 = tail call double @llvm.fma.f64(double %747, double 0x3FF921FB54442D18, double %699)
  %749 = tail call double @llvm.fma.f64(double %747, double 0x3C91A62633145C00, double %748)
  %750 = tail call double @llvm.fma.f64(double %747, double 0x397B839A252049C0, double %749)
  %751 = fcmp ult double %739, 0x41E0000000000000
  br i1 %751, label %__nv_sin.exit68, label %752

752:                                              ; preds = %743
  %753 = tail call fastcc { double, i32 } @__internal_trig_reduction_slowpathd(double %699) #7
  %newret162 = extractvalue { double, i32 } %753, 0
  %newret164 = extractvalue { double, i32 } %753, 1
  br label %__nv_sin.exit68

__nv_sin.exit68:                                  ; preds = %741, %743, %752
  %z.0.i62 = phi double [ %742, %741 ], [ %newret162, %752 ], [ %750, %743 ]
  %i.0.i63 = phi i32 [ 0, %741 ], [ %newret164, %752 ], [ %745, %743 ]
  %754 = fcmp oeq double %739, 0x7FF0000000000000
  %755 = and i32 %i.0.i63, 1
  %756 = shl nuw nsw i32 %755, 3
  %757 = zext nneg i32 %756 to i64
  %758 = getelementptr inbounds nuw double, ptr addrspace(1) @__cudart_sin_cos_coeffs, i64 %757
  %759 = load <2 x double>, ptr addrspace(1) %758, align 16, !invariant.load !6
  %760 = extractelement <2 x double> %759, i32 0
  %761 = extractelement <2 x double> %759, i32 1
  %762 = getelementptr inbounds nuw i8, ptr addrspace(1) %758, i64 16
  %763 = load <2 x double>, ptr addrspace(1) %762, align 16, !invariant.load !6
  %764 = extractelement <2 x double> %763, i32 0
  %765 = extractelement <2 x double> %763, i32 1
  %766 = getelementptr inbounds nuw i8, ptr addrspace(1) %758, i64 32
  %767 = load <2 x double>, ptr addrspace(1) %766, align 16, !invariant.load !6
  %768 = extractelement <2 x double> %767, i32 0
  %769 = extractelement <2 x double> %767, i32 1
  br i1 %754, label %770, label %772

770:                                              ; preds = %__nv_sin.exit68
  %771 = tail call double @llvm.nvvm.mul.rn.d(double %699, double 0.000000e+00) #7
  br label %__nv_cos.exit104

772:                                              ; preds = %__nv_sin.exit68
  %773 = fmul double %699, 0x3FE45F306DC9C883
  %774 = tail call i32 @llvm.nvvm.d2i.rn(double %773) #7
  %775 = sitofp i32 %774 to double
  %776 = fneg double %775
  %777 = tail call double @llvm.fma.f64(double %776, double 0x3FF921FB54442D18, double %699)
  %778 = tail call double @llvm.fma.f64(double %776, double 0x3C91A62633145C00, double %777)
  %779 = tail call double @llvm.fma.f64(double %776, double 0x397B839A252049C0, double %778)
  %780 = fcmp ult double %739, 0x41E0000000000000
  br i1 %780, label %__internal_trig_reduction_kerneld.exit.i95, label %781

781:                                              ; preds = %772
  %782 = tail call fastcc { double, i32 } @__internal_trig_reduction_slowpathd(double %699) #7
  %newret = extractvalue { double, i32 } %782, 0
  %newret148 = extractvalue { double, i32 } %782, 1
  br label %__internal_trig_reduction_kerneld.exit.i95

__internal_trig_reduction_kerneld.exit.i95:       ; preds = %781, %772
  %t.i1.0.i96 = phi double [ %newret, %781 ], [ %779, %772 ]
  %q.i.0.i97 = phi i32 [ %newret148, %781 ], [ %774, %772 ]
  %783 = add nsw i32 %q.i.0.i97, 1
  br label %__nv_cos.exit104

__nv_cos.exit104:                                 ; preds = %770, %__internal_trig_reduction_kerneld.exit.i95
  %z.0.i98 = phi double [ %771, %770 ], [ %t.i1.0.i96, %__internal_trig_reduction_kerneld.exit.i95 ]
  %i.0.i99 = phi i32 [ 1, %770 ], [ %783, %__internal_trig_reduction_kerneld.exit.i95 ]
  %784 = and i32 %i.0.i99, 1
  %785 = shl nuw nsw i32 %784, 3
  %786 = zext nneg i32 %785 to i64
  %787 = getelementptr inbounds nuw double, ptr addrspace(1) @__cudart_sin_cos_coeffs, i64 %786
  %788 = load <2 x double>, ptr addrspace(1) %787, align 16, !invariant.load !6
  %789 = extractelement <2 x double> %788, i32 0
  %790 = extractelement <2 x double> %788, i32 1
  %791 = getelementptr inbounds nuw i8, ptr addrspace(1) %787, i64 16
  %792 = load <2 x double>, ptr addrspace(1) %791, align 16, !invariant.load !6
  %793 = extractelement <2 x double> %792, i32 0
  %794 = extractelement <2 x double> %792, i32 1
  %795 = getelementptr inbounds nuw i8, ptr addrspace(1) %787, i64 32
  %796 = load <2 x double>, ptr addrspace(1) %795, align 16, !invariant.load !6
  %797 = extractelement <2 x double> %796, i32 0
  %798 = extractelement <2 x double> %796, i32 1
  %799 = tail call double @llvm.fma.f64(double %696, double 0x3FF71547652B82FE, double 0x4338000000000000)
  %800 = tail call i32 @llvm.nvvm.d2i.lo(double %799) #7
  %801 = tail call double @llvm.nvvm.add.rn.d(double %799, double 0xC338000000000000) #7
  %802 = tail call double @llvm.fma.f64(double %801, double 0xBFE62E42FEFA39EF, double %696)
  %803 = tail call double @llvm.fma.f64(double %801, double 0xBC7ABC9E3B39803F, double %802)
  %804 = tail call double @llvm.fma.f64(double %803, double 0x3E5ADE1569CE2BDF, double 0x3E928AF3FCA213EA)
  %805 = tail call double @llvm.fma.f64(double %804, double %803, double 0x3EC71DEE62401315)
  %806 = tail call double @llvm.fma.f64(double %805, double %803, double 0x3EFA01997C89EB71)
  %807 = tail call double @llvm.fma.f64(double %806, double %803, double 0x3F2A01A014761F65)
  %808 = tail call double @llvm.fma.f64(double %807, double %803, double 0x3F56C16C1852B7AF)
  %809 = tail call double @llvm.fma.f64(double %808, double %803, double 0x3F81111111122322)
  %810 = tail call double @llvm.fma.f64(double %809, double %803, double 0x3FA55555555502A1)
  %811 = tail call double @llvm.fma.f64(double %810, double %803, double 0x3FC5555555555511)
  %812 = tail call double @llvm.fma.f64(double %811, double %803, double 0x3FE000000000000B)
  %813 = tail call double @llvm.fma.f64(double %812, double %803, double 1.000000e+00)
  %814 = tail call double @llvm.fma.f64(double %813, double %803, double 1.000000e+00)
  %815 = tail call i32 @llvm.nvvm.d2i.lo(double %814) #7
  %816 = tail call i32 @llvm.nvvm.d2i.hi(double %814) #7
  %817 = shl i32 %800, 20
  %818 = add i32 %816, %817
  %819 = tail call double @llvm.nvvm.lohi.i2d(i32 %815, i32 %818) #7
  %820 = tail call i32 @llvm.nvvm.d2i.hi(double %696) #7
  %821 = bitcast i32 %820 to float
  %822 = tail call float @llvm.nvvm.fabs.f32(float %821)
  %823 = fcmp olt float %822, 0x4010C46560000000
  br i1 %823, label %__nv_exp.exit133, label %__internal_fast_icmp_abs_lt.exit.i130

__internal_fast_icmp_abs_lt.exit.i130:            ; preds = %__nv_cos.exit104
  %824 = fcmp olt double %696, 0.000000e+00
  %825 = fadd double %696, 0x7FF0000000000000
  %z.0.i131 = select i1 %824, double 0.000000e+00, double %825
  %826 = fcmp olt float %822, 0x4010E90000000000
  br i1 %826, label %827, label %__nv_exp.exit133

827:                                              ; preds = %__internal_fast_icmp_abs_lt.exit.i130
  %828 = sdiv i32 %800, 2
  %829 = shl i32 %828, 20
  %830 = add i32 %816, %829
  %831 = tail call double @llvm.nvvm.lohi.i2d(i32 %815, i32 %830) #7
  %832 = sub nsw i32 %800, %828
  %833 = shl i32 %832, 20
  %834 = add nsw i32 %833, 1072693248
  %835 = tail call double @llvm.nvvm.lohi.i2d(i32 0, i32 %834) #7
  %836 = fmul double %835, %831
  br label %__nv_exp.exit133

__nv_exp.exit133:                                 ; preds = %__nv_cos.exit104, %__internal_fast_icmp_abs_lt.exit.i130, %827
  %z.2.i132 = phi double [ %819, %__nv_cos.exit104 ], [ %836, %827 ], [ %z.0.i131, %__internal_fast_icmp_abs_lt.exit.i130 ]
  %837 = and i32 %i.0.i99, 2
  %.not1.i102 = icmp eq i32 %837, 0
  %.not.i100 = icmp eq i32 %784, 0
  %838 = select i1 %.not.i100, double 0x3DE5DB65F9785EBA, double 0xBDA8FF8320FD8164
  %839 = tail call double @llvm.nvvm.mul.rn.d(double %z.0.i98, double %z.0.i98) #7
  %840 = tail call double @llvm.fma.f64(double %838, double %839, double %789)
  %841 = tail call double @llvm.fma.f64(double %840, double %839, double %790)
  %842 = tail call double @llvm.fma.f64(double %841, double %839, double %793)
  %843 = tail call double @llvm.fma.f64(double %842, double %839, double %794)
  %844 = tail call double @llvm.fma.f64(double %843, double %839, double %797)
  %845 = tail call double @llvm.fma.f64(double %844, double %839, double %798)
  %846 = tail call double @llvm.fma.f64(double %845, double %z.0.i98, double %z.0.i98)
  %847 = tail call double @llvm.fma.f64(double %845, double %839, double 1.000000e+00)
  %spec.select.i101 = select i1 %.not.i100, double %846, double %847
  %848 = fsub double 0.000000e+00, %spec.select.i101
  %.1.i103 = select i1 %.not1.i102, double %spec.select.i101, double %848
  %849 = fmul double %z.2.i128, %.1.i103
  %850 = fneg double %.unpack39193
  %851 = fcmp ole double %51, %850
  %852 = fcmp ogt double %.unpack37192, %48
  %853 = fcmp ole double %.unpack37192, %49
  %854 = and i1 %852, %853
  %855 = and i32 %i.0.i63, 2
  %.not1.i66 = icmp eq i32 %855, 0
  %.not.i64 = icmp eq i32 %755, 0
  %856 = select i1 %.not.i64, double 0x3DE5DB65F9785EBA, double 0xBDA8FF8320FD8164
  %857 = tail call double @llvm.nvvm.mul.rn.d(double %z.0.i62, double %z.0.i62) #7
  %858 = tail call double @llvm.fma.f64(double %856, double %857, double %760)
  %859 = tail call double @llvm.fma.f64(double %858, double %857, double %761)
  %860 = tail call double @llvm.fma.f64(double %859, double %857, double %764)
  %861 = tail call double @llvm.fma.f64(double %860, double %857, double %765)
  %862 = tail call double @llvm.fma.f64(double %861, double %857, double %768)
  %863 = tail call double @llvm.fma.f64(double %862, double %857, double %769)
  %864 = tail call double @llvm.fma.f64(double %863, double %z.0.i62, double %z.0.i62)
  %865 = tail call double @llvm.fma.f64(double %863, double %857, double 1.000000e+00)
  %spec.select.i65 = select i1 %.not.i64, double %864, double %865
  %866 = fsub double 0.000000e+00, %spec.select.i65
  %.1.i67 = select i1 %.not1.i66, double %spec.select.i65, double %866
  %867 = fmul double %z.2.i128, %.1.i67
  %868 = fmul double %z.2.i128, %867
  %869 = fmul double %.1.i67, %z.2.i132
  %870 = and i1 %854, %851
  %871 = fcmp olt double %52, %850
  %872 = fcmp oeq double %z.2.i132, 0x7FF0000000000000
  %873 = fmul double %z.2.i128, %849
  %874 = fmul double %.1.i103, %z.2.i132
  %875 = fcmp oeq double %699, 0.000000e+00
  %876 = select i1 %872, double %868, double %869
  %877 = and i1 %871, %870
  %878 = fcmp ogt double %53, %850
  %879 = select i1 %872, double %873, double %874
  %880 = select i1 %875, double 0.000000e+00, double %876
  %881 = and i1 %878, %877
  %882 = fcmp oge double %54, %850
  %883 = getelementptr inbounds i8, ptr addrspace(1) %252, i64 48
  %884 = load <2 x double>, ptr addrspace(1) %883, align 16, !invariant.load !6
  %.unpack40184 = extractelement <2 x double> %884, i32 0
  %.unpack42185 = extractelement <2 x double> %884, i32 1
  %885 = and i1 %882, %881
  %886 = fmul double %.unpack40184, %879
  %887 = fmul double %.unpack42185, %880
  %888 = fsub double %886, %887
  %889 = fmul double %.unpack42185, %879
  %890 = fmul double %.unpack40184, %880
  %891 = fadd double %889, %890
  %892 = getelementptr inbounds i8, ptr addrspace(1) %262, i64 48
  %893 = load <2 x double>, ptr addrspace(1) %892, align 16
  %.unpack43186 = extractelement <2 x double> %893, i32 0
  %.unpack45187 = extractelement <2 x double> %893, i32 1
  %894 = select i1 %885, double %888, double 0.000000e+00
  %895 = fadd double %.unpack43186, %894
  %896 = select i1 %885, double %891, double 0.000000e+00
  %897 = fadd double %.unpack45187, %896
  %898 = insertelement <2 x double> poison, double %895, i32 0
  %899 = insertelement <2 x double> %898, double %897, i32 1
  store <2 x double> %899, ptr addrspace(1) %892, align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_add_fusion_1(ptr noalias align 256 captures(none) dereferenceable(8) %0, ptr noalias readnone align 256 captures(none) dereferenceable(8) %1) local_unnamed_addr #2 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = load i64, ptr addrspace(1) %3, align 256
  %5 = add i64 %4, 1
  store i64 %5, ptr addrspace(1) %3, align 256
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.fabs.f64(double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.nvvm.mul.rn.d(double, double) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.d2i.rn(double) #3

; Function Attrs: nofree noinline nosync nounwind memory(none)
define internal fastcc { double, i32 } @__internal_trig_reduction_slowpathd(double %a) unnamed_addr #5 {
  %result = alloca [5 x i64], align 8
  %1 = addrspacecast ptr %result to ptr addrspace(5)
  %2 = tail call i32 @llvm.nvvm.d2i.hi(double %a) #7
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
  %15 = tail call i32 @llvm.smin.i32(i32 %14, i32 18) #7
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
  %28 = tail call { i64, i64 } asm "{\0A\09.reg .u32 r0, r1, r2, r3, alo, ahi, blo, bhi, clo, chi;\0A\09mov.b64         {alo,ahi}, $2;    \0A\09mov.b64         {blo,bhi}, $3;    \0A\09mov.b64         {clo,chi}, $4;    \0A\09mad.lo.cc.u32   r0, alo, blo, clo;\0A\09madc.hi.cc.u32  r1, alo, blo, chi;\0A\09madc.hi.u32     r2, alo, bhi,   0;\0A\09mad.lo.cc.u32   r1, alo, bhi,  r1;\0A\09madc.hi.cc.u32  r2, ahi, blo,  r2;\0A\09madc.hi.u32     r3, ahi, bhi,   0;\0A\09mad.lo.cc.u32   r1, ahi, blo,  r1;\0A\09madc.lo.cc.u32  r2, ahi, bhi,  r2;\0A\09addc.u32        r3,  r3,   0;     \0A\09mov.b64         $0, {r0,r1};      \0A\09mov.b64         $1, {r2,r3};      \0A\09}", "=l,=l,l,l,l"(i64 %27, i64 %11, i64 %p.129.07) #8, !srcloc !7
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
  br i1 %ov, label %._crit_edge, label %.lr.ph, !llvm.loop !8

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
  %66 = tail call { i64, i64 } asm "{\0A\09.reg .u32 r0, r1, r2, r3, a0, a1, a2, a3, b0, b1, b2, b3;\0A\09mov.b64         {a0,a1}, $2;\0A\09mov.b64         {a2,a3}, $3;\0A\09mov.b64         {b0,b1}, $4;\0A\09mov.b64         {b2,b3}, $5;\0A\09sub.cc.u32      r0, a0, b0; \0A\09subc.cc.u32     r1, a1, b1; \0A\09subc.cc.u32     r2, a2, b2; \0A\09subc.u32        r3, a3, b3; \0A\09mov.b64         $0, {r0,r1};\0A\09mov.b64         $1, {r2,r3};\0A\09}", "=l,=l,l,l,l,l"(i64 0, i64 0, i64 %59, i64 %58) #8, !srcloc !10
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
  %71 = tail call { i64, i64 } asm "{\0A\09.reg .u32 r0, r1, r2, r3, alo, ahi, blo, bhi;\0A\09mov.b64         {alo,ahi}, $2;   \0A\09mov.b64         {blo,bhi}, $3;   \0A\09mul.lo.u32      r0, alo, blo;    \0A\09mul.hi.u32      r1, alo, blo;    \0A\09mad.lo.cc.u32   r1, alo, bhi, r1;\0A\09madc.hi.u32     r2, alo, bhi,  0;\0A\09mad.lo.cc.u32   r1, ahi, blo, r1;\0A\09madc.hi.cc.u32  r2, ahi, blo, r2;\0A\09madc.hi.u32     r3, ahi, bhi,  0;\0A\09mad.lo.cc.u32   r2, ahi, bhi, r2;\0A\09addc.u32        r3, r3,  0;      \0A\09mov.b64         $0, {r0,r1};     \0A\09mov.b64         $1, {r2,r3};     \0A\09}", "=l,=l,l,l"(i64 %spec.select6, i64 -3958705157555305931) #8, !srcloc !11
  %72 = extractvalue { i64, i64 } %71, 1
  %73 = icmp sgt i64 %72, 0
  %74 = add nuw nsw i64 %ctlz, 1
  %75 = extractvalue { i64, i64 } %71, 0
  br i1 %73, label %76, label %79

76:                                               ; preds = %69
  %77 = tail call { i64, i64 } asm "{\0A\09.reg .u32 r0, r1, r2, r3, a0, a1, a2, a3, b0, b1, b2, b3;\0A\09mov.b64         {a0,a1}, $2;\0A\09mov.b64         {a2,a3}, $3;\0A\09mov.b64         {b0,b1}, $4;\0A\09mov.b64         {b2,b3}, $5;\0A\09add.cc.u32      r0, a0, b0; \0A\09addc.cc.u32     r1, a1, b1; \0A\09addc.cc.u32     r2, a2, b2; \0A\09addc.u32        r3, a3, b3; \0A\09mov.b64         $0, {r0,r1};\0A\09mov.b64         $1, {r2,r3};\0A\09}", "=l,=l,l,l,l,l"(i64 %75, i64 %72, i64 %75, i64 %72) #8, !srcloc !12
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
declare i32 @llvm.smax.i32(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fma.f64(double, double, double) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.fshl.i64(i64, i64, i64) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nofree nosync nounwind memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #5 = { nofree noinline nosync nounwind memory(none) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { nounwind }
attributes #8 = { nounwind memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!nvvmir.version = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!3 = !{i32 2, i32 0}
!4 = !{i32 0, i32 96100}
!5 = !{i32 0, i32 128}
!6 = !{}
!7 = !{i32 161521, i32 161525, i32 161594, i32 161642, i32 161690, i32 161738, i32 161786, i32 161834, i32 161882, i32 161930, i32 161978, i32 162026, i32 162074, i32 162122, i32 162170, i32 162218, i32 162266}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.count", i32 1}
!10 = !{i32 159255, i32 159259, i32 159330, i32 159372, i32 159414, i32 159456, i32 159498, i32 159540, i32 159582, i32 159624, i32 159666, i32 159708, i32 159750}
!11 = !{i32 160296, i32 160300, i32 160359, i32 160406, i32 160453, i32 160500, i32 160547, i32 160594, i32 160641, i32 160688, i32 160735, i32 160782, i32 160829, i32 160876, i32 160923, i32 160970}
!12 = !{i32 158057, i32 158061, i32 158132, i32 158174, i32 158216, i32 158258, i32 158300, i32 158342, i32 158384, i32 158426, i32 158468, i32 158510, i32 158552}
