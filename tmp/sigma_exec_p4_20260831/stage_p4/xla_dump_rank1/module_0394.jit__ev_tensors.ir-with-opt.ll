; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_6_0 = local_unnamed_addr addrspace(1) global [64 x i8] zeroinitializer, align 256
@buffer_for_constant_5_0 = local_unnamed_addr addrspace(1) global [64 x i8] zeroinitializer, align 256

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_add_multiply_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(1179648) %0, ptr noalias readonly align 16 captures(none) dereferenceable(4718592) %1, ptr noalias readonly align 256 captures(none) dereferenceable(16) %2, ptr noalias readonly align 256 captures(none) dereferenceable(4) %3, ptr noalias readonly align 256 captures(none) dereferenceable(16) %4, ptr noalias readonly align 16 captures(none) dereferenceable(24772608) %5, ptr noalias writeonly align 256 captures(none) dereferenceable(24772608) %6, ptr noalias writeonly align 256 captures(none) dereferenceable(24772608) %7) local_unnamed_addr #0 {
  %9 = addrspacecast ptr %3 to ptr addrspace(1)
  %10 = addrspacecast ptr %4 to ptr addrspace(1)
  %11 = addrspacecast ptr %2 to ptr addrspace(1)
  %12 = addrspacecast ptr %5 to ptr addrspace(1)
  %13 = addrspacecast ptr %1 to ptr addrspace(1)
  %14 = addrspacecast ptr %0 to ptr addrspace(1)
  %15 = addrspacecast ptr %6 to ptr addrspace(1)
  %16 = addrspacecast ptr %7 to ptr addrspace(1)
  %17 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %18 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %19 = shl nuw nsw i32 %17, 7
  %20 = or disjoint i32 %19, %18
  %21 = udiv i32 %20, 3
  %22 = urem i32 %21, 12
  %23 = load i32, ptr addrspace(1) %9, align 256, !invariant.load !4
  %24 = tail call i32 @llvm.umin.i32(i32 %23, i32 3)
  %25 = zext nneg i32 %24 to i64
  %26 = getelementptr inbounds i32, ptr addrspace(1) %10, i64 %25
  %27 = load i32, ptr addrspace(1) %26, align 4, !invariant.load !4
  %28 = tail call i32 @llvm.smax.i32(i32 %27, i32 0)
  %29 = tail call i32 @llvm.umin.i32(i32 %28, i32 12)
  %30 = add nuw nsw i32 %29, %22
  %31 = getelementptr inbounds i32, ptr addrspace(1) %11, i64 %25
  %32 = load i32, ptr addrspace(1) %31, align 4, !invariant.load !4
  %33 = tail call i32 @llvm.smax.i32(i32 %32, i32 0)
  %34 = tail call i32 @llvm.umin.i32(i32 %33, i32 12)
  %35 = mul i32 %21, 3
  %.decomposed = sub i32 %20, %35
  %36 = shl nuw nsw i32 %.decomposed, 2
  %37 = shl nuw nsw i32 %18, 2
  %38 = shl nuw nsw i32 %17, 9
  %39 = or disjoint i32 %37, %38
  %40 = zext nneg i32 %39 to i64
  %41 = getelementptr inbounds { double, double }, ptr addrspace(1) %12, i64 %40
  %42 = load <2 x double>, ptr addrspace(1) %41, align 16, !invariant.load !4
  %.unpack64 = extractelement <2 x double> %42, i32 0
  %.unpack265 = extractelement <2 x double> %42, i32 1
  %43 = fmul double %.unpack64, 0x402B361E0E9094F8
  %44 = fmul double %.unpack265, 0.000000e+00
  %45 = fsub double %43, %44
  %46 = fmul double %.unpack265, 0x402B361E0E9094F8
  %47 = fmul double %.unpack64, 0.000000e+00
  %48 = fadd double %47, %46
  %49 = udiv i32 %20, 36
  %50 = and i32 %49, 511
  %51 = mul nuw nsw i32 %50, 576
  %52 = mul nuw nsw i32 %30, 24
  %53 = add nuw nsw i32 %52, %51
  %54 = add nuw nsw i32 %53, %34
  %55 = add nuw nsw i32 %54, %36
  %56 = zext nneg i32 %55 to i64
  %57 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %56
  %58 = load <2 x double>, ptr addrspace(1) %57, align 16, !invariant.load !4
  %.unpack378 = extractelement <2 x double> %58, i32 0
  %.unpack579 = extractelement <2 x double> %58, i32 1
  %59 = fmul double %.unpack378, 0x402B361E0E9094F8
  %60 = fmul double %.unpack579, 0.000000e+00
  %61 = fsub double %59, %60
  %62 = fmul double %.unpack579, 0x402B361E0E9094F8
  %63 = fmul double %.unpack378, 0.000000e+00
  %64 = fadd double %63, %62
  %65 = fadd double %45, %61
  %66 = fadd double %48, %64
  %67 = mul nuw nsw i32 %22, 12
  %68 = add nuw nsw i32 %67, %36
  %69 = mul nuw nsw i32 %50, 144
  %70 = add nuw nsw i32 %68, %69
  %71 = zext nneg i32 %70 to i64
  %72 = getelementptr inbounds { double, double }, ptr addrspace(1) %14, i64 %71
  %73 = load <2 x double>, ptr addrspace(1) %72, align 16, !invariant.load !4
  %.unpack680 = extractelement <2 x double> %73, i32 0
  %.unpack881 = extractelement <2 x double> %73, i32 1
  %74 = fadd double %.unpack680, %65
  %75 = fadd double %.unpack881, %66
  %76 = getelementptr inbounds { double, double }, ptr addrspace(1) %15, i64 %40
  %77 = insertelement <2 x double> poison, double %74, i32 0
  %78 = insertelement <2 x double> %77, double %75, i32 1
  store <2 x double> %78, ptr addrspace(1) %76, align 64
  %79 = getelementptr inbounds { double, double }, ptr addrspace(1) %16, i64 %40
  %80 = insertelement <2 x double> poison, double %45, i32 0
  %81 = insertelement <2 x double> %80, double %48, i32 1
  store <2 x double> %81, ptr addrspace(1) %79, align 64
  %82 = getelementptr inbounds i8, ptr addrspace(1) %41, i64 16
  %83 = load <2 x double>, ptr addrspace(1) %82, align 16, !invariant.load !4
  %.unpack1366 = extractelement <2 x double> %83, i32 0
  %.unpack1567 = extractelement <2 x double> %83, i32 1
  %84 = fmul double %.unpack1366, 0x402B361E0E9094F8
  %85 = fmul double %.unpack1567, 0.000000e+00
  %86 = fsub double %84, %85
  %87 = fmul double %.unpack1567, 0x402B361E0E9094F8
  %88 = fmul double %.unpack1366, 0.000000e+00
  %89 = fadd double %88, %87
  %90 = zext nneg i32 %36 to i64
  %91 = zext nneg i32 %54 to i64
  %92 = add i64 %91, %90
  %93 = getelementptr inbounds { double, double }, ptr addrspace(1) %13, i64 %92
  %94 = getelementptr inbounds i8, ptr addrspace(1) %93, i64 16
  %95 = load <2 x double>, ptr addrspace(1) %94, align 16, !invariant.load !4
  %.unpack1672 = extractelement <2 x double> %95, i32 0
  %.unpack1873 = extractelement <2 x double> %95, i32 1
  %96 = fmul double %.unpack1672, 0x402B361E0E9094F8
  %97 = fmul double %.unpack1873, 0.000000e+00
  %98 = fsub double %96, %97
  %99 = fmul double %.unpack1873, 0x402B361E0E9094F8
  %100 = fmul double %.unpack1672, 0.000000e+00
  %101 = fadd double %100, %99
  %102 = fadd double %86, %98
  %103 = fadd double %89, %101
  %104 = getelementptr inbounds i8, ptr addrspace(1) %72, i64 16
  %105 = load <2 x double>, ptr addrspace(1) %104, align 16, !invariant.load !4
  %.unpack1982 = extractelement <2 x double> %105, i32 0
  %.unpack2183 = extractelement <2 x double> %105, i32 1
  %106 = fadd double %.unpack1982, %102
  %107 = fadd double %.unpack2183, %103
  %108 = getelementptr inbounds i8, ptr addrspace(1) %76, i64 16
  %109 = insertelement <2 x double> poison, double %106, i32 0
  %110 = insertelement <2 x double> %109, double %107, i32 1
  store <2 x double> %110, ptr addrspace(1) %108, align 16
  %111 = getelementptr inbounds i8, ptr addrspace(1) %79, i64 16
  %112 = insertelement <2 x double> poison, double %86, i32 0
  %113 = insertelement <2 x double> %112, double %89, i32 1
  store <2 x double> %113, ptr addrspace(1) %111, align 16
  %114 = getelementptr inbounds i8, ptr addrspace(1) %41, i64 32
  %115 = load <2 x double>, ptr addrspace(1) %114, align 16, !invariant.load !4
  %.unpack2668 = extractelement <2 x double> %115, i32 0
  %.unpack2869 = extractelement <2 x double> %115, i32 1
  %116 = fmul double %.unpack2668, 0x402B361E0E9094F8
  %117 = fmul double %.unpack2869, 0.000000e+00
  %118 = fsub double %116, %117
  %119 = fmul double %.unpack2869, 0x402B361E0E9094F8
  %120 = fmul double %.unpack2668, 0.000000e+00
  %121 = fadd double %120, %119
  %122 = getelementptr inbounds i8, ptr addrspace(1) %93, i64 32
  %123 = load <2 x double>, ptr addrspace(1) %122, align 16, !invariant.load !4
  %.unpack2974 = extractelement <2 x double> %123, i32 0
  %.unpack3175 = extractelement <2 x double> %123, i32 1
  %124 = fmul double %.unpack2974, 0x402B361E0E9094F8
  %125 = fmul double %.unpack3175, 0.000000e+00
  %126 = fsub double %124, %125
  %127 = fmul double %.unpack3175, 0x402B361E0E9094F8
  %128 = fmul double %.unpack2974, 0.000000e+00
  %129 = fadd double %128, %127
  %130 = fadd double %118, %126
  %131 = fadd double %121, %129
  %132 = getelementptr inbounds i8, ptr addrspace(1) %72, i64 32
  %133 = load <2 x double>, ptr addrspace(1) %132, align 16, !invariant.load !4
  %.unpack3284 = extractelement <2 x double> %133, i32 0
  %.unpack3485 = extractelement <2 x double> %133, i32 1
  %134 = fadd double %.unpack3284, %130
  %135 = fadd double %.unpack3485, %131
  %136 = getelementptr inbounds i8, ptr addrspace(1) %76, i64 32
  %137 = insertelement <2 x double> poison, double %134, i32 0
  %138 = insertelement <2 x double> %137, double %135, i32 1
  store <2 x double> %138, ptr addrspace(1) %136, align 32
  %139 = getelementptr inbounds i8, ptr addrspace(1) %79, i64 32
  %140 = insertelement <2 x double> poison, double %118, i32 0
  %141 = insertelement <2 x double> %140, double %121, i32 1
  store <2 x double> %141, ptr addrspace(1) %139, align 32
  %142 = getelementptr inbounds i8, ptr addrspace(1) %41, i64 48
  %143 = load <2 x double>, ptr addrspace(1) %142, align 16, !invariant.load !4
  %.unpack3970 = extractelement <2 x double> %143, i32 0
  %.unpack4171 = extractelement <2 x double> %143, i32 1
  %144 = fmul double %.unpack3970, 0x402B361E0E9094F8
  %145 = fmul double %.unpack4171, 0.000000e+00
  %146 = fsub double %144, %145
  %147 = fmul double %.unpack4171, 0x402B361E0E9094F8
  %148 = fmul double %.unpack3970, 0.000000e+00
  %149 = fadd double %148, %147
  %150 = getelementptr inbounds i8, ptr addrspace(1) %93, i64 48
  %151 = load <2 x double>, ptr addrspace(1) %150, align 16, !invariant.load !4
  %.unpack4276 = extractelement <2 x double> %151, i32 0
  %.unpack4477 = extractelement <2 x double> %151, i32 1
  %152 = fmul double %.unpack4276, 0x402B361E0E9094F8
  %153 = fmul double %.unpack4477, 0.000000e+00
  %154 = fsub double %152, %153
  %155 = fmul double %.unpack4477, 0x402B361E0E9094F8
  %156 = fmul double %.unpack4276, 0.000000e+00
  %157 = fadd double %156, %155
  %158 = fadd double %146, %154
  %159 = fadd double %149, %157
  %160 = getelementptr inbounds i8, ptr addrspace(1) %72, i64 48
  %161 = load <2 x double>, ptr addrspace(1) %160, align 16, !invariant.load !4
  %.unpack4586 = extractelement <2 x double> %161, i32 0
  %.unpack4787 = extractelement <2 x double> %161, i32 1
  %162 = fadd double %.unpack4586, %158
  %163 = fadd double %.unpack4787, %159
  %164 = getelementptr inbounds i8, ptr addrspace(1) %76, i64 48
  %165 = insertelement <2 x double> poison, double %162, i32 0
  %166 = insertelement <2 x double> %165, double %163, i32 1
  store <2 x double> %166, ptr addrspace(1) %164, align 16
  %167 = getelementptr inbounds i8, ptr addrspace(1) %79, i64 48
  %168 = insertelement <2 x double> poison, double %146, i32 0
  %169 = insertelement <2 x double> %168, double %149, i32 1
  store <2 x double> %169, ptr addrspace(1) %167, align 16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 3024}
!3 = !{i32 0, i32 128}
!4 = !{}
