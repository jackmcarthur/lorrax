; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_add_multiply_fusion(ptr noalias align 16 dereferenceable(1179648) %0, ptr noalias align 16 dereferenceable(4718592) %1, ptr noalias align 256 dereferenceable(16) %2, ptr noalias align 256 dereferenceable(4) %3, ptr noalias align 256 dereferenceable(16) %4, ptr noalias align 16 dereferenceable(24772608) %5, ptr noalias align 256 dereferenceable(24772608) %6, ptr noalias align 256 dereferenceable(24772608) %7) #0 {
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %11 = mul i32 %9, 128
  %12 = add i32 %11, %10
  %13 = udiv i32 %12, 3
  %14 = urem i32 %13, 12
  %15 = getelementptr inbounds [1 x i32], ptr %3, i32 0, i32 0
  %16 = load i32, ptr %15, align 4, !invariant.load !3
  %17 = call i32 @llvm.umin.i32(i32 %16, i32 3)
  %18 = getelementptr inbounds [4 x i32], ptr %4, i32 0, i32 %17
  %19 = load i32, ptr %18, align 4, !invariant.load !3
  %20 = call i32 @llvm.smin.i32(i32 %19, i32 12)
  %21 = call i32 @llvm.smax.i32(i32 %20, i32 0)
  %22 = add i32 %14, %21
  %23 = getelementptr inbounds [4 x i32], ptr %2, i32 0, i32 %17
  %24 = load i32, ptr %23, align 4, !invariant.load !3
  %25 = call i32 @llvm.smin.i32(i32 %24, i32 12)
  %26 = call i32 @llvm.smax.i32(i32 %25, i32 0)
  %27 = urem i32 %12, 3
  %28 = mul i32 %27, 4
  %29 = mul i32 %10, 4
  %30 = mul i32 %9, 512
  %31 = add i32 %29, %30
  %32 = getelementptr inbounds [1548288 x { double, double }], ptr %5, i32 0, i32 %31
  %33 = load { double, double }, ptr %32, align 8, !invariant.load !3
  %34 = extractvalue { double, double } %33, 0
  %35 = extractvalue { double, double } %33, 1
  %36 = fmul double %34, 0x402B361E0E9094F8
  %37 = fmul double %35, 0.000000e+00
  %38 = fsub double %36, %37
  %39 = fmul double %35, 0x402B361E0E9094F8
  %40 = fmul double %34, 0.000000e+00
  %41 = fadd double %39, %40
  %42 = insertvalue { double, double } poison, double %38, 0
  %43 = insertvalue { double, double } %42, double %41, 1
  %44 = add i32 %28, %26
  %45 = udiv i32 %12, 36
  %46 = urem i32 %45, 512
  %47 = mul i32 %46, 576
  %48 = mul i32 %22, 24
  %49 = add i32 %47, %48
  %50 = add i32 %49, %44
  %51 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %50
  %52 = load { double, double }, ptr %51, align 8, !invariant.load !3
  %53 = extractvalue { double, double } %52, 0
  %54 = extractvalue { double, double } %52, 1
  %55 = fmul double %53, 0x402B361E0E9094F8
  %56 = fmul double %54, 0.000000e+00
  %57 = fsub double %55, %56
  %58 = fmul double %54, 0x402B361E0E9094F8
  %59 = fmul double %53, 0.000000e+00
  %60 = fadd double %58, %59
  %61 = fadd double %38, %57
  %62 = fadd double %41, %60
  %63 = mul i32 %14, 12
  %64 = add i32 %28, %63
  %65 = mul i32 %46, 144
  %66 = add i32 %64, %65
  %67 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %66
  %68 = load { double, double }, ptr %67, align 8, !invariant.load !3
  %69 = extractvalue { double, double } %68, 0
  %70 = fadd double %61, %69
  %71 = extractvalue { double, double } %68, 1
  %72 = fadd double %62, %71
  %73 = insertvalue { double, double } poison, double %70, 0
  %74 = insertvalue { double, double } %73, double %72, 1
  %75 = getelementptr inbounds [1548288 x { double, double }], ptr %6, i32 0, i32 %31
  store { double, double } %74, ptr %75, align 8
  %76 = getelementptr inbounds [1548288 x { double, double }], ptr %7, i32 0, i32 %31
  store { double, double } %43, ptr %76, align 8
  %77 = add i32 %28, 1
  %78 = add i32 %31, 1
  %79 = getelementptr inbounds [1548288 x { double, double }], ptr %5, i32 0, i32 %78
  %80 = load { double, double }, ptr %79, align 8, !invariant.load !3
  %81 = extractvalue { double, double } %80, 0
  %82 = extractvalue { double, double } %80, 1
  %83 = fmul double %81, 0x402B361E0E9094F8
  %84 = fmul double %82, 0.000000e+00
  %85 = fsub double %83, %84
  %86 = fmul double %82, 0x402B361E0E9094F8
  %87 = fmul double %81, 0.000000e+00
  %88 = fadd double %86, %87
  %89 = insertvalue { double, double } poison, double %85, 0
  %90 = insertvalue { double, double } %89, double %88, 1
  %91 = add i32 %77, %26
  %92 = add i32 %49, %91
  %93 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %92
  %94 = load { double, double }, ptr %93, align 8, !invariant.load !3
  %95 = extractvalue { double, double } %94, 0
  %96 = extractvalue { double, double } %94, 1
  %97 = fmul double %95, 0x402B361E0E9094F8
  %98 = fmul double %96, 0.000000e+00
  %99 = fsub double %97, %98
  %100 = fmul double %96, 0x402B361E0E9094F8
  %101 = fmul double %95, 0.000000e+00
  %102 = fadd double %100, %101
  %103 = fadd double %85, %99
  %104 = fadd double %88, %102
  %105 = add i32 %66, 1
  %106 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %105
  %107 = load { double, double }, ptr %106, align 8, !invariant.load !3
  %108 = extractvalue { double, double } %107, 0
  %109 = fadd double %103, %108
  %110 = extractvalue { double, double } %107, 1
  %111 = fadd double %104, %110
  %112 = insertvalue { double, double } poison, double %109, 0
  %113 = insertvalue { double, double } %112, double %111, 1
  %114 = getelementptr inbounds [1548288 x { double, double }], ptr %6, i32 0, i32 %78
  store { double, double } %113, ptr %114, align 8
  %115 = getelementptr inbounds [1548288 x { double, double }], ptr %7, i32 0, i32 %78
  store { double, double } %90, ptr %115, align 8
  %116 = add i32 %28, 2
  %117 = add i32 %31, 2
  %118 = getelementptr inbounds [1548288 x { double, double }], ptr %5, i32 0, i32 %117
  %119 = load { double, double }, ptr %118, align 8, !invariant.load !3
  %120 = extractvalue { double, double } %119, 0
  %121 = extractvalue { double, double } %119, 1
  %122 = fmul double %120, 0x402B361E0E9094F8
  %123 = fmul double %121, 0.000000e+00
  %124 = fsub double %122, %123
  %125 = fmul double %121, 0x402B361E0E9094F8
  %126 = fmul double %120, 0.000000e+00
  %127 = fadd double %125, %126
  %128 = insertvalue { double, double } poison, double %124, 0
  %129 = insertvalue { double, double } %128, double %127, 1
  %130 = add i32 %116, %26
  %131 = add i32 %49, %130
  %132 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %131
  %133 = load { double, double }, ptr %132, align 8, !invariant.load !3
  %134 = extractvalue { double, double } %133, 0
  %135 = extractvalue { double, double } %133, 1
  %136 = fmul double %134, 0x402B361E0E9094F8
  %137 = fmul double %135, 0.000000e+00
  %138 = fsub double %136, %137
  %139 = fmul double %135, 0x402B361E0E9094F8
  %140 = fmul double %134, 0.000000e+00
  %141 = fadd double %139, %140
  %142 = fadd double %124, %138
  %143 = fadd double %127, %141
  %144 = add i32 %66, 2
  %145 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %144
  %146 = load { double, double }, ptr %145, align 8, !invariant.load !3
  %147 = extractvalue { double, double } %146, 0
  %148 = fadd double %142, %147
  %149 = extractvalue { double, double } %146, 1
  %150 = fadd double %143, %149
  %151 = insertvalue { double, double } poison, double %148, 0
  %152 = insertvalue { double, double } %151, double %150, 1
  %153 = getelementptr inbounds [1548288 x { double, double }], ptr %6, i32 0, i32 %117
  store { double, double } %152, ptr %153, align 8
  %154 = getelementptr inbounds [1548288 x { double, double }], ptr %7, i32 0, i32 %117
  store { double, double } %129, ptr %154, align 8
  %155 = add i32 %28, 3
  %156 = add i32 %31, 3
  %157 = getelementptr inbounds [1548288 x { double, double }], ptr %5, i32 0, i32 %156
  %158 = load { double, double }, ptr %157, align 8, !invariant.load !3
  %159 = extractvalue { double, double } %158, 0
  %160 = extractvalue { double, double } %158, 1
  %161 = fmul double %159, 0x402B361E0E9094F8
  %162 = fmul double %160, 0.000000e+00
  %163 = fsub double %161, %162
  %164 = fmul double %160, 0x402B361E0E9094F8
  %165 = fmul double %159, 0.000000e+00
  %166 = fadd double %164, %165
  %167 = insertvalue { double, double } poison, double %163, 0
  %168 = insertvalue { double, double } %167, double %166, 1
  %169 = add i32 %155, %26
  %170 = add i32 %49, %169
  %171 = getelementptr inbounds [294912 x { double, double }], ptr %1, i32 0, i32 %170
  %172 = load { double, double }, ptr %171, align 8, !invariant.load !3
  %173 = extractvalue { double, double } %172, 0
  %174 = extractvalue { double, double } %172, 1
  %175 = fmul double %173, 0x402B361E0E9094F8
  %176 = fmul double %174, 0.000000e+00
  %177 = fsub double %175, %176
  %178 = fmul double %174, 0x402B361E0E9094F8
  %179 = fmul double %173, 0.000000e+00
  %180 = fadd double %178, %179
  %181 = fadd double %163, %177
  %182 = fadd double %166, %180
  %183 = add i32 %66, 3
  %184 = getelementptr inbounds [73728 x { double, double }], ptr %0, i32 0, i32 %183
  %185 = load { double, double }, ptr %184, align 8, !invariant.load !3
  %186 = extractvalue { double, double } %185, 0
  %187 = fadd double %181, %186
  %188 = extractvalue { double, double } %185, 1
  %189 = fadd double %182, %188
  %190 = insertvalue { double, double } poison, double %187, 0
  %191 = insertvalue { double, double } %190, double %189, 1
  %192 = getelementptr inbounds [1548288 x { double, double }], ptr %6, i32 0, i32 %156
  store { double, double } %191, ptr %192, align 8
  %193 = getelementptr inbounds [1548288 x { double, double }], ptr %7, i32 0, i32 %156
  store { double, double } %168, ptr %193, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 3024}
!2 = !{i32 0, i32 128}
!3 = !{}
