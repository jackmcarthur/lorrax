; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @input_transpose_fusion(ptr noalias align 16 dereferenceable(243793920) %0, ptr noalias align 256 dereferenceable(121896960) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = urem i32 %3, 32
  %6 = icmp sle i32 %5, 23
  br i1 %6, label %7, label %32

7:                                                ; preds = %2
  %8 = urem i32 %4, 20
  %9 = mul i32 %8, 1536
  %10 = udiv i32 %4, 20
  %11 = mul i32 %10, 29760
  %12 = add i32 %9, %11
  %13 = udiv i32 %3, 32
  %14 = mul i32 %13, 48
  %15 = add i32 %12, %14
  %16 = add i32 %15, %5
  %17 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %16
  %18 = load { double, double }, ptr %17, align 8, !invariant.load !3
  %19 = mul i32 %5, 33
  %20 = add i32 %19, %13
  %21 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %20
  store { double, double } %18, ptr %21, align 8
  %22 = add i32 %16, 192
  %23 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %22
  %24 = load { double, double }, ptr %23, align 8, !invariant.load !3
  %25 = add i32 %20, 4
  %26 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %25
  store { double, double } %24, ptr %26, align 8
  %27 = add i32 %16, 384
  %28 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %27
  %29 = load { double, double }, ptr %28, align 8, !invariant.load !3
  %30 = add i32 %20, 8
  %31 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %30
  store { double, double } %29, ptr %31, align 8
  br label %32

32:                                               ; preds = %7, %2
  %33 = urem i32 %4, 20
  %34 = mul i32 %33, 8
  %35 = add i32 %34, 3
  %36 = icmp sle i32 %35, 154
  %37 = and i1 %6, %36
  br i1 %37, label %38, label %54

38:                                               ; preds = %32
  %39 = mul i32 %33, 1536
  %40 = udiv i32 %4, 20
  %41 = mul i32 %40, 29760
  %42 = add i32 %39, %41
  %43 = udiv i32 %3, 32
  %44 = mul i32 %43, 48
  %45 = add i32 %42, %44
  %46 = add i32 %45, %5
  %47 = add i32 %46, 576
  %48 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %47
  %49 = load { double, double }, ptr %48, align 8, !invariant.load !3
  %50 = mul i32 %5, 33
  %51 = add i32 %50, %43
  %52 = add i32 %51, 12
  %53 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %52
  store { double, double } %49, ptr %53, align 8
  br label %54

54:                                               ; preds = %38, %32
  %55 = add i32 %34, 4
  %56 = icmp sle i32 %55, 154
  %57 = and i1 %6, %56
  br i1 %57, label %58, label %74

58:                                               ; preds = %54
  %59 = mul i32 %33, 1536
  %60 = udiv i32 %4, 20
  %61 = mul i32 %60, 29760
  %62 = add i32 %59, %61
  %63 = udiv i32 %3, 32
  %64 = mul i32 %63, 48
  %65 = add i32 %62, %64
  %66 = add i32 %65, %5
  %67 = add i32 %66, 768
  %68 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %67
  %69 = load { double, double }, ptr %68, align 8, !invariant.load !3
  %70 = mul i32 %5, 33
  %71 = add i32 %70, %63
  %72 = add i32 %71, 16
  %73 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %72
  store { double, double } %69, ptr %73, align 8
  br label %74

74:                                               ; preds = %58, %54
  %75 = add i32 %34, 5
  %76 = icmp sle i32 %75, 154
  %77 = and i1 %6, %76
  br i1 %77, label %78, label %94

78:                                               ; preds = %74
  %79 = mul i32 %33, 1536
  %80 = udiv i32 %4, 20
  %81 = mul i32 %80, 29760
  %82 = add i32 %79, %81
  %83 = udiv i32 %3, 32
  %84 = mul i32 %83, 48
  %85 = add i32 %82, %84
  %86 = add i32 %85, %5
  %87 = add i32 %86, 960
  %88 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %87
  %89 = load { double, double }, ptr %88, align 8, !invariant.load !3
  %90 = mul i32 %5, 33
  %91 = add i32 %90, %83
  %92 = add i32 %91, 20
  %93 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %92
  store { double, double } %89, ptr %93, align 8
  br label %94

94:                                               ; preds = %78, %74
  %95 = add i32 %34, 6
  %96 = icmp sle i32 %95, 154
  %97 = and i1 %6, %96
  br i1 %97, label %98, label %114

98:                                               ; preds = %94
  %99 = mul i32 %33, 1536
  %100 = udiv i32 %4, 20
  %101 = mul i32 %100, 29760
  %102 = add i32 %99, %101
  %103 = udiv i32 %3, 32
  %104 = mul i32 %103, 48
  %105 = add i32 %102, %104
  %106 = add i32 %105, %5
  %107 = add i32 %106, 1152
  %108 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %107
  %109 = load { double, double }, ptr %108, align 8, !invariant.load !3
  %110 = mul i32 %5, 33
  %111 = add i32 %110, %103
  %112 = add i32 %111, 24
  %113 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %112
  store { double, double } %109, ptr %113, align 8
  br label %114

114:                                              ; preds = %98, %94
  %115 = add i32 %34, 7
  %116 = icmp sle i32 %115, 154
  %117 = and i1 %6, %116
  br i1 %117, label %118, label %134

118:                                              ; preds = %114
  %119 = mul i32 %33, 1536
  %120 = udiv i32 %4, 20
  %121 = mul i32 %120, 29760
  %122 = add i32 %119, %121
  %123 = udiv i32 %3, 32
  %124 = mul i32 %123, 48
  %125 = add i32 %122, %124
  %126 = add i32 %125, %5
  %127 = add i32 %126, 1344
  %128 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %127
  %129 = load { double, double }, ptr %128, align 8, !invariant.load !3
  %130 = mul i32 %5, 33
  %131 = add i32 %130, %123
  %132 = add i32 %131, 28
  %133 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %132
  store { double, double } %129, ptr %133, align 8
  br label %134

134:                                              ; preds = %118, %114
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %135 = mul i32 %33, 32
  %136 = add i32 %135, %5
  %137 = icmp sle i32 %136, 619
  br i1 %137, label %138, label %176

138:                                              ; preds = %134
  %139 = udiv i32 %3, 32
  %140 = mul i32 %139, 33
  %141 = add i32 %140, %5
  %142 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %141
  %143 = load { double, double }, ptr %142, align 8
  %144 = udiv i32 %4, 20
  %145 = mul i32 %144, 14880
  %146 = add i32 %135, %145
  %147 = mul i32 %139, 620
  %148 = add i32 %146, %147
  %149 = add i32 %148, %5
  %150 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %149
  store { double, double } %143, ptr %150, align 8
  %151 = add i32 %141, 132
  %152 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %151
  %153 = load { double, double }, ptr %152, align 8
  %154 = add i32 %149, 2480
  %155 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %154
  store { double, double } %153, ptr %155, align 8
  %156 = add i32 %141, 264
  %157 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %156
  %158 = load { double, double }, ptr %157, align 8
  %159 = add i32 %149, 4960
  %160 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %159
  store { double, double } %158, ptr %160, align 8
  %161 = add i32 %141, 396
  %162 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %161
  %163 = load { double, double }, ptr %162, align 8
  %164 = add i32 %149, 7440
  %165 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %164
  store { double, double } %163, ptr %165, align 8
  %166 = add i32 %141, 528
  %167 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %166
  %168 = load { double, double }, ptr %167, align 8
  %169 = add i32 %149, 9920
  %170 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %169
  store { double, double } %168, ptr %170, align 8
  %171 = add i32 %141, 660
  %172 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %171
  %173 = load { double, double }, ptr %172, align 8
  %174 = add i32 %149, 12400
  %175 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %174
  store { double, double } %173, ptr %175, align 8
  br label %176

176:                                              ; preds = %138, %134
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

define ptx_kernel void @loop_complex_fusion_1(ptr noalias align 16 dereferenceable(243793920) %0, ptr noalias align 256 dereferenceable(121896960) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = mul i32 %4, 2
  %6 = mul i32 %3, 256
  %7 = add i32 %5, %6
  %8 = udiv i32 %7, 155
  %9 = urem i32 %8, 2
  %10 = mul i32 %9, 310
  %11 = mul i32 %3, 128
  %12 = add i32 %11, %4
  %13 = udiv i32 %12, 155
  %14 = urem i32 %13, 24
  %15 = mul i32 %14, 620
  %16 = add i32 %10, %15
  %17 = udiv i32 %12, 3720
  %18 = mul i32 %17, 29760
  %19 = add i32 %16, %18
  %20 = mul i32 %4, 4
  %21 = mul i32 %3, 512
  %22 = add i32 %20, %21
  %23 = urem i32 %22, 310
  %24 = add i32 %19, %23
  %25 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %24
  %26 = load { double, double }, ptr %25, align 8, !invariant.load !3
  %27 = extractvalue { double, double } %26, 1
  %28 = extractvalue { double, double } %26, 0
  %29 = fneg double %27
  %30 = insertvalue { double, double } poison, double %28, 0
  %31 = insertvalue { double, double } %30, double %29, 1
  %32 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %22
  store { double, double } %31, ptr %32, align 8
  %33 = add i32 %22, 1
  %34 = urem i32 %33, 310
  %35 = add i32 %19, %34
  %36 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %35
  %37 = load { double, double }, ptr %36, align 8, !invariant.load !3
  %38 = extractvalue { double, double } %37, 1
  %39 = extractvalue { double, double } %37, 0
  %40 = fneg double %38
  %41 = insertvalue { double, double } poison, double %39, 0
  %42 = insertvalue { double, double } %41, double %40, 1
  %43 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %33
  store { double, double } %42, ptr %43, align 8
  %44 = add i32 %7, 1
  %45 = udiv i32 %44, 155
  %46 = urem i32 %45, 2
  %47 = mul i32 %46, 310
  %48 = add i32 %47, %15
  %49 = add i32 %48, %18
  %50 = add i32 %22, 2
  %51 = urem i32 %50, 310
  %52 = add i32 %49, %51
  %53 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %52
  %54 = load { double, double }, ptr %53, align 8, !invariant.load !3
  %55 = extractvalue { double, double } %54, 1
  %56 = extractvalue { double, double } %54, 0
  %57 = fneg double %55
  %58 = insertvalue { double, double } poison, double %56, 0
  %59 = insertvalue { double, double } %58, double %57, 1
  %60 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %50
  store { double, double } %59, ptr %60, align 8
  %61 = add i32 %22, 3
  %62 = udiv i32 %61, 310
  %63 = urem i32 %62, 2
  %64 = mul i32 %63, 310
  %65 = add i32 %64, %15
  %66 = add i32 %65, %18
  %67 = urem i32 %61, 310
  %68 = add i32 %66, %67
  %69 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %68
  %70 = load { double, double }, ptr %69, align 8, !invariant.load !3
  %71 = extractvalue { double, double } %70, 1
  %72 = extractvalue { double, double } %70, 0
  %73 = fneg double %71
  %74 = insertvalue { double, double } poison, double %72, 0
  %75 = insertvalue { double, double } %74, double %73, 1
  %76 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %61
  store { double, double } %75, ptr %76, align 8
  ret void
}

define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 dereferenceable(3149004800) %0, ptr noalias align 256 dereferenceable(787251200) %1, ptr noalias align 256 dereferenceable(3149004800) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %6 = mul i32 %5, 4
  %7 = mul i32 %4, 512
  %8 = add i32 %6, %7
  %9 = getelementptr inbounds [196812800 x { double, double }], ptr %0, i32 0, i32 %8
  %10 = load { double, double }, ptr %9, align 8, !invariant.load !3
  %11 = mul i32 %4, 128
  %12 = add i32 %11, %5
  %13 = udiv i32 %12, 155
  %14 = urem i32 %13, 310
  %15 = mul i32 %14, 310
  %16 = udiv i32 %12, 96100
  %17 = mul i32 %16, 96100
  %18 = add i32 %15, %17
  %19 = urem i32 %8, 310
  %20 = add i32 %18, %19
  %21 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %20
  %22 = load { double, double }, ptr %21, align 8, !invariant.load !3
  %23 = extractvalue { double, double } %10, 0
  %24 = extractvalue { double, double } %10, 1
  %25 = extractvalue { double, double } %22, 0
  %26 = extractvalue { double, double } %22, 1
  %27 = fmul double %23, %25
  %28 = fmul double %24, %26
  %29 = fsub double %27, %28
  %30 = fmul double %24, %25
  %31 = fmul double %23, %26
  %32 = fadd double %30, %31
  %33 = fmul double %29, 0xBFA6A09E667F3BCC
  %34 = fmul double %32, 0.000000e+00
  %35 = fsub double %33, %34
  %36 = fmul double %32, 0xBFA6A09E667F3BCC
  %37 = fmul double %29, 0.000000e+00
  %38 = fadd double %36, %37
  %39 = insertvalue { double, double } poison, double %35, 0
  %40 = insertvalue { double, double } %39, double %38, 1
  %41 = getelementptr inbounds [196812800 x { double, double }], ptr %2, i32 0, i32 %8
  store { double, double } %40, ptr %41, align 8
  %42 = add i32 %8, 1
  %43 = getelementptr inbounds [196812800 x { double, double }], ptr %0, i32 0, i32 %42
  %44 = load { double, double }, ptr %43, align 8, !invariant.load !3
  %45 = urem i32 %42, 310
  %46 = add i32 %18, %45
  %47 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %46
  %48 = load { double, double }, ptr %47, align 8, !invariant.load !3
  %49 = extractvalue { double, double } %44, 0
  %50 = extractvalue { double, double } %44, 1
  %51 = extractvalue { double, double } %48, 0
  %52 = extractvalue { double, double } %48, 1
  %53 = fmul double %49, %51
  %54 = fmul double %50, %52
  %55 = fsub double %53, %54
  %56 = fmul double %50, %51
  %57 = fmul double %49, %52
  %58 = fadd double %56, %57
  %59 = fmul double %55, 0xBFA6A09E667F3BCC
  %60 = fmul double %58, 0.000000e+00
  %61 = fsub double %59, %60
  %62 = fmul double %58, 0xBFA6A09E667F3BCC
  %63 = fmul double %55, 0.000000e+00
  %64 = fadd double %62, %63
  %65 = insertvalue { double, double } poison, double %61, 0
  %66 = insertvalue { double, double } %65, double %64, 1
  %67 = getelementptr inbounds [196812800 x { double, double }], ptr %2, i32 0, i32 %42
  store { double, double } %66, ptr %67, align 8
  %68 = add i32 %8, 2
  %69 = getelementptr inbounds [196812800 x { double, double }], ptr %0, i32 0, i32 %68
  %70 = load { double, double }, ptr %69, align 8, !invariant.load !3
  %71 = urem i32 %68, 310
  %72 = add i32 %18, %71
  %73 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %72
  %74 = load { double, double }, ptr %73, align 8, !invariant.load !3
  %75 = extractvalue { double, double } %70, 0
  %76 = extractvalue { double, double } %70, 1
  %77 = extractvalue { double, double } %74, 0
  %78 = extractvalue { double, double } %74, 1
  %79 = fmul double %75, %77
  %80 = fmul double %76, %78
  %81 = fsub double %79, %80
  %82 = fmul double %76, %77
  %83 = fmul double %75, %78
  %84 = fadd double %82, %83
  %85 = fmul double %81, 0xBFA6A09E667F3BCC
  %86 = fmul double %84, 0.000000e+00
  %87 = fsub double %85, %86
  %88 = fmul double %84, 0xBFA6A09E667F3BCC
  %89 = fmul double %81, 0.000000e+00
  %90 = fadd double %88, %89
  %91 = insertvalue { double, double } poison, double %87, 0
  %92 = insertvalue { double, double } %91, double %90, 1
  %93 = getelementptr inbounds [196812800 x { double, double }], ptr %2, i32 0, i32 %68
  store { double, double } %92, ptr %93, align 8
  %94 = add i32 %8, 3
  %95 = getelementptr inbounds [196812800 x { double, double }], ptr %0, i32 0, i32 %94
  %96 = load { double, double }, ptr %95, align 8, !invariant.load !3
  %97 = urem i32 %94, 310
  %98 = add i32 %18, %97
  %99 = getelementptr inbounds [49203200 x { double, double }], ptr %1, i32 0, i32 %98
  %100 = load { double, double }, ptr %99, align 8, !invariant.load !3
  %101 = extractvalue { double, double } %96, 0
  %102 = extractvalue { double, double } %96, 1
  %103 = extractvalue { double, double } %100, 0
  %104 = extractvalue { double, double } %100, 1
  %105 = fmul double %101, %103
  %106 = fmul double %102, %104
  %107 = fsub double %105, %106
  %108 = fmul double %102, %103
  %109 = fmul double %101, %104
  %110 = fadd double %108, %109
  %111 = fmul double %107, 0xBFA6A09E667F3BCC
  %112 = fmul double %110, 0.000000e+00
  %113 = fsub double %111, %112
  %114 = fmul double %110, 0xBFA6A09E667F3BCC
  %115 = fmul double %107, 0.000000e+00
  %116 = fadd double %114, %115
  %117 = insertvalue { double, double } poison, double %113, 0
  %118 = insertvalue { double, double } %117, double %116, 1
  %119 = getelementptr inbounds [196812800 x { double, double }], ptr %2, i32 0, i32 %94
  store { double, double } %118, ptr %119, align 8
  ret void
}

define ptx_kernel void @wrapped_slice(ptr noalias align 16 dereferenceable(243793920) %0, ptr noalias align 256 dereferenceable(121896960) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !4
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %5 = mul i32 %3, 128
  %6 = add i32 %5, %4
  %7 = urem i32 %6, 6
  %8 = mul i32 %7, 4
  %9 = udiv i32 %6, 6
  %10 = mul i32 %9, 48
  %11 = add i32 %8, %10
  %12 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %11
  %13 = load { double, double }, ptr %12, align 8, !invariant.load !3
  %14 = mul i32 %4, 4
  %15 = mul i32 %3, 512
  %16 = add i32 %14, %15
  %17 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %16
  store { double, double } %13, ptr %17, align 8
  %18 = add i32 %11, 1
  %19 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %18
  %20 = load { double, double }, ptr %19, align 8, !invariant.load !3
  %21 = add i32 %16, 1
  %22 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %21
  store { double, double } %20, ptr %22, align 8
  %23 = add i32 %11, 2
  %24 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %23
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !3
  %26 = add i32 %16, 2
  %27 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %26
  store { double, double } %25, ptr %27, align 8
  %28 = add i32 %11, 3
  %29 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %28
  %30 = load { double, double }, ptr %29, align 8, !invariant.load !3
  %31 = add i32 %16, 3
  %32 = getelementptr inbounds [7618560 x { double, double }], ptr %1, i32 0, i32 %31
  store { double, double } %30, ptr %32, align 8
  ret void
}

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 128}
!2 = !{i32 0, i32 10240}
!3 = !{}
!4 = !{i32 0, i32 14880}
!5 = !{i32 0, i32 384400}
