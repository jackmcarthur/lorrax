; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_compare_fusion(ptr noalias align 256 dereferenceable(16) %0, ptr noalias align 256 dereferenceable(4) %1, ptr noalias align 256 dereferenceable(16) %2, ptr noalias align 256 dereferenceable(96100) %3) #0 {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %7 = mul i32 %5, 128
  %8 = add i32 %7, %6
  %9 = icmp sle i32 %8, 96099
  br i1 %9, label %10, label %31

10:                                               ; preds = %4
  %11 = udiv i32 %8, 310
  %12 = urem i32 %8, 310
  %13 = sext i32 %11 to i64
  %14 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  %15 = load i32, ptr %14, align 4, !invariant.load !3
  %16 = call i32 @llvm.umin.i32(i32 %15, i32 3)
  %17 = getelementptr inbounds [4 x i32], ptr %0, i32 0, i32 %16
  %18 = load i32, ptr %17, align 4, !invariant.load !3
  %19 = mul i32 %18, 310
  %20 = sext i32 %19 to i64
  %21 = sext i32 %12 to i64
  %22 = getelementptr inbounds [4 x i32], ptr %2, i32 0, i32 %16
  %23 = load i32, ptr %22, align 4, !invariant.load !3
  %24 = mul i32 %23, 310
  %25 = sext i32 %24 to i64
  %26 = add i64 %13, %20
  %27 = add i64 %21, %25
  %28 = icmp eq i64 %26, %27
  %29 = zext i1 %28 to i8
  %30 = getelementptr inbounds [96100 x i8], ptr %3, i32 0, i32 %8
  store i8 %29, ptr %30, align 1
  br label %31

31:                                               ; preds = %10, %4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #2

define ptx_kernel void @input_reduce_fusion(ptr noalias align 256 dereferenceable(96100) %0, ptr noalias align 16 dereferenceable(1537600) %1, ptr noalias align 256 dereferenceable(4960) %2) #3 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !4
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !5
  %6 = mul i32 %5, 8
  %7 = udiv i32 %4, 32
  %8 = add i32 %6, %7
  %9 = icmp sle i32 %8, 309
  br i1 %9, label %10, label %87

10:                                               ; preds = %3
  %11 = mul i32 %7, 310
  %12 = mul i32 %5, 2480
  %13 = add i32 %11, %12
  %14 = urem i32 %4, 32
  %15 = add i32 %13, %14
  %16 = getelementptr inbounds [96100 x i8], ptr %0, i32 0, i32 %15
  %17 = load i8, ptr %16, align 1, !invariant.load !3
  %18 = getelementptr inbounds [96100 x { double, double }], ptr %1, i32 0, i32 %15
  %19 = load { double, double }, ptr %18, align 8, !invariant.load !3
  %20 = trunc i8 %17 to i1
  %21 = select i1 %20, { double, double } %19, { double, double } zeroinitializer
  %22 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } zeroinitializer, { double, double } %21)
  %23 = add i32 %15, 32
  %24 = getelementptr inbounds [96100 x i8], ptr %0, i32 0, i32 %23
  %25 = load i8, ptr %24, align 1, !invariant.load !3
  %26 = getelementptr inbounds [96100 x { double, double }], ptr %1, i32 0, i32 %23
  %27 = load { double, double }, ptr %26, align 8, !invariant.load !3
  %28 = trunc i8 %25 to i1
  %29 = select i1 %28, { double, double } %27, { double, double } zeroinitializer
  %30 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %22, { double, double } %29)
  %31 = add i32 %15, 64
  %32 = getelementptr inbounds [96100 x i8], ptr %0, i32 0, i32 %31
  %33 = load i8, ptr %32, align 1, !invariant.load !3
  %34 = getelementptr inbounds [96100 x { double, double }], ptr %1, i32 0, i32 %31
  %35 = load { double, double }, ptr %34, align 8, !invariant.load !3
  %36 = trunc i8 %33 to i1
  %37 = select i1 %36, { double, double } %35, { double, double } zeroinitializer
  %38 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %30, { double, double } %37)
  %39 = add i32 %15, 96
  %40 = getelementptr inbounds [96100 x i8], ptr %0, i32 0, i32 %39
  %41 = load i8, ptr %40, align 1, !invariant.load !3
  %42 = getelementptr inbounds [96100 x { double, double }], ptr %1, i32 0, i32 %39
  %43 = load { double, double }, ptr %42, align 8, !invariant.load !3
  %44 = trunc i8 %41 to i1
  %45 = select i1 %44, { double, double } %43, { double, double } zeroinitializer
  %46 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %38, { double, double } %45)
  %47 = add i32 %15, 128
  %48 = getelementptr inbounds [96100 x i8], ptr %0, i32 0, i32 %47
  %49 = load i8, ptr %48, align 1, !invariant.load !3
  %50 = getelementptr inbounds [96100 x { double, double }], ptr %1, i32 0, i32 %47
  %51 = load { double, double }, ptr %50, align 8, !invariant.load !3
  %52 = trunc i8 %49 to i1
  %53 = select i1 %52, { double, double } %51, { double, double } zeroinitializer
  %54 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %46, { double, double } %53)
  %55 = add i32 %15, 160
  %56 = getelementptr inbounds [96100 x i8], ptr %0, i32 0, i32 %55
  %57 = load i8, ptr %56, align 1, !invariant.load !3
  %58 = getelementptr inbounds [96100 x { double, double }], ptr %1, i32 0, i32 %55
  %59 = load { double, double }, ptr %58, align 8, !invariant.load !3
  %60 = trunc i8 %57 to i1
  %61 = select i1 %60, { double, double } %59, { double, double } zeroinitializer
  %62 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %54, { double, double } %61)
  %63 = add i32 %15, 192
  %64 = getelementptr inbounds [96100 x i8], ptr %0, i32 0, i32 %63
  %65 = load i8, ptr %64, align 1, !invariant.load !3
  %66 = getelementptr inbounds [96100 x { double, double }], ptr %1, i32 0, i32 %63
  %67 = load { double, double }, ptr %66, align 8, !invariant.load !3
  %68 = trunc i8 %65 to i1
  %69 = select i1 %68, { double, double } %67, { double, double } zeroinitializer
  %70 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %62, { double, double } %69)
  %71 = add i32 %15, 224
  %72 = getelementptr inbounds [96100 x i8], ptr %0, i32 0, i32 %71
  %73 = load i8, ptr %72, align 1, !invariant.load !3
  %74 = getelementptr inbounds [96100 x { double, double }], ptr %1, i32 0, i32 %71
  %75 = load { double, double }, ptr %74, align 8, !invariant.load !3
  %76 = trunc i8 %73 to i1
  %77 = select i1 %76, { double, double } %75, { double, double } zeroinitializer
  %78 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %70, { double, double } %77)
  %79 = add i32 %15, 256
  %80 = getelementptr inbounds [96100 x i8], ptr %0, i32 0, i32 %79
  %81 = load i8, ptr %80, align 1, !invariant.load !3
  %82 = getelementptr inbounds [96100 x { double, double }], ptr %1, i32 0, i32 %79
  %83 = load { double, double }, ptr %82, align 8, !invariant.load !3
  %84 = trunc i8 %81 to i1
  %85 = select i1 %84, { double, double } %83, { double, double } zeroinitializer
  %86 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %78, { double, double } %85)
  br label %88

87:                                               ; preds = %3
  br label %88

88:                                               ; preds = %10, %87
  %89 = phi { double, double } [ zeroinitializer, %87 ], [ %86, %10 ]
  br label %90

90:                                               ; preds = %88
  %91 = urem i32 %4, 32
  %92 = icmp sle i32 %91, 21
  %93 = and i1 %9, %92
  br i1 %93, label %94, label %107

94:                                               ; preds = %90
  %95 = mul i32 %7, 310
  %96 = mul i32 %5, 2480
  %97 = add i32 %95, %96
  %98 = add i32 %97, %91
  %99 = add i32 %98, 288
  %100 = getelementptr inbounds [96100 x i8], ptr %0, i32 0, i32 %99
  %101 = load i8, ptr %100, align 1, !invariant.load !3
  %102 = getelementptr inbounds [96100 x { double, double }], ptr %1, i32 0, i32 %99
  %103 = load { double, double }, ptr %102, align 8, !invariant.load !3
  %104 = trunc i8 %101 to i1
  %105 = select i1 %104, { double, double } %103, { double, double } zeroinitializer
  %106 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %89, { double, double } %105)
  br label %108

107:                                              ; preds = %90
  br label %108

108:                                              ; preds = %94, %107
  %109 = phi { double, double } [ %89, %107 ], [ %106, %94 ]
  br label %110

110:                                              ; preds = %108
  %111 = extractvalue { double, double } %109, 0
  %112 = bitcast double %111 to i64
  %113 = bitcast i64 %112 to <2 x i32>
  %114 = extractelement <2 x i32> %113, i32 0
  %115 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %114, i32 16, i32 31)
  %116 = insertelement <2 x i32> undef, i32 %115, i32 0
  %117 = extractelement <2 x i32> %113, i32 1
  %118 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %117, i32 16, i32 31)
  %119 = insertelement <2 x i32> %116, i32 %118, i32 1
  %120 = bitcast <2 x i32> %119 to double
  %121 = extractvalue { double, double } %109, 1
  %122 = bitcast double %121 to i64
  %123 = bitcast i64 %122 to <2 x i32>
  %124 = extractelement <2 x i32> %123, i32 0
  %125 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %124, i32 16, i32 31)
  %126 = insertelement <2 x i32> undef, i32 %125, i32 0
  %127 = extractelement <2 x i32> %123, i32 1
  %128 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %127, i32 16, i32 31)
  %129 = insertelement <2 x i32> %126, i32 %128, i32 1
  %130 = bitcast <2 x i32> %129 to double
  %131 = insertvalue { double, double } poison, double %120, 0
  %132 = insertvalue { double, double } %131, double %130, 1
  %133 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %109, { double, double } %132)
  %134 = extractvalue { double, double } %133, 0
  %135 = bitcast double %134 to i64
  %136 = bitcast i64 %135 to <2 x i32>
  %137 = extractelement <2 x i32> %136, i32 0
  %138 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %137, i32 8, i32 31)
  %139 = insertelement <2 x i32> undef, i32 %138, i32 0
  %140 = extractelement <2 x i32> %136, i32 1
  %141 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %140, i32 8, i32 31)
  %142 = insertelement <2 x i32> %139, i32 %141, i32 1
  %143 = bitcast <2 x i32> %142 to double
  %144 = extractvalue { double, double } %133, 1
  %145 = bitcast double %144 to i64
  %146 = bitcast i64 %145 to <2 x i32>
  %147 = extractelement <2 x i32> %146, i32 0
  %148 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %147, i32 8, i32 31)
  %149 = insertelement <2 x i32> undef, i32 %148, i32 0
  %150 = extractelement <2 x i32> %146, i32 1
  %151 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %150, i32 8, i32 31)
  %152 = insertelement <2 x i32> %149, i32 %151, i32 1
  %153 = bitcast <2 x i32> %152 to double
  %154 = insertvalue { double, double } poison, double %143, 0
  %155 = insertvalue { double, double } %154, double %153, 1
  %156 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %133, { double, double } %155)
  %157 = extractvalue { double, double } %156, 0
  %158 = bitcast double %157 to i64
  %159 = bitcast i64 %158 to <2 x i32>
  %160 = extractelement <2 x i32> %159, i32 0
  %161 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %160, i32 4, i32 31)
  %162 = insertelement <2 x i32> undef, i32 %161, i32 0
  %163 = extractelement <2 x i32> %159, i32 1
  %164 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %163, i32 4, i32 31)
  %165 = insertelement <2 x i32> %162, i32 %164, i32 1
  %166 = bitcast <2 x i32> %165 to double
  %167 = extractvalue { double, double } %156, 1
  %168 = bitcast double %167 to i64
  %169 = bitcast i64 %168 to <2 x i32>
  %170 = extractelement <2 x i32> %169, i32 0
  %171 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %170, i32 4, i32 31)
  %172 = insertelement <2 x i32> undef, i32 %171, i32 0
  %173 = extractelement <2 x i32> %169, i32 1
  %174 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %173, i32 4, i32 31)
  %175 = insertelement <2 x i32> %172, i32 %174, i32 1
  %176 = bitcast <2 x i32> %175 to double
  %177 = insertvalue { double, double } poison, double %166, 0
  %178 = insertvalue { double, double } %177, double %176, 1
  %179 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %156, { double, double } %178)
  %180 = extractvalue { double, double } %179, 0
  %181 = bitcast double %180 to i64
  %182 = bitcast i64 %181 to <2 x i32>
  %183 = extractelement <2 x i32> %182, i32 0
  %184 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %183, i32 2, i32 31)
  %185 = insertelement <2 x i32> undef, i32 %184, i32 0
  %186 = extractelement <2 x i32> %182, i32 1
  %187 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %186, i32 2, i32 31)
  %188 = insertelement <2 x i32> %185, i32 %187, i32 1
  %189 = bitcast <2 x i32> %188 to double
  %190 = extractvalue { double, double } %179, 1
  %191 = bitcast double %190 to i64
  %192 = bitcast i64 %191 to <2 x i32>
  %193 = extractelement <2 x i32> %192, i32 0
  %194 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %193, i32 2, i32 31)
  %195 = insertelement <2 x i32> undef, i32 %194, i32 0
  %196 = extractelement <2 x i32> %192, i32 1
  %197 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %196, i32 2, i32 31)
  %198 = insertelement <2 x i32> %195, i32 %197, i32 1
  %199 = bitcast <2 x i32> %198 to double
  %200 = insertvalue { double, double } poison, double %189, 0
  %201 = insertvalue { double, double } %200, double %199, 1
  %202 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %179, { double, double } %201)
  %203 = extractvalue { double, double } %202, 0
  %204 = bitcast double %203 to i64
  %205 = bitcast i64 %204 to <2 x i32>
  %206 = extractelement <2 x i32> %205, i32 0
  %207 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %206, i32 1, i32 31)
  %208 = insertelement <2 x i32> undef, i32 %207, i32 0
  %209 = extractelement <2 x i32> %205, i32 1
  %210 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %209, i32 1, i32 31)
  %211 = insertelement <2 x i32> %208, i32 %210, i32 1
  %212 = bitcast <2 x i32> %211 to double
  %213 = extractvalue { double, double } %202, 1
  %214 = bitcast double %213 to i64
  %215 = bitcast i64 %214 to <2 x i32>
  %216 = extractelement <2 x i32> %215, i32 0
  %217 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %216, i32 1, i32 31)
  %218 = insertelement <2 x i32> undef, i32 %217, i32 0
  %219 = extractelement <2 x i32> %215, i32 1
  %220 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %219, i32 1, i32 31)
  %221 = insertelement <2 x i32> %218, i32 %220, i32 1
  %222 = bitcast <2 x i32> %221 to double
  %223 = insertvalue { double, double } poison, double %212, 0
  %224 = insertvalue { double, double } %223, double %222, 1
  %225 = call { double, double } @region_0_1_reduce_sum_2_0({ double, double } %202, { double, double } %224)
  %226 = icmp eq i32 %91, 0
  %227 = and i1 %226, %9
  %228 = icmp sle i32 %4, 224
  %229 = and i1 %227, %228
  br i1 %229, label %230, label %232

230:                                              ; preds = %110
  %231 = getelementptr inbounds [310 x { double, double }], ptr %2, i32 0, i32 %8
  store { double, double } %225, ptr %231, align 8
  br label %232

232:                                              ; preds = %230, %110
  ret void
}

define internal { double, double } @region_0_1_reduce_sum_2_0({ double, double } %0, { double, double } %1) {
  %3 = extractvalue { double, double } %0, 0
  %4 = extractvalue { double, double } %1, 0
  %5 = fadd double %3, %4
  %6 = extractvalue { double, double } %0, 1
  %7 = extractvalue { double, double } %1, 1
  %8 = fadd double %6, %7
  %9 = insertvalue { double, double } poison, double %5, 0
  %10 = insertvalue { double, double } %9, double %8, 1
  ret { double, double } %10
}

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #4

define ptx_kernel void @input_reduce_fusion_1(ptr noalias align 256 dereferenceable(4960) %0, ptr noalias align 256 dereferenceable(16) %1) #5 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !6
  %4 = getelementptr inbounds [310 x { double, double }], ptr %0, i32 0, i32 %3
  %5 = load { double, double }, ptr %4, align 8, !invariant.load !3
  %6 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } zeroinitializer, { double, double } %5)
  %7 = add i32 %3, 32
  %8 = getelementptr inbounds [310 x { double, double }], ptr %0, i32 0, i32 %7
  %9 = load { double, double }, ptr %8, align 8, !invariant.load !3
  %10 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %6, { double, double } %9)
  %11 = add i32 %3, 64
  %12 = getelementptr inbounds [310 x { double, double }], ptr %0, i32 0, i32 %11
  %13 = load { double, double }, ptr %12, align 8, !invariant.load !3
  %14 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %10, { double, double } %13)
  %15 = add i32 %3, 96
  %16 = getelementptr inbounds [310 x { double, double }], ptr %0, i32 0, i32 %15
  %17 = load { double, double }, ptr %16, align 8, !invariant.load !3
  %18 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %14, { double, double } %17)
  %19 = add i32 %3, 128
  %20 = getelementptr inbounds [310 x { double, double }], ptr %0, i32 0, i32 %19
  %21 = load { double, double }, ptr %20, align 8, !invariant.load !3
  %22 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %18, { double, double } %21)
  %23 = add i32 %3, 160
  %24 = getelementptr inbounds [310 x { double, double }], ptr %0, i32 0, i32 %23
  %25 = load { double, double }, ptr %24, align 8, !invariant.load !3
  %26 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %22, { double, double } %25)
  %27 = add i32 %3, 192
  %28 = getelementptr inbounds [310 x { double, double }], ptr %0, i32 0, i32 %27
  %29 = load { double, double }, ptr %28, align 8, !invariant.load !3
  %30 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %26, { double, double } %29)
  %31 = add i32 %3, 224
  %32 = getelementptr inbounds [310 x { double, double }], ptr %0, i32 0, i32 %31
  %33 = load { double, double }, ptr %32, align 8, !invariant.load !3
  %34 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %30, { double, double } %33)
  %35 = add i32 %3, 256
  %36 = getelementptr inbounds [310 x { double, double }], ptr %0, i32 0, i32 %35
  %37 = load { double, double }, ptr %36, align 8, !invariant.load !3
  %38 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %34, { double, double } %37)
  %39 = icmp sle i32 %3, 21
  br i1 %39, label %40, label %45

40:                                               ; preds = %2
  %41 = add i32 %3, 288
  %42 = getelementptr inbounds [310 x { double, double }], ptr %0, i32 0, i32 %41
  %43 = load { double, double }, ptr %42, align 8, !invariant.load !3
  %44 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %38, { double, double } %43)
  br label %46

45:                                               ; preds = %2
  br label %46

46:                                               ; preds = %40, %45
  %47 = phi { double, double } [ %38, %45 ], [ %44, %40 ]
  br label %48

48:                                               ; preds = %46
  %49 = extractvalue { double, double } %47, 0
  %50 = bitcast double %49 to i64
  %51 = bitcast i64 %50 to <2 x i32>
  %52 = extractelement <2 x i32> %51, i32 0
  %53 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %52, i32 16, i32 31)
  %54 = insertelement <2 x i32> undef, i32 %53, i32 0
  %55 = extractelement <2 x i32> %51, i32 1
  %56 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %55, i32 16, i32 31)
  %57 = insertelement <2 x i32> %54, i32 %56, i32 1
  %58 = bitcast <2 x i32> %57 to double
  %59 = extractvalue { double, double } %47, 1
  %60 = bitcast double %59 to i64
  %61 = bitcast i64 %60 to <2 x i32>
  %62 = extractelement <2 x i32> %61, i32 0
  %63 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %62, i32 16, i32 31)
  %64 = insertelement <2 x i32> undef, i32 %63, i32 0
  %65 = extractelement <2 x i32> %61, i32 1
  %66 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %65, i32 16, i32 31)
  %67 = insertelement <2 x i32> %64, i32 %66, i32 1
  %68 = bitcast <2 x i32> %67 to double
  %69 = insertvalue { double, double } poison, double %58, 0
  %70 = insertvalue { double, double } %69, double %68, 1
  %71 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %47, { double, double } %70)
  %72 = extractvalue { double, double } %71, 0
  %73 = bitcast double %72 to i64
  %74 = bitcast i64 %73 to <2 x i32>
  %75 = extractelement <2 x i32> %74, i32 0
  %76 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %75, i32 8, i32 31)
  %77 = insertelement <2 x i32> undef, i32 %76, i32 0
  %78 = extractelement <2 x i32> %74, i32 1
  %79 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %78, i32 8, i32 31)
  %80 = insertelement <2 x i32> %77, i32 %79, i32 1
  %81 = bitcast <2 x i32> %80 to double
  %82 = extractvalue { double, double } %71, 1
  %83 = bitcast double %82 to i64
  %84 = bitcast i64 %83 to <2 x i32>
  %85 = extractelement <2 x i32> %84, i32 0
  %86 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %85, i32 8, i32 31)
  %87 = insertelement <2 x i32> undef, i32 %86, i32 0
  %88 = extractelement <2 x i32> %84, i32 1
  %89 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %88, i32 8, i32 31)
  %90 = insertelement <2 x i32> %87, i32 %89, i32 1
  %91 = bitcast <2 x i32> %90 to double
  %92 = insertvalue { double, double } poison, double %81, 0
  %93 = insertvalue { double, double } %92, double %91, 1
  %94 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %71, { double, double } %93)
  %95 = extractvalue { double, double } %94, 0
  %96 = bitcast double %95 to i64
  %97 = bitcast i64 %96 to <2 x i32>
  %98 = extractelement <2 x i32> %97, i32 0
  %99 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %98, i32 4, i32 31)
  %100 = insertelement <2 x i32> undef, i32 %99, i32 0
  %101 = extractelement <2 x i32> %97, i32 1
  %102 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %101, i32 4, i32 31)
  %103 = insertelement <2 x i32> %100, i32 %102, i32 1
  %104 = bitcast <2 x i32> %103 to double
  %105 = extractvalue { double, double } %94, 1
  %106 = bitcast double %105 to i64
  %107 = bitcast i64 %106 to <2 x i32>
  %108 = extractelement <2 x i32> %107, i32 0
  %109 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %108, i32 4, i32 31)
  %110 = insertelement <2 x i32> undef, i32 %109, i32 0
  %111 = extractelement <2 x i32> %107, i32 1
  %112 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %111, i32 4, i32 31)
  %113 = insertelement <2 x i32> %110, i32 %112, i32 1
  %114 = bitcast <2 x i32> %113 to double
  %115 = insertvalue { double, double } poison, double %104, 0
  %116 = insertvalue { double, double } %115, double %114, 1
  %117 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %94, { double, double } %116)
  %118 = extractvalue { double, double } %117, 0
  %119 = bitcast double %118 to i64
  %120 = bitcast i64 %119 to <2 x i32>
  %121 = extractelement <2 x i32> %120, i32 0
  %122 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %121, i32 2, i32 31)
  %123 = insertelement <2 x i32> undef, i32 %122, i32 0
  %124 = extractelement <2 x i32> %120, i32 1
  %125 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %124, i32 2, i32 31)
  %126 = insertelement <2 x i32> %123, i32 %125, i32 1
  %127 = bitcast <2 x i32> %126 to double
  %128 = extractvalue { double, double } %117, 1
  %129 = bitcast double %128 to i64
  %130 = bitcast i64 %129 to <2 x i32>
  %131 = extractelement <2 x i32> %130, i32 0
  %132 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %131, i32 2, i32 31)
  %133 = insertelement <2 x i32> undef, i32 %132, i32 0
  %134 = extractelement <2 x i32> %130, i32 1
  %135 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %134, i32 2, i32 31)
  %136 = insertelement <2 x i32> %133, i32 %135, i32 1
  %137 = bitcast <2 x i32> %136 to double
  %138 = insertvalue { double, double } poison, double %127, 0
  %139 = insertvalue { double, double } %138, double %137, 1
  %140 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %117, { double, double } %139)
  %141 = extractvalue { double, double } %140, 0
  %142 = bitcast double %141 to i64
  %143 = bitcast i64 %142 to <2 x i32>
  %144 = extractelement <2 x i32> %143, i32 0
  %145 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %144, i32 1, i32 31)
  %146 = insertelement <2 x i32> undef, i32 %145, i32 0
  %147 = extractelement <2 x i32> %143, i32 1
  %148 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %147, i32 1, i32 31)
  %149 = insertelement <2 x i32> %146, i32 %148, i32 1
  %150 = bitcast <2 x i32> %149 to double
  %151 = extractvalue { double, double } %140, 1
  %152 = bitcast double %151 to i64
  %153 = bitcast i64 %152 to <2 x i32>
  %154 = extractelement <2 x i32> %153, i32 0
  %155 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %154, i32 1, i32 31)
  %156 = insertelement <2 x i32> undef, i32 %155, i32 0
  %157 = extractelement <2 x i32> %153, i32 1
  %158 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %157, i32 1, i32 31)
  %159 = insertelement <2 x i32> %156, i32 %158, i32 1
  %160 = bitcast <2 x i32> %159 to double
  %161 = insertvalue { double, double } poison, double %150, 0
  %162 = insertvalue { double, double } %161, double %160, 1
  %163 = call { double, double } @region_0_1_reduce_sum_2_01({ double, double } %140, { double, double } %162)
  %164 = icmp eq i32 %3, 0
  br i1 %164, label %165, label %167

165:                                              ; preds = %48
  %166 = getelementptr inbounds [1 x { double, double }], ptr %1, i32 0, i32 0
  store { double, double } %163, ptr %166, align 8
  br label %167

167:                                              ; preds = %165, %48
  ret void
}

define internal { double, double } @region_0_1_reduce_sum_2_01({ double, double } %0, { double, double } %1) {
  %3 = extractvalue { double, double } %0, 0
  %4 = extractvalue { double, double } %1, 0
  %5 = fadd double %3, %4
  %6 = extractvalue { double, double } %0, 1
  %7 = extractvalue { double, double } %1, 1
  %8 = fadd double %6, %7
  %9 = insertvalue { double, double } poison, double %5, 0
  %10 = insertvalue { double, double } %9, double %8, 1
  ret { double, double } %10
}

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { "nvvm.reqntid"="256,1,1" }
attributes #4 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #5 = { "nvvm.reqntid"="32,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 751}
!2 = !{i32 0, i32 128}
!3 = !{}
!4 = !{i32 0, i32 256}
!5 = !{i32 0, i32 39}
!6 = !{i32 0, i32 32}
