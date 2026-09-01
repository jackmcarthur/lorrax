; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_0 = private addrspace(3) global [1056 x { double, double }] undef

define ptx_kernel void @wrapped_transpose(ptr noalias align 16 dereferenceable(243793920) %0, ptr noalias align 256 dereferenceable(243793920) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = urem i32 %4, 20
  %6 = mul i32 %5, 32
  %7 = urem i32 %3, 32
  %8 = add i32 %6, %7
  %9 = icmp sle i32 %8, 619
  br i1 %9, label %10, label %42

10:                                               ; preds = %2
  %11 = udiv i32 %4, 20
  %12 = urem i32 %11, 2
  %13 = mul i32 %12, 19840
  %14 = add i32 %13, %6
  %15 = udiv i32 %4, 40
  %16 = mul i32 %15, 29760
  %17 = add i32 %14, %16
  %18 = udiv i32 %3, 32
  %19 = mul i32 %18, 620
  %20 = add i32 %17, %19
  %21 = add i32 %20, %7
  %22 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %21
  %23 = load { double, double }, ptr %22, align 8, !invariant.load !3
  %24 = mul i32 %7, 33
  %25 = add i32 %24, %18
  %26 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %25
  store { double, double } %23, ptr %26, align 8
  %27 = add i32 %21, 2480
  %28 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %27
  %29 = load { double, double }, ptr %28, align 8, !invariant.load !3
  %30 = add i32 %25, 4
  %31 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %30
  store { double, double } %29, ptr %31, align 8
  %32 = add i32 %21, 4960
  %33 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %32
  %34 = load { double, double }, ptr %33, align 8, !invariant.load !3
  %35 = add i32 %25, 8
  %36 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %35
  store { double, double } %34, ptr %36, align 8
  %37 = add i32 %21, 7440
  %38 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %37
  %39 = load { double, double }, ptr %38, align 8, !invariant.load !3
  %40 = add i32 %25, 12
  %41 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %40
  store { double, double } %39, ptr %41, align 8
  br label %42

42:                                               ; preds = %10, %2
  %43 = udiv i32 %4, 20
  %44 = urem i32 %43, 2
  %45 = mul i32 %44, 8
  %46 = add i32 %45, 4
  %47 = icmp sle i32 %46, 11
  %48 = and i1 %9, %47
  br i1 %48, label %49, label %66

49:                                               ; preds = %42
  %50 = mul i32 %44, 19840
  %51 = add i32 %50, %6
  %52 = udiv i32 %4, 40
  %53 = mul i32 %52, 29760
  %54 = add i32 %51, %53
  %55 = udiv i32 %3, 32
  %56 = mul i32 %55, 620
  %57 = add i32 %54, %56
  %58 = add i32 %57, %7
  %59 = add i32 %58, 9920
  %60 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %59
  %61 = load { double, double }, ptr %60, align 8, !invariant.load !3
  %62 = mul i32 %7, 33
  %63 = add i32 %62, %55
  %64 = add i32 %63, 16
  %65 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %64
  store { double, double } %61, ptr %65, align 8
  br label %66

66:                                               ; preds = %49, %42
  %67 = add i32 %45, 5
  %68 = icmp sle i32 %67, 11
  %69 = and i1 %9, %68
  br i1 %69, label %70, label %87

70:                                               ; preds = %66
  %71 = mul i32 %44, 19840
  %72 = add i32 %71, %6
  %73 = udiv i32 %4, 40
  %74 = mul i32 %73, 29760
  %75 = add i32 %72, %74
  %76 = udiv i32 %3, 32
  %77 = mul i32 %76, 620
  %78 = add i32 %75, %77
  %79 = add i32 %78, %7
  %80 = add i32 %79, 12400
  %81 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %80
  %82 = load { double, double }, ptr %81, align 8, !invariant.load !3
  %83 = mul i32 %7, 33
  %84 = add i32 %83, %76
  %85 = add i32 %84, 20
  %86 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %85
  store { double, double } %82, ptr %86, align 8
  br label %87

87:                                               ; preds = %70, %66
  %88 = add i32 %45, 6
  %89 = icmp sle i32 %88, 11
  %90 = and i1 %9, %89
  br i1 %90, label %91, label %108

91:                                               ; preds = %87
  %92 = mul i32 %44, 19840
  %93 = add i32 %92, %6
  %94 = udiv i32 %4, 40
  %95 = mul i32 %94, 29760
  %96 = add i32 %93, %95
  %97 = udiv i32 %3, 32
  %98 = mul i32 %97, 620
  %99 = add i32 %96, %98
  %100 = add i32 %99, %7
  %101 = add i32 %100, 14880
  %102 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %101
  %103 = load { double, double }, ptr %102, align 8, !invariant.load !3
  %104 = mul i32 %7, 33
  %105 = add i32 %104, %97
  %106 = add i32 %105, 24
  %107 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %106
  store { double, double } %103, ptr %107, align 8
  br label %108

108:                                              ; preds = %91, %87
  %109 = add i32 %45, 7
  %110 = icmp sle i32 %109, 11
  %111 = and i1 %9, %110
  br i1 %111, label %112, label %129

112:                                              ; preds = %108
  %113 = mul i32 %44, 19840
  %114 = add i32 %113, %6
  %115 = udiv i32 %4, 40
  %116 = mul i32 %115, 29760
  %117 = add i32 %114, %116
  %118 = udiv i32 %3, 32
  %119 = mul i32 %118, 620
  %120 = add i32 %117, %119
  %121 = add i32 %120, %7
  %122 = add i32 %121, 17360
  %123 = getelementptr inbounds [15237120 x { double, double }], ptr %0, i32 0, i32 %122
  %124 = load { double, double }, ptr %123, align 8, !invariant.load !3
  %125 = mul i32 %7, 33
  %126 = add i32 %125, %118
  %127 = add i32 %126, 28
  %128 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %127
  store { double, double } %124, ptr %128, align 8
  br label %129

129:                                              ; preds = %112, %108
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %130 = mul i32 %44, 32
  %131 = add i32 %130, %7
  %132 = icmp sle i32 %131, 47
  br i1 %132, label %133, label %158

133:                                              ; preds = %129
  %134 = udiv i32 %3, 32
  %135 = mul i32 %134, 33
  %136 = add i32 %135, %7
  %137 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %136
  %138 = load { double, double }, ptr %137, align 8
  %139 = mul i32 %5, 1536
  %140 = add i32 %130, %139
  %141 = udiv i32 %4, 40
  %142 = mul i32 %141, 29760
  %143 = add i32 %140, %142
  %144 = mul i32 %134, 48
  %145 = add i32 %143, %144
  %146 = add i32 %145, %7
  %147 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %146
  store { double, double } %138, ptr %147, align 8
  %148 = add i32 %136, 132
  %149 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %148
  %150 = load { double, double }, ptr %149, align 8
  %151 = add i32 %146, 192
  %152 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %151
  store { double, double } %150, ptr %152, align 8
  %153 = add i32 %136, 264
  %154 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %153
  %155 = load { double, double }, ptr %154, align 8
  %156 = add i32 %146, 384
  %157 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %156
  store { double, double } %155, ptr %157, align 8
  br label %158

158:                                              ; preds = %133, %129
  %159 = mul i32 %5, 8
  %160 = add i32 %159, 3
  %161 = icmp sle i32 %160, 154
  %162 = and i1 %132, %161
  br i1 %162, label %163, label %180

163:                                              ; preds = %158
  %164 = udiv i32 %3, 32
  %165 = mul i32 %164, 33
  %166 = add i32 %165, %7
  %167 = add i32 %166, 396
  %168 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %167
  %169 = load { double, double }, ptr %168, align 8
  %170 = mul i32 %5, 1536
  %171 = add i32 %130, %170
  %172 = udiv i32 %4, 40
  %173 = mul i32 %172, 29760
  %174 = add i32 %171, %173
  %175 = mul i32 %164, 48
  %176 = add i32 %174, %175
  %177 = add i32 %176, %7
  %178 = add i32 %177, 576
  %179 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %178
  store { double, double } %169, ptr %179, align 8
  br label %180

180:                                              ; preds = %163, %158
  %181 = add i32 %159, 4
  %182 = icmp sle i32 %181, 154
  %183 = and i1 %132, %182
  br i1 %183, label %184, label %201

184:                                              ; preds = %180
  %185 = udiv i32 %3, 32
  %186 = mul i32 %185, 33
  %187 = add i32 %186, %7
  %188 = add i32 %187, 528
  %189 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %188
  %190 = load { double, double }, ptr %189, align 8
  %191 = mul i32 %5, 1536
  %192 = add i32 %130, %191
  %193 = udiv i32 %4, 40
  %194 = mul i32 %193, 29760
  %195 = add i32 %192, %194
  %196 = mul i32 %185, 48
  %197 = add i32 %195, %196
  %198 = add i32 %197, %7
  %199 = add i32 %198, 768
  %200 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %199
  store { double, double } %190, ptr %200, align 8
  br label %201

201:                                              ; preds = %184, %180
  %202 = add i32 %159, 5
  %203 = icmp sle i32 %202, 154
  %204 = and i1 %132, %203
  br i1 %204, label %205, label %222

205:                                              ; preds = %201
  %206 = udiv i32 %3, 32
  %207 = mul i32 %206, 33
  %208 = add i32 %207, %7
  %209 = add i32 %208, 660
  %210 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %209
  %211 = load { double, double }, ptr %210, align 8
  %212 = mul i32 %5, 1536
  %213 = add i32 %130, %212
  %214 = udiv i32 %4, 40
  %215 = mul i32 %214, 29760
  %216 = add i32 %213, %215
  %217 = mul i32 %206, 48
  %218 = add i32 %216, %217
  %219 = add i32 %218, %7
  %220 = add i32 %219, 960
  %221 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %220
  store { double, double } %211, ptr %221, align 8
  br label %222

222:                                              ; preds = %205, %201
  %223 = add i32 %159, 6
  %224 = icmp sle i32 %223, 154
  %225 = and i1 %132, %224
  br i1 %225, label %226, label %243

226:                                              ; preds = %222
  %227 = udiv i32 %3, 32
  %228 = mul i32 %227, 33
  %229 = add i32 %228, %7
  %230 = add i32 %229, 792
  %231 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %230
  %232 = load { double, double }, ptr %231, align 8
  %233 = mul i32 %5, 1536
  %234 = add i32 %130, %233
  %235 = udiv i32 %4, 40
  %236 = mul i32 %235, 29760
  %237 = add i32 %234, %236
  %238 = mul i32 %227, 48
  %239 = add i32 %237, %238
  %240 = add i32 %239, %7
  %241 = add i32 %240, 1152
  %242 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %241
  store { double, double } %232, ptr %242, align 8
  br label %243

243:                                              ; preds = %226, %222
  %244 = add i32 %159, 7
  %245 = icmp sle i32 %244, 154
  %246 = and i1 %132, %245
  br i1 %246, label %247, label %264

247:                                              ; preds = %243
  %248 = udiv i32 %3, 32
  %249 = mul i32 %248, 33
  %250 = add i32 %249, %7
  %251 = add i32 %250, 924
  %252 = getelementptr inbounds [1056 x { double, double }], ptr addrspacecast (ptr addrspace(3) @shared_0 to ptr), i32 0, i32 %251
  %253 = load { double, double }, ptr %252, align 8
  %254 = mul i32 %5, 1536
  %255 = add i32 %130, %254
  %256 = udiv i32 %4, 40
  %257 = mul i32 %256, 29760
  %258 = add i32 %255, %257
  %259 = mul i32 %248, 48
  %260 = add i32 %258, %259
  %261 = add i32 %260, %7
  %262 = add i32 %261, 1344
  %263 = getelementptr inbounds [15237120 x { double, double }], ptr %1, i32 0, i32 %262
  store { double, double } %253, ptr %263, align 8
  br label %264

264:                                              ; preds = %247, %243
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 128}
!2 = !{i32 0, i32 20480}
!3 = !{}
