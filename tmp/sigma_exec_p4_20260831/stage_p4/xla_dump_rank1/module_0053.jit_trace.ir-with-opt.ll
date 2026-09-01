; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_7_0 = local_unnamed_addr addrspace(1) global [64 x i8] zeroinitializer, align 256
@buffer_for_constant_12 = local_unnamed_addr addrspace(1) global [64 x i8] zeroinitializer, align 256

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_compare_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(16) %0, ptr noalias readonly align 256 captures(none) dereferenceable(4) %1, ptr noalias readonly align 256 captures(none) dereferenceable(16) %2, ptr noalias writeonly align 256 captures(none) dereferenceable(96100) %3) local_unnamed_addr #0 {
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %0 to ptr addrspace(1)
  %7 = addrspacecast ptr %2 to ptr addrspace(1)
  %8 = addrspacecast ptr %3 to ptr addrspace(1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %10 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %11 = shl nuw nsw i32 %9, 7
  %12 = or disjoint i32 %11, %10
  %13 = icmp samesign ult i32 %12, 96100
  br i1 %13, label %14, label %36

14:                                               ; preds = %4
  %15 = udiv i32 %12, 310
  %16 = mul i32 %15, 310
  %.decomposed = sub i32 %12, %16
  %17 = zext nneg i32 %15 to i64
  %18 = load i32, ptr addrspace(1) %5, align 256, !invariant.load !4
  %19 = tail call i32 @llvm.umin.i32(i32 %18, i32 3)
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds i32, ptr addrspace(1) %6, i64 %20
  %22 = load i32, ptr addrspace(1) %21, align 4, !invariant.load !4
  %23 = mul i32 %22, 310
  %24 = sext i32 %23 to i64
  %25 = zext nneg i32 %.decomposed to i64
  %26 = getelementptr inbounds i32, ptr addrspace(1) %7, i64 %20
  %27 = load i32, ptr addrspace(1) %26, align 4, !invariant.load !4
  %28 = mul i32 %27, 310
  %29 = sext i32 %28 to i64
  %30 = add nsw i64 %24, %17
  %31 = add nsw i64 %29, %25
  %32 = icmp eq i64 %30, %31
  %33 = zext i1 %32 to i8
  %34 = zext nneg i32 %12 to i64
  %35 = getelementptr inbounds i8, ptr addrspace(1) %8, i64 %34
  store i8 %33, ptr addrspace(1) %35, align 1
  br label %36

36:                                               ; preds = %14, %4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #2

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion(ptr noalias readonly align 256 captures(none) dereferenceable(96100) %0, ptr noalias readonly align 16 captures(none) dereferenceable(1537600) %1, ptr noalias writeonly align 256 captures(none) dereferenceable(4960) %2) local_unnamed_addr #3 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !5
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !6
  %9 = shl nuw nsw i32 %8, 3
  %10 = lshr i32 %7, 5
  %11 = or disjoint i32 %9, %10
  %12 = icmp samesign ult i32 %11, 310
  br i1 %12, label %13, label %._crit_edge

._crit_edge:                                      ; preds = %3
  %.pre = and i32 %7, 31
  br label %103

13:                                               ; preds = %3
  %14 = mul nuw nsw i32 %10, 310
  %15 = mul nuw nsw i32 %8, 2480
  %16 = and i32 %7, 31
  %17 = add nuw nsw i32 %16, %15
  %18 = add nuw nsw i32 %17, %14
  %19 = zext nneg i32 %18 to i64
  %20 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %19
  %21 = load i8, ptr addrspace(1) %20, align 1, !invariant.load !4
  %22 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %19
  %23 = load <2 x double>, ptr addrspace(1) %22, align 16, !invariant.load !4
  %.unpack44 = extractelement <2 x double> %23, i32 0
  %.unpack345 = extractelement <2 x double> %23, i32 1
  %24 = trunc i8 %21 to i1
  %25 = fadd double %.unpack44, 0.000000e+00
  %26 = select i1 %24, double %25, double 0.000000e+00
  %27 = fadd double %.unpack345, 0.000000e+00
  %28 = select i1 %24, double %27, double 0.000000e+00
  %29 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 32
  %30 = load i8, ptr addrspace(1) %29, align 1, !invariant.load !4
  %31 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 512
  %32 = load <2 x double>, ptr addrspace(1) %31, align 16, !invariant.load !4
  %.unpack446 = extractelement <2 x double> %32, i32 0
  %.unpack647 = extractelement <2 x double> %32, i32 1
  %33 = trunc i8 %30 to i1
  %34 = select i1 %33, double %.unpack446, double 0.000000e+00
  %35 = fadd double %26, %34
  %36 = select i1 %33, double %.unpack647, double 0.000000e+00
  %37 = fadd double %28, %36
  %38 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 64
  %39 = load i8, ptr addrspace(1) %38, align 1, !invariant.load !4
  %40 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 1024
  %41 = load <2 x double>, ptr addrspace(1) %40, align 16, !invariant.load !4
  %.unpack748 = extractelement <2 x double> %41, i32 0
  %.unpack949 = extractelement <2 x double> %41, i32 1
  %42 = trunc i8 %39 to i1
  %43 = select i1 %42, double %.unpack748, double 0.000000e+00
  %44 = fadd double %35, %43
  %45 = select i1 %42, double %.unpack949, double 0.000000e+00
  %46 = fadd double %37, %45
  %47 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 96
  %48 = load i8, ptr addrspace(1) %47, align 1, !invariant.load !4
  %49 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 1536
  %50 = load <2 x double>, ptr addrspace(1) %49, align 16, !invariant.load !4
  %.unpack1050 = extractelement <2 x double> %50, i32 0
  %.unpack1251 = extractelement <2 x double> %50, i32 1
  %51 = trunc i8 %48 to i1
  %52 = select i1 %51, double %.unpack1050, double 0.000000e+00
  %53 = fadd double %44, %52
  %54 = select i1 %51, double %.unpack1251, double 0.000000e+00
  %55 = fadd double %46, %54
  %56 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 128
  %57 = load i8, ptr addrspace(1) %56, align 1, !invariant.load !4
  %58 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 2048
  %59 = load <2 x double>, ptr addrspace(1) %58, align 16, !invariant.load !4
  %.unpack1352 = extractelement <2 x double> %59, i32 0
  %.unpack1553 = extractelement <2 x double> %59, i32 1
  %60 = trunc i8 %57 to i1
  %61 = select i1 %60, double %.unpack1352, double 0.000000e+00
  %62 = fadd double %53, %61
  %63 = select i1 %60, double %.unpack1553, double 0.000000e+00
  %64 = fadd double %55, %63
  %65 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 160
  %66 = load i8, ptr addrspace(1) %65, align 1, !invariant.load !4
  %67 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 2560
  %68 = load <2 x double>, ptr addrspace(1) %67, align 16, !invariant.load !4
  %.unpack1654 = extractelement <2 x double> %68, i32 0
  %.unpack1855 = extractelement <2 x double> %68, i32 1
  %69 = trunc i8 %66 to i1
  %70 = select i1 %69, double %.unpack1654, double 0.000000e+00
  %71 = fadd double %62, %70
  %72 = select i1 %69, double %.unpack1855, double 0.000000e+00
  %73 = fadd double %64, %72
  %74 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 192
  %75 = load i8, ptr addrspace(1) %74, align 1, !invariant.load !4
  %76 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 3072
  %77 = load <2 x double>, ptr addrspace(1) %76, align 16, !invariant.load !4
  %.unpack1956 = extractelement <2 x double> %77, i32 0
  %.unpack2157 = extractelement <2 x double> %77, i32 1
  %78 = trunc i8 %75 to i1
  %79 = select i1 %78, double %.unpack1956, double 0.000000e+00
  %80 = fadd double %71, %79
  %81 = select i1 %78, double %.unpack2157, double 0.000000e+00
  %82 = fadd double %73, %81
  %83 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 224
  %84 = load i8, ptr addrspace(1) %83, align 1, !invariant.load !4
  %85 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 3584
  %86 = load <2 x double>, ptr addrspace(1) %85, align 16, !invariant.load !4
  %.unpack2258 = extractelement <2 x double> %86, i32 0
  %.unpack2459 = extractelement <2 x double> %86, i32 1
  %87 = trunc i8 %84 to i1
  %88 = select i1 %87, double %.unpack2258, double 0.000000e+00
  %89 = fadd double %80, %88
  %90 = select i1 %87, double %.unpack2459, double 0.000000e+00
  %91 = fadd double %82, %90
  %92 = getelementptr inbounds i8, ptr addrspace(1) %20, i64 256
  %93 = load i8, ptr addrspace(1) %92, align 1, !invariant.load !4
  %94 = getelementptr inbounds i8, ptr addrspace(1) %22, i64 4096
  %95 = load <2 x double>, ptr addrspace(1) %94, align 16, !invariant.load !4
  %.unpack2560 = extractelement <2 x double> %95, i32 0
  %.unpack2761 = extractelement <2 x double> %95, i32 1
  %96 = trunc i8 %93 to i1
  %97 = select i1 %96, double %.unpack2560, double 0.000000e+00
  %98 = fadd double %89, %97
  %99 = select i1 %96, double %.unpack2761, double 0.000000e+00
  %100 = fadd double %91, %99
  %101 = insertvalue { double, double } poison, double %98, 0
  %102 = insertvalue { double, double } %101, double %100, 1
  br label %103

103:                                              ; preds = %._crit_edge, %13
  %.pre-phi = phi i32 [ %.pre, %._crit_edge ], [ %16, %13 ]
  %104 = phi { double, double } [ zeroinitializer, %._crit_edge ], [ %102, %13 ]
  %105 = icmp ult i32 %11, 310
  %106 = icmp samesign ult i32 %.pre-phi, 22
  %107 = and i1 %106, %105
  br i1 %107, label %108, label %131

108:                                              ; preds = %103
  %109 = mul nuw nsw i32 %10, 310
  %110 = mul nuw nsw i32 %8, 2480
  %111 = zext nneg i32 %109 to i64
  %112 = zext nneg i32 %.pre-phi to i64
  %113 = zext nneg i32 %110 to i64
  %114 = add i64 %113, %112
  %115 = add i64 %114, %111
  %116 = getelementptr inbounds i8, ptr addrspace(1) %4, i64 %115
  %117 = getelementptr inbounds i8, ptr addrspace(1) %116, i64 288
  %118 = load i8, ptr addrspace(1) %117, align 1, !invariant.load !4
  %119 = getelementptr inbounds { double, double }, ptr addrspace(1) %5, i64 %115
  %120 = getelementptr inbounds i8, ptr addrspace(1) %119, i64 4608
  %121 = load <2 x double>, ptr addrspace(1) %120, align 16, !invariant.load !4
  %.unpack2842 = extractelement <2 x double> %121, i32 0
  %.unpack3043 = extractelement <2 x double> %121, i32 1
  %122 = trunc i8 %118 to i1
  %123 = extractvalue { double, double } %104, 0
  %124 = select i1 %122, double %.unpack2842, double 0.000000e+00
  %125 = fadd double %123, %124
  %126 = extractvalue { double, double } %104, 1
  %127 = select i1 %122, double %.unpack3043, double 0.000000e+00
  %128 = fadd double %126, %127
  %129 = insertvalue { double, double } poison, double %125, 0
  %130 = insertvalue { double, double } %129, double %128, 1
  br label %131

131:                                              ; preds = %108, %103
  %132 = phi { double, double } [ %130, %108 ], [ %104, %103 ]
  %133 = icmp ult i32 %11, 310
  %134 = extractvalue { double, double } %132, 0
  %135 = bitcast double %134 to <2 x i32>
  %136 = extractelement <2 x i32> %135, i64 0
  %137 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %136, i32 16, i32 31)
  %138 = insertelement <2 x i32> poison, i32 %137, i64 0
  %139 = extractelement <2 x i32> %135, i64 1
  %140 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %139, i32 16, i32 31)
  %141 = insertelement <2 x i32> %138, i32 %140, i64 1
  %142 = bitcast <2 x i32> %141 to double
  %143 = extractvalue { double, double } %132, 1
  %144 = bitcast double %143 to <2 x i32>
  %145 = extractelement <2 x i32> %144, i64 0
  %146 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %145, i32 16, i32 31)
  %147 = insertelement <2 x i32> poison, i32 %146, i64 0
  %148 = extractelement <2 x i32> %144, i64 1
  %149 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %148, i32 16, i32 31)
  %150 = insertelement <2 x i32> %147, i32 %149, i64 1
  %151 = bitcast <2 x i32> %150 to double
  %152 = fadd double %134, %142
  %153 = fadd double %143, %151
  %154 = bitcast double %152 to <2 x i32>
  %155 = extractelement <2 x i32> %154, i64 0
  %156 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %155, i32 8, i32 31)
  %157 = insertelement <2 x i32> poison, i32 %156, i64 0
  %158 = extractelement <2 x i32> %154, i64 1
  %159 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %158, i32 8, i32 31)
  %160 = insertelement <2 x i32> %157, i32 %159, i64 1
  %161 = bitcast <2 x i32> %160 to double
  %162 = bitcast double %153 to <2 x i32>
  %163 = extractelement <2 x i32> %162, i64 0
  %164 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %163, i32 8, i32 31)
  %165 = insertelement <2 x i32> poison, i32 %164, i64 0
  %166 = extractelement <2 x i32> %162, i64 1
  %167 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %166, i32 8, i32 31)
  %168 = insertelement <2 x i32> %165, i32 %167, i64 1
  %169 = bitcast <2 x i32> %168 to double
  %170 = fadd double %152, %161
  %171 = fadd double %153, %169
  %172 = bitcast double %170 to <2 x i32>
  %173 = extractelement <2 x i32> %172, i64 0
  %174 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %173, i32 4, i32 31)
  %175 = insertelement <2 x i32> poison, i32 %174, i64 0
  %176 = extractelement <2 x i32> %172, i64 1
  %177 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %176, i32 4, i32 31)
  %178 = insertelement <2 x i32> %175, i32 %177, i64 1
  %179 = bitcast <2 x i32> %178 to double
  %180 = bitcast double %171 to <2 x i32>
  %181 = extractelement <2 x i32> %180, i64 0
  %182 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %181, i32 4, i32 31)
  %183 = insertelement <2 x i32> poison, i32 %182, i64 0
  %184 = extractelement <2 x i32> %180, i64 1
  %185 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %184, i32 4, i32 31)
  %186 = insertelement <2 x i32> %183, i32 %185, i64 1
  %187 = bitcast <2 x i32> %186 to double
  %188 = fadd double %170, %179
  %189 = fadd double %171, %187
  %190 = bitcast double %188 to <2 x i32>
  %191 = extractelement <2 x i32> %190, i64 0
  %192 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %191, i32 2, i32 31)
  %193 = insertelement <2 x i32> poison, i32 %192, i64 0
  %194 = extractelement <2 x i32> %190, i64 1
  %195 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %194, i32 2, i32 31)
  %196 = insertelement <2 x i32> %193, i32 %195, i64 1
  %197 = bitcast <2 x i32> %196 to double
  %198 = bitcast double %189 to <2 x i32>
  %199 = extractelement <2 x i32> %198, i64 0
  %200 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %199, i32 2, i32 31)
  %201 = insertelement <2 x i32> poison, i32 %200, i64 0
  %202 = extractelement <2 x i32> %198, i64 1
  %203 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %202, i32 2, i32 31)
  %204 = insertelement <2 x i32> %201, i32 %203, i64 1
  %205 = bitcast <2 x i32> %204 to double
  %206 = fadd double %188, %197
  %207 = fadd double %189, %205
  %208 = bitcast double %206 to <2 x i32>
  %209 = extractelement <2 x i32> %208, i64 0
  %210 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %209, i32 1, i32 31)
  %211 = extractelement <2 x i32> %208, i64 1
  %212 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %211, i32 1, i32 31)
  %213 = bitcast double %207 to <2 x i32>
  %214 = extractelement <2 x i32> %213, i64 0
  %215 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %214, i32 1, i32 31)
  %216 = extractelement <2 x i32> %213, i64 1
  %217 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %216, i32 1, i32 31)
  %218 = icmp eq i32 %.pre-phi, 0
  %219 = icmp samesign ult i32 %7, 225
  %220 = and i1 %219, %218
  %221 = and i1 %220, %133
  br i1 %221, label %222, label %235

222:                                              ; preds = %131
  %223 = zext nneg i32 %11 to i64
  %224 = getelementptr inbounds { double, double }, ptr addrspace(1) %6, i64 %223
  %225 = insertelement <2 x i32> poison, i32 %215, i64 0
  %226 = insertelement <2 x i32> %225, i32 %217, i64 1
  %227 = bitcast <2 x i32> %226 to double
  %228 = fadd double %207, %227
  %229 = insertelement <2 x i32> poison, i32 %210, i64 0
  %230 = insertelement <2 x i32> %229, i32 %212, i64 1
  %231 = bitcast <2 x i32> %230 to double
  %232 = fadd double %206, %231
  %233 = insertelement <2 x double> poison, double %232, i32 0
  %234 = insertelement <2 x double> %233, double %228, i32 1
  store <2 x double> %234, ptr addrspace(1) %224, align 16
  br label %235

235:                                              ; preds = %222, %131
  ret void
}

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #4

; Function Attrs: norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @input_reduce_fusion_1(ptr noalias readonly align 256 captures(none) dereferenceable(4960) %0, ptr noalias writeonly align 256 captures(none) dereferenceable(16) %1) local_unnamed_addr #5 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = addrspacecast ptr %1 to ptr addrspace(1)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !7
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds { double, double }, ptr addrspace(1) %3, i64 %6
  %8 = load <2 x double>, ptr addrspace(1) %7, align 16, !invariant.load !4
  %.unpack37 = extractelement <2 x double> %8, i32 0
  %.unpack238 = extractelement <2 x double> %8, i32 1
  %9 = fadd double %.unpack37, 0.000000e+00
  %10 = fadd double %.unpack238, 0.000000e+00
  %11 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 512
  %12 = load <2 x double>, ptr addrspace(1) %11, align 16, !invariant.load !4
  %.unpack339 = extractelement <2 x double> %12, i32 0
  %.unpack540 = extractelement <2 x double> %12, i32 1
  %13 = fadd double %9, %.unpack339
  %14 = fadd double %10, %.unpack540
  %15 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 1024
  %16 = load <2 x double>, ptr addrspace(1) %15, align 16, !invariant.load !4
  %.unpack641 = extractelement <2 x double> %16, i32 0
  %.unpack842 = extractelement <2 x double> %16, i32 1
  %17 = fadd double %13, %.unpack641
  %18 = fadd double %14, %.unpack842
  %19 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 1536
  %20 = load <2 x double>, ptr addrspace(1) %19, align 16, !invariant.load !4
  %.unpack943 = extractelement <2 x double> %20, i32 0
  %.unpack1144 = extractelement <2 x double> %20, i32 1
  %21 = fadd double %17, %.unpack943
  %22 = fadd double %18, %.unpack1144
  %23 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 2048
  %24 = load <2 x double>, ptr addrspace(1) %23, align 16, !invariant.load !4
  %.unpack1245 = extractelement <2 x double> %24, i32 0
  %.unpack1446 = extractelement <2 x double> %24, i32 1
  %25 = fadd double %21, %.unpack1245
  %26 = fadd double %22, %.unpack1446
  %27 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 2560
  %28 = load <2 x double>, ptr addrspace(1) %27, align 16, !invariant.load !4
  %.unpack1547 = extractelement <2 x double> %28, i32 0
  %.unpack1748 = extractelement <2 x double> %28, i32 1
  %29 = fadd double %25, %.unpack1547
  %30 = fadd double %26, %.unpack1748
  %31 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 3072
  %32 = load <2 x double>, ptr addrspace(1) %31, align 16, !invariant.load !4
  %.unpack1849 = extractelement <2 x double> %32, i32 0
  %.unpack2050 = extractelement <2 x double> %32, i32 1
  %33 = fadd double %29, %.unpack1849
  %34 = fadd double %30, %.unpack2050
  %35 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 3584
  %36 = load <2 x double>, ptr addrspace(1) %35, align 16, !invariant.load !4
  %.unpack2151 = extractelement <2 x double> %36, i32 0
  %.unpack2352 = extractelement <2 x double> %36, i32 1
  %37 = fadd double %33, %.unpack2151
  %38 = fadd double %34, %.unpack2352
  %39 = getelementptr inbounds i8, ptr addrspace(1) %7, i64 4096
  %40 = load <2 x double>, ptr addrspace(1) %39, align 16, !invariant.load !4
  %.unpack2453 = extractelement <2 x double> %40, i32 0
  %.unpack2654 = extractelement <2 x double> %40, i32 1
  %41 = fadd double %37, %.unpack2453
  %42 = fadd double %38, %.unpack2654
  %43 = icmp samesign ult i32 %5, 22
  br i1 %43, label %44, label %48

44:                                               ; preds = %2
  %sunkaddr = getelementptr inbounds i8, ptr addrspace(1) %7, i64 4608
  %45 = load <2 x double>, ptr addrspace(1) %sunkaddr, align 16, !invariant.load !4
  %.unpack2735 = extractelement <2 x double> %45, i32 0
  %.unpack2936 = extractelement <2 x double> %45, i32 1
  %46 = fadd double %41, %.unpack2735
  %47 = fadd double %42, %.unpack2936
  br label %48

48:                                               ; preds = %44, %2
  %.pn34 = phi double [ %46, %44 ], [ %41, %2 ]
  %.pn32 = phi double [ %47, %44 ], [ %42, %2 ]
  %49 = bitcast double %.pn34 to <2 x i32>
  %50 = extractelement <2 x i32> %49, i64 0
  %51 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %50, i32 16, i32 31)
  %52 = insertelement <2 x i32> poison, i32 %51, i64 0
  %53 = extractelement <2 x i32> %49, i64 1
  %54 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %53, i32 16, i32 31)
  %55 = insertelement <2 x i32> %52, i32 %54, i64 1
  %56 = bitcast <2 x i32> %55 to double
  %57 = bitcast double %.pn32 to <2 x i32>
  %58 = extractelement <2 x i32> %57, i64 0
  %59 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %58, i32 16, i32 31)
  %60 = insertelement <2 x i32> poison, i32 %59, i64 0
  %61 = extractelement <2 x i32> %57, i64 1
  %62 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %61, i32 16, i32 31)
  %63 = insertelement <2 x i32> %60, i32 %62, i64 1
  %64 = bitcast <2 x i32> %63 to double
  %65 = fadd double %.pn34, %56
  %66 = fadd double %.pn32, %64
  %67 = bitcast double %65 to <2 x i32>
  %68 = extractelement <2 x i32> %67, i64 0
  %69 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %68, i32 8, i32 31)
  %70 = insertelement <2 x i32> poison, i32 %69, i64 0
  %71 = extractelement <2 x i32> %67, i64 1
  %72 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %71, i32 8, i32 31)
  %73 = insertelement <2 x i32> %70, i32 %72, i64 1
  %74 = bitcast <2 x i32> %73 to double
  %75 = bitcast double %66 to <2 x i32>
  %76 = extractelement <2 x i32> %75, i64 0
  %77 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %76, i32 8, i32 31)
  %78 = insertelement <2 x i32> poison, i32 %77, i64 0
  %79 = extractelement <2 x i32> %75, i64 1
  %80 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %79, i32 8, i32 31)
  %81 = insertelement <2 x i32> %78, i32 %80, i64 1
  %82 = bitcast <2 x i32> %81 to double
  %83 = fadd double %65, %74
  %84 = fadd double %66, %82
  %85 = bitcast double %83 to <2 x i32>
  %86 = extractelement <2 x i32> %85, i64 0
  %87 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %86, i32 4, i32 31)
  %88 = insertelement <2 x i32> poison, i32 %87, i64 0
  %89 = extractelement <2 x i32> %85, i64 1
  %90 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %89, i32 4, i32 31)
  %91 = insertelement <2 x i32> %88, i32 %90, i64 1
  %92 = bitcast <2 x i32> %91 to double
  %93 = bitcast double %84 to <2 x i32>
  %94 = extractelement <2 x i32> %93, i64 0
  %95 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %94, i32 4, i32 31)
  %96 = insertelement <2 x i32> poison, i32 %95, i64 0
  %97 = extractelement <2 x i32> %93, i64 1
  %98 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %97, i32 4, i32 31)
  %99 = insertelement <2 x i32> %96, i32 %98, i64 1
  %100 = bitcast <2 x i32> %99 to double
  %101 = fadd double %83, %92
  %102 = fadd double %84, %100
  %103 = bitcast double %101 to <2 x i32>
  %104 = extractelement <2 x i32> %103, i64 0
  %105 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %104, i32 2, i32 31)
  %106 = insertelement <2 x i32> poison, i32 %105, i64 0
  %107 = extractelement <2 x i32> %103, i64 1
  %108 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %107, i32 2, i32 31)
  %109 = insertelement <2 x i32> %106, i32 %108, i64 1
  %110 = bitcast <2 x i32> %109 to double
  %111 = bitcast double %102 to <2 x i32>
  %112 = extractelement <2 x i32> %111, i64 0
  %113 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %112, i32 2, i32 31)
  %114 = insertelement <2 x i32> poison, i32 %113, i64 0
  %115 = extractelement <2 x i32> %111, i64 1
  %116 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %115, i32 2, i32 31)
  %117 = insertelement <2 x i32> %114, i32 %116, i64 1
  %118 = bitcast <2 x i32> %117 to double
  %119 = fadd double %101, %110
  %120 = fadd double %102, %118
  %121 = bitcast double %119 to <2 x i32>
  %122 = extractelement <2 x i32> %121, i64 0
  %123 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %122, i32 1, i32 31)
  %124 = extractelement <2 x i32> %121, i64 1
  %125 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %124, i32 1, i32 31)
  %126 = bitcast double %120 to <2 x i32>
  %127 = extractelement <2 x i32> %126, i64 0
  %128 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %127, i32 1, i32 31)
  %129 = extractelement <2 x i32> %126, i64 1
  %130 = tail call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %129, i32 1, i32 31)
  %131 = icmp eq i32 %5, 0
  br i1 %131, label %132, label %143

132:                                              ; preds = %48
  %133 = insertelement <2 x i32> poison, i32 %128, i64 0
  %134 = insertelement <2 x i32> %133, i32 %130, i64 1
  %135 = bitcast <2 x i32> %134 to double
  %136 = fadd double %120, %135
  %137 = insertelement <2 x i32> poison, i32 %123, i64 0
  %138 = insertelement <2 x i32> %137, i32 %125, i64 1
  %139 = bitcast <2 x i32> %138 to double
  %140 = fadd double %119, %139
  %141 = insertelement <2 x double> poison, double %140, i32 0
  %142 = insertelement <2 x double> %141, double %136, i32 1
  store <2 x double> %142, ptr addrspace(1) %4, align 256
  br label %143

143:                                              ; preds = %132, %48
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "nvvm.reqntid"="256,1,1" }
attributes #4 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #5 = { norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "nvvm.reqntid"="32,1,1" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 751}
!3 = !{i32 0, i32 128}
!4 = !{}
!5 = !{i32 0, i32 256}
!6 = !{i32 0, i32 39}
!7 = !{i32 0, i32 32}
