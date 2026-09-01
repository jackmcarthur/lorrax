; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@global_smem = external local_unnamed_addr addrspace(3) global [0 x i8], align 16

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32) #1

; Function Attrs: nounwind
define ptx_kernel void @input_reduce_fusion(ptr noalias align 16 dereferenceable(12582912) %arg0, ptr noalias align 256 dereferenceable(262144) %arg1) local_unnamed_addr #3 {
  %1 = addrspacecast ptr %arg0 to ptr addrspace(1)
  %2 = addrspacecast ptr %arg1 to ptr addrspace(1)
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %4 = zext nneg i32 %3 to i64
  %5 = shl nuw nsw i64 %4, 3
  %6 = tail call range(i32 0, 64) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = shl nuw nsw i32 %6, 13
  %8 = and i32 %7, 491520
  %9 = and i32 %6, 3
  %10 = shl nuw nsw i32 %9, 1
  %11 = and i32 %6, 7
  %12 = zext nneg i32 %11 to i64
  %13 = or disjoint i32 %10, %8
  %14 = zext nneg i32 %13 to i64
  %15 = getelementptr double, ptr addrspace(1) %1, i64 %5
  %16 = getelementptr double, ptr addrspace(1) %15, i64 %14
  %17 = getelementptr i8, ptr addrspace(1) %16, i64 4194304
  %18 = getelementptr i8, ptr addrspace(1) %16, i64 8388608
  %19 = getelementptr i8, ptr addrspace(1) %16, i64 12582912
  %20 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %16, i1 true) #5
  %21 = extractvalue { i64, i64 } %20, 0
  %22 = extractvalue { i64, i64 } %20, 1
  %23 = bitcast i64 %21 to double
  %24 = bitcast i64 %22 to double
  %25 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %17, i1 true) #5
  %26 = extractvalue { i64, i64 } %25, 0
  %27 = extractvalue { i64, i64 } %25, 1
  %28 = bitcast i64 %26 to double
  %29 = bitcast i64 %27 to double
  %30 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %18, i1 true) #5
  %31 = extractvalue { i64, i64 } %30, 0
  %32 = extractvalue { i64, i64 } %30, 1
  %33 = bitcast i64 %31 to double
  %34 = bitcast i64 %32 to double
  %35 = tail call { i64, i64 } asm sideeffect "mov.u64 $0, $2;\0A\09mov.u64 $1, $3;\0A\09@$5 ld.global.v2.b64 { $0, $1 }, [ $4 + 0 ];", "=l,=l,l,l,l,b"(i64 0, i64 0, ptr addrspace(1) %19, i1 false) #5
  %36 = fadd double %23, %28
  %37 = fadd double %33, 0.000000e+00
  %38 = fadd double %36, %37
  %39 = fadd double %24, %29
  %40 = fadd double %34, 0.000000e+00
  %41 = fadd double %39, %40
  %bc = bitcast double %38 to <2 x i32>
  %42 = extractelement <2 x i32> %bc, i64 0
  %43 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %42, i32 16, i32 31)
  %44 = extractelement <2 x i32> %bc, i64 1
  %45 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %44, i32 16, i32 31)
  %46 = insertelement <2 x i32> poison, i32 %43, i64 0
  %47 = insertelement <2 x i32> %46, i32 %45, i64 1
  %48 = bitcast <2 x i32> %47 to double
  %49 = fadd double %38, %48
  %bc2 = bitcast double %49 to <2 x i32>
  %50 = extractelement <2 x i32> %bc2, i64 0
  %51 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %50, i32 8, i32 31)
  %52 = extractelement <2 x i32> %bc2, i64 1
  %53 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %52, i32 8, i32 31)
  %54 = insertelement <2 x i32> poison, i32 %51, i64 0
  %55 = insertelement <2 x i32> %54, i32 %53, i64 1
  %56 = bitcast <2 x i32> %55 to double
  %57 = fadd double %49, %56
  %bc4 = bitcast double %57 to <2 x i32>
  %58 = extractelement <2 x i32> %bc4, i64 0
  %59 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %58, i32 4, i32 31)
  %60 = extractelement <2 x i32> %bc4, i64 1
  %61 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %60, i32 4, i32 31)
  %62 = insertelement <2 x i32> poison, i32 %59, i64 0
  %63 = insertelement <2 x i32> %62, i32 %61, i64 1
  %64 = bitcast <2 x i32> %63 to double
  %65 = fadd double %57, %64
  %bc6 = bitcast double %41 to <2 x i32>
  %66 = extractelement <2 x i32> %bc6, i64 0
  %67 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %66, i32 16, i32 31)
  %68 = extractelement <2 x i32> %bc6, i64 1
  %69 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %68, i32 16, i32 31)
  %70 = insertelement <2 x i32> poison, i32 %67, i64 0
  %71 = insertelement <2 x i32> %70, i32 %69, i64 1
  %72 = bitcast <2 x i32> %71 to double
  %73 = fadd double %41, %72
  %bc8 = bitcast double %73 to <2 x i32>
  %74 = extractelement <2 x i32> %bc8, i64 0
  %75 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %74, i32 8, i32 31)
  %76 = extractelement <2 x i32> %bc8, i64 1
  %77 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %76, i32 8, i32 31)
  %78 = insertelement <2 x i32> poison, i32 %75, i64 0
  %79 = insertelement <2 x i32> %78, i32 %77, i64 1
  %80 = bitcast <2 x i32> %79 to double
  %81 = fadd double %73, %80
  %bc10 = bitcast double %81 to <2 x i32>
  %82 = extractelement <2 x i32> %bc10, i64 0
  %83 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %82, i32 4, i32 31)
  %84 = extractelement <2 x i32> %bc10, i64 1
  %85 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %84, i32 4, i32 31)
  %86 = insertelement <2 x i32> poison, i32 %83, i64 0
  %87 = insertelement <2 x i32> %86, i32 %85, i64 1
  %88 = bitcast <2 x i32> %87 to double
  %89 = fadd double %81, %88
  %90 = shl nuw nsw i32 %9, 5
  %91 = lshr i32 %6, 1
  %92 = and i32 %91, 16
  %93 = or disjoint i32 %90, %92
  %94 = zext nneg i32 %93 to i64
  %95 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %94
  %96 = insertelement <2 x double> poison, double %65, i64 0
  %97 = insertelement <2 x double> %96, double %89, i64 1
  store <2 x double> %97, ptr addrspace(3) %95, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %98 = shl nuw nsw i32 %6, 2
  %99 = and i32 %98, 16
  %100 = or disjoint i32 %90, %99
  %101 = zext nneg i32 %100 to i64
  %102 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i64 %101
  %103 = load <2 x double>, ptr addrspace(3) %102, align 16
  %104 = extractelement <2 x double> %103, i32 0
  %105 = extractelement <2 x double> %103, i32 1
  %bc14 = bitcast double %104 to <2 x i32>
  %106 = extractelement <2 x i32> %bc14, i64 0
  %107 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %106, i32 4, i32 31)
  %108 = extractelement <2 x i32> %bc14, i64 1
  %109 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %108, i32 4, i32 31)
  %110 = insertelement <2 x i32> poison, i32 %107, i64 0
  %111 = insertelement <2 x i32> %110, i32 %109, i64 1
  %112 = bitcast <2 x i32> %111 to double
  %113 = fadd double %104, %112
  %bc16 = bitcast double %105 to <2 x i32>
  %114 = extractelement <2 x i32> %bc16, i64 0
  %115 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %114, i32 4, i32 31)
  %116 = extractelement <2 x i32> %bc16, i64 1
  %117 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %116, i32 4, i32 31)
  %118 = insertelement <2 x i32> poison, i32 %115, i64 0
  %119 = insertelement <2 x i32> %118, i32 %117, i64 1
  %120 = bitcast <2 x i32> %119 to double
  %121 = fadd double %105, %120
  %122 = getelementptr double, ptr addrspace(1) %2, i64 %5
  %123 = getelementptr double, ptr addrspace(1) %122, i64 %12
  %124 = and i32 %6, 4
  %125 = icmp eq i32 %124, 0
  %126 = select i1 %125, double %113, double %121
  %127 = select i1 %125, double %121, double %113
  %128 = and i32 %6, 24
  %129 = and i32 %91, 3
  %130 = and i32 %6, 1
  %131 = icmp eq i32 %130, 0
  %132 = shl nuw nsw i32 %130, 2
  %133 = or disjoint i32 %129, %132
  %134 = or disjoint i32 %133, %128
  %bc18 = bitcast double %126 to <2 x i32>
  %135 = extractelement <2 x i32> %bc18, i64 0
  %136 = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %135, i32 %134, i32 31)
  %137 = extractelement <2 x i32> %bc18, i64 1
  %138 = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %137, i32 %134, i32 31)
  %139 = insertelement <2 x i32> poison, i32 %136, i64 0
  %140 = insertelement <2 x i32> %139, i32 %138, i64 1
  %141 = xor i32 %134, 4
  %bc20 = bitcast double %127 to <2 x i32>
  %142 = extractelement <2 x i32> %bc20, i64 0
  %143 = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %142, i32 %141, i32 31)
  %144 = extractelement <2 x i32> %bc20, i64 1
  %145 = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %144, i32 %141, i32 31)
  %146 = insertelement <2 x i32> poison, i32 %143, i64 0
  %147 = insertelement <2 x i32> %146, i32 %145, i64 1
  %.v = select i1 %131, <2 x i32> %140, <2 x i32> %147
  %148 = icmp samesign ult i32 %6, 8
  %149 = bitcast <2 x i32> %.v to i64
  tail call void asm sideeffect "@$2 st.global.b64 [ $1 + 0 ], { $0 };", "l,l,b"(i64 %149, ptr addrspace(1) %123, i1 %148) #5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @loop_multiply_fusion(ptr noalias align 256 captures(none) dereferenceable(262144) %0, ptr noalias readnone align 256 captures(none) dereferenceable(262144) %1) local_unnamed_addr #4 {
  %3 = addrspacecast ptr %0 to ptr addrspace(1)
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !2
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !3
  %6 = shl nuw nsw i32 %4, 7
  %7 = or disjoint i32 %6, %5
  %8 = zext nneg i32 %7 to i64
  %9 = getelementptr inbounds double, ptr addrspace(1) %3, i64 %8
  %10 = load double, ptr addrspace(1) %9, align 8
  %11 = fmul double %10, 0x3F95555555555555
  store double %11, ptr addrspace(1) %9, align 8
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { nounwind "nvvm.reqntid"="64,1,1" }
attributes #4 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="128,1,1" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1}
!nvvm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 0, i32 256}
!3 = !{i32 0, i32 128}
