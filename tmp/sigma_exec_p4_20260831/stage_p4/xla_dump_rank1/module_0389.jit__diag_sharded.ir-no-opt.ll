; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @loop_select_fusion(ptr noalias align 16 dereferenceable(24772608) %0, ptr noalias align 256 dereferenceable(4) %1, ptr noalias align 256 dereferenceable(4128768) %2) #0 {
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %6 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  %7 = load i32, ptr %6, align 4, !invariant.load !3
  %8 = lshr i32 %7, 1
  %9 = and i32 %8, 1
  %10 = mul i32 %9, 12
  %11 = sext i32 %10 to i64
  %12 = and i32 %7, 1
  %13 = mul i32 %12, 12
  %14 = sext i32 %13 to i64
  %15 = mul i32 %4, 128
  %16 = add i32 %15, %5
  %17 = urem i32 %16, 6
  %18 = mul i32 %17, 4
  %19 = sext i32 %18 to i64
  %20 = sub i64 %19, %11
  %21 = icmp sge i64 %20, 0
  %22 = icmp slt i64 %20, 12
  %23 = and i1 %21, %22
  %24 = sub i64 %19, %14
  %25 = icmp sge i64 %24, 0
  %26 = and i1 %23, %25
  %27 = icmp slt i64 %24, 12
  %28 = and i1 %26, %27
  %29 = call i64 @llvm.smax.i64(i64 %20, i64 0)
  %30 = call i64 @llvm.smin.i64(i64 %29, i64 11)
  %31 = icmp slt i64 %30, 0
  %32 = add i64 %30, 12
  %33 = select i1 %31, i64 %32, i64 %30
  %34 = trunc i64 %33 to i32
  %35 = call i32 @llvm.smin.i32(i32 %34, i32 11)
  %36 = call i32 @llvm.smax.i32(i32 %35, i32 0)
  %37 = call i64 @llvm.smax.i64(i64 %24, i64 0)
  %38 = call i64 @llvm.smin.i64(i64 %37, i64 11)
  %39 = icmp slt i64 %38, 0
  %40 = add i64 %38, 12
  %41 = select i1 %39, i64 %40, i64 %38
  %42 = trunc i64 %41 to i32
  %43 = call i32 @llvm.smin.i32(i32 %42, i32 11)
  %44 = call i32 @llvm.smax.i32(i32 %43, i32 0)
  %45 = udiv i32 %16, 6
  %46 = mul i32 %45, 144
  %47 = mul i32 %36, 12
  %48 = add i32 %46, %47
  %49 = add i32 %48, %44
  %50 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %49
  %51 = load { double, double }, ptr %50, align 8, !invariant.load !3
  %52 = select i1 %28, { double, double } %51, { double, double } zeroinitializer
  %53 = mul i32 %5, 4
  %54 = mul i32 %4, 512
  %55 = add i32 %53, %54
  %56 = getelementptr inbounds [258048 x { double, double }], ptr %2, i32 0, i32 %55
  store { double, double } %52, ptr %56, align 8
  %57 = add i32 %18, 1
  %58 = sext i32 %57 to i64
  %59 = sub i64 %58, %11
  %60 = icmp sge i64 %59, 0
  %61 = icmp slt i64 %59, 12
  %62 = and i1 %60, %61
  %63 = sub i64 %58, %14
  %64 = icmp sge i64 %63, 0
  %65 = and i1 %62, %64
  %66 = icmp slt i64 %63, 12
  %67 = and i1 %65, %66
  %68 = call i64 @llvm.smax.i64(i64 %59, i64 0)
  %69 = call i64 @llvm.smin.i64(i64 %68, i64 11)
  %70 = icmp slt i64 %69, 0
  %71 = add i64 %69, 12
  %72 = select i1 %70, i64 %71, i64 %69
  %73 = trunc i64 %72 to i32
  %74 = call i32 @llvm.smin.i32(i32 %73, i32 11)
  %75 = call i32 @llvm.smax.i32(i32 %74, i32 0)
  %76 = call i64 @llvm.smax.i64(i64 %63, i64 0)
  %77 = call i64 @llvm.smin.i64(i64 %76, i64 11)
  %78 = icmp slt i64 %77, 0
  %79 = add i64 %77, 12
  %80 = select i1 %78, i64 %79, i64 %77
  %81 = trunc i64 %80 to i32
  %82 = call i32 @llvm.smin.i32(i32 %81, i32 11)
  %83 = call i32 @llvm.smax.i32(i32 %82, i32 0)
  %84 = mul i32 %75, 12
  %85 = add i32 %46, %84
  %86 = add i32 %85, %83
  %87 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %86
  %88 = load { double, double }, ptr %87, align 8, !invariant.load !3
  %89 = select i1 %67, { double, double } %88, { double, double } zeroinitializer
  %90 = add i32 %55, 1
  %91 = getelementptr inbounds [258048 x { double, double }], ptr %2, i32 0, i32 %90
  store { double, double } %89, ptr %91, align 8
  %92 = add i32 %18, 2
  %93 = sext i32 %92 to i64
  %94 = sub i64 %93, %11
  %95 = icmp sge i64 %94, 0
  %96 = icmp slt i64 %94, 12
  %97 = and i1 %95, %96
  %98 = sub i64 %93, %14
  %99 = icmp sge i64 %98, 0
  %100 = and i1 %97, %99
  %101 = icmp slt i64 %98, 12
  %102 = and i1 %100, %101
  %103 = call i64 @llvm.smax.i64(i64 %94, i64 0)
  %104 = call i64 @llvm.smin.i64(i64 %103, i64 11)
  %105 = icmp slt i64 %104, 0
  %106 = add i64 %104, 12
  %107 = select i1 %105, i64 %106, i64 %104
  %108 = trunc i64 %107 to i32
  %109 = call i32 @llvm.smin.i32(i32 %108, i32 11)
  %110 = call i32 @llvm.smax.i32(i32 %109, i32 0)
  %111 = call i64 @llvm.smax.i64(i64 %98, i64 0)
  %112 = call i64 @llvm.smin.i64(i64 %111, i64 11)
  %113 = icmp slt i64 %112, 0
  %114 = add i64 %112, 12
  %115 = select i1 %113, i64 %114, i64 %112
  %116 = trunc i64 %115 to i32
  %117 = call i32 @llvm.smin.i32(i32 %116, i32 11)
  %118 = call i32 @llvm.smax.i32(i32 %117, i32 0)
  %119 = mul i32 %110, 12
  %120 = add i32 %46, %119
  %121 = add i32 %120, %118
  %122 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %121
  %123 = load { double, double }, ptr %122, align 8, !invariant.load !3
  %124 = select i1 %102, { double, double } %123, { double, double } zeroinitializer
  %125 = add i32 %55, 2
  %126 = getelementptr inbounds [258048 x { double, double }], ptr %2, i32 0, i32 %125
  store { double, double } %124, ptr %126, align 8
  %127 = add i32 %18, 3
  %128 = sext i32 %127 to i64
  %129 = sub i64 %128, %11
  %130 = icmp sge i64 %129, 0
  %131 = icmp slt i64 %129, 12
  %132 = and i1 %130, %131
  %133 = sub i64 %128, %14
  %134 = icmp sge i64 %133, 0
  %135 = and i1 %132, %134
  %136 = icmp slt i64 %133, 12
  %137 = and i1 %135, %136
  %138 = call i64 @llvm.smax.i64(i64 %129, i64 0)
  %139 = call i64 @llvm.smin.i64(i64 %138, i64 11)
  %140 = icmp slt i64 %139, 0
  %141 = add i64 %139, 12
  %142 = select i1 %140, i64 %141, i64 %139
  %143 = trunc i64 %142 to i32
  %144 = call i32 @llvm.smin.i32(i32 %143, i32 11)
  %145 = call i32 @llvm.smax.i32(i32 %144, i32 0)
  %146 = call i64 @llvm.smax.i64(i64 %133, i64 0)
  %147 = call i64 @llvm.smin.i64(i64 %146, i64 11)
  %148 = icmp slt i64 %147, 0
  %149 = add i64 %147, 12
  %150 = select i1 %148, i64 %149, i64 %147
  %151 = trunc i64 %150 to i32
  %152 = call i32 @llvm.smin.i32(i32 %151, i32 11)
  %153 = call i32 @llvm.smax.i32(i32 %152, i32 0)
  %154 = mul i32 %145, 12
  %155 = add i32 %46, %154
  %156 = add i32 %155, %153
  %157 = getelementptr inbounds [1548288 x { double, double }], ptr %0, i32 0, i32 %156
  %158 = load { double, double }, ptr %157, align 8, !invariant.load !3
  %159 = select i1 %137, { double, double } %158, { double, double } zeroinitializer
  %160 = add i32 %55, 3
  %161 = getelementptr inbounds [258048 x { double, double }], ptr %2, i32 0, i32 %160
  store { double, double } %159, ptr %161, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 504}
!2 = !{i32 0, i32 128}
!3 = !{}
