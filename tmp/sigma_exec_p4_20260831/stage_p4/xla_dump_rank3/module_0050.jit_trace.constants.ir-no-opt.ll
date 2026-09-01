; ModuleID = 'jit_trace_consts'
source_filename = "jit_trace_consts"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_7_0 = global [64 x i8] zeroinitializer, align 256
@buffer_for_constant_12 = global [64 x i8] zeroinitializer, align 256
