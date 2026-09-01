; ModuleID = 'jit__f_consts'
source_filename = "jit__f_consts"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@buffer_for_constant_2_0 = global [2048 x i8] zeroinitializer, align 256
