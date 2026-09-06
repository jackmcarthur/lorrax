#!/bin/bash
set -euo pipefail
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906 rev-parse HEAD > source_head.txt
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906 diff HEAD -- src services tests > source.diff
git -C /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tmp/worktrees/wt_bisp_prof_s_codex_20260906 diff --exit-code 71ae0bde -- src services > source_pin.diff
export LORRAX_DEBUG_PRINT=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$PWD/xla_dump_rank${SLURM_PROCID}"
NSYS=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys
if [ "$SLURM_PROCID" = 0 ]; then
 "$NSYS" profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-flush-interval=1000 --cuda-graph-trace=node --sample=none --cpuctxsw=none -t cuda,nvtx,osrt -o nsys_rank0 python3 -u profile_driver.py -i cohsex.in > driver.rank0.log 2>&1
 "$NSYS" stats --report nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_gpu_kern_sum --format csv --output stats nsys_rank0.nsys-rep > nsys_stats.log 2>&1
else
 python3 -u profile_driver.py -i cohsex.in > driver.rank${SLURM_PROCID}.log 2>&1
fi
