# Multi-node collective probe (no solver)

`mn_collectives_probe.py` brings the communicator stack up exactly as a driver
does and then times and bit-checks every collective LORRAX issues, on a
(nodes, GPUs-per-node) mesh inside `shard_map`:
`psum`, `psum_scatter`, `all_gather`, `all_to_all`, `ppermute`,
`lxkit.ordered_psum`, `lxkit.ordered_psum_scatter` (over the full mesh and the
inter-node axis `x`), and `common.collectives.barrier`. The buffer sizes run
from 1 KiB to 256 MiB per rank, in c128 and f64.

Inputs are integer-valued with power-of-two rank weights, so every result is
exact and a misrouted or mis-summed term shows up as a non-zero error.
Each operation runs under a faulthandler watchdog (`--op-timeout`), so a hang
dumps every thread's stack and exits instead of holding the allocation.
Per-rank JSONL records are written to `--out`.

It needs no FFI library and no square mesh (P = 8 on 2 nodes is fine).

```bash
source config/perlmutter/gpu_env.sh
srun -N 2 -n 8 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh \
  .venv/bin/python -u tests/multinode/mn_collectives_probe.py --stack lorrax --out probe_out
```

| option | meaning |
|---|---|
| `--stack lorrax` | `runtime.bootstrap()` (network planner included), then the one-GPU gate and `nccl_warmup` |
| `--stack lorrax --startup full` | also the `initialize_communicator_stack` steps that need neither a square mesh nor the FFI: 5b–5d, 7 and 7b (compile agreement, the P>1 compile cache, env freeze) |
| `--stack jax` | plain `jax.distributed.initialize()`: no LORRAX import before the backend starts, so the planner does not run |

Exit codes: 0 when everything is bit-exact, 3 when any check fails, 1 on a
watchdog hang (with stack dumps on stderr).
