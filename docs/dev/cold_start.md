# Cold start: where the seconds before the first physics go

**Scope.** The *first* run of a LORRAX driver on a new system size, on nodes
that have never touched the software. That is the only startup that matters:
a user runs a given system size once, so the persistent compile cache — whose
key includes shapes — misses every entry and is irrelevant to this question.

Everything below is a differenced timestamp or a `mincore(2)` residency count
from a named job. Nothing is estimated.

---

## 1. The headline

`gw.kin_ion_io`, MoS₂ 4×4, nb=256, compile cache OFF, one cell per **fresh**
compute node, variant→node assignment shuffled (job **7882055**). Seconds from
`apptainer exec` handing control to the shell, to the driver's full import
graph being resolved:

| variant | cold total | `import jax` | `jax.devices()` | rest |
|---|---|---|---|---|
| **as shipped** (venv on `/work2`, CUDA plugin discovery on) | **44–88 s** (n=4) | 8.7–22 s | **34–73 s** | ~2 s |
| **+ skip GPU plugin discovery** (`src/runtime`, no packaging change) | **11.3, 19.6 s** | 8.6–16 s | **0.06, 0.13 s** | ~2 s |
| **+ node-local runtime bundle** (`config/frontera/`) | **4.6, 4.7 s** | 1.5 s | 0.02 s | 0.6 s |
| python interpreter alone, in-container, cold | 0.38 s | — | — | — |
| the same "as shipped" cell re-run on the **same** node (warm) | 3.1 s | 1.9 s | 0.23 s | 1.0 s |

The last two rows are the controls. The interpreter and the container are not
the problem (0.38 s). The warm re-run is the negative control for the whole
table: the same cell on the same node went 44.0 s → 3.1 s, so the instrument
demonstrably distinguishes cold from warm.

Note the *spread* of the shipped configuration: 44 s to 88 s for identical
work. Cold start today is not merely slow, it is unpredictable by a factor of
two, because it is latency-bound on a shared filesystem under someone else's
load.

---

## 1b. The same thing on the real driver

Job **7882076** ran `gw.kin_ion_io` itself — same deck, same nb, one cell per
fresh node, shuffled — with the fix armed and explicitly disarmed:

| cell | arm | wall | first output | driver banner |
|---|---|---|---|---|
| c4 | `LORRAX_CPU_SKIP_GPU_PLUGINS=0` | **125 s** | 10.5 s | **89.3 s** |
| c5 | same, and no `JAX_PLATFORMS` set at all | **141 s** | 18.6 s | 99.3 s |
| c1 | fix armed (`JAX_PLATFORMS=cpu`) | **37 s** | **9.9 s** | 11.4 s |
| c3 | fix armed via the no-GPU-device arm | 37 s | 10.6 s | 12.0 s |
| c2 | fix armed + node-local bundle | 29 s | 6.5 s | 7.1 s |
| c6 | fix armed + node-local bundle | **28 s** | **4.5 s** | 5.2 s |

c4's log times the mechanism to the second: the opt-out banner prints at
10.5 s, `cuInit(0)` fails at 87.4 s, the driver banner follows at 89.3 s.
**76.9 s inside `_load_nvidia_libraries()`**, on a CPU node, for libraries that
are then discarded.

c4 reproduces job 7881949's cold cell (126 s wall, `== kin_ion_io ==` at
76.7 s): the disarmed fix is the old code's timing. Compare on the banner,
which is the unambiguous "physics has started" marker in both —

> **`== kin_ion_io ==` on a cold node: 76.7 s (7881949) → 5.2 s (7882076 c6).
> Wall: 126 s → 28 s.**

(Do not compare "time to first output" across the two: with the fix present the
first line printed is `runtime`'s own banner at 10.5 s, whereas without it the
first line is jax's `cuInit` failure — which in 7881949 landed at 75.0 s and in
c4 at 87.4 s. Same event, different name.)

Two gates, both required to have been seen failing:

* **G1 = PASS.** Cells with the fix disarmed must show the `cuInit` banner and
  no skip banner; cells with it armed must show the opposite. Both directions
  are asserted, so neither can pass by an absent grep. The `autoskip` cell
  additionally has to prove it armed through the *no-GPU-device* branch and not
  through `JAX_PLATFORMS`.
* **G2 = PASS, bit-identical.** Every cell's `kin_ion.h5` is byte-for-byte the
  disarmed reference (`max|d| = 0.000e+00`, `bit-identical=True`, 5 of 5). The
  comparator was falsified first: injecting one ULP into the reference makes it
  report `7.105e-15`, so a comparator that could only ever print zero is ruled
  out.

---

## 2. Why `jax.devices()` costs a minute on a CPU node

`xla_bridge.backends()` calls `_discover_and_register_pjrt_plugins()` **before**
it looks at `jax_platforms` (jax 0.9.1, `jax/_src/xla_bridge.py:797` vs `:808`).
Discovery imports `jax_plugins.xla_cuda12`, and the first statement of that
module's `initialize()` is `_load_nvidia_libraries()` — a `ctypes` dlopen of
libcudart, libnvrtc, libcublas, libcublasLt, libnccl, libcupti, libcusparse,
libcusolver, libcufft, libnvshmem_host, libcudnn.

`JAX_PLATFORMS=cpu` does not prevent one byte of that. Nor does the absence of a
GPU: jax's own `if platform == "cuda" and not has_visible_nvidia_gpu(): continue`
(`xla_bridge.py:829`) decides to skip CUDA — *after* discovery has already
loaded the libraries. The `RuntimeError: ... operation cuInit(0) failed` banner
in every LORRAX CPU log is the receipt for work that was thrown away.

**It is not a bandwidth cost.** `mincore(2)` over the venv immediately after a
cold start (same job; probe self-test falsified in job 7882047 — RED leg 0.0 %,
GREEN leg 100.0 %):

| | resident after cold start |
|---|---|
| shipped configuration | **173.2 MB** of the 5576 MB venv |
| with plugin discovery skipped | **54.7 MB** |
| difference, all CUDA | **118.5 MB** (libcublasLt 68.1, cudnn_graph 16.1, nvrtc 12.7, nvJitLink 10.5, cufft 3.4, cusparse 1.6, cupti 1.4, cusolver 1.4, cublas 1.3) |

118 MB at the measured `/work2` streaming rate of 70 MB/s (job 7882047) is under
two seconds. The minute comes from *how* those bytes are fetched: dlopen
relocation faults scattered 4 KB pages, each fault a separate Lustre RPC. Tens
of thousands of round trips, not a transfer.

That also explains why staging wins so completely: the bundle is one contiguous
tar read at `/scratch2`'s 560 MB/s, and every later fault is served by a local
XFS SSD.

---

## 3. The two fixes

### 3.1 `runtime.skip_gpu_plugin_discovery()` — in-tree, no packaging change

A `sys.meta_path` finder answers imports under the `jax_plugins` namespace with
a stub whose `initialize()` does nothing. jax's discovery loop finds a plugin,
calls it, gets a no-op; nothing is dlopened and nothing is registered. **No jax
file is modified and no package is removed** — the same installed venv still
runs GPU jobs.

It arms on two conditions, each one a case where jax was *already* going to end
up on CPU:

1. `JAX_PLATFORMS` resolves to exactly `cpu`.
2. `JAX_PLATFORMS` asks for a GPU but no NVIDIA device node is visible. This is
   the arm that reaches a user who never set `JAX_PLATFORMS`, since every
   driver's own `setdefault("JAX_PLATFORMS", "cuda,cpu")` lands here on a CPU
   node. Because it means "this run is CPU", it also does what the demotion in
   `fallback_to_cpu_if_no_gpu_backend()` would have done — pins
   `JAX_PLATFORMS=cpu` and re-states the CPU collectives implementation — so
   nothing downstream can tell the difference except the clock.

Called from `bootstrap()`, `set_default_env()`, `init_jax_distributed()` and
`fallback_to_cpu_if_no_gpu_backend()`, so a driver with the three-call header
(`gw.kin_ion_io`, `psp.run_nscf`, …) gets it without changing.

`LORRAX_CPU_SKIP_GPU_PLUGINS=0` restores the old behaviour, and says so.

The gate is checkable without a job, because `runtime` imports no jax. On a
login node (which has no `/dev/nvidia*`, i.e. permanently arm 2's condition),
`PYTHONPATH=src python3` gives:

| `JAX_PLATFORMS` | `LORRAX_CPU_SKIP_GPU_PLUGINS` | armed |
|---|---|---|
| `cpu` | — | yes (arm 1) |
| `cuda,cpu` | — | yes (arm 2) |
| `gpu` | — | yes (arm 2) |
| *unset* | — | **no** |
| `cpu` | `0` | **no** |
| `cpu` | `off` | **no** |
| `cpu` | `1` | yes |

and with it armed, `importlib.import_module("jax_plugins.xla_cuda12")` returns
the stub whose `initialize()` returns `None`, while
`pkgutil.iter_modules(jax_plugins.__path__)` returns `[]` — the two lookups
`discover_pjrt_plugins()` actually performs, both answered, neither reaching
`_load_nvidia_libraries()`.

### 3.2 The node-local runtime bundle — `config/frontera/`

```bash
# once per venv/source revision, inside the SIF (needs the container python):
apptainer exec --bind /home1,/work2,/scratch1,/scratch2 $LORRAX_SIF \
    config/frontera/build_cpu_runtime_bundle.sh          # -> $SCRATCH/lorrax_bundle/

# in the job's container-side runner, before python:
export LORRAX_BUNDLE=$SCRATCH/lorrax_bundle/lorrax_cpu_bundle.tar
. $LORRAX_ROOT/config/frontera/stage_runtime.sh
export PYTHONPATH=$LORRAX_OVERLAY_DIR:$LORRAX_SRC_DIR
$LORRAX_PY -u -m gw.kin_ion_io ...
```

`build_cpu_runtime_bundle.sh` packs venv + MPI overlay + `src/` into one
`tar` on `/scratch2`, minus the packages a CPU run cannot use
(`nvidia/*`, `jax_plugins/`, `jax_cuda12_plugin/`, and the
`jax_cuda12_pjrt-*.dist-info` that advertises the plugin entry point), and
byte-compiles what uv left uncompiled (measured: 1290 `.pyc` for 3760 `.py` in
the installed venv; jax itself 226 of 610). 5.6 GB → **769 MB**.

`stage_runtime.sh` unrolls it onto `/tmp` (a real 144 GB local XFS SSD on
Frontera CLX, writable inside apptainer — job 7882047) **once per node**, under
`flock`, and exports `LORRAX_PY` / `LORRAX_SRC_DIR` / `LORRAX_OVERLAY_DIR`. A
missing or unreadable bundle falls back to the Lustre venv **loudly**: a silent
fallback would turn "the startup fix regressed" into "the startup fix does
nothing", which no timing table can catch.

Measured staging cost: **1.5–2.2 s per node**, included in the 4.6 s totals
above.

Neither of these is a patched dependency. Every file in the bundle is a
byte-for-byte copy of what uv installed; removing a package a run cannot use is
a deployment choice, and the GPU venv is untouched.

### 3.3 Staging at large N — and why `sbcast` is NOT the answer here

Staging as written is **O(N) Lustre traffic**: every node streams the same
782 MB. That is 25 GB at 32 nodes and would be 782 GB at 1000, so the obvious
move is `sbcast` — SLURM pushes a file from the batch node to every node's
local storage over the job's own network, hierarchically, Lustre read once.

**Measured, it loses badly on Frontera** (jobs 7882128 at N=8 and 7882121 at
N=32; the comparison is deliberately rigged in sbcast's favour — sbcast ran
first with the tar cold, the Lustre arm ran second with the same 782 MB already
warm on the servers, and on a `stripe_count=1` tar, which is the *worst* case
for the Lustre arm):

| | N=8 | N=32 |
|---|---|---|
| `sbcast` of the 782 MB tar | **54.0 s** | **158.8 s** |
| then extracting the node-local copy | 0.80 s | 0.79 s |
| every node reading the tar off Lustre and extracting | **1.26 s** | **2.56 s** |

Forty to sixty times slower, with the thumb on its scale, and `sbcast` grows
roughly *linearly* in N (3× for 4× the nodes) while the concurrent Lustre read
grows *sub*-linearly (2× for 4× the nodes, on one OST). Do not use `sbcast` for
this.

What to do instead: **stripe the tar wide** and let the nodes read it
concurrently. Note that `build_cpu_runtime_bundle.sh` cannot do the striping
itself when it runs inside the SIF — `python:3.12-bookworm` carries no Lustre
client tools, so the tar lands on one OST (observed: job 7882121 read a
`stripe_count=1` bundle). The script now says so and prints the one-liner to
run from a login node. Both numbers above are therefore the *unstriped* case;
striping can only help the arm that already wins.

**The honest limit:** concurrent staging is measured at N=8 and N=32 only, and
2.56 s at N=32 is still growing. Extrapolating to the thousands of processes
LORRAX targets is not warranted from two points — the aggregate Lustre read is
the thing to instrument next, on a striped tar, at the largest N available.

---

## 3.4 Is the data I/O a separate target? At this deck, no.

The driver's own recorded timings, same three cells of job 7882076 (post-banner,
so none of the import cost is in them):

| | `knob0` (fix off) | `skip` (fix on, Lustre venv) | `staged` |
|---|---|---|---|
| `load_wfn` | 4.673 s | 4.637 s | **1.725 s** |
| `kin_ion_k` | 7.430 s | 7.366 s | 7.352 s |
| `write_h5` | 0.063 s | 0.078 s | 0.095 s |
| total recorded | 22.777 s | 21.886 s | 19.352 s |

Three things follow.

* **`write_h5` is not a line item at this deck** — 0.06–0.10 s in every arm.
* **`load_wfn` drops 4.67 s → 1.73 s from staging the *software*.** The WFN
  data file is on Lustre in all three arms and is *not* in the bundle, so that
  2.9 s was never data I/O: it was first-touch of h5py and the read machinery.
  The packaging lever reaches further into the driver than the import phase.
* **`kin_ion_k` is 7.35–7.43 s in all three arms.** The actual physics does not
  move, which is a second, independent check on the fix alongside the
  bit-identical h5 outputs.

So for this deck the cold-start problem really is first-touch of the software
stack, essentially end to end. A deck whose WFN is large enough for the data
read to dominate would need its own measurement; nothing here licenses a claim
about one.

---

## 4. What is irreducible

After both fixes, a cold `gw.kin_ion_io` reaches its first output in **4.5 s**
(job 7882076 c6) and resolves its whole import graph in **4.6 s** (job 7882055),
of which:

| | s |
|---|---|
| stage the 769 MB bundle to the node's SSD (once per node) | 1.5–2.2 |
| python interpreter + venv site init | 0.4 |
| `import numpy` + `import jax` from local disk | 1.5 |
| `jax.devices()` (CPU backend) | 0.02 |
| the LORRAX import graph | 0.6 |

The staging read and the ~2.5 s of interpreter-and-import work are the floor
without changing dependencies. `import jax` from a warm local page cache is
1.9 s (job 7882055, warm control), so 1.5 s from a local SSD is already at
CPython's import-machinery limit for that module count, not an I/O limit.

**Outside our control** and *not* included above: `srun` step launch inside an
allocation, measured at **0.6 s to 20.4 s** for identical steps (jobs 7881949
and 7882055). This is SLURM, it dwarfs everything the code does, and no change
here touches it.

---

## 5. `jax.distributed` at scale — it was never the problem

Job 7881949 recorded **43.8 s of "distributed init" at P=16** from rank 0's
clock alone: the gap between rank 0's first output line and the driver banner.
That interval contains everything rank 0 waits for. Job **7882070** re-measured
it with every rank writing its own timestamps.

Warm nodes, every rank released into `initialize()` at the same wall-clock
instant (so the handshake is measured with nobody missing):

| P | `initialize()` | `jax.devices()` | first collective | total |
|---|---|---|---|---|
| 1 | 0.00 | 0.22 | — | |
| 2 | 0.01 | 0.68 | 0.06 | |
| 4 | 0.01 | 0.72 | 0.06 | |
| 8 | 0.01 | 0.73 | 0.06 | |
| 16 | 0.01 | 0.76 | 0.06 | |
| 32 | 0.01 | 0.81 | 0.06 | |
| 64 | 0.01 | **0.95** | 0.06 | |

`jax.distributed` bring-up is **flat in P to 64 processes** and costs about a
second in total. Without the wall-clock barrier, `initialize()` reads 1.20 s at
P=16 and 1.17 s at P=64 — a *constant*, not a scaling term; it is the client's
connect retry when ranks arrive within ~0.2 s of each other rather than
simultaneously.

**That sweep was run in ascending P**, which is the ordering that produced a
wrong conclusion in job 7881887. Job **7882139** re-runs it fully interleaved —
every (P, barrier) point three times, all cells shuffled into one sequence — so
positional drift can be compared against the trend it might be faking:

All 42 cells, n=3 everywhere, `psum_ok=True` in every one:

| P | mode | at cells | `initialize()` mean [min,max] | `jax.devices()` mean [min,max] |
|---|---|---|---|---|
| 1 | sync | 4, 6, 40 | 0.000 [0.000, 0.000] | 0.220 [0.220, 0.220] |
| 1 | nosync | 12, 35, 42 | 0.000 [0.000, 0.000] | 0.223 [0.220, 0.230] |
| 2 | sync | 5, 24, 29 | 0.010 [0.010, 0.010] | 0.727 [0.720, 0.730] |
| 2 | nosync | 3, 19, 25 | 0.947 [0.790, 1.140] | 0.763 [0.730, 0.830] |
| 4 | sync | 1, 32, 41 | 0.010 [0.010, 0.010] | 0.723 [0.710, 0.730] |
| 4 | nosync | 9, 16, 34 | 1.077 [0.930, 1.170] | 0.737 [0.730, 0.750] |
| 8 | sync | 7, 11, 33 | 0.010 [0.010, 0.010] | 0.880 [0.740, 1.080] |
| 8 | nosync | 20, 22, 28 | 0.980 [0.820, 1.140] | 0.853 [0.750, 1.030] |
| 16 | sync | 2, 15, 31 | 0.010 [0.010, 0.010] | 0.800 [0.750, 0.890] |
| 16 | nosync | 21, 27, 30 | 1.137 [1.120, 1.160] | 0.783 [0.760, 0.830] |
| 32 | sync | 17, 26, 38 | 0.010 [0.010, 0.010] | 0.823 [0.820, 0.830] |
| 32 | nosync | 8, 10, 14 | 1.127 [1.110, 1.150] | 0.877 [0.810, 0.950] |
| 64 | sync | 13, 18, 37 | 0.010 [0.010, 0.010] | 0.943 [0.930, 0.950] |
| 64 | nosync | 23, 36, 39 | 1.037 [0.850, 1.150] | 0.990 [0.950, 1.040] |

Reading it:

* `initialize()` is **0.010 s at every P from 2 to 64, with zero spread across
  all three replicates at every point**, when the ranks are released together;
  and **0.95–1.14 s at every P ≥ 2** when they are not. Both are flat in P. The
  ~1.1 s is a fixed connect retry, not a scaling term.
* `jax.devices()` shows one real step — **0.22 s at P=1 → 0.72 s at P=2**, ranges
  nowhere near touching. That is the one-time cost of being multi-process at all
  (building the CPU client with MPI collectives), not a scaling term either.
* **From P=2 to P=64, `devices()` grows 0.727 → 0.943 s** (sync; ranges
  [0.720,0.730] and [0.930,0.950] do not overlap, so the endpoints are
  resolved). But the *intermediate* points are not: P=8 sync spans
  [0.740, 1.080], wider than the entire P=2→P=64 difference. The honest
  statement is a real but small ~0.2 s growth over a 32× increase in P, with the
  shape of the curve between unresolved at n=3.
* Replicates of one point sit at widely separated cell positions (P=1 sync at
  4/6/40, P=64 nosync at 23/36/39, P=4 sync at 1/32/41) and the largest
  replicate range anywhere in the table is 0.34 s — larger than the whole
  P=2→P=64 trend. **Ordering is not driving these numbers**, and every
  interleaved point reproduces the ascending sweep of job 7882070 to within
  0.02–0.12 s.

This probe never calls `warm_mesh_cliques`, so the 0.36–5.26 s
`collective_warmup` variance seen elsewhere is not inside any figure here; the
first collective it does time, `process_allgather`, is 0.00–0.08 s throughout.

Cold nodes, no barrier — what a user actually pays:

| cell | P | `initialize()` | `jax.devices()` | total to a verified collective |
|---|---|---|---|---|
| A_P16_cold_base | 16 | **1.35** | **34.01** | 47.2 |
| A_P64_cold_base | 64 | **0.98** | **45.15** | 58.1 |
| D_P64_warm_nosync_stage | 64 | 1.17 | 0.60 | **4.3** |

So the 43.8 s was **not** the handshake. It was the same CUDA-plugin cold load,
inside `jax.devices()`, on the far side of the first output line. Two things
made it look like distributed init:

* the driver's first output line is the *CUDA plugin failure banner*, which
  jax prints part-way through `backends()` — so "time to first output" and
  "time from first output to the banner" both straddle the plugin load;
* rank 0 in 7881949 sat on the node the four preceding single-rank cells had
  warmed, while its peers sat on cold ones, so rank 0's "init" also contained
  its peers' import time. Job 7882070's cold cells have all nodes equally cold
  and the import spread collapses to 0.0 s (P=16) / 7.7 s (P=64).

Every collective in these cells was checked, not assumed: each rank
`process_allgather`s its own index and the aggregator requires the union to be
exactly `range(P)` (`psum_ok=True` above), and it names the ranks if fewer than
P reported.

The real driver with both fixes and a node-local bundle:

| | wall | `== kin_ion_io ==` | `cuInit` lines | skip banner |
|---|---|---|---|---|
| P=16 (job 7882128) | **17 s** | 6.54 s | 0 | 1 |
| P=64 (job 7882121) | **17 s** | 6.83 s | 0 | 1 |
| P=16, job 7881949 (before) | **117 s** | ~60.5 s | — | — |

Flat from 16 to 64 processes. Stated precisely: those nodes had been used by the
job's earlier phases, so their page cache was warm and the bundle tar was
already node-local; this is the multi-process *steady state*, not a cold-node
number. The cold multi-process numbers are the `A_*` rows above, and the cold
single-process driver numbers are §1b.

---

## 6. Instruments

All under `/scratch2/08271/jackmc/lorrax_setup/wk_REL/rel/`.

| file | what it does | how it was falsified |
|---|---|---|
| `pagecache_probe.py` | `mincore(2)` residency per file over a tree | `--selftest` drives the same file to both ends: write+fsync+`fadvise(DONTNEED)` → 0.0 % resident, read it back → 100.0 % (job 7882047). Refuses to report if the RED leg is not near zero. |
| `stair.py` | cumulative import staircase against the shell's own exec stamp | the warm-repeat cell: same probe, same node, 44.0 s → 3.1 s |
| `distprobe.py` / `distagg.py` | per-rank `jax.distributed` decomposition | reports `slack` and refuses the handshake number if any rank reached the wall-clock barrier late; names the missing ranks if fewer than P reported |
| `h5cmp.py` | dataset-by-dataset comparison of driver output | `--selfcheck` injects one ULP into the reference and shows the comparator seeing it |
| `stage_runtime.sh` | node-local staging, loud on fallback | driven to all four ends by sourcing it directly (it is pure bash, no container needed): bundle unset → warns + `staged=0`; bundle missing → warns + `staged=0`; `LORRAX_STAGE=0` → says DISABLED + `staged=0`; a real (tiny) bundle → `staged=1` with a node-local `python`. A fallback that could not announce itself is ruled out. |
| `build_cpu_runtime_bundle.sh` | refuses to ship a bundle whose exclusions did not take effect | the same `grep` over the tar listing counts 0 for `nvidia` / `jax_plugins` / `jax_cuda12_plugin` and 231 for `jaxlib`, 4066 for `.pyc` — a grep that can only ever print 0 is ruled out. The source venv has 2096 `nvidia` files. |

Caveat on `pagecache_probe.py`: residency is counted in whole pages against the
file's exact size, so files smaller than 4 KB report >100 %. Totals over a tree
are dominated by the large `.so` and are unaffected in any way that matters.

### What this page does not claim

* **No memory figure appears anywhere above.** None of these instruments samples
  `/proc/PID/status` or `ru_maxrss` — they were built on `kincold_inner.sh`'s
  timestamping, which carries no memory sampler. (The truncating VmHWM sampler
  that motivates this note lives in `kinstart_inner.sh` and six other
  harnesses; a `grep -l VmHWM` finds it in those and in none of the seven files
  in the table above.) Nothing here needs re-deriving on that account.
* **Ordering.** The cold-start cells (7882055, 7882076) each get their own fresh
  node *and* a shuffled variant→node assignment. The P-scan in 7882070 was run
  in ascending P, which is the pattern that produced a wrong conclusion in
  7881887; it is re-run fully interleaved with three replicates per point in job
  **7882139**, and the per-P replicate range is reported next to the per-P mean
  so ordering drift can be compared against the trend it might be faking.
* **One deck.** Everything is MoS₂ 4×4, nb=256. The import costs are
  deck-independent by construction; the driver-work numbers in §3.4 are not.
