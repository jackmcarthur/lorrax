# SlabIO — the sharded-slab HDF5 transport

*Verified against `src/` and against measurements taken inside the Shifter
container on Perlmutter, 2026-08-05/06 (allocation 56389339, 4 × A100-40G
nodes). Where this page and older prose disagree, this page wins. Every
number below names the command that produced it.*

**This page owns SlabIO**: the tiers, the routing, the striping, the
certification. It does not own the owner rulings behind them
([`decisions.md`](decisions.md)), the native layer underneath
([`ffi_layout.md`](ffi_layout.md)), or knob spellings
([`../dev/env_vars.md`](../dev/env_vars.md)) — see the
[register](../index.md#register).

Conventions used here, same as [`services.md`](services.md):

- **Level / imports.** `file_io.slab_io` is L3; it imports `ffi.io` lazily.
- **Announce-or-refuse.** A tier that cannot run says so, naming the probe
  that declined it. Nothing about which writer ran is ever silent.
- **Shapes are logical.** Files store the logical shape; padding for mesh
  divisibility never reaches disk.

---

## The contract {#contract}

**One rank writes one tile. Nothing larger than one rank's tile is ever
materialised.**

That is the whole design constraint, and it is a hard one, not a
preference. The production envelope is arrays that need hundreds of GPUs
to hold at all — ζ(q, μ, G) and V_q for a converged cell do not fit in one
node's host RAM, let alone one device's. Any code path that gathers a
whole array to one rank is therefore not a slow path, it is a path that
**cannot run the workload it exists for**.

Consequences that follow directly, and that callers must not work around:

- `write_slab` takes the array the rank already owns, in its existing
  sharding. It does not reshard, gather, or replicate.
- `read_slab` returns a sharded array. The rank reads its own hyperslab
  and no one else's.
- Peak host memory attributable to SlabIO is one rank's tile plus the
  pinned staging buffer, independent of process count.

### `H5PY_ALLGATHER` is a refusal, not a fallback

The `H5PY_ALLGATHER` tier gathers the array to rank 0 via
`process_allgather` and writes it with serial h5py. **It is not a
supported production path.** It violates the contract above by
construction: rank 0 materialises the entire global array.

It exists for exactly two situations:

1. single-process development on a machine with no parallel HDF5 at all;
2. small fixtures and tests, where the array demonstrably fits.

Treat `Routing SlabIO through H5PY_ALLGATHER` in a multi-rank production
log as a **failed run that has not noticed yet** — it will either OOM rank
0 or serialise the whole write behind one process's disk bandwidth. The
correct response is to fix the tier-1 probe's stated reason, not to
proceed.

**Enforced since 2026-08-06** (the note that used to stand here recorded
this as a known gap). One rule now covers every route into the tier:

> **`H5PY_ALLGATHER` is reachable at exactly one process, and nowhere
> else.**

At one process the gather and the per-rank write are the *same operation*,
so there is no contract to violate and nothing to refuse about — laptop
debugging, `-n 1` smoke runs and the config test suite are untouched, and
the tier announces itself when selected. Above one process, both routes in
raise at parse time:

| route in | at `process_count() == 1` | at `process_count() > 1` |
|---|---|---|
| `slab_io=auto`, both parallel tiers declined | selected, announced with the tier-1 reason and the note that the same config refuses at larger P | `_refuse_slab_io_no_parallel_writer` |
| explicit `slab_io = h5py_allgather`, or deprecated `use_ffi_io = false` | honoured, announced | `_refuse_explicit_h5py_allgather` |

Both refusals live in `gw/gw_config.py`, both always raise, and both sit
*after* the whole precedence chain, so no deck branch escapes them.

> **Scope of "raises at parse time": the DECK. Not the library.** Verified
> 2026-08-06. `file_io/slab_io.py:88-90` still returns
> `SlabIOBackend.H5PY_ALLGATHER` when `use_ffi_io is None`, at **any**
> process count, and `:97-98` maps `use_ffi_io=False` the same way. Neither
> path consults the process count, because neither goes through the deck
> router. A caller constructing `SlabIO` directly — a test, a bench driver, a
> notebook — therefore still reaches the tier above one process. The ruling
> is implemented where decks are parsed; closing it at the library entry
> point is open work, and until then "it refuses at P>1" is a statement about
> configured runs only.

The auto refusal names **which** of the two tier-1 probes declined and
gives the repair *for that probe* — `probe_target`'s three states have
three different fixes, and "the writer is unavailable" would be barely
better than the silent demotion it replaced. The explicit refusal is the
one case where a named deck value does *not* win: explicit-wins exists so a
deck can gate a stack, not so it can authorise a transport that cannot hold
the run's own arrays.

The process count is `max(jax.process_count(), SLURM_NTASKS,
SLURM_STEP_NUM_TASKS, PMI_SIZE, OMPI_COMM_WORLD_SIZE)`. The launcher's world
size is consulted alongside JAX's so that a deck parsed before
`jax.distributed.initialize` cannot see 1 on all sixteen ranks and wave the
run through; over-counting at worst refuses a run that could have limped,
under-counting restores the exact failure this rule deletes.

---

## The tiers {#tiers}

Three backends behind one `SlabIO` facade
(`src/file_io/slab_io.py`, dispatch in `SlabIO.__init__`, lazy import per
branch so the FFI only loads when selected):

| tier | module | mechanism | requires |
|---|---|---|---|
| `PHDF5_FFI` | `_slab_io_ffi.py` | collective MPI-IO from C++ via `ffi.io`; the tile goes D2H into a pinned buffer and out through HDF5's MPI-IO VFD | a 2-D mesh; the FFI lib exports `lorrax_phdf5_write` / `lorrax_phdf5_read`; MPI can bootstrap |
| `PHDF5_HOST` | `_slab_io_mpi_host.py` | the same MPI-IO, driven from Python by mpi4py + `HDF5_MPI=ON` h5py | a 2-D mesh; the overlay; **exactly one addressable device per process** |
| `H5PY_ALLGATHER` | `_slab_io_allgather.py` | gather to rank 0, serial h5py | **`process_count() == 1`** — see the refusal note above |

`PHDF5_FFI` is the only tier that runs on both platforms from one C++
core: the same sources compile into the CUDA lib (`liblorrax_ffi.so`) and
into the CUDA-free host lib (`liblorrax_ffi_host.so`, `LORRAX_FFI_NO_CUDA`),
where the D2H staging degrades to an in-place read of the XLA host buffer.

**Writes are asynchronous.** The FFI backend owns a single writer thread
per file with a task queue (`src/ffi/cpp/phdf5/ctx.h`). One thread, not
per-call detached threads, because MPI collectives must be issued in the
same order on every rank. `SlabIO.close()` drains the queue, joins the
thread, and only then calls `H5Fclose` collectively — the three-line
`[SlabIO.close] …` banner in a run log is that sequence.

---

## Routing {#routing}

`slab_io` is an **input-deck key only** — there is no environment variable
and no CLI flag. Values: `auto` (default), `phdf5_ffi`, `phdf5_host`,
`h5py_allgather`; anything else raises at parse time
(`gw_config.py`, `from_input_file`).

An explicit value is honoured verbatim and **skips every probe** — that is
what makes it usable as a gate. `slab_io=phdf5_ffi` on a broken stack fails
loudly instead of quietly producing a slower correct answer, which is
exactly what you want when the question is "does this path work".

One exception, and only one: `h5py_allgather` is additionally gated on
`process_count() == 1` (see [the refusal
note](#h5py_allgather-is-a-refusal-not-a-fallback)). Every other value is
honoured as written, at any process count.

`auto` runs `_route_cpu_slab_io` or `_route_gpu_slab_io`. Both apply the
same two tier-1 conditions:

1. `ffi_loader.probe_target('lorrax_phdf5_write', <platform>)` — a
   three-state probe, not a boolean. It distinguishes *unknown target*
   (wrong platform), *library could not be loaded* (a `LD_LIBRARY_PATH`
   problem — the handler may be perfectly well compiled), and *loaded but
   does not export the symbol* (the only case that means "rebuild").
2. `_probe_mpi_bootstrap_ffi(<platform>)` — handler presence is not
   capability, because the tier calls `MPI_Init_thread`. A launcher PMI
   environment satisfies it; on a bare launch it runs the init in a
   throwaway subprocess, because on some stacks a failed init `abort()`s
   the process rather than returning an error.

Every decision prints the tier, the probe's reason, and — since
2026-08-05 — the **run geometry** (`_slab_io_geometry()`).

### Node count is not a routing condition

Until 2026-08-05 `_route_gpu_slab_io` declined `PHDF5_FFI` outright
whenever `SLURM_JOB_NUM_NODES > 1`, **without running the probe**, citing a
"known cross-node failure on multi-node GPU stacks".

That branch is deleted. The record:

- The failure that justified it was Intel MPI refusing a launch configured
  with `I_MPI_FABRICS=shm` on **Frontera** — a launcher misconfiguration,
  on an MPI stack and a machine this router does not run on. The
  Perlmutter GPU path is Cray MPICH through Shifter.
- The transport it declared broken cross-node was demonstrably working
  cross-node **in the same process**: in a 4-node job the phdf5 FFI *read*
  path succeeded while the *write* path was declined by policy, from the
  same `.so`.
- Measured directly (see [Certification](#certification)): 16 ranks on 4
  nodes, `MPI_Comm_size()` asserted == 16 on every rank, write and read
  bit-exact, payload byte-identical to a 4-rank single-node write of the
  same logical array.
- The branch was **live**, not merely wrong. It read the max of
  `SLURM_JOB_NUM_NODES` and `SLURM_NNODES`. Only the *first* spelling is
  absent inside the container — it is a batch/allocation-level variable,
  not a step-level one. `SLURM_NNODES` **is** exported by every `srun`
  step and measured `4` on the 4-node run, so every multi-node GPU run
  really was demoted off the FFI writer, which is what the archived logs
  show.

  (This is worth stating because the first pass of this investigation got
  it backwards: an early probe read only `SLURM_JOB_NUM_NODES`, saw
  nothing, and concluded the branch was dead code. It was not. Measured
  inside the container, `srun -N 4 -n 16`: `SLURM_NNODES=4`,
  `SLURM_NTASKS=16`, `SLURM_JOB_NODELIST=nid[001033,001644,003837,003840]`
  present; `SLURM_JOB_NUM_NODES` absent.)

That variable-spelling trap is why `_slab_io_geometry()` reports JAX's
process/device counts first: they do not depend on the launcher's
vocabulary.

---

## Launcher requirements {#launcher}

### `srun --mpi=cray_shasta` is mandatory

Shifter's `--module=mpich` bind-mounts Cray MPICH. Cray MPICH speaks
`cray_shasta` PMI. Launch it under `--mpi=pmi2` or `--mpi=pmix` and MPI
initialises **as a singleton**: every rank gets its own `MPI_COMM_WORLD` of
size 1.

This is the single most dangerous failure mode in the whole subsystem,
because **it can produce a correct-looking file**. Under independent I/O
each rank writes its own hyperslab with no collective handshake, so 16
singleton ranks write 16 disjoint regions of one file and the result looks
right. Under collective I/O (the current default) it does not survive, but
the historical record includes runs that "worked" on pmi2 for exactly this
reason.

**Therefore: a phdf5 success is not evidence of working MPI unless the run
asserted its world size.** The assertion is three lines and there is no
excuse for omitting it from a gate:

```python
import ctypes
mpi = ctypes.CDLL("libmpi.so.12")          # MPICH: MPI_COMM_WORLD == 0x44000000
sz = ctypes.c_int(-1)
mpi.MPI_Comm_size(ctypes.c_int(0x44000000), ctypes.byref(sz))
assert sz.value == jax.process_count(), f"SINGLETON MPI: {sz.value}"
```

### The rest of the launch

```
srun --jobid=$JID --overlap --mpi=cray_shasta \
     --gres=gpu:4 -N $NNODES -n $NTASKS --cpu-bind=cores \
     src/ffi/cpp/select_gpu.sh \
     shifter --image=... --module=gpu,mpich --volume=... --env=... \
     src/ffi/cpp/in_container.sh \
     <command>
```

- `select_gpu.sh` sets `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`. It is a
  **script file**, not an inline `bash -c`, so srun execs it as a single
  process image and the `PMI_*` variables reach `MPI_Init_thread` intact.
- `in_container.sh` re-exports `MPICH_GPU_SUPPORT_ENABLED=1`, which
  Shifter's mpich module explicitly unsets.
- `MPICH_GPU_SUPPORT_ENABLED` is **irrelevant to phdf5** and was never
  A/B'd against it. phdf5 writes from host pinned buffers
  (`cudaMallocHost`, `src/ffi/cpp/phdf5/context.cc`). The variable exists
  for SLATE / cuSOLVERMp GPU-Direct RDMA.
- The module's `lxrun` helper hardcodes `-N 1`. **It is not a multi-node
  launcher.** For multi-node, write the `srun` line above.

---

## Tuning {#tuning}

All figures: 2.000 GiB complex128 (32768 × 4096), 2-D mesh, `/pscratch`,
`slab_io=phdf5_ffi` forced, two reps per configuration, best of two,
world size asserted. Produced by one process per geometry that sets
`os.environ` and opens a fresh path per configuration (the C++ context
reads its knobs at `open_file` time), so all rows in a column share one
container bring-up.

Two geometries:

- **N1** — 1 node, 4 ranks, 2×2 mesh, 512 MiB per rank.
- **N4** — 4 nodes, 16 ranks, 4×4 mesh, 128 MiB per rank.

| configuration | N1 write | N1 read | N4 write | N4 read |
|---|---|---|---|---|
| **default** (collective W+R, non-collective meta, 16 × 4 MiB) | 0.638 | 1.212 | 1.988 | 3.147 |
| default, repeated (noise) | 0.640 | 1.209 | 2.045 | 3.208 |
| `COLLECTIVE_WRITES=0` (independent writes) | **0.943** | — | **0.068** | — |
| `COLL_META=1` (collective metadata) | 0.644 | 1.215 | 1.963 | 3.177 |
| `INDEPENDENT=1` (independent reads) | — | **3.744** | — | 1.708 |
| stripe 1 × 1 MiB (the `/pscratch` default) | 0.616 | 1.006 | 0.695 | 0.761 |
| stripe 4 × 4 MiB | 0.688 | 1.209 | 1.377 | 1.487 |
| stripe 8 × 4 MiB | 0.660 | 1.211 | 1.735 | 2.332 |
| stripe 48 × 4 MiB | 0.568 | 1.220 | 1.829 | 3.452 |
| **stripe 16 × 1 MiB** | **0.802** | **2.240** | **2.941** | **4.267** |
| stripe 16 × 16 MiB | 0.777 | 1.633 | 2.532 | 4.042 |
| `CB_NODES=4`, `CB_WRITE=enable` | 0.641 | 1.206 | 1.397 | 1.854 |
| `ALIGN_MB=0` (alignment off) | 0.662 | 1.204 | 2.101 | 3.196 |

Units: GiB/s aggregate for the whole file. Repeat rows bracket the noise
at **±1.5 % or better**, so differences below ~5 % are not real.

**Reads are warm-cache.** Each read follows the write of the same file in
the same process, so the data is in the client page cache. The read column
therefore measures the MPI-IO path, not Lustre read bandwidth; use it to
compare configurations, not to size a cold restart.

### What the numbers say

**Collective writes stay on, and the reason is variance, not the mean.**
The headline number is that independent writes measured **0.068 GiB/s at
N4** — a 29× cliff, a run that never finishes. But the honest version is
more interesting, and a second sweep found it: that cliff is
*striping-dependent*. At 16 × 1 MiB, independent writes measured 2.927
GiB/s at N4 and 0.915 at N1 — level with collective, or slightly better.
The catastrophe was independent writes **at 4 MiB striping** specifically.

So the case for collective is not "always faster". It is:

| | collective | independent |
|---|---|---|
| N1, 16×4M | 0.654 | 0.943 |
| N4, 16×4M | 2.07 | **0.068** |
| N1, 16×1M | 0.818 | 0.915 |
| N4, 16×1M | 2.93 | 2.927 |

Collective never varies by more than ~10 % from the best available at its
geometry. Independent varies by **43×** across a single unrelated knob,
and one of those corners is unusable. With hundreds of ranks in the
envelope and no way to sweep every deployment, take the configuration
whose worst case is bounded. Two-phase aggregation is doing exactly what
it is for: making the strided 2-D tile pattern insensitive to how the
filesystem underneath is laid out.

> Note for anyone holding an older brief: `use_collective_write` was
> flipped to `true` on 2026-07-27 (`ctx.h`, `context.cc`). Any statement
> that "the GPU write path is independent" describes a configuration that
> was abandoned, and any "4.45 GB/s collective" figure predates the
> current one. The measurements above are on the current default.

**Collective reads stay on** for the same directional reason (1.84×
faster at N4, though slower at N1), and because the read column is
cache-warm and so is the weaker evidence of the two. Not changed.

**Collective metadata stays off.** Within noise at both geometries
(≤ 1.3 % either way). No measured reason to turn it on; keep the simpler
default.

**ROMIO collective-buffering hints stay unset.** `cb_nodes` + `cb_write`
cost 30 % at N4 and were neutral at N1. Cray MPICH's own heuristics beat
hand-set hints here. Leave them alone.

**Alignment is not load-bearing** at this size — turning it off is within
noise at both geometries.

**Striping is the largest single lever, and the folklore about it was
wrong in an instructive way.** `lxrun`'s comment claimed the `/pscratch`
default of 1 × 1 MiB "caps ~30 MB/s/rank". The default is confirmed
(`lfs getstripe -d /pscratch/sd/j/jackm` → `stripe_count: 1
stripe_size: 1048576`), but the cap is **not per-rank**: a 1-stripe file
measured 0.616 GiB/s at 4 ranks and 0.695 GiB/s at 16 ranks. Quadrupling
the ranks bought 13 %. It is a **per-file, single-OST ceiling of roughly
0.65 GiB/s**, and adding ranks does not move it. Stated per-rank the
number looks like something more parallelism can fix; stated per file it
is obviously a striping problem.

Stripe layouts were verified applied, not assumed — `lfs getstripe -c -S`
on each swept file returned exactly the requested count and size, so the
`LORRAX_PHDF5_STRIPE_*` hints do reach Lustre through ROMIO.

---

## Defaults {#defaults}

| knob | default | basis |
|---|---|---|
| `LORRAX_PHDF5_COLLECTIVE_WRITES` | `1` | 29× cliff at N4 when off; the asymmetry decides |
| `LORRAX_PHDF5_INDEPENDENT` (reads) | `0` (collective) | 1.84× at N4; read evidence is cache-warm, so no change on it |
| `LORRAX_PHDF5_COLL_META` | `0` | within noise both geometries — keep the simpler one |
| `LORRAX_PHDF5_STRIPE_COUNT` | `16` | 8 wins at N1 and loses at N4; 32 the reverse; 16 wins both. **Superseded in principle, not yet in this branch** — see below |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | `1M` | **changed from `4M`** — the only layout that wins at both geometries (see below) |
| `LORRAX_PHDF5_ALIGN_MB` | `4` | unchanged; `4`/`1`/`0` all inside repeat noise, so not worth a second knob to keep in sync |
| `LORRAX_PHDF5_CB_*` | unset | hand-set hints cost 30 % at N4 |
| `lxrun` pre-stripe | `-c 16 -S 1M` | same basis as `STRIPE_SIZE_FS` |

Everything except the stripe size is the configuration that was already
in the tree; the sweep is its measured justification rather than a change.

> **A fixed stripe count of 16 is the wrong shape, and the replacement is
> approved but NOT in this branch.** The reason 16 wins the sweep above is
> that the sweep ran at 4 and 16 ranks. ROMIO sets
> `cb_nodes = min(striping_factor, nranks)`, so **the stripe count IS the
> aggregator count** — pinning it to 16 caps aggregation at 16 however many
> ranks write, which is exactly backwards for a design envelope of hundreds.
> The owner approved `stripe_count = nranks` on 2026-08-05.
>
> **Status, verified 2026-08-06:** it is implemented on
> `feat/slab-io-stripe-nranks-2026-08-06` (`e5c9618`), which is **not an
> ancestor of this branch**. There, `_stripe_policy(nranks)` clamps to
> [4, 128], ramps the unit 1 → 4 MiB, and **refuses a negative count**
> (a negative `striping_factor` means "every OST on the filesystem", the
> maximum-contention layout). Note that even on that branch the policy is
> Python-side only: `context.cc:463` is still the literal `"16"`.
>
> Until that lands, both sites in this branch default to 16 and the numbers
> in the table above are the ones that apply.

The stripe-size choice, across both geometries (GiB/s write / read):

| layout | 1 node / 4 ranks | 4 nodes / 16 ranks |
|---|---|---|
| 16 × 4 MiB *(previous default)* | 0.654 / 1.30 | 2.07 / 3.23 |
| **16 × 1 MiB** *(new default)* | **0.818 / 2.29** | **2.93 / 4.74** |
| 16 × 2 MiB | 0.626 / 1.30 | **3.12 / 5.06** |
| 32 × 1 MiB | 0.754 / 2.28 | 2.78 / **5.52** |
| 8 × 1 MiB | **0.867** / 2.28 | 2.52 / 3.53 |

2 MiB is the fastest configuration measured at 4 nodes and the *worst*
measured at 1 node. 32-wide has the best read at 4 nodes and a poor write
at 1. 8-wide has the best write at 1 node and a poor read at 4. Only
16 × 1 MiB is at or near the top of every column, so it is the default —
elegance here means one layout that does not have to be tuned per job,
not the maximum of any single cell.

Changed in three places that must agree: `src/ffi/cpp/phdf5/context.cc`
(the C++ writer's `striping_unit`), `src/file_io/_slab_io_mpi_host.py`
(the Python writer's, so one environment means one layout in every
writer), and `lxrun`'s `lfs setstripe` pre-stripe in the modulefile.

---

## Certification {#certification}

Scope of every claim below: **inside the Shifter container, on compute
nodes**, launched by raw `srun … select_gpu.sh … shifter … in_container.sh`
— not `lxrun` (which is `-N 1` only), and not from a login node. Job
56389339, branch `fix/run-shifter-nvhpc-subpath-2026-08-05`.

All four cells, `slab_io=phdf5_ffi` **forced** so the router cannot mask
the result, each asserting `MPI_Comm_size()` against `jax.process_count()`
before writing anything:

| cell | geometry | world size | round-trip | payload md5 |
|---|---|---|---|---|
| **GPU** write + read | 1 node, 4 ranks, 2×2 | 4, asserted | bit-exact | `b738748c4803951a7dc5230121e1e5b5` |
| **GPU** write + read | 4 nodes, 16 ranks, 4×4 | 16, asserted | bit-exact | `b738748c4803951a7dc5230121e1e5b5` |
| **CPU** write + read | 1 node, 4 ranks, 2×2 | 4, asserted | bit-exact | `27a17944d643fad537a0b98e2774bec7` |
| **CPU** write + read | 4 nodes, 16 ranks, 4×4 | 16, asserted | bit-exact | `27a17944d643fad537a0b98e2774bec7` |

The md5 is of the **dataset payload**, re-read serially with plain h5py,
for the same logical 8192 × 4096 complex128 array. Within each platform
the 4-node/16-rank and 1-node/4-rank digests are **equal**: the same
logical array written through a 4×4 mesh and a 2×2 mesh is byte-identical
on disk. That is the multi-node write certification.

The GPU and CPU digests differ from each other, and that is not an I/O
result — the test array is built with `jnp.sin`, which differs in the last
ulp between the CUDA and host backends. The I/O claim is the round-trip
column and the within-platform digest match.

(The raw HDF5 *files* differ in metadata/layout bytes even when payloads
match, which is why the payload is hashed rather than the file.)

The routing decision was verified separately at 4 nodes after the fix —
`slab_io=auto` on 16 GPUs now prints:

```
[config] slab_io=auto on GPU backend: CUDA FFI exports the collective phdf5
write handler and MPI can bootstrap (launcher PMI environment present
(PMI_CONTROL_PORT)).  Routing SlabIO through PHDF5_FFI [processes=16,
devices=16, local_devices=1, SLURM_NNODES=4, SLURM_NTASKS=16].
```

### Re-verified against the rebuilt library

The stripe-size default change means a rebuilt `liblorrax_ffi.so`
(sha256 `323f4f0e…`, stamped `PROVENANCE`). With **no tuning environment
set at all**, that binary produced 16 × 1 MiB files (`lfs getstripe`
confirmed) and:

| geometry | write | read | round-trip |
|---|---|---|---|
| 1 node, 4 ranks | 0.812 GiB/s | 2.261 GiB/s | bit-exact |
| 4 nodes, 16 ranks | 2.655 GiB/s | 4.717 GiB/s | bit-exact |

against 0.654 / 1.30 and 2.07 / 3.23 for the old default — **+24 % / +74 %
and +28 % / +46 %**, for a one-word change.

---

## Failure modes {#failures}

**Singleton MPI.** *Looks like:* everything succeeds; output is plausible;
at 16 ranks the file even has the right size. *Detect:* the world-size
assertion above. *Cause:* `--mpi=pmi2` / `--mpi=pmix` against Shifter's
Cray MPICH. *Fix:* `--mpi=cray_shasta`.

**Swallowed MPI bring-up.** *Looks like:* one line,
`[phdf5 init_mpi] skipped: …`, then a run that appears to work. *Fixed
2026-08-05* — `_setup_runtime` now raises, naming `--mpi=cray_shasta`.
This is how a silent bring-up failure could be mistaken for a
platform-level "phdf5 doesn't work multi-node" conclusion.

**FFI library will not load.** *Looks like:* the router demotes with
*"the … FFI library could not be loaded: OSError: lib….so: cannot open
shared object file"*. **This says nothing about whether the handler is
compiled** — the three-state probe exists precisely to keep you from
rebuilding a library that is fine. It is a `LD_LIBRARY_PATH` / staging
problem. Run `ldd` on the `.so` *inside the container*.

Worked example, the CPU leg on 2026-08-05. `config/perlmutter/build_ffi_host.sh`
links the host lib against `cray-hdf5-parallel/1.14.3.7` and `cray-fftw`,
and against the `gpu_backend=none` SLATE build. None of those are in the
container by default:

| missing SONAME | where it lives | how to expose it |
|---|---|---|
| `libhdf5_parallel_gnu.so.310` | cray-hdf5-parallel **1.14** | **DONE 2026-08-06**, and it was not enough on its own: the stage is now `$HOME/software/lorrax_phdf5_cray_1.14.3.7/stage` (`site_config.sh`), but the DEVICE leg linked the old 1.12 stage, so staging 1.14 *beside* it would have mapped two HDF5 majors into one process instead of failing loudly. Both legs were moved together and GATE 7 (`src/ffi/cpp/gate_one_hdf5.sh`) now refuses the skew and the two-HDF5 "repair" alike |
| `libfftw3.so.mpi31.3` (+ `f`, `_omp` variants) | `/opt/cray/pe/fftw/*/x86_milan/lib` | `/opt/cray` is **not** a valid Shifter `--volume` source; copy under `$HOME/software` and bind-mount |
| `libslate.so.2`, `libblaspp`, `liblapackpp` | `$HOME/software/slate_builds/cpu/install/lib64` | `/global/homes` is siteFs-visible, so put it on `LD_LIBRARY_PATH` directly |

`phdf5_stage_cray.sh` used to fall back to a hardcoded 1.12 path when
`HDF5_DIR` was unset, so `module load cray-hdf5-parallel/1.14.3.7` inside a
non-login shell that never initialised Lmod produced a 1.12 stage with no
warning. **Since 2026-08-06 it refuses instead**, naming `HDF5_DIR` and the
`module load` that sets it (same for `MPICH_DIR`). `CRAY_HDF5_PATH` /
`CRAY_MPICH_PATH` still take precedence — an explicit path is a stated
fact, not a guess.

**`MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is
not linked`.** *Looks like:* an immediate MPI abort (rank 0 rc=255, peers
segfault) the moment anything calls `MPI_Init_thread`. *Cause (fixed
2026-08-06):* `in_container.sh` exported `MPICH_GPU_SUPPORT_ENABLED=1`
**unconditionally**, which is right for the GPU leg (it is paired with an
`LD_PRELOAD` of the CUDA-12 `libmpi_gtl_cuda.so.0`) and wrong for a
CPU-platform launch that has no GTL to preload — so the CPU FFI leg was
unusable in-container without a manual override. The variable is now
**per platform**: `run_shifter.sh` resolves `LORRAX_PLATFORM` (inferred
from `JAX_PLATFORMS` when unset) and `in_container.sh` exports `1` on
`gpu`, `0` on `cpu`. It is re-derived inside the container rather than
passed through because shifter's `--module=mpich` *unsets*
`MPICH_GPU_SUPPORT_ENABLED` on the way in; the two `LORRAX_*` variables
survive that and carry the intent.

So the CPU leg is now simply:

```
JAX_PLATFORMS=cpu src/ffi/cpp/run_shifter.sh python3 …
```

*Escape hatch:* `LORRAX_MPICH_GPU_SUPPORT=0|1` overrides the inference
outright. (The CPU cells in [Certification](#certification) predate the fix
and were run with the old manual
`… in_container.sh env MPICH_GPU_SUPPORT_ENABLED=0 python3 …`.)

**`ldd` on the login node lies about the MPI closure.** *Looks like:*
`libmpi_gnu_91` and `libmpi_gnu_123` resolving to two distinct real files,
i.e. an apparent two-MPI-ABI defect. **There is no such defect** — the second
name is a deliberate SONAME alias that only resolves inside the container.
This inspection produced a false defect report on 2026-08-05, retracted
2026-08-06. *Rule: any claim about the library closure must be measured
inside the container, on a compute node.* Mechanism, the alias, and the
`gate_one_mpi.sh` in-container gate: [`ffi_layout.md`](ffi_layout.md) §7c,
which owns it.

**Routing to `H5PY_ALLGATHER` at scale.** *Looks like:* a routing banner
naming that tier, then a run that OOMs rank 0 or takes forever. See
[the refusal note](#h5py_allgather-is-a-refusal-not-a-fallback).

**`ad_cray_write_coll.c:669` OOM.** *Looks like:* an MPI-IO abort inside
the collective write at ≳ 1 GiB per rank on Cray MPICH. Historical, from
before the current defaults; not reproduced at 512 MiB/rank in this
campaign. `LORRAX_PHDF5_COLLECTIVE_WRITES=0` is the escape hatch, at the
cost documented above — prefer splitting the write.

**Slow first write with no other symptom.** Check the file's Lustre
layout: `lfs getstripe -c -S <file>`. A 1-stripe file is capped near
0.65 GiB/s no matter how many ranks write it.

---

## Build provenance {#provenance}

Both Perlmutter build legs now stamp a `PROVENANCE` file beside the `.so`
(`src/ffi/cpp/stage/stamp_provenance.sh`, called from
`src/ffi/cpp/build.sh` and `config/perlmutter/build_ffi_host.sh`). It
records the git rev, branch, dirty flag, sha256, build time, host,
exported-symbol count, and the leg's resolved vendor paths.
`ffi_loader.build_provenance()` prints it in every run's startup report.

This is not bookkeeping. On 2026-08-05 a 4-node log was analysed by
inspecting the on-disk `liblorrax_ffi.so`, which had been rebuilt **seven
minutes after that log's last write** — so the ELF being read was provably
not the ELF that produced the numbers, and nothing on disk said so. An
unstamped build makes "which source produced this artifact" permanently
unanswerable, and this project asks that question constantly.

---

## Reproducing {#reproducing}

The gate and sweep drivers used for everything on this page are
`slabio_gate.py` and `slabio_sweep.py` (world-size assertion, forced
backend, bit-exact round-trip, geometry-invariant payload digest;
`--router-only` prints what `auto` would choose without doing I/O). Both
must be launched with the raw `srun` line in
[Launcher requirements](#launcher) — a measurement taken in the wrong
environment is worse than no measurement.
