# SlabIO — the sharded-slab HDF5 transport

*Verified against `src/` and against measurements taken inside the Shifter
container on Perlmutter, 2026-08-05/06 (allocations 56389339, 56405696,
4 × A100-40G nodes). Where this page and older prose disagree, this page
wins. Every number below names the command that produced it and the
filesystem it ran on.*

**This page owns SlabIO**: the tile contract, the caller-facing API, the
launcher requirement, the striping measurements, and the certification. It
does not own the owner rulings behind them
([`decisions.md`](decisions.md)), the native layer underneath
([`ffi_layout.md`](ffi_layout.md)), or knob spellings
([`../dev/env_vars.md`](../dev/env_vars.md)) — see the
[register](../index.md#register).

Conventions used here, same as [`services.md`](services.md):

- **Level / imports.** `file_io.slab_io` is L3 and imports only downhill.
  It used to reach *up* into `gw.gw_config` for the backend enum; that
  import is gone with the enum (2026-08-06).
- **Announce-or-refuse.** A deployment that cannot run says so, naming the
  probe that declined. Nothing about the I/O path is ever silent.
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

The owner's statement of it, verbatim:

> there should always exist a valid path that does not materialize
> N_mu^2 on any proc because it is a guaranteed OOM.

Consequences that follow directly, and that callers must not work around:

- `write_slab` takes the array the rank already owns, in its existing
  sharding. It does not reshard, gather, or replicate.
- `read_slab` returns a sharded array. The rank reads its own hyperslab
  and no one else's.
- Peak host memory attributable to SlabIO is one rank's tile plus the
  pinned staging buffer, independent of process count.

**Since 2026-08-06 the contract is enforced by construction rather than by
a check: there is one transport, so there is nothing else to select.**

---

## The API — everything a caller must know {#api}

```python
from file_io.slab_io import SlabIO

with SlabIO(path, mode="w", mesh=mesh) as io:
    io.create_dataset("V_qmunu", shape=(n_q, n_mu, n_mu), dtype=c128)
    io.write_slab("V_qmunu", V)                    # V may be padded
    W = io.read_slab("V_qmunu", partition_spec=P(None, "x", "y"))
```

That is the entire surface. A caller states **a path, a mode, a mesh, and
logical shapes**. A caller does *not*:

| ...need to know | because |
|---|---|
| which backend | there is one |
| Lustre stripe count / size | resolved from the rank count by `_stripe_policy` and exported before `H5Fcreate`; see [Tuning](#tuning) |
| ROMIO / MPI-IO hints | set by the C++ context; read them back with `MPI_File_get_info`, never assume |
| whether the MPI world matches the JAX world | asserted at the first collective open, unconditionally ([Failure modes](#failures)) |
| the mesh-divisible extent | omit `shape` and `read_slab` rounds up for you (below) |
| whether the deployment can do parallel I/O | `assert_available()` runs at open and refuses naming the probe |

**Padding is SlabIO's business, not the caller's** (decisions.md
2026-08-04):

- `write_slab(name, A, offset=...)` accepts any `A`. What reaches the file
  is `min(A.shape, dataset - offset)` per dim, derived from the dataset —
  a buffer padded for mesh divisibility needs no argument at all.
- `read_slab(name, partition_spec=spec)` **with no `shape`** returns the
  dataset rounded UP to the mesh-divisible extent under `spec`,
  zero-filled past the dataset. That is the padded consumer buffer.
  This is new in 2026-08-06 and it is the point: `shape=None` used to mean
  "the dataset's own shape", which *refuses* whenever that shape is not
  mesh-divisible — the normal case, since N_mu is a physics number and not
  a multiple of the device count. Every caller therefore computed the
  round-up itself. The easy call is now the correct call.
- `shape` may still be stated exactly and is returned exactly. It must be
  mesh-divisible under `partition_spec`: the return value is a `jax.Array`
  of that shape sharded that way, and JAX will not build one at a
  non-divisible extent, so there is nothing to trim to.
- `valid_shape` survives **only** as the ragged-chunk override — a chunk
  buffer whose tail is genuinely not part of this write. Routine calls do
  not pass it.

Two rounding rules exist and must not be confused. `mesh_divisible_shape`
rounds each dim by the product of the mesh axes that shard *that dim*;
`runtime.padding.padded_mu_extent` rounds the μ extent by the **total**
device count, which is the separate in-memory convention `Meta.n_rmu_padded`
carries. The restart path deliberately uses the latter.

---

## Availability — one probe, one refusal {#availability}

`SlabIO.__init__` calls `assert_available()` before touching the inode. It
checks two things, which fail for unrelated reasons and are reported
separately:

1. this platform's FFI library exports `lorrax_phdf5_write`, probed with
   `ffi_loader.probe_target` — whose three states ("unknown target" /
   "could not be loaded" / "does not export") are three *different*
   repairs, which is why the refusal quotes its reason verbatim instead of
   reducing it to a bool;
2. MPI can bootstrap here. Handler presence is not capability: the write
   calls `MPI_Init_thread`, and on a bare launch with no PMI environment
   Intel MPI aborts inside `MPIR_pmi_init` rather than returning an error
   (job 7884926), so the probe runs in a throwaway subprocess.

The probe is cached per process. **Nothing here keys on process count.**
The old refusals all did, because the tier they guarded was legal at
exactly one process; this one guards the only path there is, so it either
works or the deployment is broken.

### There used to be three tiers and a router {#tiers-history}

Kept because the deletion is the design decision, and because the shape of
the mistake recurs.

`PHDF5_FFI` (this one), `PHDF5_HOST` (the same collective MPI-IO driven
from Python by mpi4py + h5py-parallel) and `H5PY_ALLGATHER` (gather the
whole global array onto rank 0, write it with serial h5py), selected by a
`slab_io` deck key with an `auto` platform router and a deprecated
`use_ffi_io` boolean.

**`H5PY_ALLGATHER` was refused at SEVEN separate doors**, each closure
landed and reported as complete, and an eighth door kept being found. The
doors were: the `auto` router running out of tiers; the deck naming the
tier (or `use_ffi_io = false`) post-checked after the whole precedence
chain; three separate entries in `SlabIO.__init__`'s argument normaliser
(named enum, unstated default, legacy boolean); the shared gate those
three funnel through; and a resolve-time check in the sharded-σ layout
path. An eighth, entirely ungated route survived all of them:
`gw/gw_init.py` imported `_to_host` **directly from the allgather backend
module**, bypassing every refusal.

A tier that must be refused at seven doors is not a tier; it is dead code
wearing a safety label. Deleted 2026-08-06 — the module, the enum member,
the deck value, all seven refusals, and the direct import (repointed at
`common.collectives.gather_to_host`, the sanctioned L3 gather it was a
private copy of).

**`PHDF5_HOST` was deleted with it**, and the evidence for that is worth
stating because it is a different argument. Its *only* selection condition
was a host FFI library built before workstream AE, which exports the phdf5
read symbols and not `PhdfWriteHostFfi`. Measured 2026-08-06 with `nm -D`:

| library | `PhdfWriteHostFfi` | dated |
|---|---|---|
| Perlmutter `lorrax_P/.../build_host/liblorrax_ffi_host.so` | **present** | live |
| Perlmutter `lorrax_P/.../build/liblorrax_ffi.so` (`PhdfWriteFfi`) | **present** | live |
| Frontera `$WORK/lorrax_ffi_unified/build_host_W/` | **present** | 2026-07-26 |
| Frontera `$WORK/lorrax_ffi_unified/build_host_V/` | absent | 2026-07-26 |
| Frontera `$WORK/lorrax_ffi_unified/build_host/` | absent | 2026-07-25 |

So the tier's condition is false on every deployed library; the two that
fail the probe are dated A/B control builds in a staging directory, and a
post-AE build sits beside them. Frontera's live tree (`$WORK/lorrax`) has
**no compiled `.so` at all**, so nothing there resolves either tier today.
And `PHDF5_HOST` additionally needs an mpi4py + `HDF5_MPI=ON` h5py overlay
that the FFI path does not — on Frontera's default python, `h5py` does not
even import (`libhdf5.so.103: cannot open shared object file`).

A tier that requires *more* to do the *same* thing, selected only by a
stale artifact, is not a fallback. The correct response to a stale `.so`
is a refusal naming it, which is the repo-wide contract (CLAIMS 81) and is
what `assert_available` now does.

**Not deleted, and not a tier:** `bse_io`'s serial h5py readers. They
hyperslab exactly one rank's (μ, ν) tile, allgather nothing, and are
memory-correct at any process count — they are simply ~17× slower
(0.17 GiB/s at P=4, CLAIMS 76, vs 2.919 GiB/s for the tile path at 16
ranks, CLAIMS 69) because they issue `nq × μ/px` short row-runs one at a
time with no collective buffering. `bse_io` falls back to them, loudly,
when `probe_availability()` declines. Slow and correct is a legitimate
fallback; the tier that was deleted was neither.

---

## Restart at P>1 {#restart}

`restart = true` reads `V_qmunu`, `S_qmunu`, `V0_noG0_munu`,
`psi_full_y` and (bispinor) `psi_full_y_transverse` back from
`isdf_tensors_<n_rmu>.h5`. Until 2026-08-06
`tagged_arrays.read_restart_state_from_h5` read every one of them with
`[:]` — the whole `(nq, μ, μ)` tensor **on every rank** — and only then
applied `jnp.pad` on both μ axes and `with_sharding_constraint`. Measured
at P=4 (job 56389339, MoS2 6×6, N_mu=1496, nq=36): **+1.53 GiB VmHWM per
rank**, silently. At the envelope (N_mu=20000, nq=64) the same read is
**381.47 GiB per rank**.

It was therefore *guarded off* above one process, which removed a
capability that had worked at deck scale and left `restart = true` with no
P>1 story. It is now on the tile path, following
`bse_io.load_bse_data_from_restart_sharded`:

1. one serial-h5py pass reads **shapes** and the small replicated arrays
   (`enk_full`, `G0_mu_nu`, the stamps), then closes — two live handles on
   one file is a hazard nobody needs;
2. SlabIO reads the N_mu²-class tensors and ψ as per-rank tiles, asked for
   at the **padded** extent so the zero-fill past the dataset *is* the pad.
   No `jnp.pad`, and no sharding constraint applied to an already-resident
   global array.

What stays on serial h5py is μ-class or smaller (`G0` is 320 KB at the
envelope) and is needed whole on every rank. The doctrine forbids
materialising an N_mu²-class object, not reading a vector.

**Spinor and bispinor.** `nspinor` is read from the ψ dataset and carried
through as a replicated axis at its on-disk extent — 2 for spinor, 4 for
bispinor — and is **never padded**. It is gated to `{1, 2, 4}`: an
unexpected extent would otherwise sail through as a perfectly shardable
replicated axis and misindex every downstream ψ contraction with no shape
error. The bispinor `psi_full_y_transverse` is read at its **own** μ
extent, since the transverse centroid count differs from the charge one
and carries its own pad; the stamped `n_rmu_transverse_logical` is
cross-checked against the dataset's extent *before any bytes move*.

**Parity is bit equality**, not a tolerance: this is an element-*selection*
change, not a reduction-order one. `tests/multi_device/restart_sharded_parity.py`
asserts every returned element equals the serial-h5py read of the same
file at the same index, at `RESTART_NS` ∈ {1, 2, 4}, and additionally that
the pad rows are exact zeros and that no rank's shard is the whole array.
`tests/test_restart_pad_roundtrip.py` covers the same round trip at 1×1.

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

### Striping is not a constant: the stripe count is the aggregator count

Everything above this heading was measured at **16 ranks**, where the old
default of `16 × 1 MiB` happens to equal `nranks × 1 MiB`. That coincidence
is what made a constant look like a tuning.

The mechanism, **read back from ROMIO rather than assumed**
(`LORRAX_PHDF5_DUMP_HINTS=1`, which dumps `MPI_File_get_info` after
`H5Fcreate` — the hints ROMIO *retained*, not the ones we asked for):

| requested `striping_factor` | 4 ranks → `cb_nodes` | 16 ranks → `cb_nodes` |
|---|---|---|
| 1 | — | 1 |
| 4 | **4** | 4 |
| 16 | **4** | 16 |
| 64 | — | 16 |
| 128 | **4** | 16 |
| 366 (`-1`, every OST) | — | 16 |

`cb_nodes = min(striping_factor, nranks)`, exactly, at every point measured
— and we never set `cb_nodes` ourselves (`romio_cb_write = automatic`,
`cb_config_list = *:*` in the same dump). **So the stripe count IS the
collective-buffering aggregator count.** A fixed 16 pins `cb_nodes = 16`
forever, and every rank past the sixteenth is a rank that does not
aggregate. Per-rank throughput rolls over just past 16 ranks for that
reason and no other.

Write bandwidth, GiB/s aggregate, `/pscratch`, phase-separated harness:

| ranks | payload | `16 × 1 MiB` | `nranks` stripes | gain | artifact |
|---|---|---|---|---|---|
| 4 | 2.00 GiB | 0.803 | **0.873** (4 × 1M) | +8.6 % | `w_P4.json` |
| 4 | 2.00 GiB | 0.813 | **0.906** (4 × 1M) | +11 % | `w_SC4.json` |
| 16 | 8.00 GiB | 3.152 | 3.152 (16 × 1M) | *null control* | `w_SC16.json` |
| 64 | 32.00 GiB | 5.189 | **10.630** (64 × 4M) | +105 % | `w_SC64.json` |
| 64 | **381.47 GiB** | 7.403 | **13.222** (64 × 4M) | **+79 %** | `w_EV64.json` |
| 100 | 50.00 GiB | 7.872 | **15.216** (100 × 4M) | +93 % | `w_R100.json` |

The 16-rank row is the **null control**, not a win: there the two policies
are literally the same configuration, and a policy that changed that row
would be a policy with a bug. The 381 GiB row is the real envelope payload
— `V_qmunu`, `(nq=64, 20000, 20000)` complex128 — where the old default
leaves **44 % of the write bandwidth on the floor**. `stripe = nranks` won
or tied at every rank count measured.

### The read advantage in the table above was the page cache

The `16 × 4M → 16 × 1M` change was justified partly on a **76 % read
improvement**. That number was warm: `slabio_sweep.py` read the file it had
just written, in the same process, so every byte came from the writing
node's page cache. A production restart reads files nobody just wrote.

Re-measured with write and read as separate `--phase` invocations on
**disjoint node sets** (write `nid003837`, read `nid003840`; `lctl` is
absent from the compute image and `drop_caches` needs root, so a disjoint
node set is the only honest cold-read instrument here):

| 4 ranks, 2.00 GiB | warm read | cold read | write |
|---|---|---|---|
| `16 × 1 MiB` | 2.29 | 0.932 | 0.803 |
| `4 × 1 MiB` (policy) | — | 0.932 | 0.873 |

Cold read ≈ write throughput, and the layout advantage shrinks from 76 % to
**5.7 %**. Reads are not where the lever is. **Any benchmark on this
filesystem must use disjoint node sets or it is measuring the cache**, and
a read number quoted without its node sets is not a measurement.

### What the policy is

`file_io._slab_io_ffi._stripe_policy(nranks)` — one pure function of the
rank count, no env, no MPI, no filesystem, so it is testable at rank counts
no allocation on this machine can reach:

```
striping_factor = min(max(nranks, 4), 128)
striping_unit   = the power of two NEAREST IN LOG2 to (nranks/16) MiB,
                  clamped to [1 MiB, 4 MiB]
```

which is `1 MiB` below 22.6 ranks, `2 MiB` to 45.3, `4 MiB` above.

**The unit is a ramp, not a step at 64.** Both ends are measured
(`16 × 4M` = 2.07 vs `16 × 1M` = 2.93 at 16 ranks; `64 × 1M` = 7.068 vs
`64 × 4M` = 10.630 at 64), and between them the rule is the geometric
interpolation those two anchors imply, with no free parameter to pick.
It has to be argued rather than measured, because **the midpoint of that
range cannot be run here**: `resolve_mesh` requires a perfect-square device
count, so 32 ranks is not a legal geometry, and `16 < n < 64` contains no
power-of-two rank count that is also a legal mesh. The legal points in the
gap are 25, 36 and 49.

**4 MiB is a ceiling, not a trend to extrapolate.** The per-rank tile knee
is 4 MiB — at 16 ranks, tiles of 0.06 / 0.25 / 1 / 4 / 16 MiB gave
0.126 / 0.609 / 1.779 / 3.165 / 3.551 GiB/s: flat above 4 MiB, a cliff
below (`w_TS16.json`). `V_qmunu` at `N_mu = 20000` on 1024 ranks is
**~6.1 MiB per rank**, only 1.5× above that knee, so the envelope sits on
the edge of the cliff and a stripe unit above the per-rank tile would
starve aggregators.

**`-1` is refused, not passed through.** It reads like a maximum and
measures like a failure: on `/pscratch` (370 OSTs) it expands to 366
stripes and gave **0.105 GiB/s** at 64 ranks / 32 GiB and 1.118 at 16 ranks
/ 8 GiB, against 10.63 and 3.15 for the policy. Striping across every OST
is the maximum-*contention* layout, not the maximum-bandwidth one.

**Layouts are verified granted, not assumed.** `lfs getstripe -c -S` from a
login node (the compute image has no `lfs`) returned exactly the requested
`4 × 1M`, `16 × 1M` and `128 × 4M` on the files this branch wrote.

#### The upper clamp is NOT verified in-policy {#clamp-open}

`striping_factor = 128` is emitted only at `nranks >= 128`, and **no run at
that scale has been made**: the interactive QOS maximum is 4 nodes / 16
GPUs, so the largest legal mesh reachable here is 16 ranks. What *is*
measured is that 128 stripes behave badly when the rank count is *below*
128 — 1.014 GiB/s at 64 ranks and 0.825 at 100 ranks with `128 × 4M`,
against 10.63 and 15.22 for `nranks × 4M`, and 0.501 vs 0.869 at 4 ranks.
Those points are **off-policy** (the policy never emits a count above
`nranks`), so they do not condemn the clamp — but they do mean the clamp
endpoint has never been observed in the regime where the policy actually
uses it, and 128 is anomalous even among its neighbours: at 100 ranks
`sf = 150` and `sf = 200` measured 14.07 and 12.96 while `sf = 128`
measured 0.416. **Measure `nranks = 128` (or 121, the nearest legal mesh)
before an envelope run leans on it.**

---

## Defaults {#defaults}

| knob | default | basis |
|---|---|---|
| `LORRAX_PHDF5_COLLECTIVE_WRITES` | `1` | 29× cliff at N4 when off; the asymmetry decides |
| `LORRAX_PHDF5_INDEPENDENT` (reads) | `0` (collective) | 1.84× at N4; read evidence is cache-warm, so no change on it |
| `LORRAX_PHDF5_COLL_META` | `0` | within noise both geometries — keep the simpler one |
| `LORRAX_PHDF5_STRIPE_COUNT` | **`nranks`, clamped to [4, 128]** | **changed from `16`** — the stripe count is the aggregator count (`cb_nodes = min(striping_factor, nranks)`, read back from ROMIO); a constant pins the aggregator count. +79 % on the 381 GiB envelope payload at 64 ranks. `-1` is refused |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | **`1M`/`2M`/`4M` by rank count** | **now a ramp** — 1 MiB at <=16 ranks, 4 MiB at >=64, the power of two nearest in log2 to `nranks/16` MiB between; both ends measured, the midpoint is not runnable (32 ranks is not a legal mesh) |
| `LORRAX_PHDF5_ALIGN_MB` | `4` | unchanged; `4`/`1`/`0` all inside repeat noise, so not worth a second knob to keep in sync |
| `LORRAX_PHDF5_CB_*` | unset | hand-set hints cost 30 % at N4 |
| `lxrun` pre-stripe | `-c 16 -S 1M` | same basis as `STRIPE_SIZE_FS` |

Both striping rows are resolved by ONE pure function, `file_io._slab_io_ffi._stripe_policy(nranks)`, and an explicit `LORRAX_PHDF5_STRIPE_*` still wins over it. The FFI writer's hints are built in C++ whose only input is `getenv`, so the resolved values are exported back into `os.environ` before `open_file` — which is why a run log's environment remains a complete description of the layout, and why the `lfs getstripe` readback means anything.

> **Why the stripe count is `nranks` and not a constant.** The reason 16
> won the older sweep is that the sweep ran at 4 and 16 ranks. ROMIO sets
> `cb_nodes = min(striping_factor, nranks)`, so **the stripe count IS the
> aggregator count** -- pinning it to 16 caps aggregation at 16 however many
> ranks write, which is exactly backwards for a design envelope of hundreds.
> The owner approved `stripe_count = nranks` on 2026-08-05.
>
> **Status, verified 2026-08-06 in `integration/2026-08-06`:** implemented
> and IN this tree, via `feat/slab-io-stripe-nranks-2026-08-06` (`e5c9618`,
> `252b80d`), which reaches here as an ancestor of
> `feat/bse-slabio-2026-08-06`. `_stripe_policy(nranks)` clamps the count to
> [4, 128], ramps the unit 1 -> 4 MiB, and **refuses a negative count**
> (a negative `striping_factor` means "every OST on the filesystem", the
> maximum-contention layout). The measurements behind it are recorded above
> `_stripe_policy` in `src/file_io/_slab_io_ffi.py`.
>
> **The policy is now on BOTH sides, and the C++ side is UNBUILT.** It
> used to be Python-only: `context.cc` carried the literals `"16"` and
> 1 MiB as the fallback for an unset `LORRAX_PHDF5_STRIPE_*`, and on every
> LORRAX path the variables are set before C++ can read them
> (`_FfiBackend.__init__` calls `_export_striping_env()` immediately before
> `_open_file()`, which is the call in which `context.cc` builds the
> `MPI_Info` and `H5Fcreate` applies the layout). That masked, rather than
> removed, a real disagreement: a caller reaching `ffi.io.open_file`
> **without** going through `_FfiBackend` got 16 × 1 MiB and no warning,
> at any rank count. Two writers requesting *different* layouts is worse
> than both being wrong the same way, so `context.cc` now derives the
> layout from `ctx->world_size` via `stripe_policy_count` /
> `stripe_policy_unit`, which transcribe `_stripe_policy` line for line.
> `tests/test_slab_io_routing.py::test_cpp_stripe_policy_transcribes_the_python_one`
> extracts those two functions, compiles them with the host `g++` and diffs
> them against the Python over 0..4100 ranks.
>
> **Not verified, and here is the exact gap:** that test compiles two
> `static long` functions in isolation. `liblorrax_ffi*.so` has **not been
> rebuilt** with this change — no cluster, no toolchain, on the branch that
> made it — so nothing yet shows a deployed library requesting the policy
> layout. Required before leaning on it: rebuild both libs, then one
> `lfs getstripe -c -S` readback of a file written through
> `ffi.io.open_file` with `LORRAX_PHDF5_STRIPE_COUNT` **unset** at a rank
> count away from 16 (the old literal), which is the only geometry where
> the two answers differ.

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

**A gather at scale.** *Used to look like:* a routing banner naming
`H5PY_ALLGATHER`, then a run that OOMs rank 0 or takes forever. That tier
no longer exists ([history](#tiers-history)); if you see a whole global
array on one rank now, it is a caller doing its own `process_allgather`,
not SlabIO.

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
