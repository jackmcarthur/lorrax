# The emulated-mesh SlabIO tier — branch report

`feat/emulated-multidevice-cpu-2026-08-27`. Written because three claims on
this branch lived only in commit bodies, and a commit body is not a place
another lane reads: contract rule 11 wants API-shape changes *exchanged*,
and rule 25 wants a relayed number travelling with its artifact path.

**Status.** Local only. `git ls-remote --heads origin
feat/emulated-multidevice-cpu-2026-08-27` returns nothing, so nothing here
is pushed, landed or banked.

---

## 1. What the branch does, in one paragraph

`file_io.slab_io.SlabIO` gains a second backend for exactly one geometry:
an EMULATED mesh — `jax.process_count() == 1` with more mesh cells than
processes, which is what `--xla_force_host_platform_device_count=N`
produces. The phdf5 transport cannot serve it (its C++ handler derives
every hyperslab from `ctx->rank`, a per-process scalar, so four devices in
one process would all touch shard (0,0)), and `ffi.io.open_file` refuses it
at `src/ffi/io.py:120`. That refusal is untouched. The new tier,
`file_io._slab_io_serial._SerialBackend`, is chosen from a mesh PREDICATE
before any transport is constructed — never from a transport that failed —
and takes no deck key, no env var and no argument.

---

## 2. API-shape changes other lanes cannot see in a three-way merge

A required-parameter addition, a removed default, a changed return arity or
a new raise merges clean and breaks at run time. These are this branch's:

| # | Change | Commit | Who it can bite |
|---|--------|--------|-----------------|
| a | `SlabIO(mesh=…)` now READS the mesh at construction (`mesh_is_emulated`) and refuses anything without `.devices`. A duck-typed or sentinel mesh that used to sail through now raises `TypeError`. | `9681eda6` | anyone constructing `SlabIO` with a stub mesh in a test |
| b | `_slab_io_ffi._local_shard_and_global_offset` DELETED (no caller in the tree; the live copy is `wfn_loader._collectives`, untouched). | `71319bf7` | anything reaching it by `getattr`, or an unmerged branch |
| c | `_SerialBackend.__init__` now RAISES on a non-`cpu` mesh platform. | this round | a single-process multi-GPU mesh — see §3 |
| d | `_SerialBackend.__init__` now RAISES on a mode outside `w`/`a`/`r`. `r+`, `x` and `w-` were accepted before and are not any more. | this round | a caller passing an h5py mode SlabIO never supported |

(c) and (d) are refusals the FFI tier already had; the branch's first cut
gave the emulated tier a *wider* surface than the production one, which is
the parsed-but-unvalidated defect class.

---

## 3. The platform guard, and why it is not scope creep

`common.collectives.mesh_is_emulated` reads `devices.size` against
`process_count()` and nothing else. It is therefore **platform-blind**, so
it is equally true of a SINGLE-PROCESS MULTI-GPU mesh — `resolve_mesh()` on
a 4-GPU box, or a test harness child handed all four GPUs. That geometry is
not an emulation: it is the arm deleted with the cuSOLVERMg backend
(`src/bse/STATUS.md`), and on base it refused at `src/ffi/io.py:120` like
any other `p*q != P` mesh.

Routing it to a serial h5py tier would have RE-OPENED a deleted arm through
a path nobody chose — a downgrade where TASTE rule 10 requires a refusal.
`_SerialBackend` now refuses when `ffi.gate.mesh_ffi_platform(mesh) != "cpu"`.
The predicate itself is deliberately NOT narrowed: "more devices than
processes" is a true statement about that mesh, and callers that only ask
the question want the honest answer. Its docstring now names this third
direction.

Falsifying case, run:
`tests/test_slab_io_emulated_mesh.py::test_the_serial_tier_refuses_a_single_process_multi_gpu_mesh`
builds a 2x2 of devices whose `.platform` is `"gpu"` and asserts, on the
object itself, that the REAL `mesh_is_emulated` says True and the REAL
`mesh_ffi_platform` says `CUDA` — then that construction refuses before
making an inode. With the guard disabled the cell fails.

---

## 4. The route receipt — WHERE a parity claim should read the transport

Three receipts, and they do not have the same reach:

| receipt | pytest / script | driver, `LORRAX_DEBUG_PRINT=1` | driver, production default |
|---|---|---|---|
| the printed `[SlabIO] transport = serial` line | YES | YES | **NO** |
| `LORRAX_H5_JOURNAL=1` → `stack=h5py op=open` in `h5_journal.rank0.log` | YES | YES | **YES** |
| the driver's scientific report | — | — | not implemented |

A production driver installs `runtime.production_stream.ProductionStdout`,
which points `sys.stdout` at `/dev/null`; the printed line is ordinary
component chatter by that module's own definition and is discarded exactly
as designed. **So a parity claim made from an emulated arm should quote the
JOURNAL line, not the printed one.** The journal writes to a file through a
stream `file_io.h5_journal` owns, which the sink does not touch. Both facts
are executed rather than asserted in prose, by
`test_the_receipt_is_discarded_by_the_production_stdout_sink` and
`test_the_journal_receipt_survives_the_production_stdout_sink`.

The structural receipt, scoped: at P=1/D>1 the phdf5 transport cannot open a
file at all, so on an emulated run **HDF5 output that went through SlabIO**
went through this tier. It says nothing about the ~20 writers in the tree
holding their own `h5py.File` (`file_io.wfn_writer`, `gw.kin_ion_io`,
`bse.bse_window`, `bse.absorption_common`, `file_io.qp_wfn`, …).

### OWNER DECISION 1 — put the transport in the scientific report?

The report is the one surface `ProductionStdout.emit` writes to. Adding
`Slab transport : serial (emulated 2x2)` beside `Wavefunctions : <backend>
reader` would change the report format in four drivers and needs a ruling
on where the reporter gets the run's mesh. Not taken here.

### OWNER DECISION 2 — the zeta_loader anti-pattern row

`docs/services/zeta_loader.md` previously listed "handing the transport an
emulated multi-device mesh" as an ANTI-PATTERN. The branch rewrote the row
so the call is supported and only the CLAIM ("this is a 2x2 parallel run")
is forbidden. That is a doctrine move, not a doc fix, and it touches the
rule that names single-process multi-GPU geometries as a deleted arm.
Mitigating: the branch name is the mandate, this is CPU emulation rather
than multi-GPU, the tree already carries the pattern
(`test_distrib_la_emulated_mesh.py`, `test_wfn_loader_emulated_mesh.py`,
wfn_loader's auto-selected `eager` tier), and `slab_io.md` already blesses a
slow-and-correct serial h5py reader as "not a tier". It still wants one
explicit owner line rather than a commit body.

---

## 5. Memory: say which number you mean

`docs/architecture/slab_io.md` says peak host memory attributable to SlabIO
is one rank's tile. On the collective transport the tile and the result
nearly coincide. **On this tier they do not**, and an early version of the
module docstring claimed the smaller number for both.

MEASURED — WSL2, jax 0.9.1, `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, D=4,
`tracemalloc` around one call, 1024² f64 (8.39 MB global, 2.10 MB/shard),
`P('x','y')`:

```
peak-minus-baseline during the read  = 10.49 MB
still resident after it returns      =  8.39 MB   <- the RETURNED array,
                                                     4 shards, all in this
                                                     process
peak MINUS the returned array        =  2.10 MB   <- SlabIO staging = 1 shard
write_slab, same global              =  0.02 MB
```

The returned array is global-sized *by construction* on an emulated mesh —
that is what emulation is, and it is the tier's own thesis (the process
already holds every shard). What this layer is answerable for is the
staging above it, and the staging is one shard.

The audit that raised this read the global-sized peak as the fault of
`read_slab`'s replica cache. It is not: the CPU
`make_array_from_callback` aliases each numpy block straight into the device
buffer, so the cache held the very objects the result holds, and
before/after peaks are both 10.49 MB. The cache was still the wrong
structure on the spec that matters, and is now built only when replicas
exist — see §6.

---

## 6. Replica de-duplication, counted

MEASURED, callback calls / h5py reads for one 4x4 read on a 2x2:

| spec | cb calls | h5py reads | cache built |
|---|---|---|---|
| `P(None, None)` | 1 | 1 | no (jax de-duplicates this itself) |
| `P('x', None)` | 4 | 2 | yes |
| `P(None, 'y')` | 4 | 2 | yes |
| `P('x', 'y')` | 4 | 4 | no — every device has its own index |

`P(None,'y')` is the row that matters: its indices arrive ALTERNATING
(col0, col1, col0, col1), because devices are enumerated row-major. A
one-entry memo — the first design tried here — hits zero times on it and
reads 4. Measurement rejected that design; the row is in the parametrize
list so it cannot come back silently.

---

## 7. Evidence, with paths

SCOPE FOR EVERYTHING BELOW: this WSL2 box only. jax/jaxlib 0.9.1 via
`/home/jackm/projects/lorrax_cloud13/.venv/bin/python`, `JAX_PLATFORMS=cpu`.
NO cluster, NO GPU, NO multi-process leg, no `JAX09_ENV_OK` compute-node
launch. Snapshots, not properties.

**The environment IS part of the command.** Without these two the file_io
suites give 13 failed / 232 passed, because the `.so` lives in
`build_host_cloud/` (not on `ffi_loader`'s default search list) and needs a
libhdf5 from outside the worktree:

```bash
export LORRAX_FFI_HOST_SO=<wt>/src/ffi/cpp/build_host_cloud/liblorrax_ffi_host.so
export LD_LIBRARY_PATH=/home/jackm/projects/lorrax_cloud13/.native/phdf5-1.14.6/lib
export PYTHONPATH=<wt>/src:<wt>/services/*/src
```

### Cross-tier end-to-end, `gnppm_debug`

Two arms, same tree, same deck (`memory_per_device_gb = 2` in BOTH — the
fixture pins 28, which an emulated run takes as 4x28 GB of one host's RAM,
so it is set the same on both sides rather than being a difference between
them). The only difference is the device count, which is what selects the
transport. Both completed; stderr carries no traceback, refusal or
exception in either.

```
python -m gw.gw_jax -i gnppm_test.in                     # D=1, phdf5 FFI tier
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
python -m gw.gw_jax -i gnppm_test.in                     # D=4, serial tier
```

ROUTE RECEIPT, read from the journal FILE on a production driver run (this
is the receipt §4 says to quote, and here it is doing the job):

| arm | `stack=ffi` lines | `stack=h5py` lines |
|---|---|---|
| D=1 | **100** | 6 (the FFI tier's own metadata touches) |
| D=4 | **0** | **102** |

Zero `stack=ffi` at D=4 is the fact: the collective transport did not move
a byte, and the tier under test moved all of them.

SIGMA — `sigma_diag_gnppm_test.dat`, 414 (k, band) rows over 9 k-points:

| column | max abs delta |
|---|---|
| sigX, Re/Im sigC, Re/Im sigXC, VH, Eo | **0.000000e+00 eV** (bit-identical) |

Other artifacts: `sigma_freq_debug.dat` (414 x 14) bit-identical;
`eqp0.dat`/`eqp1.dat` agree over 423 rows at max 1.0e-09 eV, which is the
last printed digit of the writer's format; `sigma_mnk.h5` — the file the
tier under test WROTE — agrees to max 9.5e-10 absolute, all of it in the
OFF-DIAGONAL `sigma_c_kij_ev` / `sigma_total_kij_ev` / `hartree_kij_ev`
blocks at relative 1e-11 or better, which is reduction-order round-off
between a 1-device and a 4-device run. The diagonal, which is what the
frozen gate reads, is exact.

Evidence: `scratchpad/fix_multidev/acceptance_result.txt`, run dirs
`acc_d1/` and `acc_d4/` beside it.

The BASE tree at the same geometry dies at `src/ffi/io.py:121`
`ValueError: mesh 2x2=4 != jax.process_count()=1` inside
`src/gw/isdf_fitting.py:821` — which is what makes the D=4 arm evidence
that the tier ran, rather than evidence that nothing changed.

### Suites

| command | result |
|---|---|
| the seven-file D=1 batch | 251 passed, 2 skipped |
| `tests/test_slab_io_emulated_mesh.py` at D=4 | 33 passed |
| the same file at D=1 | 33 skipped, 0 passed |
| the 66-file affected surface, this tree | 23 failed / 1365 passed / 43 skipped / 11 errors |
| the same, pre-fix tip | 23 failed / 1352 passed / 43 skipped / 11 errors |

The FAILED/ERROR node-id sets are IDENTICAL between the last two rows
(`comm -23` and `comm -13` both empty) — the +13 is the new cells. All 23
failures and 11 errors are pre-existing on this box and unrelated.

---

## 8. What none of this can see

No P>1 leg anywhere, so nothing here says the collective transport is
correct — only that the serial one agrees with it at one geometry on one
deck. No GPU leg: the tier now refuses one by construction, and
`tests/test_slab_io_emulated_mesh.py::_mesh` asks for `jax.devices("cpu")`,
so on a GPU box the whole file skips. The suite's only runner is a
hand-typed `XLA_FLAGS=--xla_force_host_platform_device_count=4`; nothing
in-tree exports it, so the file is a no-op on the default leg (20→33
SKIPPED at D=1, 0 passed — no cell passes vacuously off the flag).
