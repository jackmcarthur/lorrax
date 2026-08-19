# JAX CPU collectives on MPI (`impl=mpi`) and the LORRAX MPIwrapper

*How LORRAX runs `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`: the common JAX
contract, the different Intel-MPI and Cray-MPICH thread-level mechanisms, and
the two machine launch recipes.*

## STATUS: thread-main override superseded; thread level is site-specific

**A locally-patched MPI shim is not a portable dependency this project should
spread to every machine.** The `MPI_Is_thread_main` override is no longer
needed anywhere; ordinary Python in this repo replaces it. Frontera retains a
small patch only to upgrade Intel MPI's thread-level request. Perlmutter uses
unmodified upstream MPIwrapper and obtains the required thread level from
Cray MPICH's supported async-progress control.

### What replaced it: a main-thread mesh-clique warm-up

`common.collectives.warm_mesh_cliques(mesh)` creates one MPI communicator per
mesh axis, plus one over all axes, **from the Python main thread** at
mesh-construction time — three 8-byte `psum`s, ~150 ms once per process,
independent of `N_mu` / `N_k` / `N_q` / `P`.

It works because jaxlib's guard fires only on communicator **creation**, and
`xla::cpu::AcquireCommunicator` caches communicators in a process-global clique
map keyed *only* by the participating-device set: it takes the map lock and
calls `CreateCommunicator` only on a **miss**. Each warm-up `psum` is small
enough (one 8-byte buffer, <= 8 thunks) that XLA takes
`ThunkExecutor::ExecuteSequential` and runs the thunk inline on the caller, so
`MPI_Is_thread_main` is true and the split succeeds. Every later collective —
including the ones a pool worker issues inside the BSE Lanczos jit — is a cache
hit and never reaches the guard.

The caching is **per-clique**, which is the whole point and is the thing nobody
had tested. Warming the world clique alone **fails**; `x` alone fails; `x + y`
without the world fails; only `x + y + world` passes. An earlier helper warmed
the world clique only — exactly the failing cell — which is why the old
"world-collective-first contract" looked falsified. Warm-up always mattered; it
was warming the wrong device sets.

On the real 785c BSE deck at P=4 with the driver unmodified and the wrapper
gate **unset**: no warm-up gives 8 refusals and death; warm-up gives rc=0 and
eigenvalues `[1.30537661 1.3504201 1.42411254 1.50449023]`, character-identical
to the wrapper-override run and to the gloo reference. A four-way agreement.

It is also **safer** than the override. `MPI_Comm_split` is collective over
`MPI_COMM_WORLD`. The override lets XLA call it from arbitrary pool workers;
`AcquireCommunicator` serialises creation *within* a process but nothing
serialises it *across* ranks, so two cliques becoming ready in different orders
on different ranks is a latent deadlock. Creating them all from one thread in a
program-defined order before any jit runs removes that exposure.

And it changes no compiled HLO, so it cannot move the collective table or the
allocation table.

### What the wrapper is still for

MPIwrapper is always required as the ABI adapter that JAX's bundled
MPItrampoline loads.  It is not the MPI implementation.  The thread-level
policy carried by that adapter is machine-specific:

| machine | production composition |
|---|---|
| **Frontera / Intel MPI** | The locally patched adapter upgrades XLA's request to `MPI_THREAD_MULTIPLE`.  Its optional `MPI_Is_thread_main` override is **SUPERSEDED** by `warm_mesh_cliques`; production leaves `LORRAX_MPI_FORCE_THREAD_MAIN` unset. |
| **Perlmutter / Cray MPICH** | The adapter is exact, unmodified upstream MPIwrapper. HPE's public `MPICH_ASYNC_PROGRESS=1` setting makes Cray MPICH grant MULTIPLE despite XLA requesting FUNNELED. No local MPIwrapper patch is used. |

MULTIPLE remains required for the full application: XLA's collective
operations can run on pool threads while native parallel-HDF5 or distributed
linalg is using MPI.  The remaining upstream exit is jaxlib requesting
`MPI_THREAD_MULTIPLE` itself (and checking `provided`), and/or relaxing its
`MPI_Is_thread_main` communicator-creation guard.

Full analysis of all four routes, including the falsified ones:
`wk_REL/jax_threadmain_alternatives.md`.

## Why not gloo

The two CPU collectives backends jaxlib 0.9.1 offers are `gloo` (the default)
and `mpi`. LORRAX ran on gloo through 2026-07-27. Three measured results moved
it off:

1. **gloo's `reduce-scatter` silently corrupts results.** Under
   `JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo`, `jax.lax.psum_scatter` over a 2-D
   mesh intermittently returns wrong data with no error, no warning and a zero
   exit code — ~5% of executions, ~80% of process lifetimes, always output
   segment 0, at a magnitude of order the correct answer (a ~24% error against
   a 2.8e-14 association floor). It reproduces with no LORRAX imports at all.
   `impl=mpi` on the identical program is clean in 504/504 executions with a
   gloo positive control corrupting 4 of 4 process lifetimes in the same
   allocations, and a negative control proving the MPI reduce-scatter is
   genuinely on the critical path.
2. **The performance case for gloo has evaporated.** `impl=mpi` is 1.18x
   end-to-end at P=16 against gloo *on its ib0 pin*, with the collective-bound
   stages 1.4-8.2x. On identical payloads (1.12 GB all-reduce / 2.24 GB
   all-gather / 1.12 GB reduce-scatter) mpi takes 0.83 / 1.05 / 0.63 s where
   gloo takes 14.99 / 31.11 / 11.98 s.
3. **gloo in this jaxlib has no non-TCP transport**, so that gap is structural
   rather than a tuning matter. `GLOO_SOCKET_IFNAME` is inert — the string
   appears in no `.so` in jaxlib 0.9.1.

## What blocks `impl=mpi`, and why it is not an MPI thread level

jaxlib's `xla::cpu::MpiCollectives::CreateCommunicators()` opens with

```c
int flag; MPI_Is_thread_main(&flag);
if (!flag) return absl::UnknownError(
    "MPI: Communicator requested from a thread that is not the one MPI was "
    "initialized from. Multiple threads/devices per process are not yet "
    "supported.");
```

and only then calls `MPI_Comm_split(MPI_COMM_WORLD, color, key, &comm)`.

Three properties of that guard, all confirmed by disassembling
`CreateCommunicators` in `jaxlib/libjax_common.so`:

* **It is a `MPI_Is_thread_main` test, not a thread-LEVEL test.**
  `MPI_Is_thread_main` is false on any non-initialising thread at *every*
  level, including `MPI_THREAD_MULTIPLE`. No `MPI_THREAD_*` setting satisfies
  it. A THREAD_MULTIPLE-patched wrapper alone does not help — that was fixing
  the wrong layer.
* **It fires only on communicator CREATION**, once per clique key.
  `MpiCommunicator::AllReduce/ReduceScatter/AllGather/...` carry no such check.
* **The discriminator is which XLA:CPU execution path the program takes**, not
  the shape of the jaxpr. `ThunkExecutor::ExecuteSequential` runs thunks inline
  on the caller (main) thread, so small graphs pass; the parallel
  `ThunkExecutor::Execute<ReadyQueue>` path dispatches thunks to intra-op pool
  workers, so real graphs fail. A clean-room probe of "collectives inside
  `lax.scan` inside `shard_map` inside one jit" — the shape earlier docs named
  as the discriminator — **passed** under `impl=mpi`, and so did a bare
  subgroup `psum` with no warm-up. There is also no config knob: the complete
  `set_xla_cpu_*` DebugOptions list in this jaxlib contains nothing that forces
  sequential thunk execution, and `jax_cpu_enable_async_dispatch=0` is not the
  lever.

Consequently the *ordering* story ("a world collective must come first") is
wrong — but **warm-up is still the answer**, just per-clique rather than
ordered: create each mesh-axis communicator, and the world one, from the main
thread before any real jit runs. See STATUS above. The earlier probes that
appeared to falsify warm-up altogether were void: every cell in them was small
enough to take the sequential executor and so passed with no warm-up at all.

`MPI_Is_thread_main` in `libjax_common.so` is an MPItrampoline stub
(`jmpq *MPIABI_Is_thread_main`) resolved at `dlopen` from the MPIwrapper named
by `MPITRAMPOLINE_LIB` — a library **we build**. On Frontera the adapter also
carries the historical override; on Perlmutter the adapter is upstream and
the guard is satisfied only by the in-tree clique warm-up.

## The wrapper

*(Interim — see STATUS at the top.)*

The Frontera adapter is built by `config/frontera/build_mpiwrapper.sh` from upstream MPIwrapper
v2.11.1 (`eschnett/MPIwrapper`, commit `966f4231…`) plus exactly one patch,
`config/frontera/mpiwrapper/lorrax_thread.patch`. Upstream is external source
under its own licence and is fetched, not vendored; only the patch is in the
repo. The patch adds two overrides and nothing else.

### Perlmutter: unmodified adapter plus Cray controls

`config/perlmutter/build_mpiwrapper.sh` builds the same pinned MPIwrapper
commit with no patch and refuses a dirty upstream checkout.  The build uses
Cray `cc`/`CC`/`ftn`, strips the GPU and Darshan modules, checks the required
MPItrampoline ABI exports, rejects CUDA-GTL/Darshan dependencies, and runs the
one-MPI dynamic-closure gate.

Two Cray launch controls are load-bearing and are set by
`config/perlmutter/cpu_mpi_env.sh`:

* `LD_PRELOAD=/opt/cray/pe/lib64/libpmi.so.0` must be in force before Python. Without it,
  `jax.distributed.initialize()` starts coordination threads and the later MPI
  initialization segfaults in `_pmi_spawn_init -> PMI2_Init`. Preloading
  `libpmi2.so.0` does not fix it. The tracked prelude uses Cray's stable,
  unversioned absolute symlink and verifies that it resolves under
  `/opt/cray/pe`, so `LD_LIBRARY_PATH` cannot shadow it with a foreign PMI.
* `MPICH_ASYNC_PROGRESS=1` makes Cray MPICH promote XLA's explicit
  FUNNELED request to `MPI_THREAD_MULTIPLE`. The default-thread CVARs are
  inert because XLA made an explicit request. Async progress creates a
  progress thread; reserve a hardware thread per rank and benchmark its cost.

Measured on allocation 57261316: one- and two-node P=4 clique/allreduce/
reduce-scatter probes pass with `MPI_Query_thread=MULTIPLE`; a frozen two-node
P=4 GN-PPM run is exact in 2,484/2,484 reference cells. The same tracked
launcher also passes the collective proof at P=16/four nodes on a 4×4 mesh,
with 16 logical CPU affinity slots per rank (256 across the step, not 256
physical cores or full-node occupancy). The tracked P=4 proof is step
`lx-Xg1-205224-2180345-6630`, P=16 is
`lx-Xg1-210631-2288631-7393`, and GN-PPM is
`lx-Xg1-204038-2097590-2094`. This certifies the collective/runtime layer at
P=16 and the full GW path at P=4; it is not a large-physics-run performance
certificate.

### Frontera override 1 — THREAD_MULTIPLE upgrade (always on)

`MPI_Init` / `MPI_Init_thread` forward to
`PMPI_Init_thread(..., MPI_THREAD_MULTIPLE, ...)`. Requests are upgraded,
never downgraded; the init **order** is unchanged.

XLA's `MpiCollectives::Init()` calls
`MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided)` and never reads
`provided`. With h5py/mpi4py collective MPI-IO on the Python main thread and
XLA's collectives on an executor thread, a FUNNELED grant is undefined
behaviour, and it was measured as such: **4 failures in 14 runs (~29%)** at
P=16 x 8 nodes — 3 segfaults plus 1 hang, provider-independent, every one at
the ζ-write / `V_q` boundary, with backtraces showing two threads of one rank
simultaneously inside `MPID_Progress_wait`. Upgrading the grant to MULTIPLE
makes Intel MPI's global lock serialize them. P=4 single-node never failed
(shm netmod).

Rejected alternatives on the Intel-MPI route, for the record:
`I_MPI_THREAD_LEVEL_DEFAULT=MULTIPLE` and
`MPIR_CVAR_DEFAULT_THREAD_LEVEL=multiple` are **inert** (MPICH grants the
explicit request, not the default); an `LD_PRELOAD` interposer does not resolve
through the trampoline's `dlopen`ed scope; `LORRAX_MPI_INIT_FIRST=mpi4py` does
move the granted level but then hangs the trampoline on a pre-initialized MPI
and is a documented DO-NOT-USE.

### Frontera override 2 — `MPI_Is_thread_main`, gated on `LORRAX_MPI_FORCE_THREAD_MAIN`

> **SUPERSEDED — do not enable in production.** `warm_mesh_cliques` (STATUS,
> above) achieves the same thing in-repo, with no patched dependency, and
> without exposing the cross-rank `MPI_Comm_split` ordering hazard described
> below. This section is retained because the code path still exists as a
> fallback and as the positive control in the gates.

With the gate set, the wrapper reports "yes, thread-main" to every caller,
which removes XLA's refusal and lets `MPI_Comm_split` run on the executor
thread. **Default OFF**: unset, the wrapper's behaviour is byte-for-byte
override 1 alone.

Legality rests on two facts, both of which must stay true:

* `MPI_Comm_split` from a pool worker is legal MPI **only** because override 1
  has already made the grant `MPI_THREAD_MULTIPLE`. The two overrides are a
  pair; the gate must never be used with an unpatched wrapper.
* XLA creates cliques in a deterministic order that is identical on every
  rank, so all ranks split in the same order.

Blast radius is exactly XLA's CPU collectives. mpi4py, h5py and the FFI host
`.so` all link Intel `libmpi.so.12` directly and never route through
MPItrampoline, so they never see either override.

### Building the adapters

```bash
export LORRAX_ROOT=/path/to/lorrax
config/frontera/build_mpiwrapper.sh --fresh      # Intel-MPI patched adapter
config/perlmutter/build_mpiwrapper.sh --fresh    # Cray-MPICH upstream adapter
```

The Frontera script compiles Fortran bindings outside its py312 container,
applies the tracked patch, and verifies its overrides **in the machine code**
rather than in the source: it disassembles `MPIABI_Init_thread` and asserts the `required`
argument is hard-set to 3 (`MPI_THREAD_MULTIPLE`), and disassembles
`MPIABI_Is_thread_main` and asserts it reads the gate and still falls through
to `PMPI_Is_thread_main` when unset. A wrapper that silently grants FUNNELED
looks and loads exactly like a good one, so a source-level check is not enough.

Set `LORRAX_MPIWRAPPER_REFERENCE_SO` on Frontera to compare `.text` against a known-good
build (whole-file equality is not achievable — the build CWD is embedded in
`.note.gnu.build-id` and, on TACC, `.note.xalt.info`).

## Launch recipes

### Perlmutter

Build on a CPU compute node and source the prelude **inside every `lx` rank
shell before Python**.  This is the canonical minimal launcher; it pins the
JAX 0.9 module and source checkout and checks the live JAX generation before
the driver starts:

```bash
export LORRAX_CHECKOUT=/path/to/lorrax
export CPU_JAX_VENV=/path/to/jax-0.9.1-venv
export LX_BASE_MODULE=lorrax_A
export LORRAX_CPUS_PER_TASK=16
export PYTHONPATH="$LORRAX_CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}"
lx run --cpu -N 2 -n 4 -- bash -c '
  set -euo pipefail
  export PATH="$CPU_JAX_VENV/bin:$PATH"
  export LORRAX_CPU_SKIP_GPU_PLUGINS=1
  export OMP_NUM_THREADS=14
  . "$LORRAX_CHECKOUT/config/perlmutter/cpu_mpi_env.sh"
  python3 -u "$LORRAX_CHECKOUT/tools/require_jax09.py"
  python3 -u -m gw.gw_jax -i gw.in
'
```

`LORRAX_CHECKOUT` is the source pin consumed by `lx` when the caller is in a
data directory; setting only `PYTHONPATH` does not replace it. The explicit
rank-shell `PATH` is the interpreter pin. The tracked metadata-only checker
then refuses a JAX/JAXLIB series other than 0.9 without importing JAX.

The prelude validates the adapter's pinned source, MPI ABI and SHA256 manifest;
rejects stale Frontera overlays and conflicting MPI/PMI preloads; forces CPU,
one JAX device per rank and `impl=mpi`; sets PMI/async-progress controls;
disables Cray GPU support; and unsets `LORRAX_MPI_FORCE_THREAD_MAIN`. It does
not choose an OpenMP team size. The example requests room for progress, but
that is not a certified progress/XLA-thread affinity policy: XLA workers do
not obey `OMP_NUM_THREADS`, so production affinity still needs a thread-census
and async-on/off performance measurement.

Release names include the adapter content hash and the builder/prelude/site
recipe hashes. Rebuilding identical adapter bytes after changing the recipe
therefore cannot silently reuse an older provenance manifest.

### Frontera

**Executable form: `config/frontera/templates/gw_dev.sbatch`** — the
certified launch block, vendored; the fragments below are its anatomy.

```bash
# --- collectives ----------------------------------------------------------
export JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi
export MPITRAMPOLINE_LIB=$WORK/lorrax_mpiwrapper/install/lib64/libmpiwrapper.so
export LORRAX_MPI_FINALIZE_FIX=skip_atexit
# LORRAX_MPI_FORCE_THREAD_MAIN is deliberately NOT set: the in-repo
# warm_mesh_cliques() replaces it.  Setting it would only mask a missing
# warm-up call site.
PYTHONPATH=$WORK/lorrax_env_mpi_overlay/site:$PYTHONPATH   # the sitecustomize
# The overlay (mpi4py 4.1.2 + parallel h5py 3.16.0 + sitecustomize.py) is
# now buildable from the repo: config/frontera/build_mpi_overlay.sh.

# --- Intel-MPI transport --------------------------------------------------
# The whole block (PMI2 glue, fabrics, provider case-block, UCX
# setdefaults, I_MPI_DEBUG=4 banner) is config/frontera/mpi_transport_env.sh
# — source it instead of hand-copying exports.  The PMI2 lib it points at
# is staged once by config/frontera/stage_host_pmi.sh.
. $LORRAX_ROOT/config/frontera/mpi_transport_env.sh
export LORRAX_MPI_PROVIDER=auto   # auto => FI_PROVIDER unset => mlx (default)

# --- container binds ------------------------------------------------------
# NEVER bind anything under /dev.  RDMA userspace staging:
#   --bind /usr/lib64:/hostlibs:ro,/usr/lib64/libibverbs,/etc/libibverbs.d
# plus the staged-symlink block that APPENDS to LD_LIBRARY_PATH (a bare
# /hostlibs on the path shadows container glibc).
```

The Frontera composition has these load-bearing pieces:

| variable | omit it and |
|---|---|
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` | you are on gloo, i.e. on the corrupting reduce-scatter |
| `MPITRAMPOLINE_LIB` | MPItrampoline refuses loudly at startup |
| a `warm_mesh_cliques()` call on every mesh | BSE (and any grouped clique first created inside a jit) dies on every rank with the communicator refusal. This is a code call site, not an env var — `common.collectives.warm_mesh_cliques`, invoked from the mesh factories and from `contract_bands_block_reshard` |
| `LORRAX_MPI_FINALIZE_FIX=skip_atexit` + the overlay `sitecustomize` | **every run exits rc=1 after succeeding** — jax's atexit `collectives.Finalize` runs, then post-atexit C++ teardown makes one more MPI call and Intel MPI reports "Attempting to use an MPI routine after finalizing MPICH" |

`MPITRAMPOLINE_LIB` is deliberately **not** auto-defaulted from `src/`: it is a
machine fact naming a build artifact outside the repo, the hazardous-vs-good
choice must stay visible in the harness, and MPItrampoline already refuses
loudly when it is missing.

## Interaction with the rest of the stack

* `common.collectives.warm_mesh_cliques(mesh)` must be called on every mesh
  that will carry a grouped collective, synchronously on every rank. It is a
  no-op off `impl=mpi`, in single-process runs, and on an already-warmed mesh.
* `runtime.announce_cpu_collectives()`, called from `bootstrap()`, prints the
  resolved implementation once from rank 0 and warns if a multi-process CPU
  run has landed on gloo. It is the only place in `src/` that reads the
  collectives implementation at all, and it changes nothing but the log.
  The MPI transport itself is selected by the site launch recipe: Intel MPI's
  `FI_PROVIDER`/`LORRAX_MPI_PROVIDER` on Frontera and Cray MPICH/Slingshot on
  Perlmutter.
* `ffi/cpp/phdf5/context.cc` and `ffi/cpp/slate/context.cc` only call
  `MPI_Init_thread(MULTIPLE)` when nothing initialized MPI first, so they
  coexist with XLA's init by construction. The phdf5 open warns when the
  granted level is below MULTIPLE — that warning firing means the wrapper is
  not on the path and the ~29% race regime is back.
