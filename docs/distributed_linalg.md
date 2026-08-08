# Distributed linear algebra: why LORRAX ships its own layer

LORRAX runs multi-process SPMD JAX — one process per device, a two-dimensional
`('x','y')` mesh, `shard_map` and `NamedSharding` throughout, complex128 from end
to end. Three dense linear-algebra operations sit on its hot path, and all three
have to work on matrices whose size is set by the physics rather than by what
fits on a GPU:

| operation | where it is used | matrix |
|---|---|---|
| `eigh` | BSE band interpolation (`bandstructure/bse_setup.py`), coarse exchange tiles (`bse/vq_interp.py`) | (rank, rank), rank ≈ nspinor·n_μ, 10²–10⁴ |
| `cholesky` | the ISDF ζ-fit, charge channel (`isdf/core.py`) | (n_q, n_μ, n_μ), batched, Hermitian positive-definite |
| `solve_lu` | the ζ-fit's transverse channels; the W Dyson solve (`gw/w_isdf.py`) | (n_q, n_μ, n_μ), batched, Hermitian indefinite |

**JAX cannot do these**, and it is worth being precise about what "cannot"
means, because JAX is not short of linear algebra. XLA gives you excellent
*per-device* factorizations, including batched ones: hand it a stack of matrices
that each fit in one device's memory and it will do very well. What it does not
give you is a factorization of a **single matrix whose storage is sharded across
devices**. There is no distributed `eigh` in XLA. The moment one of the matrices
above stops fitting on one device, JAX's own answer is to replicate it — which is
not an answer, because replication is the thing you just ran out of memory doing.

The libraries that *do* solve sharded dense factorizations are the HPC ones:
ScaLAPACK, SLATE, cuSOLVERMp. They are C++ or Fortran, they want block-cyclic
layouts and MPI communicators rather than JAX arrays, and each carries its own
constraints on platform, mesh geometry and build. So `distrib_la` is a **layer
between LORRAX and the CPU/GPU C++ backends**: it takes a JAX array on a JAX
mesh, gets it into a form a vendor library will accept, makes the call, and hands
back a JAX array with the sharding and conventions LORRAX expects — for whichever
library is present and correct on the machine you are actually on.

## Why the routing is the hard part

You might expect the difficult code here to be the marshalling. It is not. The
difficult part is deciding *which* library to call, because a violated
constraint in this family does not usually produce an error. SLATE's host
eigensolver SIGSEGVs deterministically against MKL and LibSci LAPACK — every
size, every mesh, both settings of the eigenvector flag (bug L-2). cuSOLVERMp's
`syevd` on a rectangular mesh does not raise; it **deadlocks inside a
collective**, so the job hangs until the queue kills it. cuSOLVERMp refuses
`compute_evecs=False` with a bare status code at every size (bug L-3), and
SLATE's CUDA eigensolver takes down every rank of the job with `srun rc=139` at
n ≥ 4096 on a multi-rank mesh, while returning perfectly good answers at
n ≤ 2048 on the same mesh (bug L-4). Underneath all of them sits a subtler trap:
the two platform libraries LORRAX ships both declare `NEEDED libslate.so.2`
resolved out of *different* builds, and `ld.so` keys a loaded object by SONAME —
so whichever opens first decides which `blas::get_device_count()` the other one
calls, and the host build's is a compiled-in zero. Both loaders therefore open
CUDA before host, deliberately.

Before the service existed, the checks guarding all of this were spread over ten
`src/` modules in four packages, and they disagreed. The worst measured
consequence was a run that silently took a different route, finished with `rc=0`,
and printed a quasiparticle gap wrong by **−161 eV**. Every guard now runs in one
place, at one time, and the design follows from that.

## Resolve first, then call

`distrib_la` has two phases and they stay two.
**`plan(op, mesh, backend=…, n=…)` resolves.** It is eager: it `dlopen`s, probes
for the compiled handler, reads the process count, and runs the guard ladder —
vocabulary, platform, known-broken combinations, capability, process coverage,
mesh geometry, divisibility. **The resolved backend name is a promise**: every
guard passed, and the call cannot then fail for an availability reason.

**The returned `Plan` computes.** `plan(A)` takes one tile at `P('x','y')`;
`plan.batched(A_stack)` takes a stack at `P(None,'x','y')`. This phase is
trace-safe — no `dlopen`, no `device_put`, no process count inside — so it can
live in a `jit`. The split is not stylistic: platform and handler facts are
knowable before you have built an operand, while dtype, rank and extent are
trace-time facts, so a single-phase API would have to lie about when it checked
one or the other.

The other rule worth internalising is what happens when something is
unavailable. **An explicit request refuses**, naming the guard that failed and
the fix; only `auto` demotes, and it announces once, from the rank it happened
on. There is no silent fall-through anywhere, because a silent fall-through is
what the −161 eV was. Even the exception types are API — `ValueError` and
`RuntimeError`, each constructible from one string — because
`bandstructure/bse_setup.py` re-raises `type(exc)(_why)`, and a bespoke exception
class would escape that handler and delete a refusal message.

## Choosing a backend, and what "capability" means

Backend **choice** is deck data: `eigh_backend`, `distributed_cholesky`,
`distributed_lu`, `w_dyson_solver`, with `use_low_mem_eigh` naming the intent for
`eigh`. Backend **capability** is what the machine happens to have, probed once.
The environment sits only on the capability side: `LORRAX_FFI_SO` and
`LORRAX_FFI_HOST_SO` pin which `.so` to open. `distrib_la.resolve` reads no
environment at all, and adding an environment variable that selects a backend is
an explicit antipattern — a pin that cannot be honoured is a refusal.

The vocabularies are declarative data. `BACKEND_CHOICES` is importable with no
`.so` anywhere on the machine — a door promise, so a deck parser never needs the
FFI layer to read a deck — and the `eigh` parser reads its legal spellings from it
for that reason. The two had drifted once: the parser accepted only
`auto|off|cusolvermp|slate` while the resolver had grown `distributed` and
`scalapack`, so the low-memory eigh could not be requested through an input file
on CPU, the one platform that needs it. The `cholesky` and `solve_lu`
vocabularies are still hardcoded duplicates on the LORRAX side and can drift the
same way.

| op | accepted spellings | `auto` gives | `distributed` means |
|---|---|---|---|
| `eigh` | `auto off distributed cusolvermp slate scalapack` | `native` | cpu → ScaLAPACK, CUDA → cuSOLVERMp, ROCm → SLATE |
| `cholesky` | `auto off native2d cusolvermp slate` | `native` | *(no such spelling, deliberately)* |
| `solve_lu` | `auto off distributed cusolvermp scalapack` | `native` | cpu → ScaLAPACK, CUDA → cuSOLVERMp |

`distributed` exists for `eigh` and `solve_lu` because each has exactly one right
library per platform, so naming the library at the call site is redundant and
drifts; `cholesky` deliberately has no such spelling, because its CPU story is a
channel-policy ladder in the ζ-fit caller rather than one library.

`native` — the pure-JAX in-tree implementation — is the floor on every platform
and a first-class backend rather than a fallback of last resort. `native2d` is
the two-dimensional block-distributed tiled Cholesky, also pure JAX and also
everywhere, but a *different algorithm*: at n = 10⁴ on 128 processes the
replicated route costs 1.6 GB per device and this one 5 MB. `auto` never picks
it, because an algorithm whose cost model differs by three orders of magnitude in
**both** directions is the caller's decision, not a default. ROCm is a
**declared-untested** tier: the rows exist so the routing question has an answer,
but LORRAX builds no ROCm `.so`, and `Device.platform` reports `'gpu'` for both
vendors on the jaxes this tree runs, so a real ROCm mesh would land on the CUDA
row. Fixing that is one row of code and a machine.

## What the performance numbers actually say

They say **use `native`** — Perlmutter, one node, four A100s, jobid 56447670,
complex128, warm medians, one matrix per call:

| n | native (replicated) | cuSOLVERMp | SLATE | cuSOLVERMp / native |
|---|---|---|---|---|
| 64 | **0.00149 s** | 1.586 | 0.401 | 1064× |
| 256 | **0.00412 s** | 1.561 | 0.546 | 378× |
| 1024 | **0.02662 s** | 1.754 | 1.387 | 66× |
| 2048 | **0.07275 s** | 1.932 | 5.444 | 27× |
| 4096 | **0.39550 s** | 2.739 | **SIGSEGV** | 6.9× |

cuSOLVERMp on a four-process mesh costs a flat ~1.55 s **per matrix, independent
of size** — 64×64 and 1024×1024 cost the same, which is the signature of a fixed
collective and context charge rather than of arithmetic. Profiling closes the
question: `cusolverMpSyevd` itself is 99.998 % of the warm call, `plan()` plus
`resolve_backend` are 9.6 µs, the context-cache hit is 0.72 µs, and block size
does not move it. There is no edit in this tree that makes that number smaller.
The CPU story is the same without needing an extrapolation: ScaLAPACK `pzheevd`
never beat native replicated on one node, and the gap widens with n (4.0× at 64,
2.77× at 2048). `auto → native` is vindicated on both platforms at every size
anyone has measured. Do not change it.

SLATE looks like the reasonable compromise and is not: cheaper floor (0.40 s vs
1.59 s at n = 64), beats cuSOLVERMp up to n ≈ 1024, worse scaling, dies at 4096 —
it fails exactly where a distributed eigensolver becomes the thing you need. That
is why ELPA is a registered candidate: not built, not wired, nothing depends on
it, but the question has a written answer.

**The honest summary is that `distributed` is for capacity, not speed.** Every
row in that table fits on one device, which is precisely the regime where a
distributed library has no reason to win. Extrapolating the fitted exponents puts
break-even near n ≈ 1.9 × 10⁴ — read the *decade*, never the digits; the fit
window moves it between 1.4 × 10⁴ and 2.8 × 10⁴ — and that lands in the same
decade as the single-device capacity wall, where an n×n complex128 matrix plus
its eigenvector copy and workspace stops fitting in 40 GB, around n ≈ 2.7 × 10⁴.
Below the wall native wins by between 1064× and 6.9×; at and above it native
cannot run at all and nobody has measured what the alternatives cost. So an
explicit `distributed` or `cusolvermp` eigh below n = 16384 prints what that
route costs, once per mesh geometry on rank 0 — and then **runs exactly what was
asked for**. A cost notice is not a demote.

The largest gap, stated plainly so nobody re-derives it: every number here is one
node and four ranks, and the case `distributed` exists for is a mesh spanning
nodes, of which there is not one measurement. The rest — the capacity regime,
float64 rows, `compute_evecs=False`, a genuine CPU partition, non-square meshes —
are enumerated in [`services/distrib_la.md`](services/distrib_la.md).

## Traps

* **Do not compare eigenvectors across meshes.** A degenerate subspace has no
  canonical basis, so two meshes return different, equally correct columns.
  Compare eigenvalues or a gauge-invariant contraction; a test that diffs `Z`
  across meshes passes on silicon and fails on the first degenerate system.
* **Do not read inside a `FactorToken`.** `factor()` factors once and `solve()`
  back-solves many; the token carries a block-cyclic handle laid out for one
  process grid, and is deliberately not a JAX pytree so a `jit` boundary refuses
  it by name rather than tracing it. Feed it back verbatim.
* **Do not branch on `plan.backend` outside the package.** It is introspection —
  a banner, a test message. A caller that branches on it has re-implemented the
  resolver, and the two will drift.
* **Do not wrap a refusal in `try`/`except` at the call site.** Six probe calls
  in `isdf/core.py` exist *only* for their raise; wrapping any of them converts a
  loud refusal into a silent different-backend run.
* **Do not "fix" the SLATE SONAME collision by unsetting
  `CUDA_VISIBLE_DEVICES`.** It was measured; the failing leg had exactly one
  visible GPU. Load order is the fix.

[`docs/services/distrib_la.md`](services/distrib_la.md) is the contract: the API
table, every guard, the baselines, the antipatterns, and the procedure for adding
a backend. `docs/dev/linalg_ffi.md` is the internals reference for the C++ side.
For which library each routine reaches on each machine, see
[The FFI layer](architecture/ffi_layout.md).
