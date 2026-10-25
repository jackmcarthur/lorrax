# distrib_la

`import distrib_la as dla` Dense linear algebra on the `('x','y')` device mesh: Hermitian eigensolves, LU and Cholesky solves, planned GEMMs, the polar factor and the fixed-capacity subspace algebra of iterative eigensolvers.
{: .page-summary }

Every dense matrix operation in a LORRAX workflow goes through `distrib_la`. The QSGW quasiparticle Hamiltonian $H_\bk$, the H-transform's $f_H(\bq)$ eigensolves, the BSE Coulomb interpolation and the downfolded models are diagonalized by an eigh `plan`. The $W$ Dyson equation $\varepsilon_\bq W_\bq = v_\bq$ and the transverse ISDF ζ factor are solved by an LU `plan`. The Green's-function products $G_\bk = \sum_n \psi_{n\bk}\, w_n\, \psi_{n\bk}^\dagger$ and the Σ projections inside the time loop run through a `gemm_plan`, restricted to an active band window. The parallel-transport links of the dipole utility are polar factors $L = UV^\dagger$ of overlap matrices. The Davidson (NSCF) and Lanczos (BSE, exciton bands) solvers build their projected eigenproblems with `plan_subspace`. Each call names an operation and a mesh. The service then picks a library for the platform (cuSOLVERMp on CUDA, ScaLAPACK on CPU, ELPA on ROCm, or pure JAX), checks that it can run, and returns the result in the same face layout and conventions whichever library ran. A new stage that needs a dense factorization, a large GEMM or an iterative eigensolver should call this package rather than `jnp.linalg` on sharded operands.
{: .lead }

<div class="glance" markdown>

Import
:   `import distrib_la as dla` (top-level names only)

Depends on
:   JAX, NumPy, [`lxkit`](lxkit.md); the LORRAX native libraries for the FFI backends (optional)

Devices
:   CPU and CUDA tested; ROCm declared. Operands `float64` or `complex128`

Used by
:   GW (W Dyson solve, G and Σ GEMMs, QSGW eigh), BSE, H-transform, dipole links, NSCF, ISDF ζ factor

</div>

## Mesh and layouts {: #layouts }

Every call takes a `jax.sharding.Mesh` with axes `('x','y')` of extents $P_x \times P_y$, $P = P_xP_y$ devices. LORRAX builds square meshes, $P_x = P_y = p$. The FFI backends also need one JAX process per device, with process $xP_y + y$ on mesh cell $(x, y)$, because each library builds its own MPI or NCCL grid from the process ranks. The native (pure-JAX) routes have no such requirement.

| layout | shape | `PartitionSpec` | per-rank piece |
|---|---|---|---|
| face, one matrix | `(n, n)` | `P('x','y')` | `(n/P_x, n/P_y)` |
| face, a stack | `(B, n, n)` | `P(None,'x','y')` | `(B, n/P_x, n/P_y)` |
| batch | `(B_p, n, n)` | `P(('x','y'),None,None)` | `B_p/P` whole matrices |
| axis (GEMM operands) | `A (q, m, k)`, `B (q, k, n)` | `P(None,'x',None)`, `P(None,None,'y')` | rows of A, columns of B, complete `k` |
| row face (`contract_faces`) | `(b, m, K)` | `P(None,'x',None)` or `P(None,'y',None)` | rows on one axis, complete `K` |

$B_p = \lceil B/P\rceil P$ is the batch padded to the device count; pad rows are zero. Face extents must tile the mesh, `n % P_x == n % P_y == 0`. The service never pads a face, because padding changes the factor the caller gets: pad the matrix yourself (zero rows, or sentinel diagonal entries for an eigh) and slice the result.

## Two phases and the promise {: #contract }

Every planned entry point (`plan`, `gemm_plan`, `plan_polar_factor`, `plan_subspace`) runs in two phases.

1. **Eager resolution.** The planner opens the native library, probes the handler, reads `jax.process_count()` and checks the mesh geometry. Call it outside any `jit`, once, and hoist it out of k-point and frequency loops: the first FFI call also builds a BLACS or cuSOLVERMp context and compiles a module (1.4 to 2.7 s measured), which a plan built inside a loop pays every iteration.
2. **Trace-safe call.** The returned object can be called inside the caller's `jax.jit` or `lax.scan`. It checks only operand dtype, rank and extent.

**A returned backend is a promise.** When resolution returns a backend name, every guard has passed: vocabulary, platform, known-broken combinations, compiled handler, one process per device, mesh geometry and, when `n` is given, divisibility. The call cannot then fail for an availability or geometry reason. An explicit request that cannot be honoured raises at resolve time, naming the failed guard and the fix. Only `auto` may fall back to another backend, and it announces the fallback once on rank 0.

Refusals are `ValueError` (vocabulary, platform, coverage, geometry, divisibility) or `RuntimeError` (handler not compiled or library not loadable); misuse raises `TypeError` or `NotImplementedError`. Each is constructible from one string, so a caller may re-raise `type(exc)(context + str(exc))`, as `src/bandstructure/bse_setup.py` does.

## Dense factorizations: `plan` {: #plan }

```python
plan(op, mesh_xy, *, backend="auto", n=None, batched_route=None,
     budget_bytes=None, price_only=False) -> Plan
```

| argument | meaning |
|---|---|
| `op` | `"eigh"`, `"cholesky"` or `"solve_lu"` (`OPS`) |
| `mesh_xy` | the `('x','y')` mesh |
| `backend` | a name from `BACKEND_CHOICES[op]` ([backends](#backends)); `"auto"` by default |
| `n` | matrix extent, when known. Moves the divisibility check to resolve time. Independent of any operand, so a caller that will pad can ask first |
| `batched_route` | `"batch_reshard"` (default without a room) or `"auto"` ([batched routes](#batched-routes)). Leave unset when passing `budget_bytes` |
| `budget_bytes` | the caller's free device memory per rank beside its own live set, identical on every rank. With it, each eigh stack is routed by capacity ([eigh stacks with a room](#eigh-room)). An `int` or a `RoomExhausted` |
| `price_only` | build the plan for pricing only (workspace and stack-byte queries), without preparing to run |

The returned `Plan` has:

| member | operands | result |
|---|---|---|
| `Plan(A)` | one matrix at `P('x','y')` | `eigh`: `(w, Z)`; `cholesky`: the factor; `solve_lu`: `Plan(A, B) -> X` |
| `Plan.batched(A, *args, cut=None)` | a stack `(B, n, n)` at `P(None,'x','y')`, or an eigh stack already in batch layout | the same per matrix, stacked |
| `Plan.native_fn` | | a pure trace-safe closure of the math, native backends only, for use inside the caller's own `jit` |
| `Plan.decision(A, *args)` | arrays or `jax.ShapeDtypeStruct`s | the host-side `(backend, route, rounds)` that `batched` will trace for that signature; key compiled programs on it rather than on the room |
| `Plan.stack_route(shape, dtype)` | | the `StackRoute` an eigh plan with a room takes for one stack |
| `.backend`, `.is_native`, `.batched_route`, `.route_for(shape, dtype)`, `.describe()`, `.donates` | | resolved facts, for logs and reports |

**Conventions, identical on every backend.** Eigenvalues return ascending and replicated, `float64`. Eigenvectors return as columns, $AZ = Z\,\mathrm{diag}(w)$, in the input's face layout. `solve_lu` solves $AX = B$ with `B` of shape `(B, n, nrhs)` and returns `X` in `B`'s sharding.

**Donation is declared per operation** in `DONATES`: `eigh` donates nothing, `cholesky` operand 0, `solve_lu` operands 0 and 1. Pass a fresh copy of any operand you need afterwards. `src/isdf/core.py` copies its matrix before the transverse ζ inverse because a residual check reads it later:

```python
lu = distrib_la.plan("solve_lu", mesh_xy, backend="distributed", n=int(n),
                     batched_route="auto",
                     budget_bytes=shared_device_room("transverse zeta inverse"))
inverse = lu.batched(jax.jit(jnp.copy, out_shardings=face)(C_reg), eye)
```

The QSGW quasiparticle eigensolve in `src/gw/qsgw_density.py` pads the band extent to a multiple of the mesh with sentinel diagonal entries, solves the stack, and drops the pad by count:

```python
plan = distrib_la.plan("eigh", mesh, backend=distrib_la_backend, n=nb_pad,
                       batched_route=distrib_la_batched_route)
E, U = plan.batched(H_j)                    # H_j: (nk, nb_pad, nb_pad) at P(None,'x','y')
if plan.batch_in_sharding is not None:
    U = distrib_la.ensure_sharding(U, plan.batch_in_sharding)
```

**Native Cholesky and LU.** Under `backend="native"`, `Plan(A)` implements `eigh` only; single-matrix `cholesky` and `solve_lu` raise `NotImplementedError`, because the replicated native form is a policy of the calling stage. Reach them through `batched_route="batch_reshard"`, through `backend="native2d"` (Cholesky), or through an FFI backend.

### Result checks {: #result-checks }

The distributed libraries can return wrong eigenvectors with a success status, so no distributed result is returned unchecked.

- Exact-zero rows of an eigh operand are replaced by distinct negative diagonal sentinels and returned as exact zero eigenvalues with unit vectors, on every route.
- Every distributed eigh is checked against 8 fixed probe columns $X$: $\lVert(AZ - Z\,\mathrm{diag}(w))X\rVert/(\lVert A\rVert\lVert X\rVert)$ and $\lVert(Z^\dagger Z - I)X\rVert/\lVert X\rVert$ must be within `roundoff_tol(n)`. A failure is retried (unshifted at a smaller block, shifted by $s\,I$, then gathered on every rank when $n \le 1448$ complex), each retry noted on rank 0's stderr.
- A caller that keeps only $w > c\,\max(w)$ passes `cut=c` to `batched`, so only the kept columns are checked.
- Every distributed LU and Cholesky is checked on its solution through sketches taken before the library consumes the operands; there is no retry.
- A result that still fails is NaN-poisoned on every rank, and an eager call prints and raises `GATE distrib_la_result_check` naming the op, `n` and the call site.

`checked_program(fn, mesh, out_shardings, in_shardings=None, compiler_options=None, donate_argnums=())` jits a caller's program that contains checked solves in two phases: the first holds each solve's first attempt and returns a mesh-reduced failure flag; only when it is set does the full-retry program run. `checked_shapes(fn, *args)` is `jax.eval_shape` of the first phase. `native_eigh_jacobi` is the Jacobi local eigensolver used as a retry.

### Batched routes {: #batched-routes }

`Plan.batched_route` decides how a stack of `B` matrices runs (`BATCHED_ROUTES`):

| route | constant | what runs |
|---|---|---|
| (a) `scan` | `ROUTE_SCAN` | `lax.scan` over the backend's single-matrix call (`BATCHED_SCAN_UNROLL = 1`) |
| (b) `backend_batched` | `ROUTE_BACKEND_BATCHED` | the library's own stacked entry, one descriptor and workspace for the stack |
| (c) `batch_reshard` | `ROUTE_BATCH_RESHARD` | move the batch axis onto the mesh, solve whole matrices with the local JAX kernel, move the outputs back |

The caller chooses `batched_route` from `BATCHED_ROUTE_CHOICES = ("auto", "batch_reshard")`; `BATCHED_ROUTE_DEFAULT = "batch_reshard"`. `"auto"` takes (b) when the backend has a stacked entry and (a) otherwise. Route (c) is the default because for a matrix that fits one device a distributed library's cost is its fixed per-call overhead. Its exchanges are two single-axis `all_to_all`s each way inside one `shard_map`; nothing crosses the host. Each device holds $\lceil B/P\rceil$ whole $n\times n$ matrices plus outputs and solver workspace, so when one matrix does not fit one device, use `"auto"` with a distributed backend. The input file's `linalg` key ([GW keywords](../drivers/gw/keywords.md#linalg)) sets this for every stage: `local` gives `batch_reshard`, `distributed` gives `auto`.

### Eigh stacks with a room {: #eigh-room }

A caller holding a face stack passes its free memory per rank, `plan("eigh", mesh, n=n, backend=request, budget_bytes=room)`, and the service decides each stack (`StackRoute`): route (c) in as few slices as fit the room, otherwise the whole-mesh distributed backend. Every rank decides alike from shapes and the room alone. Under `backend="auto"` with no distributed backend available on the mesh, a stack that needs the whole mesh is refused by name (`GATE eigh_capacity_route`).

| name | purpose |
|---|---|
| `StackRoute` | `NamedTuple`: the route, matrices per rank, rounds and program bytes of one stack |
| `RoomExhausted(stage, priced_bytes, available_bytes)` | a room of 0 that the caller knows about; the stack takes the minimum-footprint whole-mesh route and says so once |
| `room_or_exhausted(room, *, stage, priced_bytes, available_bytes)` | `room` when positive, else a `RoomExhausted` |
| `room_class(cache, key, room, build)` | build once per class of rooms that decide alike, so a changed room does not recompile |
| `new_stack_routes()` | one line per stack decision since the last call, for the driver log |

### Resident operands: `batch_layout`, `local_batch` {: #batch-layout }

A factor used against many right-hand sides is placed in the batch layout once:

```python
F_b = dla.batch_layout(F_face, mesh)                       # once: (B_p, n, n), q-local
run = dla.local_batch(lambda F, Z: F @ Z, mesh, resident=(0,))
X = run(F_b, Z_face)                                       # per call: only Z moves
```

`batch_layout(a, mesh)` takes a face stack or a replicated array and returns `(B_p, ...)` at `P(('x','y'),None,...)`; rank $xP_y + y$ owns rows $[rB_p/P, (r+1)B_p/P)$. `local_batch(kernel, mesh, *, resident=(), out_layout="face", nbatch=None)` runs `kernel` on whole local matrices with one exchange each way for the non-resident operands; `out_layout="batch"` leaves outputs in batch layout. `is_batch_layout(a, mesh)` tests the layout. `ensure_sharding(x, sharding)` is the one reshard helper: a tracer gets a sharding constraint, an array already in place passes untouched, anything else is placed process-locally. `src/gw/tangential_directions.py` uses `batch_layout` / `is_batch_layout` to keep its factors resident.

## Backends {: #backends }

`distrib_la` links no vendor library. It opens the LORRAX native libraries at run time (`liblorrax_ffi.so` on CUDA, `liblorrax_ffi_host.so` on CPU, the ROCm library on ROCm). When none is present the package still imports, the native routes work, and every FFI backend is refused by name. The environment only grants capability: no backend is ever selected from an environment variable. The input file's `linalg = local | distributed` key and the caller's explicit `backend=` choose.

| op | `BACKEND_CHOICES[op]` | `auto` resolves to | `distributed` resolves to |
|---|---|---|---|
| `eigh` | `auto off distributed cusolvermp elpa scalapack one_device` | `native` | CPU `scalapack`; CUDA `cusolvermp`; ROCm `one_device` for $n \le 16384$, else `elpa` |
| `cholesky` | `auto off native2d cusolvermp` | `cusolvermp` on a CUDA mesh with $P_x, P_y \ge 2$ when compiled, else `native` | not offered |
| `solve_lu` | `auto off distributed cusolvermp scalapack one_device` | as `cholesky` (ROCm: `native`) | CPU `scalapack`; CUDA `cusolvermp`; ROCm `one_device` within the room, refused beyond it |

- `off` (alias `native`, the constant `NATIVE`) is the pure-JAX path, `jnp.linalg.eigh` or the caller's own Cholesky and LU.
- `native2d` is a pure-JAX 2-D block-distributed tiled Cholesky on every platform. At $n = 10^4$, $P = 128$ it uses 5 MB per device against 1.6 GB for the replicated route. `auto` never picks it.
- `one_device` moves each matrix whole onto one device (route (c)'s exchange), solves it there with the native kernel and returns the face layout. It keeps the FFI operand contract, so a `distributed` call site runs it unchanged. On CUDA, given a room, `distributed` resolves to `one_device` when one whole matrix's program fits the room, else to `cusolvermp`.
- `auto` eigh is always native: a q-batched local eigh solves $P$ matrices at once, while a distributed library solves one matrix $P$ ways and walks the batch serially. `distributed` is a capacity route, for a matrix too large for one device.

Geometry guards: cuSOLVERMp eigh and ELPA need a square mesh; cuSOLVERMp Cholesky and LU need $P_x \ge 2$ and $P_y \ge 2$; ScaLAPACK needs a square or 1-D mesh. cuSOLVERMp eigh with `compute_evecs=False` is refused (the library returns an error status at every $n$): always ask for eigenvectors.

### Resolution and introspection

| name | signature | purpose |
|---|---|---|
| `resolve_backend` | `(op, requested, mesh_xy, *, n=None, compute_evecs=True, size_route=True, room=None) -> str` | the raising probe; returns `"native"`, `"native2d"`, `"one_device"` or an FFI name |
| `list_backends` | `(op, mesh_xy) -> dict[str, str]` | every concrete backend mapped to `"available"` or `"unavailable: <guard>"`; never raises, for startup banners |
| `backend_module` | `(backend) -> module` | the implementation module of a resolved backend (none for `native`) |
| `BACKEND_CHOICES`, `EIGH_BACKENDS`, `CHOLESKY_BACKENDS`, `LU_BACKENDS`, `OPS`, `NATIVE` | | the vocabularies; importable with no native library, so an input-file parser needs no FFI |
| `mesh_key` | `(mesh) -> tuple` | hashable mesh identity: axes, extents, platform, device ids. Key any cache whose value does not retain the mesh on it |
| `mesh_platform`, `mesh_is_cpu` | `(mesh)` | `"cpu"`, `"CUDA"` or `"ROCM"`, from the device vendor; and the CPU predicate |
| `probe_target`, `has_target` | `(target_name, platform)` | `(usable, reason)` with the absent-versus-broken reason ([lxkit probes](lxkit.md#probe)); and the never-raising bool |
| `dial_key` | `() -> tuple` | the cache-key component of factory-time settings |

## Matrix products {: #gemm }

| entry | for | phases |
|---|---|---|
| `matmul` | an eager product of any shape | resolves on every call |
| `gemm_plan`, `local_gemm_plan` | one fixed shape called many times inside a `jit` or `lax.scan` (G construction, Σ projection) | eager plan, trace-safe call |
| `panel_matmul`, `batch_gram`, `contract_faces` | face products whose contraction axis must never be gathered whole on a rank | trace-safe, no provider |

### `matmul` {: #matmul }

```python
matmul(A, B, C=None, *, mesh, alpha=1, beta=0, transa="N", transb="N",
       backend="auto", batched_route="batch_reshard", budget_bytes=None)
```

Computes $D = \alpha\,\mathrm{op}(A)\,\mathrm{op}(B) + \beta C$ with $\mathrm{op} \in \{N, T, C\}$ (none, transpose, conjugate transpose). Rank-2 operands are at `P('x','y')`, rank-3 stacks at `P(None,'x','y')` with one shared leading batch; the output keeps the input rank and layout. `C` may be omitted only when `beta = 0`; real operands refuse complex `alpha` or `beta`. The default staged route exchanges whole matrices onto devices, runs one local GEMM and exchanges back, so each device holds $\lceil B/P\rceil$ whole A, B and D. `batched_route="auto"` runs `panel_matmul` instead. With `budget_bytes`, the staged route runs when it fits and `panel_matmul` otherwise. `MATMUL_BACKEND_CHOICES` is the accepted vocabulary; every name except `off` runs the same face GEMM, and `resolve_matmul_backend(requested, mesh, *, batched_route)` says which. The Galerkin ζ refit in `src/isdf/galerkin.py` binds one with its room and lets the service choose per call:

```python
room = shared_device_room("Galerkin refit rows GEMM")
mm = partial(face_matmul, mesh=mesh_xy, batched_route="auto", budget_bytes=room)
```

### `gemm_plan` {: #gemm-plan }

```python
gemm_plan(mesh, *, m, k, n, nq, dtype, backend="auto", alpha=1.0, beta=0.0,
          layout="face", reduction_axis=None, out_spec=None,
          enable_active_range=False, warmup=True) -> GemmPlan
```

Fixes one N,N shape, $D_q = \alpha A_q B_q + \beta C_q$ for $q < n_q$, with `A (nq, m, k)`, `B (nq, k, n)`, `C`, `D (nq, m, n)`. `nq` is usually the k-point count; a spinor axis is merged into `m`, `k` or `n`. `GemmPlan(A, B, C=None, *, out=None)` is trace-safe.

| `layout` | operands | contraction |
|---|---|---|
| `"face"` | all at `P(None,'x','y')`; `m % P_x`, `k % P_x`, `k % P_y`, `n % P_y` all zero | `panel_matmul` |
| `"axis"` | `A (q, m_X, k)` at `P(None,'x',None)`, `B (q, k, n_Y)` at `P(None,None,'y')`, `D` at `P(None,'x','y')` | a local GEMM, no collective (`local_gemm_plan`) |

- **N,N only.** A caller with a transposed operand stores it once in the complementary layout.
- `alpha` and `beta` are compile-time constants. `out=` donates an existing buffer for its storage and is legal only with `beta = 0`. Pass `C` or `out`, not both.
- `warmup=True` compiles and runs an axis plan once on dummy operands so a timed loop pays nothing; `warmup=False` defers compilation to the first call, for a plan used inside one outer `jit`.
- `backend="off"` is refused: a planned GEMM exists to avoid whole operands on every device.
- For `layout="axis"`, `out_spec=P(None,'x',None)` or `P(None,None,'y')` keeps one centroid axis sharded, and `reduction_axis="x"` or `"y"` contracts a tiled axis with a local GEMM and a `psum_scatter`. `local_gemm_plan(mesh, *, m, k, n, nq, dtype, alpha, beta, reduction_axis, out_spec, enable_active_range, warmup)` builds this layout directly.

The COHSEX Green's function in `src/gw/cohsex_sigma.py` plans one face GEMM per run, contracting bands for each k point:

```python
g_plan = gemm_plan(mesh_xy, m=n_rmu * ns, k=nb_full, n=n_rmu * ns, nq=nk,
                   dtype=jnp.complex128, layout=layout)
```

### Active ranges {: #active-ranges }

A Green's function restricted to an energy window contracts only a band interval $[lo, hi)$, but its operands keep their full band extent $K$ so compiled shapes never change. With `enable_active_range=True`:

```python
D = plan.active_range(A, B, lo, hi, C=None, *, out=None, weights=None, conj_b=False)
```

computes $D_q = \alpha\,(A_q\,\mathrm{diag}(w_q))_{:,lo:hi}\,B_{q,lo:hi,:} + \beta C_q$ without allocating anything of the interval's size. `lo` and `hi` are replicated integer scalars or `(nq,)` arrays; `weights` has shape `(nq, K)`; `conj_b=True` contracts $\bar B$ without forming it. $0 \le lo \le hi \le K$ is required: invalid Python integers raise eagerly; invalid traced bounds give an all-NaN result rather than clamping. An empty interval applies the `beta` semantics without reading A or B. `plan.prepare_active_range(lo, hi)` captures bounds known before tracing as compile-time metadata and returns `fn(A, B, C=None, *, out=None, weights=None)`, with no per-call device-to-host transfer; keep one per recurring interval. `active_weights_on_load(platform)` says whether the platform's kernel applies the weights as it loads `A` (no weighted copy) or forms `A diag(w)` first. `src/gw/greens_function_kernel.py` builds G this way:

```python
G_flat = (gemm(A, B) if band_range is None
          else gemm.active_range(A, B, *band_range, weights=phases, conj_b=conj_on_load))
```

and `src/gw/mpa/sigma.py` plans its Σ window GEMMs with `enable_active_range=True, warmup=False` inside the window executable.

### Bounded face products {: #panel-matmul }

`panel_matmul(a, b, *, mesh, panel_bytes, bounds=None, weights=None, partner=False, transa="N", transb="N", compiler_options=None)` forms $AB$ for `a (q, m, k)` and `b (q, k, n)` at `P(None,'x','y')` on a square $p\times p$ mesh as a batched 2-D SUMMA inside one `shard_map`. Each panel gathers $p\,w \le k/p$ columns, with $w$ the largest width keeping the two live panels under `panel_bytes`; no rank ever holds a band-complete row or column panel. `bounds (q, 2)` names each row's live contraction interval, `weights (q, k)` scales `a`'s columns on the way into the gather, and `partner=True` also returns $\bar A\,\mathrm{diag}(w)\,\bar B$ from the same exchange (a Green's function and its time-reversal partner). `panel_bytes` bounds the two live panels only; the input and output faces are the caller's to budget. `src/gw/epsmat_export.py` calls it directly.

`batch_gram(b, weights, bounds, *, mesh, nbatch, right=None, partner=False)` forms $W_q = b_q\,\mathrm{diag}(w_q)\,c_q^\dagger$ over each row's interval when the factors already sit in batch layout, and returns $W$ on the face. `contract_faces(b_X, b_Y, weights, start, stop, *, mesh, return_transpose=False)` forms $(b_X\,\mathrm{diag}(w))\,b_Y^\dagger$ for row faces `(b, m, K)` at `P(None,'x',None)` and `P(None,'y',None)` with replicated weights `(b, K)`; the output `(b, m, m)` lands at `P(None,'x','y')` with no collective.

## Polar factor and spectral directions {: #polar }

The parallel-transport link between neighbouring k points is the unitary closest to the overlap matrix $A$, $L = UV^\dagger$ for $A = U\Sigma V^\dagger$. `distrib_la` computes it from one Hermitian eigensolve of extent $2n$ on the dilation

$$
H = \begin{pmatrix} 0 & A \\ A^\dagger & 0 \end{pmatrix}, \qquad
H \begin{pmatrix} u_i \\ \pm v_i \end{pmatrix} = \pm\sigma_i \begin{pmatrix} u_i \\ \pm v_i \end{pmatrix},
$$

then forms $L = \sum_{\sigma_i > \mathrm{rcond}\,\sigma_{\max}} u_i v_i^\dagger$. It never forms $A^\dagger A$, which would square the condition number and lose the small singular values the link-quality check reads.

```python
plan_polar_factor(mesh, *, n, backend="distributed", rcond=None,
                  batched_route="auto") -> PolarPlan
polar_factor(A, mesh, *, backend="distributed", rcond=None) -> (L, s)
```

- **Input:** a square `float64`/`complex128` matrix at `P('x','y')` with `n` divisible by both mesh axes, or for `PolarPlan.batched(A)` a stack `(B, n, n)` at `P(None,'x','y')`. Zero-pad a non-divisible `n` and slice the leading block of `L`; the pad's null directions fall below the cutoff.
- **Output:** `L` with A's shape and sharding; `s`, length `n`, real, descending, replicated. `rcond` is relative to $\sigma_{\max}$ (default $n\varepsilon$), so a rank-deficient $A$ gives its partial isometry.
- **Cost:** one $2n$ eigensolve and one $n^3$ GEMM, $O(n^2/P)$ memory per process.
- `polar_factor` caches eager plans and refuses a tracer; inside a `jit`, call a `PolarPlan` built beforehand.

The dipole utility's link stage (`src/file_io/parallel_transport.py`) builds one plan per run and solves a stack of overlaps per flush:

```python
polar_plan = plan_polar_factor(mesh, n=nb_padded, backend=str(eigh_backend),
                               rcond=float(rcond), batched_route=str(batched_route))
link, values = polar_plan.batched(stack_faces(*overlaps))
```

Three related functions extract leading spectral subspaces from a Hermitian or rectangular-normal operator, all closing a whole multiplet at the cut (relative gap $\le$ `multiplet_tol`) so a degenerate subspace is never split. `W` is a square face, a face stack or a batch-layout stack; `eigh_plan` is the caller's resolved plan of extent `m`, and its route decides local or distributed.

| function | returns |
|---|---|
| `right_singular_vectors(W, tau, *, eigh_plan, column_extent, multiplet_tol=1e-6, real_rows=None, max_rank=None)` | right singular directions with $\sigma/\sigma_{\max} > \tau$, from the eigenvectors of $W^\dagger W$; refuses $\tau^2 < 10^4 m\varepsilon$ |
| `leading_eigenvectors(W, r, *, eigh_plan, column_extent, multiplet_tol=1e-6, real_rows=None, rcond=None)` | the leading `r` eigenvectors, optionally bounded by `rcond * max|λ|`; refuses a non-Hermitian `W` |
| `leading_eigenvectors_above(perp, top, cutoff, *, real, eigh_plan, column_extent, multiplet_tol=1e-6, real_rows=None)` | leading eigenvectors of each row of `perp` above `cutoff**2` times `top`'s largest eigenvalue |
| `retain_leading_eigenvectors(Q, values, r, *, mesh, column_extent, multiplet_tol=1e-6, real_rows=None)` | a narrower cut of a previous result, with no new eigensolve |

Only the $O(m)$ spectra cross the host, broadcast from one process as bit patterns so every rank makes the same cut. `src/gw/tangential_directions.py` uses these for the tangential screening directions.

## Local (one-device) solvers {: #local }

Replicated or device-local matrices that need no mesh go through these drop-in replacements for `jnp.linalg`, which route to the platform's best local kernel and fall back to JAX elsewhere.

| function | equivalent | notes |
|---|---|---|
| `eigh_local(a, UPLO=None, symmetrize_input=True)` | `jnp.linalg.eigh` | ascending eigenvalues, eigenvector columns, batched over leading axes |
| `eigh_route(dtype, leg_key=None)` | | the route this platform uses for `dtype` (`None`: `jnp.linalg.eigh` unchanged); `COMPLEX_EIGH_ROUTE` holds the per-platform rows |
| `solve_local(a, b, native=None)` | `jnp.linalg.solve` | `a [..., n, n]`, `b [..., n, nrhs]` or `[..., n]`; a singular matrix gives non-finite `X` |
| `lu_solve_local(lu, piv, b, native=None)` | `jax.scipy.linalg.lu_solve((lu, piv), b)` | on JAX's own factors |
| `tri_solve_local(l, b, *, transpose=False, conjugate=False, native=None)` | `lax.linalg.triangular_solve` on the lower triangle | |
| `local_lu_routes(a_shape, b_shape, dtype, leg_key=None) -> bool` | | whether the platform route takes the system; `LOCAL_LU_ROUTE` holds the rows |

The Lanczos and Davidson solvers diagonalize their small tridiagonal or projected matrices this way, for example `evals_T, vecs_T = eigh_local(T)` in `src/solvers/lanczos.py`.

## Active subspace for iterative eigensolvers {: #subspace }

Davidson and Lanczos grow a basis at run time, while `jit` fixes every shape at compile time. A subspace plan reconciles the two: buffers of fixed capacity $M$, of which only an active window, named by traced integers, is ever computed on. Changing `start`, `count` or `active` never recompiles; changing $M$, the block width $b$ or the vector shape does.

```python
plan_subspace(*, capacity, n_eig, vector_sharding=None, max_block_size=None,
              native_collectives=False)
plan_orthogonalization(*, capacity, max_block_size, vector_sharding=None,
                       native_collectives=False)
plan_local_subspace(*, capacity, n_eig, max_block_size=None) -> LocalSubspacePlan
```

`plan_subspace` checks $1 \le$ `n_eig` $\le$ `capacity` $< 2^{31}$, that JAX x64 is enabled, and that `vector_sharding` (a `NamedSharding` whose first, row axis is unsharded and whose vector axes cover every mesh axis of extent above 1) is legal. It then resolves one provider from the platform (`SUBSPACE_ROUTE`, read by `subspace_route(leg_key=None)`): native cuBLAS/cuSolverDn kernels on CUDA and ROCm, NumPy LAPACK callbacks on CPU, and a `shard_map` wrapper whenever the vectors span more than one device. All arrays are `complex128`.

| method | computes |
|---|---|
| `store(V, HV, P, HP, start, count)` | writes `count` new rows at `start`, in place |
| `project(V, HV, active, H, start, count)` | the new rows and columns of $H_{ij} = \langle v_i\vert H\vert v_j\rangle$ |
| `gram(V, P, active, start=)` | $C = \bar V_{[start, start+active)} P^{\mathsf T}$ |
| `eigh(H, active)` | the lowest `n_eig` eigenpairs of the leading `active × active` block; missing values are $+\infty$ |
| `reconstruct(V, HV, C, active, template, columns=, compute_image=, start=)` | $X = C^{\mathsf T}V$ and $HX = C^{\mathsf T}HV$ |
| `orthogonalize(V, P, active, start=)` | classical Gram–Schmidt applied twice (CGS2) of a block against the selected rows |
| `qr(P)`, `normalize(P, rank)` | reduced QR (TSQR when distributed); whitening of a correction block |

`plan.workspace_specs` lists the XLA-owned scratch the plan adds, for memory admission before lowering. `plan_orthogonalization` returns the CGS2 callable alone; `ortho.backend` names the implementation. `plan_local_subspace` is the single-device native plan of a GPU process and refuses on CPU (`GATE subspace_native`) or when a handler is missing. The Davidson NSCF solver (`src/solvers/davidson_fixed.py`) plans two:

```python
subspace = plan_subspace(capacity=capacity, n_eig=n_eig, vector_sharding=vector_sharding)
normalizer = plan_subspace(capacity=n_eig, n_eig=n_eig, vector_sharding=vector_sharding)
```

## Face-pinned block glue {: #block-glue }

Eager slicing, concatenation and $a + a^\dagger$ of face-sharded arrays come out replicated, so a `(b, R, R)` block would cost $16R^2$ bytes per rank instead of $16R^2/P$. These helpers run the same elementwise program with the operand's face as the output sharding, bitwise equal to the eager form. Traced or unsharded operands take the plain function.

| call | result |
|---|---|
| `hermitian_part(a)` | $(a + a^\dagger)/2$ for `(b, R, R)` |
| `hermitian_block(block, off, corner)` | $\begin{pmatrix}\text{block} & \text{off}^\dagger \\ \text{off} & \text{corner}\end{pmatrix}$, `(b, R+r, R+r)` |
| `join_columns(a, b)` | column panels `(b, n, R)` and `(b, n, r)` concatenated |
| `diagonal_like(values, like)` | $\mathrm{diag}(\text{values})$ in `like`'s dtype and face |
| `on_face(fn, out, *operands, **static)` | a module-level `fn` evaluated with outputs placed on `out` |
| `face_sharding(array)` | the `NamedSharding` of a concrete array sharded on every axis, else `None` |

## Workspace and memory queries {: #workspace }

The native libraries allocate workspace outside XLA's memory planner, so a stage that sizes itself asks first. None of these allocate; all accept `float64` or `complex128` only. Use them to build a `lxkit.memory.Price` ([lxkit memory model](lxkit.md#memory)).

| function | returns (bytes per rank) |
|---|---|
| `workspace_bytes_per_rank(plan, op, shapes, dtype)` | device workspace of a resolved CUDA `Plan` or `GemmPlan`; 0 off CUDA. Query collectively on the real mesh |
| `matmul_workspace_bytes_per_rank(mesh, shapes, dtype, *, backend="auto", batched_route="batch_reshard")` | the same for a `matmul` route |
| `eigh_stack_bytes(plan, shape, dtype, phase="all")` | what a batched eigh of the face stack `shape` adds beside its operand (`phase="first"`: first attempt only) |
| `eigh_stack_proposal(mesh, shape, dtype)` | closed-form route-(c) first attempt, before the run's program is known |
| `lu_workspace_bytes(mesh, n, dtype)` | measured cuSOLVERMp LU workspace for an $n\times n$ system with $n$ right-hand sides |
| `lu_workspace_query(mesh, n, nrhs, dtype, cap)` | the library's own size query for the next LU solve |
| `route_bytes(mesh, a_shape, b_shape, dtype, transa="N", transb="N", route=..., backend="off")` | what `matmul` allocates beside its operands on `route` |
| `staged_fits(mesh, a_shape, b_shape, dtype, transa, transb, budget_bytes) -> bool` | whether the staged `matmul` route fits |

## Round-off tolerance {: #roundoff }

`roundoff_tol(n, dtype=complex128)` is `ROUNDOFF_MARGIN * n * eps(dtype)` with `ROUNDOFF_MARGIN = 64`. It is the relative bar of every Hermiticity and residual check in the service and of several in LORRAX (`max|A - A^H| / max|A|`). Use it in new checks instead of a fixed `1e-12`, which is below the round-off of a GEMM-built Hermitian matrix once $n\varepsilon$ exceeds it (about $n = 4500$ in `complex128`). `src/psp/run_dense_h.py` checks the dense DFT Hamiltonian with it.

## Environment variables {: #env }

`distrib_la` has no capability dials: the backend is chosen by argument and by the input file, never by the environment. These variables locate libraries or add diagnostics.

| variable | effect |
|---|---|
| `LORRAX_FFI_SO`, `LORRAX_FFI_HOST_SO`, `LORRAX_FFI_ROCM_SO` | absolute path of the CUDA, CPU or ROCm native library. A pinned path that is missing or of another ABI refuses instead of falling through to a search |
| `LORRAX_DEBUG_PRINT=1` | print the ScaLAPACK workspace per call |

Inside ScaLAPACK calls the MKL thread team is capped at four threads, and the previous setting is restored afterwards. A library with no ABI stamp loads with a one-time `LEGACY-UNSEALED` warning, unless a build manifest is found beside it, in which case it refuses.

## Invariants and gotchas {: #gotchas }

- **Import top-level names only.** `from distrib_la.<submodule> import ...` from outside the package fails the layering lint.
- **Hoist plans.** Resolve outside loops and outside `jit`; call inside.
- **Pad faces yourself.** Every face extent must divide both mesh axes. Pad an eigh operand with distinct sentinel diagonal entries below the spectrum and drop them by count, never by value.
- **Do not catch refusals.** An explicit backend that cannot be honoured must stop the run; never wrap a refusal in `try/except` or branch on `plan.backend`.
- **Donated operands are gone.** `cholesky` consumes operand 0, `solve_lu` operands 0 and 1. Copy what you will read later.
- **Eigenvectors are gauge-dependent.** Never compare them across meshes or backends. Compare eigenvalues or invariant contractions ($Z\,\mathrm{diag}(w)Z^\dagger$, projectors, $|Z^\dagger Z'|$). Distributed and local results agree to $\kappa\varepsilon$, not bitwise.
- **Pass `cut=` when you discard small eigenvalues.** Otherwise the check runs on every column, including near-null ones you never use.
- **NaN means refused.** A result that fails every retry is NaN-poisoned on every rank; keep a finiteness check downstream.
- **No Python loop over the batch axis.** Use `Plan.batched`; a loop compiles and launches `B` separate calls.
- **`gemm_plan` is N,N with compile-time `alpha`, `beta`.** `out=` requires `beta = 0`. Traced active-range bounds out of range give NaN, not an error.
- **Collective calls.** Plans with FFI backends, `plan_subspace(native_collectives=True)` and every subspace operation must be issued by every rank in the same order with the same `start` and `count`.
- **Cache keys.** Use `mesh_key(mesh)`, not `id(mesh)`, for any cache whose value does not keep the mesh alive.
- **Precision.** Operands are `float64` or `complex128`; the subspace plans require `jax_enable_x64`.

## Where LORRAX uses it {: #consumers }

| file | purpose |
|---|---|
| `src/gw/w_isdf.py` | `plan("solve_lu")` for the $W$ Dyson solve in the ISDF basis |
| `src/gw/qsgw_density.py` | `plan("eigh")` for the QSGW quasiparticle Hamiltonian $H_\bk$ |
| `src/gw/cohsex_sigma.py`, `src/gw/plane_wave_pipeline.py`, `src/gw/wavefunction_bundle.py` | `gemm_plan` for Green's-function and band-projection GEMMs |
| `src/gw/greens_function_kernel.py` | `GemmPlan.active_range` for windowed $G$ construction |
| `src/gw/mpa/sigma.py`, `src/gw/mpa/sector_sigma.py` | `gemm_plan` with active ranges for Σ over pole windows |
| `src/gw/response_bank.py` | Dyson programs keyed on `Plan.decision`; `roundoff_tol` for the bare-response check |
| `src/gw/tangential_execution.py`, `tangential_sectors.py`, `tangential_capacity.py` | eigh stacks with a room, `eigh_stack_bytes`, `checked_program` |
| `src/gw/tangential_directions.py`, `tangential_reduction.py` | `leading_eigenvectors`, `right_singular_vectors` and face-pinned block glue for tangential screening |
| `src/isdf/galerkin.py`, `src/isdf/zeta_mubatch.py`, `src/gw/tangential_head.py` | `matmul` for face GEMMs routed by room |
| `src/gw/downfold.py` | `plan("eigh")` of the downfolded Hamiltonian stack |
| `src/gw/epsmat_export.py` | `panel_matmul`, `hermitian_part` and eigh for $\varepsilon^{-1}$ export |
| `src/isdf/core.py` | `gemm_plan` in the ζ fit; `plan("solve_lu")` for the transverse ζ factor |
| `src/bandstructure/htransform.py` | `plan("eigh")` for the fine-grid $f_H$ eigensolves |
| `src/bse/vq_interp.py`, `src/bse/bsemat_export.py` | eigh of the BSE Coulomb interpolation; `plan_polar_factor` for gauge fixing on export |
| `src/bse/bse_lanczos.py`, `src/bse/exciton_bands.py` | `plan_subspace` for Lanczos and thick-restart Lanczos |
| `src/solvers/davidson_fixed.py`, `src/solvers/lanczos.py` | `plan_subspace`, `eigh_local` in the iterative eigensolvers |
| `src/file_io/parallel_transport.py` | `plan_polar_factor` for the parallel-transport links |
| `src/psp/run_dense_h.py` | dense DFT Hamiltonian eigh and its `roundoff_tol` check |
| `src/common/zeta_projection.py` | `eigh_local` of the ζ Gram matrix |
