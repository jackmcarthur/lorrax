# The band-projection primitive (`common.contract_bands`)

`contract_bands_block_reshard` is the one owner of the congruence

$$
\mathrm{out}[e?,k,m,n] = \sum_{s,\mu}\sum_{s',\nu}
\overline{\psi_L}[k,m,s,\mu]\; O[e?,k,s,\mu,s',\nu]\; \psi_R[k,s',\nu,n]
$$

on the 2-D `('x','y')` mesh (`conj` is applied to `psi_left` inside). Its
invariant: no `(m, n, k)`, `(m, μ)` or `(μ, μ)` object is ever materialised on
one rank; the output leaves block-sharded `P(None, 'x', 'y')`, so downstream
coefficient multiplies stay local.

## API

```python
from common.contract_bands import contract_bands_block_reshard, bands_gemm_ffi_enabled

project = contract_bands_block_reshard(
    mesh_xy,
    channels="none",          # "none" | "split_reim"
    extra="none",             # "none" | "leading" | "minor"   (legacy only)
    axes=("x", "y"),          # (μ/m axis, ν/n axis); axes[1] must be the mesh's last axis
    layout="legacy",          # "legacy" | "face" | "axis"
    face_shape=None,          # (nk, nb_full, n_μ, nspinor), required for face/axis
    right_face_shape=None,    # rectangular operator: right endpoint's shape
    face_band_extent=None,
    row_block=None,           # face/axis: widest x block's local μ rows (xn·bx); default all
)
out = project(psi_left, O, psi_right)
# face/axis, an operator in x blocks on the slab pieces (every spin): one reduction per call
faces = project.prepare(psi_left, psi_right)
acc = None
for rows in project.row_blocks(n):        # common.contract_bands.face_row_blocks
    acc = project.accumulate(faces, O_block, rows=rows, acc=acc)
out = project.finish(acc)
```

`project` is safe to jit or trace into a larger kernel. The factory is a
collective: the legacy and face factories call
`common.collectives.warm_mesh_cliques` ([MPI collectives](mpi_collectives.md)).
Call it synchronously on every rank, outside any trace.

### Layouts: one projection, three mechanisms

| `layout` | operands | mechanism | collectives | consumers |
|---|---|---|---|---|
| `legacy` | ψ band axis replicated going in (`psi_xr`, `psi_yn`) | one `shard_map`: right GEMM → `psum_scatter('y')` → left GEMM → `psum_scatter('x')` | two, the large one on `'y'` | `bse.bse_ring_comm` (`extra="leading"`), `common.zeta_projection` |
| `face` | band-distributed ψ (`psi_nmu`, `psi_mun`, 1/P); square mesh | the operator never moves. `prepare`: transpose `ppermute` of each ψ tile, then an `all_to_all` over `'y'` into the ψ_l slab (all bands, `μ/P`). `accumulate`, per band chunk sized against one local operator tile: ψ_r chunk `all_gather('x')`, local `T = O·ψ_r`, `psum_scatter('y')` of T onto the slab's μ piece, slab contraction into a rank-local `(nb, nb)` partial. `finish`: `reduce_scatter_to_band_block` | per rank `16·nk·[nb·ns·(μ_l+μ_r)/p + nb² + 3·nb·ns·μ/P]` bytes; transients ≤ one operator tile | GW Σ (`gw.ppm_tau_kernel`, `cohsex_sigma`, `photon_sigma`, `mpa.sector_sigma`) on the band-distributed ψ carrier |
| `axis` | every band local, centroid split over one mesh axis | local slab contraction into the partial, then `reduce_scatter_to_band_block` | one, `nb²` per rank | axis-layout ψ carriers (none in production) |

Face and axis require `face_shape`, refuse any `extra` other than `"none"`
(call once per slice instead), and accept `channels ∈ {"none", "split_reim"}`.
Their `(nb, nb)` partial is `16·nk·nb²` bytes per rank per channel,
P-independent; a projection whose operator comes in spin blocks accumulates
every block into it and reduces once.
All three return `(nk, m, n)` at `P(None, ax_x, ax_y)`.

### Legacy operand layout

| operand | shape | spec |
|---|---|---|
| `psi_left` | `(nk, m, s, μ)` | `P(None, None, None, 'x')` |
| `O` | `(nk, s, μ, s', ν)` | `P(None, None, 'x', None, 'y')` |
| `O`, `extra="leading"` | `(E, nk, s, μ, s', ν)` | `P(None, None, None, 'x', None, 'y')` |
| `O`, `extra="minor"` | `(nk, s, μ, s', ν, E)` | `P(None, None, 'x', None, 'y', None)` |
| `psi_right` | `(nk, s', ν, n)` | `P(None, None, 'y', None)` |
| returns | `(nk, m, n)` (+E leading/minor) | `P(None, 'x', 'y')` (+`None`) |
| returns, `split_reim` | `(S_R, S_I)`, each `(nk, m, n)` | `P(None, 'x', 'y')` each |

`channels="none"` runs one chain at `O`'s dtype. `channels="split_reim"`
requires a complex `O`, splits it into `(Re O, Im O)` before projection and
runs each real channel on its own chain: the two-channel Σ plan for consumers
that weight the channels independently. The channel pair is the stacked axis,
so `split_reim` with `extra` refuses; stack `(Re O, Im O)` yourself as a real
`extra="leading"` operand if a further batch axis is needed.

### Refusals (all before any collective, fix named)

1. `mesh.axis_names[-1] != axes[1]`: inverted mesh. Build the mesh with the
   ν axis minor or pass `axes=(major, minor)`.
2. `split_reim` with `extra != "none"`, or with a real `O` (a real `O` is one
   channel: use `channels="none"`).
3. Operand rank or extent disagreeing with `O`, naming the axis.
4. Band extents `m`, `n` that are not authenticated padded carriers for
   `P(None, 'x', …)` / `P(…, 'y')` (`runtime.padding.authenticate_padded_axis`;
   pad each axis independently per [mesh-padded axes](../architecture/padding.md)).
5. Face/axis without `face_shape`, or with `extra != "none"`.
6. GEMM body dtype outside f64/f32/c128/c64, or a mismatched real/complex
   pair (a de-promotion bug upstream).

## Why the legacy chain is staged

```
right    = contract (s', ν_local) of O_local with ψ_right_local
right_rs = psum_scatter(right, 'y', scatter_dim=n)     # large payload
left     = contract (s, μ_local) of conj(ψ_left_local) with right_rs
out      = psum_scatter(left, 'x', scatter_dim=m)      # small payload
```

Each `psum_scatter` completes a μ or ν reduction and tiles a band axis in the
same collective. A single-stage form needs either the replicated `(m, n, k)`
result or a `(μ, μ)` gather. The body lives inside one `shard_map`, so the SPMD
partitioner cannot re-plan it into replicated intermediates.

## Encoded policies

* **Stacked payloads.** Every channel or `extra` slice rides one
  `psum_scatter` per mesh axis, stacked on the channel axis. Bit-exact:
  reduce-scatter sums elementwise over the same groups in the same order.
* **Large payload on the node-local axis.** The ν-side contraction runs first,
  so the large `(k, s, μ_loc, n)` partial reduce-scatters over `'y'`, whose
  replica groups are consecutive ranks on a process-ordered device layout; only
  the small final block crosses the strided `'x'` groups. This is why an
  inverted mesh refuses. A hand-permuted device mesh passes `axes` matching its
  own layout.
* **f64-split de-promotion.** XLA (CPU and GPU) lowers a mixed f64 × c128 dot
  by converting the real operand to c128 (a full-tile materialisation, about
  100× the temp bytes of the de-promoted form at production shape) and then
  runs a complex GEMM at twice the flops. Whenever a real operand meets a
  complex one in the large right GEMM, the module splits the complex operand
  into f64 parts, runs pure-f64 GEMMs and recombines with one `lax.complex`.
  Genuinely complex × complex chains are not split: that is slower.
* **Vendor-BLAS right GEMM on CPU.** On a CPU mesh the large right
  contraction routes through the host handler `lorrax_mklblas_gemm_batch`
  ([vendor GEMM service](vendor_gemm_service.md)); the small left dots stay on
  XLA. The dial is `LORRAX_BANDS_GEMM_FFI` ([gate contract](ffi_gate_contract.md)):
  required and on by default, a missing handler refuses at startup, `=0` is an
  announced debug escape. Two structural exclusions keep the XLA einsum, and
  only these: a non-CPU mesh (XLA:GPU's dot already calls
  cuBLAS; the handler is in the host symbol table only) and `extra="minor"`
  (the contracted axis is not reachable by a strided batched GEMM without a
  full-tile transpose). Read at factory time: consumers must key kernel caches
  on `bands_gemm_ffi_enabled()` (see `gw.ppm_tau_kernel`'s pipeline key).
* **No aliasing.** The GEMM handler declares no `input_output_aliases`; a
  `(BA, M, N)` output cannot alias a `(BA, M, K)` or `(BB, K, N)` operand.
* **`extra` order.** Both orders are first-class; `"leading"` is the default
  and the only one the GEMM body serves.

## Gating a change to this module

1. State the value-parity class: bit-exact (pure data movement: stacking,
   indexing) asserts byte equality; value-level (any reassociation: GEMM
   split/merge, contraction order, BLAS backend) gates at 1e-12 (1e-14 on small
   unit shapes) and claims nothing more.
2. HLO pins on the 4-emulated-device mesh, asserted on optimized text: zero
   rank ≥ 2 `convert(f64)→c128`, dot dtype/shape classes, exact collective
   count, dtype and payload shape. `tests/test_contract_bands.py` and
   `tests/test_projection_lgemm.py` are the reference pattern.
3. Collective tables on production dumps: reduce-scatter payloads unchanged,
   and no collective carries a full `(μ, μ)` tile. HLO and collective-table
   gates are valid only from a cache-cold compile (`ISDF_JAX_CACHE_DIR=""`,
   fresh dump directory).
4. A factory-time env read added here joins every consumer's kernel cache key.
5. For a `Hermitian O` congruence the output is Hermitian for any ψ; that
   check tests the machinery for free, and `common.zeta_projection` runs it.
