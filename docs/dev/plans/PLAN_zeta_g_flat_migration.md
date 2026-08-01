# Plan: zeta_q → G-flat on-disk format + IBZ solve + ρ-cutoff support

Working doc for the rchunk ↔ G_flat migration the user signed off on after
the orbit-aware kmeans test cleared.  Captures every code reference and
design rule from the planning conversation so the agent can resume after
context compaction.

## Order

| Phase | What | Risk | Bit-id smoke target |
|---|---|---|---|
| A | rchunk ↔ G_flat transform pair (phase-after-slice + accumulator) | low | MoS2 3×3 xonly |
| B | IBZ-only triangular solve via `q_irr_full_idx` gather | low | same |
| C | G-flat on-disk zeta layout + bare_coulomb_cutoff up to ρ-cutoff | medium | same |
| D | transverse V_q^{ij} IBZ unfold (R·V·Rᵀ + centroid double-permute) | medium | requires bispinor smoke |

Reference allocation: ``SLURM_JOBID=52841861``; runs/MoS2/00_mos2_3x3_cohsex/02_lorrax_xonly
with ``cohsex_orbit.in`` + ``centroids_frac_641.txt`` (orbit-closed,
exercises the IBZ writer + V_q unfold paths).

## Phase A — rchunk ↔ G_flat transform pair

Goal: a clean mirrored pair of cached transforms in
``common/wfn_transforms.py``, sharing one internal FFT helper in
``common/fft_helpers.py``.  No user-supplied workspace argument
(pure-JAX → not really lowerable to in-place anyway); instead, the
G_flat partial-sum accumulator is donated and is the only persistent
buffer.  Phase-after-slice in ``to_rchunk`` (mathematically equivalent;
keeps phase work tight on the slab).  Mirror in reverse for the new
``accumulate_rchunk_to_gflat``.

### A1. ``wfn_transforms.to_rchunk``: phase-after-slice

Current (``wfn_transforms.py:243``) order:
```
G_flat → _box_kernel → ifftn → apply_bloch_phase(WHOLE BOX) → reshape → slice
```
New order:
```
G_flat → _box_kernel → ifftn → reshape → dynamic_slice_in_dim → apply_bloch_phase_on_slice
```

Where ``apply_bloch_phase_on_slice`` is a flat-r variant of
``apply_bloch_phase`` (``wfn_transforms.py:301``) — same separable
``exp(±2πi k·r)`` math, but indexed at the flat-r positions
``r0..r0+r_len`` rather than the full box.

Concretely: given flat r-index ``r_flat`` in ``[r0, r0+r_len)``, decode
``(rx, ry, rz) = (r_flat // (ny·nz), (r_flat // nz) % ny, r_flat % nz)``
on the fly, evaluate ``px[k, rx] · py[k, ry] · pz[k, rz]``.  Each per-axis
1D phase ``px``, ``py``, ``pz`` is built once at jit-trace; the per-r-cell
product is the only thing computed per call.  Cost: ``n_k × r_len``
complex multiplies, vs current ``n_k × nx · ny · nz`` (~``r_len ×
nx·ny·nz/r_len`` smaller).  No correctness change.

Keep the existing ``_RCHUNK_KERNEL_CACHE`` shape-keying.

### A2. ``wfn_transforms.accumulate_rchunk_to_gflat``

New mirror transform: r-chunk slab → G_flat partial contribution,
**accumulating** into a donated buffer.

```
rchunk → apply_bloch_phase_on_slice(sign=-1) → zero-pad → reshape →
fftn → flat sphere-gather → ADD into donated gflat_acc
```

Signature (proposed):
```python
def accumulate_rchunk_to_gflat(
    rchunk: jax.Array,                # (n_q, n_mu_local, r_len) c128
    gflat_acc: jax.Array,             # (n_q, n_mu_local, n_G_sph) c128, donated
    *,
    fft_shape: tuple[int, int, int],  # (nx, ny, nz)
    r0: int,                          # static
    r_len: int,                       # static = rchunk.shape[-1]
    sphere_idx: np.ndarray | None,    # static flat-G indices to gather
    qvec_frac: jax.Array | None,      # (n_q, 3) for exp(-2πi q·r) ; None skips
    chunk_count: int = 1,             # internal FFT batch chunking
) -> jax.Array:                       # updated gflat_acc
```

Internal pipeline:
1. ``apply_bloch_phase_on_slice(rchunk, qvec_frac, fft_shape, r0,
   r_len, sign=-1)`` — only if ``qvec_frac is not None``.
2. ``zero-pad`` rchunk into a flat (n_q, n_mu_local, n_rtot) buffer
   via ``jnp.zeros + lax.dynamic_update_slice`` (this is the one big
   live tensor; minimised by the chunked FFT below).
3. ``reshape`` to (n_q, n_mu_local, nx, ny, nz).
4. ``local_3d_fft_chunked`` — see A3 — runs the actual FFT over the
   batch axis.
5. ``flatten`` spatial back to n_rtot; ``jnp.take(box_flat,
   sphere_idx, axis=-1)`` for the G-sphere subset.
6. ``gflat_acc + contribution`` — under jit, donation makes this an
   in-place add.

Note: this is "G-flat partial sum on the n_G_sph subset", not "full FFT
box stored".  The sphere gather happens INSIDE the jit so the
non-sphere cells are never written to host memory.

### A3. ``fft_helpers.local_3d_fft_chunked``

Shared internal helper.  Single source of truth for "local 3D FFT over
a flattened batch with optional fixed chunking".  Used by both
``to_rchunk`` (IFFT side) and ``accumulate_rchunk_to_gflat`` (FFT side).

Signature:
```python
def make_local_3d_fft_chunked(
    *, fft_shape: tuple[int, int, int],
    direction: Literal['fwd', 'inv'],
    norm: str = 'backward',
    chunk_count: int = 1,             # static
) -> Callable[[jax.Array], jax.Array]:
    """Returns a function that runs an n-D batched 3D FFT.

    Fast path: chunk_count == 1 → direct jnp.fft.fftn / ifftn.
    Chunked path: chunk_count > 1 → lax.scan over leading batch axis
    in ``chunk_count`` evenly-sized chunks.

    The batch axes are everything before the trailing (nx, ny, nz) of
    the FFT; chunking happens only along batch dims, never spatial.
    """
```

Rules (from chat):
- ``chunk_count`` is static (compile-time argument).
- ``chunk_count == 1`` has a hard fast path (no scan, no extra body
  closure).
- Chunk only over local batch axes (never spatial).
- Use ``lax.scan`` for chunked path so XLA sees the loop.
- Use ``with_sharding_constraint`` to keep input sharding stable per
  chunk (Pallas/refs are excluded by user; pure JAX only).

### A4. Phase A tests

Write ``tests/test_wfn_transforms_rchunk_gflat.py`` covering:
1. **Round-trip**: ``g_flat → to_rchunk(g_flat)`` then
   ``accumulate_rchunk_to_gflat(rchunk, 0)`` is approximately the
   identity (modulo cell-edge effects we set up the math to absorb).
2. **Phase-after-slice equivalence**: new ``to_rchunk`` matches the
   old ``to_rchunk`` to float-roundoff on random input.
3. **Accumulator donation**: confirm device buffer is reused (via
   ``jax.tree_util.tree_map`` + buffer-id check).
4. **chunk_count >1**: results match chunk_count=1 to float-roundoff
   on the same input.

Smoke target after Phase A: MoS2 3×3 ``cohsex_orbit.in``; eqp0.dat
bit-equivalent ±1 ULP × 5 × 9 baseline.

## Phase B — IBZ-only triangular solve

Goal: gather IBZ rows of ``L_q`` and ``Z_q`` right before the solve;
solve only those; everything downstream iterates IBZ q only.  The C / Z
construction (FFT-based convolution) stays symmetry-agnostic.

### B1. Gather IBZ rows

In ``common/isdf_fitting.py`` at the solve seam (search for
``triangular_solve``; current line is somewhere in fit_zeta_to_h5
around line 1500ish).  Insert:
```python
q_irr_full_idx = sym_maps.find_irreducible_qpoints()[3]    # (n_q_ibz,) int32
L_q_ibz = L_q[q_irr_full_idx]                              # gather along q-axis
Z_q_ibz = Z_q[q_irr_full_idx]
zeta_q_ibz = jax.scipy.linalg.solve_triangular(L_q_ibz, Z_q_ibz, lower=True)
```

The full-BZ ``L_q`` Cholesky stays as-is (chat: "fine to decompose
L_q for the full BZ, not a bottleneck").  Only the trsolve changes
shape.

### B2. Downstream iteration over IBZ only

After solve, the rchunk write loop already does the right thing if
``n_q_disk = n_q_ibz`` — the writer slices to ``q_irr_full_idx`` at
isdf_fitting.py:~1996.  Confirm that block matches the new pre-solve
gather (we now do the gather earlier; the writer slice becomes a
no-op identity).

### B3. Phase B tests

- Existing ``test_zeta_reader.py`` / ``test_zeta_loader.py`` still
  pass.
- Smoke (MoS2 3×3 orbit): bit-id eqp0.dat ±1 ULP.

## Phase C — G-flat on-disk zeta layout + ρ-cutoff bare Coulomb

Goal: replace the r-space-on-disk format with the G-flat partial-sum
written via Phase A's accumulator.  Two independent sub-tasks: (C1)
write G-flat; (C2) raise ``bare_coulomb_cutoff`` past ``ecutwfc``.

### C1. G-flat on-disk format

#### C1.1 isdf_header schema

Add ``zeta_layout`` to ``file_io/isdf_header.py`` ``IsdfHeader``
dataclass:
```python
zeta_layout: Literal['r_space', 'G_flat'] = 'r_space'
```
- ``'r_space'`` for legacy files (default; ``read_isdf_header``
  treats missing field as 'r_space').
- ``'G_flat'`` for new writes.
- Writer also emits ``isdf_header/n_G_sph`` (int, size of the G-flat
  trailing axis), ``isdf_header/sphere_idx`` (int32 flat-G indices),
  and ``isdf_header/bare_coulomb_cutoff_Ry`` (the cutoff used to
  build the sphere).

#### C1.2 Writer change (``common/isdf_fitting.py:fit_zeta_to_h5``)

Replace the r-space chunked write with an accumulator-based one:
```python
gflat_acc = jnp.zeros((n_q_ibz, n_rmu_padded_local, n_G_sph),
                     dtype=jnp.complex128)
# Sharding: same μ-axis sharding as the existing r-space chunked
# write so the disk write goes through the same SlabIO collective.
for r_chunk in r_chunks:
    zeta_rchunk = solve_for_rchunk(...)         # existing solve-rchunk math
    gflat_acc = accumulate_rchunk_to_gflat(
        rchunk=zeta_rchunk, gflat_acc=gflat_acc,
        fft_shape=meta.fft_grid, r0=r_chunk.start, r_len=r_chunk.size,
        sphere_idx=sphere_idx, qvec_frac=q_irr_kgrid_frac,
    )
slab_io.write_slab('zeta_q_G', gflat_acc, ...)
```
The on-disk dataset name changes from ``zeta_q`` (r-space) to
``zeta_q_G`` (G-flat).  Reader auto-detects via ``zeta_layout``.

Memory at write time: only ``gflat_acc`` is persistent
(``n_q_ibz · n_mu_local · n_G_sph · 16`` bytes per rank).  The
``zeta_rchunk`` buffer lives for one r-chunk's lifetime.  The full
r-space ``zeta_q`` tensor is never materialised.

#### C1.3 Reader change (``file_io/zeta_loader.py``)

``ZetaLoader.load(layout='G_flat')`` already does an r-space read
followed by an FFT (the ``_do_disk_to_G`` path at
``zeta_reader.py:301``).  When the file's ``isdf_header.zeta_layout
== 'G_flat'``, that FFT becomes a NO-OP: the on-disk dataset IS the
G-flat data.  Add a branch:
```python
if self._zeta_layout == 'G_flat':
    return self._read_g_flat(...)        # direct read from zeta_q_G
else:
    return self._read_r_space_then_fft(...)   # existing path
```

#### C1.4 Tests

- Add ``test_isdf_header_zeta_layout_roundtrip``: round-trip the new
  field; check legacy file (no field) defaults to 'r_space'.
- Add ``test_zeta_loader_g_flat_disk``: write a synthetic G-flat
  zeta_q.h5 + read via ZetaLoader; bit-identical to a direct h5py
  fetch.
- Smoke (MoS2 3×3 orbit) with the new writer: V_q kernel reads the
  G-flat zeta and produces the same V_q + eqp0.dat (±1 ULP).

### C2. Bare-Coulomb cutoff up to ρ-cutoff (4·ecutwfc)

Current limitation (mentioned in chat): ``bare_coulomb_cutoff`` can
only go up to ``ecutwfc`` because it uses the G-vector indices from
WFN.h5 (which are built at ecutwfc).  Raising to 4·ecutwfc (ρ-cutoff)
requires regenerating the G-vector list at the larger cutoff.

Helper to use: ``src/psp/gvec_utils.py:build_master_gvec_list(crystal)``
— builds the full G-vector list for a crystal at the rho cutoff.

#### C2.1 Wire build_master_gvec_list

In ``compute_vcoul`` or wherever the bare Coulomb sphere is built
(search ``bare_coulomb_cutoff``), use ``build_master_gvec_list``
when the requested cutoff exceeds the WFN G-list extent.  Sphere
indices for the new G-list go into the G-flat zeta header
(``isdf_header/sphere_idx``).

#### C2.2 Smoke

MoS2 3×3 with ``bare_coulomb_cutoff = 30`` (4·ecutwfc=30 Ry) should
now succeed; verify V_q values are different from the ecutwfc-bound
case in a physically reasonable direction (larger v(G=0) head term,
more G-vectors contributing).

## Phase D — Transverse V_q^{ij} IBZ unfold

Goal: extend the existing ``_unfold_v_q_ibz_to_full`` to handle the
3 × 3 polarization block (transverse vertex), with the R·V·Rᵀ rotation
on the polarization indices ON TOP OF the centroid double-permute.

### D1. Math

Per the chat:
```
V^{ij}_{Sq}(π_S μ, π_S ν) = Σ_{ab} R^{ia}(S) R^{jb}(S) V^{ab}_q(μ, ν)
```
where ``R(S)`` is the 3×3 spatial rotation (Cartesian).  Centroid
permutation ``π_S`` is the same as the scalar case.

### D2. Implementation

Add ``_unfold_v_q_ij_ibz_to_full`` (sibling of
``_unfold_v_q_ibz_to_full`` at ``gw/v_q_tile.py:1454``):
```python
def _unfold_v_q_ij_ibz_to_full(
    V_q_ij_ibz: jax.Array,     # (n_q_ibz, 3, 3, n_rmu, n_rmu)
    *,
    full_to_irr_idx: np.ndarray,
    full_to_irr_sym: np.ndarray,
    sym_perm: np.ndarray,       # centroid perm
    R_cart: np.ndarray,         # (n_sym, 3, 3) cartesian rotations
    mesh_xy: Mesh,
) -> jax.Array:                  # (n_q_full, 3, 3, n_rmu, n_rmu)
```

Body: gather along q-axis (as scalar), apply centroid double-permute
(as scalar), then ``einsum('sia,sjb,sqabmn->sqijmn', R_cart_q,
R_cart_q, V_perm)``.  All take_along_axis calls use
``mode='promise_in_bounds'`` (the Phase 49b7f84 XLA fix).

The R_cart_q for each full-BZ q is just ``R_cart[full_to_irr_sym[q]]``.

### D3. Tests

- Unit test in ``tests/test_v_q_transverse_unfold.py``: synthetic
  ``V_q_ij_ibz`` + identity sym → output equals input (identity
  R V Rᵀ).  Non-trivial sym (90° rotation) → expected polarization
  mixing matches by hand.
- Smoke with bispinor MoS2: requires ``cfg.bispinor=True`` + transverse
  centroids.  Out of scope until earlier phases are bit-identical;
  document as future test.

## Key file/line references (post-compaction recovery)

| File | Lines | What |
|---|---|---|
| ``src/common/wfn_transforms.py`` | 243 | ``to_rchunk`` (refactor to phase-after-slice in A1) |
| ``src/common/wfn_transforms.py`` | 301 | ``apply_bloch_phase`` (separable 1D, current; extend to flat-r in A1) |
| ``src/common/wfn_transforms.py`` | (new) | ``accumulate_rchunk_to_gflat`` (A2) |
| ``src/common/fft_helpers.py`` | 304-334 | ``make_sharded_ifftn_3d`` / ``make_sharded_fftn_3d`` (use as the primitive) |
| ``src/common/fft_helpers.py`` | 37 | ``query_fft_peak_bytes`` (model the rchunk-FFT peak) |
| ``src/common/fft_helpers.py`` | (new) | ``make_local_3d_fft_chunked`` (A3) |
| ``src/common/isdf_fitting.py`` | ~1500 | triangular solve seam (B1) |
| ``src/common/isdf_fitting.py`` | ~1996 | writer's q-IBZ slice (B2, becomes no-op identity) |
| ``src/common/isdf_fitting.py`` | 1742 | ``fit_zeta_to_h5`` header-write entry (C1.2: replace r-space chunked write with G-flat accumulator) |
| ``src/common/isdf_fitting.py`` | 2092 | post-write ``mark_zeta_done`` (already there) |
| ``src/common/symmetry_maps.py`` | 346 | ``find_irreducible_qpoints`` returns ``q_irr_full_idx`` (4-tuple, last entry) |
| ``src/file_io/zeta_reader.py`` | 301 | ``_do_disk_to_G`` (existing r→G post-FFT; G-flat path skips this) |
| ``src/file_io/zeta_loader.py`` | — | ``ZetaLoader.load`` (add ``zeta_layout`` dispatch in C1.3) |
| ``src/file_io/isdf_header.py`` | — | Add ``zeta_layout`` field + ``n_G_sph`` + ``sphere_idx`` (C1.1) |
| ``src/gw/v_q_tile.py`` | 1454 | ``_unfold_v_q_ibz_to_full`` (template; transverse mirror in D2) |
| ``src/gw/v_q_tile.py`` | 1115 | ``_round_up_to_mesh`` (μ-padding pattern; already used by V_q) |
| ``src/centroid/orbit_syms.py`` | 209 | ``compute_centroid_sym_perm`` (centroid π_S) |
| ``src/centroid/orbit_syms.py`` | 340 | ``compute_rgrid_sym_perm`` (r-grid σ_S; from Pass-2) |
| ``src/psp/gvec_utils.py`` | — | ``build_master_gvec_list(crystal)`` (C2.1) |
| ``src/file_io/_slab_io_ffi.py`` | 339-410 | async write dispatch loop (reference; not modified) |

## Smoke target settings (MoS2 3×3)

- ``cohsex_orbit.in`` at ``runs/MoS2/00_mos2_3x3_cohsex/02_lorrax_xonly/``
  with ``centroids_file = centroids_frac_641.txt`` (orbit-closed)
- ``SLURM_JOBID=52841861`` (4-node 4-hour allocation)
- lxrun module: ``lorrax_D/0.1.0`` + ``lorrax_agent``
- Direct srun template (when lxrun is contended)::

    srun --jobid=$SLURM_JOBID --overlap --mpi=cray_shasta -N1 -n4 \
         --gres=gpu:4 \
         /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/ffi/cpp/select_gpu.sh \
         $LORRAX_SHIFTER \
         /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/ffi/cpp/in_container.sh \
         python3 -u -m gw.gw_jax -i cohsex_orbit.in

- Bit-identity reference: 5 lines × 9 k-points × ±1 ULP diff vs
  prior smoke output is acceptable (matches the recurring baseline).
