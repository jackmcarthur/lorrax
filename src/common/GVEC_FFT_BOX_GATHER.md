# `gvec_fft_box` — a shared gather primitive for "sparse G → dense FFT box"

## The pattern

Several parts of LORRAX take a small vector of wavefunction
coefficients `cnk` indexed by G-vectors and place them into a dense
three-dimensional FFT box (so it can be inverse-FFT'd, gathered at
centroid positions, multiplied against a dense G-space kernel, etc.):

```
psi[..., gvec[:,0], gvec[:,1], gvec[:,2]] = cnk
```

On GPU via `jax`, writing this as an `.at[...].set(...)` scatter is
**catastrophically slow** — measured ~800 ms at MoS2 3×3 scale for
9 k-points × 20 bands × 2 spinors × 1963 G-vectors (that's ≈700 k
atomic writes into a 46 k-slot FFT box; GPU HBM-atomic bandwidth with
contention puts this out at ~1 MB/s effective).  Even with
`unique_indices=True` hinting and `mode='drop'`, the scatter kernel
JAX emits has 10–100× worse throughput than the equivalent gather.

## The fix — gather with a precomputed inverse index

Flip the direction.  Instead of iterating over G-vectors and writing
to FFT slots, iterate over FFT slots and read from G-vectors.  The
sparsity structure — which G-slots map to which FFT cells — is fixed
by the WFN file (doesn't change during a calculation), so precompute
once on host:

```
inv[k, nx, ny, nz] = g_local_within_k    if (gvecs[k, g] % fft_grid) == (nx, ny, nz)
                   = ngkmax (sentinel)   otherwise
```

At runtime, pad `cnk` with one zero row at index `ngkmax`, then:

```
psi[..., k, nx, ny, nz] = cnk_padded[..., k, inv[k, nx, ny, nz]]
```

One `jnp.take` over the flattened `(k,G)` plane uses
`flat_idx = k*(ngkmax+1) + g_index[k,...]`.  Scatter correctness drops out
automatically because the sentinel row is zero.

## Current ownership

`src/common/gvec_fft_box.py` owns the padded G-vector representation and the
host-built inverse index:

```python
def build_g_index_for_fft_box(
    gvecs_per_k, fft_grid, ngkmax, *, ngk_valid=None,
) -> np.ndarray:
    """g_index[k, nx, ny, nz] = position of this box cell's coefficient
    within k's coefficient slab, or ngkmax (sentinel) if empty."""

```

`src/common/wfn_transforms.py` owns the sole device gather and all FFT
variants.  Its public `to_box`, `to_rbox`, `to_rmu`, and `to_rchunk` routes
share `_box_kernel`, and their FFTs route through `common.fft_helpers`.
Charge density, current density, matrix-element sweeps, qsgw density, kmeans,
htransform, and the reusable Galerkin source consume that same implementation.

Note the two DIFFERENT sentinels this file talks about:

* the **empty-cell sentinel** `ngkmax`, a value in `g_index`, which the
  runtime kernel turns into a zero by appending one zero coefficient
  slot.  That is the subject of everything above.
* the **pad sentinel**, a G-VECTOR (`fft_box_pad_sentinel`) that fills
  the `[ngk[k], ngkmax)` rows of the padded G table.  See the module
  docstring of `gvec_fft_box.py`.

They are unrelated; the shared word is historical.

## Design notes

- `g_index` is keyed by `(gvecs_per_k, fft_grid)` which for a given WFN
  file is constant.  `WfnLoader.box_index_dev` builds/places it once and
  reuses the same canonical device buffer across band chunks and drivers.
- The sentinel scheme (`ngkmax` as "empty") only works if `cnk_padded`
  always has a zero at that slot.  `_box_kernel` owns that append.
- For variable-ngk (nosym/irreducible-wedge): the COEFFICIENTS are
  zero-padded per k to `ngkmax`, so the gather covers the shape
  uniformly with no per-k dynamic logic at runtime.  The G-VECTORS are
  padded with the pad sentinel instead, and the index build masks them
  off using `ngk_valid`.  Do NOT "pad then scatter everything": pad rows
  have higher `g`, so in a last-writer-wins scatter they would take the
  sentinel's box cell away from any real G sitting there.  (No real G
  may sit there — `pad_gvecs_to_sentinel` refuses tables where one does
  — but the masked build does not have to rely on that, and relying on
  invariants you can avoid relying on is how they rot.)
- Memory: `inv` is `nk × nx × ny × nz × 4` bytes.  At Si 10×10×10
  (nk=1000, fft=24³): ~55 MB per rank, replicated — still fine.  If
  that ever bites, shard `inv` on the k axis and build the gather
  under `shard_map`.

The device implementation is deliberately not exposed from
`gvec_fft_box`: doing so would create a second compiled transform family and
split FFT/sharding conventions from `wfn_transforms`.
