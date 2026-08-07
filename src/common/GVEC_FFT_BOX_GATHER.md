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

One `jnp.take` per k (trivially vmapped or absorbed into a flat
single-gather via `flat_idx = k*(ngkmax+1) + inv[k,...]`).  Measured
in `common/phdf5_wfn_read_test.py`: **800 ms scatter → 90 ms gather**
at MoS2 3×3 scale; scatter correctness drops out automatically
because the sentinel row is zero.

## Proposed utility

`src/common/gvec_fft_box.py`:

```python
def build_g_index_for_fft_box(
    gvecs_per_k, fft_grid, ngkmax, *, ngk_valid=None,
) -> np.ndarray:
    """g_index[k, nx, ny, nz] = position of this box cell's coefficient
    within k's coefficient slab, or ngkmax (sentinel) if empty."""

def make_fft_box_kernel(mesh, nk, ngkmax, nb_padded, nspinor, fft_grid):
    """Returns a jitted callable (cnk_slab, g_index) -> psi_G_fft_box
    that fills a sharded FFT box from a re/im-packed coefficient slab
    in one gather (no scatter, no per-k loop)."""
```

Note the two DIFFERENT sentinels this file talks about:

* the **empty-cell sentinel** `ngkmax`, a value in `g_index`, which the
  runtime kernel turns into a zero by appending one zero coefficient
  slot.  That is the subject of everything above.
* the **pad sentinel**, a G-VECTOR (`fft_box_pad_sentinel`) that fills
  the `[ngk[k], ngkmax)` rows of the padded G table.  See the module
  docstring of `gvec_fft_box.py`.

They are unrelated; the shared word is historical.

Both are already used by `PhdfWfnReader` in
`common/phdf5_wfn_reader.py`; available to any caller that holds
per-k ngkmax-wide slabs in the same layout.

## Call sites that should migrate

Hot (inner loop, big win):

- **`common/load_wfns.py::read_Gvecs_to_devices`** — the band-chunked
  G-space loader.  Already replaced in the new `phdf5_wfn_reader`;
  the legacy host-numpy path stays as a fallback for non-FFI builds.

Setup-time (correctness-consistency win, modest throughput gain):

- **`psp/get_DFT_mtxels.py:227`** — `buf.at[Gx, Gy, Gz].set(row)` for
  the DFT-matrix-element buffer.  Runs once at psp setup; migration
  is a 3-line edit.
- **`centroid/get_charge_density.py:37`** — `fft_box.at[ix, iy, iz].set(data_1d)`
  for the charge density used as kmeans weight.  Runs once at
  preprocessing.

Verification/debug (migrate opportunistically):

- **`gw/w_from_eps0_0d_check.py:243`** — 0D check code.

Leave alone (host-numpy scatter is already fast):

- `load_wfns.py:41` (`load_kpoint_fftbox`) — host numpy.
- `load_wfns.py:223` (legacy `read_Gvecs_to_devices` k-loop) — host
  numpy.  The new phdf5 reader obsoletes this path but we keep it
  around as a non-FFI fallback.

## Design notes for the utility

- `inv` is keyed by `(gvecs_per_k, fft_grid)` which for a given WFN
  file is constant.  Build once, reuse across every band chunk, every
  iteration of every driver.
- The sentinel scheme (`ngkmax` as "empty") only works if `cnk_padded`
  always has a zero at that slot — the utility should either own the
  padding or document the contract loudly.
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

## Why this is not a sweeping refactor

Only the hot inner-loop site (`read_Gvecs_to_devices`) paid a real
cost.  The setup-time JAX scatters are two-edit migrations worth
doing the next time we're editing those files — not a dedicated
project.  Host-numpy sites are already fine.  The win was always
concentrated in the band-chunked GW loop.
