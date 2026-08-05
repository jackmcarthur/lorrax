# `bse_k_grid`: general BSE-init coarse→fine interpolation

**Branch** `agent/bse-kgrid` (off `agent/bse-coarse-w` @ a91a831, which has
`pad_W_R_to_grid`). **System** MoS2 640-centroid slab. **Date** 2026-07-20.

## What it does

A production config knob `bse_k_grid NX NY NZ` that makes the **general BSE
init** interpolate the ENTIRE BSE problem — wavefunctions, QP energies, V_Q
exchange, and the W direct term — from the coarse restart grid onto a finer
k-grid **before any solve**, so **every** BSE solver (exciton_bands, feast,
nontda, kpm, resolvent) transparently runs on the fine grid. The coarse-W pad
is driven automatically.

## Where the general init is + exactly what changed

The single choke point where every BSE solver obtains its `data` bundle is
**`bse.bse_io.load_bse_data_from_restart_sharded`** (feast, nontda, kpm, jax,
lanczos, pseudopoles, davidson, haydock, exciton_bands all call it). The
interpolation lives there, so it is not exciton-bands-specific.

| File | Change |
|---|---|
| `src/bse/bse_io.py` | New `bse_k_grid` param on the loader (auto-read from the cohsex.in key when `None`). New `_interpolate_bse_data_to_grid()` — the coarse→fine densification. `_parse_grid_spec` / `_resolve_bse_k_grid` helpers. Fast path: unset or == coarse → coarse bundle returned untouched. |
| `src/bse/vq_interp.py` | New `build_vq_evaluator()` — ONE orchestration of the arbitrary-Q exchange model (stages 1–3), shared by exciton_bands AND the general init. `minibz_head_vlr()` gains a `kgrid=` override (fine mini-BZ head). |
| `src/bse/exciton_bands.py` | Refactored its inline `vq_interp` model build to call the shared `build_vq_evaluator` (consolidation — one exchange-interp path, not two). |
| `src/gw/gw_config.py` | New `bse_k_grid` default (`""`). |
| `docs/docs_gwjax/COHSEX_INPUT.md` | `bse_k_grid` reference. |
| `tests/test_bse_kgrid.py` | New gate (parser + on-grid identity + densify). |

`_interpolate_bse_data_to_grid` reuses existing machinery, single-sourced:
psi/eps <- `bandstructure.bse_setup.compute_wfns_fi(kgrid_fi=...)`; V_Q <- the
shared `vq_interp.build_vq_evaluator` eval'd at Q=0 with the fine mini-BZ head;
W <- `bse_io.pad_W_R_to_grid` (fft'd back to a fine `W_q`). Band/mu dims and the
q=0 head projectors are grid-invariant; only the k axis (and W's k-axes) grow.

## Validation (real numbers, real logs — `logs/`)

### 1. On-grid identity — EXACT

`bse_k_grid == coarse` -> the `data` bundle is byte-identical to the no-flag
path, and a full solve gives identical eigenvalues.

```
byte-identical bundle: True
   max|d psi_c_X|=0.000e+00  max|d psi_v_X|=0.000e+00  max|d M_X|=0.000e+00
   max|d eps_c|=0.000e+00    max|d eps_v|=0.000e+00
   max|d W_q|=0.000e+00      max|d V_q0|=0.000e+00
identity solve max|d eig| (Ry): 0.000e+00
```

**exciton_bands** (owner's named check): a full Gamma->M solve with
`bse_k_grid = 3 3 1` (== coarse) vs no flag -> the eigenvalue `.dat` rows are
identical (only the `# input:` filename comment differs). `logs/exb_noflag.dat`
vs `logs/exb_kg3.dat`.

### 2. Fine-grid physics — coarse 3x3 -> `bse_k_grid 12 12 1` vs native 12x12

Lowest TDA excitons (eV), MoS2 640c, DFT energies, 4v4c, n_eig=6, 1 GPU
(`logs/densify_3to12_vs_native.npz`):

| grid | lowest exciton (eV) | delta vs coarse |
|---|---|---|
| coarse 3x3 (no flag) | 0.1794 | — |
| interp 12x12 (`bse_k_grid`) | 1.2209 | +1042 meV |
| native 12x12 (own restart) | 1.1426 | +963 meV |

Interpolation error (interp-native), lowest 6 states (meV):
`[78.3, 78.8, 115.3, 119.4, 73.9, 73.9]` -> ~78 meV on the lowest state.

Physics reading: the pathologically-coarse 3x3 grid massively over-binds; finer
k-sampling relaxes the binding (+1042 meV). The RPA kernel (D+V, no W) is
essentially unchanged across the densification (1.703 eV at both 3x3 and 12x12)
— so the entire shift is the screened-direct-term W(k-k') k-convergence, which
the pad handles exactly. The interpolation recovers 963/1042 ~ 93% of the true
shift; the 78 meV residual is the htransform psi/eps error from a 4x
densification off a 9-k-point base plus the exchange head-convention difference
(interp uses the `vq_interp` mini-BZ LR head; native uses the GW-computed
`vhead` rank-1). Smaller densifications interpolate better (3x3->6x6, 2v2c:
lowest 0.184->1.150 eV, +966 meV; `logs/densify_3to6.npz`).

### 3. Generality — a SECOND, unmodified production driver

`python -m bse.bse_feast -i cohsex_kg12.in --n-val 4 --n-cond 4 --tda`, where
`cohsex_kg12.in` carries `bse_k_grid = 12 12 1`. bse_feast passes only
`input_file` to the loader; the general init read the key and densified — feast
never knew (`logs/feast_interp12.log`):

```
[bse_k_grid] coarse 3x3x1 -> fine 12x12x1 (144 k-pts); interpolating psi/eps (htransform), V_Q (vq_interp), W (zero-pad in R)
[bse_k_grid] V_q0 exchange tile via vq_interp eval_vq(Q=0), fine mini-BZ head <v_LR>=13.2337 (gstar=164)
[bse_k_grid] W zero-padded in R 3x3x1->12x12x1 (exact trig-interp; direct term now on the fine grid)
--- Spectral bounds (Lanczos) --- E_min (diag gap): 1.700 eV   E_max: 5.600 eV
```

The `bse_kgrid_validate.py` harness also solves the interp-fine bundle with a
second kernel (RPA, `kernel="rpa"`) — proving the fine bundle is consumed by
more than one matvec path.

### 4. W-pad fired

Every fine run prints the `[bse_k_grid] W zero-padded in R ...->...` banner
(above) — `bse_k_grid` drove the coarse-W pad automatically.

### 5. Golden gates

`test_bse_kgrid` (new), `test_exciton_bands`, `test_bse_vq_interp`,
`test_coarse_w_pad`, `test_gw_jax_regression -k "not bispinor"` — see
`logs/golden_gates.log`.

## Consolidation — one interpolation path, not two

- V_Q exchange: exciton_bands and the general init both call the SAME
  `vq_interp.build_vq_evaluator` (its inline stage-1->3 sequence was extracted
  into that one function). No second copy of the setup.
- psi/eps: both call the SAME `bandstructure.bse_setup.compute_wfns_fi`
  (exciton_bands with `q_list={k+Q}`, the init with `kgrid_fi=fine`).
- W: both call the SAME `bse_io.pad_W_R_to_grid`.

The exciton Q-path dispersion stays a distinct use, but its wfn/V_Q/W
interpolation now routes through the identical helpers the general init uses.

## Caveats / next

- Slab (2D) only for the V_Q leg — `vq_interp` asserts `q_z=0` (same restriction
  as exciton_bands). psi/eps and W legs are dimension-agnostic.
- `nband` must stay modest (<=~48 for MoS2/640c) or the htransform eps recovery
  over-packs (the `10_lorrax_exciton_bands` finding); the validation used
  nband=40.
- The exchange q=0 head uses the `vq_interp` mini-BZ LR-head convention; a
  future refinement could reconcile it with the GW `vhead` rank-1 for a tighter
  interp-vs-native match (the lowest exciton is direct-term dominated, so this
  is secondary).
