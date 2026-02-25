# RPA `W(0)` / `Wc(0)=W(0)-v` Bug Guide (Casida/GMRES Path)

This note is a handoff summary for agents debugging why the **Casida shifted-solve** pipeline (in `bse_w_exact.py`) does not reproduce the **sum-over-states (SOS) + Dyson** reference for `q=0` static screening.

The goal quantity for comparison is typically:

- `W(0) = (I - v chi0(0))^-1 v`
- `Wc(0) = W(0) - v`

All comparisons below use **no-head** (`V_qmunu_nohead`, `W0_qmunu_nohead`) when present.

## 1) Reference (SOS + Dyson) That We Trust

For a restart file `tmp/isdf_tensors_600.h5` containing `psi_l`, `psi_r`, `enk_l`, `enk_r`, and `V_qmunu_nohead`:

1. Build the transition vertex (SOS convention):
   - `M[k,c,v,mu] = sum_s conj(psi_v[k,v,s,mu]) * psi_c[k,c,s,mu]`
2. Independent-particle polarizability:
   - `chi0_raw[mu,nu] = sum_{k,c,v} M[k,c,v,mu] * (1/ΔE[k,c,v]) * conj(M[k,c,v,nu])`
   - `chi0 = -(2/Nk) * chi0_raw`
3. Dyson:
   - `W = solve(I - V @ chi0, V)`  (matrix solve, not explicit inverse)
   - `Wc = W - V`

This SOS+Dyson implementation matches the `cohsex_jax` static `W0_nohead` extremely well (sub-1e-3 relative Frobenius in previous runs).

For local validation in this repo, use:
- `scripts/compare_rpa_chi_exact_to_sos.py` (compares `chi` columns from `bse_w_exact` against SOS+Dyson).

## 2) What We Implemented for the Casida/GMRES “Exact” Pipeline

The shifted-solve pipeline computes **column actions** by driving the Casida/Liouvillian system with a density-space basis vector `g = e_nu`:

- The density drive uses `u = V g` (unless in debug mode we drive with `u=g`).
- The transition-space RHS uses two couplings for complex Bloch spinors:
  - `f    = d^† u`
  - `fbar = d^T u`
- For non-TDA, we solve a block system for `[X, Y]`:
  - `(z I - S) [X; Y] = [f; -fbar]`
  - `z = (omega + i eta) / Ry_to_eV`, and we care most about `omega=0`, `eta=0` for static.
- The density readout is **fixed** to:
  - `rho = d X + d* Y`
  - and the output “screening channel” is `V rho` (and then we interpret it as `Wc` with an output scale).

Important: we removed older variants like `R(X+Y)` from the code paths to reduce confusion.

## 3) “Green Light” Validations (Things That Work)

### 3.1 D-only / noninteracting limit is correct

If we set the operator to `S = D` (i.e. `--d-only` so kernel blocks are zeroed), then:

- Casida shifted solve reproduces the SOS `chi0` / `V chi0 V` action to ~machine precision.
- This strongly suggests the following pieces are correct:
  - `d` vertex construction in the drive/readout (including spin sums)
  - the `fbar = d^T u` convention for complex wavefunctions
  - GMRES plumbing / sharded matvec wiring

### 3.2 The RPA `V`-term contraction matches an explicit dense formula

Using a tiny `(n_val,n_cond)` subset and explicit dense contractions, we verified:

- The ring `apply_V_ring` implements a kernel of the form:
  - `m_cv[k,c,v,mu] = sum_s conj(psi_c[k,c,s,mu]) * psi_v[k,v,s,mu]`
  - `A_V(X) = (1/Nk) * m_cv^† V (m_cv X)`
  - `B_V(Y) = (1/Nk) * m_cv^† V (m_cv^* Y)`

This is checked by `../tests_isdf/cohsex_prod/check_rpa_kernel_conventions.py`.

## 4) The Failing Symptom (The Bug)

When we turn on the interacting RPA kernel (`S = D + V`, i.e. `--rpa` without `--d-only`):

- The computed response does **not** satisfy the Dyson identity implied by the SOS reference.
- In particular, when we output `chi` columns from the Casida solve (driving with `u=g` and inverting `V` on the host):
  - `||(I - chi0 V) chi_bse - chi0|| / ||chi0||` is O(1) for the tested cases.

This indicates the mismatch is **not** GMRES tolerance or a simple global scalar.

## 4.1) Status Update: Fix Found (k-coupling in the RPA `V` term)

The core bug was that the RPA Coulomb term was treated as **k-diagonal**. For
`q=0` density response, the Hartree/RPA density must **sum over all k**, so all
transitions couple through the shared density channel.

After enabling k-coupling in the RPA `V` term (`apply_V_ring(..., couple_k=True)`
plumbed via `v_couples_k=True` for `include_W=False` / `--rpa`), the results
improve dramatically:

- `bse_w_exact` (RPA, `q=0`, nohead) matches SOS+Dyson `Wc(0)` to ~machine precision
  **with `--output-scale 1.0`**.
- The remaining discrepancy previously observed was an output scaling artifact
  from using an incorrect default `output_scale` (a legacy `4/Nk` fudge); with the
  corrected k-coupling, the natural comparison uses `output_scale=1.0`.

This is now the recommended baseline configuration for any further debugging.

## 5) What We Tried That Did *Not* Fix It

- Sweeping global scale factors on `V` (e.g. `0.3, 0.4444, 0.5, 0.6, 1.0, 2.0`) can make plots “look closer” but does not make the operator satisfy Dyson or reduce the error to ~1e-4.
- Splitting `V` scaling into “kernel” vs “coupling” factors changes the magnitude but still does not restore Dyson-consistent response.
- Switching the folded-sign convention (`S = JH` vs `S = HJ`) improved agreement with `chi0` (noninteracting) but did not fix interacting Dyson residuals.

## 6) Leading Suspects (Where the Real Bug Likely Is)

The noninteracting limit constrains the **outer chain** (drive + readout + D solve) but does not fully constrain the **interacting structure**. The most plausible culprits:

0. **Wrong RPA/Hartree contraction: V term must couple all k even at q=0**
   - For q=0, each *χ0 contribution* is a same-k transition (k→k), but the **Dyson resummation happens after summing over k**:
     - `chi0 = sum_k chi0(k)` then `chi = (I - chi0 V)^-1 chi0`.
   - A V-kernel that applies `V` independently for each k (i.e. keeps an explicit k index through the Coulomb step) instead implements the *non-Dyson* approximation
     - `chi_approx = sum_k (I - chi0(k) V)^-1 chi0(k)`,
     which generically differs by O(1) once V is on.
   - The Dyson-consistent RPA/Hartree kernel in transition space is:
     - `rho_total(mu) = sum_{k,c,v} M(k,c,v,mu) X(k,c,v)`
     - `phi = V @ rho_total`
     - `(VX)(k,c,v) = M*(k,c,v,mu) phi(mu) / Nk`

1. **Dyson/response identity mismatch for the chosen Liouvillian fold**
   - The mapping between `(zI - S)^-1` and `(I - v chi0)^-1` may require an additional `J` on the input or output side when translating between the Casida resolvent and the physical density response.
   - This can preserve D-only correctness while breaking interacting response.

2. **Mismatch of transition vertex convention between SOS reference and Casida kernel**
   - SOS reference uses `m_vc = psi_v^* psi_c`.
   - Ring kernel uses `m_cv = psi_c^* psi_v`.
   - These are related by conjugation/transpose, and `chi0` can still match, but the interacting `A/B` block structure for complex spinors is sensitive to the exact placement of conjugations/transposes.

3. **Bottom-block / antiresonant coupling subtlety**
   - The correct non-TDA structure uses elementwise conjugates `(-B*, -A*)` (not Hermitian adjoints).
   - This was previously a known bug on one branch; it has been addressed in current code, but the remaining mismatch could still be tied to antiresonant coupling conventions (especially for complex spinors).

## 7) Minimal Repro Commands (cohsex_prod dataset)

All runs below are typically done on CPU if the current JAX GPU backend is unstable.

Compute a few `Wc` columns:
```bash
JAX_PLATFORMS=cpu uv run python -m isdf.bse_isdf.bse_w_exact \
  -i cohsex_prod.in --n-val 26 --n-cond 44 --rpa --nohead \
  --cols 0,1,2,3,4,5,6,7,8 \
  --gmres-max-iter 80 --gmres-tol 1e-12 \
  --out bse_w_exact_cols0_8.h5
```

Compare to SOS-Dyson reference:
```bash
uv run python compare_w_exact_to_sos.py --restart tmp/isdf_tensors_600.h5 \
  --n-val 26 --n-cond 44 --bse bse_w_exact_cols0_8.h5
```

Compute `chi` columns (debug output):
```bash
JAX_PLATFORMS=cpu uv run python -m isdf.bse_isdf.bse_w_exact \
  -i cohsex_prod.in --n-val 26 --n-cond 44 --rpa --nohead \
  --drive-kind potential --write-kind chi \
  --cols 0,1,2,3,4,5,6,7,8 \
  --gmres-max-iter 80 --gmres-tol 1e-10 \
  --out bse_chi_cols0_8.h5
```

Compare `chi` to SOS:
```bash
uv run python compare_chi_exact_to_sos.py --restart tmp/isdf_tensors_600.h5 \
  --n-val 26 --n-cond 44 --bse bse_chi_cols0_8.h5
```

## 8) Recommended Next Debugging Step

Focus on proving (or falsifying) the exact transfer-function identity between:

- the physical density response `chi(0)` used in `W = (I - V chi0)^-1 V`, and
- the Casida/Liouvillian resolvent `(zI - S)^-1` driven/read out with the current `f/fbar` and `dX+d*Y` conventions.

The D-only case already matches, so the discrepancy must be in how the interacting kernel maps to the physical Dyson equation.
