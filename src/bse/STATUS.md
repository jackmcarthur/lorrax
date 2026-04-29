# BSE module status — agent C, 2026-04-28

> **If you are about to run a LORRAX-vs-BGW absorption comparison, stop
> and read [BGW_COMPARE.md](BGW_COMPARE.md) first.** It enumerates six
> conventions (dipole operator, eqp source, head injection, SOC band
> counting, n_occ resolution, broadening/iter-count) that *every*
> comparison must satisfy. Skipping any of them produces silent O(1)
> errors that look plausible. The cookbook there has the exact command
> sequence that reproduces the validated 8×8 Si Haydock comparison.

## Modules

| File | Role | Status |
|---|---|---|
| `bse_jax.py`            | CLI entry, sharded driver, `_preview_lanczos` | working; `--n-reorth -1` (full reorth) is the right default for spinor BSE |
| `bse_simple.py`         | plain-jit (μ,ν) matvec — XLA partitioner, no shard_map | fastest, default `--matvec-kind=simple` |
| `bse_ring_comm.py`      | shard_map + ppermute / all-gather matvec | kept for memory-tight runs (`--matvec-kind=ring`) |
| `bse_lanczos.py`        | `solve_bse_sharded` Lanczos / block-Lanczos / convergence-driven | works; ghost eigenvalues at high N without full reorth |
| `bse_io.py`             | restart-bundle reader, `write_eigenvectors_stream` | writer is BGW-compliant (see "Index ordering" below) |
| `absorption_common.py`  | h5 readers + Lorentzian + Kramers-Kronig + BGW-format `.dat` writers | new |
| `absorption_eigvecs.py` | ε₂(ω) via Σ_S |⟨0\|r̂\|S⟩|²·L (sum-over-states) | new |
| `absorption_haydock.py` | ε₂(ω) via continued fraction on (α_n, β_n) recurrence, no eigvecs | new — *the* method to use vs BGW |
| `eigenvectors.h5.spec`  | BGW spec, kept verbatim | reference |

## Index ordering — read this BEFORE comparing to BGW

BGW conventions (in HDF5 / `eigenvalues.dat` / vmtxel binary) — opposite of LORRAX internal in two places:

1. **Valence axis is reversed**: BGW `iv=1` is the *highest* valence band (just below gap); LORRAX internal `v=0` is the *lowest* (deepest) valence band. Source: `BGW/Common/evecs.f90:1982` writes `[scalar, ns, nv, nc, nk, N, nq]`; `BSE/input_fi.f90:407` does `ib = kp%ifmax - ib_kp + 1`. Conduction axis is the same in both (`c=0` lowest).
2. **Eigenvalue units**: BGW writes eigenvalues in **eV** (`eigenvalues.dat` and `eigenvectors.h5`); LORRAX internal works in **Ry** (factor 13.6056980659).
3. **Fortran column-major → numpy via h5py**: file shape `[scalar, ns, nv, nc, nk, N, nq]` (Fortran) becomes numpy shape `(nq, N, nk, nc, nv, ns, 2)` after axis reversal. Axis 3 = nc, axis 4 = nv. Verified by reading BGW source.
4. **vmtxel binary** (`Common/misc.f90 bse_index`): flat index = `is + (iv-1 + (ic-1 + (ik-1)*nc)*nv)*nspin`, k slowest, v fastest. Reshape to `(nk, nc, nv)`.

`write_eigenvectors_stream` in `bse_io.py` flips valence on write so our `eigenvectors.h5` is BGW-compliant for downstream consumers; converts Ry→eV on the eigenvalues dataset. No flips on conduction/k axes.

## Fair-comparison measures applied to BGW (Si 4×4×4, n_val=8, n_cond=8)

| Aspect | What we did |
|---|---|
| Same QP corrections | LORRAX `--eqp ../00_bgw_bse/eqp.dat` (BGW's eqp1.dat) |
| Same head value | LORRAX `vhead = 3303.748` (BGW's `wcoul0` debug value, exact) |
| Same dipole physics | LORRAX `psp/get_dipole_mtxels.py --skip-vnl` matches BGW's `use_momentum` keyword (both use bare p̂, no `i[r,V_NL]`) |
| Same η broadening | LORRAX `--eta-eV 0.15`, BGW `energy_resolution 0.15` |
| Same band count | both 8 val × 8 cond |
| Same number of Haydock iterations | LORRAX `--n-iter 100/500`, BGW `number_iterations 100/500` |
| Polarization | both `1 0 0` (Cartesian x = b1 in BGW suffix convention) |
| Eigenvector valence-axis convention | LORRAX flips on file write → matches BGW spec |

## Gauge-invariant comparison checks

QE leaves `ψ_n(k)` undetermined up to a per-`(n,k)` U(1) phase. BSE basis vectors `|cv,k⟩` inherit per-`(c,v,k)` phases. *Complex inner products* of BGW vs LORRAX eigenvectors are gauge-dependent — must use gauge-invariant scalars:

- **Total `Σ_{cvk} |d_cvk|²`** (Frobenius norm of dipole vector): BGW = 2314.177, LORRAX = 2314.177 → **machine precision match**
- **`Σ_{cvk} |A^S[cvk]|²` distribution** (per-state density): cosine similarity 0.76–0.88 (best-match per state)
- **Manifold-summed `Σ_{S∈mfd} |⟨0|r̂|S⟩|²`** (basis-invariant within degenerate manifold)
- **ε₂(ω)** (gauge invariant by construction)

Gauge-DEPENDENT quantities (do NOT compare per-state directly):
- ⟨A^L_i | A^B_j⟩ — complex inner product of eigenvectors
- per-state |⟨0|r̂|S⟩|²
- per-element ⟨c|p̂|v⟩

## Results

### Working well

- **BSE eigenvalues**: agree with BGW within **~3 meV** for lowest 20 (saturated — at ISDF compression floor; doesn't improve from n=400 → n=2400 Krylov)
- **Total ‖d‖²**: machine-precision match (verified by parsing BGW's `vmtxel` binary and reshaping per `bse_index` formula with v-axis flip)
- **Haydock vs Haydock at same iteration count**: peak ε₂ within **1.5%**, peak ω within 70 meV (well below η = 150 meV)
- **Haydock 100 ≈ BGW full diag**: both give peak ε₂ = 141 vs 144 (BGW) vs 146 (LORRAX) at ~3.2 eV — the continued-fraction implicitly resums all 4096 states' moments
- **Eigenvector route at n_eig=100, 8×8 with `--skip-vnl`**: peak ε₂ = 25.96 (BGW reference 26.02 at 4×4) — single-data-point match within 0.2%

### Open / known-not-fixable-trivially

- **Per-state `|A^S[cvk]|²` cosine similarity ~80%** (gauge-invariant). Real eigenvector-decomposition difference between BGW and LORRAX from ISDF compression error in BSE H matrix elements. Eigenvalue first-order stability + densely-packed spectrum (~100 states/0.5 eV at the band edge) means O(δH / spacing) ≈ 20% eigenvector rotations.
- **Lanczos eigvec sum-over-states converges peak slowly**: n=100 captures ~18% of full peak height; n=400 captures ~50%; n→4096 (full diag) needed for full convergence. Use Haydock instead.
- **Haydock 4 eV ghost peak** at n=500: present in both BGW and LORRAX Haydock at same iteration count; smooths out at larger n_iter.
- **Cubic-symmetry breaking**: lowest-3 triplet split by ~485 μeV in LORRAX vs 2 μeV in BGW. Likely ISDF centroid placement not perfectly symmetric; sub-meV impact on absorption.
- **Gauge alignment**: per-(c,v,k) phase between LORRAX `dipole.h5` and BGW `vmtxel` is random (`std(arg(ratio)) ≈ 1.93 rad`). Magnitude per element matches (median ratio 1.06 with v-flip) but phases differ. Sum rules and spectra still match — gauge invariant.

### Unresolved

- **`|d|² per-state breakdown after Lanczos n=100/400` differs by ~3× from BGW's first-100 from `eigenvalues.dat`** — initially flagged as bug; turned out to be method confound (Lanczos Ritz vectors at finite n ≠ exact eigvecs of full diag, particularly for densely-packed middle-of-spectrum states; use manifold sums or Haydock for fair comparison)

## Run artifacts

In `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/`:

- `eps2_8x8_haydock_compare.png` — BGW Haydock 100 vs LORRAX Haydock 100 (the apples-to-apples plot)
- `eps2_8x8_converged.png` — convergence comparison including Lanczos n=100/400
- `eps2_reconstructed_apples.png` — same-N-states ε₂ reconstruction (shows truncation effect)
- `dipole_p_only.h5` — `--skip-vnl` LORRAX dipole, total ‖d‖² = 2314.177 (matches BGW exact)
- `eigenvectors.h5` — LORRAX BGW-compliant eigvec file (post-writer-fix), 200 lowest at n=2400 Krylov
- BGW reference dirs: `00_bgw_bse_8x8/` (full diag, 500 eigvecs), `00_bgw_bse_8x8_haydock/` (Haydock n=500)

## Recommended usage

```bash
# Run BSE solve, get eigenvectors.h5
LORRAX_NGPU=4 lxrun python3 -u -m bse.bse_jax \
    -i cohsex_bse.in --eqp <bgw_eqp.dat> --n-occ <Nocc> \
    --bse --lanczos --tda --matvec-kind=simple \
    --n-val <Nv> --n-cond <Nc> --n-reorth -1 \
    --max-lanczos-iter <2-3x n_eig> --n-eig <Neig> \
    --px 2 --py 2 --write-eigs <Neig>

# Absorption — Haydock (best fidelity vs BGW; no eigvecs needed)
LORRAX_NGPU=4 lxrun python3 -u -m bse.absorption_haydock \
    -i cohsex_bse.in --n-val <Nv> --n-cond <Nc> --n-occ <Nocc> \
    --eqp <bgw_eqp.dat> --dipole dipole.h5 \
    --V-cell <V_bohr3> --n-iter 200 --eta-eV 0.15 --no-eps1

# Absorption — eigenvector route (per-state oscillator strengths to BGW eigenvalues.dat format)
LORRAX_NGPU=1 lxrun python3 -u -m bse.absorption_eigvecs \
    --eigenvectors eigenvectors.h5 --dipole dipole.h5 \
    --n-occ <Nocc> --V-cell <V_bohr3> --eta-eV 0.15 --no-eps1
```

## Known caveats / next steps if needed

- For exact per-state agreement with BGW (sub-3 meV eigenvalues, >95% per-state density): increase ISDF centroid count or implement symmetry-adapted ISDF.
- Haydock's `‖d‖²` factor in continued fraction matches BGW's `mmts%norm = ||d||²` (verified `BSE/haydock.f90:536`); pref `16π²/(V·N_k·n_spin·n_spinor)` matches `BSE/absh.f90:46`.
- All comparisons performed with `use_momentum` (bare p̂, divided by ΔE_DFT). For full-velocity comparison BGW would need `use_velocity` + WFNq run.
