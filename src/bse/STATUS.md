# BSE module status — last revised 2026-07-22

> **Running a LORRAX-vs-BGW absorption comparison?** Read
> [BGW_COMPARE.md](BGW_COMPARE.md) first. It enumerates six conventions
> (dipole operator, eqp source, head injection, SOC band counting, n_occ
> resolution, broadening/iter-count) that *every* comparison must satisfy.
> Skipping any produces silent O(1) errors that look plausible.
>
> **Running an arbitrary-Q exciton bandstructure?** Read
> [EXCITON_BANDS.md](EXCITON_BANDS.md) first. Both failures in that pipeline
> to date were silent — run completes, on-grid gates pass, bands wrong by eV.
> It documents the two traps, how to gate off-grid, and the sizing rules.

## Modules

Absorption / eigensolvers (the 2026-04 arc, validated vs BGW below):

| File | Role | Status |
|---|---|---|
| `bse_jax.py`            | CLI entry, sharded driver, `_preview_lanczos` | working; `--n-reorth -1` (full reorth) is the right default for spinor BSE |
| `bse_simple.py`         | plain-jit (μ,ν) matvec — XLA partitioner, no shard_map | fastest matvec, but opt-in via `--matvec-kind=simple`; the solver default is `ring`, not this (bse_lanczos.py:159) |
| `bse_ring_comm.py`      | shard_map + ppermute / all-gather matvec | the DEFAULT matvec (`--matvec-kind=ring`); also the memory-tight choice |
| `bse_lanczos.py`        | `solve_bse_sharded` Lanczos / block-Lanczos / convergence-driven | works; ghost eigenvalues at high N without full reorth |
| `bse_stack_matvec.py`   | batched trial-stack matvec, one T-tensor regardless of `n_trials` | working |
| `bse_nontda.py`         | structure-preserving non-TDA (full-BSE) eigensolver | working |
| `bse_feast.py`          | FEAST contour-integration eigensolver | see `context/feast_accuracy_notes.md` |
| `bse_kpm.py`            | KPM Chebyshev moments → BSE density of states | working |
| `bse_pseudopoles.py`    | FEAST-based pseudopole construction, density-biased seeds | working |
| `bse_io.py`             | restart-bundle reader, padding utils, `write_eigenvectors_stream` | writer is BGW-compliant (see "Index ordering" below); also `pad_W_R_to_grid` / `bse_k_grid` coarse→fine |
| `absorption_common.py`  | h5 readers + Lorentzian + Kramers-Kronig + BGW-format `.dat` writers | working |
| `absorption_eigvecs.py` | ε₂(ω) via Σ_S \|⟨0\|r̂\|S⟩\|²·L (sum-over-states) | working |
| `absorption_haydock.py` | ε₂(ω) via continued fraction on (α_n, β_n), no eigvecs | *the* method to use vs BGW |
| `eigenvectors.h5.spec`  | BGW spec, kept verbatim | reference |

Finite-/arbitrary-Q and screened-W (the 2026-07 arc — see EXCITON_BANDS.md):

| File | Role | Status |
|---|---|---|
| `vq_interp.py`          | arbitrary-Q bare-exchange tile `V_Q`, F-scheme + b26p | working; `build_vq_evaluator` is the entry point |
| `exciton_bands.py`      | `E_S(Q)` along a Q path, finite-momentum TDA | working; single-compile `lax.scan`, `--extra-q` for the symmetry gate |
| `w_omega_chain.py`      | full-frequency `W_q(ω)` via one block-Lanczos chain | working |
| `bse_w_exact.py`        | exact `W_c(ω)` by shifted solves on the non-TDA RPA resolvent | cross-validation reference |

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

## Handoff notes, 2026-07-22

Cross-cutting facts established during the arbitrary-Q work that are easy to
re-derive the hard way.

**One JAX process per GPU, always.** This is the LORRAX process model, chosen
because it is what research clusters prefer. The distributed-linalg FFIs
(cuSOLVERMp, cuBLASMp, SLATE) requiring one process per device is a *match*,
not a constraint to work around. Single-process multi-GPU is not a supported
geometry — the cuSOLVERMg backend built on it has been deleted. Single-*device*
runs are unrelated and must keep working: gates must not require 16 GPUs.

**One eigh dispatcher.** `ffi.common.dispatch.dispatch_eigh` is the single
backend switch (`auto|off|cusolvermp|slate`) for both `vq_interp`'s coarse
`C_q` and `bandstructure/bse_setup`'s per-q `fH_q`. `vq_interp._eigh_backend`
was deleted; do not reintroduce a parallel dispatcher. Native batched is the
default — the FFI arms are exact but 11–41× slower per matrix and do not
reduce the high-water mark (see EXCITON_BANDS.md § Memory).

**Silent-failure pattern, twice now.** Both bugs in this arc kept the run
alive and the usual gates green: the Galerkin Gram cross-term loss (Cholesky
still succeeds; only `ctilde` orthogonality moves), and the charge ζ-solve
silently downgrading from `rank_truncate` to Cholesky above a hardcoded
replication cap (produced ζ 4.5× too large; now raises instead, and the cap
honours `LORRAX_ZETA_REPLICATE_CAP_GIB`). When adding a knob or a fallback,
make the unsupported path *refuse*; a config key that is parsed and quietly
ignored has cost this project multiple days.

**Gate metrics must move when the thing under test moves.** An on-grid
`|Δε_c|` gate that returns a bit-identical number across two different fH
windows is not measuring the window. Before trusting a gate, perturb the
input and confirm the metric responds.
