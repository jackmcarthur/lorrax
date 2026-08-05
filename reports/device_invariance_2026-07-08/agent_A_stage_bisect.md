# Agent A — stage bisect: 4-GPU vs 16-GPU device invariance (MoS2)

Date: 2026-07-08. Artifacts analyzed: `runs/MoS2/Z_memplanner_validation_2026-07-06/{A_charge,B_bispinor}/head_{4g,16g}` (head runs). No reruns were needed — the runs already dumped every intermediate (`tmp/zeta_q*.h5`, `tmp/isdf_tensors_*.h5`, `tmp/v_q_bispinor.h5`, `sigma_mnk.h5`). All arrays are complex128. Analysis script: session scratchpad `diff_outputs.py` (a `sigma_diag.dat` / `eqp{0,1}.dat` parser — not previously in `skills/compare/SKILL.md`).

## Verdict (one paragraph)

**The first diverging stage in BOTH systems is the ζ fit (stage 1).** Every fit input is bit-identical between 4g and 16g (centroids `r_mu_crystal`/`r_mu_fft_idx`, `psi_full_y`, `enk_full`: max|d| = 0). ζ output differs at ~2–9e-7 Frobenius-relative in float64 — 1e9× machine eps, but the divergence is **present at the same magnitude in the bispinor charge channel where n_rmu = 640 is divisible by both 4 and 16 (zero padding at either P)**, so **μ-padding is exonerated as the necessary cause**. All pad rows/cols in every dumped 16g padded tensor (`V_qmunu`, `W0_qmunu`, `psi_full_y`, `G0_mu_nu`, `g0_mu`) are **exactly zero** — no pad-row leak at any dumped stage. The ~1e-7 ζ noise is device-count-dependent reduction order (different psum tree/local matmul shapes at P=4 vs P=16) amplified by the ζ-fit solve's conditioning. Downstream, the noise is amplified into eV by two *different* ill-conditioned consumers: (A_charge) near-pole GN-PPM Σ_C evaluation; (B_bispinor) the near-singular indefinite transverse (TT) ζ fit whose output feeds Σ^B.

## Stage-by-stage divergence table

### A_charge (GN-PPM, 1204 centroids, nk=16; padding 1204→1216 exists only at P=16)

| stage | quantity | 4g-vs-16g divergence | verdict |
|---|---|---|---|
| 0 fit inputs | centroids, `psi_full_y` (valid block), `enk_full` | max\|d\| = 0 (bit-identical) | clean |
| 0 padding | 16g pad rows/cols of `V_qmunu`,`W0_qmunu`,`psi_full_y`,`G0_mu_nu`,`g0_mu` | max\|.\| = 0.000e+00 exactly | **no pad leak** |
| 1 ζ fit | `zeta_q_G` (16,1204,1964) | frobrel 2.4e-7 – 8.7e-7 per q (max\|d\| 4.5e-2 of \|a\|max 3.8e4) | **FIRST DIVERGENCE** |
| 2 V_q / W0 | `V_qmunu`, `W0_qmunu` valid 1204² block | median rel 1.2e-7; frac rel>1e-6 = 7% | inherits ζ noise, no growth |
| 3 Σ_SX / Hartree | `sigma_sx_kij_ev`, `hartree_kij_ev` | frobrel 1.1e-10 / 8.5e-10 | contraction averages noise DOWN |
| 4 Σ_C(ω) GN-PPM | `sigma_c_kij_ev` (41,16,80,80) | frobrel 8.7e-5, max\|d\| = 2552 eV | **amplified ~500× rel; eV-abs near poles** |
| 5 eqp0/eqp1 | eqp1.dat | max\|dEqp\| = 5.64 eV; 1240/1280 bands > 1 meV | inherited from Σ_C(E_dft) |

- `sigma_diag.dat`: max|dsigX| = 0 (to 1e-6 eV print precision) at all (k,n); Re dΣ_C up to **5.66 eV**, at ALL 16 k (per-k max ranges 0.2–5.7 eV). Not a k≠0-only effect.
- Worst (ik,n): (12,0/1) Re dΣ_C = +5.66 eV; (9,0/1) +5.60; (11,0/1) −5.37; (4,0/1) −5.35 — all deep semicore bands.
- **The run sits in a pathological GN-PPM regime**: |Im Σ_C| ≥ 1 eV for all 1280 (k,n); |Im Σ_C| ∈ [1e4, 6.2e6] eV for 1030/1280 — the evaluation energy is essentially ON PPM poles. dEqp tracks pole proximity monotonically:

| \|Im Σ_C\| (eV) | N | max\|dEqp\| (eV) | median (eV) |
|---|---|---|---|
| 1–100 | 26 | 0.27 | 8.1e-3 |
| 1e2–1e4 | 224 | 0.65 | 6.7e-3 |
| 1e4–6e6 | 1030 | **5.64** | 7.4e-2 |

- eqp0 and eqp1 have identical max divergence (5.639 eV) → the QP linearization/Newton step adds **no** further amplification; the divergence is fully present in the direct Σ(E_DFT) evaluation. The QP solve is deterministic given its inputs; its inputs (Σ_C(E_dft)) already differ by eV.

### B_bispinor (COHSEX + bare Breit, 640 charge + 668 transverse centroids, nk=9)

| stage | quantity | 4g-vs-16g divergence | verdict |
|---|---|---|---|
| 0 fit inputs | centroids, `psi_full_y` (9,32,4,640), `enk_full` | max\|d\| = 0 | clean |
| 1 ζ charge (640, **no padding at either P**) | `zeta_q_G` | frobrel 1.8e-7 – 3.0e-7 per q | **diverges WITHOUT padding — padding exonerated** |
| 2 V_CC / W0 | `V_qmunu`, `W0_qmunu` | frobrel 2.5e-7 / 2.2e-7 | inherits ζ noise |
| 3 Σ_SX/Σ_COH diag | `sigma_diag.dat` | **byte-identical** except last digit of VH on 8 lines | clean at ALL 9 k |
| 1' ζ transverse (668→672 pad only at P=16; indefinite μ_L, LU) | `zeta_q_mu{1,2,3}` | **ORDER 1 – ORDER 1e3**: mu1/mu3 frobrel ≈ 0.9 with systematic 3.1× norm ratio (frob 2.0e5 → 6.4e4); mu2 q1 frob **1.9e5 → 2.3e8** (1000× blowup at 16g) | **catastrophic; near-singular fit** |
| 2' V_TT tiles | `V_qmunu_TT_22` | max\|d\| = 3.8e12 vs max\|a\| = 2.8e7 | inherits ζ_T explosion |
| 3' Σ^B (from V_TT, enters eqp via `sig_x`; NOT printed in sigma_diag) | dΣ^B diag = dEqp0 | **max 2.535 eV, median 69 meV, min 26 meV; ALL 270 (k,n) > 25 meV; ALL 9 k incl. k=0 (2.22 eV)** | real physical divergence |
| 5 eqp0/eqp1 | | max\|dEqp\| = 2.535 eV, 270/270 > 1 meV | = dΣ^B |

**Resolution of the "Σ bit-identical at k=0 but eqp diverges" discriminator:** `sigma_diag.dat` prints only the charge-channel sigSX/sigCOH (`gw_output.py` passes `results.sig_sx`/`sig_coh`); the transverse Breit Σ^B is computed in `src/gw/sigma_x_bispinor.py` from the `V_qmunu_TT_ij` tiles and folded into `results.sig_x`, which feeds eqp{0,1} but is never printed in sigma_diag. Σ^B diverges by eV at every k, including k=0. Nothing diverges "between Σ and eqp" — the diverging Σ term simply isn't in the printed diagnostic.

## Amplification-vs-leak verdict

1. **No pad-row leak observed.** Every pad row/col of every dumped padded tensor is exactly 0.0, and the ζ-stage divergence appears at the same ~2e-7 level in a channel with zero padding at both device counts (bispinor CC, 640). Padding is not the mechanism.
2. **The base divergence is a device-count-dependent roundoff realization inside the ζ-fit solve** (~1e-7 rel in f64 from bit-identical inputs ⇒ effective amplification of ~1e-16 arithmetic noise by ~1e9, i.e. the ζ-fit normal-equations solve has cond ~1e9). This is P-dependent (different collective/reduction topology), so it can never be removed by chunk-order fixes at fixed P — consistent with the orchestrator's ≤6e-5 eV chunk-order measurement.
3. **The eV-scale output divergence is amplification by two known-ill-conditioned consumers**, not a leak:
   - A_charge: GN-PPM Σ_C evaluated essentially on poles (|Im Σ_C| up to 6.2e6 eV). 1e-7-level W noise → 8.7e-5 rel Σ_C → up to 5.7 eV Re Σ_C on near-pole bands → 5.6 eV eqp. Healthy-pole bands (|Im Σ_C| < 100 eV) still show up to 0.27 eV — the whole run is in the invalid-PPM regime, so no band is truly healthy here.
   - B_bispinor: the indefinite transverse ζ fit (LU on near-singular μ_L=i CCT) turns the same ~1e-7 input noise into order-1..1e3 changes in ζ_T/V_TT, and the physical Σ^B moves by 26 meV – 2.5 eV. This is NOT pure gauge (physical Σ^B and eqp change), matching the known TT-conditioning concern (`reports/bispinor_tt_conditioning_2026-06-16`).

## Recommended follow-ups (for other agents)

- The actionable defect is **conditioning of the ζ-fit solves**, worst in the transverse channel: regularize/stabilize the TT fit (pivoted factorization with null-space handling or Tikhonov), and quantify cond(CCT) for the charge channel.
- For A_charge-type GN-PPM runs, the |Im Σ_C| ~ 1e4–6e6 eV values indicate invalid PPM modes dominate; device-variance of eqp is a symptom, `ppm_invalid_mode` handling is the physics knob.
- Padding audit not needed for THIS bug, but keep the zero-pad invariant gate (pad rows verified exactly zero here at V/W/ψ/G0 — a cheap regression check).
