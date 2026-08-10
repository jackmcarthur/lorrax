# AMENDMENT — `wq_resolvent` DIAGNOSED AND STRUCK (2026-08-08)

**`test_bse_w0_resolvent::test_wq_resolvent_matches_restart_finite_q` is STRUCK
from this file.**  The last undiagnosed red of its family — registered by the
RE-CUT WAVE amendment below, fingerprinted at `rel_err=6.87e-01` and A/B-proven
pre-existing from four bases (`013aad92`, `81a285af`, `e495fc45`, `f0435e9a`) —
was a real code defect.  It is diagnosed and fixed.

| | |
|---|---|
| machine | Perlmutter, lx pool (JID 56522011), 4×A100, Shifter, `lx test`, 1 GPU, `--workers=1` |
| module | `LX_BASE_MODULE=lorrax_J070`, jax 0.7.0 |
| tree | `/pscratch/sd/j/jackm/wq_phase_0808/wt`, branch `fix/wq-resolvent-phase-2026-08-08` off `main` @ `fef002e9` |
| commit | `b5c0cf15` — `src/bse/bse_w_exact.py`, one function (`build_finite_q_data`) |
| prose record | `~/lorrax_bse_perf_2026-08-08/FIX_wq_resolvent.md` |

**THE DEFECT — a CONJUGATION, and it is exactly invisible at q=0.**  The W_q
resolvent's four pair-density vertices carry one fixed conjugation convention
(`K^x = M V M†`, conjugate on the ENCODE leg), pinned by the optical BSE
exchange term.  Composed, that chain assembles `conj(χ₀) = χ₀ᵀ`, while the GW
producer's χ₀(q) — the object whose Dyson solve wrote the `W0_qmunu[q]` tile the
cell is scored against — is the other one.  At q=0 the k-sum runs over ±k pairs
whose pair densities are complex conjugates under TRS, so **χ₀(0) is REAL**
(‖χ₀−χ₀ᵀ‖/‖χ₀‖ = 4.7e-11 measured on the gnppm fixture) and the two conventions
are the same matrix.  At q≠0 they are not (2.9e-01 on the same fixture), and the
chain resums χ₀(−q) against the +q Coulomb tile — a hybrid that is no stored tile
in any conjugation, which is why an earlier grid-wide argmin scan over every q′
in `T(q′)`, `conj(T(q′))`, `conj(T(−q′))` found no match and returned "operator,
not label".  **Fix: conjugate ψ on both legs in `build_finite_q_data`**; each
vertex is bilinear in (ψ_c, ψ_v) with exactly one conj, so this flips all four at
once.  Exact — no TRS assumption.

**THE DIAGNOSIS IS AN A/B, NOT AN ARGUMENT.**  A dense numpy model of the same
chain, run on the restart on a login node with no jax and no allocation,
reproduces the observed per-column failure **digit for digit** under the shipped
convention and closes under the GW one:

| col | rel(GW convention) | rel(shipped convention) | observed on GPU, pre-fix |
|---:|---:|---:|---:|
| 179 | 2.4575e-08 | **6.8742e-01** | **6.8742e-01** |
| 375 | 2.4575e-08 | **6.8742e-01** | **6.8742e-01** |
| 337 | 2.4608e-08 | **3.5347e-01** | **3.5347e-01** |
| 253 | 2.4570e-08 | **7.0595e-01** | **7.0595e-01** |

**GATES.**

| leg | selection | result |
|---|---|---|
| the cell's own file + its chain sibling | `test_bse_w0_resolvent.py test_bse_w_omega_chain.py` | **5 passed** (34.03 s) |
| BSE subset, 9 files | + `test_bse_dense_reference`, `test_bse_w_donation`, `test_bse_matvec_opts`, `test_bse_stack_matvec`, `test_fft_shardmap_context`, `test_bse_feast_runner_cache`, `test_bse_nontda_restart_preflight` | **67 passed, 1 deselected** (71.05 s) — **zero reds** |
| red twin A — revert the flip | same file | **1 failed, 2 passed**, and the failure is `q=(0,1,0) col 179: rel_err=6.87e-01`, the historical fingerprint reproduced |
| red twin B — flip the conduction leg only | same file | **1 failed, 2 passed**, `rel_err=9.28e-01` — a half-flip is a third, also-wrong operator |
| restore | | md5 `bb9584c3d182c98f52fe5eba74a25147` on both sides, `git status` clean — **RESTORE EXACT** |

The cell now closes at **2.459e-08** against its own unchanged 1e-6 gate, and the
**q=0 sibling is unmoved at 2.157e-09** — it does not go through this function.
Both twins leave the q=0 sibling and `test_kgrid_shift_map_matches_roll` green, so
the twin measures the finite-q vertex and nothing global.

**The whole symmetry-reduced q grid**, via `bse_w_exact --compare-wq` on the gnppm
fixture — every q≠0 was `6.87e-01`-class before:

```
 iq   q (kgrid)  max_rel_err      median   max_resid
  0   (0, 0, 0)    3.203e-09   2.253e-09   4.219e-10
  1   (0, 1, 0)    2.854e-08   2.444e-08   6.031e-10
  2   (1, 0, 0)    2.905e-08   2.626e-08   3.481e-10
  3   (1, 1, 0)    7.895e-08   4.871e-08   1.456e-10
  4   (1, 2, 0)    2.820e-08   2.464e-08   3.766e-10
```

That driver already printed "Closure at the GW minimax-quadrature floor confirms
`W_q = v_q(0-H_RPA^q)^-1 v_q + v_q` at every symmetry-reduced q".  The sentence is
now true.

**BLAST RADIUS: one function, three callers, none on a solve path.**
`build_finite_q_data` is called only by `bse_w_exact`'s own `--compare-wq` and
`--w-omega-chain` arms and by the two test files that gate them
(`git grep build_finite_q_data`).  Nothing in `bse_nontda`, the Lanczos/FEAST
solvers or `bse_jax` reaches it, so the non-TDA coupling cross-check is untouched
by construction, and was re-measured on both sides of the branch point to
confirm it: base `fef002e9` and this branch return the Si non-TDA and TDA arms
**identical to every printed digit** (`FIX_wq_resolvent.md` §6.3), so the
non-TDA coupling correction is 0.698 meV on both.

Every leg above was re-taken after the rebase onto `fef002e9`; the branch was cut
at `ed11a955`, gated there in full, and the rebase was conflict-free with an
identical diff.

**Two hypotheses this closes, both previously live.**  The stale-artifact family
was already dead (`FIX_nontda_feature.md` §4); this adds that the artifact was
never the question — the fresh restart and an Aug-7 one give **identical** dense
closure to all printed digits.  And the umklapp-phase lead its own docstring
ranked first is **wrong**: 6.87e-01 does sit inside the 0.6–3.2 band that
docstring quotes, but the band was coincidence.  The `no umklapp Bloch phase`
bullet stands, verified — the phase structure was exact at every q all along
(|ratio| = 1.00000, arg = 0.003° between the two constructions of χ₀).
