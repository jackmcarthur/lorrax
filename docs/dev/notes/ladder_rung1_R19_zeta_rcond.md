# ladder_rung1_notes §R19 — zeta_rcond 2x2: window fixes, truncation-loosening destroys

Extracted 2026-07-31 from wk_REL/docs/ladder_rung1_notes.md (campaign evidence,
machine-local; wk_REL/ = /scratch2/08271/jackmc/lorrax_setup/wk_REL/). Cited by
`src/common/rank_criterion.py`. Context: MoS2 4x4, nb=1024, mu≈10k, zeta_rcond
swept with everything else fixed.

## R19 — **2x2 COMPLETE.** Axis A fixes it; axis B is CATASTROPHIC.

All cells: nb=1024, mu 10015-10037, n_pad=10048, same WFN, weapons ON.

| cell | prune window | rcond | fit n_keep | eqp0 | eqp1 |
|---|---|---|---|---|---|
| baseline | (0,52) | 1e-8 | 6700 | 0.3645 | -0.3639 |
| **A1** | **(0,256)** | 1e-8 | 6793 | **3.1350** | **3.0710** |
| **A2** | **(0,1024)** | 1e-8 | 6788 | **3.7227** | **3.4551** |
| B1 | (0,52) | **1e-10** | 8290 | **-206.83** | **-1039.84** |
| B2 | (0,52) | **1e-12** | 9461 | **-5049.59** | **-304.20** |

### R19.1 — axis B is not "worse", it is DESTROYED

`zeta_rcond = 1e-10` gives a QP gap of **-206.8 eV**; `1e-12` gives
**-5049.6 eV**.  These are not degraded answers, they are numerical wreckage —
hundreds to thousands of eV on a 2.2 eV DFT gap.  This confirms the
`gw_config.py:617-628` plateau far more violently than its own cited datum
(MoS2 4x4/1204c, MAE 1.4 eV at 1e-10): at THIS size 1e-10 is off by 200 eV.

**My R13/R14.4 reasoning is now refuted from both ends and I want that on the
record without hedging.**  I argued the truncation was "discarding real content
six decades above the f64 noise floor" and proposed sizing rcond against that
floor.  The arithmetic was right; the inference was exactly backwards.
Retaining those modes — 6700 -> 8290 -> 9461, i.e. **+41% more retained
rank** — moves the answer from wrong-by-2.8-eV to wrong-by-5000-eV.  They are
not "real content": they are the over-complete, near-null directions whose
pseudo-inverse amplifies noise by 1/lambda, and truncating them is the CURE.
`zeta_rcond=1e-8` is doing exactly the job it was measured into.

**Corollary, and it is the sharpest statement this campaign produced:**

> Retained rank is not basis quality — in EITHER direction.
> Axis A: +1.4% rank  ->  +2.8 eV of correctness (0.3645 -> 3.1350).
> Axis B: +41% rank   ->  -5000 eV of correctness.
> What matters is WHICH directions the basis spans, never how many.

### R19.2 — the owner's ruling is vindicated twice

Parking the conditioning physics and refusing to let a physics-affecting default
be changed on my noise-floor argument was correct.  Had `zeta_rcond` been
lowered on the strength of R13's arithmetic, the result would have been a
5000 eV error shipped behind a plausible-sounding rationale.  The two guards
that stopped it were (a) the standing rule not to change physics defaults
silently, and (b) reading the config's own recorded measurement history before
proposing a rule.  Both are cheap; both were decisive.

### R19.3 — what is settled and what is not

SETTLED: the rung-5 defect is the prune-window clamp (R12), the fix restores
the gap monotonically (0.3645 -> 3.1350 -> 3.7227), and `zeta_rcond=1e-8` must
not be lowered.
NOT SETTLED, for the numerical-stability pass: the 0.59 eV spread between the
two corrected windows; whether `--prune-window vc_x_vc` (adds cxc pair
densities) is better still; and the rank-saturation curve of R13, every point of
which was taken on a rank-deficient basis and which must be re-measured before
any "max usable nb" rule is published.

