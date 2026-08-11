# The two-window contract is in, the fifth wall is down at Γ by four orders — and the tile null still refuses, on a SIXTH wall that is FINITE q and is not the m-leg (2026-08-11)

**STATUS: THE NAMED FIX IS IMPLEMENTED, GATED AND MEASURED. THE TILE NULL
STILL REFUSES, SO THERE IS NO `.dat`, NO `.png` AND NO CERT. NO TOLERANCE
MOVED: the tile bracket is 5.0e-02, the fH ortho cap 1.0e-06, the reference
cert grade 0.01 meV, `DEGENERACY_TOL_RY` 1.000 meV — all untouched, and all
still refusing where they refused. `LORRAX_FH_ORTHO_TOL` was NEVER set;
`LORRAX_FI_FSHOULDER_TOL=-1` was set on exactly four legs, all of them RED
TWINS whose job is to reproduce a known-bad run, and it is announced by the
resolver in each of their logs.**

Base `origin/main` `0cd4c503`, branch `lane/two-window-refit-2026-08-11`.
Workspace `/pscratch/sd/j/jackm/twowin_0811/`, sole writer; the two probed
parents (`zsolve_0811/p2628n60z52` — the decoupled ζ=52/Σ=60/μ2628 parent —
and `zeta52_0811/dp2628n20`) were read and not touched, and so were the
`si_bse_debug` fixture, the 2628-centroid table and `triangle_0810`'s
`dipole.h5`.

This row takes the fix that
`2026-08-11-fifth-wall-is-the-f-transform-shoulder.md` §5 named and declined
to take — the TWO-WINDOW CONTRACT — and reports what happened.

## 1. What landed

* **`htransform.initialize_wfns(n_guard_bands=…)`** widens the Galerkin window
  ABOVE the deck's, and only above it. Default `0` is the historical window
  band for band. It raises `ncond` **and `nband` together**, because `Meta`
  zero-pads ψ above `nband` and a guard band the deck does not name would
  arrive as exact zeros with nothing downstream to say so; it refuses when the
  WFN cannot supply the bands.
* **`vq_interp.refit_prepare(n_guard=…)`**, `REFIT_N_GUARD_DEFAULT = 4`,
  driven by **`exciton_bands --refit-guard-bands`**. fH spans the ζ window plus
  the guards; **ζ′ is still fitted on the producer's own window, exactly**, so
  `refit_ongrid_null` stays the gate. ψ is streamed over the fH window because
  `B_full = W_proj ψ` is a wide-window object; `rst["psi_r"]` is the ζ
  sub-block, because guards shape fH and are never pair density. Two Galerkin
  residuals are printed and the ζ one is named as the tile's floor. The `.dat`
  carries a `# refit-fh-window:` stamp so a curve cannot be separated from the
  contract it was drawn under.
* **The f-shoulder tripwire in `compute_wfns_fi`**: `min_k |f(ε_b)|/max|f|`
  over the RETURNED window, refusing at zero, with the same announce-or-refuse
  grammar as the `ctilde` orthonormality gate four lines away and one
  deliberate difference — `0` is the DEFAULT here, not the off switch, because
  a band whose `f` is exactly zero is ABSENT rather than inaccurate, so
  disabling it needs a negative value.
* **The splash radius of §7, audited**: `htransform`'s `get_centroids_fi` path
  (`wfn_fi_max` unset = the full band count = zero guards) and `bse_densify`
  (permits `b_max == nb_window`) both warn now; the tripwire is what refuses.
* One bug the contract exposed: `refit_prepare`'s ψ chunk loop broadcast the
  loader's band-padded chunk whole. Every window it had ever seen had a band
  count divisible by the device count (52, 24, 20 at P=4), so the pad never
  materialised; the first odd `nb_fh` died with a numpy shape message naming
  neither the loader nor the pad.

## 2. THE FIX WORKS, AND Γ IS WHERE THAT IS PROVABLE

At an on-grid q the answer is exactly known, and at **Γ** it is known without
any q-labelling convention entering. On `dp2628n20`, four real processes,
mesh 2×2, everything else identical:

| `dp2628n20`, guards | fH window | Γ tile null | α-overlap `min_{m,k} ‖O[m,:]‖` | `min_b min_k \|f\|/max\|f\|` | on-grid \|Δε\| |
|---|---|---|---|---|---|
| **0** | [0, 20) | **1.267e+00** | **0.225204** | **0.000000** | 3.867e-10 meV |
| 1 | [0, 21) | **4.211e-06** | — | — | — |
| 2 | [0, 22) | **4.146e-06** | — | — | — |
| **4** | [0, 24) | **4.688e-06** | **1.000000** | **6.107e-03** | 1.975e-10 meV |
| `m_leg="stored"` (the formulation null, the floor) | — | **3.695e-06** | — | — | — |

**Four orders, onto the formulation null**, and the instrument that convicted
the zero-guard configuration reverses completely: the α-space overlap
`‖O[m,:]‖`, which read **0.225** on the collapsed bands, reads **1.000000** for
every band at every k. That is the fifth wall, and it is down.

**One guard is enough, and that is a structural statement rather than a lucky
number.** `f` is zero only at or above `shift = max_k ε[nb_fh−1]`, and
eigenvalues are ascending in the band index, so a band below the top can be
zeroed ONLY by being exactly degenerate with the top band at the k that
defines the shift — which is what a 4-fold irrep at a symmetry point does, and
is why the dead set was four bands deep. Move the top up by one band and the
whole multiplet is strictly below the new shift. The measurement agrees: the Γ
null is **flat at 4.1–4.7e-06 across guard counts 1, 2 and 4**, and the finite-q
worst is **1.165e+00 in all three**, unmoved to four digits.

**AND THE GUARD EDGE CUTTING A MULTIPLET DOES NOT MATTER — QUANTIFIED.** On
this SOC deck every band edge in 21…24 (and 53…57) splits a multiplet, so
every guard edge here is degeneracy-ILLEGAL. It does not move the tile: the
three guard counts above cut three different multiplets at the load edge and
the Γ null does not move outside 4.1–4.7e-06, nor does the finite-q worst move
at all. That is the "the shoulder weight there is small by construction"
expectation, measured rather than assumed — the f-weight at the fH edge is the
`min_b min_k |f|/max|f|` column, six decades under the alive bands.

**Both red twins fire.** The zero-guard tile null reproduces the published
numbers to every digit: **1.292e+00** over 7 coarse q on the decoupled parent
and **1.409e+00** on `dp2628n20`. The tripwire's own red twin — the same
zero-guard run with no override — refuses by name before any tile is built:
`band 50 of the RETURNED window [0, 52) is invisible to fH … 16 exactly-zero
(band, k) slot(s)`.

## 3. COLLISION 1 — THE ORTHO CAP ADMITS **ZERO** GUARDS ON THE DECOUPLED PARENT

The capacity print says `nb < 57.91`. The gate that actually decides crosses
four bands earlier than that, and it crosses at the FIRST guard. Decoupled
parent, ζ=52, μ=2628, nk=64, `rtol` untouched:

| `n_guard` | `nb_fh` | rank / nk·nb_fh | ortho `max\|C Cᴴ−I\|` | cap 1.0e-06 | Galerkin ζ-window residual |
|---|---|---|---|---|---|
| **0** | 52 | 3327 / 3328 | **3.444e-07** | **PASS** | 4.494e-07 |
| 1 | 53 | 3388 / 3392 | **1.326e-06** | REFUSE | 9.173e-07 |
| 2 | 54 | 3440 / 3456 | 2.328e-06 | REFUSE | 1.457e-06 |
| 3 | 55 | 3492 / 3520 | 2.288e-06 | REFUSE | 1.898e-06 |
| 4 | 56 | 3534 / 3584 | 1.869e-06 | REFUSE | 2.439e-06 |
| 8 | 60 | 3706 / 3840 | **3.467e-06** | REFUSE | 5.428e-06 |

The 0 and 8 rows reproduce the two numbers the previous lanes published
(3.444e-07 and 3.467e-06) to every digit, which is what makes the middle of the
table a measurement rather than four new numbers.

**So on THIS parent the two-window contract is unreachable at μ = 2628.** The
lever is centroids, and it is the same lever
`2026-08-11-zeta-window-refit-needs-psi-rank-not-mu-count.md` named: the rank
deficit grows 1 → 4 → 16 → 28 → 50 → 134 as the window widens, i.e. the basis
stops spanning immediately above nb = 52. It is NOT a conditioning problem and
`rtol` is not the lever (the ortho gate's own docstring measured that).

On `dp2628n20` (ζ=20, same 2628 centroids) there is room and to spare —
`nk·nb_fh = 1536` against `n_μ·n_s = 5256` — which is why §2's measurement
exists at all.

## 4. COLLISION 2 — THE ON-GRID CLIFF, MEASURED, AND IT IS OUTSIDE THE CERT'S OWN BRACKET

The documented two-sided warning is that a LARGER interp window corrupts
on-grid energies past a system-dependent cliff (MoS2/640c: ~1 meV at nband ≤
48, ~955 meV at nband 80). On this deck it does not cliff — it degrades
smoothly by about a factor of six — but the reference grade is 0.01 meV and
that is enough to cross it. Implied on-grid energy error, from the gate's own
measured conversion (on-grid max|Δε| ≈ 9.0e3 × ortho):

| `n_guard` | 0 | 1 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|---|
| implied on-grid \|Δε\| (meV) | **3.10e-03** | 1.19e-02 | 2.10e-02 | 2.06e-02 | 1.68e-02 | 3.12e-02 |
| vs the 0.01 meV reference grade | 3× INSIDE | 1.2× out | 2.1× out | 2.1× out | 1.7× out | 3.1× out |

**And on the parent where the window IS affordable there is no cliff at all.**
Directly measured rather than proxied on `dp2628n20` — recovered-vs-stored DFT
energies over the returned window at the coarse k, which is the quantity the
warning is about: **3.867e-10 meV at zero guards, 1.975e-10 meV at four**. It
IMPROVES, and both are eight orders inside the 0.01 meV grade. The `ortho`
there is 8.882e-15 → 8.549e-15 and the rank is full (1536/1536) on both sides.
So the cliff is not a property of guard bands; it is the capacity of §3 wearing
a different instrument.

**Read together with §3 this is a STOP on the decoupled parent**, and it is
the stop the dispatch pre-registered: guards there put the on-grid numbers
outside the cert's own bracket, and the ortho gate refuses them first anyway.

## 5. COLLISION 3 — THE TILE NULL, AND THE SIXTH WALL

`dp2628n20`, four guards, the same 7-coarse-q population the 1.409 zero-guard
number was measured on:

| q | refit vs stored `V_qmunu` |
|---|---|
| **(0,0,0) Γ** | **4.688e-06** |
| (0,2,2) | 1.135e+00 |
| (3,1,2) | 1.126e+00 |
| (2,2,2) | 1.156e+00 |
| (3,2,3) | 1.106e+00 |
| (1,2,2) | 1.165e+00 |
| (3,2,2) | 1.141e+00 |
| **worst** | **1.165e+00 — REFUSES against 5.0e-02** |

Γ moved four orders and **nothing else moved at all**. That is not a partial
fix; it is a second, independent defect that the first one was hiding, and the
previous lane had already seen its shadow: the fifth-wall row's own honesty
note records that its two zone-boundary points read **1.135 and 1.156 on BOTH
legs** — including `m_leg="stored"`, where the m-leg is the producer's own
stored ψ and there is no interpolation anywhere — and called them "a
mis-indexed reference, not a physical number". Those are the same two numbers.

**So this lane measured it, with the exact m-leg, against every stored tile.**
For each coarse q: compute `refit_vq(q)` with `m_leg="stored"` and then scan
ALL 64 stored tiles for the best match, plus the conjugate, transpose and
Hermitian variants at the gate's own slot, on both parents.

| parent | q | gate slot | negated-q slot | BEST of all 64 | 2nd best |
|---|---|---|---|---|---|
| `dp2628n20` | Γ | **3.695e-06** | 3.695e-06 | **3.695e-06** (j=0) | 9.461e-01 |
| `dp2628n20` | (2,2,2) | 1.156e+00 | 1.156e+00 | 1.002e+00 (j=63) | 1.013e+00 |
| `dp2628n20` | (1,2,2) | 1.165e+00 | 1.200e+00 | 1.014e+00 (j=48) | 1.026e+00 |
| `dp2628n20` | (0,2,2) | 1.135e+00 | 1.135e+00 | 1.002e+00 (j=15) | 1.014e+00 |
| `p2628n60z52` | Γ | **9.157e-06** | 9.157e-06 | **9.157e-06** (j=0) | 9.273e-01 |
| `p2628n60z52` | (2,2,2) | 1.237e+00 | 1.237e+00 | 1.018e+00 (j=63) | 1.031e+00 |
| `p2628n60z52` | (1,2,2) | 1.248e+00 | 1.320e+00 | 1.056e+00 (j=48) | 1.068e+00 |

Read the Γ rows first. They land on the two published formulation nulls —
3.695e-06 and 9.157e-06 — and the SECOND-best tile is 0.93–0.95 away, so at Γ
the identification is unambiguous and exact. At every finite q the best of all
sixty-four stored tiles is **1.00–1.06**, i.e. no stored tile is reproduced at
all, and conj / transpose / Hermitian at the gate slot change nothing.

**The sixth wall is therefore: at FINITE q, `refit_vq` does not compute any
stored `V_qmunu` tile, with the interpolation removed entirely.** It is not
the m-leg (this is the exact one), not the ζ solve (fixed and re-verified —
every leg logs `path=replicated_rank_truncate rcond=1.0e-10`), not the window
(§2), not a q-index order, not a q sign, and not a Hermitian convention. What
is left is inside `refit_vq`'s own finite-q machinery: the centroid winding
phase `e^{−2πi r_μ·q}`, the sphere `_sphere_millers(zx, qw)` / `flat_idx`
selection, or `v_on_set(qw, GS)` — all of which are exactly no-ops at q = 0,
which is precisely why Γ is clean and nothing else is. **That is the next
lane's object, and it is a one-file question.**

## 6. WHY NO PLOT, SAID PLAINly

The dispatch's condition was the tile null inside 5.0e-02. It reads 1.165 on
the parent where the contract is affordable and the contract is unaffordable
on the parent the curve would be OF (§3). So: no `.dat`, no `.png`, no cert,
and no bracket touched. The driver refused on its own; nothing had to be
withheld by hand.

## 7. GATES

One combined P=4 pytest leg per tree, `-G 4 -n 1`, over the refit/xbands
suites, `test_bse_setup_qchunk`, `test_refit_vq_shard_p4`, `test_layering`,
`test_env_grammar` and all seven `services/*/tests`:

| tree | result |
|---|---|
| branch, final HEAD | **17 failed / 1221 passed / 13 skipped / 1 xfailed** (254 s) |
| branch, mid-lane HEAD `3e586e7e` | 17 failed / 1220 passed / 13 skipped / 1 xfailed |
| base `0cd4c503` | **17 failed / 1203 passed / 13 skipped / 1 xfailed** (263 s) |

**The failure NAME sets are identical — empty set-diff — and 1221 − 1203 = 18
is exactly this branch's new cells**, which is also the collected-count check
(the mid-lane row is the same file one cell earlier, before the ψ-chunk cell
was added, and 1220 − 1203 = 17 there):
the base arm's suite list omits `tests/test_f_shoulder_two_window.py` because
the base tree does not have it, so the difference IS the file. Two of the
seventeen reds are `test_bse_setup_qchunk` ×2, which this lane added to the
list and which `KNOWN_FAILURES.md` rows 804/805 already carry with the exact
fingerprint this run reproduces (`_maxdiff = 1.3743988419548263`); the other
fifteen are the same fifteen the zsolve lane reported.

**Si deck at four processes: rc 0.**

**Default-deck A/B where the refit is NOT invoked** — the production decoupled
parent through `--vq-mode ongrid`, which exercises `initialize_wfns` and
`compute_wfns_fi` (where the new gate lives) and no refit at all, run on both
trees:

| | data bytes | md5 (comments stripped) |
|---|---|---|
| branch | 346 | `0e41fb06ea7a2fbc` |
| base | 346 | `0e41fb06ea7a2fbc` |

**Data-identical.** The only differing line in the whole file is `# input:`,
which names each arm's own workdir. The f-shoulder gate on that path reports
`[8, 16) of 52 (36 guard band(s) above): min_b min_k |f|/max|f| = 3.778457e-01,
0 exactly-zero slots` — i.e. the production BSE window is nowhere near the
shoulder, which is why the default path is untouched.

## 8. EVIDENCE

`/pscratch/sd/j/jackm/twowin_0811/EVIDENCE.md`.
