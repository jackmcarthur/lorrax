# The fifth wall named: `refit_vq` asks the htransform for the top of its own band window, and that is exactly where `f(ε)` is zero (2026-08-11)

**STATUS: DIAGNOSTIC ONLY. MECHANISM CONVICTED BY MEASUREMENT ON BOTH
PROBED PARENTS; NO FIX TAKEN, BECAUSE THE FIX IS NOT BOUNDED. NO SOURCE
FILE CHANGED, NO TOLERANCE MOVED, NO `.dat`, NO `.png`,
`LORRAX_FH_ORTHO_TOL` NEVER SET, NOTHING MERGED.**
Base `origin/main` `9bef1b13`, branch `lane/fifth-wall-mleg-diag-2026-08-11`
(this row is its only commit). Workspace `/pscratch/sd/j/jackm/mleg_0811/`,
sole writer; tree `/pscratch/sd/j/jackm/mleg_0811/tree` @ `9bef1b13`,
`dirty-count 0`, HEAD asserted in both legs. `zeta52_0811`'s two probed
parents (`dp2628n20`, `p2628n52`) were read and not touched; nothing was
written anywhere else.

This row answers the handoff left by
`2026-08-11-refit-zeta-solve-fixed-and-the-fifth-wall-is-the-gram.md` §4,
which measured that one leg — `compute_wfns_fi`'s ψ at wrap(k−q) — owns the
entire remaining tile-null residual, and named finding out *why* as the next
lane's job.

## 1. The mechanism, in one paragraph

`build_fH_R` builds the interpolating Hamiltonian as
`fH_k = Σ_n f(ε_n,k) c_n,k c_n,kᴴ`, where `f` is the htransform paper's
bandwidth-bound transform. `f` is **identically zero for ε ≥ shift**, and
`htransform._f_params_from_energies` sets

    shift := max_k ε[nb−1]

— the maximum over k of the **top band of `ctilde`'s own window**. So the top
band contributes **exactly nothing** to `fH` at the k where it attains that
maximum, and `f` vanishes to second order approaching it, so a whole shoulder
of bands below the top carries only a per-cent or less of `fH`'s weight. Those
eigenvector slots are therefore degenerate, or nearly so, with `fH`'s own
(rank − nb)-dimensional null space, and `jnp.linalg.eigh` fills them with
arbitrary directions out of that null space.

Every other caller in the tree knows this and keeps **guard bands** above the
window it returns. `bse.exciton_bands` computes `n_guard = nb_window − b_max`
explicitly and warns below four of them; `bse.bse_densify`'s own error text
calls the bands above the BSE window "conduction guard bands". The refit
m-leg is the one caller with **zero** guards:

```python
# src/bse/vq_interp.py, refit_vq
bundle = compute_wfns_fi(
    ctilde=rst["ctilde"], ..., band_window_fi=(0, nb), ...)
```

`nb` here is the **whole** `ctilde` window, because `refit_prepare` pins
`nb == zx["nb"]`: ζ′ has to be re-fitted on the producer's own ζ window, so
the refit's htransform window *is* the ζ window and there is nowhere for a
guard band to come from. The refit asks the htransform for precisely the
bands the htransform cannot represent.

## 2. Measurement 1 — the overlap, and it is NOT block-unitary

One combined P=4 leg per parent, four real processes, `mesh 2x2`, on-grid
q = Γ. The overlap is taken in the **α space**, where it is exact and carries
none of the Galerkin residual:

    O[k,m,n] = Σ_α ctilde[k,m,α] · conj(coeffs_fi[k,α,n])

`‖O[m,:]‖ = 1` means stored band `m` is fully inside the returned set — the
only thing the charge Gram needs, since the Gram over pair densities *is*
invariant under any unitary mixing of a **complete** m-band set (that is why
the Kramers gauge freedom has never mattered here). The controls first, and
they reproduce the previous row to every digit it published:

| parent | `m_leg=stored` | `m_leg=htransform` | bracket |
|---|---|---|---|
| `dp2628n20` (nb 20, rank 1280) | **3.6954e-06** | **1.2673e+00** | 5.0e-02 |
| `p2628n52`  (nb 52, rank 3328) | **9.1567e-06** | **1.1754e+00** | 5.0e-02 |

And the overlap:

| parent | `max_k ‖O Oᴴ − I‖` | bands recovered at 1.000000 | bands that collapse |
|---|---|---|---|
| `dp2628n20` | **9.4928e-01** | 0 … 15, at every k | **16, 17, 18, 19** — `min_k ‖O[m,:]‖` = 0.274 / 0.257 / 0.261 / 0.225 |
| `p2628n52`  | **9.9301e-01** | 0 … 49, at every k | **50, 51** — `min_k ‖O[m,:]‖` = 0.0836 / 0.0882 |

It is not a gauge, and it is not an interpolation *error* either: the bad
bands are not slightly wrong, they are **absent**, and only at some k. On
`dp2628n20`, `‖O[19,:]‖` reads `1.0000` at fifteen of the first sixteen k and
`0.3394` at k10 — which is `argmax_k ε[19]`, the k that *defines* `shift`.

The f-transform table, printed from the run's own `enk_sigma`, says why, and
it is the whole row in four lines:

    dp2628n20   a=1.119560 Ry  shift=1.27040772 Ry ( = max_k eps[19] )
      band |  min_k |f|/max|f|   max_k |f|/max|f|   n_k with f == 0
         0 |   7.195437e-01      1.000000e+00       0
        15 |   3.956010e-04      8.760386e-02       0
        16 |   0.000000e+00      1.979889e-02       3
        19 |   0.000000e+00      1.210574e-02       3

    p2628n52    a=1.229864 Ry  shift=2.80319195 Ry ( = max_k eps[51] )
        49 |   6.898657e-04      6.402771e-03       0
        50 |   0.000000e+00      5.761011e-03       8
        51 |   0.000000e+00      5.761011e-03       8

Read the last column. On `dp2628n20` **four** bands are exactly zeroed at
three k each; on `p2628n52`, two bands at eight k each. And note that it is
not only the top band: `f(ε) = 0` for **every state in the window whose
energy is at or above `max_k ε[nb−1]`**, which on a dispersive window is
several bands deep. The bands that survive are the ones whose `|f|` never
gets closer to zero than ~4e-4 of `max|f|` (band 15 on one parent, bands
48/49 on the other) — those come back at `1.000000` at every k.

`ctilde` orthonormality was 8.882e-15 and 3.444e-07 on the two parents,
against the untouched 1.0e-06 cap, so this is emphatically **not** the
failure mode that gate catches. The representation is fine; the request is
wrong.

The centroid-space overlap `|S̃|` is also printed and tells the same story
(0.16 for the collapsed bands against 0.6–0.7 for the good ones), but it is
the blunter instrument: ψ at centroids is not an orthonormal set, so even a
perfectly recovered band spreads over its Kramers partner and never reads
1.0. Quote the α-space number.

## 3. Measurement 2 — the rank shift is broadband, not a cut artefact

The dispatch asked whether the 195 moved directions concentrate at the
spectral cut (a tolerance-scale reshuffle, which `common/spectral_closure`'s
drop-block rule would own) or spread across the spectrum (genuinely different
pair densities). **They spread.** Both Gram spectra, on both parents:

| parent | leg | `lam_max` | `n_keep` (rcond 1e-10) |
|---|---|---|---|
| `dp2628n20` | stored | 4.15259934e-02 | 1072/2628 |
| `dp2628n20` | fi     | 4.11810759e-02 | 1267/2628 |
| `p2628n52`  | stored | 1.15463869e-01 | 1591/2628 |
| `p2628n52`  | fi     | 1.15194741e-01 | 1979/2628 |

The whole spectrum is lifted, hardest a long way **above** the cut. On
`dp2628n20`, direction i=536 goes 2.08e-08 → 4.85e-08 (2.3×) while the cut
sits at 1072; the decade census moves 55 → 97 directions into (1e-04, 1e-02]
and 86 → 158 into (1e-06, 1e-04], while the bottom decade drains 566 → 461.
`p2628n52` is the same shape, 425 → 165 and 110 → 4 in the two bottom
decades. A tolerance-scale reshuffle cannot do that; different densities can,
and these are different densities.

One more signature worth having on the record: the **stored** spectrum is
exactly Kramers-paired (i=1977/1978/1979 all read 2.4047e-12 on `p2628n52`)
and the interpolated one **splits those pairs** (2.412e-09 / 2.397e-09 /
2.377e-09). Whatever fills the collapsed slots is not a physical state.

## 4. What the top band alone is worth — the splice

To weigh the top band against the rest of the shoulder, the production
`m_leg="htransform"` tile was re-run with **only band nb−1** of the
interpolated bundle replaced by its exact Galerkin value `ctilde[:, nb−1, :]`
(everything else the interpolation's own; `compute_wfns_fi` monkeypatched at
the module attribute `refit_vq` imports it from, so the production code path
is untouched):

| parent | production | top band spliced | `m_leg=stored` | bracket |
|---|---|---|---|---|
| `dp2628n20` | 1.2673 | **0.9469** | 3.6954e-06 | 5.0e-02 |
| `p2628n52`  | 1.1754 | **1.1099** | 9.1567e-06 | 5.0e-02 |

So the top band is a **minority** of the error. Dropping the top band from
both legs of the Gram tells the same story: the `lam_max` gap does not close,
it *widens* (−0.83 % → −2.62 % on `dp2628n20`). The wall is the whole f→0
shoulder, not one band, and any fix has to clear the shoulder rather than
patch its top row.

## 5. Why no fix was taken

The dispatch's condition was a bounded fix — a knob, a normalization, a
multiplet completion. This is none of the three.

* **There is no knob.** `a_band_index` moves `a`, not `shift`. `shift` is
  `max_k ε[nb−1]` with no override anywhere, and it is not a tuning
  parameter: `f ≡ 0` above the window is what makes `fH` finite-rank and the
  interpolation well-posed. Moving it is re-deriving the htransform, on a
  function every other consumer and every frozen reference depends on.
* **It is not a multiplet completion.** Every band window here is already
  degeneracy-clean (`edge 52 min gap 6.87 meV`, and `strict` throughout); the
  set that is incomplete is not a degenerate multiplet but the set of bands
  `fH` can represent at all.
* **The real fix is a two-window contract, and that is a design change.**
  The refit's fH window has to become a strict **superset** of the ζ-fit
  window — four or so guard bands, judging by where `min_k |f|/max|f|` stops
  reaching zero on these two parents — while ζ′ stays fitted on exactly the
  producer's window. That splits `refit_prepare`'s single window in two and
  moves its `nb == zx["nb"]` identity, its `band_range` refusal, the ψ stream
  and `B_full`; it needs the deck to load more bands; it tightens the
  Galerkin capacity bound `n_μ·n_s ≥ nk·nb_wide`; and it runs into the
  documented two-sided warning that a *larger* interp window corrupts on-grid
  energies past a system-dependent cliff (`exciton_bands`, MoS2/640c: ~1 meV
  at nband ≤ 48, ~955 meV at nband 80). That is a change with its own
  certification, not a knob, so this lane stops here with the mechanism
  named.

## 6. The certification question, stated plainly so nobody quietly "fixes" it

**`refit_ongrid_null` must NOT be re-pointed at `m_leg="stored"`.** The
gate's whole job is to bound the OFF-grid refit by an on-grid proxy, and the
only reason an on-grid q can stand in for an off-grid one is that the *same*
machinery computes both. The off-grid m-leg is `compute_wfns_fi` by
construction — there are no stored wavefunctions at an off-grid q, which is
why the refit exists. A cert that takes the m-leg from `psi_full_y` skips the
interpolation the off-grid path depends on, so it bounds nothing about it: it
would pass at 3.7e-06 on a path whose real error is order 1, which is exactly
the "certify where consumed" failure the preamble's measurement rule 6
records. `m_leg="stored"` is a **formulation null** — its own docstring says
so — and it is a diagnostic, never a gate. The 3.7e-06 in this row is
evidence that the fit conventions, the solve, the window and the Coulomb
pairing are right; it is not evidence that any tile is right.

## 7. Splash radius — two other callers with the same shape

Neither was measured here; both are named because they are the same request
against the same function.

* **`bandstructure.htransform`'s `get_centroids_fi` path** takes
  `b_max = int(params["wfn_fi_max"]) or int(ctilde.shape[1])` — i.e. it
  **defaults to the full band count, zero guard bands**, the exact
  configuration this row convicts.
* **`bse.bse_densify`** checks only `b_max > nb_window` and so *permits*
  `b_max == nb_window`; a deck whose `nband` equals `nval + ncond` has no
  conduction guard and lands in the same place. Its error text already tells
  the user to load guard bands, but nothing enforces it.

A cheap, bounded, and separate follow-up would be to make `compute_wfns_fi`
**report** `min_k |f(ε_b)| / max|f|` over the returned window and refuse (or
at least announce) when a returned band's minimum is zero. That is a gate in
the one function both consumers pass through, it is the same shape as the
`ctilde` orthonormality gate that already lives there, and it would have
turned this five-lane hunt into one line of a startup banner. It is written
down here as a proposal, not taken, because this lane's dispatch was bounded
to the diagnosis.

## 8. Evidence

`/pscratch/sd/j/jackm/mleg_0811/EVIDENCE.md`; logs
`/pscratch/sd/j/jackm/mleg_0811/_logs/f2628n20.log` (exit 0, 44 s) and
`.../fp2628n52.log` (exit 0, 68 s); probe
`/pscratch/sd/j/jackm/mleg_0811/probe_fifth.py`; manifest `m_fifth.jsonl`,
two legs at `-N 2 -G 4 -n 4 -P 2`, own allocation 56640776, placed one per
node (`nid001140`, `nid001141`) — the `-P 4`-on-one-node coordination-service
failure the previous lane recorded as void did not recur. Both legs printed,
from inside the step, `git HEAD 9bef1b13`, `git dirty-count 0`,
`The run's device mesh is 2x2 over axes ('x', 'y')` and
`[probe] jax.device_count()=4 process_count()=4 mesh_xy.shape={'x': 2, 'y': 2}`.
The fine arm's separate `RESOURCE_EXHAUSTED` wall was not touched.
