# FIXED — the downfold built its transfer at −q and applied it at +q (2026-08-11)

**STATUS: ROOT CAUSE FOUND AND FIXED**, branch
`diag/wedge-child-composition-2026-08-10`, pushed, **NOT merged**.  This closes
the "mechanism is not identified" state that
`tests/known_failures/2026-08-10-downfold-orbit-economics-owner-row.md` left
open, and it convicts a two-day-old diff rather than any established machinery.

## The defect, in one paragraph

`gw/downfold.py::pair_density_gram` builds the pair-density Gram by calling the
ISDF fit's own kernel, `isdf.core.c_q_from_psi_sm`. That kernel returns the
Gram labelled by **−q**. Its own caller, the ISDF fit, is self-consistent in
that labelling and is correct; the downfold is the caller that has to agree
with something else — the restart tensors' q axis — because it multiplies this
Gram's transfer into `V_qmunu[q]`. Nobody checked the two against each other.
The transfer the downfold built was therefore `conj(T)`, the transfer belonging
to `−q`, and the congruence applied it to the tensor at `+q`, at every q with
`−q ≢ q`. On a 4×4×4 grid that is **56 of 64 q**. The fix is a q-axis relabel
in `pair_density_gram` — `negate_q_index`, a pure permutation, exact, and
correct for symmetric and asymmetric windows alike. `c_q_from_psi_sm` is NOT
touched, and no other caller of it exists.

Introduced by `0dd61f20` (2026-08-09), the downfold's own birth commit.
`pair_density_gram` has no caller outside `gw/downfold*.py` and its tests, so
the blast radius is the downfold and everything downstream of a downfolded
child.

## The convicting numbers

Deck `si_bse_debug`, μ_L = 480, μ_S = 168, window 0:20 symmetric,
`downfold_rcond = 1.1e-6`. Read against the child the orbit-floor lane actually
shipped, `/pscratch/sd/j/jackm/orbitfloor_0810/child_floor`. Evidence:
`/pscratch/sd/j/jackm/wedgechild_0811/`.

| measurement | value |
|---|---|
| the shipped child, reproduced by `conj(T) V Tᵀ` (the −q transfer) | **7.541e-05** on all 64 q (`W0`: 1.682e-04) |
| the shipped child, reproduced by `T V T^H` (the +q transfer) | **9.947e-01** — order one |
| wedge-composition covariance of the shipped child | **1.170e+00** / `W0` **1.241e+00** — the driver's own gate, reproduced to the digit by an independent host implementation |
| wedge-composition covariance of `T V T^H`, same deck/keep/rcond/tables | **3.721e-08** / `W0` **3.001e-08** |
| q unaffected | the **8** with `−q ≡ q`: stars `q_irr` = (0,0,0), (0,0,½), (0,½,½) |

The host chain reproduces the driver's transfer solve on every number the
driver prints — retained rank per q **elementwise identical** (120/123/124),
`eigen_rank_kept` 122 of 168 at q=0, `kappa_eff` 8.94e+05 against the driver's
8.945e+05 — so it is the same solve, and the only thing that differs is which
q the Gram belongs to.

Confirmed a second, independent way, with no symmetry machinery involved at
all: on an **asymmetric** window (the only shape that can see a leg mapping)
`c_q_from_psi_sm`'s output matches a dense double loop over the definition
**only after the q axis is reversed**.

## What this REFUTES, and each was measured, not argued

Every hypothesis that was live when this lane opened is dead, and the
measurements are cheap enough that they should be re-run rather than trusted.

* **The pair-density Gram is NOT non-covariant.** `S_LL[q]` against
  `U S_LL[i(q)] U†`, host-side on the parent's own stored tables: **3.109e-10**,
  and `S_SS` and `S_cross` the same. The band projector `P(k)` is covariant at
  **2.069e-15**. This was the decisive measurement the previous lane named, and
  it came back the opposite way to the steer.
* **The band window is NOT cutting a multiplet.** `[0,20)` has a minimum
  boundary gap of **1.678e-02 Ry** over all 64 k — 0 of 64 cut. (It is `[0,24)`
  that would cut, 7 of 64.)
* **The spectral-closure guard is INERT on this run**, so neither it nor its
  drop-block flip can be the mechanism: `df_floor.log:195` reads "the cut falls
  in a gap on all 64 q", minimum relative gap **1.374e-01**.
* **The unfold monomials are unitary.** All 96 op rows of `sym_perm` are
  bijections, on the parent (480) and on the child (168).
* **The child's unfold tables are NOT a bare permutation and are NOT rebuilt
  per q.** The tables stored beside the child are **bit-identical** to the
  parent's `sym_perm` and `L_table` restricted to `keep`, phases included.
* **The conditioning is not the mechanism either**, though it is close enough
  to be worth stating: the pinv's amplification of the Gram's 3.1e-10
  covariance residual accounts for the 3.7e-08 floor and nothing beyond it,
  and it scales with `rcond` exactly as a pinv perturbation should
  (`kappa_eff` 8.9e5 → 3.7e-08; 9.9e1 → 1.2e-09).

## Why a suite with a "matches the definition" cell stayed green

`tests/test_downfold.py::_dense_gram` is described in its own docstring as
"the slow honest way, straight off the definition". It carried **the same
q-sign error as the code it checks**: its prose said the left (m) window sits
at `k−q`, its code put it at `k+q`. Three cells compared the implementation
against it and agreed, because both were at `−q`.

This is the failure mode `AGENT_PREAMBLE`'s measurement discipline exists to
refuse, one level up from where it is usually caught: a reference written from
the same hand as the code is not a reference. The repair is a cell that could
not have been written that way —
`test_the_gram_is_labelled_by_PLUS_q_analytically` — which uses a **closed
form**: with one band and `psi(k, r_mu) = exp(2 pi i k R_mu)` the pair density
at momentum transfer q is `exp(2 pi i q R_mu)`, so
`S(q)[mu,nu] = nk * exp(2 pi i q (R_mu - R_nu))` exactly, and that is not
invariant under `q -> -q`. `_dense_gram` is corrected too, and the closed form
now arbitrates between it and the kernel rather than the two arbitrating each
other.

The red twin is the shipped call — the kernel's output with no relabel — and it
is held to the *shape* of the failure and not merely its size: it must agree
EXACTLY at the self-inverse q and miss by order one everywhere else. That
signature is what named the mechanism in the first place, and it is what
distinguishes it from a missing unfold phase, which would have tracked the
umklapp wrap count instead.

## What this poisons, and what it does NOT settle

**Every downfolded child produced between 2026-08-09 and this fix is wrong at
56 of 64 q**, so everything measured off one is void. In particular the
orbit-floored-vs-point-picked accuracy comparison recorded in the orbit
economics row — `eps_W` medians 4.319e-02 vs 1.056e-02, the point-picked arm's
**−348.6 meV** exciton error, the orbit-floored arm's FEAST non-convergence and
its +47 % Lanczos `E_max` — was taken with **both arms** carrying this defect.
It measured two flavours of the same bug and cannot be read as a statement
about orbit flooring. **That comparison was NOT re-run by this lane** (it is a
follow-on decision, and it is now cheap to re-take correctly).

It does explain the orbit-floored arm's non-convergence at least as well as the
direction count did, and this row does not claim to have separated them: a
transfer built at the wrong q is not a small perturbation, and an arm whose
`eps_W` is 4× the other's is the arm with less headroom to absorb it.

What this does **not** touch: the orbit floor itself, the ruling behind it, the
spectral-closure guard, `child_unfold_tables`, or `c_q_from_psi_sm`. The
economics finding — that an orbit-closed basis on this deck buys 0.73
independent directions per point against 0.92 — is a statement about the
selection Gram at q = 0, where `−q ≡ q`, so it is untouched by this defect and
stands.

## Gates

**WSL CPU, 1×1, both sides run, worktree pin proven by `__file__` before
measuring.** Base `origin/main` `7a7fe7a8`; branch
`diag/wedge-child-composition-2026-08-10`.

*(filled in below by the A/B and the P=4 leg)*
Suites: `tests/test_downfold.py` + `tests/test_exciton_bands_downfold_dropin.py`
+ `tests/test_spectral_closure.py` + `tests/test_layering.py`.

| arm | HEAD | result | collected |
|---|---|---|---|
| base | `7a7fe7a8` | **184 passed, 10 skipped** | 194 |
| branch | this tree | **187 passed, 10 skipped** | 197 |

Delta is exactly the three new cells — the closed-form q-labelling gate, its
red twin, and the relabel's involution/fixed-point check — and **zero
regressions**. The 10 skips are one reason on both sides, the documented WSL
`liblorrax_ffi_host.so` condition. The count is checked rather than read off a
green bar, and it caught a real defect in this lane's own method: the first
base arm was taken in a worktree that turned out to sit at `ad8d342f`, ten
commits behind, and reported a +17 delta. A stale arm is not a control
(`AGENT_PREAMBLE`, measurement discipline rule 4); the numbers above are from a
worktree checked out at `7a7fe7a8` with `git rev-parse HEAD` printed.
