# `use_band_extrapolation`: default-on, hard-gated, and driving the SC iteration

**Date** 2026-08-16 · **Branch** `feat/use-band-extrapolation-sc-2026-08-16`
(cut from `feat/band-extrapolation-sampling-2026-08-15` @ `97f6f544`) ·
**Worktree** `/pscratch/sd/j/jackm/wt_bandextrap_20260815` ·
**Runs** `sandbox:runs/Si/51_band_extrap_sc_wiring_20260816/`

---

## 1. What now refuses that did not before

The blast radius is **much smaller than the raw arithmetic suggests**, and the
reason is the per-stage auto-disable in §5 rather than anything about band
counts. Sweeping every committed deck in `tests/` against the new gate:

| deck | `nband` | `n_occ` | `n_cond` | `compute_mode` | verdict |
|---|---|---|---|---|---|
| `regression/gnppm_debug/gnppm_test.in` | 46 | 26 | 20 | **gn_ppm** | **WOULD REFUSE** -> key set `false` |
| `regression/bispinor_debug/bispinor_test.in` | 32 | 26 | 6 | **gn_ppm** | **WOULD REFUSE** -> key set `false` |
| `regression/cohsex_debug/cohsex_test.in` | 40 | 26 | 14 | cohsex | auto-disabled, runs |
| `regression/cohsex_debug/cohsex_test_minimax_selfcheck.in` | 40 | 26 | 14 | cohsex | auto-disabled, runs |
| `regression/gnppm_debug/cohsex_ibz_test.in` | 46 | 26 | 20 | cohsex | auto-disabled, runs |
| `regression/si_cohsex_debug/cohsex_si_test.in` | 60 | 8 | 52 | cohsex | clears anyway |
| `regression/si_cohsex_debug/cohsex_si_fast.in` | 20 | 8 | 12 | cohsex | clears anyway |
| `regression/si_bse_debug/bse_si_test.in` | 60 | 8 | 52 | cohsex | clears anyway |
| `regression/hbn_cohsex_debug/cohsex_hbn_test.in` | 80 | 16 | 64 | cohsex | clears anyway |
| `archive/projects/test_isdf/cohsex_test.in` | 110 | 26 | 84 | cohsex | clears anyway |

`n_occ` is **not a deck key** - it is `b2 - b0 = wfn.nelec`, read from the
fixture WFN (`src/common/meta.py:82`). So five decks are under `2*n_occ`, and
**exactly two of them refuse**: the two GN-PPM ones. The three COHSEX decks
below the threshold never reach the gate, because a defaulted key
auto-disables on a non-PPM mode.

**Both refusing decks were fixed by setting `use_band_extrapolation = false`,
not by raising `nband`**, and that was a deliberate choice in both cases:

* `gnppm_test.in` is a frozen-reference fixture (`sigma_diag_gnppm_ref.dat`,
  `eqp_rotations_fixedpoint_ref.npy`, `test_invariance_gates.py`, the
  one-shot == SC-iteration-1 gate). Raising `nband` 46 -> 52 changes every
  number it exists to compare, which would retire the comparison rather than
  preserve it. What it measures is GN-PPM Sigma against a pin, not band
  convergence.
* `bispinor_test.in` **cannot** have its band count raised: its fixture
  `WFN.h5` carries only 34 bands against the 52 the gate wants. Setting the
  key false is the only way it runs at all.

A deck that wants to exercise the extrapolation should be a new variant with
enough bands - which is what `runs/.../11_gnppm_sc_extrap_on/` is.

**Also relaxed, not tightened, in one place.** The shipped gate refused on
`n_cond <= n_occ`; the owner asked for `>=`. The equality case
`n_cond == n_occ` now **runs**. The "counts collapsed" refusal further down
`plan_band_brackets` quoted `2*n_occ + 1` as its floor; it now quotes
`2*n_occ`, so the two refusals name one threshold and a deck that satisfies
the message it was given cannot land on the other one with a different
number.

---

## 2. The tolerance-vs-uncertainty ruling - **WARN, loudly; do not refuse**

This was the call flagged as the most important robustness point, so the
reasoning is given in full. Two independent arguments, either sufficient.

**(1) A refusal would fire on the shipped default.** `sc_tol_ev` defaults to
`1.0e-4` eV = **0.1 meV** (`gw_config._DEFAULTS`). `use_band_extrapolation`
now defaults **TRUE**. The p90 bar is 15 % of the applied correction, which on
the calibration deck runs to **tens of meV**. So "tolerance inside the bar" is
not an edge case - it is the *default state of the code*, by two to three
orders of magnitude. A gate that refuses the configuration the code ships with
is not a safety property; it is a build that cannot run, and it would be
switched off by the first operator who needed a number.

**(2) The two quantities answer different questions, so "inside" is not an
inconsistency to refuse - it is a misreading to prevent.** This is where I
push back on the framing in the task. `sc_tol_ev` bounds the
**iteration-to-iteration displacement** of E_nk: *has the fixed point been
reached?* The extrapolation bar is a **systematic uncertainty on the absolute
Sigma_c**: *where is the fixed point?* A systematic, iteration-independent bias
in Sigma_c does not stop the loop converging to 0.1 meV - it **moves** the
fixed point. So a run reporting "converged, RMS dE = 8e-5 eV" is making a
*true* statement about the iteration and would be making a *false* one about
the accuracy of E_nk. Refusing would answer a question about accuracy by
breaking a mechanism about convergence. The defensible response is to make the
two impossible to confuse, every iteration, in the log.

Implemented as `band_extrapolation.sc_tolerance_ruling`, printed once per SC
iteration beside the fit. When inside the bar it prints a `***`-marked block
carrying: `sc_tol_ev` in meV; the p90 bar as **median over states and max**
(the max alone is set by the top of the QP window and describes that state,
not the calculation); the ratio; the statement that these are not the same
number; and an explicit instruction not to quote the SC residual as the
accuracy of E_nk.

**Measured on the live SC run (8.3), the ruling's premise holds
quantitatively:** `sc_tol_ev` = 0.1 meV against a p90 bar of **184.3 meV
median / 704.9 meV max** - the loop is being asked to converge **1,843x
tighter** than the uncertainty on the quantity it is converging. That block
printed on every one of the 7 iteration-map calls.

**And the second failure mode is real, not hypothetical.** I said above that
what *would* justify a refusal is the correction *wobbling between iterations*
by more than `tol_ev` - iteration noise injected into the fixed-point map,
which can prevent convergence rather than relocate it. The run measures it:
the extrapolation's correction to the **gap** ranges over
**-0.084 ... -0.202 eV** across the seven calls, a **118 meV swing, i.e.
~1,180x `sc_tol_ev`**. So the loop cannot reach 0.1 meV until the correction
itself settles, and the tolerance is unreachable for a second, independent
reason. A cross-iteration wobble check is the one diagnostic that could
honestly gate this, and `sc_tolerance_ruling` cannot compute it from a single
fit. **Registered as open work, not implemented.**

---

## 3. The self-consistency coupling

**Extrapolate Sigma, then diagonalize** - implemented, and the implementation
is what makes the Hermiticity argument exact rather than approximate.

Ordinary least squares of `S(N) = S_inf + A/N` is **linear in the
observations**, so its intercept is a fixed affine combination of the three
cumulative bracket sums whose coefficients depend only on the band *counts*:

```
c_i = 1/n - xbar*(x_i - xbar)/S_xx ,   x_i = 1/N_i ,   sum(c_i) = 1
```

`band_extrapolation.extrapolation_weights` returns exactly these, as
`float64`. Three consequences, all tested:

* **`c` is real**, so `sum_b c_b S_b` of Hermitian `S_b` is Hermitian
  *elementwise and bitwise*: `S_b[j,i]` is exactly `conj(S_b[i,j])` out of the
  kernel, a real scalar multiply commutes with conjugation exactly in IEEE,
  and the reduction is over the **leading** axis so its order is identical for
  `(i,j)` and `(j,i)`. Gated by
  `test_extrapolated_sigma_is_hermitian_to_machine_precision`, which asserts
  `np.array_equal` against the conjugate transpose - **bitwise, not
  `allclose`**.
* **`sum(c_i) = 1`**, so a Sigma that does not depend on band count passes
  through unchanged rather than being rescaled.
* **Pad-band inertness survives**: a real-weighted sum of exact zeros is an
  exact zero, so rCROP's bit-for-bit pad check is unaffected
  (`test_extrapolation_is_pad_band_inert`).

The rejected order is tested too:
`test_eigenvalue_extrapolation_is_not_the_same_operation` asserts that
extrapolating each bracket's *spectrum* gives a **different** answer, so the
ruling is a real choice and cannot be quietly inverted. Eigenvalues are not
linear in the matrix, so an extrapolated spectrum is the spectrum of no
Hamiltonian and the next iteration's eigenvectors would belong to a different
operator than its energies.

**One estimator, two consumers.** The weights (which drive E_nk) and
`fit_band_extrapolation` (which produces the logged numbers) are pinned
together by `test_weights_reproduce_the_fit_intercept`, so the run cannot
report one correction and apply another.

**Two diagonalizations per iteration, not four.** The pipeline now returns the
un-extrapolated N3 body cube alongside the extrapolated one; the finalizer
builds a second QSGW matrix from it through the *same* head, omega grid, E_qp
and clipping policy, and `gw_iteration_map` assembles its Hamiltonian through
the same k-select -> rotate-to-DFT -> V_H path before diagonalizing **both
with `eigvalsh`** (eigenvalues only - nothing feeds back). The log then
reports VBM, CBM, gap and the over-all-states mean/RMS/max shift, plus the
fraction of states that moved **down** - the 1/N form is documented to
undershoot one-signed against a measured S(508), so a *mixed* sign is a
statement about the deck.

The comparison is made on the **pre-partition** Hamiltonians deliberately:
`apply_band_partition` only alters non-protected off-diagonals and
out-of-range diagonals, so on the QP window the pair differs in exactly one
thing - the band-sum tail in Sigma_c - rather than also differing in a scissor
refitted separately on each arm.

**rCROP checked.** It is a generic fixed-point accelerator on `H`: it
re-Hermitises before feeding the map (`sc_iteration.py:1831`), zero-pads bands
to a divisor, and converts the tolerance from per-band RMS eV to an L2
residual. It reads no band count and makes no assumption that the input is a
raw full-band value. The only two properties it needs from the new input -
exact Hermiticity and pad-band inertness - are the two proven above.

---

## 4. The key and its deprecated alias

`use_band_extrapolation`, **default `True`** (`USE_BAND_EXTRAPOLATION_DEFAULT`).
`sigma_band_extrapolation` is kept as a transitional alias and marked
deprecated in `docs/input_reference.md`.

Both are **tri-state** (`_NULLABLE_BOOL`, default `None` = "no deck named
it"), because the resolver has to distinguish *the deck said false* from *the
deck said nothing*. That distinction is load-bearing twice: it decides whether
the default stands, and it decides whether a non-PPM mode refuses or
auto-disables.

`resolve_band_extrapolation` returns `(enabled, explicit)` and **refuses by
name** when both spellings are named and disagree - no winner is picked,
because whichever precedence were chosen, half the decks that hit it would
silently get the other one. Naming both with the same value is accepted.

---

## 5. Interaction with static modes - reconciling two gates instead of fighting them

**The premise as briefed does not hold: `sc_stage_N_type` does not exist.**
`grep -rn "sc_stage"` over the checkout returns one unrelated prose hit, and
`git log --all -S"sc_stage"` is **empty** - it has never existed on any
branch. The staging axis that does exist is `compute_mode`, one per run. I
mapped "per stage" onto that and recorded the discrepancy in
`KNOWN_SANDBOX_ERRORS.md`.

`sigma_dispatch.py:554` already refused non-PPM modes when the key was set.
With the key defaulting on, that refusal would have killed every COHSEX / MPA
/ X_ONLY deck in the tree. The reconciliation splits on **provenance**, which
is the only thing that distinguishes the two situations:

* **explicitly named + non-PPM -> REFUSE.** The operator wrote the knob down;
  silently doing nothing with it is how a green A/B measures nothing.
* **defaulted + non-PPM -> DISABLE, and say so.** A loud log line carrying the
  measurement that justifies it (static COHSEX **94.9 -> 288.2 meV** MAE as
  nband goes 60 -> 124, anti-converging, ~340 meV past the exact answer,
  against GN-PPM improving **171.3 -> 32.8**). It is printed at the Sigma seam
  every iteration rather than once at startup, so it sits beside the stage it
  describes.

The physics guard is intact in both branches - no static-mode Sigma is ever
extrapolated. What changed is who gets refused. Both are tested.

---

## 6. Verification - scope, and what each check does *not* cover

### 6.1 Test suites (`-rs`, skip counts reported as required)

| invocation | result |
|---|---|
| **bare** `lx run ... pytest test_band_extrapolation.py test_band_extrapolation_star_covariance.py -q -rs` (baseline, pre-change) | **18 passed, 11 skipped** |
| **with h5py overlay**, same two files, pre-change | **29 passed, 0 skipped** |
| **with h5py overlay**, post-change, + `test_sigma_result_basis.py` + `test_qp_solver_config.py` | **75 passed, 0 skipped, 0 failed** |

**The documented "8 tests skip silently" understates it by three.** The 11
skips are **8 h5py** (1 h5 payload, 1 star-extraction registration, 6 in the
whole star-covariance file) **plus 3 FFT-FFI** (`FfiLibraryNotBuilt`), and the
FFI three include `test_brackets_partition_the_band_sum` - the gate the module
docstring calls *the* load-bearing one. So bare `lx run` drops both the
persistence layer and the partition gate.

**The invocation that sees h5py** (recorded in `KNOWN_SANDBOX_ERRORS.md`):

```
LX_BASE_MODULE=lorrax_J070 lx run -G 1 -n 1 bash -lc \
  'source runs/Si/51_band_extrap_sc_wiring_20260816/env_prelude.sh; cd <checkout>; python3 -m pytest ... -q -rs'
```

`PROV h5py 3.16.0 /global/cfs/cdirs/m4598/jackm/at_risk_artifacts_2026-08-15/h5py_site/h5py/__init__.py`.
`LX_BASE_MODULE=lorrax_J070` is **not optional and its omission is not
obvious** - without it the driver refuses at `runtime.jax_support` with
`got jax 0.5.3.dev20260816 ... want jax >= 0.7.0`, a message that names jax and
says nothing about the prelude. It cost one wasted 4-GPU launch this session.
**No h5py code was touched.**

### 6.2 Bit-identity of the default-off path - **proven, not asserted**

Two runs of the *same committed deck* (`gnppm_debug/gnppm_test.in`, GN-PPM,
one-shot), differing only in the presence of `use_band_extrapolation = false`:

* arm A - this branch, deck carrying the new key set false (`12_gnppm_off/`)
* arm B - a clean `97f6f544` worktree, unmodified deck, feature at its old
  default (`13_gnppm_baseline/`)

Every artifact either run produced, `md5sum`:

| artifact | md5 | |
|---|---|---|
| `eqp0.dat` | `a68727c5000858ab2949ebde5205b45d` | IDENTICAL |
| `eqp1.dat` | `91a1e9ec78fbe65732ac242c17701b80` | IDENTICAL |
| `eqp_g0w0.dat` | `466a1383fe2f296e64ac649903fe6cd4` | IDENTICAL |
| `sigma_diag_gnppm_test.dat` | `5a65293036acafac442497eaa3c60c45` | IDENTICAL |
| `sigma_freq_debug.dat` | `fb6225ca708d17d349e3795ab41a9f4b` | IDENTICAL |
| `qp_wfn_rotations.h5` | `1f74c8b90f4392c5d31b45b3b77c911e` | IDENTICAL |
| `sigma_mnk.h5` (19.5 MB) | `47ec0efa974ec3dd5acfde3e569b5267` | IDENTICAL |
| `WFN_qp.h5` (46.5 MB) | `b5ee7a21cd0d9a53542080bd28fa19a2` | IDENTICAL |

**8/8 byte-identical, including both HDF5 files.** Both arms exit 0.

*Scope of this check:* one deck, one mode (GN-PPM), one-shot, P=1. It does not
cover the SC path with the key off, MPA, or P>1.

### 6.3 `fastloop`

**`fastloop` does not exist on Perlmutter** - `run_fastloop.sbatch` is
hard-wired to TACC (`-p development`, `-A PHY25006`, `/scratch2/...`,
`tacc-apptainer`). It was **not run**, and this is not a claim that the gate
passed. What was run instead is 6.1 and 6.2 above.

---

## 7. Defects found and registered (not fixed - scope)

Three rows added to `KNOWN_LORRAX_ISSUES.md`, one section to
`KNOWN_SANDBOX_ERRORS.md`:

1. **`ppm_tau_kernel.py:464` - the bracketed G build slices the wrong axis of
   a 3-D mask.** `mask_A[:, lo:hi]` is the band axis only when the mask is 2-D
   `(nk, nb)`; on a **1x1 processor mesh** it is `(1, nk, nb)` and the slice
   cuts *nk*. `greens_function_kernel.py:145-151` already carries a reshape
   workaround whose comment names this exact failure ("crashed GN-PPM on 1
   GPU"), but it runs *inside* `build_G_tau`, i.e. **after** the corrupting
   slice. **PRE-EXISTING**: reproduced identically on a clean `97f6f544`
   worktree with an unmodified deck, and it bites the **single-bracket** path
   too, so it is not specific to extrapolation. Observed as
   `cannot reshape (1, 9, 52) into (9, 42)` and `(1, 20, 20) -> (64, 20)`.
   Workaround used here: run on a 2x2 mesh.
2. **Static-CH contamination inside a GN-PPM Sigma.** The whole justification
   for the PPM-only guard is that the 1/N limit is wrong for a static Coulomb
   hole - but `ppm_invalid_mode = "static_limit"`, *the shipping default*,
   injects an analytic static-COHSEX term for every invalid pole, **inside the
   band sum being extrapolated**. Measured live on the MoS2 deck this session:
   **1.73 % invalid modes (22944/1327104), `max|Sigma_static| = 0.2358 eV`**.
   The mode-level guard cannot see this because the mode *is* `gn_ppm`.
   Unquantified - nobody has measured whether those states' extrapolation
   error carries the static arm's sign.
3. **`test_band_extrapolation_star_covariance.py:90-96` absorbs a refusal as a
   pass.** `BandExtrapolationRefused` subclasses `ValueError`, which that
   test's except clause accepts as a pass - so every parametrised arm could be
   refusing outright and the suite would stay green. The 2026-08-16 gate change
   passed this test both before *and* after, for that reason.

---

## 8. The end-to-end SC run

`11_gnppm_sc_extrap_on/` is a variant of the GN-PPM deck at **`nband = 52`**,
`qp_solver = self_consistent`, key left at its new default. `n_occ = 26`, so
`n_cond = 26 == n_occ`: **this deck sits exactly on the boundary the
relaxation opened**, and would have refused before 2026-08-16. It is the
tightest deck the owner's rule admits.

Ran **P=4 (2x2 mesh), exit 0 in 37 s**, 3 rCROP iterations = 7
`gw_iteration_map` calls. Log preserved at
`runs/Si/51_band_extrap_sc_wiring_20260816/11_gnppm_sc_extrap_on/run_sc_extrap_on_P4.log`,
with per-iteration `eqp0_iter000{0..6}.dat` snapshots beside it.
P=1 is blocked by the pre-existing defect (1) in 7; **2x2 is the workaround,
not a fix.**

### 8.1 The extrapolation is live and driving

```
Sigma_c band extrapolation: ON - 3 disjoint band brackets ((0, 42), (42, 46), (46, 52))
against ONE W(tau) per tau; band counts (42, 46, 52) (requested (42, 47, 52))
```

Cut 47 snapped down to 46 on a degenerate boundary, as designed. The
`[driving]` line confirming that E_nk is built from `S_inf` rather than
`S(N3)` printed on all 7 calls.

### 8.2 The eqp-level correction, side by side (iteration 0)

```
-- band-extrapolation effect on E_nk, iteration 0 (two eigvalsh: extrapolated vs N3) --
   VBM   S(N3) =   -4.438460  ->  S_inf =   -5.787895 eV   (-1.349435)
   CBM   S(N3) =   -1.789880  ->  S_inf =   -3.237087 eV   (-1.447207)
   gap   S(N3) =   +2.648580  ->  S_inf =   +2.550808 eV   (-0.097772)
   over all (k, band): mean -1.591455  RMS 1.893143  max |dE| 4.352440 eV
   100.0 % of states moved DOWN.
```

**100 % of states moved down at every iteration**, which is exactly the
one-signed undershoot the module docstring documents for the 1/N form against
a measured S(508). The sign diagnostic behaved as designed.

Per-iteration, showing both the size and the *stability* of the correction:

| iter | gap S(N3) -> S_inf | delta | mean dE | max abs dE |
|---|---|---|---|---|
| 0 | +2.648580 -> +2.550808 | -0.097772 | -1.591455 | 4.352440 |
| 1 | -2.979594 -> -3.082004 | -0.102410 | -1.665098 | 4.115041 |
| 2 | +1.713470 -> +1.511132 | **-0.202338** | -1.600670 | 4.022150 |
| 3 | +2.445561 -> +2.361698 | **-0.083862** | -1.520869 | 3.884821 |
| 4 | +2.162512 -> +2.016807 | -0.145704 | -1.523413 | 3.878044 |
| 5 | +1.857219 -> +1.760799 | -0.096420 | -1.586159 | 4.082644 |
| 6 | +2.053974 -> +1.964416 | -0.089558 | -1.561669 | 4.028768 |

(eV. The gap itself swings because 3 rCROP iterations do not converge this
deck - see 8.4.)

### 8.3 The tolerance ruling, as printed every iteration

```
*** SC TOLERANCE IS INSIDE THE EXTRAPOLATION BAR ***
  sc_tol_ev              =      0.1000 meV  (per-band RMS dE between SC iterations)
  extrapolation p90      =    184.2672 meV median over states, 704.8652 meV max (15 % of Delta_tail)
  The loop is being asked to converge 1,843x TIGHTER than the uncertainty on the quantity it is converging.
```

### 8.4 Two findings the run produced that were not asked for

**(a) The gate is necessary but NOT sufficient, and this deck proves it.** At
`nband = 52` - the *minimum* the owner's rule admits - the extrapolation's own
trust diagnostic returns **`NOT TRUSTWORTHY`** on the VBM, the CBM and the
envelope: `Delta_model/Delta_tail = 0.51` and `0.65`, against the 0.35
threshold, i.e. *"the pairwise intercepts disagree by an appreciable fraction
of the correction, so these three counts do not resolve the 1/N tail. Use more
bands."* `Delta_tail` at the VBM is **1.35 eV** and `A/N3` is **1.33 eV** -
the band sum is nowhere near finished. So `nband >= 2*n_occ` admits decks on
which the estimator does not work, and passing the gate must not be read as
the extrapolation being trustworthy. The two are independent, and only the
gate is enforced. **Worth putting to the owner: the hard gate as specified is
a floor on the band count, not a guarantee about the fit, and this deck sits
where the two disagree.**

**(b) The correction is not converged either.** Its `Delta_tail` at the VBM
(1.35 eV) is comparable to the QP corrections themselves, and the p90 envelope
(184 meV median) exceeds the whole SC residual budget. On this deck the
extrapolation is applying a very large, poorly-resolved correction - which is
a statement about `nband = 52` on MoS2, not about the wiring.

## 9. Files changed

`src/gw/band_extrapolation.py` (gate threshold, `extrapolation_weights`,
`tolerance_bar_ev`, `sc_tolerance_ruling`) · `src/gw/gw_config.py` (new key,
alias resolver, tri-state) · `src/gw/ppm_pipeline.py` (`_extrapolated_point`,
driving Sigma, ruling print) · `src/gw/sigma_dispatch.py` (provenance-split
guard, second QSGW build, new `SigmaResult` field + basis registration) ·
`src/gw/sc_iteration.py` (second diagonalization + eqp report, finalize
rotation) · `docs/input_reference.md` · two regression decks ·
`tests/test_band_extrapolation.py` (+16 tests).
