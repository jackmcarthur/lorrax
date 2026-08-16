# Wiring `self_energy_eval_type` turns `test_sc_iteration1_equals_one_shot` RED, and the reason names the next design decision (2026-08-16)

**STATUS: RED GATE, ATTRIBUTABLE, DELIBERATELY NOT "FIXED". No tolerance
moved, no test edited, no assertion relaxed.** Branch
`feat/se-eval-type-wire-2026-08-16` @ `865d3469`, off
`origin/feat/staged-sc-2026-08-15` @ `98289d77`. Workspace
`/pscratch/sd/j/jackm/seevaltype_20260816/`, sole writer; the Si deck inputs
under `sandbox_v2_docs_consolidation_2026-08-14/runs/Si/01_staged_sc_2026-08-14/qe/`
were read-only and are untouched.

## 1. The set-diff, both sides run

Same four modules, same FFI, same module, one process, `-G 4 -n 1`:

| tree | result | failing |
|---|---|---|
| parent `98289d77` | 1 failed, 23 passed | `test_fixed_point_frozen_qp_rotations` |
| this branch `865d3469` | 2 failed, 22 passed | the above **+ `test_sc_iteration1_equals_one_shot`** |

`test_fixed_point_frozen_qp_rotations` is **pre-existing** — red on both
sides, not mine, not investigated here. `test_sc_iteration1_equals_one_shot`
is **new and attributable**.

Logs: `logs/batch4/GPU_gates.log` (branch), `logs/gpu_gates_parent.log` (parent).

## 2. What the gate compares, and why the wiring moves it

`tests/test_invariance_gates.py::test_sc_iteration1_equals_one_shot` runs the
gnppm fixture twice — once `qp_solver = one_shot_dft`, once
`qp_solver = self_consistent` + `sc_max_iter = 1` — and asserts

1. `sigma_diag` rows agree to `1e-6`, then
2. `eqp0.dat` / `eqp1.dat` numeric tokens agree to `atol=1e-6`.

**Assertion 1 still PASSES.** The run reaches line 201, so Σ is unmoved: the
physics claim the gate was built for — SC iteration 1 reproduces one-shot —
still holds exactly as before.

**Assertion 2 fails, BY DESIGN of the feature.** `self_energy_eval_type`
resolves from `qp_solver`: `one_shot_dft → linearized`,
`self_consistent → hermitianized`. Before the wiring both arms reported the
same linearized Newton numbers because the key selected nothing. Now the
one-shot arm reports the BGW Newton/Z pair and the SC arm reports the
rediagonalised H_qp eigenvalues, so the gate compares two different
definitions of the QP energy.

    Mismatched elements: 414 / 1692 (24.5%)
    Max absolute difference among violations: 21.70858649

414 is exactly the state count of the fixture's `sigma_diag` reference
(`sigma_diag_gnppm_ref.dat`), i.e. **every** QP value differs, which is the
signature of "different definition" rather than "a few bad bands".

## 3. The 21.7 eV is not off-diagonal Σ, and that is the finding

On a CLEAN deck the two definitions differ by the off-diagonal Σ and no more.
Measured, Si 4×4×4, one-shot GN-PPM, 24-band window, 192 states:

    E_qp(hermitianized) - eqp0(linearized):
      mean +0.0000 eV   RMS 0.0521 eV   range [-0.3447, +0.3610] eV
      trace identity residual  max 2.0e-09 eV over 8 k

(The mean is zero and the trace residual is at print precision because
eqp0 IS diag(H) and the eigenvalue sum is its trace.)

21.7 eV is two orders of magnitude beyond that, and the fixture says why:

| | fixture `gnppm_debug` | my Si deck |
|---|---|---|
| window | `nval 26 / ncond 20 / nband 46` — all 46 bands | 24 bands |
| E_DFT span | **[-40.09, -2.56] eV** | [-5.70, 19.49] eV |
| ω-grid | **[-10, +10] eV** | [-15, +8] eV |

The great majority of the fixture's Σ window lies BELOW its ω-grid floor.
Those bands' Σ_c is clamped at the grid edge — it is not a self-energy, it is
an edge value. The linearized report is per-band, so clamped bands stay
confined to their own rows. **A rediagonalisation mixes them across the whole
window through the off-diagonals**, which is how an edge artefact at -40 eV
becomes a 21.7 eV move on states that were fine.

This is not a new hazard: `sc_iteration.run_sc_driver` already handles exactly
it, and says so —

> In-range mask: bands whose E_DFT lies inside [σ_ω_min, σ_ω_max] at *every* k.
> Bands outside the ω-grid get the per-iteration scissor (otherwise their Σ_c
> is clamped at the grid edge → the QSGW H-build feeds garbage diagonals that
> explode the iteration).

— via `classify_bands_in_grid` → `BandPartition(protected/in_range)` →
`apply_band_partition`, printed each run as `SC partition: protected/in-range`.

## 4. The decision this hands back, and why it was not taken here

**The one-shot hermitianized path does NOT apply that partition.** It reports
the eigh the driver already performs on the raw `kin_ion + Σ_xc + V_H`.

Giving it the partition means importing `BandPartition`,
`classify_bands_in_grid` and `apply_band_partition` into the one-shot seam AND
deciding a scissor policy for a run that has no iteration to fit one on — the
SC scissor is refitted per iteration against the in-grid bands. That is design
work on the physics, not "expose one pass of the SC construction", so it was
stopped at the report line rather than pushed through.

Three options, for the owner:

1. **Refuse.** `hermitianized` refuses a deck whose Σ window leaves the
   ω-grid, naming the offending bands — the same shape as the band-degeneracy
   guard this branch already added for the sliced-multiplet case. Honest,
   cheap, and makes the fixture deck simply ineligible.
2. **Partition.** Reuse the SC band partition on the one-shot path so
   out-of-grid bands take a scissor and carry no off-diagonal mixing. Most
   faithful to what SC does; costs the scissor-policy decision above.
3. **Warn.** Report the out-of-grid count and proceed. Weakest — it is the
   "one seam only whispered" pattern the `band_degeneracy` ruling rejected.

Option 1 is the recommendation: it is the smallest thing that cannot be
silently wrong, and option 2 remains available afterwards without rework.

## 5. What to do with the red gate meanwhile

Do **not** relax `atol`. The gate is comparing two quantities that are now
legitimately different, so the repair is to the COMPARISON, not the tolerance.
The natural rewrite — expressible only because of this wiring — is to run the
one-shot arm with an explicit `self_energy_eval_type = hermitianized` so both
arms report the same definition, keeping the `sigma_diag` leg exactly as is.
That edits a landed acceptance gate, which is owner-scoped, so it is proposed
here and not done.

Until then this row is the accounting: **one new red, named, diagnosed, and
not a physics regression** — Σ is bit-unmoved and the default reporting path is
data-identical to the parent (`eqp0/eqp1/sigma_diag` sha256-equal modulo the
provenance stamp, `sigma_mnk.h5` equal over all 7 datasets).
