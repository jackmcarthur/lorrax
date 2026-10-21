# CrI3 750-band q-grid campaign helpers

These are reviewed backups of the external 20 October 2026 measurement
harnesses, not new production drivers. No input data, allocation, SSH session
or automatic job submission is bundled here. Launch collectively through the
normal authenticated native environment and a separately enforced wall cap.
The small tests check metadata/owner contracts, not GPU or material accuracy.

## Independent collision audit, 21 October

The occupation-bound failure is separate from the converged fresh-number
residual. Three independent audits check the derivation, actual native
execution and saved material matrices; no collision prescription is changed.
The read-only `independent_pauli_audit.py DATA_ROOT` reproduces the all64-k
localization and authenticates its saved inputs. The
[localization receipt](evidence/INDEPENDENT_PAULI_LOCALIZATION_20261021.json)
includes a closed passive three-level counterexample with zero trace but an
outward filled-space direction, row/column controls, rotations and the
limitations of deleting dispersive shell differences. The seven tiny tests
in `../test_column_shell_boundary_control_cpu.py` reproduce those algebraic
controls without material data or a GPU. These results identify a failure
of the literal collision closure; they are not a validated replacement,
equilibrium ground state, or 10 fs damping comparison.

The [native positive-time audit](evidence/UPPER_ENDPOINT_LITERAL_TIME_AUDIT_20261021.json)
checks the full complex kernel against independently integrated time-domain
Green functions, including coherences and wrong-order negative controls.
It reproduces the passive three-level violation through the actual stored-W,
FFT, projection and collision owners. The
[endpoint/reference toy](evidence/MGKBA_ENDPOINT_REFERENCE_TOY_v1.json)
separates a kinetic stationary state from a true correlated ground state and
shows why deleting dispersive terms can break coherent-state number balance.
Its internal E are fixed; it is not a self-consistent instantaneous-H solve.
The combined focused check is 20 passing CPU tests, with no new GPU jobs
and no runtime collision-law change. The paired full-covariance channel
alternative remains research-only until its dynamical accuracy is tested.

## Accepted reference and integration status

Latest actual reference: allocation 60056663 step4 completed eight q4 Fejér
SC maps on source `861d228`, meeting the 1 meV native criterion with a
0.974317 meV residual. The completed driver authenticates the accepted INPUT
gap 3.073808 eV, not the final raw OUTPUT gap. The exact provenance and timings
are in `evidence/Q4_FEJER_ACCEPTED_REFERENCE_60056663.json`. Warm P16 material
maps take about 45.2 s for W and 17.3 s for the protected160-output Sigma
sweep; these are not compact44 or coherent TD-refresh timings. Fejér is an
explicit response-smoothing policy, not a successful sharp-cutoff replay or
a conserving-closure claim. A matching finite-window q4 BSE and coherent
static-A calibration subsequently completed; raw-collision and updated-W
material dynamics remain unadmitted, as distinguished below.
See [PERFORMANCE.md](PERFORMANCE.md) for the measured cold/warm phase costs,
compact-destination contract, actual backend limits and remaining unmeasured
TD costs. The native tiny BSE mapping proof is recorded separately in
`evidence/P4_COARSE_BSE_60056663.json`; it is not a material eigensolve.

The prepared BSE and static-TD control share the native `make_bosonic_q_comb`
owner: inject the coarse screened head once, embed nested q rows with Nk/Nq,
then use existing fine-grid Fourier owners. This is ordinary coarse-q
quadrature, not W interpolation. Bare Hartree/exchange keep their fine-cell
normalization. Enlarged Fourier carriers are compiled/priced before dispatch.
Tiny literal full-complex BSE/static-control oracles pass on CPU; the separate
`diagnostics/verify_bse_coarse_q_p4.py` is the bounded native GPU proof, not a
material eigensolve or trajectory.

`run_sc_reference_frames.py` runs the unchanged native GW CLI and records the
actual input of each scalar SC map before donation. It saves full current E/f,
protected U, original-basis input H and small centroid/k provenance, never
retaining spatial faces. A reference is ready only when a proper converged
solver returns that exact input H and the native driver subsequently completes.
Budget-ended and one-map results are never promoted to converged references.
The campaign's original external copy ran with source `1134957`. This
repository helper is now a SAFER variant, not byte-identical to archived
runner SHA `31f545b6…`: periodic Python traceback walking defaults OFF
(`--stack-after 0`), and its wall watchdog emits a marker and exits without
walking Python frames. An explicit positive interval opts back into periodic
walking. Fatal handlers and external GDB remain unchanged. Job 60052556
captured a real SIGSEGV in CPython's `faulthandler_thread -> dump_frame`;
that fault must not be mislabeled a constructor eigensolver failure.
Changing this repository copy does not alter archived or running payloads.

The preserved diagnostic wrappers intentionally authenticate the ORIGINAL
archived runner SHA `31f545b6…`; they cannot use this safer repository helper
without a separately reviewed hash change. Their repository defaults now
pass `--stack-after 86400`, beyond their maximum two-hour process cap, because
the archived runner accepts only positive intervals. These changed wrapper
hashes are not the hashes of earlier actual launches. An explicit launch
argument still controls the recorded run. The active external corrected
launch already passes 86400 and is not modified here.

`native_frozen_map_probe.py` is a prepared, **not yet hardware-verified**
same-input q8/q4 probe. Its scope is full64 electronic rows, 750 bands and a
160-band protected block. After native coarse-q setup, it invokes the existing
map twice on an authenticated accepted input. It holds the full input E/f/U,
while each leg computes its own native W/head/Hartree/Sigma and spectral fit.
The raw750 and extrapolated full complex shell pairs are saved separately.
It intentionally stops at the external probe boundary, not a completed SC
driver. Native gates and memory admission remain authoritative.

The optional fine-q synchronous BSE callback consumes current orbitals, W0 and
head before Sigma can donate its inputs. The existing callback is
`../cri3_full_microscopic/native_bse_probe.py`; it finds finite-window bright
sub-gap candidates and measures true post-solve residuals. Its CPU toy payload
test does not certify CrI3 screening. A post-Sigma TD callback must explicitly
reconstruct current faces and bind the actual producer's spectral fit; holding
a Python reference is not a donation-lifetime guarantee.

The current probe reports full microscopic W comparison and reproduction of
the original accepted fine Sigma as **NOT_MEASURED**. Band-operator differences
alone are not a cutoff-convergence certificate. Neither helper weakens a
scientific gate, clips populations/poles, or substitutes a projected W model.

The actual source68 full64 fine-grid replay in allocation 60055335 refused
the Gram gate before BSE, Sigma or TD, as did its q4 SC client. A subsequent
audit found a deterministic file-bank ordering defect: requests used the
original member order while consumption was dense-first, and sequential
reads ignored the requested member row. The converged SC reference used the
unaffected indexed device bank. Fix `d34629c` aligns the orders and witnesses
row/q-span association; eight focused tests include a real tiny file-bank
round-trip. Its material replay remains pending. The detailed witness is
`evidence/STREAMED_BANK_ORDERING_DIAGNOSIS_60055335.json`. In particular, the
earlier q4 refusal is not evidence that its cutoff alone caused the failure.

`inputs/` preserves the corrected protected-form q8 deck (SHA `534a2c24…`)
and a prepared q4 deck (SHA `1712d907…`) with a deliberately absent
`seed_from_accepted_q8` path. The q4 seed can be staged only after the genuine
q8 terminal artifact, accepted input and completed driver are authenticated.
`evidence/` preserves the small header-only native guard (SHA `c697f567…`)
and the pending q4 handoff plan. These are input/provenance checks, not
converged SC results. `diagnostics/README.txt` records the actual fault-time
CPython stack, selector-policy correction, and explicitly partial progress
of corrected run 60053348; no current intermediate map is a reference.

## Accepted-SC750 short zero-field callback

The probe also supports `--only-qgrid 4` or `--only-qgrid 8` to evaluate one
response leg without rebuilding the other. A selected q4 callback uses
`coarse_bse`, `coarse_td`, or `coarse_pre_sigma` output paths, and records a
single-leg completion rather than a fine/coarse comparison. Its reference
must be an actually converged **q4 input frame**; the accepted q8 state may
not be relabeled as that reference. Pass `--accepted-deck` with the exact
producer deck (or use its identical copy as `--setup-deck`): its SHA must
match the accepted frame, and the native parser authenticates qgrid and
sharp/Fejér window before W is built. The existing paired frozen-state
comparison remains a separate calculation on one fixed input.

Electronic k rows remain full64 and virtual bands remain750. Bare HF uses
the fine-cell `head_resolver.direct_at(0j).vc0/(volume*64)`; the correlation
kernel keeps the actual q4 or q8 screened head and quadrature. Zero/driven
receipts expose the response grid/window, and coarse driven prerequisites
must use BSE and zero-field results from that same accepted response policy.
These are prepared adapter contracts, not a material TD or cutoff verdict.

The supplied four-band zero artifact is **preflight only**, not Pauli or
stationarity evidence for the resolved30v14c driven window. Before either
qualified MGKBA driven arm, the driven callback runs both B and C at
zero field for0.6fs on the actual resolved window and requires a completed
fresh refresh, raw number/boundary checks, trace/Pauli health and the explicit
zero-carrier-drift bound. `run_zero_controls=false` can disable the optional
A-zero control, never these B/C safety gates. Their actual artifacts and
response policy are listed in `matching_active_zero_gates.json`; a failure
prevents both MGKBA driven arms. A amplitude calibration remains permitted
before this larger-window gate. Identical equilibrium U/E/rho reuse the
same checked equilibrium W and Sigma row; the actual C refresh is not skipped.

For `execute_mgkba=false`, staticA amplitude calibration requires the actual
authenticated BSE target/reference but does **not** read raw-collision zero
artifacts. A missing or scientifically refused zero4 therefore need not
block independent staticA carrier-target measurements. Its manifest records
`raw_zero_status=NOT_REQUIRED_STATIC_A_ONLY`, no zero evidence and B/C not
attempted; ordinary coherent trace/Pauli/Hermiticity checks remain active.
Enabling qualified B/C still requires the original rawzero preflight and both mandatory
matching-window gates. An A-only completion is not an A/B/C comparison or
evidence that raw MGKBA is safe.

An explicit `diagnostic_pilot` plan may instead run B or C immediately,
without archival zero artifacts, static-A recalibration, or the paired zero
campaign. It names `arms`, an `amplitudes_v_m` subset of the finite candidate
list, and a nonempty `purpose`; `execute_mgkba=true` and
`run_zero_controls=false` are required. BSE/live-source authentication and
all instantaneous number, boundary, trace, Pauli, finite-value and resource
checks remain unchanged. Every pilot result is labeled
`UNQUALIFIED_DRIVEN_PILOT`; the top-level output is `pilot_complete.json`,
not the campaign's `complete.json`. It cannot admit a matching-zero gate or
claim damping or a qualified B/C comparison. Numerical fit tolerance is an
explicit plan value, recorded separately from these physical checks; a 5%
pilot is not evidence of a passed 2% fit.

`compare_sc750_driven.py --calibration-only` validates and plots only completed
native10fs static-A legs, even when a finite carrier-target search ends without
a selected bracket. Optional `--calibration-tags A_calibration_00 ...` names an
explicit verified subset for bounded data transfers; omitted committed tags
are reported. Raw P/E/electron/hole traces and amplitude-normalized P are not
a B/C comparison or a damping measurement. Default plotting still requires a
complete three-arm campaign.

`sc750_td_callback.py` is a prepared, not material-verified adapter from the
authenticated accepted input to the existing native snapshot and TD owners.
The accepted q8 input of run 60053348 is recorded separately in
`evidence/Q8_ACCEPTED_REFERENCE_60053348.json`; historical prepared/ongoing
records above are not its convergence receipt. The SC producer revision and
actual TD execution revision are recorded separately.

After a frozen native fine-q map, the probe commits its raw750/extrapolated
complex shell pairs, spectral fit and receipt BEFORE the optional TD callback.
Separate STARTED and CALLBACK_RETURNED records distinguish request from return;
a TD refusal leaves the completed GW evidence intact and aborts the probe.
To obtain the matched q8/q4 GW comparison independently of a possible TD
refusal, run first without either TD option. No full microscopic W comparison
or cutoff-accuracy verdict is implied by those small-band results.

The callback explicitly rerotates the original DFT carrier with the full
accepted 160-band unitary before selecting active [128,132), preserves the
logical750/padded752 ladder and actual native spectral fit, and uses the SC
anchor once plus bare-HF delta and MH minus its equilibrium value. Rates remain
raw, full-coherent, active-only, with no equilibrium collision subtraction,
PSD/population clipping or trace repair. Fullvirtual MH includes the tail.
`dynamic` is the legal parser value retaining complete MH in joint mode.

Launch collectively in a separately authenticated native environment; the
following is a command template, NOT a submitted job:

```sh
python CAMPAIGN/native_frozen_map_probe.py \
  --source IMMUTABLE_EXEC_SOURCE --source-revision EXEC_REVISION \
  --setup-deck PRIVATE_CORRECTED_Q4_SETUP.in \
  --frames ACCEPTED_Q8_FRAME_DIRECTORY --out FRESH_PROBE_DIRECTORY \
  --td-helper CAMPAIGN/sc750_td_callback.py \
  --td-deck CAMPAIGN/inputs/sc750_zero_joint_0p6fs.toml \
  --wall-cap 900 --stack-after 0
```

`LEG_HEAD` must equal EXEC_REVISION. The execution source needs the native
logical750/padded752 fix, authenticated SC-reference builder, legal dynamic-MH
default, spectral-tail adapter and full-grid seed join. The setup deck must
use fresh private outputs and the actual accepted q8 artifact, never a failed
map's output. This template does not select an allocation or waive native
memory admission. Optional `--bse-helper` uses the separate reviewed native
BSE callback; its finite-window target is not automatically a 2v2c TD target.

The zero-field deck is callback-only: its LIVE_AUTHENTICATED_SC750_CALLBACK
markers are not a loadable old160 manifest. The callback emits resolved runtime
paths and execution/module hashes; source paths describe the live prepared
callback, not a standalone reloadable bundle. It requests 0.6 fs with a 0.5 fs
updated-W refresh, source collision eta (currently 0.25 eV), and separate
response-lag gamma 0.05 eV. Completion requires an actual refreshed snapshot,
and reports fresh same-gauge boundary compressions, raw/coherent collision,
trace and Pauli excursions. Static SC convergence does not certify finite-eta
Markov stationarity or driven admission. Any refusal is evidence, not repaired.
Fine full-coherent TD has 64 q rows rather than the SC run's 10 symmetry
parents; no SC wall time is claimed as its refresh cost. The callback now
accepts only the two explicit zero-gate windows: `[128,132)` (2v2c) and
`[100,144)` (30v14c). The latter has its own prepared
`inputs/sc750_zero_joint_44band_0p6fs.toml`, full44 `rho_full` measurement,
44-dimensional reference/rates and truthful gauge/window labels. It is not
admitted by a completed four-band gate. Both retain full750 virtual bands
and require the same own-grid accepted source/head/tail authentication.
The single-trial observer retains only its latest full native row, checks
basis-object identity at capture, and leaves the actual evaluator caches
unchanged. Host tests do not qualify material44 dynamics; the matching
zero-field, bright-target/window and resource gates remain required before
a nonlinear MGKBA pulse; the independent static-A calibration does not
assert that collision admission.

Focused local parser/gauge/provenance tests are
`test_sc750_td_callback.py` and `test_native_frozen_map_probe.py`; they do not
replace a real multihost material callback execution.

`sc750_static_control.py` prepares the distinct coherent arm
`A_STATIC_SEX_HARTREE_NO_COLLISION` from the live fine-grid static-W owner.
It reuses the native full-centroid exchange and Hartree kernels without
exporting zeta. Hartree keeps the original G=0-excluded bare V0; exchange
receives the actual W(0) body and its head once. `HeadSample.wcoul0` is FULL
screened W, not Wc: the adapter injects that value alone, then sets the
separate scalar exchange-head term to zero. Its CPU literal-k-sum oracle
checks full coherences, bare-Hartree preservation and this head convention.
Static-A active44 calibration completed for 10 fs on the accepted q4 input;
the measured midpoint populations are 0.00813454 and 0.15942947 e/cell.
See [the preserved calibration record](evidence/STATIC_A_MIDPOINT_60056663.json).
This is not collision, updated-W or damping evidence. Turning collisions
off on the SC builder's bare-HF model is NOT this static TD-aGW arm.
The BSE callback also publishes bounded k-summed root/c/v
weights and central-2v2c containment. This measures the existing finite BSE
solution, not convergence in a larger window. It does not alter the pinned
source68 callback used by the already prepared material probe.

## Prepared larger-window driven comparison

`sc750_driven_callback.py` is a separate, prepared post-Sigma callback; it
does not weaken or replace the short zero-field gate above. Its explicit JSON
plan comes from `example_plan()` and must name the actual matched BSE and
completed zero-field artifacts. The requested 30v14c window is resolved by
the existing multiplet owner, not silently split across a degeneracy.

The three labels distinguish `A_STATIC_TDAGW_NO_COLLISION` (bare Hartree plus
fixed-W screened exchange), `B_FROZEN_W_MGKBA` and `C_UPDATED_W_MGKBA`.
Actual complete A trajectories calibrate two finite amplitudes near 0.01
and 0.1 carriers/cell; B/C use those exact pulses and share the same initial
equilibrium W. A differs in its coherent functional, while C minus B includes
both real/coherent and collision changes. Neither difference alone measures
irreversible damping. The internal `execute_mgkba` flag selects B/C execution;
it is not a request for more user approval.

Full density samples are in each native H5 file. Small refresh snapshots
preserve material H/U/E, full complex shell pairs and fresh collision/boundary
diagnostics; separate checkpoint health records measure held-rate drift.
Gap records describe the propagated window, not a complete750-band gap.
The actual frozen/reference basis and source producer are recorded separately.
These adapter tests alone establish neither a material target nor dynamics.
The separate material BSE and static-A receipts above establish their limited
results; no 10fs B/C completion or screening-induced damping is established.

`compare_sc750_driven.py CAMPAIGN --out NEW_DIRECTORY` reads only a completed
three-arm campaign, requiring actual checkpoint coverage, matched fields,
times, k points and weights. It plots raw polarization, carriers, C-minus-B
and scale-separated waveform differences, plus available material-gap
snapshots. Its report also compares polarization divided by the actual
pulse amplitude across the two intensities. It performs no interpolation,
equilibrium subtraction or damping-rate fit, and refuses calibration-only
output. The host tests use explicitly synthetic small fixtures only.

## Active-destination cost benchmark

`--pre-sigma-helper active_sigma_probe.py` on the frozen native probe runs
two identical-input compact native Sigma sweeps after fine W, then stops
cleanly before the protected160 Sigma map, BSE, coarse leg or TD. Only
external [100,144) energies and 44x44 matrices are requested; all750 physical
internal bands remain. The two sweeps expose cold and held-session
costs, actual window/tau counts and full complex branch pairs. The first can
serve one-shot rules while building padded held rules for the second; these
need not be the same executable or agree at machine precision. Their observed
matrix difference is reported, not promoted to a convergence verdict. Wall time per
window/tau pair is an amortized whole-sweep cost, not isolated GPU latency.
This first benchmark measures the raw750 equilibrium correlation BODY only;
it does not include head evaluation, tail extrapolation or construction of
the general coherent refreshed-W model. Those costs must be reported
separately, not inferred from this sweep.

The source-fit G0 handoff also accepts the native basis's exact-zero canonical
suffix padding (3000 logical / 3008 canonical in the actual material setup).
The captured producer/basis/value receipt and small vector are preserved
before maps; only the existing basis owner performs orbit packing. BSE
validates this before allocating static W. No G0 fit or large zeta export is
substituted.

## Verification-only dielectric and source-frozen collision diagnostic

The first full750 coherent response stream completed in 405.294 s, but the
general reduced W was refused for a growing q0 pole before any ordered Sigma
or t0 collision. See [PERFORMANCE.md](PERFORMANCE.md). The saved electronic
input is an exact equilibrium projector; no population inversion explains
that fitted pole. A retarded-fit defect or genuine dielectric zero must be
distinguished with a native dielectric witness, not by clipping poles.

`screen_general_shared_poles` accepts an explicit snapshot-only
`direct_dielectric_probe={"q": 0, "z_ry": [real, imag]}` diagnostic. It uses
the same current full750/four-block Green/chi owners at one upper-half-plane
site, saves scalar physical-space epsilon and checked-inverse witnesses,
then raises `GATE shared_general_dielectric_diagnostic_stop` before fitting
or Sigma. Only resolved inverse bounds establish nonsingularity; the named
stop or an `inverse_error` record alone does not. No microscopic face is
gathered to host and no default production route changes.

`sc750_source_frozen_callback.py` separately binds the production current
U/E/full-rho Sigma and authenticated tail/head to the exact signed conversion
of the native equilibrium source W. Its optional first dielectric witness
uses a fresh capacity ledger and catches only the explicit diagnostic stop.
The callback requires the private source artifact policy `write_poles=true`
and `inputs/sc750_source_frozen_zero44_0p6fs.toml`. Source W is **unshifted**:
its applied response-lag gamma is zero, not the configured general-W .05 eV.
All results carry `SOURCE_FROZEN_DIAGNOSTIC`, `zero_gate=false` and
`matched_BC_admission=false`; both driven prerequisite validators reject
them. This is a bounded raw-collision investigation, not a matching B/C zero
gate, an updated-W calculation or a successful material trajectory.

## Saved-W active-rate quadrature replay

`same_w_rate_replay.py` authenticates the accepted source frame, saved t0
projector and committed general-W model before invoking the existing native
ordered Sigma owner with `internal='active'`. Four or sixteen ranks are
supported; the actual material replay used the original P16 layout. Native
initialization runs, but no SC map, new W, upstream160-output Sigma or TD
step runs. Only the exact binary64k/30v14c projector is accepted: headless
rate-body C and boundary compressions match the full collision there, not
for an arbitrary coherent state. Head, virtual MH and tail are omitted.

Actual jobs60078170/60078338 reduced the fresh trace residual to
`-9.64e-12 Ry` at quadrature1e-8, while retaining a **-24.859 meV** outward
occupied-boundary eigenvalue. The tight rate body cost30.64s. This is
number-quadrature convergence, not a positivity fix or damping trajectory;
see [timings and scope](PERFORMANCE.md#same-w-active-rate-convergence-number-recovered-pauli-boundary-not-recovered).

Optional `--spectral-bands 0 1 ... 43 --spectral-k 0 --quadrature-eps 1e-8`
selects a **grid-only** diagnostic instead. Each frequency is the selected
band's saved instantaneous energy minus the common numerical origin; the
electronic energies/density and W are unchanged. One native sweep retains
full44×44 both-retarded branch matrices at those common energies, then saves
only the selected k, raw widths `i(R-R†)` and their unclipped eigenvalues.
This supplies data missing from row/column on-shell tables. There are no
matrix roots, new collision closure, extra on-shell sweep or PSD assertion.

`diagnostics/benchmark_general_screen_device.py` preserves the bounded native
pre-map screening benchmark. It authenticates the accepted frame and saved
t0 projector, then reconstructs all16 current W rows twice: cold and with
held numerical fit spaces. Current response, poles/residues and checks are
recomputed on both calls; source band extrapolation is not disabled. It
stops before Sigma or propagation. The explicit pilot W fit tolerance is5%,
not a material convergence claim. Invocation requires a separately reviewed
native launcher, provenance environment and wall cap; it submits no jobs.
