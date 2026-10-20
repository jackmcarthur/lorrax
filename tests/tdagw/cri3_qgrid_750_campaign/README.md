# CrI3 750-band q-grid campaign helpers

These are reviewed backups of the external 20 October 2026 measurement
harnesses, not new production drivers. No input data, allocation, SSH session
or automatic job submission is bundled here. Launch collectively through the
normal authenticated native environment and a separately enforced wall cap.
The small tests check metadata/owner contracts, not GPU or material accuracy.

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
parents; no SC wall time is claimed as its refresh cost. A later 30v14c window
and nonlinear pulse need separate bright-target/window and resource gates.

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
No material/GPU trajectory or active44/pulse calibration is claimed. Turning
collisions off on the SC builder's bare-HF model is NOT this static TD-aGW arm.
The next-version BSE callback also publishes bounded k-summed root/c/v
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
No 10fs material completion, usable BSE target or nonlinear response has yet
been established by these adapter tests.

`compare_sc750_driven.py CAMPAIGN --out NEW_DIRECTORY` reads only a completed
three-arm campaign, requiring actual checkpoint coverage, matched fields,
times, k points and weights. It plots raw polarization, carriers, C-minus-B
and scale-separated waveform differences, plus available material-gap
snapshots. Its report also compares polarization divided by the actual
pulse amplitude across the two intensities. It performs no interpolation,
equilibrium subtraction or damping-rate fit, and refuses calibration-only
output. The host tests use explicitly synthetic small fixtures only.
