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
The campaign's original external copy ran with source `1134957`; changing
these backup tests does not alter its recorded source/helper hashes.

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
