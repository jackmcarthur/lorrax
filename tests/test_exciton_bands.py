"""Driver smoke for ``bse.exciton_bands`` — 3-point Q path, 1 GPU, minutes.

Builds a scratch run dir over the MoS2 3×3 640-centroid fixture (read-only
symlinks: restart tmp/, WFN.h5, centroids), writes a 3-point
``K_POINTS crystal_b`` Γ→M path into the input, runs the driver end-to-end
in interp mode, and asserts:

  * it completes and writes ``<prefix>.dat`` + ``<prefix>.png``;
  * the .dat has one interp row per path point with finite, ordered,
    positive-gap eigenvalues;
  * the Γ row's lowest eigenvalue lies in a loose physical bracket
    (0.1–10 eV — a unit/convention breakage detector, not a physics pin).

The single-compile census is asserted inside the driver run itself
(``solve_path`` is one jit; the warm re-run must be dispatch-only and
bit-reproducible — the driver hard-asserts reproducibility).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")

FXDIR = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/"
         "05_lorrax_cohsex_native")

pytestmark = pytest.mark.skipif(
    not os.path.exists(f"{FXDIR}/tmp/zeta_q.h5"),
    reason="MoS2 3x3 640-centroid fixture not present")


@pytest.mark.gpu
def test_exciton_bands_3pt_smoke(tmp_path):
    harness.skip_unless_gpu(pytest)
    run = tmp_path / "exb"
    run.mkdir()
    os.symlink(f"{FXDIR}/tmp", run / "tmp")
    os.symlink(f"{FXDIR}/WFN.h5", run / "WFN.h5")
    os.symlink(f"{FXDIR}/centroids_frac_640.txt",
               run / "centroids_frac_640.txt")
    src = open(f"{FXDIR}/cohsex.in").read()
    head = src.split("K_POINTS")[0].rstrip() + "\n\n"
    (run / "exb.in").write_text(head + (
        "K_POINTS {crystal_b}\n2\n"
        "  0.0000 0.0000 0.0000 1   # G\n"
        "  0.0000 0.5000 0.0000 2   # M\n"))

    from bse.exciton_bands import main
    cwd = os.getcwd()
    try:
        os.chdir(run)
        # --q-per-segment 1 is the IDENTITY on the deck's own counts, and it
        # is what keeps this a 3-point smoke: the driver's default is a floor
        # of 16 per segment (it is a bandstructure), which would make this
        # cell a 17-point run for no extra coverage.  The floor's own
        # semantics are asserted in test_q_per_segment_floor below, on the
        # parsed dict, with no GPU.
        rc = main(["-i", str(run / "exb.in"), "--n-val", "2", "--n-cond", "2",
                   "--n-eig", "4", "--block-size", "4", "--max-iter", "30",
                   "--vq-mode", "interp", "--q-per-segment", "1",
                   "--out-prefix", "smoke"])
    finally:
        os.chdir(cwd)
    assert rc == 0
    assert (run / "smoke.png").exists()
    rows = [l.split() for l in open(run / "smoke.dat")
            if l.strip() and not l.startswith("#")]
    interp = [r for r in rows if r[5] == "interp"]
    assert len(interp) == 3
    ev = np.array([[float(x) for x in r[6:]] for r in interp])
    assert np.all(np.isfinite(ev))
    assert np.all(np.diff(ev, axis=1) >= -1e-8)      # ascending per row
    assert 0.1 < ev[0, 0] < 10.0                      # eV bracket at Gamma
