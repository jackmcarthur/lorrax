"""Phase C acceptance gate: SC iteration 1 ≡ one-shot G0W0.

The self-consistent QSGW path (``sc_iteration.gw_iteration_map`` →
``screening.compute_screening`` → ``sigma_dispatch.compute_sigma_xc``)
and the one-shot ``gw_jax.main()`` path are the SAME physics pipeline;
one QSGW iteration started from ``H^(0) = diag(E_DFT)`` must reproduce
the one-shot G0W0 result (``make_initial_state_from_dft`` docstring).

Historically main() carried an inlined second copy of the pipeline, so
this equivalence was structural-but-unverified (and in fact broken at
the seams: full-BZ vs IBZ static W, wfn.efermi vs midgap E_F in the
QSGW build).  This gate pins the equivalence on the GN-PPM MoS2 3×3
fixture so the driver unification (IDEAL_SCAFFOLD_VS_LORRAX.md §5
Phase C) cannot silently regress either path.

Compared outputs:

* ``sigma_diag_gnppm_test.dat`` rows (sigX / sigC / sigXC / VH) — the Σ
  decomposition at E_DFT.
* ``eqp0.dat`` / ``eqp1.dat`` — the at-DFT QP energies + Z-linearised
  update (covers kin_ion + V_H + Σ_x + Σ_c(E_DFT) + the Z-factor).

NOT compared: ``qp_wfn_rotations.h5`` — the SC carry applies the band
partition (off-diagonal masking + scissor for out-of-ω-grid bands), so
its eigh family legitimately differs from the one-shot full-matrix eigh
whenever some bands are out of range (28/80 on this fixture).  The Σ
level is the "same pipeline" statement this gate pins.

Sensitivity note (measured 2026-07-09, this fixture): the GN-PPM
pipeline amplifies a +1 ulp perturbation of every WFN energy to
max|ΔΣ_c(ω)| = 1.28 eV (near-threshold pole modes) while the run itself
is bit-deterministic.  That is why iteration 0 of the SC map must use
the exact eigensystem of its diagonal carry rather than eigh — see
``sc_iteration.gw_iteration_map``.

atol=1e-6: both runs are deterministic; the tolerance only absorbs
GPU-nondeterministic last-ULP drift (same budget as the frozen gates).
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

# Reuse the e2e subprocess runner + GPU gating from the regression harness
# (tests/ is a package; put its dir on sys.path for a bare sibling import
# under any pytest import mode — same pattern as test_mu_pad_invariance).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_gw_jax_regression import (          # noqa: E402
    _REG,
    _gpu_available,
    _parse_eqp_rows,
    _requested_platform,
    _run_gw_jax,
)

_LABELS = ("sigX", "sigC", "sigXC")
_OUT = "sigma_diag_gnppm_test.dat"


def _run_case(tmp_path, name, input_mutator):
    case_dir = _REG / "gnppm_debug"
    run_dir = tmp_path / name
    shutil.copytree(
        case_dir, run_dir,
        ignore=shutil.ignore_patterns("tmp", "sigma_diag*.dat", "eqp*.dat"))
    input_file = run_dir / "gnppm_test.in"
    input_file.write_text(input_mutator(input_file.read_text()))
    res = _run_gw_jax(run_dir, "gnppm_test.in", _requested_platform())
    if res.returncode != 0:
        pytest.fail(
            f"{name} run failed.\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    out = run_dir / _OUT
    assert out.exists(), f"no output written: {out}"
    return run_dir


def _numeric_tokens(path):
    """All numeric tokens of a whitespace-separated .dat file, in order."""
    toks = []
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        for t in line.split():
            try:
                toks.append(float(t))
            except ValueError:
                pass
    return np.asarray(toks, dtype=np.float64)


@pytest.mark.regression
def test_sc_iteration1_equals_one_shot(tmp_path):
    platform = _requested_platform()
    if platform in {"gpu", "cuda"} and not _gpu_available():
        pytest.skip("CUDA GPU not available.")

    one_shot = _run_case(tmp_path, "one_shot", lambda text: text)

    def _to_sc(text):
        assert "qp_solver = one_shot_dft" in text
        return text.replace(
            "qp_solver = one_shot_dft",
            "qp_solver = self_consistent\nsc_max_iter = 1")

    sc1 = _run_case(tmp_path, "sc_iter1", _to_sc)

    # Σ decomposition rows must match.
    rows_os = _parse_eqp_rows(one_shot / _OUT, _LABELS)
    rows_sc = _parse_eqp_rows(sc1 / _OUT, _LABELS)
    assert rows_os.shape == rows_sc.shape, (
        f"row-count mismatch: one-shot {rows_os.shape} vs SC-1 {rows_sc.shape}")
    np.testing.assert_allclose(
        rows_sc[:, :6], rows_os[:, :6], rtol=0.0, atol=1e-6,
        err_msg="SC iteration 1 sigma_diag differs from one-shot G0W0")

    # At-DFT QP energies (eqp0) + Z-linearised update (eqp1) must match.
    for fname in ("eqp0.dat", "eqp1.dat"):
        t_os = _numeric_tokens(one_shot / fname)
        t_sc = _numeric_tokens(sc1 / fname)
        assert t_os.shape == t_sc.shape, (
            f"{fname}: token-count mismatch {t_os.shape} vs {t_sc.shape}")
        np.testing.assert_allclose(
            t_sc, t_os, rtol=0.0, atol=1e-6,
            err_msg=f"SC iteration 1 {fname} differs from one-shot G0W0")
