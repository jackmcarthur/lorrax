import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "tests" / "regression" / "cohsex_debug"
INPUT_FILE = CASE_DIR / "cohsex_test.in"
REFERENCE_FILE = CASE_DIR / "eqp_ref.dat"
OUTPUT_FILE = CASE_DIR / "eqp_test.dat"


def _gpu_available() -> bool:
    try:
        import jax

        return any(getattr(dev, "platform", "") in {"gpu", "cuda"} for dev in jax.devices())
    except Exception:
        return False


def _requested_platform() -> str:
    # Default to JAX's native backend selection (typically GPU on test nodes).
    platform = os.environ.get("ISDF_COHSEX_TEST_PLATFORM", "auto").strip().lower()
    valid = {"cpu", "gpu", "cuda", "auto"}
    if platform not in valid:
        raise ValueError(
            f"Invalid ISDF_COHSEX_TEST_PLATFORM={platform!r}. "
            f"Expected one of {sorted(valid)}."
        )
    return platform


def _parse_eqp_rows(path: Path, labels=("sigSX", "sigCOH", "sigTOT")) -> np.ndarray:
    float_re = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    imag_opt = rf"(?:\+\s*{float_re}i)?"  # optional imaginary part (not captured when absent)
    a, b, c = labels  # COHSEX: sigSX/sigCOH/sigTOT ;  GN-PPM: sigX/sigC/sigXC
    # Match: n=<band> <a>=<re>[+<im>i] <b>=<re>[+<im>i] <c>=<re>[+<im>i] VH=<re>[+<im>i]
    # (a trailing Eo= column, if present, is ignored — the regex is unanchored.)
    data_re = re.compile(
        rf"n=\s*(\d+)\s+"
        rf"{a}=\s*{float_re}{imag_opt}\s+"
        rf"{b}=\s*{float_re}{imag_opt}\s+"
        rf"{c}=\s*{float_re}{imag_opt}\s+"
        rf"VH=\s*{float_re}{imag_opt}"
    )
    kpt_re = re.compile(r"k-point\s+(\d+)\s*:")

    kpt = -1
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        k_match = kpt_re.search(line)
        if k_match:
            kpt = int(k_match.group(1))
            continue

        m = data_re.search(line)
        if not m:
            continue

        band = int(m.group(1))
        # Groups: 2=SX_re, 3=SX_im, 4=COH_re, 5=COH_im, 6=TOT_re, 7=TOT_im, 8=VH_re, 9=VH_im
        sx_re = float(m.group(2))
        coh_re = float(m.group(4))
        tot_re = float(m.group(6))
        vh_re = float(m.group(8))
        vh_im = float(m.group(9)) if m.group(9) else 0.0
        rows.append([float(kpt), float(band), sx_re, coh_re, tot_re, vh_re, vh_im])

    if not rows:
        raise ValueError(f"No Sigma data rows were parsed from {path}")
    return np.asarray(rows, dtype=np.float64)


def _run_gw_jax(run_dir, input_name, platform, extra_env=None):
    """Run ``python -m gw.gw_jax -i <input_name>`` in run_dir; return the process."""
    env = os.environ.copy()
    cache_dir = Path(env.get("JAX_COMPILATION_CACHE_DIR", str(REPO_ROOT / ".pytest_jax_cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    env.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    env.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    env.setdefault("JAX_ENABLE_X64", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if platform == "cpu":
        env["JAX_PLATFORMS"] = "cpu"; env["JAX_PLATFORM_NAME"] = "cpu"
    elif platform in {"gpu", "cuda"}:
        env["JAX_PLATFORMS"] = "cuda,cpu"; env["JAX_PLATFORM_NAME"] = "gpu"
    else:
        env.pop("JAX_PLATFORMS", None); env.pop("JAX_PLATFORM_NAME", None)
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "gw.gw_jax", "-i", input_name],
        cwd=run_dir, env=env, capture_output=True, text=True, timeout=900, check=False,
    )


# (case_id, subdir, input_name, output_name, reference_name, sigma_labels, atol)
# COHSEX: static Σ (sigSX/sigCOH/sigTOT) on WFNsmall; GN-PPM: dynamic Σc
# (sigX/sigC/sigXC) on a full MoS2 3×3 WFN (WFNsmall is incompatible with the
# dynamic build_G path — see reports/gw_refactor_map_2026-07-01/BUGS_FOUND.md).
#
# si_cohsex_3d is the FIRST BGW-anchored, sys_dim=3 e2e gate.  The two MoS2
# cases above are 2D self-consistency freezes (their reference is a re-frozen
# LORRAX number); this one exercises the 3D Coulomb/analytic-head path and its
# frozen value is pinned to BerkeleyGW.  System: bulk Si 4×4×4 (nosoc), the run
# proven to agree with BGW COHSEX at MAE 0.12 meV / max|Δ| 0.48 meV — see
# reports/cohsex_si_444_gamma_agreement_2026-05-02/.  The exact 0.12-meV config
# needs a 185 MB BGW vcoul-body dump (not git-committable), so the fixture uses
# LORRAX's *native* finite-q Coulomb body (identical to BGW-noavg 4π/|q+G|²)
# plus BGW's q→0 head as two scalars (vhead / whead_0freq).  Verified vs BGW
# sigma_hp.log at Γ: sigCOH tracks to 0.22 meV, sigTOT to 3.2 meV spread —
# matching the validated overlay run (see tests/regression/si_cohsex_debug/
# README.md).  atol=1e-3 eV (1 meV): a physical bound above the 0.12 meV BGW
# agreement, tight enough to catch a real 3D regression.  Two independent GPU
# runs reproduce the reference bit-for-bit (max|Δ|=0).
#
# bispinor is the FIRST e2e gate on the bispinor path (nspinor=2, screened-
# charge COHSEX Σ_SX+Σ_COH on W⁰⁰ plus bare Breit Σ^B).  It exercises the 4
# ζ-channel / 7 V_q-tile / transverse-γ̃ machinery that the scalar gates never
# touch.  System: MoS2 3×3 (nspinor=2), the run hand-validated in
# reports/bispinor_screened_a_validation_2026-06-16/.  Σ^B is folded into the
# sigSX column, so the sigma_diag labels are the same sigSX/sigCOH/sigTOT as
# scalar COHSEX (the bispinor path does NOT emit separate Σ^B columns).  The
# fixture WFN.h5 is the source 82-band WFN truncated to 34 bands (52 MB, > the
# nband=32 window) — verified BIT-IDENTICAL to the full-WFN run (bands above 32
# never enter the COHSEX/ζ window).  Runs on 1 GPU (LORRAX_NGPU=1) at
# memory_per_device_gb=30, so the standard pytest harness drives it.  The IBZ
# closure check fails on this non-orbit-closed centroid set so the run is
# full-BZ-direct; Σ_X/Σ^B is covariant (IBZ-unfold == full-BZ-direct, see
# reports/bispinor_tt_conditioning_2026-06-16/), so the frozen value is a valid
# regression freeze.  atol=1e-6: pure freeze (deterministic run, bit-identical
# on reruns; 1e-6 only absorbs GPU-nondeterministic last-ULP drift).
_REG = REPO_ROOT / "tests" / "regression"
_CASES = [
    ("cohsex", _REG / "cohsex_debug", "cohsex_test.in", "eqp_test.dat",
     "eqp_ref.dat", ("sigSX", "sigCOH", "sigTOT"), 1e-6),
    ("gnppm", _REG / "gnppm_debug", "gnppm_test.in", "sigma_diag_gnppm_test.dat",
     "sigma_diag_gnppm_ref.dat", ("sigX", "sigC", "sigXC"), 1e-6),
    ("si_cohsex_3d", _REG / "si_cohsex_debug", "cohsex_si_test.in",
     "eqp_si_test.dat", "eqp_si_ref.dat", ("sigSX", "sigCOH", "sigTOT"), 1e-3),
    ("bispinor", _REG / "bispinor_debug", "bispinor_test.in",
     "sigma_diag_bispinor_test.dat", "sigma_diag_bispinor_ref.dat",
     ("sigSX", "sigCOH", "sigTOT"), 1e-6),
]


@pytest.mark.regression
@pytest.mark.parametrize(
    "case_id,case_dir,input_name,output_name,ref_name,labels,atol",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_gw_jax_matches_reference(
    tmp_path, case_id, case_dir, input_name, output_name, ref_name, labels, atol
):
    platform = _requested_platform()
    if platform in {"gpu", "cuda"} and not _gpu_available():
        pytest.skip("CUDA GPU not available for requested ISDF_COHSEX_TEST_PLATFORM=gpu.")

    input_src = case_dir / input_name
    reference_file = case_dir / ref_name
    assert input_src.exists(), f"Missing regression input: {input_src}"
    assert reference_file.exists(), f"Missing regression reference: {reference_file}"

    run_dir = tmp_path / case_dir.name
    shutil.copytree(
        case_dir,
        run_dir,
        ignore=shutil.ignore_patterns(
            "tmp", output_name,
            "eqp_test.dat", "eqp0_test.dat", "eqp1_test.dat",
            "sigma_diag.dat", "eqp0.dat", "eqp1.dat",
        ),
    )
    input_file = run_dir / input_name
    output_file = run_dir / output_name

    result = _run_gw_jax(run_dir, input_name, platform)
    cmd = [sys.executable, "-m", "gw.gw_jax", "-i", input_name]
    if result.returncode != 0:
        pytest.fail(
            f"{case_id} regression run failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    assert output_file.exists(), f"Expected output file was not written: {output_file}"

    if output_file.read_text() == reference_file.read_text():
        return

    ref_rows = _parse_eqp_rows(reference_file, labels)
    out_rows = _parse_eqp_rows(output_file, labels)
    assert out_rows.shape == ref_rows.shape, (
        f"Row-count mismatch: output shape {out_rows.shape}, reference shape {ref_rows.shape}"
    )

    # Compare only real-valued physics columns: kpt, band, <3 Σ components>, VH_re
    # (exclude the imaginary/VH_imag noise-level parts; byte-identity above is the
    # primary check, this atol path only absorbs GPU-nondeterministic last-ULP drift).
    try:
        np.testing.assert_allclose(out_rows[:, :6], ref_rows[:, :6], rtol=0.0, atol=atol)
    except AssertionError as exc:
        pytest.fail(f"{case_id} output differs from reference beyond tolerance.\n{exc}")


@pytest.mark.regression
def test_gnppm_fixed_point_reproduces_frozen_qp_rotations(tmp_path):
    """``qp_solver = fixed_point`` must reproduce the pre-toggle eigh outputs.

    The 2026-07-08 ``qp_solver`` toggle changed the DEFAULT eigh-family
    behavior for dynamic modes (``E_qp_ry`` / ``qp_wfn_rotations.h5`` /
    ``WFN_qp.h5``) from the diagonal on-shell fixed point to the at-DFT
    G0W0 evaluation (``one_shot_dft``).  This gate proves the default flip
    is a pure re-labeling, not a physics change: ``qp_solver = fixed_point``
    on the GN-PPM fixture reproduces the ``E_qp_nk_rydberg`` eigenvalues
    frozen from the pre-toggle HEAD (620b501) run of this exact fixture
    (``eqp_rotations_fixedpoint_ref.npy``).  atol=1e-6 Ry only absorbs
    GPU-nondeterministic last-ULP drift — the run is deterministic.
    """
    import h5py

    platform = _requested_platform()
    if platform in {"gpu", "cuda"} and not _gpu_available():
        pytest.skip("CUDA GPU not available for requested ISDF_COHSEX_TEST_PLATFORM=gpu.")
    case_dir = _REG / "gnppm_debug"
    ref_file = case_dir / "eqp_rotations_fixedpoint_ref.npy"
    assert ref_file.exists(), f"Missing frozen reference: {ref_file}"

    run_dir = tmp_path / "gnppm_fixed_point"
    shutil.copytree(
        case_dir, run_dir,
        ignore=shutil.ignore_patterns("tmp", "sigma_diag*.dat", "eqp*.dat"))
    input_file = run_dir / "gnppm_test.in"
    text = input_file.read_text()
    assert "qp_solver = one_shot_dft" in text
    input_file.write_text(
        text.replace("qp_solver = one_shot_dft", "qp_solver = fixed_point"))

    result = _run_gw_jax(run_dir, "gnppm_test.in", platform)
    if result.returncode != 0:
        pytest.fail(
            f"fixed_point regression run failed.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    rot_file = run_dir / "qp_wfn_rotations.h5"
    assert rot_file.exists(), f"missing {rot_file}"
    with h5py.File(rot_file, "r") as f:
        e_qp_ry = np.asarray(f["E_qp_nk_rydberg"])
    e_ref_ry = np.load(ref_file)
    assert e_qp_ry.shape == e_ref_ry.shape, (
        f"shape mismatch: {e_qp_ry.shape} vs frozen {e_ref_ry.shape}")
    np.testing.assert_allclose(e_qp_ry, e_ref_ry, rtol=0.0, atol=1e-6)


@pytest.mark.regression
def test_ibz_full_bz_equivalence(tmp_path):
    """IBZ cascade must reproduce full-BZ-direct to numerical precision.

    Runs the MoS2 3x3 fixture (cascade active: 5 IBZ / 9 full-BZ) with the IBZ
    cascade (default) and forced full-BZ-direct (LORRAX_FORCE_FULL_BZ=1), and
    asserts the sigma_diag rows agree.  This is the gate that catches a WRONG
    symmetry unfold: the IBZ-only regression gate cannot, because its frozen
    reference was produced with the same unfold.  No golden file -- it tests the
    physical invariant (IBZ + unfold == full BZ).  Static COHSEX is used because
    its unfold is purely algebraic (holds at ~1e-9 eV); the dynamic GN-PPM path
    shows a benign ~0.12 meV q-set path-dependence from the nonlinear PPM fit.
    """
    platform = _requested_platform()
    if platform in {"gpu", "cuda"} and not _gpu_available():
        pytest.skip("CUDA GPU not available for requested ISDF_COHSEX_TEST_PLATFORM=gpu.")
    case_dir = _REG / "gnppm_debug"          # MoS2 3x3, 642 orbit-closed centroids
    input_name, out_name = "cohsex_ibz_test.in", "sigma_diag_ibzgate.dat"
    labels = ("sigSX", "sigCOH", "sigTOT")   # static COHSEX: unfold is exact (algebraic)

    def _run(force_full_bz):
        run_dir = tmp_path / ("full" if force_full_bz else "ibz")
        shutil.copytree(case_dir, run_dir,
                        ignore=shutil.ignore_patterns("tmp", out_name, "sigma_diag.dat"))
        res = _run_gw_jax(run_dir, input_name, platform,
                          extra_env={"LORRAX_FORCE_FULL_BZ": "1" if force_full_bz else "0"})
        if res.returncode != 0:
            pytest.fail(f"LORRAX_FORCE_FULL_BZ={int(force_full_bz)} run failed.\n"
                        f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
        out = run_dir / out_name
        assert out.exists(), f"no output written: {out}"
        return _parse_eqp_rows(out, labels)

    ibz = _run(False)
    full = _run(True)
    assert ibz.shape == full.shape, f"row-count mismatch: IBZ {ibz.shape} vs full {full.shape}"
    np.testing.assert_allclose(ibz[:, :6], full[:, :6], rtol=0.0, atol=1e-6)
