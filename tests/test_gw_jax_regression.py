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


# (case_id, subdir, input_name, output_name, reference_name, sigma_labels)
# COHSEX: static Σ (sigSX/sigCOH/sigTOT) on WFNsmall; GN-PPM: dynamic Σc
# (sigX/sigC/sigXC) on a full MoS2 3×3 WFN (WFNsmall is incompatible with the
# dynamic build_G path — see reports/gw_refactor_map_2026-07-01/BUGS_FOUND.md).
_REG = REPO_ROOT / "tests" / "regression"
_CASES = [
    ("cohsex", _REG / "cohsex_debug", "cohsex_test.in", "eqp_test.dat",
     "eqp_ref.dat", ("sigSX", "sigCOH", "sigTOT")),
    ("gnppm", _REG / "gnppm_debug", "gnppm_test.in", "sigma_diag_gnppm_test.dat",
     "sigma_diag_gnppm_ref.dat", ("sigX", "sigC", "sigXC")),
]


@pytest.mark.regression
@pytest.mark.parametrize(
    "case_id,case_dir,input_name,output_name,ref_name,labels",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_gw_jax_matches_reference(
    tmp_path, case_id, case_dir, input_name, output_name, ref_name, labels
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
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
    elif platform in {"gpu", "cuda"}:
        env["JAX_PLATFORMS"] = "cuda,cpu"
        env["JAX_PLATFORM_NAME"] = "gpu"
    else:
        env.pop("JAX_PLATFORMS", None)
        env.pop("JAX_PLATFORM_NAME", None)

    src_path = str(REPO_ROOT / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_path

    cmd = [sys.executable, "-m", "gw.gw_jax", "-i", input_file.name]
    result = subprocess.run(
        cmd,
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
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
        np.testing.assert_allclose(out_rows[:, :6], ref_rows[:, :6], rtol=0.0, atol=1e-6)
    except AssertionError as exc:
        pytest.fail(f"{case_id} output differs from reference beyond tolerance.\n{exc}")
