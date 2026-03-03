import os
import re
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
    platform = os.environ.get("ISDF_COHSEX_TEST_PLATFORM", "cpu").strip().lower()
    valid = {"cpu", "gpu", "cuda", "auto"}
    if platform not in valid:
        raise ValueError(
            f"Invalid ISDF_COHSEX_TEST_PLATFORM={platform!r}. "
            f"Expected one of {sorted(valid)}."
        )
    return platform


def _parse_eqp_rows(path: Path) -> np.ndarray:
    float_re = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    data_re = re.compile(
        rf"n=\s*(\d+)\s+sigSX=\s*{float_re}\s+sigCOH=\s*{float_re}\s+"
        rf"sigTOT=\s*{float_re}\s+VH=\s*{float_re}\+\s*{float_re}i"
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
        values = [float(m.group(i)) for i in range(2, 7)]
        rows.append([float(kpt), float(band), *values])

    if not rows:
        raise ValueError(f"No COHSEX data rows were parsed from {path}")
    return np.asarray(rows, dtype=np.float64)


@pytest.mark.regression
def test_cohsex_jax_matches_reference():
    platform = _requested_platform()
    if platform in {"gpu", "cuda"} and not _gpu_available():
        pytest.skip("CUDA GPU not available for requested ISDF_COHSEX_TEST_PLATFORM=gpu.")

    assert INPUT_FILE.exists(), f"Missing regression input: {INPUT_FILE}"
    assert REFERENCE_FILE.exists(), f"Missing regression reference: {REFERENCE_FILE}"

    OUTPUT_FILE.unlink(missing_ok=True)

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

    cmd = [sys.executable, "-m", "isdf.gw_isdf.cohsex_jax", "-i", INPUT_FILE.name]
    result = subprocess.run(
        cmd,
        cwd=CASE_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "COHSEX regression run failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    assert OUTPUT_FILE.exists(), f"Expected output file was not written: {OUTPUT_FILE}"

    ref_text = REFERENCE_FILE.read_text()
    out_text = OUTPUT_FILE.read_text()
    if out_text == ref_text:
        return

    ref_rows = _parse_eqp_rows(REFERENCE_FILE)
    out_rows = _parse_eqp_rows(OUTPUT_FILE)
    assert out_rows.shape == ref_rows.shape, (
        f"Row-count mismatch: output shape {out_rows.shape}, reference shape {ref_rows.shape}"
    )

    try:
        np.testing.assert_allclose(out_rows, ref_rows, rtol=0.0, atol=1e-6)
    except AssertionError as exc:
        pytest.fail(f"COHSEX output differs from reference beyond tolerance.\n{exc}")
