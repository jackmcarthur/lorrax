"""Shared helpers for the e2e regression/invariance gates.

One home for the subprocess runner, output parsers, and fixture-dir
copying that the Tier-1 frozen gates (``test_gw_jax_regression``), the
Tier-2 invariance gates (``test_invariance_gates``), and the session
fixtures in ``conftest.py`` all use.  Not a test module.

Suite architecture (2026-07-09 redesign):

* **Tier 1** — frozen e2e pins, one fresh ``gw.gw_jax`` subprocess per
  fixture (si_cohsex_3d / cohsex / gnppm / bispinor GN-PPM).
* **Tier 2** — self-checking invariances (restart≡fresh, μ-pad flip,
  kij↔kij_stream, SC-iter1≡one-shot, fixed-point rotations, IBZ≡full-BZ)
  run as cheap ``restart = true`` variants from a COPY of the Tier-1
  gnppm session state (the ISDF ζ-fit + V_q are not redone).  Each
  variant copies the session ``tmp/`` because the driver WRITES W0 +
  head scalars back into the restart file (``persist_w0_and_head``).
* **Tier 3** — unit tests for what the gates cannot see.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REG = REPO_ROOT / "tests" / "regression"

# Output files never copied from a fixture dir into a run dir.
_FIXTURE_IGNORE = (
    "tmp", "eqp_test.dat", "eqp0_test.dat", "eqp1_test.dat",
    "sigma_diag*.dat", "eqp0.dat", "eqp1.dat", "eqp_g0w0.dat",
    "sigma_mnk.h5", "*_qp.h5", "qp_wfn_rotations.h5",
)


def gpu_available() -> bool:
    try:
        import jax

        return any(getattr(dev, "platform", "") in {"gpu", "cuda"}
                   for dev in jax.devices())
    except Exception:
        return False


def requested_platform() -> str:
    # Default to JAX's native backend selection (typically GPU on test nodes).
    platform = os.environ.get("ISDF_COHSEX_TEST_PLATFORM", "auto").strip().lower()
    valid = {"cpu", "gpu", "cuda", "auto"}
    if platform not in valid:
        raise ValueError(
            f"Invalid ISDF_COHSEX_TEST_PLATFORM={platform!r}. "
            f"Expected one of {sorted(valid)}."
        )
    return platform


def skip_unless_gpu(pytest):
    """Common gate: skip when the requested platform needs a missing GPU."""
    if requested_platform() in {"gpu", "cuda"} and not gpu_available():
        pytest.skip("CUDA GPU not available for the requested platform.")


def copy_fixture(case_dir: Path, run_dir: Path, *, tmp_from: Path = None):
    """Copy a regression fixture dir into a run dir, minus outputs.

    ``tmp_from`` (a previous run dir) additionally copies its ``tmp/``
    (the ISDF restart state) — used by the Tier-2 from-restart variants.
    Each variant needs its OWN copy: the driver mutates the restart file
    in place (``persist_w0_and_head`` writes W0_qmunu + head scalars back).
    """
    shutil.copytree(
        case_dir, run_dir,
        ignore=shutil.ignore_patterns(*_FIXTURE_IGNORE))
    if tmp_from is not None:
        src_tmp = Path(tmp_from) / "tmp"
        assert src_tmp.is_dir(), f"no restart state to copy: {src_tmp}"
        shutil.copytree(src_tmp, run_dir / "tmp")
    return run_dir


def gw_env(platform=None, extra_env=None):
    """The environment every test-driven gw.gw_jax process runs under."""
    if platform is None:
        platform = requested_platform()
    env = os.environ.copy()
    cache_dir = Path(env.get("JAX_COMPILATION_CACHE_DIR",
                             str(REPO_ROOT / ".pytest_jax_cache")))
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
    env["PYTHONPATH"] = src_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if extra_env:
        env.update(extra_env)
    return env


def run_gw_jax(run_dir, input_name, platform=None, extra_env=None,
               timeout=900):
    """Run ``python -m gw.gw_jax -i <input_name>`` in run_dir; return the process."""
    return subprocess.run(
        [sys.executable, "-m", "gw.gw_jax", "-i", input_name],
        cwd=run_dir, env=gw_env(platform, extra_env),
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def run_variant_bundle(variants, spec_dir, platform=None, timeout=1800):
    """Run prepared gw_jax variants in ONE subprocess (tests/run_variant_bundle.py).

    ``variants`` = [{"name", "run_dir", "input_name", "env"}] with run dirs
    already prepared (copy_fixture + mutate_input).  Returns
    {name: (status, stdout)} — stdout from <run_dir>/variant_stdout.txt.
    Raises AssertionError only if the bundle process itself died before
    writing results; per-variant failures are reported in the mapping so
    each test fails on its own variant.
    """
    spec_dir = Path(spec_dir)
    spec_path = spec_dir / "bundle_spec.json"
    results_path = spec_dir / "bundle_results.json"
    spec_path.write_text(json.dumps(
        {"variants": [{**v, "run_dir": str(v["run_dir"])} for v in variants]}))
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tests" / "run_variant_bundle.py"),
         str(spec_path), str(results_path)],
        cwd=spec_dir, env=gw_env(platform), capture_output=True, text=True,
        timeout=timeout, check=False,
    )
    assert results_path.exists(), (
        f"variant bundle process died before writing results "
        f"(rc={proc.returncode}).\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}")
    results = {r["name"]: r for r in json.loads(results_path.read_text())}
    out = {}
    for v in variants:
        r = results.get(v["name"], {"status": "missing", "error": ""})
        stdout_file = Path(v["run_dir"]) / "variant_stdout.txt"
        stdout = stdout_file.read_text() if stdout_file.exists() else ""
        if r["status"] != "ok" and r.get("error"):
            stdout += f"\n[bundle error]\n{r['error']}"
        out[v["name"]] = (r["status"], stdout)
    return out


def parse_eqp_rows(path: Path, labels=("sigSX", "sigCOH", "sigTOT")) -> np.ndarray:
    """Parse sigma_diag rows → (nrows, 7): kpt, band, 3 Σ columns, VH re/im."""
    float_re = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    imag_opt = rf"(?:\+\s*{float_re}i)?"  # optional imaginary part
    a, b, c = labels  # COHSEX: sigSX/sigCOH/sigTOT ;  GN-PPM: sigX/sigC/sigXC
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
        # Groups: 2=A_re, 3=A_im, 4=B_re, 5=B_im, 6=C_re, 7=C_im, 8=VH_re, 9=VH_im
        rows.append([float(kpt), float(band), float(m.group(2)),
                     float(m.group(4)), float(m.group(6)), float(m.group(8)),
                     float(m.group(9)) if m.group(9) else 0.0])
    if not rows:
        raise ValueError(f"No Sigma data rows were parsed from {path}")
    return np.asarray(rows, dtype=np.float64)


def normalize_dat(text: str) -> str:
    """Drop the run-timestamp header ('# Generated by LORRAX ... at ...');
    every numeric byte still participates in identity checks."""
    return "\n".join(
        ln for ln in text.splitlines()
        if not ln.startswith("# Generated by LORRAX")
    )


def census_lines(log_text: str) -> tuple:
    """PPM census + adaptive-window signature from a run log — the integer
    quantities that must be exactly invariant under μ-pad flips."""
    windows = tuple(re.findall(
        r'window "(\w+)" \((?:crossing|Laplace)\): (\d+) nodes', log_text))
    m = re.search(r"GN invalid modes: (\d+)/(\d+)", log_text)
    u = re.search(r"unfulfilled=([\d.]+)%", log_text)
    return (windows,
            m.groups() if m else None,
            u.group(1) if u else None)


def eqp_column(path: Path) -> np.ndarray:
    """E_qp column of an eqp0/eqp1.dat file (data rows are 'ik n Edft Eqp';
    k-point header rows are 4 floats — distinguished by '.' in field 0)."""
    vals = []
    for ln in path.read_text().splitlines():
        s = ln.split()
        if len(s) == 4 and not ln.startswith('#') and '.' not in s[0]:
            vals.append(float(s[3]))
    return np.asarray(vals, dtype=np.float64)


def numeric_tokens(path: Path) -> np.ndarray:
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


def mutate_input(path: Path, replacements: dict[str, str], append: str = ""):
    """Apply exact-string replacements to an input file (each must hit)."""
    text = path.read_text()
    for old, new in replacements.items():
        assert old in text, f"{path}: expected {old!r} in input"
        text = text.replace(old, new)
    if append:
        text += "\n" + append + "\n"
    path.write_text(text)
