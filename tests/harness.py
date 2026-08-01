"""Shared helpers for the e2e regression/invariance gates.

One home for the subprocess runner, output parsers, and fixture-dir
copying that the Tier-1 frozen gates (``test_gw_jax_regression``), the
Tier-2 invariance gates (``test_invariance_gates``), and the session
fixtures in ``conftest.py`` all use.  Not a test module.

Suite architecture (2026-07-09 redesign):

* **Tier 1** — frozen e2e pins, one fresh ``gw.gw_jax`` subprocess per
  fixture (si_cohsex_3d / cohsex / gnppm / bispinor GN-PPM).
* **Tier 2** — self-checking invariances (restart≡fresh, μ-pad flip,
  SC-iter1≡one-shot, fixed-point rotations, IBZ≡full-BZ)
  run as cheap ``restart = true`` variants from a COPY of the Tier-1
  gnppm session state (the ISDF ζ-fit + V_q are not redone).  Each
  variant copies the session ``tmp/`` because the driver WRITES W0 +
  head scalars back into the restart file (``persist_w0_and_head``).
* **Tier 3** — unit tests for what the gates cannot see.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
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
    # copytree preserves modes, and the fixtures themselves are kept
    # READ-ONLY at rest (see ``protect_fixtures``).  Restore owner-write on
    # the COPY: Tier-2 variants edit their run dir's input file
    # (``mutate_input``) and the driver rewrites tmp/ state in place.
    make_writable(run_dir)
    return run_dir


def make_writable(root: Path) -> None:
    """Give the owner write permission on ``root`` and everything under it."""
    root = Path(root)
    os.chmod(root, os.stat(root).st_mode | stat.S_IWUSR)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            try:
                os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR)
            except OSError:
                pass


def protect_fixtures(reg_root: Path = None) -> list:
    """Make every regression FIXTURE file read-only; return what it changed.

    Why this exists
    ---------------
    Gates are staged by copying ``tests/regression/<case>/`` into a scratch
    run dir.  On 2026-07-25 one sbatch stager used ``ln -sf`` instead of
    ``cp`` — the driver then wrote its ``sigma_mnk.h5`` output THROUGH the
    symlink and silently destroyed the checked-in fixture.  Nothing failed;
    the corruption was noticed by eye.

    Defence in depth, cheapest layer first:
      1. fixtures are ``a-w`` at rest (this function, called from
         ``conftest.pytest_sessionstart``) — a write through a stray
         symlink now fails loudly with EACCES;
      2. stagers copy, never link (``cp -L`` if the source may be a link);
      3. run-dir copies get owner-write back (:func:`make_writable`).

    The protected set is exactly the **git-tracked** files under
    ``tests/regression/`` — not a filename heuristic.  That distinction
    matters: ``sigma_mnk.h5`` is in ``_FIXTURE_IGNORE`` (the driver writes a
    file of that name, so it is never copied into a run dir) and is ALSO a
    checked-in reference artifact.  It is the file the 2026-07-25 incident
    destroyed. A name-based rule would have skipped precisely the victim.

    Self-healing rather than assertive: it chmods and reports.  A hard
    failure here would strand a fresh clone whose umask left files writable,
    which is every clone.  Outside a git checkout it is a no-op.
    """
    reg_root = Path(reg_root) if reg_root is not None else REG
    if not reg_root.is_dir():
        return []
    try:
        out = subprocess.run(
            ["git", "-C", str(reg_root), "ls-files", "-z", "--full-name", "."],
            capture_output=True, timeout=60)
        if out.returncode != 0:
            return []
        top = subprocess.run(
            ["git", "-C", str(reg_root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return []
    if not top:
        return []
    changed = []
    for rel in out.stdout.decode().split("\0"):
        if not rel:
            continue
        path = Path(top) / rel
        if path.is_symlink() or not path.is_file():
            continue
        mode = os.stat(path).st_mode
        ro = mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        if ro != mode:
            try:
                os.chmod(path, ro)
                changed.append(str(path))
            except OSError:
                pass
    return changed


def run_gw_jax(run_dir, input_name, platform=None, extra_env=None,
               timeout=900):
    """Run ``python -m gw.gw_jax -i <input_name>`` in run_dir; return the process."""
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
    return subprocess.run(
        [sys.executable, "-m", "gw.gw_jax", "-i", input_name],
        cwd=run_dir, env=env, capture_output=True, text=True,
        timeout=timeout, check=False,
    )


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
