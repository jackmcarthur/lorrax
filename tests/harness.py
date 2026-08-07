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


def _visible_gpus(preset: str | None, probe) -> list[str]:
    """The device ids this process may use, in the order it may use them."""
    if preset is not None and preset.strip() != "":
        return [d for d in preset.split(",") if d != ""]
    if preset is not None:                       # explicitly masked: ""
        return []
    return [str(i) for i in range(probe())]


def _probe_nvidia_smi() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=10).stdout
        return len(out.strip().splitlines()) if out.strip() else 0
    except Exception:                                          # noqa: BLE001
        return 0


def pin_one_gpu(preset: str | None, worker_id: str = "", probe=None):
    """The ONE device this process should see, or ``None`` for "leave it".

    ``preset`` is ``CUDA_VISIBLE_DEVICES`` as the process found it
    (``None`` = unset), ``worker_id`` is ``PYTEST_XDIST_WORKER``.  With a
    worker id the pick fans out across the visible list (``gw2`` -> the
    third one, wrapping); without one it is the FIRST visible device.

    A PURE FUNCTION on purpose.  Its caller is a module-scope side effect
    in ``tests/conftest.py`` — it has to run before the first CUDA init,
    which is the one place a test cannot observe — so the DECISION lives
    here where ``tests/test_gpu_pinning.py`` can construct every case,
    including the one that regressed.
    """
    devs = _visible_gpus(preset, probe or _probe_nvidia_smi)
    if not devs:
        return None
    # ``worker_id[2:].isdigit()`` rather than ``startswith("gw")``: the
    # xdist CONTROLLER sets no worker id at all and other spellings exist
    # ("master"), and a bare ``int(worker_id[2:])`` on one of them raises
    # ValueError out of a conftest at module scope — which pytest reports
    # as a collection error for the entire suite, not as one bad pin.
    tail = worker_id[2:] if worker_id.startswith("gw") else ""
    i = int(tail) % len(devs) if tail.isdigit() else 0
    return devs[i]

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


# ---------------------------------------------------------------------------
# BerkeleyGW anchor comparison (the ONE external check in the suite).
#
# Every other gate in this repo compares LORRAX against LORRAX's own frozen
# output.  That catches "the code changed" but is structurally blind to "the
# code drifted away from BerkeleyGW", because BGW never enters the loop.  The
# helpers below put it back in: they read a fixture of literal BGW sigma_hp.log
# columns and line it up with a LORRAX sigma_diag .dat.
#
# COLUMN CONVENTION.  BGW's 14-column sigma_hp.log block
# (Sigma/write_result_hp.f90:88-100) writes
#     CH  = ach + achcor      Sig  = asig + achcor
#     CH` = ach               Sig` = asig
# where ``achcor`` is the STATIC REMAINDER.  LORRAX computes no static
# remainder, so the comparable columns are the PRIMED ones, and the mapping
# below applies NO offset to either side:
#     LORRAX sigSX  == X + SXmX      LORRAX sigCOH == CHp      sigTOT == Sigp
# Comparing against the UNPRIMED CH instead would show a spurious ~367 meV.
# ---------------------------------------------------------------------------

BGW_HP_COLS = ("Emf", "Eo", "X", "SXmX", "CH", "Sig", "KIH", "Eqp0", "Eqp1",
               "CHp", "Sigp", "Eqp0p", "Eqp1p", "Znk")


def parse_bgw_hp_fixture(path: Path):
    """Read a bgw_sigma_hp_*.dat fixture.

    Returns ``(kfrac (nk,3), bands (nb,), {col: (nk, nb)})``.  Rows are
    ``ik kx ky kz n <14 columns>``; ``#`` lines are commentary.
    """
    rows = [ln.split() for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"no data rows in BGW fixture {path}")
    nk = max(int(r[0]) for r in rows)
    bands = sorted({int(r[4]) for r in rows})
    bpos = {b: j for j, b in enumerate(bands)}
    kfrac = np.zeros((nk, 3))
    data = {c: np.full((nk, len(bands)), np.nan) for c in BGW_HP_COLS}
    for r in rows:
        ik = int(r[0]) - 1
        kfrac[ik] = [float(r[1]), float(r[2]), float(r[3])]
        j = bpos[int(r[4])]
        for c, v in zip(BGW_HP_COLS, r[5:]):
            data[c][ik, j] = float(v)
    for c, arr in data.items():
        if np.isnan(arr).any():
            raise ValueError(f"BGW fixture {path}: column {c} has holes")
    # The fixture must be internally consistent or the parse is wrong.
    resid = np.abs(data["X"] + data["SXmX"] + data["CHp"] - data["Sigp"]).max()
    if resid > 1e-6:
        raise ValueError(
            f"BGW fixture {path} is not self-consistent: "
            f"max|X + SXmX + CHp - Sigp| = {resid:.3e} eV (expected ~1e-9). "
            f"Column order is wrong or the file was edited by hand.")
    return kfrac, np.asarray(bands), data


def compare_to_bgw(output_file: Path, fixture: Path, labels=(
        "sigSX", "sigCOH", "sigTOT")):
    """Deviation of a LORRAX sigma_diag .dat from the BGW anchor, in meV.

    Returns ``{column: (mae, max_abs)}`` plus ``"_star_spread"`` and
    ``"_nstar"``.

    Every LORRAX k-point is used, not one representative per IBZ k.  The
    fixture holds BGW's 8 IBZ k-points; LORRAX writes the full 64-point BZ.
    Each LORRAX k is assigned to the IBZ k whose mean-field energies it
    reproduces (matching on the whole ``Eo`` vector, not on k coordinates,
    because the two codes do not order k the same way).  Symmetry-equivalent
    k MUST carry identical Sigma, so comparing all of them is both the honest
    average and a symmetry check — ``_star_spread`` is the worst per-band
    disagreement between members of one star, which an IBZ-representative
    comparison cannot see.
    """
    kfrac, bands, bgw = parse_bgw_hp_fixture(fixture)
    nb = bands.size
    rows = parse_eqp_rows(output_file, labels)
    lx = {}
    for r in rows:
        lx.setdefault(int(r[0]), []).append(r)
    lx = {k: np.asarray(v) for k, v in lx.items()}
    # parse_eqp_rows gives [kpt, band, A, B, C, VH_re, VH_im]; the driver also
    # writes an Eo column, which parse_eqp_rows drops — re-read it here since
    # the k-matching needs it.
    eo = _parse_eo_column(output_file)

    ref = {labels[0]: bgw["X"] + bgw["SXmX"],
           labels[1]: bgw["CHp"],
           labels[2]: bgw["Sigp"]}
    acc = {c: [] for c in labels}
    star_spread, nstar_total = 0.0, 0
    for ik in range(kfrac.shape[0]):
        emf = bgw["Emf"][ik, :nb]
        hits = [k for k, v in lx.items()
                if v.shape[0] >= nb and k in eo
                and np.max(np.abs(np.asarray(eo[k][:nb]) - emf)) < 2e-3]
        if not hits:
            raise AssertionError(
                f"BGW anchor: no LORRAX k-point reproduces the mean-field "
                f"energies of BGW IBZ k{ik+1} = {kfrac[ik]}.  Either the run "
                f"used a different WFN or the band ordering changed.")
        nstar_total += len(hits)
        for j, c in enumerate(labels):
            for k in hits:
                acc[c].append(lx[k][:nb, 2 + j] - ref[c][ik])
        if len(hits) > 1:
            block = np.stack([lx[k][:nb, 4] for k in hits])   # sigTOT
            star_spread = max(star_spread,
                              float((block.max(0) - block.min(0)).max()))
    out = {}
    for c in labels:
        d = np.concatenate(acc[c]) * 1e3
        out[c] = (float(np.abs(d).mean()), float(np.abs(d).max()))
    out["_star_spread"] = star_spread * 1e3
    out["_nstar"] = nstar_total
    return out


def _parse_eo_column(path: Path) -> dict:
    """{kpt: [Eo per band]} from a sigma_diag .dat (the ``Eo=`` field)."""
    out, ik = {}, -1
    for ln in Path(path).read_text().splitlines():
        m = re.search(r"k-point\s+(\d+)\s*:", ln)
        if m:
            ik = int(m.group(1))
            out[ik] = []
            continue
        m = re.search(r"\bEo=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", ln)
        if m and ik >= 0:
            out[ik].append(float(m.group(1)))
    return {k: v for k, v in out.items() if v}
