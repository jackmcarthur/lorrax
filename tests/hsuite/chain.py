"""The driver chain on the magnetic H2+ spinor fixture, at P4 on one node.

Every stage is a production driver's entry point, called in sequence in one
Python process on each of the four srun ranks (one GPU each, MPI world 4;
one runtime, one FFI load, one compile cache), in one shared run directory,
in the order a user runs them:

    kmeans -> kin_ion -> dipole -> gwjax GN-PPM one-shot
           -> gwjax shared-pole QSGW (2 maps) -> BSE -> htransform
           -> exciton bands -> restarted: COHSEX, GN-PPM SC (1 map),
              shared-pole one-shot with W and pole exports
           -> bispinor kin_ion -> four-current shared-pole one-shot
           -> restarted: four-current shared-pole QSGW (2 maps)

Each stage is then checked against the stored outputs in ``reference/``
(eqp columns, numeric members of the written h5 files, solver outputs)
within the tolerances below, and every rank log is scanned for failure
signatures.  An exit code alone is never a pass.

    lx run -N 1 -G 4 -n 4 -- python3 -m tests.hsuite.chain --out DIR
    lx run -N 1 -G 4 -n 4 -- python3 -m tests.hsuite.chain --out DIR --regenerate

The second rewrites ``reference/``.  A local four-process launch is not
enough: the drivers' parallel HDF5 needs an MPI world of four, which only
srun provides.  At one rank (``lx test``'s launch shape) the same chain runs
at P1 with a 1x1 excited-state mesh and is checked against the same P4
references.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from tests.hsuite import rank_session  # noqa: E402

FIXTURE = HERE / "fixture"
REFERENCE = HERE / "reference"

# Tolerances.  Static objects (centroid-free preprocessing, ISDF fits) are
# deterministic to round-off; Sigma carries the quadrature rule set, which is
# rebuilt every run (no rule cache) and may split boxes differently by P, so
# the Sigma tolerance is the 0.5 meV rule-set budget of the old core GN cell.
ATOL = {
    "h5": 1.0e-8,        # kin_ion / dipole members (relative + absolute)
    "w_bank": 1.0e-6,    # exported shared-pole W samples (relative)
    "eqp_ev": 5.0e-4,    # eqp0/eqp1 columns, GN-PPM and shared pole
    "bse_ev": 2.0e-3,    # BSE and exciton-band eigenvalues: Krylov solves
                         # on a P-dependent padded space (P1 vs P4 1.2 meV)
}

# h5 members not compared.  qp_diag_self_consistent_ev is a diagnostic
# diagonal fixed point E = h0 + Re Sigma(E) with plain mixing; on this fixture
# (SC map gain ~5) it lands on different roots from Sigma inputs that agree to
# 30 ueV (P1 vs P4: 0.79 eV on one state).  The terminal eqp columns and the
# Sigma matrices it is built from are compared instead.  line_charge_* are
# the W bank's selected direction states: a gauge per direction, and padded
# to the mesh (P1 6x1, P4 8x2); Wc, dWc_ds and the moments are compared.
H5_UNCOMPARED = ("qp_diag_self_consistent_ev", "_w.h5:line_charge_")

FAILURE_SIGNATURES = (
    "Traceback (most recent call last)",
    "Fatal Python error",
    "Segmentation fault",
    "CUDA_ERROR",
    "CUBLAS_STATUS_",
    "CUFFT_",
    "RESOURCE_EXHAUSTED",
    "out of memory",
    "MPI_Abort",
    "TIMED OUT",
)

_DECK_COMMON = """[cohsex]
centroids_file = {centroids}
nval = 1
ncond = 2
number_bands = 7
sys_dim = 3
bispinor = false
fermi_reference = midgap
wfn_file = WFN.h5
kin_ion_file = kin_ion.h5
write_restart_tensors = true
sigma_freq_debug_output = false
"""

# GN-PPM one-shot: fresh zeta fit, local dense algebra, default heads (the
# dipole written by the stage before) and default band extrapolation.
GNPPM_DECK = _DECK_COMMON + """restart = false
compute_mode = gn_ppm
qp_solver = one_shot_dft
linalg = local
sigma_regularization_ev = 0.25
sigma_diag_file = gnppm_sigma.dat
eqp0_file = gnppm_eqp0.dat
eqp1_file = gnppm_eqp1.dat
report_file = gnppm.out
sigma_omega_h5_file = gnppm_sigma.h5
"""

# The production route: full-frequency QSGW with the shared-pole W, two maps,
# distributed dense algebra (local at P1, where cuSolverMp has no 2-D mesh),
# restarting from the GN-PPM run's zeta.  Time
# reversal is broken, so the store is ordered and its head is the direct
# frequency-dependent one (`full` folds wings through a TR-even body and
# refuses: GATE shared_pole_head_ordered).  One electron gives the head too
# few poles for the default 8-pole head fit, whose conditioning gate refuses.
# The fixture's QSGW maps move states by eV (max|dE| 2.18 then 1.20 eV), so
# the 1.5 eV criterion stops the run after exactly two maps, converged, with
# its terminal outputs written; the per-map residuals are checked below.
SP_SC_DECK = _DECK_COMMON + """restart = true
compute_mode = mpa
sigma_w_model = shared_pole
head_correction = no_local_fields
mpa_n_poles = 2
qp_solver = self_consistent
sc_max_iter = 2
sc_tol_ev = 1.5
linalg = {linalg}
sigma_regularization_ev = 0.25
write_qsgw_datasets = true
sigma_diag_file = sp_sigma.dat
eqp0_file = sp_eqp0.dat
eqp1_file = sp_eqp1.dat
report_file = sp.out
sigma_omega_h5_file = sp_sigma.h5
"""

_PATH = """
K_POINTS {{crystal_b}}
3
  0.000000  0.000000  0.000000  1  # G
  0.400000  0.000000  0.000000  1  # X
  0.400000  0.400000  0.000000  1  # M
"""

# Five k-points per side leave the Galerkin basis no room below the full
# state count at the default QRCP tolerance (1e-3 saturates at 80 of 80).
EXCITED_DECK = _DECK_COMMON + """restart = true
compute_mode = cohsex
qp_solver = one_shot_dft
linalg = local
htransform_qr_eps = 1e-2
""" + _PATH

_SIDE = "2" if rank_session._resolve_proc_count() == 4 else "1"

# Restarted steps: each reads the tmp/ state (zeta, V(q), W0) the chain has
# written and covers one more route.
def _restart_deck(prefix, body):
    return _DECK_COMMON + "restart = true\nlinalg = local\n" + body + f"""\
sigma_diag_file = {prefix}_sigma.dat
eqp0_file = {prefix}_eqp0.dat
eqp1_file = {prefix}_eqp1.dat
report_file = {prefix}.out
sigma_omega_h5_file = {prefix}_sigma.h5
"""


RESTART_DECKS = {
    "cohsex.in": _restart_deck("cohsex", """compute_mode = cohsex
qp_solver = one_shot_dft
"""),
    # GN-PPM self-consistency, one map: map 0's max|dE| (2.59 eV) is inside
    # the 3 eV criterion, so the SC driver stops converged after one map.
    "gnppm_sc.in": _restart_deck("gnsc", """compute_mode = gn_ppm
qp_solver = self_consistent
sc_max_iter = 1
sc_tol_ev = 3.0
sigma_regularization_ev = 0.25
write_qsgw_datasets = true
"""),
    # The shared-pole file-model path: one-shot with the W bank and the
    # pole model exported.
    "sp_export.in": _restart_deck("spx", """compute_mode = mpa
sigma_w_model = shared_pole
head_correction = no_local_fields
mpa_n_poles = 2
qp_solver = one_shot_dft
sigma_regularization_ev = 0.25
write_w = true
write_poles = true
"""),
}

# The four-current route, last in the chain: its fresh bispinor zeta fit
# rewrites the tmp/ restart the scalar restarted steps read.  SP-full
# (bispinor_gw = full_shared_pole: ordered CC/CT/TC/TT sector poles, direct
# four-current Gamma head) one-shot with its own kinetic-balance kin_ion,
# then the same route through the SC driver for two maps, restarted from
# the one-shot's zeta and V(q), with the per-map head (dft_velocity) and the
# live four-current density (density_self_consistent, required).
_BISP_COMMON = """bispinor = true
bispinor_gw = full_shared_pole
compute_mode = mpa
sigma_w_model = shared_pole
head_correction = no_local_fields
mpa_n_poles = 2
sigma_regularization_ev = 0.25
linalg = local
"""


def _bisp_deck(prefix, body):
    return (_DECK_COMMON.replace("bispinor = false\n", "")
            .replace("kin_ion_file = kin_ion.h5", "kin_ion_file = kin_ion_bisp.h5")
            + _BISP_COMMON + body + f"""\
sigma_diag_file = {prefix}_sigma.dat
eqp0_file = {prefix}_eqp0.dat
eqp1_file = {prefix}_eqp1.dat
report_file = {prefix}.out
sigma_omega_h5_file = {prefix}_sigma.h5
""")


RESTART_DECKS.update({
    "bisp_os.in": _bisp_deck("bos", """restart = false
qp_solver = one_shot_dft
"""),
    "bisp_sc.in": _bisp_deck("bsc", """restart = true
qp_solver = self_consistent
sc_max_iter = 2
sc_tol_ev = 1.5
sc_head_update = dft_velocity
density_self_consistent = true
"""),
})
_P = ["--px", _SIDE, "--py", _SIDE]

# (name, module, argv, deck name -> template).  The decks are written into
# the run directory after kmeans, which names the centroid file.
STAGES = (
    ("kmeans", "centroid.kmeans_cli",
     ["6", "--seed", "42", "--force-shard", "--orbit", "--oversample", "1.5",
      "--fit-window", "0:1,0:7", "--density-mode", "scalar"]),
    ("kin_ion", "gw.kin_ion_io", ["-i", "gnppm.in"]),
    ("dipole", "psp.get_dipole_mtxels", ["-i", "gnppm.in"]),
    ("gnppm", "gw.gw_jax", ["-i", "gnppm.in"]),
    ("shared_pole_sc", "gw.gw_jax", ["-i", "sp_sc.in"]),
    # A regression check, not physics: the DFT gap is 48 meV between the
    # exchange-split bonding states, far below the ~1 eV exciton binding of
    # this unscreened one-electron molecule, so the TDA eigenvalues come out
    # negative (-1.19, -0.81 eV).
    ("bse", "bse.bse_jax",
     ["-i", "gnppm.in", "--bse", "--lanczos", "--tda", "--solver", "davidson",
      "--n-val", "1", "--n-cond", "2", "--n-occ", "1",
      "--band-degeneracy", "off", "--max-lanczos-iter", "40",
      "--n-eig", "2", "--block-size", "1", *_P,
      "--report-file", "bse.out"]),
    ("htransform", "bandstructure.htransform",
     ["-i", "excited.in", "--guard-bands", "1", "-o", "htransform.dat",
      "--report-file", "htransform.out"]),
    # One conduction band: the window's top band must lie below the fitted
    # window's top, or fH cannot see it (compute_wfns_fi refuses).
    ("exciton_bands", "bse.exciton_bands",
     ["-i", "excited.in", "--n-val", "1", "--n-cond", "1", "--n-eig", "1",
      "--block-size", "1", "--max-iter", "40", "--vq-mode", "ongrid",
      "--q-per-segment", "1", "--band-degeneracy", "off", *_P,
      "--out-prefix", "exciton", "--report-file", "exciton.out"]),
    ("cohsex", "gw.gw_jax", ["-i", "cohsex.in"]),
    ("gnppm_sc", "gw.gw_jax", ["-i", "gnppm_sc.in"]),
    ("sp_export", "gw.gw_jax", ["-i", "sp_export.in"]),
    ("kin_ion_bisp", "gw.kin_ion_io", ["-i", "bisp_os.in"]),
    ("bisp_oneshot", "gw.gw_jax", ["-i", "bisp_os.in"]),
    ("bisp_sc", "gw.gw_jax", ["-i", "bisp_sc.in"]),
)

# What each stage leaves behind and how it is compared.
#   eqp: eqp files (column compare); h5: numeric members; rows: whitespace
#   tables (numeric columns after `skip`); text: regex list captured from a
#   report and compared as floats.
CHECKS = {
    "kin_ion": {"h5": ["kin_ion.h5"]},
    "dipole": {"h5": ["dipole.h5"]},
    "gnppm": {"eqp": ["gnppm_eqp0.dat", "gnppm_eqp1.dat"],
              "h5": ["gnppm_sigma.h5"]},
    "shared_pole_sc": {"eqp": ["sp_eqp0.dat", "sp_eqp1.dat"],
                       "h5": ["sp_sigma.h5"],
                       "report_floats": ("sp.out",
                                         r"SC iteration: call=\d+ .*?max\|dE\|=([0-9.e+-]+)")},
    "bse": {"stdout_floats": r"^\s*S\d+\s+([0-9.+-]+)\s*$"},
    "exciton_bands": {"rows": [("exciton.dat", 6)]},
    "htransform": {"rows": [("htransform.dat", 6)]},
    "cohsex": {"eqp": ["cohsex_eqp0.dat", "cohsex_eqp1.dat"]},
    "gnppm_sc": {"eqp": ["gnsc_eqp0.dat", "gnsc_eqp1.dat"],
                 "h5": ["gnsc_sigma.h5"],
                 "report_floats": ("gnsc.out",
                                   r"SC iteration: call=\d+ .*?max\|dE\|=([0-9.e+-]+)")},
    # The exported W bank is compared by value; the pole model (b, Lambda)
    # has a gauge per pole, so only its members' shapes are pinned.
    "sp_export": {"eqp": ["spx_eqp0.dat", "spx_eqp1.dat"],
                  "h5": ["spx_sigma.h5", "tmp/mpa/oneshot_w.h5"],
                  "shapes": ["tmp/mpa/oneshot_poles.h5"]},
    "kin_ion_bisp": {"h5": ["kin_ion_bisp.h5"]},
    "bisp_oneshot": {"eqp": ["bos_eqp0.dat", "bos_eqp1.dat"],
                     "h5": ["bos_sigma.h5"]},
    "bisp_sc": {"eqp": ["bsc_eqp0.dat", "bsc_eqp1.dat"],
                "h5": ["bsc_sigma.h5"],
                "report_floats": ("bsc.out",
                                  r"SC iteration: call=\d+ .*?max\|dE\|=([0-9.e+-]+)")},
}


def _env(cache_dir):
    """The drivers' environment, set in this process before any driver import."""
    env = os.environ
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    if rank_session._resolve_proc_count() == 1:
        # P1 (lx test): the drivers run on ONE device.  A single process
        # holding four devices is the in-process mesh the flat-k FFT refuses.
        visible = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        env["CUDA_VISIBLE_DEVICES"] = visible[0]
    env["ISDF_JAX_CACHE_DIR"] = str(cache_dir)
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def stage_fixture(run):
    run.mkdir(parents=True, exist_ok=False)
    for path in FIXTURE.iterdir():
        shutil.copy2(path, run / path.name)
        os.chmod(run / path.name, 0o644)


def write_decks(run):
    found = sorted(p.name for p in run.glob("centroids_frac_*.txt"))
    assert len(found) == 1, f"kmeans left {found}"
    centroids = found[0]
    (run / "gnppm.in").write_text(GNPPM_DECK.format(centroids=centroids))
    (run / "sp_sc.in").write_text(SP_SC_DECK.format(
        centroids=centroids, linalg="distributed" if _SIDE == "2" else "local"))
    (run / "excited.in").write_text(EXCITED_DECK.format(centroids=centroids))
    for fname, deck in RESTART_DECKS.items():
        (run / fname).write_text(deck.format(centroids=centroids))
    return centroids


def _release_devices():
    """Drop what one driver left behind before the next one starts."""
    import gc
    import jax
    gc.collect()
    jax.clear_caches()
    gc.collect()


def run_stage(run, name, module, argv, env, timeout):
    """Run one driver's entry point in this process on every rank.

    One runtime per rank for the whole chain: one jax.distributed world,
    one FFI load, one compile cache.  The driver's stdout/stderr (Python
    and native) go to ``<name>.rank<r>.log``.  A driver that fails on one
    rank would leave the others in a collective, so a failure exits the
    process at once and srun ends the step.
    """
    import importlib
    import inspect
    import traceback
    t0 = time.monotonic()
    rank = rank_session._resolve_proc_id()
    log = run / f"{name}.rank{rank}.log"
    sys.stdout.flush()
    sys.stderr.flush()
    # Both levels: native writes go to fds 1/2, Python writes to sys.stdout /
    # sys.stderr, which pytest's capture replaces with objects that never
    # reach fd 1.  One append-mode stream on the log serves both.
    log.write_text("")
    stream = open(log, "a", buffering=1, encoding="utf-8", errors="replace")
    saved = (os.dup(1), os.dup(2))
    saved_py = (sys.stdout, sys.stderr)
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
    sys.stdout = sys.stderr = stream
    cwd, saved_argv = os.getcwd(), sys.argv
    rc = 0
    try:
        os.chdir(run)
        sys.argv = [module, *argv]
        main = importlib.import_module(module).main
        try:
            ret = main(argv) if inspect.signature(main).parameters else main()
            rc = int(ret or 0)
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else int(exc.code is not None)
        except BaseException:                                  # noqa: BLE001
            traceback.print_exc()
            rc = 1
    finally:
        stream.flush()
        sys.stdout, sys.stderr = saved_py
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        stream.close()
        os.chdir(cwd)
        sys.argv = saved_argv
    if rc != 0:
        print(f"hsuite FAIL {name}: rc={rc} on rank {rank}; see {log}", flush=True)
        os._exit(rc)
    _release_devices()
    rcs = rank_session.exchange(rc)
    return all(r == 0 for r in rcs), rcs, time.monotonic() - t0


def signatures(run, name):
    hits = []
    for log in sorted(run.glob(f"{name}.rank*.log")):
        text = log.read_text(errors="replace")
        hits += [f"{log.name}: {sig}" for sig in FAILURE_SIGNATURES if sig in text]
    return hits


def _eqp(path):
    rows = [line.split() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    return np.asarray([[float(v) for v in row] for row in rows
                       if all(_isfloat(v) for v in row)])


def _isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _rows(path, skip):
    return np.asarray([[float(v) for v in line.split()[skip:]]
                       for line in Path(path).read_text().splitlines()
                       if line.strip() and not line.startswith("#")])


def _h5_members(path):
    import h5py
    out = {}
    with h5py.File(path, "r") as h5:
        def visit(key, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fciub":
                value = np.asarray(obj[()])
                out[key] = value.astype(np.int64) if value.dtype.kind in "ub" else value
        h5.visititems(visit)
    return out


def captured(run, name, rule):
    """The numbers a stage is judged by, as {label: array}."""
    got = {}
    for fname in rule.get("eqp", []):
        got[fname] = _eqp(run / fname)
    for fname in rule.get("h5", []):
        for key, value in _h5_members(run / fname).items():
            got[f"{fname}:{key}"] = value
    for fname in rule.get("shapes", []):
        for key, value in _h5_members(run / fname).items():
            got[f"{fname}:{key}:shape"] = np.asarray(value.shape)
    for fname, skip in rule.get("rows", []):
        got[fname] = _rows(run / fname, skip)
    if "report_floats" in rule:
        fname, pattern = rule["report_floats"]
        got[f"{fname}:residuals"] = np.asarray([float(v) for v in re.findall(
            pattern, (run / fname).read_text(errors="replace"))])
    if "stdout_floats" in rule:
        text = (run / f"{name}.rank0.log").read_text(errors="replace")
        got["stdout"] = np.asarray([float(v) for v in re.findall(
            rule["stdout_floats"], text, re.MULTILINE)])
    return got


def _tol(label):
    """(atol, relative?) for one captured array."""
    if label.endswith(":shape"):
        return 0.0, False
    if "_w.h5:" in label:
        # The W bank: chi0 and V(q) reductions reorder with P.
        return ATOL["w_bank"], True
    if "sigma" in label or "eqp" in label:
        # Sigma-derived: absolute, in the file's energy unit (eV).
        return ATOL["eqp_ev"], False
    if ".h5:" in label:
        return ATOL["h5"], True
    return ATOL["bse_ev"], False


def compare(name, got, ref):
    problems = []
    for label, want in ref.items():
        if any(tag in label for tag in H5_UNCOMPARED):
            continue
        if label not in got:
            problems.append(f"{name}: {label} missing")
            continue
        have = got[label]
        if have.shape != want.shape:
            problems.append(f"{name}: {label} shape {have.shape} != {want.shape}")
            continue
        if have.size and not np.all(np.isfinite(have)):
            problems.append(f"{name}: {label} has non-finite values")
            continue
        tol, relative = _tol(label)
        scale = max(1.0, float(np.max(np.abs(want)))) if (
            relative and want.size) else 1.0
        if label.endswith("_kij_ev"):
            # Band-basis matrices carry the wavefunction phases, which the
            # eigensolver may choose differently (P1 vs P4): compare the
            # diagonal and the element moduli, which a phase change keeps.
            have = np.concatenate([np.einsum("...ii->...i", have).ravel(),
                                   np.abs(have).ravel()])
            want = np.concatenate([np.einsum("...ii->...i", want).ravel(),
                                   np.abs(want).ravel()])
        err = float(np.max(np.abs(have - want))) if want.size else 0.0
        if err > tol * scale:
            problems.append(f"{name}: {label} max|diff| {err:.3e} > {tol * scale:.1e}")
    for label in got:
        if label not in ref:
            problems.append(f"{name}: {label} not in the reference")
    return problems


def _save(name, got):
    np.savez(REFERENCE / f"{name}.npz", **{k.replace("/", "|"): v
                                           for k, v in got.items()})


def _load(name):
    path = REFERENCE / f"{name}.npz"
    if not path.is_file():
        return None
    with np.load(path) as ref:
        return {k.replace("|", "/"): ref[k] for k in ref.files}


def run_chain(out, *, regenerate=False, cache_dir=None, timeout=600,
              only=None):
    """Run every stage on every rank; return (walls, problems) on all ranks."""
    out = Path(out).resolve()
    run = out / "run"
    lead = rank_session._resolve_proc_id() == 0
    cache_dir = Path(cache_dir) if cache_dir else out / "jax-cache"
    if lead:
        stage_fixture(run)
        cache_dir.mkdir(parents=True, exist_ok=True)
    rank_session.exchange("staged")
    env = _env(cache_dir)
    walls, problems = {}, []
    t_all = time.monotonic()
    for name, module, argv in STAGES:
        if only and name not in only:
            continue
        if name == "kin_ion" and lead:
            write_decks(run)
        rank_session.exchange(name)
        ok, rcs, walls[name] = run_stage(run, name, module, argv, env, timeout)
        hits = signatures(run, name) if lead else []
        # One verdict for every rank: a rank that walks on alone hangs.
        if rank_session.exchange(not ok or bool(hits))[0]:
            problems.append(f"{name}: rc={rcs} {hits[:4]}")
            break
        if not lead:
            continue
        rule = CHECKS.get(name, {})
        if name == "kmeans":
            found = list(run.glob("centroids_frac_*.txt"))
            pts = np.loadtxt(found[0]) if len(found) == 1 else np.zeros((0, 3))
            if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
                problems.append("kmeans: centroid file malformed")
            got = {"centroids": pts}
        else:
            got = captured(run, name, rule)
        if regenerate == "all" or (regenerate == "missing" and _load(name) is None):
            REFERENCE.mkdir(exist_ok=True)
            _save(name, got)
            continue
        ref = _load(name)
        if ref is None:
            problems.append(f"{name}: no stored reference (run --regenerate)")
            continue
        if name == "kmeans":
            # The centroid set is compared as a set: row order is not a
            # contract, and every later stage checks what it feeds.
            a = np.round(np.sort(got["centroids"], axis=0), 6)
            b = np.round(np.sort(ref["centroids"], axis=0), 6)
            if a.shape != b.shape or np.max(np.abs(a - b)) > 1e-5:
                problems.append("kmeans: centroid set differs from reference")
            continue
        problems += compare(name, got, ref)
    walls["total"] = time.monotonic() - t_all
    if lead:
        (out / "summary.json").write_text(json.dumps(
            {"walls_s": walls, "problems": problems, "regenerate": regenerate,
             "ranks": rank_session._resolve_proc_count(),
             "cache_dir": str(cache_dir)}, indent=1))
        if regenerate:
            (REFERENCE / "walls.json").write_text(json.dumps(walls, indent=1))
    return walls, rank_session.exchange(problems)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="new directory for the run")
    parser.add_argument("--regenerate", nargs="?", const="all", default=None,
                        choices=("all", "missing"),
                        help="rewrite reference/ from this run (all), or write "
                             "only the stages that have none (missing)")
    parser.add_argument("--cache-dir", default=None,
                        help="compile cache (default: <out>/jax-cache, cold)")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)
    walls, problems = run_chain(args.out, regenerate=args.regenerate,
                                cache_dir=args.cache_dir, only=args.only)
    if rank_session._resolve_proc_id() != 0:
        return 0 if not problems else 1
    for name, wall in walls.items():
        print(f"hsuite {name:16s} {wall:7.1f} s")
    for line in problems:
        print(f"hsuite FAIL {line}")
    print("hsuite PASS" if not problems else "hsuite FAILED")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
