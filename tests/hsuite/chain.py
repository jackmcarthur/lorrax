"""The driver chain on the magnetic H2⁻ spinor fixture, at P4 on one node.

Every stage is a production driver's entry point, called in sequence in one
Python process on each of the four srun ranks (one GPU each, MPI world 4;
one runtime, one FFI load, one compile cache), in one shared run directory,
in the order a user runs them:

    kmeans -> kin_ion -> dipole -> gwjax GN-PPM one-shot
           -> BSE -> htransform
           -> exciton bands -> restarted: COHSEX, shared-pole one-shot with
              W and pole exports
           -> bispinor kin_ion, dipole -> four-current shared-pole QSGW
              (2 maps, fresh zeta; its map 0 is the one-shot) -> BSE
           -> the same QSGW with the sector constructor forced to the face
           -> bcc Na (run/na): kin_ion, dipole -> metal shared-pole QSGW
              (2 maps, the production defaults)

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
import contextlib
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
# The metal + semicore fixture, staged in its own subdirectory of the run.
FIXTURE_NA = HERE / "fixture_na"
NA_DIR = "na"
REFERENCE = HERE / "reference"

# Tolerances.  Static objects (centroid-free preprocessing, ISDF fits) are
# deterministic to round-off; Sigma carries the quadrature rule set, which is
# rebuilt every run (no rule cache) and may split boxes differently by P, so
# the Sigma tolerance is the 0.5 meV rule-set budget of the old core GN cell.
ATOL = {
    "h5": 1.0e-8,        # kin_ion / dipole members (relative + absolute)
    "w_bank": 1.0e-6,    # exported shared-pole W samples (relative)
    "eqp_ev": 5.0e-4,    # eqp0/eqp1 columns, GN-PPM and shared pole
    "bse_ev": 1.0e-4,    # BSE, exciton-band and htransform eigenvalues:
                         # P1 vs P4 agree to 1 ueV (the fH basis pivots
                         # break residual ties by index, galerkin.py)
    "sc_residual_ev": 2.0e-3,  # per-map SC max|dE| read from the report
    # Sigma members of the SC stages' sigma h5 (SC_SIGMA_H5): an SC map
    # amplifies the P-dependent rule-set split, and the owner's target is
    # <= 1 meV rule-set reproducibility.  Measured P1 vs P4 on na_sc map 1:
    # 0.66 meV (main 8e33b82d3 and GRAMRP alike).  eqp files keep eqp_ev.
    "sc_sigma_ev": 1.0e-3,
}
# The SC stages' sigma h5 files, compared at ATOL["sc_sigma_ev"], and the
# GN-PPM one: its output window holds the deep occupied bands 1-2 of the
# H2- fixture, where P1 vs P4 reads 0.98 meV.
SC_SIGMA_H5 = ("bsc_sigma.h5:", "bfc_sigma.h5:", "na_sigma.h5:", "gnppm_sigma.h5:")

# h5 members not compared.  line_charge_* are the W bank's selected
# direction states: a gauge per direction, and padded to the mesh (P1 6x1,
# P4 8x2); Wc, dWc_ds and the moments are compared.
H5_UNCOMPARED = ("_w.h5:line_charge_",)

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
    "[EXC] ",            # an exception passed a stage timer (caught or not)
)
# A refusal carried by an exception (``ValueError: GATE name: ...``), caught
# or not.  A bare ``GATE name`` line is not a failure: several stages log
# declined options by gate name (the dipole's parallel-transport artifact,
# the na_sc head's zeroed Sigma term).
REFUSAL = re.compile(r"\w+(?:Error|Exception): GATE \w+")
# A GATE name logged on rank r > 0 that rank 0's log of the same stage does
# not show is rank-divergent, and a failure.
GATE_NAME = re.compile(r"GATE (\w+)")
# How long a failed rank waits for the others at the stage-end join.  Ranks
# that refuse together arrive within seconds; a rank that fails alone leaves
# the others in a collective, and then reports its own record and exits.
FAIL_JOIN_S = 60
# Test-only hook, never read by src: HSUITE_INJECT_REFUSAL="stage:rank" makes
# that rank raise a GATE refusal after the stage's driver returns (the others
# join); "stage:rank:before" raises before the driver runs (the others are
# left in a collective).  Gate for the harness itself.
INJECT = os.environ.get("HSUITE_INJECT_REFUSAL", "")
# Set when a failed rank could not join the others: the process must then
# end with os._exit after pytest's report (conftest.pytest_unconfigure), since
# a normal exit waits on peers blocked in a collective.
LONE_FAILURE = False

# nval = 3: every occupied band of the H2⁻ fixture, which htransform requires
# (it refuses a window that omits one) and every restarted stage must share.
_DECK_COMMON = """[cohsex]
centroids_file = {centroids}
nval = 3
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
# four-current Gamma head) through the SC driver for two maps, with its own
# kinetic-balance kin_ion, the per-map head (dft_velocity), the live
# four-current density (density_self_consistent, required) and the default
# band extrapolation (the CC class's band sum).  Map 0 is the
# one-shot, so there is no separate one-shot stage (suite wall).  The
# transverse zeta is fitted on the charge centroid set: one kmeans stage, and
# any point set is a legal ISDF basis for the current rows.
_BISP_COMMON = """bispinor = true
centroids_file_current = {centroids}
bispinor_gw = full_shared_pole
compute_mode = mpa
sigma_w_model = shared_pole
head_correction = no_local_fields
mpa_n_poles = 2
sigma_regularization_ev = 0.25
linalg = local
"""


def _bisp_deck(prefix, body):
    # nval = 1: the fresh bispinor fit shares no restart window with the
    # scalar stages, and three SC bands keep the stage wall.
    return (_DECK_COMMON.replace("bispinor = false\n", "")
            .replace("nval = 3\n", "nval = 1\n")
            .replace("kin_ion_file = kin_ion.h5", "kin_ion_file = kin_ion_bisp.h5")
            + _BISP_COMMON + body + f"""\
sigma_diag_file = {prefix}_sigma.dat
eqp0_file = {prefix}_eqp0.dat
eqp1_file = {prefix}_eqp1.dat
report_file = {prefix}.out
sigma_omega_h5_file = {prefix}_sigma.h5
""")


_BISP_SC = """restart = false
qp_solver = self_consistent
sc_max_iter = 2
sc_tol_ev = 3.0
sc_head_update = dft_velocity
density_self_consistent = true
"""
RESTART_DECKS.update({
    "bisp_sc.in": _bisp_deck("bsc", _BISP_SC),
    "bisp_face.in": _bisp_deck("bfc", _BISP_SC),
})
_P = ["--px", _SIDE, "--py", _SIDE]

# The metal + semicore route on bcc Na (fixture_na/): the production
# defaults the H2⁻ fixture cannot reach.  Fermi-Dirac occupations, the
# partition by number_bands_protected (2s and 2p lie below a 20 eV gap, so
# they are the coarse class, read on held windows at eta_semi and pinned at
# their DFT block by the default sc_semicore = dft), the rigid tail above the
# QP window with its min(Z, 1/Z) law, the unnamed head update
# (parallel_transport; the 3^3 links fail the window gate, so its Sigma term
# is zero on every map) and its per-map head block (gap 0 on a metal),
# spectral_shell extrapolation, and the held SC windows, all at their defaults.  Scalar WFN, fresh zeta.
NA_DECK = """[cohsex]
centroids_file = {centroids}
number_bands_protected = 8
number_bands = 13
sys_dim = 3
bispinor = false
wfn_file = WFN.h5
kin_ion_file = kin_ion.h5
restart = false
compute_mode = mpa
sigma_w_model = shared_pole
occ_smearing_width_ry = 0.01
fermi_reference = mp1_fixed_n
qp_solver = self_consistent
sc_max_iter = 2
sc_tol_ev = {sc_tol}
linalg = {linalg}
write_qsgw_datasets = true
sigma_freq_debug_output = false
sigma_diag_file = na_sigma.dat
eqp0_file = na_eqp0.dat
eqp1_file = na_eqp1.dat
report_file = na.out
sigma_omega_h5_file = na_sigma.h5
"""
# Map 1 moves 1.04 eV: the 1.5 eV criterion stops after two maps, converged.
NA_SC_TOL_EV = "1.5"

# Scratch the lead removes before a stage.  The four-current QSGW is a
# fresh model in a run directory that already holds the scalar shared-pole
# models (checked at sp_export), and a fresh run refuses to overwrite a
# completed model (GATE shared_pole_output).
_CLEAR_BEFORE = {"bisp_sc": ("tmp/mpa",), "bisp_face": ("tmp/mpa",)}

# bisp_face reruns bisp_sc with the sector constructor (CC/TT/CT) forced to
# the whole-mesh (face) route, which decks too large for whole parents per
# rank (CrI3) take for every sector.  Its programs differ from the local ones
# only in layout and eigensolver adapter, so it is judged against bisp_sc's
# reference (labels bsc -> bfc).  Test-only, never read by src.
FORCED_FACE = ("bisp_face",)
REFERENCE_OF = {"bisp_face": ("bisp_sc", "bsc", "bfc")}

# (name, module, argv, deck name -> template).  The decks are written into
# the run directory after kmeans, which names the centroid file.
STAGES = (
    ("kmeans", "centroid.kmeans_cli",
     ["6", "--seed", "42", "--force-shard", "--orbit", "--oversample", "1.5",
      "--fit-window", "0:3,0:7", "--density-mode", "scalar"]),
    ("kin_ion", "gw.kin_ion_io", ["-i", "gnppm.in"]),
    ("dipole", "psp.get_dipole_mtxels", ["-i", "gnppm.in"]),
    ("gnppm", "gw.gw_jax", ["-i", "gnppm.in"]),
    # A regression check, not physics: the DFT gap is 72 meV, far below the
    # ~1 eV exciton binding of this weakly screened molecule, so the TDA
    # eigenvalues come out negative (-0.94, -0.78 eV).
    ("bse", "bse.bse_jax",
     ["-i", "gnppm.in", "--bse", "--lanczos", "--tda", "--solver", "davidson",
      "--n-val", "1", "--n-cond", "2", "--n-occ", "3",
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
    ("sp_export", "gw.gw_jax", ["-i", "sp_export.in"]),
    ("kin_ion_bisp", "gw.kin_ion_io", ["-i", "bisp_sc.in", "-o", "kin_ion_bisp.h5"]),
    # The direct head authenticates the dipole's representation stamp, so
    # the four-component route writes its own (over the scalar one).
    ("dipole_bisp", "psp.get_dipole_mtxels", ["-i", "bisp_sc.in"]),
    ("bisp_sc", "gw.gw_jax", ["-i", "bisp_sc.in"]),
    # BSE after SP-full: the final map's W0 = V + Wc_CC(0) (charge sector
    # only; CT/TC/TT are not stored), on the four-component restart.
    ("bse_bisp", "bse.bse_jax",
     ["-i", "bisp_sc.in", "--bse", "--lanczos", "--tda", "--solver", "davidson",
      "--n-val", "1", "--n-cond", "2", "--n-occ", "3",
      "--band-degeneracy", "off", "--max-lanczos-iter", "40",
      "--n-eig", "2", "--block-size", "1", *_P,
      "--report-file", "bse_bisp.out"]),
    # After bse_bisp, so that stage reads bisp_sc's W0 restart.
    ("bisp_face", "gw.gw_jax", ["-i", "bisp_face.in"]),
    # bcc Na in run/na/.  Its centroids are stored in fixture_na/ (kmeans
    # is covered above): 56 orbit-closed points from `centroid.kmeans_cli 64
    # --seed 42 --orbit --oversample 1.5 --fit-window 0:8,0:13` (the zeta legs).
    ("na_kin_ion", "gw.kin_ion_io", ["-i", "na.in"]),
    ("na_dipole", "psp.get_dipole_mtxels", ["-i", "na.in"]),
    ("na_sc", "gw.gw_jax", ["-i", "na.in"]),
)

# Stages that run in a subdirectory of the run (their own fixture).
_STAGE_DIR = {name: NA_DIR for name in ("na_kin_ion", "na_dipole", "na_sc")}

# What each stage leaves behind and how it is compared.
#   eqp: eqp files (column compare); h5: numeric members; rows: whitespace
#   tables (numeric columns after `skip`); text: regex list captured from a
#   report and compared as floats.
CHECKS = {
    "kin_ion": {"h5": ["kin_ion.h5"]},
    "dipole": {"h5": ["dipole.h5"]},
    "gnppm": {"eqp": ["gnppm_eqp0.dat", "gnppm_eqp1.dat"],
              "h5": ["gnppm_sigma.h5"]},
    "bse": {"stdout_floats": r"^\s*S\d+\s+([0-9.+-]+)\s*$"},
    "exciton_bands": {"rows": [("exciton.dat", 6)]},
    "htransform": {"rows": [("htransform.dat", 6)]},
    "cohsex": {"eqp": ["cohsex_eqp0.dat", "cohsex_eqp1.dat"]},
    # The exported W bank is compared by value; the pole model (b, Lambda)
    # has a gauge per pole, so only its members' shapes are pinned.
    "sp_export": {"eqp": ["spx_eqp0.dat", "spx_eqp1.dat"],
                  "h5": ["spx_sigma.h5", "tmp/mpa/oneshot_w.h5"],
                  "shapes": ["tmp/mpa/oneshot_poles.h5"]},
    "kin_ion_bisp": {"h5": ["kin_ion_bisp.h5"]},
    "dipole_bisp": {"h5": ["dipole.h5"]},
    "bisp_sc": {"eqp": ["bsc_eqp0.dat", "bsc_eqp1.dat"],
                "h5": ["bsc_sigma.h5"],
                "report_floats": ("bsc.out",
                                  r"SC iteration: call=\d+ .*?max\|dE\|=([0-9.e+-]+)")},
    "bse_bisp": {"stdout_floats": r"^\s*S\d+\s+([0-9.+-]+)\s*$"},
    "bisp_face": {"eqp": ["bfc_eqp0.dat", "bfc_eqp1.dat"],
                  "h5": ["bfc_sigma.h5"],
                  "report_floats": ("bfc.out",
                                    r"SC iteration: call=\d+ .*?max\|dE\|=([0-9.e+-]+)")},
    "na_kin_ion": {"h5": ["kin_ion.h5"]},
    "na_dipole": {"h5": ["dipole.h5"]},
    "na_sc": {"eqp": ["na_eqp0.dat", "na_eqp1.dat"],
              "h5": ["na_sigma.h5"],
              "report_floats": ("na.out",
                                r"SC iteration: call=\d+ .*?max\|dE\|=([0-9.e+-]+)")},
}

# Lines a stage's rank-0 log must show: the production defaults the stage is
# there to cover, by name, so a default that silently changes fails here.
REQUIRED_LINES = {
    "na_sc": (
        ("partition by number_bands_protected",
         r"QP matrix: bands 1-8 counted \(nval=5, ncond=3\).*coarse \(semicore\) "
         r"Sigma read: 16 \(k,state\).*\(number_bands_protected\)"),
        ("2s and 2p coarse windows at eta_semi 5 eV",
         r"SC coarse windows \(plan, map 0\): \[[^\]]+\]@5, \[[^\]]+\]@5 eV"),
        ("sc_semicore = dft pin",
         r"SC semicore = dft: 16 coarse \(k,label\) hold their DFT block"),
        ("tail law with min(Z, 1/Z) weights at map 1",
         r"SC sum-band tail: scissored \[8, 13\) .*Z-weighted\)"),
        ("unnamed sc_head_update: parallel_transport, Sigma term zeroed",
         r"SC head: map 1: parallel_transport Sigma term D_k dH set to 0 "
         r"\(GATE pt_head_window_hybridized"),
        ("Fermi-Dirac fixed-N metal head",
         r"SC metal head: fixed-N fd occupations"),
        ("per-map head block at map 1, metal gap",
         r"SC head velocity, map 1:(?:.*\n){1,8}?\s+band gap: 0\.0000 eV \(metal\)"),
        ("held Sigma windows re-planned on escape at map 1",
         r"SC map event at call 1: sampled grid or Sigma rule set changed"),
    ),
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
    (run / NA_DIR).mkdir()
    for src, dst in ((FIXTURE, run), (FIXTURE_NA, run / NA_DIR)):
        for path in src.iterdir():
            shutil.copy2(path, dst / path.name)
            os.chmod(dst / path.name, 0o644)


def _centroid_file(run):
    found = sorted(p.name for p in run.glob("centroids_frac_*.txt"))
    assert len(found) == 1, f"kmeans left {found} in {run}"
    return found[0]


def write_na_deck(run_na):
    (run_na / "na.in").write_text(NA_DECK.format(
        centroids=_centroid_file(run_na), sc_tol=NA_SC_TOL_EV,
        linalg="distributed" if _SIDE == "2" else "local"))


def write_decks(run):
    centroids = _centroid_file(run)
    (run / "gnppm.in").write_text(GNPPM_DECK.format(centroids=centroids))
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


def _injected(name, rank, when):
    stage, _, rest = INJECT.partition(":")
    target, _, moment = rest.partition(":")
    if stage == name and target == str(rank) and (moment or "after") == when:
        raise ValueError(f"GATE hsuite_injected_refusal: got: HSUITE_INJECT_REFUSAL="
                         f"{INJECT!r} on rank {rank}; want: unset; why: harness test")


@contextlib.contextmanager
def _sector_route(name):
    """Force the sector constructor's route to the face for a FORCED_FACE stage."""
    if name not in FORCED_FACE:
        yield
        return
    import gw.shared_pole_sectors as sectors
    resolve = sectors.sector_execution

    def face(*args, **kwargs):
        _, rows = resolve(*args, **kwargs)
        for row in rows:
            row["mode"] = "face"
        return "face", rows
    sectors.sector_execution = face
    try:
        yield
    finally:
        sectors.sector_execution = resolve


def run_stage(run, name, module, argv, env, timeout):
    """Run one driver's entry point in this process on every rank.

    One runtime per rank for the whole chain: one jax.distributed world,
    one FFI load, one compile cache.  The driver's stdout/stderr (Python
    and native) go to ``<name>.rank<r>.log``.  Every rank then scans its own
    log and joins the others with a record (rc, failure signatures, GATE
    names, and the log tail on a failure); the return value is every rank's
    record, the same list on every rank.  A failed rank waits FAIL_JOIN_S at
    most: if the others are left in a collective, it returns its own record
    alone and sets LONE_FAILURE.
    """
    global LONE_FAILURE
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
            _injected(name, rank, "before")
            with _sector_route(name):
                ret = main(argv) if inspect.signature(main).parameters else main()
            rc = int(ret or 0)
            _injected(name, rank, "after")
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
    record = rank_record(log, name, rank, rc)
    failed = bool(rc or record["hits"])
    if failed:
        # pytest's capture holds this until the report, which prints it.
        print(f"hsuite FAIL {name}: rc={rc} on rank {rank}; see {log}", flush=True)
    _release_devices()
    try:
        records = rank_session.exchange(
            record, **({"timeout": FAIL_JOIN_S} if failed else {}))
    except (TimeoutError, OSError) as exc:
        if not failed:
            raise
        LONE_FAILURE = True
        record["hits"].append(f"the other ranks did not join within {FAIL_JOIN_S} s "
                              f"(left in a collective?): {exc!r}")
        records = [record]
    return records, time.monotonic() - t0


def rank_record(log, name, rank, rc):
    """One rank's verdict on its own stage log: signatures, GATE names, tail."""
    text = log.read_text(errors="replace")
    hits = [sig for sig in FAILURE_SIGNATURES if sig in text]
    hits += sorted(set(REFUSAL.findall(text)))
    tail = ""
    if rc or hits:
        tail = "\n".join(text.splitlines()[-60:])
    return {"stage": name, "rank": rank, "rc": rc, "hits": hits,
            "gates": sorted(set(GATE_NAME.findall(text))), "tail": tail}


def stage_verdict(name, records):
    """Problems from every rank's record of one stage (same on every rank)."""
    problems = []
    count = rank_session._resolve_proc_count()
    joined = {rec["rank"] for rec in records}
    if len(joined) < count:
        problems.append(f"{name}: ranks {sorted(set(range(count)) - joined)} "
                        "did not report")
    lead_gates = set(next((rec["gates"] for rec in records if rec["rank"] == 0), ()))
    for rec in records:
        divergent = [g for g in rec["gates"] if g not in lead_gates]
        if rec["rank"] and divergent and 0 in joined:
            rec["hits"].append(f"GATE {divergent} absent from rank 0")
        if rec["rc"] or rec["hits"]:
            problems.append(f"{name}: rank {rec['rank']} failed: rc={rec['rc']} "
                            f"{rec['hits'][:4]}\n--- {name}.rank{rec['rank']}.log "
                            f"(tail) ---\n{rec['tail']}")
    return problems


def missing_lines(run, name):
    """The REQUIRED_LINES a stage's rank-0 log does not show."""
    text = (run / f"{name}.rank0.log").read_text(errors="replace")
    return [f"{name}: log lacks {label!r} ({pattern})"
            for label, pattern in REQUIRED_LINES.get(name, ())
            if not re.search(pattern, text, re.MULTILINE)]


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
        return (ATOL["sc_sigma_ev"] if label.startswith(SC_SIGMA_H5)
                else ATOL["eqp_ev"]), False
    if ".h5:" in label:
        return ATOL["h5"], True
    if label.endswith(":residuals"):
        return ATOL["sc_residual_ev"], False
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
    walls, problems, ranks = {}, [], {}
    t_all = time.monotonic()
    for name, module, argv in STAGES:
        if only and name not in only:
            continue
        where = run / _STAGE_DIR.get(name, "")
        if name == "kin_ion" and lead:
            write_decks(run)
        if name == "na_kin_ion" and lead:
            write_na_deck(where)
        if lead:
            for rel in _CLEAR_BEFORE.get(name, ()):
                shutil.rmtree(run / rel, ignore_errors=True)
        rank_session.exchange(name)
        records, walls[name] = run_stage(where, name, module, argv, env, timeout)
        ranks[name] = [{k: rec[k] for k in ("rank", "rc", "hits", "gates")}
                       for rec in records]
        # One verdict for every rank (every rank holds the same records): a
        # rank that walks on alone hangs.
        failed = stage_verdict(name, records)
        if failed:
            problems += failed
            break
        if not lead:
            continue
        rule = CHECKS.get(name, {})
        problems += missing_lines(where, name)
        if name == "kmeans":
            found = list(where.glob("centroids_frac_*.txt"))
            pts = np.loadtxt(found[0]) if len(found) == 1 else np.zeros((0, 3))
            if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
                problems.append("kmeans: centroid file malformed")
            got = {"centroids": pts}
        else:
            got = captured(where, name, rule)
        source, old, new = REFERENCE_OF.get(name, (name, "", ""))
        if source == name and (regenerate == "all" or (
                regenerate == "missing" and _load(name) is None)):
            REFERENCE.mkdir(exist_ok=True)
            _save(name, got)
            continue
        ref = _load(source)
        if ref is not None and old:
            ref = {new + k[len(old):] if k.startswith(old) else k: v for k, v in ref.items()}
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
    summary = {"walls_s": walls, "problems": problems, "regenerate": regenerate,
               "ranks": rank_session._resolve_proc_count(),
               "rank_records": ranks, "cache_dir": str(cache_dir)}
    if LONE_FAILURE:
        # The others are blocked in a collective: no final join.
        rank = rank_session._resolve_proc_id()
        (out / f"summary.rank{rank}.json").write_text(json.dumps(summary, indent=1))
        return walls, problems
    if lead:
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
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
    if rank_session._resolve_proc_id() != 0 and not LONE_FAILURE:
        return 0 if not problems else 1
    for name, wall in walls.items():
        print(f"hsuite {name:16s} {wall:7.1f} s")
    for line in problems:
        print(f"hsuite FAIL {line}")
    print("hsuite PASS" if not problems else "hsuite FAILED")
    return 0 if not problems else 1


def _cli():
    rc = main()
    if LONE_FAILURE:
        # Peers are blocked in a collective: a normal exit would wait on them.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc or 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(_cli())
