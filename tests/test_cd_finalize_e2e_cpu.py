"""CPU end-to-end pin of the contour-deformation route through the output stage, on the bcc Na hsuite fixture.

The ZnO all-band CD production (cd/land-quadrature-v4) computed every chunk and then crashed in the output stage:
the CD body returns Sigma_c on the on-shell request's distinct frequencies (``omega_grid_ry =
request.frequencies_ry``, ``gw.sigma_dispatch._compute_mpa_sigma``), and ``gw_output.write_results`` read it as an
omega-grid cube on the deck's grid (``eqp_bgw.compute_z_factor_from_omega_grid``: ``sigma_c_omega (321, 8, 48)`` vs
217 points).  This fixture reproduces it as ``(14, 4, 8)`` vs 87.  The deck below runs ``gw.kin_ion_io`` and the
one-shot CD ``gw.gw_jax`` end to end on one CPU device (~45 s) and pins:

1. the run reaches ``write_results`` and writes eqp0.dat, eqp1.dat and sigma_diag.dat (before the fix it refuses
   in ``compute_z_factor_from_omega_grid``); the CD body goes through ``gw.cd.sigma.cd_on_shell_body`` into the
   SPM/MPA route's on-shell finalize (``OnShellSigma``);
2. the ``omega_grid_ry`` request handoff: the CD axis (``cd_sigma_own_energy.npz`` ``omega_ry`` minus mu) holds
   every covered own-row E_DFT, and sigma_diag's Sigma_c on those rows equals the own-energy accumulators (a
   separate kernel at the DFT energies) to print precision, on a reduced k wedge (4 of 27; the unfold runs);
3. the pinned residue reach: ``cd_residue_reach_ev = 16`` passes ``GATE cd_residue_coverage`` (receipt need <= 16)
   and the receipt's reach is 16 eV rounded up to the knot grid;
4. Z: 1 off the sampled range, finite on it, and eqp1 = E_DFT + Z (eqp0 - E_DFT) on every row.

Unit cell: exempt from the four-GPU rule (one CPU device: a process holding four emulated devices is the in-process
mesh the flat-k FFT refuses, tests/hsuite/chain.py).  Run in the LORRAX environment (FFI bundle on the path).
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parent / "hsuite" / "fixture_na"
REACH_EV = 16.0
OMEGA_EV = (-8.0, 12.0)

DECK = f"""[cohsex]
# CD finalize regression: bcc Na hsuite fixture, contour deformation, one-shot (tests/test_cd_finalize_e2e_cpu.py)
centroids_file = centroids_frac_56.txt
nval = 5
ncond = 3
number_bands = 8
sys_dim = 3
bispinor = false
wfn_file = WFN.h5
kin_ion_file = kin_ion.h5
restart = false
write_restart_tensors = false
compute_mode = mpa
screening_diagrams = w_rpa
screening_method = minimax
head_correction = off
qp_solver = one_shot_dft
occ_smearing_width_ry = 0.01
fermi_reference = mp1_fixed_n
use_band_extrapolation = false
linalg = local
sigma_regularization_ev = 1.0
sigma_omega_step_ev = 0.25
sigma_omega_min_ev = {OMEGA_EV[0]:g}
sigma_omega_max_ev = {OMEGA_EV[1]:g}
sigma_out_of_grid = static
sigma_frequency_integration = contour_deformation
cd_imaginary_nodes = 8
cd_knot_step_ev = 0.25
cd_residue_reach_ev = {REACH_EV:g}
cd_chi0_rule_tolerance = 1e-4
cd_own_energy_band_min = 5
cd_checkpoint = write
sigma_diag_file = sigma_diag.dat
eqp0_file = eqp0.dat
eqp1_file = eqp1.dat
report_file = gwjax.out
"""


def _env():
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1", PYTHONUNBUFFERED="1")
    flags = re.sub(r"--xla_force_host_platform_device_count=\d+", "", env.get("XLA_FLAGS", "")).strip()
    env["XLA_FLAGS"] = flags
    return env


def _run(module, cwd, log, timeout):
    with open(cwd / log, "w") as fh:
        run = subprocess.run([sys.executable, "-m", module, "-i", "na_cd.in"], cwd=cwd, env=_env(),
                             stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    return run.returncode, (cwd / log).read_text(errors="replace")


def _eqp(path):
    """``{(k, band): (E_DFT, E_QP)}`` of a BerkeleyGW eqp file (k in file order, band 1-based)."""
    rows, k = {}, -1
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if line.startswith("#") or not parts:
            continue
        if len(parts) == 4 and "." in parts[0]:
            k += 1
        elif len(parts) == 4:
            rows[(k, int(parts[1]))] = (float(parts[2]), float(parts[3]))
    return rows


def _sigma_diag(path):
    """``{(k, band): dict(Eo, sigC, Z)}`` from sigma_diag.dat (k blocks in order, ``n=`` 0-based)."""
    rows, k = {}, -1
    num = r"([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)"
    for line in Path(path).read_text().splitlines():
        if line.startswith("k-point"):
            k += 1
        m = re.match(r"n=(\d+)\s", line)
        if m:
            vals = {key: float(re.search(key + r"=\s*" + num, line).group(1)) for key in ("Eo", "sigC", "Z")}
            rows[(k, int(m.group(1)) + 1)] = vals
    return rows


def test_cd_route_reaches_write_results(tmp_path):
    for name in ("WFN.h5", "centroids_frac_56.txt", "Na.upf"):
        shutil.copy2(FIXTURE / name, tmp_path / name)
        os.chmod(tmp_path / name, 0o644)
    (tmp_path / "na_cd.in").write_text(DECK)
    rc, log = _run("gw.kin_ion_io", tmp_path, "kin_ion.log", 600)
    assert rc == 0, log[-4000:]
    rc, log = _run("gw.gw_jax", tmp_path, "gwjax.log", 1200)
    assert rc == 0, "CD one-shot did not finish the output stage:\n" + log[-6000:]
    for name in ("eqp0.dat", "eqp1.dat", "sigma_diag.dat"):
        assert (tmp_path / name).is_file(), name

    receipt = json.loads((tmp_path / "cd_sigma_receipt.json").read_text())
    assert receipt["need_ev"] <= REACH_EV < receipt["reach_ev"] <= REACH_EV + 2 * receipt["knot_step_ev"], receipt
    assert receipt["full_bz"] is False and receipt["nk_red"] < receipt["nk"], "the fixture must exercise the unfold"

    RYD_EV = float(subprocess.run([sys.executable, "-c", "from common.units import RYD_TO_EV; print(repr(RYD_TO_EV))"],
                                  env=_env(), capture_output=True, text=True, check=True).stdout.split()[-1])
    own = np.load(tmp_path / "cd_sigma_own_energy.npz")
    mu_ry = float(own["mu_ry"])
    axis_ev = (np.asarray(own["omega_ry"]) - mu_ry) * RYD_EV          # the CD body's frequency axis (from mu)
    bands0 = np.asarray(own["bands0"])
    sig = _sigma_diag(tmp_path / "sigma_diag.dat")
    e0, e1 = _eqp(tmp_path / "eqp0.dat"), _eqp(tmp_path / "eqp1.dat")
    n_cov = 0
    for k in range(int(np.asarray(own["kw_rows"]).size)):
        for j, b0 in enumerate(bands0):
            eps_ev = float(own["eps_own_ry"][k, j]) * RYD_EV
            row = sig[(k, int(b0) + 1)]
            assert abs(row["Eo"] - eps_ev) < 2e-6, (k, b0, row["Eo"], eps_ev)
            covered = OMEGA_EV[0] <= eps_ev - mu_ry * RYD_EV <= OMEGA_EV[1]
            if not covered:
                continue
            n_cov += 1
            assert np.min(np.abs(axis_ev - (eps_ev - mu_ry * RYD_EV))) < 1e-8, "E_DFT is not on the CD axis"
            want = float(np.real(own["sigma_c_own_ry"][k, j])) * RYD_EV
            assert abs(row["sigC"] - want) < 2e-6, (k, b0, row["sigC"], want)
    assert n_cov >= 8, n_cov
    for key, row in sig.items():
        assert np.isfinite(row["Z"]), key
        if not OMEGA_EV[0] <= row["Eo"] - mu_ry * RYD_EV <= OMEGA_EV[1]:
            assert row["Z"] == 1.0, (key, row)                       # off the sampled range: Z = 1
        if key in e0:
            (edft, q0), (_, q1) = e0[key], e1[key]
            assert abs(q1 - (edft + row["Z"] * (q0 - edft))) < 5e-6, (key, q0, q1, row["Z"])
