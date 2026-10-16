"""Build a synthetic 2v2c slab system and run TDGW's own driver on it.

Writes <out>/indata/kernel.h5 (BerkeleyGW TDA layout, complex flavor) and
<out>/tdgw_ref/{tdgw.h5, tdgw.log} from TDGW's aGWDriver + solve_ivp
(serial).  The header in tdgw_ref/tdgw.h5 is then the input of the LORRAX
propagator (load_header_from), so both codes integrate identical operators.

usage: python make_synthetic_tdgw.py OUT [--nk 3] [--tf 2000] [--gamma 0]
"""
import argparse, os, json, time
import numpy as np
import h5py


def build(out, nk1=3, tf=2000.0, mdt=5.0, depop=0.0, dephas=0.0, efield=None,
          atol=1e-13, rtol=1e-6, seed=7, kscale=0.03, intraband=True, kernel=True,
          nv=2, nc=2, vspan=(-1.4, -1.1), cspan=(0.6, 0.9), meas_dm=True):
    os.environ["TDGW_RINTRA_IN_EOM"] = "true" if intraband else "false"
    import tensor2
    import tensor2.core.mpio as _mpio
    _mpio.H5PY_HAS_MPI = False      # serial HDF5 (this h5py lists mpio without MPI)
    from tensor2 import TaskGrid as TG
    from tensor2.utils.mpi import COMM_W
    from TDGW.common import basis
    from TDGW.mf.structure import Structure
    from TDGW.mf.wfn import WFNHeader
    from TDGW.mf.dipole.dipwfn import DipoleFromWFN
    from TDGW.mf.dipole.dipcompute import KBMaps, covariant_olp, dipole_from_olp
    from TDGW.td.io import read_kernel
    from TDGW.td.measurement.base import Measurement
    from TDGW.td.driver.agw import aGWDriver
    from TDGW.td.ivp.ivp import solve_ivp
    from TDGW.td.utils import EFunc
    from TDGW.constants import T_IN_ATTOSEC, HA_IN_EV
    from TDGW.utils.misc import setup_logger
    import TDGW.globals as GLB
    assert GLB.RINTRA_IN_EOM == intraband

    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out, "indata"), exist_ok=True)
    refdir = os.path.join(out, "tdgw_ref")
    os.makedirs(refdir, exist_ok=True)
    grid = TG.create((COMM_W.size,), ("k",), COMM_W)
    setup_logger(os.path.join(refdir, "tdgw.log"), "debug", True)

    a = 13.0
    rprim = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 36.0]])
    stru = Structure(rprim, [24, 53], [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.52]])
    kpts_b = basis.KPoints.fullgrid([nk1, nk1, 1], [0, 0, 0])
    nk = kpts_b.size
    nb = nv + nc
    ibndmin = 5
    ivbm = ibndmin + nv - 1
    ibndmax = ibndmin + nb - 1
    bands_b = basis.SequentialBasis(ibndmin, ibndmax + 1, 1)
    kc = 2 * np.pi * kpts_b.kpts[:, :2]
    disp = np.cos(kc[:, 0]) + np.cos(kc[:, 1]) + np.cos(kc[:, 0] + kc[:, 1])
    ev = np.linspace(vspan[0], vspan[1], nv); ec = np.linspace(cspan[0], cspan[1], nc)
    amp = 0.05 + 0.05 * rng.random(nb)
    e_kn = (np.concatenate([ev, ec])[None, :]
            + np.where(np.arange(nb) < nv, 1.0, -1.0)[None, :] * amp[None, :] * disp[:, None]) / HA_IN_EV
    f_kn = np.tile(np.r_[np.ones(nv), np.zeros(nc)], (nk, 1))
    wfnhd = WFNHeader(kpts_b, bands_b, np.array([30, 30, 90]), 2,
                      tensor2.astensor(e_kn), tensor2.astensor(f_kn),
                      tensor2.astensor(np.full(nk, 1.0 / nk)))
    kb = KBMaps.from_scratch(stru, "slab", kpts_b, False, kpts_b.basis_full)
    nbvec = kb.nbvec
    olp = np.empty((nbvec, nk, nb, nb), complex)
    for ib in range(nbvec):
        for ik in range(nk):
            g = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
            g = 0.5 * (g + g.conj().T)
            w, v = np.linalg.eigh(g)
            olp[ib, ik] = 0.97 * (v * np.exp(1j * 0.15 * w)) @ v.conj().T
    olp_t = tensor2.astensor(olp)
    efull = tensor2.astensor(e_kn)
    st, oc = covariant_olp(olp_t, efull, kb, 0.05 / HA_IN_EV)
    rk = dipole_from_olp(oc, kb.wb_ib).get_data("np")
    gz = rng.normal(size=(nk, nb, nb)) + 1j * rng.normal(size=(nk, nb, nb))
    rz = 0.3 * (gz + np.conj(np.swapaxes(gz, -1, -2)))
    rinter = rk.copy(); rinter[2] = rz
    rintra = np.zeros((3, nk, nb)); rintra[2] = np.real(np.einsum("knn->kn", rz))
    for i in range(nb):
        rinter[:, :, i, i] = 0.0
    dip = DipoleFromWFN(wfnhd, "slab", kb, olp_t, st, oc, tensor2.astensor(rinter),
                        tensor2.astensor(rintra), efull)

    kfile = os.path.join(out, "indata", "kernel.h5")
    kern = None
    if kernel:
        n = nk * nc * nv
        m = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        K = kscale / HA_IN_EV * (m + m.conj().T) / np.sqrt(n)
        K -= 0.15 / HA_IN_EV * np.eye(n)
        K6 = K.reshape(nk, nc, nv, nk, nc, nv)
        f6 = np.zeros((nk, nk, nv + nc, nv + nc, nv + nc, nv + nc), complex)
        kt = np.transpose(K6, (3, 0, 1, 2, 4, 5))[:, :, :, ::-1, :, ::-1]
        f6[:, :, nv:, :nv, nv:, :nv] = -2.0 * kt
        with h5py.File(kfile, "w") as f:
            f["kernel"] = np.stack([f6.real, f6.imag], axis=-1)
            h = f.create_group("bse_header")
            h["flavor"] = 2
            h["params/nblocks"] = 1
            h["kpoints/kpts"] = kpts_b.kpts
            h["bands/n1b"] = nv + nc; h["bands/n2b"] = nv + nc
            h["bands/nvb"] = nv; h["bands/ncb"] = nc
            g = f.create_group("header")
            g["screening"] = 0; g["truncation"] = 6; g["kernel_type"] = 3
        kern = read_kernel(kfile, ivbm=ivbm, bands_b=bands_b, grid=grid)

    efield = efield or {"E_x": "1e3 * np.exp(-((t - 100) / 40)**2)",
                        "E_y": "0.5e3 * np.exp(-((t - 100) / 40)**2)", "E_z": 0}
    efunc = EFunc.from_dict(efield)
    meas_cfg = [("Efield", {}), ("polarization", {}),
                ("current", {} if kernel else {"use_v0": True})]
    if meas_dm:
        meas_cfg.append(("density_matrix", {"dataset_name": "rho_cv",
                         "brange1": [ivbm + 1, ivbm + 1 + nc],
                         "brange2": [ibndmin, ivbm + 1]}))
    meas = [Measurement.from_dict(nm, dict(kw)) for nm, kw in meas_cfg]
    t0 = 0.0
    driver = aGWDriver(t0 / T_IN_ATTOSEC, mdt / T_IN_ATTOSEC, tf / T_IN_ATTOSEC, 5000,
                       stru, wfnhd, dip, kern, efunc, depop / HA_IN_EV,
                       dephas / HA_IN_EV, meas, os.path.join(refdir, "tdgw.h5"), grid)
    t_start = time.perf_counter()
    res = solve_ivp(driver, rtol=rtol, atol=atol, max_step=np.inf)
    wall = time.perf_counter() - t_start
    deck = {
        "control": {"loglevel": "info", "calc_type": "COHSEX" if kernel else "IP",
                    "timesteps": {"t0": t0, "tf": tf, "dt_measure": mdt, "ntcheck": 5000},
                    "ivp": {"rtol": rtol, "atol": atol},
                    "files": {"indir": os.path.join(out, "indata"), "outdir": refdir}},
        "system": {"ibndmin": ibndmin, "ibndmax": ibndmax, "ivbm": ivbm,
                   "geometry": "slab", "center_of_vacuum": [0, 0, 0],
                   "esmear": 0.05, "depopulation": depop, "dephasing": dephas,
                   "Efunc": efield},
        "measurements": {"types": [m for m, _ in meas_cfg],
                         **{str(i): kw for i, (_, kw) in enumerate(meas_cfg) if kw}},
    }
    with open(os.path.join(out, "deck_tdgw.json"), "w") as f:
        json.dump(dict(deck=deck, nfev=int(res.nfev), wall=wall,
                       intraband=intraband), f, indent=1)
    return deck, res, wall


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--nk", type=int, default=3)
    ap.add_argument("--tf", type=float, default=2000.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--no-intraband", action="store_true")
    ap.add_argument("--ip", action="store_true")
    ap.add_argument("--efield", type=float, default=1e3)
    a = ap.parse_args()
    ef = {"E_x": f"{a.efield:g} * np.exp(-((t - 100) / 40)**2)",
          "E_y": f"{0.5 * a.efield:g} * np.exp(-((t - 100) / 40)**2)", "E_z": 0}
    deck, res, wall = build(a.out, nk1=a.nk, tf=a.tf, depop=a.gamma, dephas=a.gamma,
                            intraband=not a.no_intraband, kernel=not a.ip, efield=ef)
    print("TDGW nfev", res.nfev, "wall", round(wall, 2))
