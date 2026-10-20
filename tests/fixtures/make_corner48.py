"""tests/fixtures/bgw_epsmat_corner48.h5: a 48 x 48 corner of BerkeleyGW 4.0's own epsmat.h5.

Source: runs/CrI3/516_tdgw_ref_6x6_nosym_20261016/bgw/epsilon/epsmat.h5
(epsilon.cplx.x, CrI3 6x6 no-sym, 20 Ry, 160 bands, slab, freq_dep 0), q index 1 =
(0, 1/3, 0).  mats/matrix and mats/matrix-diagonal are the stored values of the
first 48 eps G (lowest |q+G|), copied verbatim; the rho list is those 48 G in
eps order (gind_eps2rho = gind_rho2eps = 1..48), so components, ekin and vcoul
are BerkeleyGW's at the same G.
"""
import sys, h5py, numpy as np
src, dst, IQ, N = sys.argv[1], sys.argv[2], 1, 48
with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
    for k in ("versionnumber", "flavor"):
        g[f"eps_header/{k}"] = f[f"eps_header/{k}"][()]
    for k, d in f["eps_header/params"].items():
        g[f"eps_header/params/{k}"] = d[()]
    for k, d in f["eps_header/freqs"].items():
        g[f"eps_header/freqs/{k}"] = d[()]
    q = f["eps_header/qpoints"]
    g["eps_header/qpoints/nq"] = np.int32(1)
    g["eps_header/qpoints/qpts"] = q["qpts"][IQ:IQ + 1]
    g["eps_header/qpoints/qgrid"] = q["qgrid"][()]
    g["eps_header/qpoints/qpt_done"] = q["qpt_done"][IQ:IQ + 1]
    gs = f["eps_header/gspace"]
    e2r = gs["gind_eps2rho"][IQ, :N] - 1
    g["eps_header/gspace/nmtx"] = np.array([N], np.int32)
    g["eps_header/gspace/nmtx_max"] = np.int32(N)
    g["eps_header/gspace/gind_eps2rho"] = np.arange(1, N + 1, dtype=np.int32)[None]
    g["eps_header/gspace/gind_rho2eps"] = np.arange(1, N + 1, dtype=np.int32)[None]
    g["eps_header/gspace/ekin"] = gs["ekin"][IQ][e2r][None]
    g["eps_header/gspace/vcoul"] = gs["vcoul"][IQ:IQ + 1, :N]
    g["mf_header/gspace/components"] = f["mf_header/gspace/components"][()][e2r]
    g["mats/matrix"] = f["mats/matrix"][IQ:IQ + 1, :, :, :N, :N, :]
    g["mats/matrix-diagonal"] = f["mats/matrix-diagonal"][IQ:IQ + 1, :N, :]
    g.attrs["provenance"] = ("48 x 48 low-|q+G| corner of BerkeleyGW 4.0 epsilon.cplx.x epsmat.h5, "
                             "CrI3 6x6 no-sym 20 Ry (LORRAX run 516), q index 1 = (0, 1/3, 0); "
                             "stored matrix values verbatim; rho list = the 48 eps G")
    print("qpt", g["eps_header/qpoints/qpts"][()].round(6).tolist(),
          "matrix", g["mats/matrix"].shape, "e2r max", int(e2r.max()))
