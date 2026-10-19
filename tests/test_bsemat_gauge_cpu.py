"""CPU equivalence of the bsemat gauge map's device route with the per-k host fit.

``bse.bsemat_export.gauge_map`` evaluates the coarse WFN's states at the
centroids with the GW face transform (``WfnLoader.load(k="ibz")``,
``gflat_to_rmu``), fits LORRAX's ψ on the mesh and takes the block polar
factors from ``distrib_la``. Here a real fixture WFN (H: nspinor 2, 9 k; Na:
nspinor 1, 4 k) supplies ψ_file, and a synthetic ψ_LX = s·ψ_file U_true (block
unitary per k, random k order) carries an out-of-window admixture, a cross-block
leak and off-span noise. The device U, the mapped ψ and all four receipts must
equal the base branch's host fit (reproduced verbatim below) to rounding.
Red twins: a re-phased file moves U by exactly its phases while every receipt
stays put (only U can see a gauge error); a permuted k map with consistent
energies blows the residual receipt; a shifted energy fails the band gate; a
grid too small for the G sphere fails the alias gate.
Unit cells: exempt from the four-GPU rule (CPU, an emulated mesh of at most
four host devices).
"""
import os
import shutil

import h5py
import numpy as np
import pytest
import jax
from jax.sharding import Mesh

from bse.bsemat_export import gauge_map

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = {"H_ns2": (os.path.join(HERE, "hsuite", "fixture", "WFN.h5"), 2, 2, 40),
            "Na_ns1": (os.path.join(HERE, "hsuite", "fixture_na", "WFN.h5"), 3, 3, 50)}
RECEIPTS = ("lsq_residual_max", "sv_spread_max", "cross_block_max", "energy_mismatch_max_ry")


def _mesh():
    n = 2 if len(jax.devices()) >= 4 else 1
    return Mesh(np.asarray(jax.devices()[:n * n]).reshape(n, n), ("x", "y"))


def _read_k(wfn_file, k, b_lo, b_hi):
    """The base branch's ``file_io.bgw_bsemat.read_wfn_coefficients``."""
    with h5py.File(wfn_file, "r") as w:
        ngk = np.asarray(w["mf_header/kpoints/ngk"][()], dtype=np.int64)
        off = np.concatenate([[0], np.cumsum(ngk)])
        g = np.asarray(w["wfns/gvecs"][off[k]:off[k + 1]], dtype=np.int64)
        c = w["wfns/coeffs"][b_lo:b_hi, :, off[k]:off[k + 1], :]
        el = np.asarray(w["mf_header/kpoints/el"][()], dtype=np.float64)
        el = el.reshape(-1, el.shape[-2], el.shape[-1])[0, k, b_lo:b_hi]
    return g, c[..., 0] + 1j * c[..., 1], el


def _host_gauge_map(psi, *, wfn_co, k_rows, bands_abs, centroid_frac, fft_grid, nv, enk_win,
                    guard=8):
    """The base branch's (d31876a78) host gauge map, verbatim."""
    grid = np.asarray(fft_grid, dtype=np.int64)
    rmu = np.mod(np.rint(np.asarray(centroid_frac) * grid[None, :]).astype(np.int64),
                 grid[None, :]) / grid[None, :]
    with h5py.File(wfn_co, "r") as w:
        nb_file = int(w["mf_header/kpoints/mnband"][()])
        rk = np.asarray(w["mf_header/kpoints/rk"][()], dtype=np.float64)
    if rk.shape[0] == 3 and rk.shape[1] != 3:
        rk = rk.T
    b_lo, b_hi = max(0, int(bands_abs[0]) - guard), min(nb_file, int(bands_abs[-1]) + 1 + guard)
    nb, n_mu = psi.shape[1], len(rmu)
    sl_v, sl_c = slice(0, nv), slice(nv, nb)
    w0 = int(bands_abs[0]) - b_lo
    U = np.zeros((len(rk), nb, nb), np.complex128)
    fig = {"lsq_residual_max": 0.0, "sv_spread_max": 0.0, "cross_block_max": 0.0,
           "energy_mismatch_max_ry": 0.0}
    for j in range(len(rk)):
        g, c, el = _read_k(wfn_co, j, b_lo, b_hi)
        de = float(np.max(np.abs(el[int(bands_abs[0]) - b_lo:int(bands_abs[-1]) + 1 - b_lo]
                                  - enk_win[k_rows[j]])))
        fig["energy_mismatch_max_ry"] = max(fig["energy_mismatch_max_ry"], de)
        if de > 5.0e-2:
            raise ValueError(f"GATE bsemat_band_offset: WFN_co row {j} window energies differ "
                             f"from the restart's by {de:.3e} Ry; want the same bands.")
        ph = np.exp(2j * np.pi * ((g + rk[j][None, :]) @ rmu.T))
        Af = np.einsum("bsg,gm->bsm", c, ph).reshape(b_hi - b_lo, -1)
        Bm = psi[k_rows[j], :, :, :n_mu].reshape(nb, -1)
        X = np.linalg.solve(Af.conj() @ Af.T, Af.conj() @ Bm.T)
        res = np.linalg.norm(Bm - X.T @ Af) / np.linalg.norm(Bm)
        Xw = X[w0:w0 + nb]
        for sl in (sl_v, sl_c):
            Wl, s, Vh = np.linalg.svd(Xw[sl, sl])
            U[j][sl, sl] = Wl @ Vh
            fig["sv_spread_max"] = max(fig["sv_spread_max"], float((s.max() - s.min()) / s.max()))
        cross = (np.linalg.norm(Xw[sl_v, sl_c]) + np.linalg.norm(Xw[sl_c, sl_v])) / np.linalg.norm(Xw)
        fig["lsq_residual_max"] = max(fig["lsq_residual_max"], float(res))
        fig["cross_block_max"] = max(fig["cross_block_max"], float(cross))
    return U, fig


def _unitary(rng, n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _system(name, seed=7, leak=1e-3, admix=1e-2, noise=1e-4, scale=0.37, guard=8):
    """ψ_LX planted on the fixture's states: the gauge_map keywords, U_true and the file."""
    path, nv, nc, n_mu = FIXTURES[name]
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as w:
        m = w["mf_header"]
        grid = np.asarray(m["gspace/FFTgrid"][()], dtype=np.int64)
        nb_file, nk = int(m["kpoints/mnband"][()]), int(m["kpoints/nrk"][()])
        el = np.asarray(m["kpoints/el"][()], dtype=np.float64).reshape(-1, nk, nb_file)[0]
    nb = nv + nc
    b0 = (nb_file - nb) // 2
    bands_abs = np.arange(b0, b0 + nb)
    b_lo, b_hi = max(0, b0 - guard), min(nb_file, b0 + nb + guard)
    w0 = b0 - b_lo
    cells = rng.choice(int(np.prod(grid)), size=n_mu, replace=False)
    idx = np.stack(np.unravel_index(cells, tuple(int(g) for g in grid)), -1)
    centroid_frac = (idx + rng.uniform(-0.3, 0.3, idx.shape)) / grid   # rounds back onto idx
    k_rows = rng.permutation(nk)
    ns = None
    psi, U_true = None, np.zeros((nk, nb, nb), np.complex128)
    with h5py.File(path, "r") as w:
        rk = np.asarray(w["mf_header/kpoints/rk"][()], dtype=np.float64)
    rk = rk.T if rk.shape[0] == 3 and rk.shape[1] != 3 else rk
    for j in range(nk):
        g, c, _ = _read_k(path, j, b_lo, b_hi)                    # (nbg, ns, ngk)
        ns = c.shape[1]
        Af = np.einsum("bsg,gm->bsm", c, np.exp(2j * np.pi * ((g + rk[j]) @ (idx / grid).T)))
        if psi is None:
            psi = rng.standard_normal((nk, nb, ns, n_mu + 3)) * (1 + 1j)   # pad μ: garbage
        U_true[j, :nv, :nv], U_true[j, nv:, nv:] = _unitary(rng, nv), _unitary(rng, nc)
        L = np.zeros((nb, nb), np.complex128)
        L[:nv, nv:] = rng.standard_normal((nv, nc)); L[nv:, :nv] = rng.standard_normal((nc, nv))
        out = [b for b in range(b_hi - b_lo) if not w0 <= b < w0 + nb]
        Win = Af[w0:w0 + nb]
        ps = (np.einsum("nm,nsr->msr", U_true[j] + leak * L, Win)
              + admix * np.einsum("om,osr->msr", rng.standard_normal((len(out), nb)), Af[out])
              + noise * np.abs(Win).mean() * (rng.standard_normal((nb, ns, n_mu))
                                              + 1j * rng.standard_normal((nb, ns, n_mu))))
        psi[k_rows[j], :, :, :n_mu] = scale * ps
    enk_win = np.zeros((nk, nb))
    enk_win[k_rows] = el[:, bands_abs] + 1.0e-5
    kw = dict(psi=psi, wfn_co=path, k_rows=k_rows, bands_abs=bands_abs,
              centroid_frac=centroid_frac, fft_grid=tuple(int(g) for g in grid), nv=nv,
              enk_win=enk_win)
    return kw, U_true, el


def _mapped(kw, U):
    """The call site's ψ_file,n = Σ_m ψ_LX,m conj(U[n, m]) in WFN row order."""
    return np.einsum("knsr,kmn->kmsr", kw["psi"][kw["k_rows"]], np.conj(U))


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_device_gauge_map_equals_host_fit(name):
    kw, U_true, _ = _system(name)
    U_h, f_h = _host_gauge_map(**kw)
    U_d, f_d = gauge_map(**kw, mesh_xy=_mesh())
    assert U_d.shape == U_h.shape and U_d.dtype == np.complex128
    assert np.max(np.abs(U_d - U_h)) < 1e-10
    p_h, p_d = _mapped(kw, U_h), _mapped(kw, U_d)
    assert np.linalg.norm(p_d - p_h) < 1e-10 * np.linalg.norm(p_h)
    for key in RECEIPTS:
        assert np.isclose(f_d[key], f_h[key], rtol=1e-6, atol=1e-15), (key, f_d[key], f_h[key])
    assert np.max(np.abs(U_d - U_true)) < 0.05          # the planted gauge, to the leak level
    assert 1e-6 < f_d["lsq_residual_max"] < 1e-2 and 1e-4 < f_d["cross_block_max"] < 1e-2


def test_red_twin_rephased_file_moves_U_by_its_phases(tmp_path):
    kw, _, _ = _system("Na_ns1", seed=11)
    twin = tmp_path / "WFN.h5"
    shutil.copy(kw["wfn_co"], twin)
    rng = np.random.default_rng(3)
    with h5py.File(twin, "r+") as w:
        ngk = np.asarray(w["mf_header/kpoints/ngk"][()], dtype=np.int64)
        off = np.concatenate([[0], np.cumsum(ngk)])
        c = w["wfns/coeffs"][()]
        cz = c[..., 0] + 1j * c[..., 1]                                # (nb, ns, ngktot)
        theta = rng.uniform(0, 2 * np.pi, (len(ngk), cz.shape[0]))
        for j in range(len(ngk)):
            cz[:, :, off[j]:off[j + 1]] *= np.exp(1j * theta[j])[:, None, None]
        w["wfns/coeffs"][...] = np.stack([cz.real, cz.imag], -1)
    m = _mesh()
    U0, f0 = gauge_map(**kw, mesh_xy=m)
    U1, f1 = gauge_map(**{**kw, "wfn_co": str(twin)}, mesh_xy=m)
    D = np.exp(-1j * theta[:, kw["bands_abs"]])                         # ψ_file' = ψ_file e^{iθ}
    assert np.max(np.abs(U1 - D[:, :, None] * U0)) < 1e-10
    assert np.max(np.abs(U1 - U0)) > 0.1                               # a gauge error U can see
    for key in RECEIPTS:
        assert np.isclose(f1[key], f0[key], rtol=1e-6, atol=1e-15)     # ... and no receipt can


def test_red_twins_k_map_energy_and_alias_gates():
    kw, _, el = _system("H_ns2", seed=5)
    m = _mesh()
    swapped = kw["k_rows"].copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    enk = kw["enk_win"].copy()
    enk[swapped] = el[:, kw["bands_abs"]]                              # energies pass the gate
    _, f = gauge_map(**{**kw, "k_rows": swapped, "enk_win": enk}, mesh_xy=m)
    assert f["lsq_residual_max"] > 0.1 and f["energy_mismatch_max_ry"] < 1e-12
    shifted = kw["enk_win"].copy()
    shifted[kw["k_rows"][2]] += 0.1
    with pytest.raises(ValueError, match="bsemat_band_offset: WFN_co row 2"):
        gauge_map(**{**kw, "enk_win": shifted}, mesh_xy=m)
    with pytest.raises(ValueError, match="bsemat_gauge_box_alias"):
        gauge_map(**{**kw, "fft_grid": (4, 4, 4)}, mesh_xy=m)
