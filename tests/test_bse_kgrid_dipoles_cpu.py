"""``bse_k_grid`` dipoles: the htransform-interpolated velocity (CPU, 2x2 host mesh).

``bse.bse_densify.densified_dipoles`` on a toy tight-binding slab whose
orbitals are the htransform nodes (all at the cell origin, unit metric), so the
coarse states ARE ``ctilde`` and the exact fine-grid dipoles are one ``eigh``
away.  Run with
``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.

1. Every fine q that is a coarse k reproduces the coarse velocity in the
   fine states (the gate's identity), and off the coarse grid the dipoles
   match the exact ones phase-coherently, in the densified states' own gauge.
2. A gauge planted on the coarse states AND their velocity changes nothing;
   planted on the velocity alone (the dipfix failure) it is seen, and the
   velocity image loses its locality.
3. Under a scissor the fine states come from the QP f(H) while ``r = v/ΔE``
   keeps the velocity file's energies: the dipoles do not move with the QP
   energies, and dividing by the QP ΔE (the red twin) would.
4. A permuted fine q list fires ``GATE densified_dipole_on_grid``;
   ``window_dipoles`` refuses a densified bundle without its own dipoles, an
   undensified bundle whose dipole has another k list, and one whose dipole
   is stamped for another WFN (or not stamped).
"""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from bandstructure.bse_setup import compute_wfns_fi  # noqa: E402
from bse.absorption_common import window_dipoles  # noqa: E402
from bse.bse_densify import densified_dipoles  # noqa: E402
from ffi.fft import xla_reference  # noqa: E402
from isdf.galerkin import NodalInterp  # noqa: E402

KC, KF = (6, 6, 1), (9, 9, 1)          # non-nested: 3x3 shared points
ONSITE = np.r_[np.linspace(-1.6, -0.7, 4), np.linspace(0.6, 1.5, 4), 5.0, 5.3]
NB, B_MIN, B_MAX, N_VAL = 8, 2, 6, 2   # fitted window, BSE window, valence rows


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _model(seed=3, hop=0.05):
    rng = np.random.default_rng(seed)
    m = len(ONSITE)
    a = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    h = {(0, 0): np.diag(ONSITE) + 0.02 * (a + a.conj().T)}
    for r in ((1, 0), (0, 1), (1, 1), (1, -1)):
        b = hop * (rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m)))
        h[r], h[(-r[0], -r[1])] = b, b.conj().T
    return h


def _grid(n):
    axes = [np.arange(v) / v for v in n]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return (g + 0.5) % 1.0 - 0.5


def _states(h, kpts, phases=None):
    """C (nk, NB, M) = ⟨orbital|ψ⟩, E (nk, NB), v (nk, 3, NB, NB) = ⟨m|∂H|n⟩."""
    cs, es, vs = [], [], []
    for i, k in enumerate(kpts):
        hk = sum(np.exp(2j * np.pi * (k[0] * r[0] + k[1] * r[1])) * x
                 for r, x in h.items())
        dk = [sum(1j * r[a] * np.exp(2j * np.pi * (k[0] * r[0] + k[1] * r[1])) * x
                  for r, x in h.items()) for a in (0, 1)]
        e, u = np.linalg.eigh(hk)
        u = u[:, :NB] * (np.ones(NB) if phases is None else phases[i])[None]
        vk = np.stack([u.conj().T @ d @ u for d in dk] + [np.zeros((NB, NB))])
        cs.append(u.T); es.append(e[:NB]); vs.append(vk)
    return np.array(cs), np.array(es), np.array(vs)


def _sym(kpts):
    nk = len(kpts)
    return SimpleNamespace(
        irr_idx_k=np.arange(nk), sym_idx_k=np.zeros(nk, np.int32),
        sym_mats_k=np.zeros((2, 3, 3)), nk_red=nk, nk_tot=nk,
        unfolded_kpts=np.asarray(kpts),
        cartesian_action=lambda rows, axial, time_odd: np.broadcast_to(
            np.eye(3), (len(np.asarray(rows)), 3, 3)))


def _nodes(c):
    """The toy's orbitals as co-located nodes; their nodal values are ``c``."""
    m = int(c.shape[2])
    return NodalInterp(jnp.asarray(c), np.zeros((m, 3)),
                       np.diag([1.0, 1.0, 10.0]), 2, 1)


def _densify(mesh, c, e, v, *, fh_energies=None, e_file=None, q_perm=None,
             gate_energies=None):
    """``compute_wfns_fi`` + ``densified_dipoles`` on the toy, XLA backend."""
    kc = _grid(KC)
    e_fh = e if fh_energies is None else fh_energies
    nodes = _nodes(c)
    with xla_reference():
        fine = compute_wfns_fi(
            ctilde=jnp.asarray(c), B_at_mu=nodes,
            enk_sigma=jnp.asarray(e_fh.T),
            kgrid_co=KC, kgrid_fi=KF, band_window_fi=(B_MIN, B_MAX),
            mesh_xy=mesh, return_coeffs=True)
        if q_perm is not None:
            fine.q_frac = fine.q_frac[q_perm]
        dip = densified_dipoles(
            (v, e if e_file is None else e_file, "toy dH/dk", "toy.h5"),
            sym=_sym(kc), fh_energies=e_fh if gate_energies is None else gate_energies,
            basis=SimpleNamespace(ctilde=jnp.asarray(c), band_range=(0, NB),
                                  basis_at_nodes=nodes),
            fine=fine, coarse_grid=KC, n_val=N_VAL, window_start=B_MIN,
            a_band_index=None,
            mesh_xy=mesh, log_fn=lambda *a: None)
    return dip, fine


def _exact_rotated(h, fine):
    """Exact d at the fine q, rotated into the densified states' own gauge."""
    ce, ee, ve = _states(h, fine.q_frac)
    lo, hi = B_MIN, B_MAX
    de = ee[:, lo + N_VAL:hi, None] - ee[:, None, lo:lo + N_VAL]
    dx = ve[:, :, lo + N_VAL:hi, lo:lo + N_VAL] / de[:, None]
    w = np.einsum("qna,qam->qnm", ce[:, lo:hi].conj(), np.asarray(fine.coeffs_fi))
    return np.einsum("qcd,qxce,qef->xqdf", w[:, N_VAL:, N_VAL:].conj(), dx,
                     w[:, :N_VAL, :N_VAL]), dx


def _rel(a, b):
    return float(np.abs(a - b)[:2].max() / np.abs(b)[:2].max())


def test_on_grid_identity_and_off_grid_dipoles():
    mesh, h = _mesh(), _model()
    c, e, v = _states(h, _grid(KC), phases=np.exp(2j * np.pi * np.random.default_rng(
        11).random((np.prod(KC), NB))))
    dip, fine = _densify(mesh, c, e, v)
    assert dip["n_on_grid"] == 9 and dip["on_grid_rel"] < 1e-12
    ref, _ = _exact_rotated(h, fine)
    assert dip["d_alpha"].shape == (3, np.prod(KF), B_MAX - B_MIN - N_VAL, N_VAL)
    # The htransform's own interpolation error on this 6x6 toy is ~1%.
    assert _rel(dip["d_alpha"], ref) < 0.02
    assert np.all(dip["de_cv"] > 0)


def test_coarse_gauge_is_invisible_and_a_velocity_only_gauge_is_seen():
    mesh, h = _mesh(), _model()
    kc = _grid(KC)
    rng = np.random.default_rng(5)
    c0, e, v0 = _states(h, kc)
    ph = np.exp(2j * np.pi * rng.random((len(kc), NB)))
    c1, _, v1 = _states(h, kc, phases=ph)
    a, fine_a = _densify(mesh, c0, e, v0)
    b, fine_b = _densify(mesh, c1, e, v1)
    ra, _ = _exact_rotated(h, fine_a)
    rb, _ = _exact_rotated(h, fine_b)
    assert _rel(a["d_alpha"], ra) < 0.02 and _rel(b["d_alpha"], rb) < 0.02
    np.testing.assert_allclose(np.abs(a["d_alpha"]), np.abs(b["d_alpha"]),
                               atol=1e-9 * np.abs(a["d_alpha"]).max())
    # Red twin: the velocity in another gauge than the states (dipfix).
    bad, fine_bad = _densify(mesh, c0, e, v1)
    rbad, _ = _exact_rotated(h, fine_bad)
    assert _rel(bad["d_alpha"], rbad) > 0.2
    assert bad["image_locality"]["v_x"][0] > 10 * a["image_locality"]["v_x"][0]


def test_scissor_keeps_the_velocity_energies():
    mesh, h = _mesh(), _model()
    c, e, v = _states(h, _grid(KC))
    qp = e + 0.3 * (np.arange(NB) >= 4)[None]            # conduction scissor
    dft, _ = _densify(mesh, c, e, v)
    sc, fine = _densify(mesh, c, e, v, fh_energies=qp)
    _, ee, _ = _states(h, fine.q_frac)
    de_x = ee[:, B_MIN + N_VAL:B_MAX, None] - ee[:, None, B_MIN:B_MIN + N_VAL]
    err = lambda de: float(np.abs(de - de_x).max())
    # The fine states come from the QP f(H); the denominators stay the
    # velocity's DFT energies (Rayleigh quotients, second order).
    assert err(sc["de_cv"]) < 1.5 * err(dft["de_cv"]) + 1e-3
    e_qp = np.asarray(fine.enk_full)
    de_qp = e_qp[:, N_VAL:, None] - e_qp[:, None, :N_VAL]
    assert err(de_qp) > 0.25                             # the bundle's ΔE is QP
    assert _rel(np.abs(sc["d_alpha"]), np.abs(dft["d_alpha"])) < 0.06
    # Red twin: the QP ΔE in the denominator is off by ΔE_DFT/ΔE_QP.
    wrong = sc["d_alpha"] * sc["de_cv"][None] / de_qp[None]
    assert _rel(np.abs(wrong), np.abs(dft["d_alpha"])) > 0.2


def test_permuted_q_list_fires_the_on_grid_gate():
    mesh, h = _mesh(), _model()
    c, e, v = _states(h, _grid(KC))
    perm = np.roll(np.arange(np.prod(KF)), 1)
    with pytest.raises(ValueError, match="GATE densified_dipole_on_grid"):
        _densify(mesh, c, e, v, q_perm=perm)


def test_window_dipoles_refusals(tmp_path, monkeypatch):
    import bse.absorption_common as absorption_common
    import file_io.dipole as dipole
    monkeypatch.setattr(dipole, "wfn_psi_basis", lambda path: "dft")
    monkeypatch.setattr(absorption_common, "_deck_wfn_fingerprint",
                        lambda path: "sha-of-the-deck-wfn")
    dense = {"densified": True, "dipoles": None, "nkx": 3, "nky": 3, "nkz": 1,
             "n_val": 1, "n_cond": 1}
    with pytest.raises(ValueError, match="GATE densified_dipoles"):
        window_dipoles(dense, tmp_path / "dipole.h5", n_occ=1, wfn_path="WFN.h5")
    path = tmp_path / "dipole.h5"
    with h5py.File(path, "w") as f:
        f["dipole_cart"] = np.zeros((3, 4, 2, 2), complex)
        f["band_energies"] = np.tile([-1.0, 1.0], (4, 1))
        f.attrs.update({"nk": 4, "nbands": 2, "basis": "dft",
                        "prov_wfn_sha256": "sha-of-the-deck-wfn"})
    coarse = dict(dense, densified=False)
    with pytest.raises(ValueError, match="GATE dipole_k_grid"):
        window_dipoles(coarse, path, n_occ=1, wfn_path="WFN.h5")
    ok = dict(coarse, nkx=2, nky=2)
    d, de = window_dipoles(ok, path, n_occ=1, wfn_path="WFN.h5")
    assert d.shape == (3, 4, 1, 1) and de.shape == (4, 1, 1)
    # Another WFN's file on the same k list, and an unstamped one (AUDIT B12).
    for stamp in ("sha-of-another-wfn", None):
        with h5py.File(path, "a") as f:
            if stamp is None:
                del f.attrs["prov_wfn_sha256"]
            else:
                f.attrs["prov_wfn_sha256"] = stamp
        with pytest.raises(ValueError, match="GATE velocity_wfn_fingerprint"):
            window_dipoles(ok, path, n_occ=1, wfn_path="WFN.h5")


def test_on_grid_identity_follows_the_fh_band_order():
    """A QP ladder that moves a band across the window edge at a shared k.

    The fH's own order picks the window there (MoS2 9x9 eqp1 at Gamma swaps
    DFT bands 18-19 out of [18, 34) for 16-17), so the on-grid state identity
    reads the window rows from ``fh_energies``; the red twin hands the gate
    the DFT order and must fire.
    """
    mesh, h = _mesh(), _model()
    c, e, v = _states(h, _grid(KC))
    i0 = int(np.flatnonzero(np.all(np.abs(_grid(KC)) < 1e-12, axis=1))[0])
    e_qp = e.copy()
    e_qp[i0, B_MIN - 1], e_qp[i0, B_MIN] = e[i0, B_MIN] + 0.02, e[i0, B_MIN - 1]
    assert list(np.argsort(e_qp[i0])[B_MIN:B_MAX]) != list(range(B_MIN, B_MAX))
    dip, _ = _densify(mesh, c, e, v, fh_energies=e_qp)
    assert dip["n_on_grid"] == 9
    assert dip["on_grid_state_dev"] < 1e-8 and dip["on_grid_rel"] < 1e-12
    with pytest.raises(ValueError, match="GATE densified_dipole_on_grid"):
        _densify(mesh, c, e, v, fh_energies=e_qp, gate_energies=e)
