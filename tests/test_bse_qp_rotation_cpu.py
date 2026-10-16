"""The QP-rotated BSE basis (``bse.bse_qp_rotation``) on toy inputs (CPU).

Every check is paired with the case that must fail it: a permuted U for the
U = I dipole identity, an unconjugated antiunitary row for the star gauge, a
degenerate QP edge and a window outside the block for the window guard.
The sharded rotation runs on a 2x2 host mesh (``JAX_PLATFORMS=cpu`` with
``--xla_force_host_platform_device_count=4``), the four-device logic case of
the four-GPU rule; the GPU leg owns the driver.
"""
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from bse.absorption_common import slice_dipole_to_bse_window
from bse.bse_qp_rotation import (certify_star_gauge, qp_window_dipoles,
                                 qp_window_from_artifact, rotate_window_psi,
                                 star_parents, star_relation_residuals)
from common.band_degeneracy import BandWindowDegeneracyError

jax.config.update("jax_enable_x64", True)
RNG = np.random.default_rng(20261016)


def _unitary(n, rng=RNG):
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _hermitian(n, rng=RNG):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return 0.5 * (a + a.conj().T)


def _dft_model(nk=3, nb=10, n_occ=6, rng=RNG):
    """DFT energies with a gap, a Hermitian velocity, and its E_m - E_n."""
    e = np.sort(rng.normal(size=(nk, nb)), axis=1)
    e[:, n_occ:] += 2.0
    v = np.stack([np.stack([_hermitian(nb, rng) for _ in range(nk)])
                  for _ in range(3)])
    de = e[:, :, None] - e[:, None, :]
    return e, v, de


def _qp(U, E, n_occ, n_val, n_cond):
    w = qp_window_from_artifact(U, E, (0, U.shape[1]), n_occ=n_occ,
                                n_val=n_val, n_cond=n_cond,
                                degeneracy_mode="strict",
                                degeneracy_tol_ry=1e-6, log=lambda *_: None)
    w.update(band_range=(0, U.shape[1]), n_occ=n_occ)
    return w


def test_identity_rotation_is_the_eqp_dipole_bitwise_and_permutation_is_not():
    nk, nb, n_occ, nv, nc = 3, 10, 6, 4, 3
    e, v, de = _dft_model(nk, nb, n_occ)
    shift = np.where(np.arange(nb) < n_occ, -0.3, 0.4) + 0.01 * RNG.normal(size=(nk, nb))
    U = np.broadcast_to(np.eye(nb, dtype=complex), (nk, nb, nb)).copy()
    d = qp_window_dipoles(v, de, _qp(U, e + shift, n_occ, nv, nc))
    ref, _ = slice_dipole_to_bse_window(v, de, n_occ, nv, nc)
    assert d.shape == ref.shape == (3, nk, nc, nv)
    assert np.array_equal(d, ref)
    # Red twin: two QP valence columns of the window exchanged.
    Up = U.copy()
    Up[:, :, [n_occ - 1, n_occ - 2]] = Up[:, :, [n_occ - 2, n_occ - 1]]
    dp = qp_window_dipoles(v, de, _qp(Up, e + shift, n_occ, nv, nc))
    assert not np.array_equal(dp, ref)
    assert np.max(np.abs(dp - ref)) > 1e-2 * np.max(np.abs(ref))


def test_cross_gap_free_rotation_rotates_the_position_matrix_exactly():
    nk, nb, n_occ, nv, nc = 2, 10, 6, 6, 4
    e, v, de = _dft_model(nk, nb, n_occ)
    U = np.zeros((nk, nb, nb), dtype=complex)
    for k in range(nk):
        U[k, :n_occ, :n_occ] = _unitary(n_occ)
        U[k, n_occ:, n_occ:] = _unitary(nb - n_occ)
    E = np.sort(e + 0.2 * RNG.normal(size=e.shape), axis=1)
    E[:, n_occ:] += 1.0
    d = qp_window_dipoles(v, de, _qp(U, E, n_occ, nv, nc))
    r_cv = v[:, :, n_occ:, :n_occ] / de[None, :, n_occ:, :n_occ]
    exact = np.einsum("kmc,akmn,knv->akcv", U[:, n_occ:, n_occ:].conj(), r_cv,
                      U[:, :n_occ, :n_occ])
    assert np.allclose(d, exact, rtol=0, atol=1e-12 * np.abs(exact).max())


def test_qp_dipole_is_the_interband_commutator_velocity_over_the_qp_gap():
    """Eq. (3) against v^QP = v^DFT + [ΔH, W] formed in the DFT basis."""
    nk, nb, n_occ = 2, 9, 5
    e, v, de = _dft_model(nk, nb, n_occ)
    cross = np.zeros((nb, nb), bool)
    cross[n_occ:, :n_occ] = cross[:n_occ, n_occ:] = True
    W = np.where(cross[None, None], v / np.where(cross, de, 1.0)[None], 0)
    U = np.empty((nk, nb, nb), complex)
    E = np.empty((nk, nb))
    ref = []
    for k in range(nk):
        dH = 0.3 * _hermitian(nb)               # mixes valence and conduction
        H = np.diag(e[k]) + dH
        E[k], U[k] = np.linalg.eigh(H)
        vqp = v[:, k] + (dH[None] @ W[:, k] - W[:, k] @ dH[None])
        vq = U[k].conj().T[None] @ vqp @ U[k][None]
        ref.append(vq[:, n_occ:, :n_occ]
                   / (E[k, n_occ:, None] - E[k, None, :n_occ])[None])
    ref = np.stack(ref, axis=1)
    d = qp_window_dipoles(v, de, _qp(U, E, n_occ, n_occ, nb - n_occ))
    assert np.allclose(d, ref, rtol=0, atol=1e-11 * np.abs(ref).max())


def test_star_residuals_follow_the_ibz_slab_rule_and_refuse_a_missing_conj():
    irr = np.array([0, 1, 1, 0, 1])
    sidx = np.array([0, 0, 3, 2, 1])           # rows >= 2 are antiunitary
    parent, anti = star_parents(irr, sidx, 2, kirr_fullids=np.array([0, 1]))
    assert parent.tolist() == [0, 1, 1, 0, 1]
    assert anti.tolist() == [False, False, True, True, False]
    base = np.stack([_hermitian(4) for _ in range(5)])
    M = np.where(anti[:, None, None], np.conj(base[parent]), base[parent])
    res, other = star_relation_residuals(M, parent, anti)
    assert res.max() < 1e-15 and other[anti].min() > 1e-2
    certify_star_gauge(M, M, parent, anti, where="toy", log=lambda *_: None)
    bad = M.copy()
    bad[2] = base[1]                           # antiunitary row left unconjugated
    with pytest.raises(ValueError, match="GATE bse_qp_star_gauge"):
        certify_star_gauge(M, bad, parent, anti, where="toy",
                           log=lambda *_: None)


def test_window_guard_refuses_a_cut_qp_edge_and_a_window_outside_the_block():
    nk, nb, n_occ = 2, 8, 4
    E = np.sort(RNG.normal(size=(nk, nb)), axis=1)
    E[:, n_occ:] += 3.0
    U = np.broadcast_to(np.eye(nb, dtype=complex), (nk, nb, nb)).copy()
    E_cut = E.copy()
    E_cut[0, n_occ + 2] = E_cut[0, n_occ + 1] + 1e-9    # degenerate at the edge
    with pytest.raises(BandWindowDegeneracyError):
        _qp(U, E_cut, n_occ, 2, 2)
    with pytest.raises(ValueError, match="GATE bse_qp_window_outside_block"):
        _qp(U, E, n_occ, n_occ + 1, 2)
    w = _qp(U, E, n_occ, 2, 2)
    assert w["diagnostics"]["dft_window_leakage_max"] == 0.0
    # A QP state of the window that is a DFT band above it: leakage 1.
    Ux = U.copy()
    Ux[:, :, [n_occ + 1, n_occ + 2]] = Ux[:, :, [n_occ + 2, n_occ + 1]]
    wx = _qp(Ux, E, n_occ, 2, 2)
    assert wx["diagnostics"]["dft_window_leakage_max"] == 1.0
    assert wx["diagnostics"]["dft_window_leakage_at"][1] == n_occ + 1


def _mesh():
    devices = np.array(jax.devices()[:4])
    if devices.size < 4:
        pytest.skip("needs --xla_force_host_platform_device_count=4")
    return Mesh(devices.reshape(2, 2), ("x", "y"))


def test_sharded_rotation_identity_bitwise_unitary_exact_star_gauge_gated():
    mesh = _mesh()
    nk, nb, n_occ, ns, nmu = 4, 6, 3, 2, 8
    irr = np.array([0, 1, 0, 1])
    sidx = np.array([0, 0, 2, 3])
    parent, anti = star_parents(irr, sidx, 2, kirr_fullids=np.array([0, 1]))
    base = RNG.normal(size=(nk, nb, ns, nmu)) + 1j * RNG.normal(size=(nk, nb, ns, nmu))
    psi = np.where(anti[:, None, None, None], np.conj(base[parent]), base[parent])
    spec = NamedSharding(mesh, P(None, None, None, "x"))
    pv = jax.device_put(psi[:, :n_occ], spec)
    pc = jax.device_put(psi[:, n_occ:], spec)
    E = np.sort(RNG.normal(size=(nk, nb)), axis=1)
    E[:, n_occ:] += 3.0
    U = np.broadcast_to(np.eye(nb, dtype=complex), (nk, nb, nb)).copy()
    qp = _qp(U, E, n_occ, 2, 2)
    qp.update(parent=parent, antiunitary=anti, certify_psi_star=True,
              path="toy.h5")
    out_v, out_c = rotate_window_psi(pv, pc, qp, mesh, log=lambda *_: None)
    assert out_v.sharding.spec == P(None, None, None, "x")
    assert np.array_equal(np.asarray(out_v), psi[:, n_occ - 2:n_occ])
    assert np.array_equal(np.asarray(out_c), psi[:, n_occ:n_occ + 2])
    Ur = np.stack([_unitary(nb) for _ in range(nk)])
    Ur[2], Ur[3] = np.conj(Ur[0]), np.conj(Ur[1])   # its star images
    qr = _qp(Ur, E, n_occ, 2, 2)
    qr.update(parent=parent, antiunitary=anti, certify_psi_star=True,
              path="toy.h5")
    out_v, _ = rotate_window_psi(pv, pc, qr, mesh, log=lambda *_: None)
    ref = np.einsum("kmsu,kmn->knsu", psi, Ur[:, :, n_occ - 2:n_occ])
    assert np.allclose(np.asarray(out_v), ref, rtol=0, atol=1e-13)
    # Red twin: the restart psi of an antiunitary k without its conjugation.
    wrong = psi.copy()
    wrong[3] = base[1]
    with pytest.raises(ValueError, match="GATE bse_qp_star_gauge"):
        rotate_window_psi(jax.device_put(wrong[:, :n_occ], spec),
                          jax.device_put(wrong[:, n_occ:], spec), qp, mesh,
                          log=lambda *_: None)
