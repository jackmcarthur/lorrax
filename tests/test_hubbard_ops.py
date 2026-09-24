"""psp.hubbard_ops: QE's DFT+U potential and its k-derivative (the i[r, V_U] velocity term).

The physical gate is a cluster leg (runs/runtime/dftu_velocity_20260923: Hellmann-Feynman on VI3 with and
without U, and a covariant-finite-difference referee for the off-diagonal); these are its cheap CPU twins:

* the Liechtenstein potential of the VI3 X6p0 ``occup.txt`` reproduces QE's printed Hubbard energies
  (``Hub. E (dc, noflip, flip, total)``, nscf12_X6p0/nscf.out:527, Ry) -- a sign/index/Y_lm convention slip in
  u_matrix or ns changes these digits;
* u_matrix averages to U and J (the Slater parametrization);
* d(O^{-1/2}) and the ortho-atomic rows' k-derivative agree with a central difference;
* v_U = d/dk <psi|V_U(k)|psi> at fixed psi (central difference), and v_U is Hermitian and band-covariant;
* the Dudarev limit and QE's j-averaging rule.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from psp import hubbard_ops as ho

FIX = Path(__file__).parent / "core" / "fixtures" / "hubbard" / "vi3_X6p0_occup.txt"
QE_HUB_E = (3.3586, 3.7085, -0.0027, 0.3472)     # QE 7.6 nscf.out, 4 decimals, from this occup.txt


def test_liechtenstein_energy_matches_qe_vi3():
    ns = ho.read_occup_nc(FIX, nat=8, ldmx=5)
    U, J = 6.0 / ho.RYTOEV, ho.default_hubbard_J(2, 0.8 / ho.RYTOEV)
    tot = np.zeros(4)
    for a in (0, 1):
        v, e = ho.v_hubbard_full_nc(ns[:, :, :, a], 2, U, J)
        tot += np.asarray(e)
        W = ho.hubbard_W(v)
        assert np.max(np.abs(W - W.conj().T)) < 1e-12
    assert np.max(np.abs(tot - np.asarray(QE_HUB_E))) < 6e-5, tot
    assert np.all(np.abs(ns[:, :, :, 2:]) == 0.0)          # non-Hubbard atoms carry no ns


def test_u_matrix_slater_averages():
    l, U, J1 = 2, 0.4, 0.07
    u = ho.hubbard_u_matrix(l, U, ho.default_hubbard_J(l, J1))
    n = 2 * l + 1
    direct = np.einsum("abab->ab", u)
    exch = np.einsum("abba->ab", u)
    assert abs(direct.mean() - U) < 1e-12
    off = ~np.eye(n, dtype=bool)
    assert abs(U - (direct[off] - exch[off]).mean() - J1) < 1e-12
    assert np.max(np.abs(u - u.transpose(2, 3, 0, 1))) < 1e-12
    assert np.max(np.abs(u - u.transpose(1, 0, 3, 2))) < 1e-12


def _rows(k, seed=3, R=7, nG=40):
    rng = np.random.default_rng(seed)
    A = [rng.normal(size=(R, nG)) + 1j * rng.normal(size=(R, nG)) for _ in range(3)]
    Z = A[0] + k * A[1] + k ** 2 * A[2]
    dZ = A[1] + 2 * k * A[2]
    return jnp.asarray(Z), jnp.asarray(np.stack([dZ, 0 * dZ, 0 * dZ]))


def test_lowdin_derivative_matches_central_difference():
    hub = jnp.asarray([2, 3, 4])
    k, h = 0.3, 1e-5
    Zt, dZt, lam = ho.lowdin_rows_with_derivative(*_rows(k), hub)
    Zp, _, _ = ho.lowdin_rows_with_derivative(*_rows(k + h), hub)
    Zm, _, _ = ho.lowdin_rows_with_derivative(*_rows(k - h), hub)
    fd = (np.asarray(Zp) - np.asarray(Zm)) / (2 * h)
    assert np.max(np.abs(fd - np.asarray(dZt[0]))) < 1e-6 * np.max(np.abs(fd))
    Zfull, _, _ = ho.lowdin_rows_with_derivative(*_rows(k), jnp.arange(7))
    O = np.conj(np.asarray(Zfull)) @ np.asarray(Zfull).T
    assert np.max(np.abs(O - np.eye(7))) < 1e-10               # orthonormal rows


def test_hubbard_velocity_is_dk_of_hubbard_matrix():
    rng = np.random.default_rng(11)
    nb, nG, ld = 5, 40, 3
    psi = jnp.asarray(rng.normal(size=(nb, 2, nG)) + 1j * rng.normal(size=(nb, 2, nG)))
    A = rng.normal(size=(2 * ld, 2 * ld)) + 1j * rng.normal(size=(2 * ld, 2 * ld))
    W = jnp.asarray((A + A.conj().T)[None])
    hub = jnp.asarray([1, 2, 3])
    k, h = -0.2, 1e-5

    def at(kk, p=psi):
        Zt, dZt, _ = ho.lowdin_rows_with_derivative(*_rows(kk), hub)
        return ho.hubbard_matrix_and_velocity(p, Zt, dZt, W)

    H, v, _ = at(k)
    Hp, _, _ = at(k + h)
    Hm, _, _ = at(k - h)
    fd = (np.asarray(Hp) - np.asarray(Hm)) / (2 * h)
    v = np.asarray(v)
    assert np.max(np.abs(fd - v[0])) < 1e-6 * np.max(np.abs(fd))
    assert np.max(np.abs(v - np.conj(np.transpose(v, (0, 2, 1))))) < 1e-10
    Q, _ = np.linalg.qr(rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb)))
    _, vq, _ = at(k, jnp.einsum("msG,mn->nsG", psi, jnp.asarray(Q)))
    assert np.max(np.abs(np.asarray(vq) - np.einsum("mi,amn,nj->aij", Q.conj(), v, Q))) < 1e-10


def test_dudarev_diagonal_limit():
    ld, U = 5, 0.3
    occ = np.linspace(0.1, 0.9, ld)
    ns = np.zeros((ld, ld, 4), dtype=complex)
    ns[:, :, 0] = np.diag(occ)
    ns[:, :, 3] = np.diag(occ[::-1])
    v, _ = ho.v_hubbard_dudarev_nc(ns, 2, U)
    assert np.allclose(np.diag(v[:, :, 0]).real, U * (0.5 - occ))
    assert np.allclose(np.diag(v[:, :, 3]).real, U * (0.5 - occ[::-1]))
    assert np.allclose(v[:, :, 1], 0) and np.allclose(v[:, :, 2], 0)


def test_j_averaging_follows_atomic_wfc_so_mag():
    r = np.linspace(0, 5, 11)
    c = {k: np.full(11, float(i + 1)) for i, k in enumerate(("s", "p32", "p12", "x"))}
    pswfc = {"r": r, "rab": np.ones(11), "chi": [
        {"label": "3S", "l": 0, "occupation": 2.0, "j": 0.5, "chi": c["s"]},
        {"label": "3P", "l": 1, "occupation": 4.0, "j": 1.5, "chi": c["p32"]},
        {"label": "3P", "l": 1, "occupation": 2.0, "j": 0.5, "chi": c["p12"]},
        {"label": "4F", "l": 3, "occupation": -1.0, "j": 3.5, "chi": c["x"]}]}
    out = ho.j_averaged_atomic_wfcs(pswfc)
    assert [(w["label"], w["l"]) for w in out] == [("3S", 0), ("3P", 1)]
    assert np.allclose(out[1]["chi"], (2 * c["p32"] + c["p12"]) / 3)


def test_read_occup_nc_is_fortran_order(tmp_path):
    ld, nat = 3, 2
    ns = (np.arange(ld * ld * 4 * nat) + 1j * np.arange(ld * ld * 4 * nat)[::-1]).reshape(
        (ld, ld, 4, nat), order="F")
    p = tmp_path / "occup.txt"
    p.write_text("\n".join(f" ({z.real:.17g},{z.imag:.17g})" for z in ns.flatten(order="F")))
    assert np.array_equal(ho.read_occup_nc(p, nat=nat, ldmx=ld), ns)
