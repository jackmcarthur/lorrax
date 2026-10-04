"""htransform's modern-theory orbital route on toy inputs (CPU, seconds).

The star-weighted parent total equals the full-BZ sum, the moved SOS pieces
equal ``orbital_response.orbital_magnetization``, and a stored velocity that
is not the WFN's, or a QP velocity with no Sigma term, refuses before any
payload read.
"""
from types import SimpleNamespace

import h5py
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from bandstructure import orbital  # noqa: E402
from file_io import dipole  # noqa: E402
from psp.orbital_response import (  # noqa: E402
    orbital_magnetization, orbital_pieces_at_k)

_R = np.diag([1.0, -1.0, -1.0])          # C2 about x: polar = axial action


def _velocity(rng, nb):
    a = rng.standard_normal((3, nb, nb)) + 1j * rng.standard_normal((3, nb, nb))
    return a + np.conj(np.swapaxes(a, 1, 2))


def test_sos_pieces_equal_the_thermodynamic_moment():
    rng = np.random.default_rng(1)
    v, e = _velocity(rng, 8), np.sort(rng.standard_normal(8))
    mu = 0.5 * (e[2] + e[3])
    pa, pb = orbital_pieces_at_k(v, e, 3, 1e-8)
    sos = 0.5 * (pa - 2 * mu * pb).sum(axis=(1, 2)).imag
    lib = np.asarray(orbital_magnetization(v, e, mu_ry=mu, width_ry=0.0))
    np.testing.assert_allclose(sos, lib, atol=1e-12)


def test_parent_total_with_axial_star_weights_equals_the_full_bz_sum():
    rng = np.random.default_rng(2)
    nb, nelec = 12, 4
    v, e = _velocity(rng, nb), np.sort(rng.standard_normal(nb))
    v_image = np.einsum("ab,bnm->anm", _R, v)      # velocity at C2x k
    sym = SimpleNamespace(
        irr_idx_k=np.array([0, 0]), nk_tot=2, active_symmetry_rows=[0, 1],
        cartesian_action=lambda rows, axial, time_odd: np.stack(
            [np.eye(3), _R])[np.asarray(rows)])
    mu, ceilings, _, m = orbital.orbital_totals(
        v[None], e[None], sym, nelec=nelec, width_ry=None, deps_tol_ry=1e-8)
    assert ceilings[-1] == nb and mu == 0.5 * (e[nelec - 1] + e[nelec])
    full = 0.5 * sum(np.asarray(orbital_magnetization(
        x, e, mu_ry=mu, width_ry=0.0)) for x in (v, v_image))
    np.testing.assert_allclose(m[-1], full, atol=1e-12)
    assert abs(m[-1][0]) > 1e-3                     # the check sees a moment


def _stamped(path, *, basis, label=None, fingerprint="wfn-a"):
    with h5py.File(path, "w") as h5:
        h5["band_energies"] = np.zeros((2, 4))
        h5.attrs["basis"] = basis
        h5.attrs["prov_wfn_sha256"] = fingerprint
        if label is not None:
            h5.attrs["velocity"] = label
    return path


@pytest.fixture
def qp_wfn(monkeypatch):
    import common.parallel_transport as pt
    monkeypatch.setattr(pt, "wfn_fingerprint", lambda wfn: "wfn-a")
    monkeypatch.setattr(dipole, "wfn_psi_basis", lambda path: "qp")
    return SimpleNamespace(kirr_fullids=[0], nk_tot=2)


def test_a_dft_velocity_on_a_qp_wfn_refuses(tmp_path, qp_wfn):
    path = _stamped(tmp_path / "dipole.h5", basis="dft")
    with pytest.raises(ValueError, match="GATE dipole_basis"):
        orbital.stored_velocity(path, wfn=None, wfn_path="WFN_qp.h5",
                                sym=qp_wfn, mesh=None)


def test_a_qp_velocity_without_a_sigma_term_refuses(tmp_path, qp_wfn):
    path = _stamped(tmp_path / "dipole_qsgw.h5", basis="qp",
                    label="v_DFT (dft_velocity: no Sigma term)")
    with pytest.raises(ValueError, match="GATE qp_velocity_sigma_term"):
        orbital.stored_velocity(path, wfn=None, wfn_path="WFN_qp.h5",
                                sym=qp_wfn, mesh=None)


def test_another_wfns_velocity_refuses(tmp_path, qp_wfn):
    path = _stamped(tmp_path / "dipole_qsgw.h5", basis="qp",
                    label="v_DFT + D_k DeltaH (parallel_transport links)",
                    fingerprint="wfn-b")
    with pytest.raises(ValueError, match="GATE velocity_wfn_fingerprint"):
        orbital.stored_velocity(path, wfn=None, wfn_path="WFN_qp.h5",
                                sym=qp_wfn, mesh=None)
