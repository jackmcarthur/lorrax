"""Orbital magnetization consumes the typed symmetry action, not raw tables."""

from types import SimpleNamespace

import numpy as np

from psp import orbital_magnetization as orbmag


class _TypedMagneticSymmetry:
    """QE typed the z-preserving operation as antiunitary, not unitary."""

    active_symmetry_rows = np.asarray([0, 3], dtype=np.int32)
    nk_tot = 1
    trs_allowed = False
    operation_typing_source = "qe-schema"

    def __init__(self):
        self.calls = []

    def cartesian_action(self, rows, *, axial, time_odd):
        self.calls.append((np.asarray(rows).copy(), axial, time_odd))
        assert axial is True
        assert time_odd is True
        # row 0: identity; row 3: time-odd antiunitary C2x preserves z and
        # reverses x.  Its unitary row-1 partner is not an active operation.
        return np.asarray([
            np.eye(3),
            np.diag([-1.0, 1.0, 1.0]),
        ])


class _OneKPointTable:
    kvecs = np.zeros((1, 3), dtype=np.float64)

    @staticmethod
    def at(_):
        return (np.zeros((1, 3), dtype=np.int32),
                np.ones(1, dtype=np.bool_))


def _run(monkeypatch, sym, *, psi=None):
    if psi is None:
        psi = np.zeros((1, 1, 2, 1), dtype=np.complex128)
    wfn = SimpleNamespace(
        nkpts=1,
        kweights=np.ones(1),
        bvec=np.eye(3),
        blat=1.0,
        energies=np.zeros((1, 1, 1)),
        load=lambda **_: psi,
    )
    monkeypatch.setattr(orbmag, "padded_gvectors",
                        lambda *_, **__: _OneKPointTable())
    monkeypatch.setattr(
        orbmag, "momentum_matrix_k",
        lambda *_, **__: np.zeros((3, 1, 1), dtype=np.complex128))

    # Make the projector visible in the returned vector.  Keeping only the
    # presumed-unitary half would leave x=1; the typed rows project it to 0.
    pa = np.asarray([1.0, 2.0, 3.0], dtype=np.complex128)[:, None, None]
    monkeypatch.setattr(
        orbmag, "orbital_pieces_at_k",
        lambda *_, **__: (pa, np.zeros_like(pa)))

    return orbmag.run_ibz(
        wfn, sym, meta=None, vnl_setup=None, nbnd=1, nocc=1,
        deps_tol=1.0e-8, m_axis=(0.0, 0.0, 1.0), sign=0)


def test_ibz_uses_active_time_odd_axial_action_and_keeps_antiunitary(
        monkeypatch):
    sym = _TypedMagneticSymmetry()
    cA, cB, pa_z, pb_z, _, _, info = _run(monkeypatch, sym)

    assert len(sym.calls) == 1
    np.testing.assert_array_equal(sym.calls[0][0], sym.active_symmetry_rows)
    np.testing.assert_array_equal(cA, np.asarray([0.0, 2.0, 3.0]))
    np.testing.assert_array_equal(cB, np.zeros(3))
    np.testing.assert_array_equal(pa_z, np.asarray([[3.0]]))
    np.testing.assert_array_equal(pb_z, np.asarray([[0.0]]))
    assert info["idx"] == [0, 3]
    assert info["nG"] == 2


def test_nonmagnetic_pure_time_reversal_projects_the_moment_to_zero(
        monkeypatch):
    class _NonmagneticSymmetry(_TypedMagneticSymmetry):
        active_symmetry_rows = np.asarray([0, 1], dtype=np.int32)
        trs_allowed = True

        def cartesian_action(self, rows, *, axial, time_odd):
            assert axial is True and time_odd is True
            return np.asarray([np.eye(3), -np.eye(3)])

    psi = np.zeros((1, 1, 2, 1), dtype=np.complex128)
    psi[0, 0, 0, 0] = 1.0
    cA, cB, pa_z, pb_z, m_spin_z, _, info = _run(
        monkeypatch, _NonmagneticSymmetry(), psi=psi)
    np.testing.assert_array_equal(cA, np.zeros(3))
    np.testing.assert_array_equal(cB, np.zeros(3))
    np.testing.assert_array_equal(pa_z, np.zeros((1, 1)))
    np.testing.assert_array_equal(pb_z, np.zeros((1, 1)))
    assert m_spin_z == 0.0
    assert info["idx"] == [0, 1]
