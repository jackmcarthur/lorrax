"""Gates for the two-component DFT-reference TRS check."""
from __future__ import annotations

import os
from contextlib import nullcontext

import h5py
import numpy as np
import pytest

from wfn_loader import WfnLoader
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from symmetry_maps import (                                     # noqa: E402
    check_density_symmetries,
    trs_check_mode,
)
from symmetry_maps.density_symmetry_check import (               # noqa: E402
    _classify_trs_evidence,
    _occupation_operator_residual,
    _plan_two_component_evidence,
    _theta_overlap,
)
import symmetry_maps.density_symmetry_check as density_check     # noqa: E402


_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "regression", "cohsex_debug", "WFNsmall.h5")


# ----------------------------------------------------------------------
# Synthetic decks
# ----------------------------------------------------------------------
def _neg_closed_gvecs() -> np.ndarray:
    """A small G-list closed under ``G -> -G`` (index 0 is G=0).

    Closure is what makes the real-space Kramers construction below
    expressible in G space: ``Θψ(r) = iσ_y ψ*(r)`` reads
    ``ψ_partner(G) = iσ_y conj(ψ(−G))``.
    """
    base = [(0, 0, 0)]
    for g in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
              (0, 1, 1), (1, 1, 1), (2, 0, 0), (0, 2, 0)]:
        base.append(g)
        base.append((-g[0], -g[1], -g[2]))
    return np.array(base, dtype=np.int32)


def _write_wfn(path: str, *, coeffs: np.ndarray, gvecs: np.ndarray,
               kpoints: np.ndarray, kgrid, nocc: int, mtrx: np.ndarray,
               tnp: np.ndarray, shift=(0.0, 0.0, 0.0)) -> str:
    """Minimal BGW-shaped WFN.h5.  ``coeffs`` is (nb, 2, ngktot) complex."""
    nb, nspinor, ngktot = coeffs.shape
    nrk = int(kpoints.shape[0])
    ngk = np.full(nrk, ngktot // nrk, dtype=np.int32)
    ntran = int(mtrx.shape[0])
    with h5py.File(path, "w") as f:
        g = f.create_group("mf_header")
        g.create_dataset("versionnumber", data=np.int32(3))
        g.create_dataset("flavor", data=np.int32(2))
        kp = g.create_group("kpoints")
        kp.create_dataset("nspin", data=np.int32(1))
        kp.create_dataset("nspinor", data=np.int32(nspinor))
        kp.create_dataset("nrk", data=np.int32(nrk))
        kp.create_dataset("mnband", data=np.int32(nb))
        kp.create_dataset("ngkmax", data=np.int32(int(ngk.max())))
        kp.create_dataset("ecutwfc", data=np.float64(20.0))
        kp.create_dataset("kgrid", data=np.asarray(kgrid, dtype=np.int32))
        kp.create_dataset("shift", data=np.asarray(shift, dtype=np.float64))
        kp.create_dataset("ngk", data=ngk)
        kp.create_dataset("ifmin", data=np.ones((1, nrk), dtype=np.int32))
        kp.create_dataset("ifmax", data=np.full((1, nrk), nocc, dtype=np.int32))
        kp.create_dataset("w", data=np.full(nrk, 1.0 / nrk, dtype=np.float64))
        kp.create_dataset("rk", data=np.asarray(kpoints, dtype=np.float64))
        kp.create_dataset("el", data=np.tile(
            np.arange(nb, dtype=np.float64), (1, nrk, 1)))
        occ = np.zeros((1, nrk, nb), dtype=np.float64)
        occ[:, :, :nocc] = 1.0
        kp.create_dataset("occ", data=occ)
        gs = g.create_group("gspace")
        gs.create_dataset("ng", data=np.int32(200))
        gs.create_dataset("ecutrho", data=np.float64(80.0))
        gs.create_dataset("FFTgrid", data=np.array([8, 8, 8], dtype=np.int32))
        sym = g.create_group("symmetry")
        sym.create_dataset("ntran", data=np.int32(ntran))
        sym.create_dataset("cell_symmetry", data=np.int32(0))
        mt = np.tile(np.eye(3, dtype=np.int32)[None], (48, 1, 1))
        mt[:ntran] = mtrx
        sym.create_dataset("mtrx", data=mt)
        tn = np.zeros((48, 3), dtype=np.float64)
        tn[:ntran] = tnp
        sym.create_dataset("tnp", data=tn)
        cr = g.create_group("crystal")
        cr.create_dataset("celvol", data=np.float64(1000.0))
        cr.create_dataset("recvol", data=np.float64(0.248))
        cr.create_dataset("alat", data=np.float64(10.0))
        cr.create_dataset("blat", data=np.float64(0.628))
        cr.create_dataset("nat", data=np.int32(2))
        cr.create_dataset("avec", data=np.eye(3, dtype=np.float64) * 10.0)
        cr.create_dataset("bvec", data=np.eye(3, dtype=np.float64) * 0.628)
        cr.create_dataset("adot", data=np.eye(3, dtype=np.float64) * 100.0)
        cr.create_dataset("bdot", data=np.eye(3, dtype=np.float64) * 0.394)
        cr.create_dataset("atyp", data=np.array([1, 1], dtype=np.int32))
        cr.create_dataset("apos", data=np.array(
            [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64))
        wf = f.create_group("wfns")
        wf.create_dataset("gvecs", data=np.tile(gvecs, (nrk, 1)))
        packed = np.stack([coeffs.real, coeffs.imag], axis=-1)
        wf.create_dataset("coeffs", data=packed)
    return path


def _kramers_deck(tmp_path, *, magnetic: bool, ntran: int = 2,
                  kpoint=(0.0, 0.0, 0.0)) -> str:
    """One-k 2-spinor deck; occupied manifold is closed under Theta or
    fully spin-polarised.

    Kramers case: bands come in pairs ``(ψ, Θψ)`` with
    ``Θψ(G) = iσ_y conj(ψ(−G))``, i.e. ``(a, b) → (conj(b(−G)),
    −conj(a(−G)))``. The occupied subspace is closed under Θ.

    Magnetic case: every occupied band is pure spin-up, so its Θ image is
    orthogonal to the occupied subspace.
    """
    rng = np.random.default_rng(0x5EED)
    gvecs = _neg_closed_gvecs()
    ngk = int(gvecs.shape[0])
    neg = {tuple(-g): i for i, g in enumerate(gvecs)}
    neg_idx = np.array([neg[tuple(g)] for g in gvecs], dtype=np.int64)

    def _rand() -> np.ndarray:
        """Random G-even spinor used to make the synthetic inversion valid."""
        c = rng.normal(size=(2, ngk)) + 1j * rng.normal(size=(2, ngk))
        return 0.5 * (c + c[:, neg_idx])

    nocc, nb = 4, 6

    def _orthogonalise(candidate: np.ndarray,
                       rows: list[np.ndarray]) -> np.ndarray:
        out = np.array(candidate, dtype=np.complex128, copy=True)
        for row in rows:
            out -= np.vdot(row, out) * row
        norm = np.linalg.norm(out)
        if norm < 1.0e-10:
            raise AssertionError("synthetic spinor basis lost rank")
        return out / norm

    rows: list[np.ndarray] = []
    if magnetic:
        for n in range(nb):
            c = _rand()
            c[1] = 0.0                    # all bands are pure spin-up
            rows.append(_orthogonalise(c, rows))
    else:
        for n in range(0, nb, 2):
            psi = _orthogonalise(_rand(), rows)
            theta = np.empty_like(psi)
            theta[0] = np.conj(psi[1, neg_idx])
            theta[1] = -np.conj(psi[0, neg_idx])
            theta = _orthogonalise(theta, rows + [psi])
            rows.extend((psi, theta))
    coeffs = np.stack(rows)
    flat = coeffs.reshape(nb, -1)
    assert np.max(np.abs(flat.conj() @ flat.T - np.eye(nb))) < 2.0e-14

    mtrx = np.tile(np.eye(3, dtype=np.int32)[None], (ntran, 1, 1))
    if ntran >= 2:
        mtrx[1] = -np.eye(3, dtype=np.int32)          # inversion
    kpoint = np.asarray(kpoint, dtype=np.float64)
    off_gamma = bool(np.max(np.abs(kpoint)) > 1.0e-12)
    name = "WFN_magnetic" if magnetic else "WFN_kramers"
    name += "_spatial.h5" if off_gamma else ".h5"
    return _write_wfn(
        str(tmp_path / name), coeffs=coeffs, gvecs=gvecs,
        kpoints=kpoint[None], kgrid=((2, 1, 1) if off_gamma else (1, 1, 1)),
        shift=((0.5, 0.0, 0.0) if off_gamma else (0.0, 0.0, 0.0)),
        nocc=nocc, mtrx=mtrx, tnp=np.zeros((ntran, 3)))


# ----------------------------------------------------------------------
# EXECUTABLE AUDIT OF THE TRS ALGEBRA
#
# ----------------------------------------------------------------------
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_ISY = np.array([[0, 1], [-1, 0]], dtype=complex)      # = i·σ_y


def test_isigma_y_matches_symmetry_maps_and_is_kramers():
    """Theta = i sigma_y K has Theta squared equal to -1."""
    from symmetry_maps.maps import _I_SIGMA_Y

    assert np.allclose(_ISY, 1j * _SY)
    assert np.allclose(np.asarray(_I_SIGMA_Y, dtype=complex), _ISY)
    # Θ²ψ = iσ_y conj(iσ_y conj(ψ)) = iσ_y conj(iσ_y) ψ = −ψ
    assert np.allclose(_ISY @ np.conj(_ISY), -np.eye(2))


def test_time_reversal_flips_pauli_expectation_values():
    for s in (_SX, _SY, _SZ):
        assert np.allclose(_SY @ s @ _SY, -np.conj(s))

    rng = np.random.default_rng(11)
    psi = rng.normal(size=2) + 1j * rng.normal(size=2)
    phi = _ISY @ np.conj(psi)                              # Θψ
    for s in (_SX, _SY, _SZ):
        m_psi = np.vdot(psi, s @ psi)
        m_phi = np.vdot(phi, s @ phi)
        assert abs(m_psi.imag) < 1e-14                     # m is real
        assert m_phi == pytest.approx(-m_psi, abs=1e-13)


def test_occupied_density_residual_is_gauge_invariant():
    """Band phases and rotations inside equally occupied blocks are inert."""
    rng = np.random.default_rng(19)
    u1, _ = np.linalg.qr(
        rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    u2, _ = np.linalg.qr(
        rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    overlap = np.zeros((4, 4), dtype=np.complex128)
    overlap[:2, :2] = u1
    overlap[2:, 2:] = u2
    occupations = np.asarray([1.0, 1.0, 0.2, 0.2])
    residual, min_singular = _occupation_operator_residual(
        overlap, occupations, occupations)
    assert residual < 5.0e-8
    assert min_singular == pytest.approx(1.0, abs=5.0e-15)

    # Mixing differently occupied sectors changes the one-particle density.
    mixed = np.eye(4, dtype=np.complex128)
    angle = 0.2
    mixed[[1, 2], [1, 2]] = np.cos(angle)
    mixed[1, 2] = np.sin(angle)
    mixed[2, 1] = -np.sin(angle)
    residual, _ = _occupation_operator_residual(
        mixed, occupations, occupations)
    assert residual > 0.1


def test_theta_overlap_handles_reciprocal_representatives_and_band_gauge():
    rng = np.random.default_rng(23)
    nb, ng = 3, 5
    flat = (rng.normal(size=(2 * ng, nb))
            + 1j * rng.normal(size=(2 * ng, nb)))
    q, _ = np.linalg.qr(flat)
    source = q.T.reshape(nb, 2, ng)
    source_g = np.stack((np.arange(ng) - 2,
                         np.zeros(ng, dtype=int),
                         np.zeros(ng, dtype=int)), axis=1)
    # target k=3/4 is the wrapped representative of -1/4, hence K=(1,0,0).
    target_g = -source_g - np.asarray([1, 0, 0])
    theta = np.empty_like(source)
    theta[:, 0] = np.conj(source[:, 1])
    theta[:, 1] = -np.conj(source[:, 0])
    gauge, _ = np.linalg.qr(
        rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb)))
    target = (gauge @ theta.reshape(nb, -1)).reshape(theta.shape)
    overlap = _theta_overlap(
        source, source_g, np.asarray([0.25, 0.0, 0.0]),
        target, target_g, np.asarray([0.75, 0.0, 0.0]))
    residual, min_singular = _occupation_operator_residual(
        overlap, np.ones(nb), np.ones(nb))
    assert residual < 5.0e-8
    assert min_singular == pytest.approx(1.0, abs=5.0e-15)


def test_evidence_planner_never_needs_an_antiunitary_row():
    inversion = -np.eye(3, dtype=np.int64)
    identity = np.eye(3, dtype=np.int64)
    kpoints = np.asarray([
        [0.25, 0.0, 0.0],
        [0.00, 0.0, 0.0],
        [0.50, 0.0, 0.0],
    ])
    evidence = _plan_two_component_evidence(
        kpoints, np.stack((identity, inversion)), np.ones(3), max_k=0)
    quarter = next(item for item in evidence if item.source == 0)
    assert quarter.kind == "spatial-pair"
    assert quarter.spatial_op == 1
    assert all(item.spatial_op is None or item.spatial_op < 2
               for item in evidence)

    raw_closed = np.asarray([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]])
    evidence = _plan_two_component_evidence(
        raw_closed, identity[None], np.ones(2), max_k=0)
    assert evidence == [density_check._TRSEvidence("raw-pair", 0, 1)]


def test_independent_failure_outranks_larger_spatial_residual():
    evidence = [
        density_check._TRSEvidence("spatial-pair", 0, 1, 1),
        density_check._TRSEvidence("trim", 2, 2),
    ]
    assert _classify_trs_evidence(
        evidence, [0.30, 0.20], tol_trs=0.10,
    ) == (True, "trim-falsified", True, "trim", 0.20)


# ----------------------------------------------------------------------
# Synthetic gates
# ----------------------------------------------------------------------
def test_kramers_manifold_at_one_trim_is_inconclusive_globally(tmp_path):
    path = _kramers_deck(tmp_path, magnetic=False)
    loader = WfnLoader(path)
    rep = loader.density_symmetry
    assert rep is not None, "check must be ON by default"
    assert rep.trs_basis == "trim-only"
    assert rep.trs_holds is False, rep.summary()
    # Kramers closure is exact; only overlap round-off survives.
    assert rep.m_rel < 1e-12, rep.summary()
    assert rep.trs_coverage == pytest.approx(1.0)
    assert loader.trs_holds is False
    assert rep.conclusive is False
    assert rep.charge == pytest.approx(rep.charge_expected, rel=1e-10)


@pytest.mark.parametrize("magnetic", [False, True])
def test_spatial_only_ibz_unfold_is_real_trs_evidence(tmp_path, magnetic):
    """Inversion reaches -k without using an antiunitary unfold row."""
    path = _kramers_deck(
        tmp_path, magnetic=magnetic, kpoint=(0.25, 0.0, 0.0))
    warning = pytest.warns(RuntimeWarning) if magnetic else nullcontext()
    with warning:
        loader = WfnLoader(path)
    rep = loader.trs_reference
    assert dict(rep.evidence_counts) == {
        "raw-pair": 0, "spatial-pair": 1, "trim": 0}
    expected_basis = (
        "spatial-conditional-failure" if magnetic else "spatial-conditional")
    assert rep.trs_basis == expected_basis
    if magnetic:
        assert rep.trs_holds is False
        assert rep.subspace_residual == pytest.approx(1.0, rel=1.0e-9)
    else:
        assert rep.trs_holds is True, rep.summary()
        assert rep.subspace_residual < 5.0e-8


def test_fractional_diagnostic_uses_complete_weighted_band_support(tmp_path):
    path = _kramers_deck(tmp_path, magnetic=False)
    occ = np.asarray([1.0, 1.0, 0.7, 0.7, 0.05, 0.05])
    with h5py.File(path, "r+") as h5:
        h5["mf_header/kpoints/occ"][0, 0, :] = occ
    loader = WfnLoader(path)
    rep = loader.density_symmetry
    assert loader.physical_density_band_stop == occ.size
    assert rep.charge_expected == pytest.approx(float(np.sum(occ)))
    assert rep.charge == pytest.approx(rep.charge_expected, rel=1e-10)
    assert rep.subspace_residual < 1.0e-12, rep.messages


def test_spin_polarised_manifold_is_caught(tmp_path):
    path = _kramers_deck(tmp_path, magnetic=True)
    with pytest.warns(RuntimeWarning):
        loader = WfnLoader(path)
    rep = loader.density_symmetry
    assert rep.trs_basis == "trim-falsified"
    assert rep.trs_holds is False, rep.summary()
    # The occupied space is orthogonal to its time-reversed image.
    assert rep.m_rel == pytest.approx(1.0, rel=1e-9)
    assert loader.trs_holds is False
    # TRS must be the ONLY thing that fails: the spatial op and the
    # normalisation are deliberately sound, so a failure there would
    # mean the check is firing for the wrong reason.
    assert rep.spatial_ops_ok, rep.messages
    assert rep.invariants_ok, rep.summary()


def test_symmaps_gate_refuses_trs_rows_for_a_magnetic_deck(tmp_path):
    """The point of the whole exercise: a broken-TRS verdict must reach
    ``SymMaps`` and remove the time-reversal rows from play."""
    path = _kramers_deck(tmp_path, magnetic=True)
    with pytest.warns(RuntimeWarning):
        loader = WfnLoader(path)
    with pytest.warns(RuntimeWarning):
        sym = loader._ensure_sym()
    assert sym.trs_allowed is False
    ntran = int(sym.sym_matrices.shape[0])
    # The table keeps its 2*ntran length (unfold_psi hard-requires it)...
    assert sym.sym_mats_k.shape[0] == 2 * ntran
    # ...but only the spatial half is eligible for selection.
    assert sym._sym_mats_k_search.shape[0] == ntran
    assert int(np.max(sym.sym_idx_k)) < ntran
    assert int(np.max(sym.sym_idx_q)) < ntran


def test_check_off_value_is_retired_instead_of_asserting_trs(
        tmp_path, monkeypatch):
    path = _kramers_deck(tmp_path, magnetic=True)
    monkeypatch.setenv("LORRAX_TRS_CHECK", "0")
    with pytest.raises(ValueError, match="retired_LORRAX_TRS_CHECK_off"):
        trs_check_mode()
    with pytest.raises(ValueError, match="retired_LORRAX_TRS_CHECK_off"):
        WfnLoader(path)


def test_strict_mode_raises_on_a_broken_deck(tmp_path, monkeypatch):
    path = _kramers_deck(tmp_path, magnetic=True)
    monkeypatch.setenv("LORRAX_TRS_CHECK", "strict")
    assert trs_check_mode() == "strict"
    with pytest.raises(RuntimeError, match="two-component TRS check FAILED"):
        WfnLoader(path)


def test_strict_mode_rechecks_policy_on_cached_report(tmp_path, monkeypatch):
    path = _kramers_deck(tmp_path, magnetic=True)
    monkeypatch.setenv("LORRAX_TRS_CHECK", "on")
    with pytest.warns(RuntimeWarning):
        WfnLoader(path)
    monkeypatch.setenv("LORRAX_TRS_CHECK", "strict")
    with pytest.raises(RuntimeError, match="two-component TRS check FAILED"):
        WfnLoader(path)


def test_strict_mode_refuses_a_trim_only_pass(tmp_path, monkeypatch):
    path = _kramers_deck(tmp_path, magnetic=False)
    monkeypatch.setenv("LORRAX_TRS_CHECK", "strict")
    with pytest.raises(RuntimeError, match="TRS check INCONCLUSIVE"):
        WfnLoader(path)


# ----------------------------------------------------------------------
# Real-fixture gates
# ----------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(_FIXTURE), reason="fixture absent")
def test_fixture_passes_two_component_reference_check():
    """cohsex_debug: nonmagnetic two-component MoS2 reference."""
    loader = WfnLoader(_FIXTURE)
    rep = check_density_symmetries(loader, max_k=0)
    print("\n" + rep.summary())
    for msg in rep.messages:
        print("   ", msg)
    assert rep.trs_holds, rep.summary()
    assert rep.trs_basis in {
        "raw-subspace", "spatial-conditional", "trim-only"}
    assert rep.trs_coverage > 0.0
    assert rep.spatial_ops_ok, rep.messages
    assert rep.invariants_ok, rep.summary()
    assert rep.charge == pytest.approx(rep.charge_expected, rel=1e-8)
    # Every op must actually have been exercised: the mesh is Γ-centred,
    # so Γ is in every little group.
    assert not rep.spatial_untested, rep.spatial_untested


@pytest.mark.skipif(not os.path.exists(_FIXTURE), reason="fixture absent")
def test_raw_read_matches_the_loader_ibz_path():
    """``_raw_ibz_psi_k`` must stay bit-identical to the loader's
    documented raw-IBZ contract, so the deliberate independence of the
    check's reader can never become a silent divergence."""
    # The bounded raw reader must agree with the loader's IBZ contract.
    from symmetry_maps.density_symmetry_check import _raw_ibz_psi_k

    loader = WfnLoader(_FIXTURE)
    nb = 6
    ref = np.asarray(loader.load(bands=(0, nb), k="ibz", sharding=None))
    for ik in range(int(loader.nkpts)):
        got = _raw_ibz_psi_k(loader, ik, nb)
        ngk = int(loader.ngk[ik])
        assert np.array_equal(got, ref[ik, :nb, :, :ngk])


@pytest.mark.skipif(not os.path.exists(_FIXTURE), reason="fixture absent")
def test_subsampling_does_not_change_the_verdict():
    loader = WfnLoader(_FIXTURE)
    full = check_density_symmetries(loader, max_k=0)
    sub = check_density_symmetries(loader, max_k=2)
    assert sub.trs_holds == full.trs_holds
    assert sub.subsampled and sub.n_k_used <= full.n_k_used
    # Compatibility charge fields remain internally consistent.
    assert sub.charge == pytest.approx(sub.charge_expected, rel=1e-8)
