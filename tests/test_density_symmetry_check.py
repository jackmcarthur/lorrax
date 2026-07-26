"""Gates for the load-time density symmetry measurement.

``common.density_symmetry_check`` replaces a decade of flag-based
inference ("ntran<=1 means nosym means ...") with a measurement taken
from the wavefunctions themselves.  These tests pin the three things
that measurement has to get right:

1. it must not cry wolf on a real, nonmagnetic, symmetric deck;
2. it must catch a manifestly TRS-broken occupied manifold; and
3. when it does, ``SymMaps`` must structurally refuse to select a
   time-reversal row.

The synthetic decks are built here rather than captured so the physics
under test (Kramers pairing vs an unpaired spin-polarised manifold) is
visible in the test source.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from common.density_symmetry_check import (
    check_density_symmetries,
    trs_check_mode,
)
from common.symmetry_maps import SymMaps
from file_io.wfn_loader import WfnLoader


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
               tnp: np.ndarray) -> str:
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
        kp.create_dataset("shift", data=np.zeros(3, dtype=np.float64))
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


def _kramers_deck(tmp_path, *, magnetic: bool, ntran: int = 2) -> str:
    """Γ-only 2-spinor deck; occupied manifold either Kramers-paired or
    fully spin-polarised.

    Kramers case: bands come in pairs ``(ψ, Θψ)`` with
    ``Θψ(G) = iσ_y conj(ψ(−G))``, i.e. ``(a, b) → (conj(b(−G)),
    −conj(a(−G)))``.  Every Pauli component of the pair's density then
    cancels identically, for ANY random ψ — the check must see m ≡ 0.

    Magnetic case: every occupied band is pure spin-up, so ``m_z ≡ ρ``
    and ``‖m‖∞/‖ρ‖∞ = 1`` — as unambiguous a TRS violation as exists.
    """
    rng = np.random.default_rng(0x5EED)
    gvecs = _neg_closed_gvecs()
    ngk = int(gvecs.shape[0])
    neg = {tuple(-g): i for i, g in enumerate(gvecs)}
    neg_idx = np.array([neg[tuple(g)] for g in gvecs], dtype=np.int64)

    def _rand() -> np.ndarray:
        """Random ψ(G) with ``c(G) = c(−G)`` ⇒ ψ(r) even ⇒ ρ(r) even, so
        the deck's declared inversion really IS a symmetry of it and the
        spatial arm of the check must pass.  Only the TRS arm is then
        under test."""
        c = rng.normal(size=(2, ngk)) + 1j * rng.normal(size=(2, ngk))
        return 0.5 * (c + c[:, neg_idx])

    nocc, nb = 4, 6
    coeffs = np.zeros((nb, 2, ngk), dtype=np.complex128)
    if magnetic:
        for n in range(nb):
            c = _rand()
            if n < nocc:
                c[1] = 0.0                # occupied manifold is pure spin-up
            coeffs[n] = c
    else:
        for n in range(0, nb, 2):
            psi = _rand()
            coeffs[n] = psi
            coeffs[n + 1, 0] = np.conj(psi[1, neg_idx])
            coeffs[n + 1, 1] = -np.conj(psi[0, neg_idx])
    # Normalise each band so ∫ρ d³r = f_spin · nocc exactly (f_spin = 1
    # for nspinor=2), making the invariants arm meaningful too.
    coeffs /= np.sqrt(np.sum(np.abs(coeffs) ** 2,
                             axis=(1, 2)))[:, None, None]

    mtrx = np.tile(np.eye(3, dtype=np.int32)[None], (ntran, 1, 1))
    if ntran >= 2:
        mtrx[1] = -np.eye(3, dtype=np.int32)          # inversion
    name = "WFN_magnetic.h5" if magnetic else "WFN_kramers.h5"
    return _write_wfn(
        str(tmp_path / name), coeffs=coeffs, gvecs=gvecs,
        kpoints=np.zeros((1, 3)), kgrid=(1, 1, 1), nocc=nocc,
        mtrx=mtrx, tnp=np.zeros((ntran, 3)))


# ----------------------------------------------------------------------
# EXECUTABLE AUDIT OF THE TRS ALGEBRA
#
# The verdict this module produces rests on three lines of Pauli algebra
# spelled out as (T1)-(T3) in ``common/density_symmetry_check.py``.  They
# are pinned here as tests so a future reader (or agent) can check the
# math without trusting a comment.
# ----------------------------------------------------------------------
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_ISY = np.array([[0, 1], [-1, 0]], dtype=complex)      # = i·σ_y


def test_T1_isigma_y_matches_symmetry_maps_and_is_kramers():
    """(T1): Θ = iσ_y K with Θ² = −1, and LORRAX's stored constant is iσ_y."""
    from common.symmetry_maps import _I_SIGMA_Y

    assert np.allclose(_ISY, 1j * _SY)
    assert np.allclose(np.asarray(_I_SIGMA_Y, dtype=complex), _ISY)
    # Θ²ψ = iσ_y conj(iσ_y conj(ψ)) = iσ_y conj(iσ_y) ψ = −ψ
    assert np.allclose(_ISY @ np.conj(_ISY), -np.eye(2))


def test_T3_time_reversal_flips_the_magnetization_density():
    """(T3): the identity σ_y σ_i σ_y = −σ_i*, and its consequence
    m_{Θψ} = −m_ψ, which is the whole basis of the TRS verdict."""
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


def test_polarisation_identities_used_to_get_m_from_the_quadrature():
    """The four-call reconstruction in ``_spin_resolved_density``:
    m_x = D(a+b) − ρ and m_y = ρ − D(a+ib), with D linear in |·|²."""
    rng = np.random.default_rng(12)
    a = rng.normal(size=8) + 1j * rng.normal(size=8)
    b = rng.normal(size=8) + 1j * rng.normal(size=8)
    rho = np.abs(a) ** 2 + np.abs(b) ** 2
    assert np.allclose(np.abs(a + b) ** 2 - rho, 2 * np.real(np.conj(a) * b))
    assert np.allclose(rho - np.abs(a + 1j * b) ** 2,
                       2 * np.imag(np.conj(a) * b))


# ----------------------------------------------------------------------
# Synthetic gates
# ----------------------------------------------------------------------
def test_kramers_manifold_measures_trs_holds(tmp_path):
    path = _kramers_deck(tmp_path, magnetic=False)
    loader = WfnLoader(path)
    rep = loader.density_symmetry
    assert rep is not None, "check must be ON by default"
    assert rep.trs_basis == "measured"
    assert rep.trs_holds is True, rep.summary()
    # Kramers cancellation is exact in exact arithmetic; only fp64 FFT
    # round-off survives, so this should be many orders below the gate.
    assert rep.m_rel < 1e-12, rep.summary()
    assert rep.trs_coverage == pytest.approx(1.0)
    assert loader.trs_holds is True
    # Nothing else may fire: the deck's inversion is a real symmetry of
    # these (G-even) coefficients and the bands are normalised.
    assert rep.ok, rep.messages
    assert rep.charge == pytest.approx(rep.charge_expected, rel=1e-10)


def test_spin_polarised_manifold_is_caught(tmp_path):
    path = _kramers_deck(tmp_path, magnetic=True)
    with pytest.warns(RuntimeWarning):
        loader = WfnLoader(path)
    rep = loader.density_symmetry
    assert rep.trs_basis == "measured"
    assert rep.trs_holds is False, rep.summary()
    # every occupied band is spin-up => m_z is the whole density
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


def test_check_is_a_no_op_when_disabled(tmp_path, monkeypatch):
    path = _kramers_deck(tmp_path, magnetic=True)
    monkeypatch.setenv("LORRAX_TRS_CHECK", "0")
    assert trs_check_mode() == "off"
    loader = WfnLoader(path)
    assert loader.density_symmetry is None
    assert loader.trs_holds is True          # permissive, historical behaviour
    sym = loader._ensure_sym()
    assert sym.trs_allowed is True


def test_strict_mode_raises_on_a_broken_deck(tmp_path, monkeypatch):
    path = _kramers_deck(tmp_path, magnetic=True)
    monkeypatch.setenv("LORRAX_TRS_CHECK", "strict")
    assert trs_check_mode() == "strict"
    with pytest.raises(RuntimeError, match="density symmetry check FAILED"):
        WfnLoader(path)


# ----------------------------------------------------------------------
# Real-fixture gates
# ----------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(_FIXTURE), reason="fixture absent")
def test_fixture_passes_every_arm():
    """cohsex_debug: ntran=12, nonmagnetic MoS2, spatially-reduced IBZ."""
    loader = WfnLoader(_FIXTURE)
    rep = check_density_symmetries(loader, max_k=0)
    print("\n" + rep.summary())
    for msg in rep.messages:
        print("   ", msg)
    assert rep.trs_holds, rep.summary()
    assert rep.trs_basis == "measured"
    assert rep.trs_coverage > 0.5
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
    from common.density_symmetry_check import _raw_ibz_psi_k

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
    # the sampled sub-density still normalises exactly
    assert sub.charge == pytest.approx(sub.charge_expected, rel=1e-8)


@pytest.mark.skipif(not os.path.exists(_FIXTURE), reason="fixture absent")
def test_a_corrupted_symmetry_op_is_caught(tmp_path):
    """Rewrite the fixture's symmetry block with a bogus op and confirm
    the density flags exactly that op."""
    import shutil
    bad = str(tmp_path / "WFN_badsym.h5")
    shutil.copy(_FIXTURE, bad)
    os.chmod(bad, 0o644)
    with h5py.File(bad, "r+") as f:
        mt = f["mf_header/symmetry/mtrx"][()]
        # A 90-degree rotation about z is NOT a symmetry of MoS2 (D3h).
        mt[1] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=mt.dtype)
        f["mf_header/symmetry/mtrx"][...] = mt
        del f["mf_header/symmetry/ntran"]
        f["mf_header/symmetry"].create_dataset("ntran", data=np.int32(2))
    loader = WfnLoader(bad)
    rep = check_density_symmetries(loader, max_k=0)
    print("\n" + rep.summary())
    assert not rep.spatial_ops_ok, rep.summary()
    assert 1 in rep.spatial_failed, rep.spatial_residual
    assert np.isfinite(rep.spatial_residual[0])
    assert rep.spatial_residual[0] < rep.tol_spatial   # identity still fine
