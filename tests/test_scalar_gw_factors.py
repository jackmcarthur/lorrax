"""nspinor=1 (scalar-relativistic) spin-degeneracy factors in the GW stack.

THE CONVENTION UNDER TEST (verified in BerkeleyGW source): the factor 2
for nspin=nspinor=1 enters ONLY the polarizability/epsilon
(epsilon_main.f90:660-665, fact = 4/(N_k·Ω·nspin·nspinor)) and the
DENSITY; the Σ band sums carry occupancy 1 per band with NO spin factor
(sigma_main.f90:2340-2348).  In LORRAX that means exactly two live sites:

  * χ₀'s prefactor  — ``gw.w_isdf._w_solve_pref_scalar`` (already correct;
    pinned here so nobody "fixes" it by moving spin into the ``-2.0``
    time-ordering factor at w_isdf.py:925/970/1672, which is NOT spin);
  * the ISDF Hartree ρ — ``gw.cohsex_sigma``'s hartree kernel (fixed
    2026-08-28: at nspinor=1 it built HALF the electron density, i.e. a
    ~500 eV V_H silently wrong by 2×, on the only production V_H path
    when hartree_source=isdf).

Every factor introduced is exactly 1.0 on the nspinor≥2 path (bispinor
meta.nspinor=4 included) — the spinor arms below are bit-identity pins,
not tolerance checks.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                    # noqa: E402
from jax.sharding import Mesh                              # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _mesh():
    devices = np.asarray(jax.devices())[:1].reshape(1, 1)
    return Mesh(devices, ("x", "y"))


# ---------------------------------------------------------------------------
# (1) χ₀ Dyson prefactor: 2/(√N_k · nspin · nspinor).
# ---------------------------------------------------------------------------

def test_w_solve_pref_scalar_pins_all_three_spin_structures():
    """The nspinor=2 value is pinned FIRST and exactly: it is the
    anti-regression anchor for the whole convention.  Someone who believes
    the ``-2.0`` at w_isdf.py:925/970/1672 is a spin factor and "fixes" it
    (or moves spin degeneracy out of this prefactor) changes this value
    and fails here before any physics run does."""
    from gw.w_isdf import _w_solve_pref_scalar
    nk = 8
    # spinor (the production MoS2/Si decks) — unchanged, exact.
    assert _w_solve_pref_scalar(NS(nk_tot=nk, nspin=1, nspinor=2, nspinor_wfnfile=2)) \
        == 2.0 / (float(nk) ** 0.5 * 1.0 * 2.0)
    # scalar: the BGW nspin=nspinor=1 factor 2 lives HERE (and in ρ),
    # nowhere else.
    assert _w_solve_pref_scalar(NS(nk_tot=nk, nspin=1, nspinor=1, nspinor_wfnfile=1)) \
        == 2.0 / (float(nk) ** 0.5)
    # bispinor (meta.nspinor=4): since the 2026-08-29 four-current refactor
    # the pref reads the FILE nspinor (a bispinor run rides a 2-component
    # file; the lift's own normalization lives in the lift) — bispinor
    # therefore shares the nspinor=2 value.
    assert _w_solve_pref_scalar(NS(nk_tot=nk, nspin=1, nspinor=4, nspinor_wfnfile=2)) \
        == 2.0 / (float(nk) ** 0.5 * 1.0 * 2.0)


# ---------------------------------------------------------------------------
# (2) The ISDF Hartree kernel — through the REAL factory and kernel.
#
# The flat-k FFT factories require the FFI .so, which the hartree kernel
# never touches; they are stubbed with REFUSING functions so this cell can
# run FFI-free while proving, not assuming, that the kernel under test
# stays off the FFT path (a stub that got called would fail the test, not
# fake a pass).
# ---------------------------------------------------------------------------

def _stub_fft_factories(monkeypatch):
    from common import fft_helpers

    def _refusing_factory(*_a, **_k):
        def _refuse(_x):
            raise AssertionError(
                "flat-k FFT invoked — the hartree kernel must not reach "
                "the FFT path; only sigma_sx/sigma_coh may, and this test "
                "does not call those.")
        return _refuse

    monkeypatch.setattr(fft_helpers, "make_flat_k_fftn", _refusing_factory)
    monkeypatch.setattr(fft_helpers, "make_flat_k_ifftn", _refusing_factory)


def _fresh_kernel_cache(monkeypatch):
    from gw import cohsex_sigma
    monkeypatch.setattr(cohsex_sigma, "_cohsex_kernel_cache", {})


def _bundle(nk, ns, n_mu, nocc, seed=7):
    """A (nb = ns·n_mu)-band bundle whose ψ rows are a unitary over the
    combined (s, μ) index — so Σ_m |ψ_m(s,μ)|² = 1 at every (s, μ) and
    Σ_{s,μ} |ψ_m(s,μ)|² = 1 for every band, which makes the traces below
    exact by algebra rather than by tolerance."""
    from gw.wavefunction_bundle import BandSlices, Wavefunctions
    nb = ns * n_mu
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
    q, _ = np.linalg.qr(a)
    psi = np.broadcast_to(q.reshape(nb, ns, n_mu), (nk, nb, ns, n_mu))
    psi_r = jnp.asarray(psi)                       # (nk, n, s, μ) — xr/yr
    psi_n = jnp.asarray(psi.transpose(0, 2, 3, 1))  # (nk, s, μ, n) — xn/yn
    slices = BandSlices.from_band_edges(0, nocc, nocc, nb, nb)
    wfns = Wavefunctions(
        psi_xn=psi_n, psi_xr=psi_r, psi_yr=psi_r, psi_yn=psi_n,
        enk=jnp.zeros((nk, nb)), occ=jnp.zeros((nk, nb)), slices=slices)
    return wfns, nb


def _hartree_trace(mesh, kgrid, f_spin, wfns, Gij, n_mu):
    from gw.cohsex_sigma import _make_cohsex_kernels
    nk = int(np.prod(kgrid))
    V_q = jnp.broadcast_to(
        jnp.eye(n_mu, dtype=jnp.complex128), (nk, n_mu, n_mu))
    _, _, hartree_k = _make_cohsex_kernels(mesh, kgrid, nk, f_spin=f_spin)
    with mesh:
        sig_h = hartree_k(wfns, Gij, V_q)
    return np.asarray(sig_h)


def test_isdf_hartree_rho_carries_the_scalar_spin_degeneracy(monkeypatch):
    """Electron-count gate through the REAL hartree kernel (the
    ``expected_electrons`` idea of kin_ion_io's exact-V_H path, restated
    for the ISDF quadrature).  With V_q[0] = 1 the kernel returns
    ⟨m|ρ/nk|n⟩, and with unitary ψ over the combined (s,μ) index

        tr Σ_H(k) = ns · Σ_μ ρ(μ)/nk = ns · f_spin · n_occ

    (the leading ns is basis completeness — each μ is projected once per
    spinor component — not physics).  At nspinor=1 the electron count is
    f_spin·n_occ = 2·n_occ; the pre-2026-08-28 kernel produced n_occ,
    which is the red twin asserted below via the f_spin=1.0 build."""
    from gw.cohsex_sigma import build_Gij
    _stub_fft_factories(monkeypatch)
    _fresh_kernel_cache(monkeypatch)
    mesh = _mesh()
    kgrid, n_mu, nocc = (2, 1, 1), 4, 2
    nk = int(np.prod(kgrid))

    # ── scalar (ns=1): ρ must carry 2 electrons per occupied band ──────
    wfns, nb = _bundle(nk, 1, n_mu, nocc)
    Gij = build_Gij(NS(nk_tot=nk, nb_sigma=nb, nelec=nocc), mesh)
    sig = _hartree_trace(mesh, kgrid, 2.0, wfns, Gij, n_mu)
    tr = np.trace(sig, axis1=1, axis2=2).real
    np.testing.assert_allclose(tr, 2.0 * nocc, rtol=0, atol=1e-12)

    # RED TWIN — the pre-fix kernel (no spin factor, i.e. f_spin=1.0)
    # counts HALF the electrons on the same scalar bundle.  ×2.0 is exact
    # in IEEE, so the equality is bitwise: the factor is applied exactly
    # once, linearly, and nowhere else in the kernel.
    sig_unfixed = _hartree_trace(mesh, kgrid, 1.0, wfns, Gij, n_mu)
    tr_unfixed = np.trace(sig_unfixed, axis1=1, axis2=2).real
    np.testing.assert_allclose(tr_unfixed, float(nocc), rtol=0, atol=1e-12)
    assert np.array_equal(sig, 2.0 * sig_unfixed)

    # ── spinor (ns=2): factor exactly 1.0 — count is n_occ, unchanged ──
    wfns2, nb2 = _bundle(nk, 2, n_mu, 3)
    Gij2 = build_Gij(NS(nk_tot=nk, nb_sigma=nb2, nelec=3), mesh)
    sig2 = _hartree_trace(mesh, kgrid, 1.0, wfns2, Gij2, n_mu)
    tr2 = np.trace(sig2, axis1=1, axis2=2).real
    np.testing.assert_allclose(tr2, 2 * 3.0, rtol=0, atol=1e-12)  # ns·n_occ


def test_cohsex_kernel_cache_is_keyed_on_f_spin(monkeypatch):
    """A kernel cached for one spin structure must never serve the other:
    without f_spin in the key, a spinor run followed by a scalar rerun in
    the same process would reuse the f_spin=1.0 hartree closure and halve
    ρ with no other symptom.  Red twin: drop f_spin from the cache key at
    cohsex_sigma.py:~168 and the first assertion fails."""
    from gw.cohsex_sigma import _make_cohsex_kernels
    _stub_fft_factories(monkeypatch)
    _fresh_kernel_cache(monkeypatch)
    mesh = _mesh()
    k_spinor = _make_cohsex_kernels(mesh, (2, 1, 1), 2, f_spin=1.0)
    k_scalar = _make_cohsex_kernels(mesh, (2, 1, 1), 2, f_spin=2.0)
    assert k_spinor is not k_scalar and k_spinor[2] is not k_scalar[2]
    # …and the cache still caches: same key, same kernels (element
    # identity — the factory rebuilds the outer tuple on a miss).
    assert _make_cohsex_kernels(mesh, (2, 1, 1), 2, f_spin=2.0)[2] is k_scalar[2]


def test_the_static_drivers_derive_f_spin_from_meta():
    """The kernel-level cells above prove the factor works; this pins that
    every production entry actually threads it — from meta, through the
    ONE canonical helper — so the factory's required argument cannot be
    satisfied with a literal that ignores the deck's spin structure."""
    import inspect
    from gw import cohsex_sigma, sigma_x_bispinor
    for fn in (cohsex_sigma.compute_cohsex_sigma,
               cohsex_sigma.compute_v_h_sigma_x,
               sigma_x_bispinor.compute_sigma_x_bispinor):
        src = inspect.getsource(fn)
        assert "_spin_capacity(meta)" in src, fn.__qualname__
        assert "_make_cohsex_kernels(" in src, fn.__qualname__
    # …and _spin_capacity itself delegates to the ONE canonical helper.
    assert "spin_degeneracy_factor" in inspect.getsource(
        cohsex_sigma._spin_capacity)


def test_spin_capacity_refuses_an_undeclared_spin_structure():
    """The canonical helper getattr-defaults a missing attr to 1, i.e. an
    underspecified duck-typed meta would take the SCALAR factor 2
    silently — into a ~500 eV V_H.  Refuse loudly instead (red twin: the
    silent path is exactly what spin_degeneracy_factor alone would do)."""
    from gw.cohsex_sigma import _spin_capacity
    with pytest.raises(AttributeError, match="nspinor"):
        _spin_capacity(NS(nk_tot=2, nb_sigma=4, nelec=2, nspin=1))
    with pytest.raises(AttributeError, match="nspin"):
        _spin_capacity(NS(nk_tot=2, nb_sigma=4, nelec=2))
    # Green twins: the three declared structures.
    assert _spin_capacity(NS(nspin=1, nspinor=1)) == 2.0
    assert _spin_capacity(NS(nspin=1, nspinor=2)) == 1.0
    assert _spin_capacity(NS(nspin=1, nspinor=4)) == 1.0   # bispinor meta


# ---------------------------------------------------------------------------
# (3) The fixed-N window check in build_Gij is capacity-weighted.
# ---------------------------------------------------------------------------

def _occ_state(f_kn, n_electrons):
    # build_Gij duck-types gw.efermi.OccupationState; only these two
    # fields are read on the checked path.
    return NS(f_kn=np.asarray(f_kn, dtype=np.float64),
              n_electrons=float(n_electrons))


def test_fixed_n_window_check_accepts_a_scalar_deck(monkeypatch):
    """``occupation_state.n_electrons`` is capacity-weighted (2 e⁻ per
    filled band on a scalar deck, efermi.py ``state_capacity``); before
    2026-08-28 build_Gij compared it against the raw band sum and refused
    EVERY scalar deck by exactly the capacity."""
    from gw.cohsex_sigma import build_Gij
    mesh = _mesh()
    meta = NS(nk_tot=2, nb_sigma=4, nelec=2, nspin=1, nspinor=1,
              nspinor_wfnfile=1)
    f = np.array([[1.0, 1.0, 0.0, 0.0]] * 2)     # 2 filled bands per k
    Gij = build_Gij(meta, mesh, _occ_state(f, n_electrons=4.0))
    got = np.asarray(Gij)
    np.testing.assert_array_equal(
        got, np.broadcast_to(np.diag(f[0]).astype(np.complex128),
                             got.shape))


def test_fixed_n_window_check_red_twins(monkeypatch):
    from gw.cohsex_sigma import build_Gij
    mesh = _mesh()
    f = np.array([[1.0, 1.0, 0.0, 0.0]] * 2)
    # RED TWIN 1 — the band-count target (what the pre-fix code compared
    # against) must now REFUSE on a scalar deck: passing it would mean the
    # capacity factor was dropped again.
    meta1 = NS(nk_tot=2, nb_sigma=4, nelec=2, nspin=1, nspinor=1)
    with pytest.raises(ValueError, match="sigma window holds"):
        build_Gij(meta1, mesh, _occ_state(f, n_electrons=2.0))
    # RED TWIN 2 — the check is not weakened: genuinely missing weight
    # (an electron carried outside the window) still refuses at nspinor=1.
    f_missing = np.array([[1.0, 0.0, 0.0, 0.0]] * 2)
    with pytest.raises(ValueError, match="sigma window holds"):
        build_Gij(meta1, mesh, _occ_state(f_missing, n_electrons=4.0))
    # Spinor bit-identity: factor exactly 1.0 — the historical contract
    # (band sum == n_electrons) is untouched.
    meta2 = NS(nk_tot=2, nb_sigma=4, nelec=2, nspin=1, nspinor=2)
    Gij = build_Gij(meta2, mesh, _occ_state(f, n_electrons=2.0))
    assert np.asarray(Gij).shape == (2, 4, 4)


# ---------------------------------------------------------------------------
# (4) kin_ion.h5 validator: the producer's nspinor stamp is enforced.
# ---------------------------------------------------------------------------

def _write_min_kin_ion(path, nspinor=None):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        ds = f.create_dataset(
            "kin_ion", data=np.zeros((2, 3, 3), dtype=np.complex128))
        # main's 2026-08-29 representation gate fails closed on a MISSING
        # bispinor stamp; stamp it so the flow reaches the nspinor check
        # (the nspinor attr keeps its own required-if-present contract).
        ds.attrs["bispinor"] = False
        if nspinor is not None:
            ds.attrs["nspinor"] = int(nspinor)
    return str(path)


def test_kin_ion_validator_refuses_a_spin_structure_mismatch(tmp_path):
    """gw.kin_ion_io stamps the source WFN's nspinor (kin_ion_io.py:1464);
    a file built from an nspinor=2 WFN consumed by a scalar run puts a
    wrong-by-construction T+V_ion into a ~500 eV cancellation.  The
    refusal names both values."""
    from file_io.kin_ion import validate_kin_ion_against_run
    p = _write_min_kin_ion(tmp_path / "kin_ion.h5", nspinor=2)
    with pytest.raises(ValueError, match=r"nspinor=2.*nspinor=1"):
        validate_kin_ion_against_run(p, nspinor=1, expected_bispinor=False,
                                     selected_hartree_source="stored",
                                     print_fn=lambda *_: None)


def test_kin_ion_validator_green_and_legacy_twins(tmp_path):
    from file_io.kin_ion import validate_kin_ion_against_run
    # Green: matching stamp passes.
    p = _write_min_kin_ion(tmp_path / "kin_ion_match.h5", nspinor=2)
    validate_kin_ion_against_run(p, nspinor=2, expected_bispinor=False,
                                 selected_hartree_source="stored",
                                 print_fn=lambda *_: None)
    # Legacy-accept: a pre-stamp file (no attr) is NOT refused — the red
    # twin of over-refusal, same contract as the sys_dim attr.
    p2 = _write_min_kin_ion(tmp_path / "kin_ion_legacy.h5", nspinor=None)
    validate_kin_ion_against_run(p2, nspinor=1, expected_bispinor=False,
                                 selected_hartree_source="stored",
                                 print_fn=lambda *_: None)


def test_the_driver_threads_wfn_nspinor_into_the_validator():
    """A check whose parameter no production caller passes is dead code
    (the x_only/screening_method defect class).  Text-level pin on the one
    production call site, without importing the heavy driver module."""
    src = (_REPO / "src" / "gw" / "gw_jax.py").read_text()
    call = src[src.index("validate_kin_ion_against_run("):]
    call = call[:call.index("\n\t)")]        # the call's closing paren line
    assert "nspinor=int(wfn.nspinor)" in call


# ---------------------------------------------------------------------------
# (5) Meta.from_system refuses an un-halved spinor deck.
# ---------------------------------------------------------------------------

def _scalar_wfn(nelec=4):
    return NS(nelec=nelec, fft_grid=(8, 8, 8), cell_volume=1.0,
              nspin=1, nspinor=1, kgrid=(2, 2, 2))


def test_meta_refuses_nval_beyond_the_wfn_occupied_bands():
    """A user porting an nspinor=2 Si deck (nval=8) onto a scalar WFN
    (nelec=4 occupied bands, 2 electrons each) currently gets a negative
    valence edge and silent nonsense; the refusal names both numbers and
    the halving."""
    from common.meta import Meta
    with pytest.raises(ValueError, match=r"nval=8.*nelec=4.*halve"):
        Meta.from_system(_scalar_wfn(nelec=4), NS(nk_tot=8),
                         8, 4, 12, 16, False)


def test_meta_accepts_nval_at_and_below_nelec():
    """Silent-otherwise twin: the guard fires ONLY past nelec — the
    boundary nval == nelec (all occupied bands in the window) is legal and
    unchanged."""
    from common.meta import Meta
    meta = Meta.from_system(_scalar_wfn(nelec=4), NS(nk_tot=8),
                            4, 4, 12, 16, False)
    assert meta.b_id_1 == 0 and meta.nelec == 4


# ---------------------------------------------------------------------------
# (6) ζ provenance stamps and enforces the source WFN's nspinor.
# ---------------------------------------------------------------------------

def _prov(nspinor_wfnfile, monkeypatch):
    # The effective-env keys must be deterministic for the stamp algebra
    # below; these are the two _zeta_fit_provenance reads through
    # deprecated_env_record.
    monkeypatch.delenv("LORRAX_ZETA_RIDGE", raising=False)
    monkeypatch.delenv("LORRAX_ZETA_RCOND", raising=False)
    from gw.gw_init import _zeta_fit_provenance
    cfg = NS(bispinor=False,
             backend=NS(zeta_ridge=0.0, zeta_rcond=1e-8,
                        charge_zeta_solve="cholesky",
                        distributed_zeta_solve="auto",
                        transverse_zeta_solve="ridge",
                        transverse_zeta_rcond=1e-10,
                        gamma_contract_mode="auto"))
    wfn = NS(ecutwfc=25.0, ecutrho=100.0)
    meta = NS(n_rmu=16, fft_grid=(8, 8, 8),
              nspinor_wfnfile=nspinor_wfnfile)
    return _zeta_fit_provenance(
        wfn=wfn, meta=meta, cfg=cfg,
        band_range_left=(0, 8), band_range_right=(0, 16),
        logical_band_stop=16,
        zeta_cutoff=10.0, zeta_vcoul_cutoff=40.0,
        write_ibz_only=True, band_norms=None)


def test_zeta_provenance_stamps_the_file_nspinor(monkeypatch):
    assert json.loads(_prov(1, monkeypatch))["nspinor"] == 1
    assert json.loads(_prov(2, monkeypatch))["nspinor"] == 2


def _stamped_zeta_file(tmp_path, stamp, name="zeta_q.h5"):
    from file_io.isdf_header import IsdfHeader, write_isdf_header
    idx = np.stack(np.unravel_index(np.arange(16), (8, 8, 8)),
                   axis=1).astype(np.int32)
    path = str(tmp_path / name)
    write_isdf_header(path, IsdfHeader.build(
        r_mu_fft_idx=idx, fft_grid=(8, 8, 8), density="n",
        vertex_mu_L=0, zeta_is_done=True, fit_provenance=stamp))
    return path, idx


def test_zeta_reuse_enforces_nspinor_symmetrically(tmp_path, monkeypatch):
    """Read-side twin of the stamp: a ζ fit from a spinor WFN must not be
    reused by a scalar rerun that hits the same (path, size) fingerprint —
    and a stamp MISSING the key is a legacy spinor fit (scalar support did
    not exist when it was written), so an nspinor=2 rerun still reuses it
    (the red twin of forcing a multi-hour refit on every old run dir)."""
    monkeypatch.delenv("LORRAX_FORCE_REFIT", raising=False)
    from gw.gw_init import _zeta_reuse_ok
    said = []
    say = said.append

    prov_spinor = _prov(2, monkeypatch)
    prov_scalar = _prov(1, monkeypatch)

    # Identical stamps reuse (the green baseline the refusals sit on).
    p, idx = _stamped_zeta_file(tmp_path, prov_spinor)
    assert _zeta_reuse_ok(p, prov_spinor, idx, say) is True

    # Both stamped, different nspinor -> refit, naming the key.
    said.clear()
    assert _zeta_reuse_ok(p, prov_scalar, idx, say) is False
    assert any("nspinor" in s for s in said)

    # Legacy stamp (key absent): spinor rerun reuses…
    legacy = dict(json.loads(prov_spinor))
    del legacy["nspinor"]
    p2, idx2 = _stamped_zeta_file(
        tmp_path, json.dumps(legacy, sort_keys=True), name="zeta_legacy.h5")
    said.clear()
    assert _zeta_reuse_ok(p2, prov_spinor, idx2, say) is True
    # …and a scalar rerun refits, naming the key it could not verify.
    said.clear()
    assert _zeta_reuse_ok(p2, prov_scalar, idx2, say) is False
    assert any("nspinor" in s for s in said)
