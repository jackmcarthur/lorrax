"""Fractional occupations in the static Σ and the head/body occupation rule.

W2.c pins the COHSEX occupation projector ``build_Gij`` (step occupations
reproduce the integer projector bit for bit, MP overshoot keeps its sign,
the metallic window coverage guard refuses) and the static Σ_x built from
``diag(f)``; W2.d pins the one-state rule between the head fit and the body
Σ (``assert_head_body_occupation_match``).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# W2.c: the cohsex occupation projector, and W2.d: the one-state rule.
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from gw.cohsex_sigma import build_Gij
from gw.mpa.sigma import assert_head_body_occupation_match


def _mesh_1x1():
    import jax as _jax
    from jax.sharding import Mesh
    return Mesh(np.asarray(_jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _state(f, mu=0.0, n_electrons=None):
    f = np.asarray(f, dtype=np.float64)
    return SimpleNamespace(
        f_kn=f, mu_ry=mu,
        n_electrons=(float(f.sum() / f.shape[0])
                     if n_electrons is None else float(n_electrons)),
        occ_hash="testhash")


def test_gij_step_occupations_bit_exact_vs_integer_projector():
    # nspin/nspinor declared since 2026-08-28: the fixed-N check is
    # capacity-weighted and refuses a meta with no spin structure; these
    # cells model the spinor decks (1 electron per band) they always did.
    meta = SimpleNamespace(nk_tot=2, nb_sigma=4, nelec=2, nspin=1, nspinor=2)
    mesh = _mesh_1x1()
    integer = np.asarray(jax.device_get(build_Gij(meta, mesh)))
    stepped = np.asarray(jax.device_get(build_Gij(
        meta, mesh, occupation_state=_state(np.tile([1.0, 1.0, 0.0, 0.0],
                                                    (2, 1))))))
    np.testing.assert_array_equal(integer, stepped)


def test_gij_keeps_mp_overshoot_sign_unclipped():
    """A negative MP occupation must enter diag(f) with its sign — the
    linear weight IS the state's Σ_x/SX/Hartree contribution, so clipping
    here flips a real (small, negative) exchange contribution to zero."""
    meta = SimpleNamespace(nk_tot=1, nb_sigma=3, nelec=2, nspin=1, nspinor=2)
    f = np.asarray([[1.02, 1.0, -0.02]])
    got = np.asarray(jax.device_get(build_Gij(
        meta, _mesh_1x1(), occupation_state=_state(f))))
    assert got[0, 2, 2] == -0.02 + 0.0j
    assert got[0, 0, 0] == 1.02 + 0.0j


def test_gij_metallic_window_coverage_guard_refuses():
    meta = SimpleNamespace(nk_tot=1, nb_sigma=2, nelec=2, nspin=1, nspinor=2)
    f = np.asarray([[1.0, 0.7, 0.3]])   # 0.3 electrons live outside nb_sigma
    try:
        build_Gij(meta, _mesh_1x1(),
                  occupation_state=_state(f, n_electrons=2.0))
    except ValueError as err:
        assert "missing weight" in str(err)
    else:
        raise AssertionError("the metallic coverage guard did not refuse")


def test_head_body_occupation_match_refuses_mismatch_and_unstamped():
    state = _state(np.asarray([[1.0, 0.5, 0.0]]), mu=0.1)
    assert_head_body_occupation_match(
        {"occ_hash": "testhash", "mu_ry": 0.1}, state)      # match: silent
    assert_head_body_occupation_match({}, None)             # insulating: skip
    for attrs in ({}, {"occ_hash": "other", "mu_ry": 0.1},
                  {"occ_hash": "testhash", "mu_ry": 0.2}):
        try:
            assert_head_body_occupation_match(attrs, state)
        except ValueError:
            pass
        else:
            raise AssertionError(f"no refusal for head attrs {attrs}")


def test_head_body_occupation_match_accepts_only_a_reproduced_legacy_hash():
    state = _state(np.asarray([[1.0, 0.5, 0.0]]), mu=0.1)
    attrs = {"occ_hash": "legacy-zero-pad", "mu_ry": 0.1}
    assert assert_head_body_occupation_match(
        attrs, state, compatible_occ_hashes={"legacy-zero-pad"}
    ) == "legacy_zero_pad"
    with pytest.raises(ValueError, match="different occupation states"):
        assert_head_body_occupation_match(
            attrs, state, compatible_occ_hashes={"not-this-hash"})


def test_mpa_head_occupation_preflight_precedes_the_body_sweep():
    import inspect

    from gw.sigma_dispatch import _compute_mpa_sigma

    mpa = inspect.getsource(_compute_mpa_sigma)
    assert mpa.index("head = _bundle_reader.read_head_fit_collective") < mpa.index(
        "body = compute_sigma_c_mpa_omega_grid")


# ---------------------------------------------------------------------------
# W2.c (continued): Σ_x itself, not just the projector.
#
# The projector cells above prove ``build_Gij`` builds diag(f).  They do NOT
# prove Σ_x consumes it — and for three commits it did not: every call site
# built the integer projector and the ``occupation_state`` parameter was
# unreachable from the driver.  These two cells are the discriminating
# direction: run the PRODUCTION static-exchange kernel both ways and require
# the fractional answer to (a) differ from the integer one and (b) equal an
# independent dense band-sum reference.
#
# Scope: nk = 1 (kgrid (1,1,1)), one spin channel, 3 bands, 3 centroids.  At
# one k-point the flat-k FFT pair in ``_convolve`` is exactly the identity, so
# this cell tests the occupation weighting and the projection, not the q-sum.
# ---------------------------------------------------------------------------

from gw.cohsex_sigma import _face_kwargs, _make_cohsex_kernels, _resolve_Gij
from gw.ppm_sigma import (
    _compute_invalid_static_sigma,
    _invalid_static_coh_by_bracket,
)
from gw.wavefunction_bundle import BandSlices, Wavefunctions


_SX_NB = 3        # bands (sigma window == full window)
_SX_NMU = 3       # centroids
_SX_NELEC = 2     # integer projector fills bands 0 and 1


def _sx_fixture():
    """One-k, one-spin ψ bundle + Hermitian V_q, all complex128."""
    rng = np.random.default_rng(20260815)

    def _c(*shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(np.complex128)

    # psi_xn (nk, s, muX, n) and psi_xr (nk, n, s, muX) are the SAME data;
    # likewise psi_yn / psi_yr.  Build one of each pair and transpose, so the
    # fixture cannot accidentally test a bundle no loader could produce.
    psi_xn = _c(1, 1, _SX_NMU, _SX_NB)
    psi_yn = psi_xn.copy()
    psi_xr = np.transpose(psi_xn, (0, 3, 1, 2)).copy()
    psi_yr = np.transpose(psi_yn, (0, 3, 1, 2)).copy()

    V = _c(_SX_NMU, _SX_NMU)
    V = 0.5 * (V + V.conj().T)
    V_q = V[None, :, :].copy()

    slices = BandSlices.from_band_edges(0, 0, _SX_NELEC, _SX_NB, _SX_NB)
    enk = np.linspace(-0.4, 0.6, _SX_NB)[None, :]
    occ = np.zeros((1, _SX_NB))
    occ[0, :_SX_NELEC] = 1.0
    wfns = Wavefunctions(
        psi_mun=jnp.asarray(psi_xn), psi_nmu=jnp.asarray(psi_yr),
        enk=jnp.asarray(enk), occ=jnp.asarray(occ), slices=slices)
    meta = SimpleNamespace(nk_tot=1, nb_sigma=_SX_NB, nelec=_SX_NELEC,
                           kgrid=(1, 1, 1), nspin=1, nspinor=2)
    return wfns, meta, psi_xn, psi_xr, psi_yr, psi_yn, V_q


def _sigma_x_dense_reference(f, psi_xn, psi_xr, psi_yr, psi_yn, V_q):
    """Σ_x[m,n] = -Σ_i f_i Σ_{s,x,t,y} ψ*_xr ψ_xn V ψ*_yr ψ_yn, by loops.

    Deliberately NOT an einsum and deliberately not reusing ``build_G`` or
    ``project``: an independent contraction order is the whole value of a
    reference.  The overall -1/√Nk is the kernel's ``_inv_sqrt_nk`` at
    Nk = 1.
    """
    out = np.zeros((_SX_NB, _SX_NB), dtype=np.complex128)
    for m in range(_SX_NB):
        for n in range(_SX_NB):
            acc = 0.0 + 0.0j
            for i in range(_SX_NB):
                if f[i] == 0.0:
                    continue
                for x in range(_SX_NMU):
                    for y in range(_SX_NMU):
                        acc += (np.conj(psi_xr[0, m, 0, x])
                                * psi_xn[0, 0, x, i]
                                * V_q[0, x, y]
                                * np.conj(psi_yr[0, i, 0, y])
                                * psi_yn[0, 0, y, n]
                                * f[i])
            out[m, n] = -acc
    return out


def _sx_parent_bundle(wfns, mesh):
    """The one-row fixture carries raw parents with the typed identity action."""
    from functools import partial
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.wavefunction_bundle import ParentGreenCarrier
    from jax.sharding import NamedSharding, PartitionSpec as P
    from symmetry_maps import spinor_rotation_for_sym_row

    ops = np.eye(3, dtype=np.int32)[None]
    sym = SimpleNamespace(
        sym_matrices=ops, translations=np.zeros((1, 3)),
        sym_mats_k=np.concatenate((ops, -ops)),
        irr_idx_k=np.asarray([0]), sym_idx_k=np.asarray([0]),
        kirr_fullids=np.asarray([0]), unfolded_kpts=np.zeros((1, 3)),
        nk_red=1, nk_tot=1,
        spinor_action=partial(spinor_rotation_for_sym_row,
                              np.ones((1, 1, 1)), n_tran=1))
    plan = build_centroid_k_unfold_plan(
        sym, np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        (3, 1, 1), mesh, nspinor=1)
    carrier = ParentGreenCarrier(
        psi_mun=jax.device_put(
            wfns.psi_mun, NamedSharding(mesh, P(None, None, "x", "y"))),
        psi_nmu=jax.device_put(
            wfns.psi_nmu, NamedSharding(mesh, P(None, "x", None, "y"))),
        enk=wfns.enk, occ=wfns.occ, plan=plan)
    return Wavefunctions(enk=wfns.enk, occ=wfns.occ, slices=wfns.slices,
                         green_parent=carrier, layout="face")


def test_invalid_static_shared_spatial_matches_dense_cohsex():
    """The invalid static SX+COH tail equals independent occupied and identity band sums."""
    wfns, meta, xn, xr, yr, yn, Wc0_q = _sx_fixture()
    mesh = _mesh_1x1()
    invalid = np.asarray([[[True, False, True],
                           [False, True, False],
                           [True, False, True]]])
    W_static = np.where(invalid, Wc0_q, 0.0j)
    reference = _sigma_x_dense_reference(
        np.asarray([1.0, 1.0, 0.0]), xn, xr, yr, yn, W_static)
    reference -= 0.5 * _sigma_x_dense_reference(
        np.ones(3), xn, xr, yr, yn, W_static)
    got = _compute_invalid_static_sigma(
        _sx_parent_bundle(wfns, mesh), jnp.asarray(Wc0_q),
        jnp.asarray(invalid), meta, mesh)
    scale = max(float(np.max(np.abs(reference))), 1.0)
    np.testing.assert_allclose(got, reference[None], rtol=2e-12,
                               atol=2e-12 * scale)


def test_invalid_static_brackets_match_dense_cohsex():
    """Each disjoint static COH bracket equals its literal intermediate-state sum."""
    wfns, meta, xn, xr, yr, yn, Wc0_q = _sx_fixture()
    mesh = _mesh_1x1()
    invalid = np.asarray([[[True, True, False],
                           [True, False, True],
                           [False, True, True]]])
    brackets = ((0, 1), (1, 3))
    W_static = np.where(invalid, Wc0_q, 0.0j)
    reference = np.stack([
        -0.5 * _sigma_x_dense_reference(
            ((np.arange(3) >= lo) & (np.arange(3) < hi)).astype(float),
            xn, xr, yr, yn, W_static)[None]
        for lo, hi in brackets])
    got = _invalid_static_coh_by_bracket(
        _sx_parent_bundle(wfns, mesh), jnp.asarray(Wc0_q),
        jnp.asarray(invalid), meta, mesh, brackets)
    scale = max(float(np.max(np.abs(reference))), 1.0)
    np.testing.assert_allclose(got, reference, rtol=2e-12,
                               atol=2e-12 * scale)


def test_sigma_x_takes_diag_f_and_differs_from_the_integer_projector():
    wfns, meta, psi_xn, psi_xr, psi_yr, psi_yn, V_q = _sx_fixture()
    mesh = _mesh_1x1()
    wfns = _sx_parent_bundle(wfns, mesh)
    sigma_sx_k, _ = _make_cohsex_kernels(
        mesh, meta.kgrid, int(meta.nk_tot), **_face_kwargs(wfns))

    f_int = np.asarray([1.0, 1.0, 0.0])
    f_frac = np.asarray([1.0, 0.625, 0.375])   # same 2 electrons, smeared
    assert f_frac.sum() == 2.0                 # exact in binary; fixed-N

    Gij_int = _resolve_Gij(None, meta, mesh, None)
    Gij_frac = _resolve_Gij(None, meta, mesh,
                            _state(f_frac[None, :], n_electrons=2.0))

    with mesh:
        sig_int = np.asarray(jax.device_get(
            sigma_sx_k(wfns, Gij_int, jnp.asarray(V_q))))[0]
        sig_frac = np.asarray(jax.device_get(
            sigma_sx_k(wfns, Gij_frac, jnp.asarray(V_q))))[0]

    ref_int = _sigma_x_dense_reference(
        f_int, psi_xn, psi_xr, psi_yr, psi_yn, V_q)
    ref_frac = _sigma_x_dense_reference(
        f_frac, psi_xn, psi_xr, psi_yr, psi_yn, V_q)

    scale = float(np.max(np.abs(ref_int)))
    np.testing.assert_allclose(sig_int, ref_int, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(sig_frac, ref_frac, rtol=0, atol=1e-12 * scale)

    # THE DISCRIMINATING DIRECTION.  Before the threading fix both calls
    # returned ``sig_int``; the fractional weights on the Fermi-shell bands
    # are exactly what was being dropped.
    delta = float(np.max(np.abs(sig_frac - sig_int)))
    assert delta > 1.0e-2 * scale, (
        f"Sigma_x did not respond to diag(f): max|delta| = {delta:.3e} "
        f"against scale {scale:.3e}")


def test_sigma_x_step_occupations_reproduce_the_integer_projector_bitwise():
    """The insulating no-delta claim, at Σ_x rather than at the projector."""
    wfns, meta, *_unused, V_q = _sx_fixture()
    mesh = _mesh_1x1()
    wfns = _sx_parent_bundle(wfns, mesh)
    sigma_sx_k, _ = _make_cohsex_kernels(
        mesh, meta.kgrid, int(meta.nk_tot), **_face_kwargs(wfns))

    Gij_int = _resolve_Gij(None, meta, mesh, None)
    Gij_step = _resolve_Gij(
        None, meta, mesh, _state(np.asarray([[1.0, 1.0, 0.0]]),
                                 n_electrons=2.0))
    with mesh:
        a = np.asarray(jax.device_get(
            sigma_sx_k(wfns, Gij_int, jnp.asarray(V_q))))
        b = np.asarray(jax.device_get(
            sigma_sx_k(wfns, Gij_step, jnp.asarray(V_q))))
    np.testing.assert_array_equal(a, b)


def test_static_sigma_refuses_an_explicit_Gij_beside_an_occupation_state():
    """Two occupation models in one Σ is a refusal, not a silent drop."""
    _wfns, meta, *_rest = _sx_fixture()
    mesh = _mesh_1x1()
    explicit = _resolve_Gij(None, meta, mesh, None)
    try:
        _resolve_Gij(explicit, meta, mesh,
                     _state(np.asarray([[1.0, 0.5, 0.5]]), n_electrons=2.0))
    except ValueError as err:
        assert "occupation_state" in str(err)
    else:
        raise AssertionError("_resolve_Gij accepted both a Gij and a state")


def test_the_occupation_state_actually_reaches_all_three_build_Gij_sites():
    """The gap this file's Σ_x cells exist to close was NOT in ``build_Gij``.

    ``build_Gij(occupation_state=...)`` was correct and reachable for three
    commits while all three call sites — ``cohsex_sigma:301``, ``:395``,
    ``ppm_sigma:763`` — called it bare, so the parameter was dead from the
    driver's point of view and no value-level cell could see it.  Pin the
    CHAIN, not just the leaf: every link takes the kwarg, and the dispatcher
    forwards the state it already carries into each of the three entries.
    """
    import inspect

    from gw import cohsex_sigma, ppm_pipeline, ppm_sigma, sigma_dispatch

    for fn in (cohsex_sigma.compute_cohsex_sigma,
               cohsex_sigma.compute_sigma_x,
               cohsex_sigma.build_Gij,
               ppm_sigma.compute_sigma_c_ppm_omega_grid,
               ppm_sigma._compute_invalid_static_sigma,
               ppm_pipeline.compute_ppm_sigma_pipeline):
        params = inspect.signature(fn).parameters
        assert "occupation_state" in params, f"{fn.__qualname__} drops it"
        assert params["occupation_state"].default is None, (
            f"{fn.__qualname__}: None must stay the insulating default")

    # No bare ``build_Gij(meta, mesh_xy)`` may survive in src/.
    for module in (cohsex_sigma, ppm_sigma):
        source = inspect.getsource(module)
        assert "build_Gij(meta, mesh_xy)" not in source, (
            f"{module.__name__} still builds the integer projector "
            "unconditionally")

    # And the dispatcher forwards it into all three static entries.
    dispatch_src = "\n".join(inspect.getsource(fn) for fn in (
        sigma_dispatch._static_sigma_channels,
        sigma_dispatch._packed_dynamic_sigma_channels,
        sigma_dispatch._compute_ppm_sigma))
    assert dispatch_src.count("occupation_state=occupation_state") >= 3, (
        "compute_sigma_xc must forward occupation_state to "
        "compute_cohsex_sigma, compute_sigma_x and the PPM pipeline")
