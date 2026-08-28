"""Metallic Σ window discrimination: weights and the Fermi straddle.

The hazard this file pins (ARCHITECTURE W2.b): with fractional occupations
the branch A-supports acquire a negative-E_A shell of width ~ few×degauss.
Two distinct failure modes exist for a naive metallization, and each test
here fails under exactly one of them:

1. Keeping ``occ > 0.5`` masks assigns each Fermi-shell state WHOLLY to one
   branch — Σ_c silently wrong by O(f·pole term) at exactly the
   Fermi-surface states (``test_mask_semantics_err_at_fermi_shell``
   measures the size of that error; it is the magnitude certificate, not
   the gate).
2. Widening the supports without splitting the plan makes the deep-pole
   slab rectangle reach zero: pre-change the planner REFUSES there
   (recorded below), and the fix routes the straddle through the existing
   crossing core by deepening the shallow/deep pole edge by the branches'
   negative excursion (``test_weighted_split_plan_matches_fractional_reference``
   is the gate — it FAILED on the pre-change tree, see the certificate note).

Failing-first certificate (recorded 2026-08-15, pre-change tree
integ/metal-mpa-qsgw-2026-08-15 @ merge base):
  - weighted-support branches -> ``ValueError: MPA sign-definite window
    reaches or crosses zero (lower denominator -0.01 Ry)`` from
    ``sigma_windows._rectangles`` (the b_slab straddle), and
    ``TypeError: unexpected keyword argument 'band_weight'`` for the
    weighted construction;
  - mask-semantics arm evaluated cleanly and disagreed with the exact
    fractional reference by max rel err ~ 2e-1 (recorded by
    ``test_mask_semantics_err_at_fermi_shell``'s floor assert).
"""

import numpy as np
import jax
import jax.numpy as jnp

from gw.mpa import sigma_windows as SW
from gw.ppm_tau_kernel import build_shared_w_tau
from gw.ppm_windows import _SigmaBranch


# One k-point, four bands around mu=0, MP-like fractional occupations.
_E_MINUS_MU = np.asarray([[-0.60, -0.10, 0.08, 0.55]])
_F = np.asarray([[1.0, 0.9, 0.3, 0.0]])

# Three scalar poles: shallow, barely-deep (the straddle trigger: with
# omega_max=0.5, eta=0.05, edge=1.5 the OLD edge is 0.575 and this pole's
# slab rectangle bottoms at -0.10 + 0.59 - 0.50 = -0.01), genuinely deep.
_OMEGA_P = np.asarray([0.30 - 0.04j, 0.59 - 0.03j, 1.80 - 0.05j])
_B_P = np.asarray([0.8 + 0.1j, 1.1 - 0.2j, 0.7 + 0.3j])

_OMEGA_GRID = np.asarray([0.0, 0.25, 0.5])
_ETA = 0.05
_EDGE = 1.5
_TOL = dict(target_error=1.0e-6, max_rank=96,
            crossing_max_nodes=SW.CROSSING_NODE_FLOOR)


def _pole_fields():
    Omega = jnp.asarray(_OMEGA_P.reshape(-1, 1, 1, 1))
    B = jnp.asarray(_B_P.reshape(-1, 1, 1, 1))
    return Omega, B


def _pos_branches(*, weighted):
    """The two +omega branches, either weighted-support or occ>0.5 masks."""
    idx = np.arange(_OMEGA_GRID.size)
    E = jnp.asarray(_E_MINUS_MU)
    f = _F
    if weighted:
        val_mask = jnp.asarray(f != 0.0)
        cond_mask = jnp.asarray(f != 1.0)
        val_w = jnp.asarray(np.where(f != 0.0, f, 0.0))
        cond_w = jnp.asarray(np.where(f != 1.0, 1.0 - f, 0.0))
        return [
            _SigmaBranch("pos_cond", E, cond_mask, "cond", False,
                         _OMEGA_GRID, idx, band_weight=cond_w),
            _SigmaBranch("pos_val", -E, val_mask, "val", False,
                         _OMEGA_GRID, idx, band_weight=val_w),
        ]
    occupied = jnp.asarray(f > 0.5)
    return [
        _SigmaBranch("pos_cond", E, ~occupied, "cond", False,
                     _OMEGA_GRID, idx),
        _SigmaBranch("pos_val", -E, occupied, "val", False,
                     _OMEGA_GRID, idx),
    ]


def _plan(branches):
    Omega, B = _pole_fields()
    summaries = SW.summarize_sigma_poles(
        Omega, B, branches,
        regularization_width_ry=_ETA, edge_factor=_EDGE)
    return SW.build_shared_sigma_windows(
        summaries, branches,
        regularization_width_ry=_ETA, edge_factor=_EDGE, **_TOL)


def _evaluate(plan_rows, omega):
    """Scalar oracle: assemble Σ(omega) from the production W(t) builder.

    Follows the reconstruction in test_mpa_sigma_windows exactly, with the
    branch band weight applied in the G factor the way the tau kernel's
    ``build_G_tau(band_weight=...)`` seam applies it.
    """
    Omega, B = _pole_fields()
    total = 0.0 + 0.0j
    for row in plan_rows:
        win = row.window
        w_idx = np.where(np.isclose(row.omega_abs, abs(omega)))[0]
        if not w_idx.size:
            continue
        t = np.asarray(jax.device_get(win.nodes.t))
        alpha = np.asarray(jax.device_get(win.nodes.alpha))
        build_all = jax.jit(jax.vmap(
            lambda tt: build_shared_w_tau(
                B, Omega, jnp.asarray(row.pole_indices),
                jnp.asarray(row.bounds), jnp.asarray(row.phase_real),
                win.E_ref_B, tt)))
        W_t = np.asarray(jax.device_get(build_all(win.nodes.t))).reshape(
            t.size, -1)[:, 0]
        E_A = np.asarray(jax.device_get(row.E_A)).reshape(-1)
        mask = np.asarray(win.mask_A).reshape(-1)
        weight = getattr(row, "band_weight", None)
        w = (np.ones_like(E_A) if weight is None
             else np.asarray(jax.device_get(weight)).reshape(-1))
        G_t = np.array([
            np.sum(w[mask] * np.exp(-1j * (E_A[mask] - win.E_ref_A) * tt))
            for tt in t])
        coeff = (win.prefactor * alpha
                 * np.exp(-1j * (win.E_ref_A + win.E_ref_B
                                 - win.omega_sign * abs(omega)) * t))
        total += np.sum(coeff * G_t * W_t)
    return total


def _exact_fractional(omega):
    """Exact weighted Σ(omega) with the plan's own +omega sign convention."""
    poles = _OMEGA_P - 1j * _ETA
    E = _E_MINUS_MU.reshape(-1)
    f = _F.reshape(-1)
    val = -sum(
        f[n] * np.sum(_B_P / (omega + (-E[n]) + poles))
        for n in range(E.size) if f[n] != 0.0)
    cond = -sum(
        (1.0 - f[n]) * np.sum(_B_P / (omega - E[n] - poles))
        for n in range(E.size) if f[n] != 1.0)
    return val + cond


def test_mask_semantics_err_at_fermi_shell():
    """Magnitude certificate: occ>0.5 masks are measurably wrong for metals.

    This arm runs on the PRE-change planner too (it uses only bool masks) —
    it is the number that makes the weighted path worth building, and it
    discriminates any regression that quietly reverts weights to masks.
    """
    plan, _ = _plan(_pos_branches(weighted=False))
    errs = []
    for omega in _OMEGA_GRID:
        got = _evaluate(plan, omega)
        want = _exact_fractional(omega)
        errs.append(abs(got - want) / abs(want))
    assert max(errs) > 5.0e-2, (
        "the mask arm now agrees with the fractional reference — either the "
        "fixture lost its Fermi shell or masks silently became weights: "
        f"{errs}")


def test_weighted_split_plan_matches_fractional_reference():
    """THE GATE: weighted supports + deepened crossing edge meet tolerance.

    Pre-change this failed twice over (TypeError on band_weight; ValueError
    'reaches or crosses zero' once supports widened) — see module docstring.
    """
    plan, geometry = _plan(_pos_branches(weighted=True))
    for omega in _OMEGA_GRID:
        got = _evaluate(plan, omega)
        want = _exact_fractional(omega)
        assert abs(got - want) / abs(want) < 5.0e-4, (
            f"weighted plan disagrees at omega={omega}: {got} vs {want}")


def test_wrong_side_sliver_in_sign_definite_branch_matches_reference():
    """Wrong-side fractional states in a STATICALLY-SIGN-DEFINITE branch.

    The gate above cannot see this cell: its shallowest pole (Re 0.30)
    sits above the sliver depth (|E_A| = 0.08), so x = omega + E_A + a
    never crosses zero and the plain Laplace window stays valid.  The
    first real metallic arm (sodium 48b, claim 0194 chain) had poles
    down to Re 0.0149 against a shell of ~0.05 Ry — x through zero in
    the +omega orientation, which the crossing family cannot represent
    (its x = E_A + a − omega has the opposite omega sign; wholesale
    reclassification was measured wrong by a factor ~−8).  This cell
    pins the fix: the sliver keeps the sign-definite family's own
    (omega_sign=−1, prefactor=−neg) and takes the damped positive rule
    (sd_core) for shallow poles plus a Laplace slab (sd_slab) for deep
    ones.

    Failing-first certificate: pre-fix this exact construction refuses
    with "MPA sign-definite window reaches or crosses zero (lower
    denominator −0.03 Ry)" — the np6 arm's refusal in miniature.  The
    unweighted negative control below keeps that refusal reachable.
    """
    shallow_poles = np.asarray([0.05 - 0.02j, 0.59 - 0.03j, 1.80 - 0.05j])
    Omega = jnp.asarray(shallow_poles.reshape(-1, 1, 1, 1))
    B = jnp.asarray(_B_P.reshape(-1, 1, 1, 1))
    branches = _pos_branches(weighted=True)
    summaries = SW.summarize_sigma_poles(
        Omega, B, branches,
        regularization_width_ry=_ETA, edge_factor=_EDGE)
    plan, geometry = SW.build_shared_sigma_windows(
        summaries, branches,
        regularization_width_ry=_ETA, edge_factor=_EDGE, **_TOL)

    # Plan shape: the wrong-side val state (E_A = −0.08) owns an sd_core
    # window in the +omega orientation, and the sd_slab serves the deep pole.
    sd_core = [r for r in plan if r.window.name == "sd_core"]
    assert sd_core, "no sd_core window for the wrong-side sliver"
    for r in sd_core:
        assert r.window.omega_sign == -1, r.window
        assert bool(np.asarray(r.window.mask_A).reshape(-1)[2])
        assert not np.asarray(r.window.mask_A).reshape(-1)[[0, 1, 3]].any()
    assert [r for r in plan if r.window.name == "sd_slab"], (
        "no sd_slab window for sliver x deep poles")

    # Values against the exact fractional reference, same pole set.
    def exact(omega):
        poles = shallow_poles - 1j * _ETA
        E = _E_MINUS_MU.reshape(-1)
        f = _F.reshape(-1)
        val = -sum(f[n] * np.sum(B_P / (omega + (-E[n]) + poles))
                   for n in range(E.size) if f[n] != 0.0
                   for B_P in [_B_P])
        cond = -sum((1.0 - f[n]) * np.sum(B_P / (omega - E[n] - poles))
                    for n in range(E.size) if f[n] != 1.0
                    for B_P in [_B_P])
        return val + cond

    def evaluate(plan_rows, omega):
        total = 0.0 + 0.0j
        for row in plan_rows:
            win = row.window
            w_idx = np.where(np.isclose(row.omega_abs, abs(omega)))[0]
            if not w_idx.size:
                continue
            t = np.asarray(jax.device_get(win.nodes.t))
            alpha = np.asarray(jax.device_get(win.nodes.alpha))
            build_all = jax.jit(jax.vmap(
                lambda tt: build_shared_w_tau(
                    B, Omega, jnp.asarray(row.pole_indices),
                    jnp.asarray(row.bounds), jnp.asarray(row.phase_real),
                    win.E_ref_B, tt)))
            W_t = np.asarray(jax.device_get(
                build_all(win.nodes.t))).reshape(t.size, -1)[:, 0]
            E_A = np.asarray(jax.device_get(row.E_A)).reshape(-1)
            mask = np.asarray(win.mask_A).reshape(-1)
            weight = getattr(row, "band_weight", None)
            w = (np.ones_like(E_A) if weight is None
                 else np.asarray(jax.device_get(weight)).reshape(-1))
            G_t = np.array([
                np.sum(w[mask] * np.exp(-1j * (E_A[mask] - win.E_ref_A) * tt))
                for tt in t])
            coeff = (win.prefactor * alpha
                     * np.exp(-1j * (win.E_ref_A + win.E_ref_B
                                     - win.omega_sign * abs(omega)) * t))
            total += np.sum(coeff * G_t * W_t)
        return total

    for omega in _OMEGA_GRID:
        got = evaluate(plan, omega)
        want = exact(omega)
        assert abs(got - want) / abs(want) < 5.0e-4, (
            f"sliver plan disagrees at omega={omega}: {got} vs {want}")

    # Negative control aimed at the real refusal: an UNWEIGHTED branch whose
    # mask includes the wrong-side state must still refuse by name — the
    # sd split is licensed by the weight semantics only.
    import pytest
    E = jnp.asarray(_E_MINUS_MU)
    idx = np.arange(_OMEGA_GRID.size)
    bad_val_mask = jnp.asarray(_F != 0.0)
    bad = [_SigmaBranch("pos_val", -E, bad_val_mask, "val", False,
                        _OMEGA_GRID, idx)]
    bad_sum = SW.summarize_sigma_poles(
        Omega, B, bad, regularization_width_ry=_ETA, edge_factor=_EDGE)
    with pytest.raises(ValueError, match="reaches or crosses zero"):
        SW.build_shared_sigma_windows(
            bad_sum, bad,
            regularization_width_ry=_ETA, edge_factor=_EDGE, **_TOL)


def test_straddle_routes_through_crossing_core():
    """Plan shape: the negative-E_A shell lands in a core (crossing) window,
    the barely-deep pole is served shallow, and every sign-definite slab
    rectangle stays strictly positive (no refusal was raised to get here)."""
    plan, geometry = _plan(_pos_branches(weighted=True))
    old_edge = geometry["omega_max_ry"] + _EDGE * geometry["eta_ry"]
    assert geometry["crossing_edge_ry"] >= old_edge + 0.10 - 1e-12, geometry
    core_rows = [r for r in plan if r.window.name == "core"
                 and r.window.mask_A.reshape(-1)[1]]
    assert core_rows, "the E_A=-0.10 shell state left the crossing core"
    # The 0.59 pole (index 1) must be served by the (shallow) core rule.
    assert any(1 in r.pole_indices.tolist() for r in core_rows), (
        "the barely-deep pole was not rerouted to the crossing core")


def test_insulating_step_occupations_change_nothing():
    """Step-function weights (exact 0/1) reproduce the mask plan node-for-node."""
    idx = np.arange(_OMEGA_GRID.size)
    E = jnp.asarray(_E_MINUS_MU)
    f_step = np.asarray([[1.0, 1.0, 0.0, 0.0]])
    occupied = jnp.asarray(f_step > 0.5)
    mask_branches = [
        _SigmaBranch("pos_cond", E, ~occupied, "cond", False,
                     _OMEGA_GRID, idx),
        _SigmaBranch("pos_val", -E, occupied, "val", False,
                     _OMEGA_GRID, idx),
    ]
    w_branches = [
        _SigmaBranch("pos_cond", E, jnp.asarray(f_step != 1.0), "cond",
                     False, _OMEGA_GRID, idx,
                     band_weight=jnp.asarray(1.0 - f_step)),
        _SigmaBranch("pos_val", -E, jnp.asarray(f_step != 0.0), "val",
                     False, _OMEGA_GRID, idx,
                     band_weight=jnp.asarray(f_step)),
    ]
    plan_m, geo_m = _plan(mask_branches)
    plan_w, geo_w = _plan(w_branches)
    assert geo_m["crossing_edge_ry"] == geo_w["crossing_edge_ry"]
    assert len(plan_m) == len(plan_w)
    for rm, rw in zip(plan_m, plan_w):
        assert rm.window.name == rw.window.name
        np.testing.assert_array_equal(np.asarray(rm.window.mask_A),
                                      np.asarray(rw.window.mask_A))
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(rm.window.nodes.t)),
            np.asarray(jax.device_get(rw.window.nodes.t)))
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(rm.window.nodes.alpha)),
            np.asarray(jax.device_get(rw.window.nodes.alpha)))


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

from gw.cohsex_sigma import _make_cohsex_kernels, _resolve_Gij, _spin_capacity
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
    psi_yn = _c(1, 1, _SX_NMU, _SX_NB)
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
        psi_xn=jnp.asarray(psi_xn), psi_xr=jnp.asarray(psi_xr),
        psi_yr=jnp.asarray(psi_yr), psi_yn=jnp.asarray(psi_yn),
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


def test_invalid_static_shared_spatial_matches_legacy_dense_cohsex():
    """Mode-3 static fallback keeps the old SX+COH algebra at P=1.

    This is deliberately a cross-implementation comparison: the reference is
    the former decomposed static-COHSEX path, while the production result uses
    the canonical fused spatial kernel and reduce-scatter projector.
    """
    wfns, meta, *_unused, Wc0_q = _sx_fixture()
    mesh = _mesh_1x1()
    invalid = np.asarray([[[True, False, True],
                           [False, True, False],
                           [True, False, True]]])
    W_static = jnp.where(jnp.asarray(invalid), jnp.asarray(Wc0_q), 0.0j)
    sigma_sx_k, sigma_coh_k, _ = _make_cohsex_kernels(
        mesh, meta.kgrid, int(meta.nk_tot), _spin_capacity(meta))
    Gij = _resolve_Gij(None, meta, mesh, None)
    with mesh:
        legacy = np.asarray(jax.device_get(
            sigma_sx_k(wfns, Gij, W_static)
            + sigma_coh_k(wfns, W_static, jnp.zeros_like(W_static))))

    got = _compute_invalid_static_sigma(
        wfns, jnp.asarray(Wc0_q), jnp.asarray(invalid), meta, mesh)
    scale = max(float(np.max(np.abs(legacy))), 1.0)
    np.testing.assert_allclose(got, legacy, rtol=2e-12,
                               atol=2e-12 * scale)


def test_invalid_static_brackets_match_legacy_dense_cohsex():
    """The memory-bounded bracket diagnostic preserves each disjoint COH part."""
    wfns, meta, *_unused, Wc0_q = _sx_fixture()
    mesh = _mesh_1x1()
    invalid = np.asarray([[[True, True, False],
                           [True, False, True],
                           [False, True, True]]])
    brackets = ((0, 1), (1, _SX_NB))
    W_static = jnp.where(jnp.asarray(invalid), jnp.asarray(Wc0_q), 0.0j)
    _, sigma_coh_k, _ = _make_cohsex_kernels(
        mesh, meta.kgrid, int(meta.nk_tot), _spin_capacity(meta))
    legacy = []
    with mesh:
        for lo, hi in brackets:
            legacy.append(np.asarray(jax.device_get(sigma_coh_k(
                wfns, W_static, jnp.zeros_like(W_static),
                ri_bands=(lo, hi)))))
    legacy = np.stack(legacy)

    got = _invalid_static_coh_by_bracket(
        wfns, jnp.asarray(Wc0_q), jnp.asarray(invalid), meta, mesh, brackets)
    scale = max(float(np.max(np.abs(legacy))), 1.0)
    np.testing.assert_allclose(got, legacy, rtol=2e-12,
                               atol=2e-12 * scale)


def test_sigma_x_takes_diag_f_and_differs_from_the_integer_projector():
    wfns, meta, psi_xn, psi_xr, psi_yr, psi_yn, V_q = _sx_fixture()
    mesh = _mesh_1x1()
    sigma_sx_k, _, _ = _make_cohsex_kernels(mesh, meta.kgrid,
                                            int(meta.nk_tot),
                                            _spin_capacity(meta))

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
    sigma_sx_k, _, _ = _make_cohsex_kernels(mesh, meta.kgrid,
                                            int(meta.nk_tot),
                                            _spin_capacity(meta))

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
               cohsex_sigma.compute_v_h_sigma_x,
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
    dispatch_src = inspect.getsource(sigma_dispatch.compute_sigma_xc)
    assert dispatch_src.count("occupation_state=occupation_state") >= 3, (
        "compute_sigma_xc must forward occupation_state to "
        "compute_cohsex_sigma, compute_v_h_sigma_x and the PPM pipeline")
