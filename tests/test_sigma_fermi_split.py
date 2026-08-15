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
    meta = SimpleNamespace(nk_tot=2, nb_sigma=4, nelec=2)
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
    meta = SimpleNamespace(nk_tot=1, nb_sigma=3, nelec=2)
    f = np.asarray([[1.02, 1.0, -0.02]])
    got = np.asarray(jax.device_get(build_Gij(
        meta, _mesh_1x1(), occupation_state=_state(f))))
    assert got[0, 2, 2] == -0.02 + 0.0j
    assert got[0, 0, 0] == 1.02 + 0.0j


def test_gij_metallic_window_coverage_guard_refuses():
    meta = SimpleNamespace(nk_tot=1, nb_sigma=2, nelec=2)
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
