"""Sharded head/wing/body Schur decomposition of W in the centroid basis.

Mirror of agent-A's bordered Sherman-Morrison reconstruction
(``sources/lorrax_A/src/psp/finite_q_head_interp.py``), recast in the
production V_FFT5D_SPEC sharding so that:

* The rank-1 head-channel subtract / rebuild are **zero-communication**
  outer products: ``g0_x*[μ] · g0_y[ν]`` is local on every device when
  ``g0_x`` is replicated on the x-axis and ``g0_y`` on the y-axis.
* The unavoidable Schur reductions (matrix-vector contracting μ_X or ν_Y;
  scalar χ_head_eff contracting both) collapse to one all-reduce per q,
  small payload.

This module is purely the **algebraic kernel**.  It does not yet replace
production W — a separate driver patch will plumb it into the static
COHSEX path once the test in ``tests/test_head_wing_schur.py`` confirms
both correctness (vs dense reference, machine precision) and that the
HLO has no collectives in the outer-product steps.

API summary
-----------
The caller provides four pre-replicated copies of every per-q vector
(``g0``, ``χ_wing``, ``χ_wing'``) — one sharded on each μ axis — so the
outer products at the start (V_body extract) and end (W rebuild) need
no all-gather.

* ``extract_V_body_sharded`` — V_body = V_q − v_head · g0_x* ⊗ g0_y
* ``solve_W_body0_sharded`` — W_body0 = (I − V_body·χ_body)⁻¹·V_body
  (delegates to existing ``w_isdf._get_w_solve_fn_local``).
* ``schur_reductions_sharded`` — assemble (χ_head_eff, A_wing, A_wing').
* ``assemble_W_sharded`` — W = W_body0 + (g0_x* + A_wing_x) · W_head ·
  (g0_y + A_wing'_y).
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# ---------------------------------------------------------------------------
# Sharding specs.  Match the production V_FFT5D_SPEC convention but
# expressed in flat-q form (nq, μ, μ) so this module composes with
# ``w_isdf._get_w_solve_fn_local`` which already takes flat-q V/χ.
# ---------------------------------------------------------------------------

V_FLATQ_SPEC = P(None, 'x', 'y')      # (nq, μ_X, μ_Y) — same as W_q in flat-q form
G0_X_SPEC    = P(None, 'x')           # (nq, μ_X) — replicated on y
G0_Y_SPEC    = P(None, 'y')           # (nq, μ_Y) — replicated on x
SCALAR_Q_SPEC = P(None,)              # (nq,) — replicated

# Intermediate sharding for the V_FLATQ_SPEC ↔ q-shard reshard.
# Same trick as ``w_isdf._get_w_solve_fn_local``: routing through
# ``P('x', None, 'y')`` lets SPMD plan the reshard as two single-axis
# all_to_alls (one on x for the q axis, one on y for the μ axis)
# instead of falling into "Involuntary full rematerialization" when
# asked to un-shard or re-shard both x and y simultaneously.  The
# helper below applies this stage at the entry of any function that
# consumes W_body0 (output of ``_get_w_solve_fn_local``) and re-imposes
# V_FLATQ_SPEC for the kernel body.
_RESHARD_MID_SPEC = P('x', None, 'y')


def _reshard_W_to_flatq(W_q: jax.Array, mesh_xy: Mesh) -> jax.Array:
    """Two-stage reshard back to V_FLATQ_SPEC, avoiding SPMD remat.

    ``with_sharding_constraint`` to ``P('x', None, 'y')`` first parks
    one axis at a time, then a second constraint to ``V_FLATQ_SPEC``
    moves the other.  Without the intermediate, XLA hits "Involuntary
    full rematerialization" when the producer (``_get_w_solve_fn_local``)
    leaves W q-sharded across the combined ('x','y') axis tuple.
    """
    W_q = jax.lax.with_sharding_constraint(
        W_q, NamedSharding(mesh_xy, _RESHARD_MID_SPEC))
    W_q = jax.lax.with_sharding_constraint(
        W_q, NamedSharding(mesh_xy, V_FLATQ_SPEC))
    return W_q


# ---------------------------------------------------------------------------
# Step 1: rank-1 head-channel subtract  (zero comm)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=('mesh_xy',))
def extract_V_body_sharded(
    V_q: jax.Array,           # (nq, μ_X, μ_Y) — V_FLATQ_SPEC
    g0_x: jax.Array,          # (nq, μ) on x — G0_X_SPEC
    g0_y: jax.Array,          # (nq, μ) on y — G0_Y_SPEC
    v_head_q: jax.Array,      # (nq,) replicated
    *,
    mesh_xy: Mesh,
) -> jax.Array:
    """V_body[q, μ, ν] = V_q[q, μ, ν] − v_head[q] · conj(g0_x[q,μ]) · g0_y[q,ν].

    Zero-comm: ``conj(g0_x)[q, μ_X, None]`` is local on each x-shard, and
    ``g0_y[q, None, μ_Y]`` is local on each y-shard.  The product is a
    pure ``μ_X × μ_Y`` outer product on every device.
    """
    rank1 = (jnp.conj(g0_x)[:, :, None]) * (g0_y[:, None, :])
    V_body = V_q - v_head_q[:, None, None] * rank1
    return jax.lax.with_sharding_constraint(
        V_body, NamedSharding(mesh_xy, V_FLATQ_SPEC))


# ---------------------------------------------------------------------------
# Step 2: body solve W_body0 = (I − V_body·χ_body)⁻¹ · V_body
# ---------------------------------------------------------------------------
# Delegates to the existing ``w_isdf._get_w_solve_fn_local`` — this already
# uses q-parallel shard_map for the LU and reshards back to V_FLATQ_SPEC,
# so we just hand off.  We keep a thin wrapper so callers see a uniform
# API alongside the rank-1 helpers.

def solve_W_body0_sharded(
    V_body_q: jax.Array,
    chi_body_q: jax.Array,
    *,
    mesh_xy: Mesh,
    pref: complex = 1.0+0j,
) -> jax.Array:
    """Return W_body0 = (1 − pref · V_body·χ_body)⁻¹ · V_body, on V_FLATQ_SPEC.

    ``pref`` matches the convention of the existing solver
    (``gw_jax`` passes ``2.0+0j`` for the spinor factor in static COHSEX);
    the test passes ``1.0+0j`` for synthetic data.
    """
    from ..w_isdf import _get_w_solve_fn_local
    nq = int(V_body_q.shape[0])
    n_mu = int(V_body_q.shape[1])
    solve_w = _get_w_solve_fn_local(mesh_xy, nq, n_mu)
    pref_arr = jnp.asarray(pref, dtype=jnp.complex128)
    W = solve_w(V_body_q, chi_body_q, pref_arr)
    # Current contract: ``_get_w_solve_fn_local`` lands W 2-D sharded at
    # ``P(None,'x','y')`` (the former replicated ``rep_3d`` output was
    # removed — scorecard J.2 #3).  The two-stage reshard here moves that
    # to V_FLATQ_SPEC without an involuntary full rematerialization, so
    # consumers (schur_reductions_sharded, assemble_W_sharded) see a
    # cleanly-sharded input and don't need any reshard themselves.
    return _reshard_W_to_flatq(W, mesh_xy)


# ---------------------------------------------------------------------------
# Step 3: Schur reductions  →  χ_head_eff, A_wing, A_wing'
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=('mesh_xy',))
def schur_reductions_sharded(
    W_body0_q: jax.Array,         # (nq, μ_X, μ_Y) — V_FLATQ_SPEC
    chi_wing_x: jax.Array,        # (nq, μ) on x — G0_X_SPEC  (left  contraction)
    chi_wing_y: jax.Array,        # (nq, μ) on y — G0_Y_SPEC  (right contraction)
    chi_wingp_x: jax.Array,       # (nq, μ) on x — G0_X_SPEC  (left  contraction)
    chi_wingp_y: jax.Array,       # (nq, μ) on y — G0_Y_SPEC  (right contraction)
    chi_head_q: jax.Array,        # (nq,) replicated — bare χ_head from ISDF SoS
    *,
    mesh_xy: Mesh,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Return (χ_head_eff, A_wing_x, A_wingp_y) per q.

    ``chi_wing_y`` contracts the ν axis of W_body0 → A_wing_x lives on x.
    ``chi_wingp_x`` contracts the μ axis of W_body0 → A_wingp_y lives on y.
    χ_head_eff contracts both → scalar per q.

    Each contraction is one psum on the contracted axis.  Total comm per q:
    one all-reduce on y for A_wing, one on x for A_wing', and one on the
    full mesh for χ_head_eff.  Three small reductions per q.

    **Caller contract:** ``W_body0_q`` MUST already be on V_FLATQ_SPEC.
    ``solve_W_body0_sharded`` returns it that way; if you got W_body0
    from a different source, two-stage-reshard via
    ``_reshard_W_to_flatq`` first.  This function does NOT defensively
    reshard, because doing so charges a 16-all-to-all reshard cost on
    every call even when the input is already correctly sharded.
    """
    # A_wing[q, μ_X] = Σ_ν W_body0[q, μ_X, ν_Y] · chi_wing_y[q, ν_Y]
    #   einsum reduces along the y axis → output on x
    A_wing_x = jnp.einsum('qmn,qn->qm', W_body0_q, chi_wing_y, optimize=True)
    A_wing_x = jax.lax.with_sharding_constraint(
        A_wing_x, NamedSharding(mesh_xy, G0_X_SPEC))

    # A_wing'[q, ν_Y] = Σ_μ chi_wingp_x[q, μ_X] · W_body0[q, μ_X, ν_Y]
    A_wingp_y = jnp.einsum('qm,qmn->qn', chi_wingp_x, W_body0_q, optimize=True)
    A_wingp_y = jax.lax.with_sharding_constraint(
        A_wingp_y, NamedSharding(mesh_xy, G0_Y_SPEC))

    # χ_head_eff[q] = χ_head[q] + Σ_μν chi_wingp_x[q,μ] · W_body0[q,μ,ν] · chi_wing_y[q,ν]
    correction = jnp.einsum('qm,qmn,qn->q',
                            chi_wingp_x, W_body0_q, chi_wing_y, optimize=True)
    chi_head_eff_q = chi_head_q + correction
    chi_head_eff_q = jax.lax.with_sharding_constraint(
        chi_head_eff_q, NamedSharding(mesh_xy, SCALAR_Q_SPEC))

    return chi_head_eff_q, A_wing_x, A_wingp_y


def W_head_scalar_per_q(v_head_q: jax.Array, chi_head_eff_q: jax.Array) -> jax.Array:
    """W_head[q] = v_head[q] / (1 − v_head[q] · χ_head_eff[q]).  Replicated."""
    return v_head_q / (1.0 - v_head_q * chi_head_eff_q)


# ---------------------------------------------------------------------------
# Step 4: rank-1 W rebuild  (zero comm)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=('mesh_xy',))
def assemble_W_sharded(
    W_body0_q: jax.Array,         # (nq, μ_X, μ_Y) — V_FLATQ_SPEC
    g0_x: jax.Array,              # (nq, μ) on x
    g0_y: jax.Array,              # (nq, μ) on y
    A_wing_x: jax.Array,          # (nq, μ) on x
    A_wingp_y: jax.Array,         # (nq, μ) on y
    W_head_scalar_q: jax.Array,   # (nq,) replicated
    *,
    mesh_xy: Mesh,
) -> jax.Array:
    """W[q, μ, ν] = W_body0[q, μ, ν]
                  + W_head[q] · (conj(g0_x) + A_wing_x)[q, μ]
                              · (g0_y + A_wingp_y)[q, ν].

    Zero-comm: every component on the rhs is already on the right
    axis (x-shard for left, y-shard for right).  The outer product is
    a pure local kernel on each device.

    **Caller contract:** ``W_body0_q`` must already be on V_FLATQ_SPEC
    (use ``solve_W_body0_sharded`` whose output respects this; or
    apply ``_reshard_W_to_flatq`` first).  This function trusts the
    sharding label and does no defensive resharding.
    """
    left  = jnp.conj(g0_x) + A_wing_x        # (nq, μ_X) on x
    right = g0_y           + A_wingp_y       # (nq, μ_Y) on y
    rank1 = W_head_scalar_q[:, None, None] * (left[:, :, None]) * (right[:, None, :])
    W = W_body0_q + rank1
    return jax.lax.with_sharding_constraint(
        W, NamedSharding(mesh_xy, V_FLATQ_SPEC))


# ---------------------------------------------------------------------------
# End-to-end driver (convenience)
# ---------------------------------------------------------------------------

def reconstruct_W_via_schur_sharded(
    V_q: jax.Array,
    chi_body_q: jax.Array,
    g0_x: jax.Array,
    g0_y: jax.Array,
    chi_wing_x: jax.Array,
    chi_wing_y: jax.Array,
    chi_wingp_x: jax.Array,
    chi_wingp_y: jax.Array,
    chi_head_q: jax.Array,
    v_head_q: jax.Array,
    *,
    mesh_xy: Mesh,
    pref: complex = 1.0+0j,
) -> jax.Array:
    """End-to-end Schur reconstruction of W from rank-1 head + body solve.

    Equivalent to ``solve_w(V_q, χ_q)`` modulo the user's ability to
    *override* the wing convention before W is rebuilt.  Pass the
    centroid-derived χ_wing for the literal centroid result; pass
    ``chi_wing_y = 0`` (and similarly chi_wingp_x = 0) to mirror BGW's
    fixwings q=0 behavior — that's the planned use case.
    """
    V_body = extract_V_body_sharded(V_q, g0_x, g0_y, v_head_q, mesh_xy=mesh_xy)
    W_body0 = solve_W_body0_sharded(V_body, chi_body_q, mesh_xy=mesh_xy, pref=pref)
    chi_head_eff_q, A_wing_x, A_wingp_y = schur_reductions_sharded(
        W_body0, chi_wing_x, chi_wing_y, chi_wingp_x, chi_wingp_y,
        chi_head_q, mesh_xy=mesh_xy)
    W_head_q = W_head_scalar_per_q(v_head_q, chi_head_eff_q)
    return assemble_W_sharded(
        W_body0, g0_x, g0_y, A_wing_x, A_wingp_y, W_head_q, mesh_xy=mesh_xy)
