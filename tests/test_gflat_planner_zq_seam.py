"""Stage-C planning of the full-BZ Z_q build (2026-08-22).

Register row closed here (sandbox KNOWN_LORRAX_ISSUES 2026-08-19/20): the
advertised Stage-C HWM omitted or undercounted the full-BZ ``Z_q`` build —
the MoS2 8x8 full-BZ fit certified 54.33 GB/dev against a 64-GB budget and
then the first ``z_q_phase`` requested another 30.24 GiB and OOMed
(JID 57269074 step lx-Xg4-005932); the naturally-unreduced 9x9/81-q run
printed its own plan at 244% of budget, proceeded, and OOMed the same way
(JID 57281385 step .28).  Both failing requests are byte-exact matches of
the SINGLE Stage-C pair-density temp arena
``slots * nk * ns^2 * (mu/p_x) * (cr/p_y) * 16``.

Three planner changes are pinned:

* the arena is additionally capped as ONE contiguous allocation at
  ``_ARENA_PLACEMENT_FRAC`` of the post-persistent headroom;
* ``Z_q`` is charged LIVE ACROSS the solve seam (``solve_t + Z_q`` — a
  sum, not the old max — because ``solve_phase`` takes the full-BZ Z_q as
  a live input whose reshard cannot alias);
* the budget-derived r-chunk cap OUTRANKS the mu-wide performance floor
  (the floor is how the 9x9 run was handed r_chunk = mu against an
  infeasible plan).

``r_chunk_override`` keeps its register-documented run-level-workaround
authority and bypasses every cap.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from gw.gflat_memory_model import (
    _ARENA_PLACEMENT_FRAC, plan_gflat_chunks)

_C16 = 16


def _fake_meta(*, nk_tot, nspinor, n_rmu, n_rtot, ngkmax):
    return SimpleNamespace(
        nk_tot=int(nk_tot), nspinor=int(nspinor), n_rmu=int(n_rmu),
        n_rmu_padded=int(n_rmu), n_rtot=int(n_rtot), ngkmax=int(ngkmax))


def _mesh_shim(p_x, p_y):
    """The planner reads mesh.shape only; a real Mesh routes Stage A
    through the compiled FFT probe, which this unit does not need."""
    return SimpleNamespace(shape={'x': p_x, 'y': p_y})


def test_budget_outranks_the_mu_floor_and_the_arena_is_placeable():
    """The measured MoS2 8x8 full-BZ geometry: pre-fix the planner chose
    r_chunk=7888 (the mu floor did not bind, but the arena was 32.47 GiB
    — over half the pool as one allocation).  Post-fix the single-arena
    placement cap must bound the arena at _ARENA_PLACEMENT_FRAC of the
    post-persistent headroom, which on this geometry also lands the
    chosen r_chunk BELOW mu — the floor no longer outranks memory."""
    nk = nq = 64
    ns = 2
    mu = 5360
    slots = 3   # pinned: GPU value, independent of this test's backend
    p_x = p_y = 4
    plan = plan_gflat_chunks(
        meta=_fake_meta(nk_tot=nk, nspinor=ns, n_rmu=mu, n_rtot=46080,
                        ngkmax=1975),
        mesh_xy=_mesh_shim(p_x, p_y),
        nb_total=608, fit_nb_total=608, ngkmax=1975, n_q_disk=nq,
        budget_gb=64.0, band_chunk_override=608,
        pair_density_slots=slots,
        distributed_zeta_solve="distributed")
    p_xy = p_x * p_y
    target = 64.0e9 * plan.target_utilization
    headroom = target - plan.persistent_bytes
    arena = slots * nk * ns * ns * (mu / p_x) * (plan.r_chunk / p_y) * _C16
    # One p_xy rounding step of slack on the cap.
    arena_slope = slots * nk * ns * ns * mu * _C16 / p_xy
    assert arena <= _ARENA_PLACEMENT_FRAC * headroom + arena_slope * p_xy, (
        f"arena {arena / 1024**3:.2f} GiB exceeds the placement cap "
        f"({_ARENA_PLACEMENT_FRAC:.2f} x {headroom / 1024**3:.2f} GiB "
        f"headroom) at r_chunk={plan.r_chunk}")
    # The mu-wide performance floor no longer overrides the budget.
    assert plan.r_chunk < mu, (
        f"r_chunk={plan.r_chunk} still floored at mu={mu}: the budget "
        f"does not outrank the performance floor")
    # And the certified HWM actually fits the budget it certified against.
    assert plan.hwm_bytes <= plan.budget_bytes


def test_zq_is_charged_live_across_the_solve_seam():
    """C-stage transient >= solve_t + Z_q on a solve-dominated geometry.
    Pre-fix ``C_t = max(C_fit_t, solve_t)`` dropped the live Z_q from
    under the solve — exactly the seam both measured OOMs sat on."""
    nk = nq = 8
    ns = 1
    mu = 64
    slots = 3
    p_x = p_y = 2
    r_override = 64
    plan = plan_gflat_chunks(
        meta=_fake_meta(nk_tot=nk, nspinor=ns, n_rmu=mu, n_rtot=4096,
                        ngkmax=200),
        mesh_xy=_mesh_shim(p_x, p_y),
        nb_total=16, fit_nb_total=16, ngkmax=200, n_q_disk=nq,
        budget_gb=32.0, band_chunk_override=8,
        r_chunk_override=r_override,
        pair_density_slots=slots,
        distributed_zeta_solve="replicated")
    p_xy = p_x * p_y
    cr = plan.r_chunk
    rhs_stacks = 2 * nq * mu * cr * _C16 / p_xy
    factor = plan.q_chunk * mu * mu * _C16
    zq_live = nq * mu * cr * _C16 / p_xy
    c_transient = plan.peak_breakdown["C_fit_one_rchunk"] \
        - plan.persistent_bytes
    assert c_transient >= rhs_stacks + factor + zq_live - 1.0, (
        f"C transient {c_transient:.0f} B < solve {rhs_stacks + factor:.0f}"
        f" + live Z_q {zq_live:.0f} B: the solve seam dropped Z_q")


def test_r_chunk_override_bypasses_every_cap():
    """The register-documented run-level workaround must keep winning."""
    nk = nq = 64
    ns = 2
    mu = 5360
    plan = plan_gflat_chunks(
        meta=_fake_meta(nk_tot=nk, nspinor=ns, n_rmu=mu, n_rtot=46080,
                        ngkmax=1975),
        mesh_xy=_mesh_shim(4, 4),
        nb_total=608, fit_nb_total=608, ngkmax=1975, n_q_disk=nq,
        budget_gb=64.0, band_chunk_override=608,
        r_chunk_override=7888,
        pair_density_slots=3,
        distributed_zeta_solve="distributed")
    assert plan.r_chunk == 7888  # already a p_xy multiple; untouched


def test_capped_r_chunk_stays_mesh_divisible_and_positive():
    """A budget small enough to bottom the cap out must still emit a
    legal (p_xy-multiple, >= p_xy) width, never zero."""
    plan = plan_gflat_chunks(
        meta=_fake_meta(nk_tot=36, nspinor=2, n_rmu=768, n_rtot=80_000,
                        ngkmax=4000),
        mesh_xy=_mesh_shim(2, 2),
        nb_total=64, fit_nb_total=64, ngkmax=4000, n_q_disk=36,
        budget_gb=6.0, band_chunk_override=16,
        pair_density_slots=3)
    assert plan.r_chunk >= 4 and plan.r_chunk % 4 == 0
    n_chunks = math.ceil(80_000 / plan.r_chunk)
    assert plan.n_r_chunks == n_chunks
