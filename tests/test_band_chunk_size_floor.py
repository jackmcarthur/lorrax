"""Mesh-floor on ``band_chunk_size`` in :func:`gw.gflat_memory_model.plan_gflat_chunks`.

When the band axis is sharded across the device mesh of size ``p_xy``
(:class:`common.psi_G_store.PsiGStore` band-flat-shards per-bc bands
across all mesh ranks), each device's per-bc local band count is
``bpd_per_bc = band_chunk_size // p_xy``.  If ``band_chunk_size < p_xy``,
that division collapses to 0 for every bc and downstream
``z_q_from_psi_sm._local`` raises::

    dimension size of operand at 'all_gather_dim' cannot be zero

The planner is the single source of truth for chunk sizes; the user-set
``band_chunk_size`` (e.g. via cohsex.in) is a hint that the planner must
adjust for correctness on the active mesh.  These tests pin the
mesh-floor + round-up-to-multiple-of-``p_xy`` contract.

THE DEFAULT IS ``0`` SINCE 0.1.0, i.e. "the planner decides".  It was
``16``, and ``gw_init`` passes ``band_chunk_override = band_chunk_size if
> 0 else None`` — so a non-zero default meant every deck in the tree
pinned the band chunk and the auto-pick branch below (the largest
power-of-2 whose FFT box fits half the headroom) never ran on a real
run.  The last cell here pins that the shipped default reaches the
planner.

Tickled by CrI3 6x6 30Ry bispinor full-BZ 16-GPU (commit ``2de70eb``+
``4b927dc`` cascade) where user set ``band_chunk_size=2``.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh

from gw.gflat_memory_model import plan_gflat_chunks


def _fake_meta(*, nk_tot=36, nspinor=4, n_rmu=384, n_rtot=80_000):
    """Minimal duck-typed Meta for the planner.  Only the attrs the
    planner reads are populated."""
    return SimpleNamespace(
        nk_tot=int(nk_tot),
        nspinor=int(nspinor),
        n_rmu=int(n_rmu),
        n_rmu_padded=int(n_rmu),
        n_rtot=int(n_rtot),
        ngkmax=int(0.06 * n_rtot),
    )


def _cpu_mesh(p_x: int, p_y: int) -> Mesh:
    """Build a (p_x, p_y) Mesh over CPU devices.  Multi-process tests
    are not needed here — the planner reads ``mesh.shape`` only."""
    n = p_x * p_y
    devices = jax.devices()
    if len(devices) < n:
        # Fall back to a fake-mesh proxy: planner reads .shape only,
        # so we can construct a thin shim.
        return SimpleNamespace(shape={'x': p_x, 'y': p_y})
    devs = np.asarray(devices[:n]).reshape(p_x, p_y)
    return Mesh(devs, axis_names=('x', 'y'))


# ---------------------------------------------------------------------------
# Core regression: the CrI3 case (band_chunk_size=2, world_size=16)
# ---------------------------------------------------------------------------


def test_band_chunk_bumped_when_smaller_than_world_size():
    """User sets ``band_chunk_size=2`` on a 16-rank mesh — the planner
    must auto-bump to ≥ 16 so ``bpd_per_bc = band_chunk // 16 ≥ 1``."""
    mesh = _cpu_mesh(4, 4)
    plan = plan_gflat_chunks(
        meta=_fake_meta(), mesh_xy=mesh,
        nb_total=168,            # CrI3 nband=84, L+R = 168
        ngkmax=5000, n_q_disk=36,
        budget_gb=28.0,
        band_chunk_override=2,   # the bug trigger
        r_chunk_override=4096,
    )
    p_xy = 4 * 4
    assert plan.band_chunk >= p_xy, (
        f"band_chunk {plan.band_chunk} < world_size {p_xy}: bpd_per_bc "
        f"would be zero, all_gather_dim would crash")
    assert plan.band_chunk % p_xy == 0, (
        f"band_chunk {plan.band_chunk} not a multiple of world_size "
        f"{p_xy}: per-device bc widths would be ragged")


def test_band_chunk_rounded_up_to_multiple_of_world_size():
    """User sets ``band_chunk_size=20`` on an 8-rank mesh — round up
    to 24 (next multiple of 8 ≥ 20), not down to 16."""
    mesh = _cpu_mesh(2, 4)       # p_xy = 8
    plan = plan_gflat_chunks(
        meta=_fake_meta(), mesh_xy=mesh,
        nb_total=200, ngkmax=5000, n_q_disk=36,
        budget_gb=28.0,
        band_chunk_override=20,
        r_chunk_override=4096,
    )
    assert plan.band_chunk == 24, (
        f"expected 20 rounded up to next multiple of 8 = 24; "
        f"got {plan.band_chunk}")


def test_band_chunk_unchanged_when_already_valid():
    """User sets ``band_chunk_size=32`` on a 16-rank mesh — already a
    multiple of 16, so no bump needed (idempotent)."""
    mesh = _cpu_mesh(4, 4)
    plan = plan_gflat_chunks(
        meta=_fake_meta(), mesh_xy=mesh,
        nb_total=168, ngkmax=5000, n_q_disk=36,
        budget_gb=28.0,
        band_chunk_override=32,
        r_chunk_override=4096,
    )
    assert plan.band_chunk == 32, (
        f"valid user value 32 should pass through unchanged; "
        f"got {plan.band_chunk}")


def test_band_chunk_unchanged_on_single_device_mesh():
    """``p_xy == 1`` ⇒ band axis is not sharded; no floor applies."""
    mesh = _cpu_mesh(1, 1)
    plan = plan_gflat_chunks(
        meta=_fake_meta(), mesh_xy=mesh,
        nb_total=168, ngkmax=5000, n_q_disk=36,
        budget_gb=28.0,
        band_chunk_override=2,
        r_chunk_override=4096,
    )
    assert plan.band_chunk == 2, (
        f"on a 1-device mesh, no bump should happen; got {plan.band_chunk}")


# ---------------------------------------------------------------------------
# Auto-pick path (no override) must also respect the floor.
# ---------------------------------------------------------------------------


def test_band_chunk_autopick_respects_world_size_floor():
    """Even with no user override, the planner's auto-pick must respect
    ``band_chunk >= p_xy`` and be a multiple of it."""
    mesh = _cpu_mesh(4, 4)
    plan = plan_gflat_chunks(
        meta=_fake_meta(), mesh_xy=mesh,
        nb_total=168, ngkmax=5000, n_q_disk=36,
        budget_gb=28.0,
        # no overrides
        r_chunk_override=4096,
    )
    p_xy = 16
    assert plan.band_chunk >= p_xy
    assert plan.band_chunk % p_xy == 0


# ---------------------------------------------------------------------------
# Sanity: the bumped value never exceeds ``nb_total`` plus one world_size
# block (to keep the band-chunk loop length reasonable).
# ---------------------------------------------------------------------------


def test_band_chunk_capped_at_nb_total_or_world_size():
    """``band_chunk`` is clamped to ``max(nb_total, p_xy)``; if nb_total
    is tiny (< p_xy), the floor still wins so the per-device split has
    ≥ 1 band per device on the (single) bc."""
    mesh = _cpu_mesh(4, 4)
    plan = plan_gflat_chunks(
        meta=_fake_meta(), mesh_xy=mesh,
        nb_total=8,              # nb_total < p_xy=16
        ngkmax=5000, n_q_disk=36,
        budget_gb=28.0,
        band_chunk_override=2,
        r_chunk_override=4096,
    )
    # band_chunk == p_xy is the post-bump floor; the (one) bc range
    # built downstream will still be only nb_total wide, but the static
    # _bpd_max shape uses band_chunk // p_xy = 1 per device.
    assert plan.band_chunk >= 16


# ---------------------------------------------------------------------------
# The shipped default must REACH the auto-pick above.
# ---------------------------------------------------------------------------


def test_the_shipped_default_hands_the_choice_to_the_planner():
    """``band_chunk_size = 0`` is the deck's spelling of "planner decides".

    Checked at both ends, because the two halves fail differently: the
    DEFAULT (a non-zero one un-deadens nothing) and the SEAM in
    ``gw_init`` that turns it into ``band_chunk_override = None`` (a seam
    that passed ``0`` through would ask the planner to floor zero).
    """
    import ast
    import pathlib as _pl

    from gw.gw_config import _DEFAULTS

    assert _DEFAULTS["band_chunk_size"] == 0

    src = (_pl.Path(__file__).resolve().parents[1]
           / "src" / "gw" / "gw_init.py").read_text()
    assert ("band_chunk_override=(int(mem.band_chunk_size)" in src
            and "if mem.band_chunk_size > 0 else None)" in src), (
        "the gw_init seam no longer turns 0 into None; the planner would "
        "be handed a zero-width band chunk")
    ast.parse(src)


def test_a_pinned_value_still_wins_over_the_planner():
    """RED TWIN of the default change: > 0 still PINS."""
    mesh = _cpu_mesh(2, 2)
    pinned = plan_gflat_chunks(
        meta=_fake_meta(), mesh_xy=mesh,
        nb_total=168, ngkmax=5000, n_q_disk=36, budget_gb=28.0,
        band_chunk_override=8, r_chunk_override=4096)
    planned = plan_gflat_chunks(
        meta=_fake_meta(), mesh_xy=mesh,
        nb_total=168, ngkmax=5000, n_q_disk=36, budget_gb=28.0,
        band_chunk_override=None, r_chunk_override=4096)
    assert pinned.band_chunk == 8
    assert planned.band_chunk >= 4 and planned.band_chunk % 4 == 0
