"""Route selection for the charge-channel ζ-solve.

The charge ζ-solve has one route -- the *replicated* one -- that carries the
rank-truncation cure (``charge_zeta_solve='rank_truncate'``, the production
default).  Every other route returns a plain Cholesky factor that is **not**
rank-conditioned.  When a rank-deficient C_q is factored that way the fit
returns a ζ that is far too large, V_q rebuilds to relF O(10) instead of
O(1e-15), and Σ comes out as nonsense while every stage still reports success.

That failure is silent: the only trace is the route name in the per-q
``Computing L_q`` line.  These tests pin the route decision so a regression
cannot reintroduce it.

The regression these were written for: on a CPU backend ``gw_config`` rewrote
``distributed_cholesky=auto`` to ``'off'`` (reasoning that cuSOLVERMp is
CUDA-only), but ``'off'`` is an *override* that short-circuits the whole policy
to ``sharded_cholesky`` -- so the replicated route, which is plain dense JAX and
perfectly valid on CPU, became unreachable there.  A full MoS2 12x12 G0W0 ran to
completion with rc=0 and produced a QP gap of **-161 eV**.

Pure decision logic: no GPU and no multi-process launch required.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
from jax.sharding import Mesh

from isdf.core import _resolve_solver_kind_charge


def _mesh(px: int, py: int) -> Mesh:
    """A (px, py) mesh over the available (CPU) devices."""
    n = px * py
    devs = jax.devices()
    if len(devs) < n:
        pytest.skip(f"needs {n} devices, have {len(devs)}")
    return Mesh(np.array(devs[:n]).reshape(px, py), ['x', 'y'])


# nq * n_mu**2 * 16 bytes vs the 4 GiB default cap.
_UNDER_CAP = dict(nq=144, n_rmu=1200)      # 3.32 GiB
_OVER_CAP = dict(nq=144, n_rmu=2412)       # 13.4 GiB stack; ~90 MiB per q-batch
# Above the PER-Q-BATCH factor cap too (the binding limit for rank_truncate
# since the _replicate_rank_truncate_ok widening: one (mu, mu) c128 matrix
# alone exceeds _REPLICATED_FACTOR_MAX_BATCH_BYTES = 4 GiB, so the refusal —
# not a batch split — is the contract).  17000^2 * 16 B = 4.31 GiB.
_OVER_FACTOR_CAP = dict(nq=144, n_rmu=17000)


class TestReplicatedRouteReachable:
    """Below the cap the replicated route must win -- on any backend."""

    def test_auto_rank_truncate_selects_replicated(self):
        # THE regression: this returned 'sharded_cholesky' on CPU because the
        # config layer had already rewritten auto -> off.
        got = _resolve_solver_kind_charge(
            _mesh(2, 2), override="auto",
            charge_zeta_solve="rank_truncate", **_UNDER_CAP)
        assert got == "replicated_rank_truncate"

    def test_auto_cholesky_selects_replicated_cholesky(self):
        got = _resolve_solver_kind_charge(
            _mesh(2, 2), override="auto",
            charge_zeta_solve="cholesky", **_UNDER_CAP)
        assert got == "replicated_cholesky"

    def test_replicated_route_is_mesh_invariant(self):
        """Same answer on 1x1, 1x2 and 2x2 -- the replicated factor does not
        depend on the process grid (that mesh-dependence is exactly why the
        block-cyclic routes drift at mildly rank-deficient n_mu)."""
        kw = dict(override="auto", charge_zeta_solve="rank_truncate",
                  **_UNDER_CAP)
        seen = {_resolve_solver_kind_charge(_mesh(px, py), **kw)
                for px, py in ((1, 1), (1, 2), (2, 2))}
        assert seen == {"replicated_rank_truncate"}


class TestOverridesStillHonoured:
    """Explicit overrides keep their documented meaning."""

    def test_off_forces_sharded(self):
        got = _resolve_solver_kind_charge(
            _mesh(2, 2), override="off",
            charge_zeta_solve="rank_truncate", **_UNDER_CAP)
        assert got == "sharded_cholesky"

    def test_off_is_a_footgun_not_a_gpu_switch(self):
        """Documents the trap: 'off' reads like "don't use the GPU backend" but
        it also discards the rank-truncation cure, even below the cap where the
        replicated route was available."""
        under = _resolve_solver_kind_charge(
            _mesh(2, 2), override="off",
            charge_zeta_solve="rank_truncate", **_UNDER_CAP)
        assert "rank_truncate" not in under


class TestAboveCap:
    """Above the cap the replicated route cannot be used."""

    def test_rank_truncate_refuses_rather_than_downgrading(self):
        # Refuse loudly: silently returning a non-rank-conditioned factor is
        # what produced the bad production ζ in the first place.  The refusal
        # threshold MOVED with the _replicate_rank_truncate_ok widening
        # (ladder R15.1): a stack-over-cap fit (the old _OVER_CAP, 13.4 GiB
        # stack but ~4 GiB per q-batch) now legitimately runs replicated
        # one q-batch at a time; only a fit whose SINGLE (mu, mu) factor
        # exceeds the per-batch cap has nowhere left to go and must refuse.
        with pytest.raises(ValueError, match="rank_truncate"):
            _resolve_solver_kind_charge(
                _mesh(2, 2), override="auto",
                charge_zeta_solve="rank_truncate", **_OVER_FACTOR_CAP)

    def test_rank_truncate_stack_over_cap_still_runs_per_q_batch(self):
        # The R15.1 widening pinned: stack-over-cap but batch-under-cap keeps
        # the rank-truncation cure (refusing here lost physics, not memory).
        got = _resolve_solver_kind_charge(
            _mesh(2, 2), override="auto",
            charge_zeta_solve="rank_truncate", **_OVER_CAP)
        assert got == "replicated_rank_truncate"

    def test_cpu_mesh_never_auto_picks_cusolvermp(self):
        """cuSOLVERMp is CUDA-only; auto must not select it on a host mesh.

        This is what makes letting ``auto`` through on CPU safe.
        """
        mesh = _mesh(2, 2)
        if mesh.devices.flat[0].platform != "cpu":
            pytest.skip("CPU-backend assertion")
        got = _resolve_solver_kind_charge(
            mesh, override="auto",
            charge_zeta_solve="cholesky", **_OVER_CAP)
        assert got == "sharded_cholesky"
