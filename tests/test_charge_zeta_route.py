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


# ---------------------------------------------------------------------------
#  D18's third contract: the SIX raising probe calls, made falsifiable
# ---------------------------------------------------------------------------
# The replumb brief required that "the refusal contracts at bse_setup:390-414,
# w_isdf:505-511, isdf/core:1060-1067 survive VERBATIM IN EFFECT".  Two of the
# three are prose: w_isdf's is a comment block whose whole effect is the
# ABSENCE of an is_native re-check, and isdf/core's is a docstring bullet in
# _resolve_channel_ladder.  "Survives verbatim in effect" is unfalsifiable for
# a comment.
#
# The falsifiable form of isdf/core's is Arm B's D2: the explicit
# 'cusolvermp'/'slate'/'scalapack'/'distributed' handlers must still CALL the
# raising probe, and the raise must still reach the caller.  There are six
# such calls and A1's site table counted them as zero, so a conversion that
# dropped one would have moved a loud refusal into a silent different-backend
# run with nothing failing.  Two cells: the sites exist, and the raise
# propagates.

_PROBE_SITES = {
    # enclosing function            -> the (op, backend) pairs it must probe
    "_resolve_solver_kind_charge":     {("cholesky", "slate"),
                                        ("cholesky", "cusolvermp")},
    "_resolve_solver_kind_transverse": {("solve_lu", "scalapack"),
                                        ("solve_lu", "cusolvermp")},
    "_resolve_zeta_gather":            {("eigh", "distributed")},
}


def _probe_calls_by_function():
    """AST: every ``_resolve_linalg_backend(op, backend, ...)`` in isdf.core,
    keyed by the top-level function that encloses it."""
    import ast
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src/isdf/core.py"
    tree = ast.parse(src.read_text())
    found = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            fn = sub.func
            if getattr(fn, "id", None) != "_resolve_linalg_backend":
                continue
            args = [a.value for a in sub.args if isinstance(a, ast.Constant)]
            assert len(args) >= 2, ast.dump(sub)
            found.setdefault(node.name, set()).add((args[0], args[1]))
    return found


def test_the_six_explicit_backends_still_run_the_raising_probe():
    """AST, not grep: the six probe calls, by enclosing function and by the
    (op, backend) pair each one asserts.

    A grep for the NAME proves nothing — the module-level alias is imported
    whether or not anything calls it.  Naming the pairs means a conversion
    that kept five calls and dropped ``('solve_lu', 'scalapack')`` fails
    here rather than in a run that quietly used a different backend.

    RED ARM: deleting any one call, or changing a backend string, fails with
    the missing pair named.
    """
    found = _probe_calls_by_function()
    for fname, pairs in _PROBE_SITES.items():
        assert fname in found, (
            f"isdf.core.{fname} makes no _resolve_linalg_backend call at "
            f"all; the explicit-backend refusals in it are gone")
        assert pairs <= found[fname], (
            f"isdf.core.{fname} lost probe(s) {sorted(pairs - found[fname])}"
            f"; it still probes {sorted(found[fname])}")
    total = sum(len(v) for v in found.values())
    assert total >= 5, found
    # ...and the alias is the module attribute the two worker suites
    # monkeypatch, not a locally re-spelled import (Arm B seams 4 and 5).
    import isdf.core
    assert callable(isdf.core._resolve_linalg_backend)


@pytest.mark.parametrize("resolver,kwargs,override", [
    ("_resolve_solver_kind_charge",
     dict(charge_zeta_solve="cholesky", **_UNDER_CAP), "slate"),
    ("_resolve_solver_kind_charge",
     dict(charge_zeta_solve="cholesky", **_UNDER_CAP), "cusolvermp"),
    ("_resolve_solver_kind_transverse",
     dict(transverse_zeta_solve="ridge", n_rmu_logical=64),
     "scalapack"),
    ("_resolve_solver_kind_transverse",
     dict(transverse_zeta_solve="ridge", n_rmu_logical=64),
     "cusolvermp"),
])
def test_a_refusing_resolver_reaches_the_isdf_caller(monkeypatch, resolver,
                                                     kwargs, override):
    """The raise PROPAGATES — it is not caught and turned into a route.

    The AST cell above proves the calls are there; it cannot prove they are
    not wrapped in a ``try/except`` that demotes.  This one makes the door
    refuse for a reason nothing else in the ladder would produce, and
    requires that exact reason out the top.  An explicit backend that cannot
    be honoured REFUSES (doctrine 3); a caught refusal would run some other
    factorisation successfully and silently.

    RED ARM: wrapping any probe call in ``try/except Exception: pass`` makes
    the call return a route string and this cell fails with it in hand.
    """
    import isdf.core as core
    mesh = _mesh(1, 1)
    sentinel = "REFUSED-BY-THE-DOOR-sentinel"

    def _refuse(op, backend, mesh_xy, **kw):
        raise RuntimeError(f"{sentinel}: {op}/{backend}")

    monkeypatch.setattr(core, "_resolve_linalg_backend", _refuse)
    fn = getattr(core, resolver)
    with pytest.raises(RuntimeError, match=sentinel) as exc:
        fn(mesh, override=override, **kwargs)
    assert override in str(exc.value), (
        f"the refusal reached the caller but lost the backend it was "
        f"about: {exc.value}")


# ---------------------------------------------------------------------------
#  The two closures Arm B left open on the FactorToken
# ---------------------------------------------------------------------------

def test_a_factor_token_cannot_enter_the_composed_kernel():
    """The pytree DECLINE, pinned — which is what makes it safe to rely on.

    ``isdf.core``'s composed ``@jax.jit _kernel`` takes ``L_q`` as a TRACED
    ARGUMENT.  ``FactorToken`` is a frozen dataclass that is deliberately
    not registered as a pytree, so jax refuses it by name at the jit
    boundary instead of tracing a handle.  The comment at :4250-4256 states
    that; nothing checked it, and the day somebody registers the token as a
    pytree "for convenience" the refusal becomes a silent trace of an opaque
    block-cyclic handle in the AOT/memory-model path — which the default
    suite does not exercise.

    RED ARM: register FactorToken as a pytree node and this raises nothing.
    """
    import ast
    import pathlib
    import isdf.core as core
    from distrib_la import FactorToken

    token = FactorToken(op="cholesky", backend="slate", mesh=_mesh(1, 1),
                        n=8, nbatch=2, _factor=object())
    with pytest.raises(TypeError):
        jax.jit(lambda L_q: L_q)(token)

    # ...and ``L_q`` really is a jit ARGUMENT of _kernel, which is the only
    # reason the decline above protects anything.
    src = pathlib.Path(core.__file__).with_suffix(".py").read_text()
    kernels = [n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.FunctionDef) and n.name == "_kernel"]
    assert kernels, "isdf.core has no _kernel"
    assert any("L_q" in [a.arg for a in k.args.args] for k in kernels), (
        "_kernel no longer takes L_q as a jit argument; the pytree decline "
        "is guarding nothing")


def test_the_token_seams_still_take_the_array_path_without_a_token():
    """The NEGATIVE branch of the ``isinstance(L_q, FactorToken)`` seams.

    Every one of them reads extents off the token instead of a ``.shape``
    the token does not have — and every one has an ``else`` that must still
    serve the plain sharded array the JAX routes return.  An isinstance seam
    is exactly the shape that gets tested on one side only.
    """
    import numpy as np
    import isdf.core as core
    from distrib_la import FactorToken

    array = np.zeros((3, 8, 8))
    assert core._factor_nbatch(array) == 3
    token = FactorToken(op="cholesky", backend="slate", mesh=_mesh(1, 1),
                        n=8, nbatch=3, _factor=object())
    assert core._factor_nbatch(token) == 3
