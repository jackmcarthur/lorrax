"""ONE replicated-eigh capacity gate, both ζ-fit channels.

Register row: "transverse resolver lacks the charge branch's capacity
gate; OOMs late above mu_T~16k".  ``charge_zeta_solve='rank_truncate'``
and ``transverse_zeta_solve='rank_truncate'`` allocate the SAME object —
one replicated ``(q_batch, n_mu, n_mu)`` complex128 eigh operand — but
only the charge resolver ever tested whether it fits.  A transverse fit
above ``n_mu_T ~ 16k`` therefore resolved cleanly and died on the
allocation hours later, after the charge fit had already been paid for.

What is pinned here:

1. **The predicate is shared, not copied.**  ``_replicate_rank_truncate_ok``
   returns the same verdict for the same ``(nq, n_mu)`` regardless of
   which channel asks — it is one function and this is the A/B that says
   so.
2. **The transverse resolver refuses where the charge resolver refuses**,
   with the same arithmetic and the same ``n_mu <= ...`` ceiling.
3. **Both escapes survive.**  ``nq=None`` (the ``gw_init`` pre-flight,
   which has only the centroid file) and ``replicated_factor_used=False``
   (``distributed_zeta_solve='distributed'`` replaces the factor, so the
   buffer is never allocated) both keep the route reachable.  Losing
   either would refuse runs on the size of a buffer they do not use --
   the defect the charge branch's own escape was added for.

Pure host: builds a 1x1 CPU mesh, no GPU, no FFI.  SCOPE: this is a
RESOLVER contract test.  It does not run a ζ fit and says nothing about
whether the eigh that is now admitted actually completes -- the refusal
text says so too ("raising the cap makes this RESOLVE, not finish").
"""
import math

import numpy as np
import pytest

import jax
from jax.sharding import Mesh

import isdf.core as core
from isdf.core import (
    _rank_truncate_capacity_error,
    _replicate_rank_truncate_ok,
    _resolve_solver_kind_transverse,
    _resolve_solver_kind_charge,
)


@pytest.fixture(scope="module")
def mesh11():
    d = jax.devices()[:1]
    return Mesh(np.array(d).reshape(1, 1), ("x", "y"))


def _mu_ceiling() -> int:
    cap = max(core._REPLICATED_CHOL_MAX_STACK_BYTES,
              core._REPLICATED_FACTOR_MAX_BATCH_BYTES)
    return int(math.isqrt(cap // 16))


# A μ comfortably past the ceiling (16,384 at the shipped 4 GiB caps) and
# one comfortably under it.  Derived from the caps, not hard-coded, so a
# cap change moves the test instead of silently exempting it.
_MU_TOO_BIG = _mu_ceiling() * 2
_MU_FINE = 512
_NQ = 8


def test_the_predicate_is_one_function_for_both_channels():
    """A/B: same (nq, n_mu) -> same verdict.  There is no second criterion."""
    assert _replicate_rank_truncate_ok(_NQ, _MU_FINE) is True
    assert _replicate_rank_truncate_ok(_NQ, _MU_TOO_BIG) is False
    # Unknown inputs are not "fits"; they are "do not decide here".
    assert _replicate_rank_truncate_ok(None, _MU_FINE) is False
    assert _replicate_rank_truncate_ok(_NQ, None) is False


def test_transverse_rank_truncate_refuses_where_charge_refuses(mesh11,
                                                               monkeypatch):
    """The gate that was missing.  Same size, same verdict, both channels."""
    monkeypatch.setattr(core, "_resolve_linalg_backend", lambda *a, **k: None)
    with pytest.raises(ValueError) as ti:
        _resolve_solver_kind_transverse(
            mesh11, "auto", n_rmu_logical=_MU_TOO_BIG,
            transverse_zeta_solve="rank_truncate", nq=_NQ)
    with pytest.raises(ValueError) as ch:
        _resolve_solver_kind_charge(
            mesh11, "auto", n_rmu=_MU_TOO_BIG, nq=_NQ,
            charge_zeta_solve="rank_truncate")
    t_msg, c_msg = str(ti.value), str(ch.value)
    # The message names the channel's OWN deck key and μ symbol...
    assert "transverse_zeta_solve='rank_truncate'" in t_msg
    assert "charge_zeta_solve='rank_truncate'" in c_msg
    assert f"n_mu_T={_MU_TOO_BIG}" in t_msg
    assert f"n_mu={_MU_TOO_BIG}" in c_msg
    # ...and both carry the SAME per-batch arithmetic and the SAME ceiling,
    # because it is the same buffer.
    ceiling = _mu_ceiling()
    assert f"n_mu_T <= {ceiling}" in t_msg
    assert f"n_mu <= {ceiling}" in c_msg
    assert "ONE q-batch, not the stack" in t_msg


def test_transverse_rank_truncate_is_admitted_at_a_size_that_fits(mesh11,
                                                                  monkeypatch):
    """The gate is not a blanket refusal: a fit-size transverse solve runs."""
    monkeypatch.setattr(core, "_resolve_linalg_backend", lambda *a, **k: None)
    kind = _resolve_solver_kind_transverse(
        mesh11, "auto", n_rmu_logical=_MU_FINE,
        transverse_zeta_solve="rank_truncate", nq=_NQ)
    assert kind == "transverse_rank_truncate"


def test_unknown_nq_keeps_the_legacy_policy(mesh11, monkeypatch):
    """gw_init's pre-flight has no nq; it must not refuse on a guess.

    The ζ-fit call site re-resolves WITH nq and refuses there, so nothing
    is lost -- but a pre-flight that refused every large-μ bispinor run
    from a missing argument would be worse than the OOM it replaced.
    """
    monkeypatch.setattr(core, "_resolve_linalg_backend", lambda *a, **k: None)
    kind = _resolve_solver_kind_transverse(
        mesh11, "auto", n_rmu_logical=_MU_TOO_BIG,
        transverse_zeta_solve="rank_truncate")
    assert kind == "transverse_rank_truncate"


def test_the_distributed_tier_escape_survives(mesh11, monkeypatch):
    """``distributed_zeta_solve='distributed'`` never allocates the buffer.

    The caller overrides the kind to
    ``distributed_transverse_rank_truncate`` on the next statement, so
    enforcing the REPLICATED capacity here would refuse a run on the size
    of a buffer it does not use.  This is the transverse twin of the
    charge branch's ``replicated_factor_used`` escape (capacity fix
    2026-07-29, ladder notes R15.1).
    """
    monkeypatch.setattr(core, "_resolve_linalg_backend", lambda *a, **k: None)
    kind = _resolve_solver_kind_transverse(
        mesh11, "auto", n_rmu_logical=_MU_TOO_BIG,
        transverse_zeta_solve="rank_truncate", nq=_NQ,
        replicated_factor_used=False)
    assert kind == "transverse_rank_truncate"


def test_the_shared_error_refuses_an_unknown_channel():
    """A new channel must add its own escape advice, not inherit charge's."""
    with pytest.raises(AssertionError):
        _rank_truncate_capacity_error(_NQ, _MU_TOO_BIG, channel="spin")
