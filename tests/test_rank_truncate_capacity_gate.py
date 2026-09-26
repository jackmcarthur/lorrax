"""The replicated-eigh capacity gate of the charge ζ factor.

``charge_zeta_solve='rank_truncate'`` allocates one replicated
``(q_batch, n_mu, n_mu)`` complex128 eigh operand.  The resolver refuses
when one q-batch of it exceeds the per-batch cap, naming the ceiling
``n_mu <= sqrt(cap/16)``.

Pure host: no mesh, no GPU, no FFI.  SCOPE: a RESOLVER contract test.  It
does not run a ζ fit.
"""
import math

import pytest

import isdf.core as core
from isdf.core import (
    _rank_truncate_capacity_error,
    _replicate_rank_truncate_ok,
    _resolve_solver_kind,
)


def _mu_ceiling() -> int:
    cap = max(core._REPLICATED_CHOL_MAX_STACK_BYTES,
              core._REPLICATED_FACTOR_MAX_BATCH_BYTES)
    return int(math.isqrt(cap // 16))


# A μ comfortably past the ceiling and one comfortably under it, derived
# from the caps so a cap change moves the test instead of exempting it.
_MU_TOO_BIG = _mu_ceiling() * 2
_MU_FINE = 512
_NQ = 8


def test_the_predicate_decides_fit_or_refuse():
    assert _replicate_rank_truncate_ok(_NQ, _MU_FINE) is True
    assert _replicate_rank_truncate_ok(_NQ, _MU_TOO_BIG) is False
    # Unknown inputs are not "fits"; they are "do not decide here".
    assert _replicate_rank_truncate_ok(None, _MU_FINE) is False
    assert _replicate_rank_truncate_ok(_NQ, None) is False


def test_the_charge_resolver_refuses_one_oversized_q_batch():
    with pytest.raises(ValueError) as ch:
        _resolve_solver_kind(0, "auto", n_rmu=_MU_TOO_BIG, nq=_NQ)
    msg = str(ch.value)
    assert "charge_zeta_solve='rank_truncate'" in msg
    assert f"n_mu={_MU_TOO_BIG}" in msg
    assert f"n_mu <= {_mu_ceiling()}" in msg
    assert "ONE q-batch, not the stack" in msg


def test_the_shared_error_refuses_an_unknown_channel():
    """A new channel must add its own advice, not inherit charge's."""
    with pytest.raises(AssertionError):
        _rank_truncate_capacity_error(_NQ, _MU_TOO_BIG, channel="spin")
