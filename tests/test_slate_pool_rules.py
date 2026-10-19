"""The SLATE pool seed's outside allowance is the memory model's; eigh's workspace rule.

The device SLATE handlers seed each rank's pool (cpp/slate/pool_seed.h ScratchSeed: that
rank's bytes from XLA's scratch allocator; eigh: a workspace output) at the measured lower envelope of the pool's
high-water -- rules in cpp/slate/{eigh,potrf,batched_solve_lu}_ffi.cc, measured with
pool_seed.h PoolWatch (MI250X 2x2, n = 4096 ... 30720; DISTRIB_LA_SCALE_REPORT addendum).
potrf seeds a rank's reservation only past ``seed_min_bytes``, which distrib_la passes from
``lxkit.legs.DISTRIBUTED_CONTEXT_BYTES``, the memory model's distributed-linalg outside
allowance.
"""
import pytest

pytest.importorskip("jax")

from distrib_la import _slate  # noqa: E402


def test_eigh_workspace_is_the_per_rank_rule():
    """A: max(lower, full - mt/2); Z: local; Z1d: mt x this rank's 1 x P columns --
    equal on every rank of 2x2, so the uniform workspace holds no rank's unused bytes."""
    nb, it = 128, 16
    for n in (4096, 8192, 16384, 20480, 30720):
        mt = n // nb
        local = (mt // 2) ** 2
        want = (local - mt // 2) + local + mt * (mt // 4)
        assert _slate.eigh_workspace_bytes(n, nb, 2, 2, it) == want * nb * nb * it
        assert _slate.eigh_workspace_bytes(n, nb, 2, 2, it, False) == (local - mt // 2) * nb * nb * it
