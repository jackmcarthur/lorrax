"""The SLATE pool seed's outside allowance is the memory model's.

The device SLATE handlers seed each rank's pool (cpp/slate/pool_seed.h ScratchSeed: that
rank's bytes from XLA's scratch allocator) at the measured lower envelope of the pool's
high-water -- rules in cpp/slate/{eigh,potrf,batched_solve_lu}_ffi.cc, measured with
pool_seed.h PoolWatch (MI250X 2x2, n = 4096 ... 30720; DISTRIB_LA_SCALE_REPORT addendum).
potrf seeds a rank's reservation only past ``seed_min_bytes``, which distrib_la passes from
``_slate._SLATE_OUTSIDE_ALLOWANCE_BYTES``; that constant must be the runtime memory
model's distributed-linalg outside allowance (distrib_la cannot import ``runtime``).
"""
import pytest

pytest.importorskip("jax")

from distrib_la import _slate  # noqa: E402


def test_allowance_is_the_memory_models():
    runtime = pytest.importorskip("runtime")
    assert _slate._SLATE_OUTSIDE_ALLOWANCE_BYTES == runtime.DISTRIBUTED_LINALG_OUTSIDE_BYTES
