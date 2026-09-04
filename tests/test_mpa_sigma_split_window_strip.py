"""Real P4 gate for the runtime-owned Sigma band carrier.

The dynamic producer publishes only a square ``P(..., 'x', 'y')`` carrier.
An indivisible logical window therefore stays live as a divisible carrier
whose tail rows and columns are exact zero.  Only a replicated/host output
consumer strips that carrier.  These cells cover both the old divisible
strip seam and the formerly-refused 5 -> 6 carrier on a real 2x2 mesh.
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    _TESTS = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.dirname(_TESTS)
    for _svc in ("lxkit", "distrib_la"):
        _src = os.path.join(_REPO, "services", _svc, "src")
        if os.path.isdir(_src) and _src not in sys.path:
            sys.path.insert(0, _src)
    from lxkit.gate import platform_from_env
    from runtime import initialize_communicator_stack
    _plat = platform_from_env()
    _RUNTIME = initialize_communicator_stack(
        platform="gpu" if _plat == "CUDA" else "cpu")

import argparse

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.ppm_sigma import sigma_band_axis, strip_sigma_window
from runtime.padding import PaddedAxis, pad_square, strip_axis

PX = PY = 2


def _to_host(x):
    from jax.experimental import multihost_utils as mhu
    return np.asarray(mhu.process_allgather(x, tiled=True))


def _sharded_sigma(mesh, *, n_omega, nk, nb_full, seed):
    """A genuine P(None,None,'x','y')-sharded (n_omega,nk,nb_full,nb_full)
    complex128 array with distinct, reproducible values -- built the SAME
    way ``DeviceOmegaAccumulator.finalize()`` produces its own output
    (device_put onto the exact spec MPA's accumulator uses), so this test
    exercises the real seam rather than a stand-in shape."""
    rng = np.random.default_rng(seed)
    full = (rng.standard_normal((n_omega, nk, nb_full, nb_full))
            + 1j * rng.standard_normal((n_omega, nk, nb_full, nb_full)))
    sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    return jax.device_put(jnp.asarray(full), sharding), full


def check_strip_divisible_window(mesh, *, nb_full, nb_real, seed):
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    sigma, full = _sharded_sigma(mesh, n_omega=3, nk=2, nb_full=nb_full,
                                 seed=seed)
    tag = PaddedAxis(
        name="test Sigma band window", logical=nb_real,
        carrier=nb_full, divisor=PX)
    got = jax.block_until_ready(
        strip_sigma_window(sigma, tag, mesh_xy=mesh))
    got_h = _to_host(got)
    want_h = full[..., :nb_real, :nb_real]
    assert got_h.shape == want_h.shape, (got_h.shape, want_h.shape)
    # Pure data movement (slice + resharding collective, no arithmetic):
    # bit-exact against the host/numpy reference, not merely close.
    np.testing.assert_array_equal(got_h, want_h)
    p0(f"  nb_full={nb_full} nb_real={nb_real}: strip bit-exact "
       f"({got_h.size} elements)")


def check_strip_full_window_is_identity(mesh, *, nb_full):
    sigma, _ = _sharded_sigma(mesh, n_omega=2, nk=1, nb_full=nb_full,
                              seed=99)
    tag = PaddedAxis(
        name="test Sigma band window", logical=nb_full,
        carrier=nb_full, divisor=PX)
    out = strip_sigma_window(sigma, tag, mesh_xy=mesh)
    assert out is sigma, "nb_real == nb_full must be a true no-op"


def check_indivisible_window_uses_zero_carrier(mesh, *, nb_real):
    """The old refusal case is a live divisible carrier with inert tails."""
    tag = sigma_band_axis(nb_real, mesh, ansatz="compute_mode = mpa")
    assert tag.logical == nb_real
    assert tag.carrier == 6
    rng = np.random.default_rng(7)
    logical = (rng.standard_normal((2, 1, nb_real, nb_real))
               + 1j * rng.standard_normal((2, 1, nb_real, nb_real)))
    carrier = np.asarray(pad_square(jnp.asarray(logical), tag))
    np.testing.assert_array_equal(
        carrier[..., :nb_real, :nb_real], logical)
    np.testing.assert_array_equal(carrier[..., nb_real:, :], 0)
    np.testing.assert_array_equal(carrier[..., :, nb_real:], 0)
    sigma = jax.device_put(
        jnp.asarray(carrier),
        NamedSharding(mesh, P(None, None, "x", "y")))
    # The sharded cube remains at carrier width.  The public diagonal gather
    # is the replicated consumer boundary, then the owner strips by receipt.
    from gw.qsgw_utils import extract_sigma_diag_replicated
    got = np.asarray(extract_sigma_diag_replicated(sigma, mesh))
    got = np.asarray(strip_axis(got, tag, axis=-1))
    np.testing.assert_array_equal(got, np.diagonal(logical, axis1=-2, axis2=-1))


_DIVISIBLE_CASES = (
    ("strip_divisible_window_half", dict(nb_full=8, nb_real=4, seed=201)),
    ("strip_divisible_window_min", dict(nb_full=8, nb_real=2, seed=202)),
)


@pytest.mark.parametrize("name,kwargs", _DIVISIBLE_CASES,
                         ids=[c[0] for c in _DIVISIBLE_CASES])
def test_strip_divisible_window(name, kwargs):
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} REAL processes (a live P('x','y')-sharded "
            f"array); got process_count={jax.process_count()}")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_strip_divisible_window(mesh, **kwargs)


def test_strip_full_window_is_identity():
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} REAL processes; "
            f"got process_count={jax.process_count()}")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_strip_full_window_is_identity(mesh, nb_full=6)


def test_indivisible_window_uses_zero_carrier():
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} REAL processes; "
            f"got process_count={jax.process_count()}")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_indivisible_window_uses_zero_carrier(mesh, nb_real=5)


def _cli_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="2x2", help="PxQ process mesh")
    args = ap.parse_args()
    px, py = (int(v) for v in args.mesh.lower().split("x"))
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()}")
    if jax.device_count() != px * py:
        p0(f"REFUSE: need exactly {px * py} devices for a {args.mesh} mesh; "
           f"got {jax.device_count()}")
        return 1
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))
    failures = 0
    total = 0
    for name, kwargs in _DIVISIBLE_CASES:
        total += 1
        try:
            check_strip_divisible_window(mesh, **kwargs)
            p0(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {name}: {exc}")
    for name, fn, kwargs in (
        ("strip_full_window_is_identity",
         check_strip_full_window_is_identity, dict(nb_full=6)),
        ("indivisible_window_uses_zero_carrier",
         check_indivisible_window_uses_zero_carrier,
         dict(nb_real=5)),
    ):
        total += 1
        try:
            fn(mesh, **kwargs)
            p0(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {name}: {exc}")
    p0(f"done: {total - failures}/{total} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_cli_main)
