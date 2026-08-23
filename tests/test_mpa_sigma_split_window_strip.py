"""Real multi-rank CUDA gate for the MPA split-Sigma-window fix
(``gw.mpa.sigma._integrate_sigma_batches``, 2026-08-23).

Companion to ``tests/test_sigma_window_pad.py`` (the shape-only, single-
device unit tests for ``pad_sigma_window``/``strip_sigma_window``) and to
``tests/test_pack_band_window.py`` (the analogous INPUT-side repack this
fix's mechanism is reused from). Neither of those exercises a genuinely
mesh-sharded ``P(...,'x','y')`` array through ``strip_sigma_window`` --
every array they build lives on ONE device (``jnp.asarray`` with no
explicit sharding), so the new device-array arm added here
(:func:`gw.ppm_sigma._strip_sharded_sigma_window_kernel`) has no real
coverage until this file runs on real multi-process CUDA.

Named gap this closes: ``gw.mpa.sigma._integrate_sigma_batches``
``layout='face'`` used to refuse outright whenever ``nb_sigma !=
nb_full`` (a split Sigma window -- the common case: a deck usually loads
more bands for chi0/screening than it evaluates Sigma over).  The fix
keeps the accumulator at ``nb_full`` (the input side is untouched --
``contract_bands``'s face GEMM plans are fixed at that width, see
``mpa/sigma.py``'s own updated comment for why narrowing the INPUT was
the wrong lever) and instead repacks the OUTPUT ``Sigma_c(omega,k,m,n)``
down to ``nb_sigma`` post-hoc, reusing
``wavefunction_bundle.pack_band_window``'s OWN mechanism
(``jax.lax.slice_in_dim`` + ``jax.lax.with_sharding_constraint``) rather
than a numpy-style slice (illegal on a live mesh-sharded axis whenever
the target extent does not itself divide the mesh).

    lx run -N 1 -G 4 -n 4 bash <wrapper.sh> \\
        tests/test_mpa_sigma_split_window_strip.py --mesh 2x2

Under plain pytest (one process) every case SKIPS rather than failing --
it names exactly why (process_count), matching every sibling face-layout
gate in this directory (TASTE.md, "a check that cannot fail is not
evidence").

Cases
-----
* ``strip_divisible_window`` -- a genuine 2x2-mesh-sharded
  ``(n_omega, nk, nb_full, nb_full)`` complex128 array, stripped to a
  SMALLER ``nb_real`` that itself divides the mesh (the achievable case
  ``_integrate_sigma_batches`` now serves): the mesh-aware kernel's
  result must equal a HOST reference (gather the full array, then plain
  numpy-slice it -- ``strip_sigma_window``'s own host/numpy arm, exercised
  on the SAME data as an independent oracle) bit-exactly (pure
  data movement -- a slice + a resharding collective, no arithmetic).
  Checked at TWO nb_real values on the same input, including nb_real ==
  the mesh size itself (a degenerate 1-per-rank window).
* ``strip_full_window_is_identity`` -- ``nb_real == nb_full`` returns the
  VERY SAME array object (the no-op fast path both layouts already relied
  on) -- checked by identity, not just value equality; confirms this fix
  did not disturb the unsplit case any existing gate already covers.
* ``strip_indivisible_window_refuses`` -- ``nb_real`` that does NOT
  divide the mesh: ``assert_sharded_sigma_window_divides_mesh`` (the SAME
  shared owner the legacy branch already calls) refuses BEFORE any
  attempt to build the illegal sharding -- the "genuinely impossible
  sub-case" the fix keeps refusing, by name, rather than reworking
  around it.
* ``strip_missing_mesh_refuses`` -- a live mesh-sharded array reaching
  ``strip_sigma_window`` with ``mesh_xy=None`` refuses loudly (the
  defensive backstop for a future caller that forgets it) rather than
  mis-indexing -- process-count-gated for the same reason as the others
  (building the P('x','y') array at all needs real ranks).
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

from gw.ppm_sigma import (assert_sharded_sigma_window_divides_mesh,
                          strip_sigma_window)

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
    assert_sharded_sigma_window_divides_mesh(
        nb_real, mesh, ansatz="compute_mode = mpa")
    got = jax.block_until_ready(
        strip_sigma_window(sigma, nb_real, mesh_xy=mesh))
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
    out = strip_sigma_window(sigma, nb_full, mesh_xy=mesh)
    assert out is sigma, "nb_real == nb_full must be a true no-op"


def check_strip_indivisible_window_refuses(mesh, *, nb_full, nb_real):
    sigma, _ = _sharded_sigma(mesh, n_omega=2, nk=1, nb_full=nb_full,
                              seed=7)
    try:
        assert_sharded_sigma_window_divides_mesh(
            nb_real, mesh, ansatz="compute_mode = mpa")
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"expected assert_sharded_sigma_window_divides_mesh to refuse "
            f"nb_real={nb_real} on a {PX}x{PY} mesh")
    # strip_sigma_window itself does not re-derive divisibility (the
    # caller's assert is the single owner) -- what it must do on an
    # indivisible target via the mesh-aware arm is either raise or produce
    # something that is NOT silently wrong; the production caller never
    # reaches this call because it asserts first (mirrored above), so this
    # only confirms the guard fires before mpa/sigma.py would ever try.
    del sigma


def check_strip_missing_mesh_refuses(mesh, *, nb_full, nb_real):
    sigma, _ = _sharded_sigma(mesh, n_omega=2, nk=1, nb_full=nb_full,
                              seed=11)
    try:
        strip_sigma_window(sigma, nb_real)
    except ValueError as exc:
        assert "mesh_xy" in str(exc), exc
    else:
        raise AssertionError(
            "expected strip_sigma_window to refuse a mesh-sharded array "
            "with mesh_xy=None rather than mis-index it")


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


def test_strip_indivisible_window_refuses():
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} REAL processes; "
            f"got process_count={jax.process_count()}")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_strip_indivisible_window_refuses(mesh, nb_full=8, nb_real=5)


def test_strip_missing_mesh_refuses():
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} REAL processes; "
            f"got process_count={jax.process_count()}")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_strip_missing_mesh_refuses(mesh, nb_full=8, nb_real=4)


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
        ("strip_indivisible_window_refuses",
         check_strip_indivisible_window_refuses,
         dict(nb_full=8, nb_real=5)),
        ("strip_missing_mesh_refuses",
         check_strip_missing_mesh_refuses, dict(nb_full=8, nb_real=4)),
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
