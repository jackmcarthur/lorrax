"""``distrib_la.panel_matmul``: one scan over equal panels of a zero-padded K (CPU host meshes).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
Every panel has the one width ``w`` and each rank's ``kl = K/p`` columns are read as
``ceil(kl/w)·w``, the excess zero.  The product must equal ``a @ b`` for 1 to 6 panels,
with and without padded columns, for every operand option; and the compiled per-device
program of one CrI3 24x24 CT-pencil-sized product must have the same instruction count on
the 2x2, 4x4 and 8x8 meshes (a subprocess with 64 host devices, compile only).
"""
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)


def _mesh(side):
    if len(jax.devices()) < side * side:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:side * side]).reshape(side, side), ("x", "y"))


def _complex(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


# (side, kl, two-panel columns L): the panel count and padded columns _interleaved_width gives.
CASES = (
    (1, 5, 64),     # one panel
    (2, 4, 64),     # two panels, no pad
    (2, 7, 64),     # three panels of 3, two zero columns
    (2, 3, 64),     # three panels of 1
    (2, 11, 12),    # four panels of 3, one zero column
    (2, 11, 8),     # six panels of 2, one zero column
)
WANT = {(1, 5, 64): (1, 0), (2, 4, 64): (2, 0), (2, 7, 64): (3, 2), (2, 3, 64): (3, 0),
        (2, 11, 12): (4, 1), (2, 11, 8): (6, 1)}


def test_equal_panels_match_matmul():
    from distrib_la import panel_matmul
    from distrib_la._panel_matmul import _interleaved_width
    rng = np.random.default_rng(7)
    q, m, n = 3, 4, 6
    for side, kl, columns in CASES:
        mesh = _mesh(side)
        face = NamedSharding(mesh, P(None, "x", "y"))
        k = side * kl
        width = _interleaved_width(k, side, columns)
        n_panel = -(-kl // width)
        assert (n_panel, n_panel * width - kl) == WANT[(side, kl, columns)], (side, kl, columns, width)
        a, b = _complex(rng, q, m, k), _complex(rng, q, k, n)
        w = _complex(rng, q, k)
        lo = rng.integers(0, k, q)
        hi = np.minimum(lo + rng.integers(1, k + 1, q), k)
        live = (np.arange(k)[None, :] >= lo[:, None]) & (np.arange(k)[None, :] < hi[:, None])
        a_live = np.where(live[:, None, :], a, 0)        # the caller zeroes a outside [lo, hi)
        bounds = np.stack([lo, hi], axis=1).astype(np.int32)
        per_column = 16 * q * (m // side + n // side)
        run = lambda x, y, **kw: panel_matmul(jax.device_put(x, face), jax.device_put(y, face), mesh=mesh,
                                              panel_bytes=per_column * columns, **kw)
        checks = (
            ("plain", run(a, b), a @ b),
            ("weighted", run(a, b, weights=w), (a * w[:, None, :]) @ b),
            ("bounded", run(a_live, b, bounds=bounds), a_live @ b),
            ("transposed", run(np.conj(np.swapaxes(a, -1, -2)), np.swapaxes(b, -1, -2), transa="C",
                               transb="T"), a @ b),
        )
        partner = run(a_live, b, bounds=bounds, weights=w, partner=True)
        checks += (("green", partner[0], (a_live * w[:, None, :]) @ b),
                   ("green partner", partner[1], (np.conj(a_live) * w[:, None, :]) @ np.conj(b)))
        for name, got, want in checks:
            assert got.sharding.spec == P(None, "x", "y"), name
            np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12, atol=1e-12,
                                       err_msg=f"{name} side={side} kl={kl} L={columns}")


_FLAT = r'''
import re, sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
jax.config.update("jax_enable_x64", True)
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
from distrib_la import panel_matmul
from gw.shared_pole_execution import FACE_PANEL_DEPTH
# The CT pencil's left = matmul(tc, qt, transa="C") at CrI3 24x24 shapes (n_T 4992, F_C 18432,
# F_T 24576, stages of 4 parents), with face_matmul's panel bytes.
q, k, m, n = 4, 4992, 18432, 24576
for side in (2, 4, 8):
    mesh = Mesh(np.array(jax.devices()[:side * side]).reshape(side, side), ("x", "y"))
    face = NamedSharding(mesh, P(None, "x", "y"))
    a = jax.ShapeDtypeStruct((q, k, m), jnp.complex128, sharding=face)
    b = jax.ShapeDtypeStruct((q, k, n), jnp.complex128, sharding=face)
    per_column = 16 * q * (m // side + n // side)
    fn = lambda a, b: panel_matmul(a, b, mesh=mesh, panel_bytes=per_column * 2 * side * FACE_PANEL_DEPTH,
                                   transa="C")
    text = jax.jit(fn).lower(a, b).compile().as_text()
    ops = sum(1 for line in text.splitlines() if re.match(r"^\s*(ROOT\s+)?%[\w.\-]+\s*=", line))
    print("OPS", side, ops, text.count(" while("))
'''


def test_program_is_flat_in_the_mesh():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=64")
    out = subprocess.run([sys.executable, "-c", _FLAT], capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    rows = [tuple(int(v) for v in line.split()[1:]) for line in out.stdout.splitlines() if line.startswith("OPS")]
    assert [r[0] for r in rows] == [2, 4, 8], out.stdout
    assert len({r[1] for r in rows}) == 1 and all(r[2] == 1 for r in rows), rows
