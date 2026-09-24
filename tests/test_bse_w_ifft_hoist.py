"""Gates for the memoised donated W-transform (``get_donated_kfft_kminor``).

``bse_lanczos`` used to build its W ifft as a fresh factory closure wrapped in
a fresh ``jax.jit`` inside ``solve_bse_sharded``.  Both objects were new on
every call, so jax's dispatch cache started empty and a byte-identical program
was re-traced, re-lowered and re-probed against the persistent compile cache
every time — measured at ~20-23 ms per call against ~1.0 ms of execution on the
Si 4x4x4 record deck at P=4 (``PRECOND_BUILD_FREE.md`` §7.2).

The transform itself is the k-convolution router's k-minor door
(``make_kfft_kminor``; nvidia-mathdx on CUDA, the plan route on cpu — here the
announced cpu test arm).  What a deck run cannot gate, and these cells do:

* the accessor returns THE SAME program object for the same key, and a
  different one when the key changes;
* the number of ``jax.jit`` PROGRAM CONSTRUCTIONS over N calls is 1, not N
  (red twin: the inline form constructs N);
* donation survives the hoist — the operand's buffer is still consumed;
* the value is numpy's ``ifftn``.
"""
from __future__ import annotations

import numpy as np
import pytest

import harness  # noqa: F401  (puts src/ on sys.path)

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402

jax.config.update("jax_enable_x64", True)

from common import fft_helpers as FH  # noqa: E402


NMU, NNU, NG = 4, 4, 3          # (mu, nu, gx, gy, gz) — the W_q layout
KG = (NG, NG, NG)
AXES = (2, 3, 4)
NORM = 'ortho'


@pytest.fixture(scope="module")
def mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1),
                axis_names=("x", "y"))


@pytest.fixture(scope="module")
def spec():
    return P("x", "y", None, None, None)


def _w_q(seed=20260808):
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((NMU, NNU, NG, NG, NG))
         + 1j * rng.standard_normal((NMU, NNU, NG, NG, NG)))
    return jnp.asarray(a, dtype=jnp.complex128)


def _get(mesh, spec, **kw):
    return FH.get_donated_kfft_kminor(mesh, kw.pop("kgrid", KG), spec,
                                      kind=kw.pop("kind", "ifftn"), norm=kw.pop("norm", NORM))


class _count_jit:
    """Count ``jax.jit`` calls — one call is one program construction."""

    def __enter__(self):
        self.n = 0
        self._orig = jax.jit

        def counting(*a, **k):
            self.n += 1
            return self._orig(*a, **k)

        jax.jit = counting
        return self

    def __exit__(self, *exc):
        jax.jit = self._orig
        return False


def test_accessor_returns_one_program_per_key(mesh, spec):
    assert _get(mesh, spec) is _get(mesh, spec)
    assert _get(mesh, spec, norm=None) is not _get(mesh, spec)
    assert _get(mesh, P("x", None, None, None, None)) is not _get(mesh, spec)


def test_construction_count_is_one_not_n(mesh, spec):
    FH._DONATED_KFFT_KMINOR.clear()
    with _count_jit() as c:
        for _ in range(4):
            _get(mesh, spec)(_w_q()).block_until_ready()
    assert c.n == 1, f"{c.n} program constructions for 4 calls"


def test_construction_count_red_twin(mesh, spec):
    """FALSE CASE — the old inline form must show one construction per call."""
    with _count_jit() as c:
        for _ in range(4):
            f = FH.make_kfft_kminor(mesh, KG, spec, kind="ifftn", norm=NORM)
            jax.jit(f, donate_argnums=(0,))(_w_q()).block_until_ready()
    assert c.n == 4, f"the inline form constructed {c.n} programs for 4 calls"


def test_donated_operand_is_consumed_and_value_is_numpy(mesh, spec):
    w = _w_q()
    ref = np.fft.ifftn(np.asarray(w), axes=AXES, norm=NORM)
    out = _get(mesh, spec)(w)
    out.block_until_ready()
    assert w.is_deleted(), "donation was declined: W_q survived the call"
    assert np.allclose(ref, np.asarray(out), rtol=0, atol=1e-13)


def test_bse_lanczos_has_no_inline_jit_left():
    """The call site must go through the accessor, by source inspection."""
    import inspect
    from bse import bse_lanczos
    src = inspect.getsource(bse_lanczos.solve_bse_sharded)
    body = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "get_donated_kfft_kminor(" in body
    assert "jax.jit(" not in body or "donate_argnums" not in body
