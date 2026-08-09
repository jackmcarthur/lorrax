"""Gates for the memoised donated W-transform (``get_donated_ifftn_3d``).

``bse_lanczos`` used to build its W ifft as a fresh ``make_sharded_ifftn_3d``
factory closure wrapped in a fresh ``jax.jit`` inside ``solve_bse_sharded``.
Both objects were new on every call, so jax's dispatch cache started empty and a
byte-identical program was re-traced, re-lowered and re-probed against the
persistent compile cache every time — the same defect
``PRECOND_BUILD_FREE.md`` §3.1 removed from the exact-diagonal build and named at
§7.2, measured here at ~20-23 ms per call against ~1.0 ms of execution on the Si
4x4x4 record deck at P=4.

What a deck run cannot gate, and these cells do:

* the accessor returns THE SAME program object for the same
  ``(mesh, spec, axes, norm)``, and a different one when any of those change;
* the number of ``jax.jit`` PROGRAM CONSTRUCTIONS over N calls is 1, not N;
* donation survives the hoist — the operand's buffer is still consumed;
* the transform is bit-identical to the old inline form.

``test_construction_count_red_twin`` is the FALSE case: it rebuilds the old
inline form and asserts the construction count is N.  It is what stops the gate
above from passing for a trivial reason (e.g. someone stubbing the counter), and
it fails the moment the wrapper goes back inside the call.
"""
from __future__ import annotations

import numpy as np
import pytest

import harness  # noqa: F401  (puts src/ on sys.path)

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

jax.config.update("jax_enable_x64", True)

from common import fft_helpers as FH  # noqa: E402


NMU, NNU, NG = 4, 4, 3          # (mu, nu, gx, gy, gz) — the W_q layout
AXES = (2, 3, 4)
NORM = 'ortho'


@pytest.fixture(scope="module")
def mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1),
                axis_names=("x", "y"))


@pytest.fixture(scope="module")
def spec():
    return P("x", "y", None, None, None)


def _w_q(seed=20260808, scale=1.0):
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((NMU, NNU, NG, NG, NG))
         + 1j * rng.standard_normal((NMU, NNU, NG, NG, NG)))
    return jnp.asarray(scale * a, dtype=jnp.complex128)


def _inline_old(Wq, mesh, spec):
    """``bse_lanczos.solve_bse_sharded`` before this branch, verbatim."""
    local = FH.make_sharded_ifftn_3d(mesh, spec, spec, axes=AXES, norm=NORM)
    return jax.jit(local, donate_argnums=(0,))(Wq)


class _count_jit:
    """Count ``jax.jit`` calls — one call is one program construction.

    A wrapper built inside a function body starts with an EMPTY dispatch cache,
    so one wrapper per call is one construction per call however warm any
    compile cache is.  Counting the constructor is therefore the direct measure
    and needs no jax-internal counters.
    """

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


# ── identity of the memo ─────────────────────────────────────────────────

def test_accessor_returns_one_program_per_key(mesh, spec):
    a = FH.get_donated_ifftn_3d(mesh, spec, axes=AXES, norm=NORM)
    b = FH.get_donated_ifftn_3d(mesh, spec, axes=AXES, norm=NORM)
    assert a is b, "the same key must return the same program object"


@pytest.mark.parametrize("axes,norm,other_spec", [
    ((2, 3, 4), None, None),                       # norm differs
    ((-3, -2, -1), 'ortho', None),                 # axes differ
    ((2, 3, 4), 'ortho', P("x", None, None, None, None)),   # spec differs
])
def test_accessor_separates_keys(mesh, spec, axes, norm, other_spec):
    """Anything that changes the emitted PROGRAM must not share an entry."""
    base = FH.get_donated_ifftn_3d(mesh, spec, axes=AXES, norm=NORM)
    other = FH.get_donated_ifftn_3d(mesh, other_spec or spec,
                                    axes=axes, norm=norm)
    assert other is not base


def test_memo_does_not_key_on_shape(mesh, spec):
    """One entry serves every payload shape — shapes are jax's own business."""
    FH._DONATED_IFFTN_3D.clear()
    for n in (3, 4):
        w = jnp.asarray(np.zeros((NMU, NNU, n, n, n)), dtype=jnp.complex128)
        FH.get_donated_ifftn_3d(mesh, spec, axes=AXES, norm=NORM)(w)
    assert len(FH._DONATED_IFFTN_3D) == 1, FH._DONATED_IFFTN_3D.keys()


# ── the construction-count gate, and its red twin ────────────────────────

def test_construction_count_is_one_not_n(mesh, spec):
    FH._DONATED_IFFTN_3D.clear()
    with _count_jit() as c:
        for _ in range(4):
            out = FH.get_donated_ifftn_3d(mesh, spec, axes=AXES,
                                          norm=NORM)(_w_q())
            out.block_until_ready()
    assert c.n == 1, (
        f"{c.n} program constructions for 4 calls — the jit wrapper is being "
        f"rebuilt per call again")


def test_construction_count_red_twin(mesh, spec):
    """FALSE CASE — the old inline form must show one construction per call.

    If this ever reports 1, the counter is not measuring what the cell above
    claims it measures and that gate is worthless.
    """
    with _count_jit() as c:
        for _ in range(4):
            out = _inline_old(_w_q(), mesh, spec)
            out.block_until_ready()
    assert c.n == 4, (
        f"the inline form constructed {c.n} programs for 4 calls; the gate "
        f"above cannot be trusted")


# ── donation and value ───────────────────────────────────────────────────

def test_donation_reaches_the_compiled_program(mesh, spec):
    """The alias directive must be in the compiled module, not just requested.

    A jitted callable exposes no ``donate_argnums`` attribute, and a request
    XLA declines is silent apart from a warning — so the gate reads the
    compiled HLO.  This is the whole reason the W transform sits at a top-level
    dispatch boundary: at production ``mu = 10015 / P = 64`` the un-aliased form
    costs 2 x 404 MB per rank.
    """
    f = FH.get_donated_ifftn_3d(mesh, spec, axes=AXES, norm=NORM)
    txt = f.lower(_w_q()).compile().as_text()
    assert "input_output_alias" in txt, (
        "the compiled W transform declares no input/output alias, so W_R does "
        "not land on W_q's buffer and the run carries two copies")


def test_donated_operand_is_consumed(mesh, spec):
    w = _w_q()
    out = FH.get_donated_ifftn_3d(mesh, spec, axes=AXES, norm=NORM)(w)
    out.block_until_ready()
    assert w.is_deleted(), (
        "donation was declined: W_q survived the call, so the run carries two "
        "copies of an N_mu^2-class tile")


def test_hoisted_matches_the_inline_form_bit_for_bit(mesh, spec):
    w1, w2 = _w_q(), _w_q()
    ref = np.asarray(_inline_old(w1, mesh, spec))
    new = np.asarray(
        FH.get_donated_ifftn_3d(mesh, spec, axes=AXES, norm=NORM)(w2))
    assert np.array_equal(ref, new), (
        f"max|Δ| = {np.max(np.abs(ref - new)):.3e}")


def test_matches_numpy_ifftn(mesh, spec):
    """The value itself, against numpy — the hoist must not move the norm."""
    w = _w_q()
    ref = np.fft.ifftn(np.asarray(w), axes=AXES, norm=NORM)
    out = np.asarray(
        FH.get_donated_ifftn_3d(mesh, spec, axes=AXES, norm=NORM)(w))
    assert np.allclose(ref, out, rtol=0, atol=1e-13)


def test_bse_lanczos_has_no_inline_jit_left():
    """The call site must go through the accessor, by source inspection.

    A value gate cannot see a REGRESSION here: re-inlining the wrapper keeps
    every number identical and only puts the ~20 ms back.  This cell is what
    makes that regression visible.
    """
    import inspect
    from bse import bse_lanczos
    src = inspect.getsource(bse_lanczos.solve_bse_sharded)
    body = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "get_donated_ifftn_3d(" in body
    assert "jax.jit(" not in body or "donate_argnums" not in body, (
        "solve_bse_sharded builds a donating jit in its own body again; use "
        "common.fft_helpers.get_donated_ifftn_3d")
