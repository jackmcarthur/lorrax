"""``face_to_batch_reshard(route=…)`` — the TWO exchange schedules.

``tests/test_staged_reshard.py`` gates the primitive; this file gates the
second route the owner proposed (2026-07-31)::

    q, mu_X, nu_Y  →  q, mu_XY, nu  →  q_XY, mu, nu     "flatten_m_first"

against the shipped one::

    q, mu_X, nu_Y  →  q_X, mu, nu_Y  →  q_XY, mu, nu    "split_b_first"

The claim being gated is ELEMENT IDENTITY: two different collective
schedules, the same output array, bit for bit — including the case where
``flatten_m_first`` has to pad M locally because ``M/p_x`` is not a
multiple of ``p_y`` (the MoS2 4x4 exciton deck at P=64: M = 672,
672/8 = 84, 84 % 8 = 4).  Speed is not gated here; it is measured on
hardware (``wk_REL/qchunk``).

Run with 4 host devices::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 pytest -q \
        tests/test_staged_reshard_routes.py
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.staged_reshard import (
    ROUTES,
    DEFAULT_ROUTE,
    face_to_batch_reshard,
    face_to_batch_reshard_supported,
)

# FOUR DEVICES.  Was a ``skipif(jax.device_count() < 4)``, which skipped in
# every suite run regardless of the node — tests/conftest.py pins each test
# process to one GPU, so ``device_count()`` is 1 by construction.  The marker
# states the requirement and the conftest supplies it; see the mesh section
# there.
pytestmark = pytest.mark.mesh(4)

FACE = P(None, 'x', 'y')
BATCH = P(('x', 'y'), None, None)


def _mesh(px=2, py=2, names=('x', 'y')):
    devs = np.array(jax.devices()[:px * py]).reshape(px, py)
    return Mesh(devs, names)


def _probe(b, m, n):
    """Values that encode their own (batch, row, col) index, so a wrong
    block map is a WRONG VALUE and not merely a wrong shape."""
    idx = np.arange(b * m * n, dtype=np.float64).reshape(b, m, n)
    return jnp.asarray(idx + 1j * (idx * 0.5 + 1.0))


def _unstaged(mesh):
    face, batch = NamedSharding(mesh, FACE), NamedSharding(mesh, BATCH)

    @jax.jit
    def f(a):
        a = jax.lax.with_sharding_constraint(a, face)
        return jax.lax.with_sharding_constraint(a, batch)
    return f


# ---------------------------------------------------------------------------
#  1. both routes reproduce the input, and each other, exactly
# ---------------------------------------------------------------------------

# (B, M, N) with M/p_x DIVISIBLE by p_y (no pad) and, second, NOT divisible
# (pad path).  On a 2x2 mesh: M=12 -> m_loc 6, 6%2==0 (no pad);
# M=10 -> m_loc 5, 5%2==1 (pad to 6, global 12, +20%).
_SHAPES = [(8, 12, 20), (8, 10, 20), (4, 6, 6)]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_route_returns_the_input_reassembled(route, shape):
    b, m, n = shape
    mesh = _mesh()
    a = jax.device_put(_probe(b, m, n), NamedSharding(mesh, FACE))
    out = jax.jit(face_to_batch_reshard(mesh, route=route))(a)
    spec = tuple(out.sharding.spec) + (None,) * (3 - len(out.sharding.spec))
    assert spec == (('x', 'y'), None, None), out.sharding.spec
    np.testing.assert_array_equal(np.asarray(out),
                                  np.asarray(_probe(b, m, n)))


@pytest.mark.parametrize("shape", _SHAPES)
def test_the_two_routes_are_element_identical(shape):
    """THE claim.  Different schedules, same bytes — so a route swap is a
    movement-only change and needs no physics review."""
    b, m, n = shape
    mesh = _mesh()
    a = jax.device_put(_probe(b, m, n), NamedSharding(mesh, FACE))
    got = {r: np.asarray(jax.jit(face_to_batch_reshard(mesh, route=r))(a))
           for r in ROUTES}
    ref = np.asarray(_unstaged(mesh)(a))
    for r, v in got.items():
        np.testing.assert_array_equal(v, ref, err_msg=f"route={r}")
    np.testing.assert_array_equal(got["split_b_first"],
                                  got["flatten_m_first"])


def test_route_comparison_detects_a_wrong_block_map():
    """RED TWIN for the identity claim.

    ``flatten_m_first``'s correctness rests on ONE unverified-by-eye
    assumption: that ``all_to_all`` over a TUPLE axis ``('x','y')`` orders
    its peers x-major, the same way ``P(('x','y'), …)`` numbers blocks.
    Build the y-major variant of exactly that stage and require the
    comparison to reject it — otherwise the passes above would also pass
    for a route that shuffles the batch.
    """
    from common.shard_map import shard_map
    b, m, n = 8, 12, 20
    mesh = _mesh()

    def _body_ymajor(t):
        t = jax.lax.all_to_all(t, 'y', split_axis=1, concat_axis=2, tiled=True)
        t = jax.lax.all_to_all(t, ('y', 'x'), split_axis=0, concat_axis=1,
                               tiled=True)
        return t

    bad = jax.jit(shard_map(_body_ymajor, mesh=mesh, in_specs=(FACE,),
                            out_specs=BATCH, check_vma=False))
    a = jax.device_put(_probe(b, m, n), NamedSharding(mesh, FACE))
    ref = np.asarray(_unstaged(mesh)(a))
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(np.asarray(bad(a)), ref)


# ---------------------------------------------------------------------------
#  2. the pad that flatten_m_first needs, and its cost
# ---------------------------------------------------------------------------

def test_flatten_m_first_pads_when_m_loc_does_not_divide_py_and_says_so():
    """M = 10 on a 2x2 mesh: m_loc = 5, 5 % 2 = 1, so the local tile is
    padded to 6 (global 12, +20 %).  The pad is a real cost and must be
    ANNOUNCED — a silent 20 % on the (rank, rank) face is exactly the kind
    of thing that makes an A/B unreadable."""
    lines = []
    mesh = _mesh()
    a = jax.device_put(_probe(8, 10, 20), NamedSharding(mesh, FACE))
    f = face_to_batch_reshard(mesh, route="flatten_m_first",
                              log_fn=lines.append)
    out = np.asarray(jax.jit(f)(a))
    np.testing.assert_array_equal(out, np.asarray(_probe(8, 10, 20)))
    banner = " ".join(lines)
    assert "flatten_m_first" in banner and "zero-padding" in banner, banner
    assert "+20.00%" in banner, banner


def test_split_b_first_never_pads():
    lines = []
    mesh = _mesh()
    a = jax.device_put(_probe(8, 10, 20), NamedSharding(mesh, FACE))
    jax.jit(face_to_batch_reshard(mesh, route="split_b_first",
                                  log_fn=lines.append))(a)
    assert not any("zero-padding" in ln for ln in lines), lines


def test_the_deck_geometry_is_the_pad_case():
    """The number the recommendation turns on: at the MoS2 4x4 exciton
    deck's M = rank = 672 on an 8x8 mesh, ``flatten_m_first`` pads and
    ``split_b_first`` does not.  Kept as arithmetic so the claim in the
    report is checkable without the machine."""
    M, px, py = 672, 8, 8
    assert M % px == 0                       # both routes accept the input
    assert M % (px * py) != 0                # flatten_m_first's extra rule
    m_loc = M // px
    m_pad = -(-m_loc // py) * py
    assert (m_loc, m_pad) == (84, 88)
    assert m_pad * px == 704
    assert abs(100.0 * (704 - 672) / 672 - 4.7619) < 1e-3


# ---------------------------------------------------------------------------
#  3. vocabulary and support predicate
# ---------------------------------------------------------------------------

def test_default_route_is_the_shipped_one():
    assert DEFAULT_ROUTE == "split_b_first"
    assert set(ROUTES) == {"split_b_first", "flatten_m_first"}


def test_unknown_route_is_refused_not_defaulted():
    mesh = _mesh()
    with pytest.raises(ValueError, match="route="):
        face_to_batch_reshard(mesh, route="flatten_n_first")
    assert not face_to_batch_reshard_supported(mesh, (8, 12, 20),
                                               route="nonsense")


def test_resolve_reshard_route_env_grammar(monkeypatch):
    """An unrecognised env token must ANNOUNCE and fall back — a typo that
    silently ran the default is how an A/B comes back 'no difference'."""
    from bandstructure.bse_setup import resolve_reshard_route
    monkeypatch.delenv("LORRAX_FACE_TO_BATCH_ROUTE", raising=False)
    assert resolve_reshard_route() == DEFAULT_ROUTE
    monkeypatch.setenv("LORRAX_FACE_TO_BATCH_ROUTE", "  Flatten_M_First ")
    assert resolve_reshard_route() == "flatten_m_first"
    said = []
    monkeypatch.setenv("LORRAX_FACE_TO_BATCH_ROUTE", "flatten")
    assert resolve_reshard_route(log_fn=said.append) == DEFAULT_ROUTE
    assert any("LORRAX SANITY" in s for s in said), said
    # an explicit argument beats the env, and a bad one raises
    monkeypatch.setenv("LORRAX_FACE_TO_BATCH_ROUTE", "flatten_m_first")
    assert resolve_reshard_route("split_b_first") == "split_b_first"
    with pytest.raises(ValueError):
        resolve_reshard_route("bogus")


# ---------------------------------------------------------------------------
#  4. residency: neither route adds a live shard-class intermediate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ROUTES)
def test_every_rank_holds_exactly_one_shard_of_the_output(route):
    b, m, n = 8, 12, 20
    mesh = _mesh()
    a = jax.device_put(_probe(b, m, n), NamedSharding(mesh, FACE))
    out = jax.jit(face_to_batch_reshard(mesh, route=route))(a)
    per_rank = {s.data.shape for s in out.addressable_shards}
    assert per_rank == {(b // 4, m, n)}, per_rank
    # ONE process owns all four devices here, so ``addressable_shards`` lists
    # every shard and their bytes sum to the WHOLE array — the per-DEVICE
    # residency is one shard, which is what the shape assertion above says.
    # (The first version of this cell summed the four and compared against
    # one; it failed green-looking on a correct implementation.)
    assert len(out.addressable_shards) == 4
    assert {s.data.nbytes for s in out.addressable_shards} == \
        {b * m * n * 16 // 4}
