"""The distributed eigh's layout, its one padding mechanism and the deflation contract (CPU, seconds).

cuSOLVERMp runs at the block ``distrib_la._cusolvermp.solve_layout`` picks: the
largest divisor of n/p up to 256, or, where that is below 128 (n/p prime or a
small multiple of one), the smallest padded tile with a divisor in [128, 256].
The plan zero-pads an operand to the backend's side (``pad_side``), deflates
every zero row (``deflate_zero_rows``: the deflated pairs come FIRST, the live
ones ascending after them) and drops the pads, which lead (head padding), by count (``drop_side``); the
wrappers refuse an operand that is not a solve side. ``jnp.linalg.eigh``
stands in for the library.
"""
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from distrib_la import _elpa, _slate  # noqa: E402
from distrib_la._cusolvermp import _BLOCK_MAX, _BLOCK_MIN, _largest_block, solve_layout  # noqa: E402
from distrib_la._result_check import deflate_zero_rows, drop_side, pad_side  # noqa: E402
from distrib_la.plan import Plan, _eigh_columns, _eigh_side  # noqa: E402


@pytest.mark.parametrize("n,p,want", [
    (7908, 2, (7912, 172)),   # CrI3 6x6 TT: 3954 = 2*3*659, block 6 before
    (7902, 2, (7904, 247)),   # 2634 * 3: 3951 = 3^2*439, block 9
    (7894, 2, (7896, 188)),   # n/p = 3947 prime, block 1
    (3960, 2, (3960, 220)),   # the 1316-point TT side: block 220, unpadded
    (16800, 2, (16800, 240)),
    (512, 2, (512, 256)),
    (778, 4, (778, 194)),     # n/p <= 256: one tile per rank, as before
])
def test_solve_layout(n, p, want):
    assert solve_layout(n, p) == want


@pytest.mark.parametrize("p", [2, 4, 8])
def test_every_padded_tile_has_a_block_in_range_and_every_side_is_a_fixed_point(p):
    """The wrappers re-derive the side from the padded operand: each layout must map a side to itself."""
    for local in range(257, 9000, 7):
        side, block = solve_layout(p * local, p)
        tile = side // p
        assert tile % block == 0 and side % p == 0
        assert _BLOCK_MIN <= block <= _BLOCK_MAX
        if side > p * local:
            assert _largest_block(local) < _BLOCK_MIN and tile - local < _BLOCK_MIN
        for backend in ("cusolvermp", "slate", "elpa"):
            for dtype in (np.float64, np.complex128):
                s = _eigh_side(backend, p * local, p, dtype)
                assert s >= p * local and s % p == 0
                assert _eigh_side(backend, s, p, dtype) == s


def _hermitian(n, rank, seed):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((rank, n)) + 1j * rng.standard_normal((rank, n))
    return b.conj().T @ b / n


def _with_zero_rows(n, live, seed, shift):
    a = np.zeros((n, n), complex)
    rows = np.sort(np.random.default_rng(seed).permutation(n)[:live])
    a[np.ix_(rows, rows)] = _hermitian(live, live // 3, seed) - shift * np.eye(live)
    return a


def test_deflation_contract_deflated_first_live_ascending():
    """Zero rows come out first as (0, unit vector); a negative live value sits after them."""
    n = 48
    a = _with_zero_rows(n, 40, 2, 0.05)                 # negative live values and 8 zero rows
    w, v = (np.asarray(x) for x in jax.jit(deflate_zero_rows(jnp.linalg.eigh))(jnp.asarray(a)))
    dead = np.flatnonzero(np.all(a == 0, axis=1))
    assert np.all(w[:len(dead)] == 0) and np.any(w[len(dead):] < 0)
    assert np.all(np.diff(w[len(dead):]) >= 0)
    # the deflated vectors are the zero rows' unit vectors, lowest row first
    assert np.array_equal(np.argmax(np.abs(v[:, :len(dead)]), axis=0), dead)
    ref = np.linalg.eigvalsh(a)
    scale = np.linalg.norm(a)
    assert np.max(np.abs(np.sort(w) - ref)) < 1e-13 * np.sqrt(n) * scale
    assert np.linalg.norm(a @ v - v * w) < 1e-12 * scale
    assert np.linalg.norm(v.conj().T @ v - np.eye(n)) < 1e-12
    assert float(jnp.min(w)) == w[len(dead)] < 0       # consumers read the minimum by value, not w[0]


def test_deflation_without_zero_rows_is_the_plain_sorted_solve():
    a = jnp.asarray(_hermitian(40, 12, 3) - 0.1 * np.eye(40))
    new = jax.jit(deflate_zero_rows(jnp.linalg.eigh))(a)
    old = jax.jit(jnp.linalg.eigh)(a)
    assert np.array_equal(np.asarray(new[0]), np.asarray(old[0]))
    assert np.array_equal(np.asarray(new[1]), np.asarray(old[1]))


@pytest.mark.parametrize("backend", ["cusolvermp", "slate", "elpa"])
def test_plan_pads_to_the_backend_side_and_drops_the_pads(backend):
    """Plan._checked_eigh: zero-pad to the side, deflate, solve once, drop by count, check.

    The stand-in library asserts it only ever sees a solve side, and returns
    cuSOLVERMp's raw V^H buffer where that backend does (``_eigh_columns``)."""
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    p = 2
    n = 2 * 263 if backend != "elpa" else 2 * 100          # sides 2*300 / 2*... (padded)
    side = _eigh_side(backend, n, p, np.complex128)
    assert side > n
    seen = []

    def library(a, *, mesh):
        seen.append(int(a.shape[-1]))
        assert _eigh_side(backend, int(a.shape[-1]), p, a.dtype) == int(a.shape[-1])
        w, v = jnp.linalg.eigh(a)
        return (w, jnp.conj(v.T)) if backend == "cusolvermp" else (w, v)
    safe = Plan._checked_eigh(SimpleNamespace(backend=backend), library, _eigh_columns)
    a = _with_zero_rows(n, n - 6, 4, 0.02)
    w, v, status = jax.jit(lambda x: safe(x, mesh=mesh))(jnp.asarray(a))
    w, v = np.asarray(w), np.asarray(v)
    assert seen == [side] and w.shape == (n,) and v.shape == (n, n)
    assert not bool(status[0])
    ref = np.linalg.eigvalsh(a)
    scale = np.linalg.norm(a)
    assert np.max(np.abs(np.sort(w) - ref)) < 1e-12 * scale
    assert np.linalg.norm(a @ v - v * w) < 1e-12 * scale
    assert np.linalg.norm(v.conj().T @ v - np.eye(n)) < 1e-12


def test_wrappers_refuse_an_operand_that_is_not_a_solve_side():
    """The one padding mechanism is the plan's: a wrapper never pads (shape rule, before any library call)."""
    import inspect
    from distrib_la import _cusolvermp
    for module in (_cusolvermp, _slate, _elpa):
        src = inspect.getsource(module.distributed_eigh)
        assert "pad it to" in src and "_padded" not in src
        assert "side" not in inspect.signature(module.distributed_eigh).parameters


def test_pad_side_and_drop_side_return_the_unpadded_eigensystem():
    n, side = 61, 67
    a = _hermitian(n, 20, 1) - 0.3 * np.eye(n)      # indefinite, rank-deficient + shift
    w, v = drop_side(*jax.jit(deflate_zero_rows(jnp.linalg.eigh))(pad_side(jnp.asarray(a), side)), n)
    w, v = np.asarray(w), np.asarray(v)
    ref = np.linalg.eigvalsh(a)
    assert np.max(np.abs(w - ref)) < 1e-12 * np.max(np.abs(ref))
    assert np.linalg.norm(a @ v - v * w) < 1e-12 * np.linalg.norm(a)
    assert np.linalg.norm(v.conj().T @ v - np.eye(n)) < 1e-12


def _low_rank_psd(n, rank, seed):
    """A rank-deficient PSD Gram: a (n - rank)-wide cluster at rounding level (the CrI3 failure shape)."""
    return _hermitian(n, rank, seed)


def _damaged(solve, columns):
    """A solver whose vectors are non-orthonormal in ``columns`` (cuSOLVERMp's near-zero-cluster defect)."""
    def bad(a, **_):
        w, v = solve(a)
        i, j = columns
        return w, v.at[:, i].set((v[:, i] + 0.3 * v[:, j]) / np.sqrt(1.09))
    return bad


@pytest.mark.parametrize("where,cut,refused", [
    ("cluster", 1e-8, False),     # defect among discarded columns: outside the check
    ("cluster", None, True),      # full check (Ritz eighs, site-5 directions): refused
    ("kept", 1e-8, True),         # defect in a used column: refused whatever the cut
])
def test_check_what_you_use(where, cut, refused):
    from distrib_la._result_check import checked_eigh
    n, rank = 64, 16
    a = jnp.asarray(_low_rank_psd(n, rank, 7))
    w = np.linalg.eigvalsh(np.asarray(a))
    assert np.sum(w > 1e-8 * w[-1]) == rank
    columns = (3, 5) if where == "cluster" else (n - 2, n - 1)
    (values, vectors), (failed, errors) = jax.jit(
        lambda x: checked_eigh(_damaged(jnp.linalg.eigh, columns), x, cut=cut))(a)
    assert bool(failed) == refused, np.asarray(errors)
    assert bool(np.isnan(np.asarray(values)).all()) == refused


def test_plan_and_route_c_take_the_cut():
    """The cut reaches the checked plan entry (padded side) and route (c)'s program."""
    from distrib_la._result_check import eigh_layout
    from distrib_la.plan import _reshard_stack_program
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    n = 2 * 263
    a = _low_rank_psd(n, 40, 8)
    safe = Plan._checked_eigh(SimpleNamespace(backend="slate"),
                              lambda x, *, mesh: _damaged(jnp.linalg.eigh, (side - n + 3, side - n + 5))(x),
                              _eigh_columns)
    side = _eigh_side("slate", n, 2, np.complex128)
    for cut, refused in ((1e-8, False), (None, True)):
        _, _, status = jax.jit(lambda x: safe(x, mesh=mesh, cut=cut))(jnp.asarray(a))
        assert bool(status[0]) == refused
    stack = jax.device_put(jnp.asarray(np.stack([a[:256, :256]] * 2)),
                           jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "x", "y")))
    (w, v), (failed, _) = _reshard_stack_program("eigh", mesh, (2, 256, 256), "complex128", 1, 1e-8)(stack)
    assert not bool(failed) and np.isfinite(np.asarray(w)).all()


@pytest.mark.parametrize("rows", [np.arange(61, 64), np.arange(3), np.arange(30, 33), np.arange(64), np.arange(0)])
def test_elpa_last_row_is_coupled_before_the_solve(rows):
    """ELPA's tail-block defect: the wrapper swaps the last row with the last coupled one; the
    solver must see a coupled last row (unless none is coupled), and the pairs come back exact."""
    from distrib_la._elpa import _last_row_coupled
    n = 64
    a = _hermitian(n, n, 9)
    a[rows, :] = 0
    a[:, rows] = 0
    a[rows, rows] = -5.0 - np.arange(len(rows))
    seen = []

    def solve(x):
        seen.append(x)
        return jnp.linalg.eigh(x)
    w, q = (np.asarray(t) for t in jax.jit(_last_row_coupled(solve))(jnp.asarray(a)))
    if len(rows) < n:
        last = np.asarray(jax.jit(lambda x: _last_row_coupled(lambda y: (y[-1], y))(x)[0])(jnp.asarray(a)))
        assert np.any(last[:-1] != 0)
    assert np.linalg.norm(a @ q - q * w) < 1e-12 * np.linalg.norm(a)
    assert np.linalg.norm(q.conj().T @ q - np.eye(n)) < 1e-12
