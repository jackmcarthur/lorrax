"""The distributed eigh's layout, padding and shifted first attempt (CPU, seconds).

cuSOLVERMp runs at the block ``distrib_la._cusolvermp.solve_layout`` picks: the
largest divisor of n/p up to 256, or, where that is below 128 (n/p prime or a
small multiple of one), the smallest padded tile with a divisor in [128, 256].
The padded rows carry sentinels below the spectrum (``pad_with_sentinels``)
and leave the result (``drop_sentinels``). The checked chain's first attempt
on cuSOLVERMp is ``deflate_zero_rows(..., shift=True)``. Each is checked here
with ``numpy.linalg.eigh`` / ``jnp.linalg.eigh`` standing in for the library.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from distrib_la._cusolvermp import (  # noqa: E402
    _BLOCK_MAX, _BLOCK_MIN, _largest_block, drop_sentinels, pad_with_sentinels, retry_block,
    solve_layout)
from distrib_la._result_check import deflate_zero_rows  # noqa: E402


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
def test_every_padded_tile_has_a_block_in_range(p):
    for local in range(257, 9000, 7):
        side, block = solve_layout(p * local, p)
        tile = side // p
        assert tile % block == 0 and side % p == 0
        assert _BLOCK_MIN <= block <= _BLOCK_MAX
        if side > p * local:
            assert _largest_block(local) < _BLOCK_MIN and tile - local < _BLOCK_MIN
        retry = retry_block(p * local, p)
        assert retry is None or (8 <= retry <= block // 2 and tile % retry == 0)


def _hermitian(n, rank, seed):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((rank, n)) + 1j * rng.standard_normal((rank, n))
    return b.conj().T @ b / n


def test_sentinel_padding_returns_the_unpadded_eigensystem():
    n, side = 61, 67
    a = _hermitian(n, 20, 1) - 0.3 * np.eye(n)      # indefinite, rank-deficient + shift
    w, q = (np.asarray(x) for x in jnp.linalg.eigh(pad_with_sentinels(jnp.asarray(a), side)))
    # the raw cuSOLVERMp buffer is V^H: rows are eigenvectors
    w, raw = drop_sentinels(w, q.conj().T, n)
    v = np.asarray(raw).conj().T
    ref = np.linalg.eigvalsh(a)
    assert np.max(np.abs(np.asarray(w) - ref)) < 1e-12 * np.max(np.abs(ref))
    assert np.linalg.norm(a @ v - v * np.asarray(w)) < 1e-12 * np.linalg.norm(a)
    assert np.linalg.norm(v.conj().T @ v - np.eye(n)) < 1e-12


def test_shifted_deflation_keeps_the_spectrum_and_the_zero_rows():
    n = 48
    a = np.zeros((n, n), complex)
    a[:40, :40] = _hermitian(40, 10, 2) - 0.05 * np.eye(40)   # negative live values too
    plain = jax.jit(deflate_zero_rows(jnp.linalg.eigh))(jnp.asarray(a))
    shifted = jax.jit(deflate_zero_rows(jnp.linalg.eigh, shift=True))(jnp.asarray(a))
    ref = np.linalg.eigvalsh(a)
    scale = np.linalg.norm(a)
    for w, v in (plain, shifted):
        w, v = np.asarray(w), np.asarray(v)
        assert np.all(np.diff(w) >= 0)
        assert np.max(np.abs(w - ref)) < 1e-13 * np.sqrt(n) * scale
        assert np.linalg.norm(a @ v - v * w) < 1e-12 * scale
        assert np.linalg.norm(v.conj().T @ v - np.eye(n)) < 1e-12
    assert np.sum(np.asarray(shifted[0]) == 0) >= n - 40


def test_unshifted_deflation_is_unchanged_bitwise():
    """``shift=False`` is the deflation every non-cuSOLVERMp route ran before."""
    n = 40
    a = np.zeros((n, n), complex)
    a[:30, :30] = _hermitian(30, 12, 3)
    a = jnp.asarray(a)

    def before(x):
        dead = jnp.all(x == 0, axis=-1)
        bound = jnp.max(jnp.sum(jnp.abs(x), axis=-1), axis=-1)
        bound = jnp.where(bound > 0, bound, 1)
        sentinel = -2 * bound[..., None] * (1 + jnp.arange(n, dtype=bound.dtype) / n)
        diagonal = jnp.where(dead, sentinel, 0).astype(x.dtype)
        eye = jnp.eye(n, dtype=x.dtype)
        values, vectors = jnp.linalg.eigh(x + diagonal[..., :, None] * eye)
        restored = jnp.where(values < -1.5 * bound[..., None], 0, values)
        live_negative = jnp.any((restored < 0) & jnp.any(dead, axis=-1, keepdims=True))

        def reorder(operands):
            w, v = operands
            order = jnp.argsort(w, axis=-1, stable=True)
            return jnp.take_along_axis(w, order, axis=-1), jnp.take_along_axis(v, order[..., None, :], axis=-1)
        return jax.lax.cond(live_negative, reorder, lambda r: r, (restored, vectors))

    new = jax.jit(deflate_zero_rows(jnp.linalg.eigh))(a)
    old = jax.jit(before)(a)
    assert np.array_equal(np.asarray(new[0]), np.asarray(old[0]))
    assert np.array_equal(np.asarray(new[1]), np.asarray(old[1]))


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
        lambda x: checked_eigh((_damaged(jnp.linalg.eigh, columns),), x, cut=cut))(a)
    assert bool(failed) == refused, np.asarray(errors)
    assert bool(np.isnan(np.asarray(values)).all()) == refused


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


def test_route_c_takes_the_cut():
    """The cut reaches route (c)'s checked program (check what you use, on every attempt of main's chain)."""
    from jax.sharding import Mesh
    from distrib_la.plan import _reshard_stack_program
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    a = _low_rank_psd(256, 40, 8)
    stack = jax.device_put(jnp.asarray(np.stack([a] * 2)),
                           jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "x", "y")))
    (w, v), failed = _reshard_stack_program("eigh", mesh, (2, 256, 256), "complex128", 1, "all", 1e-8)(stack)
    assert not bool(np.asarray(failed[0]).any()) and np.isfinite(np.asarray(w)).all()


def test_a_local_workspace_the_vendor_rejects_is_inadmissible(monkeypatch, capsys):
    """run30k: cusolverDnZheevd_bufferSize answers status 3 at n 30000. The LOCAL candidate's
    query fails closed (priced past any room, so admission takes the whole mesh); a failure of
    the distributed (context) query still raises."""
    from types import SimpleNamespace
    from distrib_la import loader, workspace

    class _Rejects:                                   # a bufferSize query that answers INVALID_VALUE
        def __call__(self, *args):
            return 3
    monkeypatch.setattr(loader, "get_lib", lambda platform: SimpleNamespace(lrx_eigh_workspace_bytes=_Rejects()))
    workspace._vendor_query.cache_clear()
    try:
        assert workspace._vendor_query(0, "eigh", (30000,), "<c16") == (workspace.UNPRICED_LOCAL_BYTES, 0)
        assert workspace.UNPRICED_LOCAL_BYTES > 1e15
        assert "the local eigh is unpriced" in capsys.readouterr().out
        with pytest.raises(RuntimeError, match="status=3"):
            workspace._vendor_query(7, "eigh", (30000,), "<c16")
    finally:
        workspace._vendor_query.cache_clear()


def test_lu_relabels_from_4x4_and_keeps_the_2x2_bound():
    """run30k: n 30000 on 4x4 (local 7500) factors block-cyclic at block 250 with a chunkable RHS
    (30 local column blocks); 2x2 keeps its measured rule (relabel only past local 8192)."""
    from distrib_la._cusolvermp import lu_blocks, rhs_block
    mb, nb = lu_blocks(30000, 4, 4)
    assert (mb, nb) == (250, 250)
    nb_b = rhs_block(30000, 30000, 4, 4, mb)
    assert nb_b == 250 and (30000 // 4) // nb_b == 30
    assert lu_blocks(30000, 2, 2) == (250, 250)          # local 15000 > 8192: relabeled before too
    assert lu_blocks(9152, 2, 2) == (4576, 4576)          # local <= 8192 on 2x2: one tile, unchanged
    assert rhs_block(9152, 9152, 2, 2, 4576) == 4576


def test_the_gathered_retry_is_priced_against_the_plans_room(monkeypatch):
    """run30k: cuSOLVERMp 0.9.1 left the n 30000 direction eigh with no correct attempt. The
    gathered retry joins the chain where 2 n^2 and the local eigh workspace fit the plan's room
    (no room: the 64 MiB rule), and the local query runs once per shape for the process."""
    import sys
    import lxkit
    import distrib_la  # noqa: F401
    from distrib_la import loader, workspace
    pl = sys.modules["distrib_la.plan"]                  # the module (distrib_la.plan is also a function)
    calls = []

    def query(ctx, n, cplx, device, host):              # cusolverDnXsyevd_bufferSize: ~2x LAPACK heevd
        calls.append(int(n))
        device._obj.value, host._obj.value = 2 * (16 * (2 * n + n * n) + 8 * (1 + 5 * n + 2 * n * n)), 0
        return 0
    monkeypatch.setattr(loader, "get_lib", lambda platform: type("Lib", (), {
        "lrx_eigh_workspace_bytes": staticmethod(query)})())
    monkeypatch.setattr(lxkit, "device_vendor", lambda device: "cuda")
    workspace._vendor_query.cache_clear()
    try:
        assert pl._gathered_admitted((1448, 1448), np.complex128) and not calls      # the fixed rule
        assert not pl._gathered_admitted((9152, 9152), np.complex128)               # no room
        assert pl._gathered_admitted((9152, 9152), np.complex128, 63.41e9)           # 8.0 GB
        assert not pl._gathered_admitted((30000, 30000), np.complex128, 63.41e9)     # 86.4 GB
        assert pl._gathered_admitted((30000, 30000), np.complex128, 90e9)
        assert sorted(calls) == [9152, 30000]                                         # once each
    finally:
        workspace._vendor_query.cache_clear()
