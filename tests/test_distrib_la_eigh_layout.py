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
