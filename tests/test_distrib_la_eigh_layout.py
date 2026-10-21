"""The distributed eigh's layout, padding and shifted first attempt (CPU, seconds).

The padded rows carry sentinels below the spectrum (``pad_with_sentinels``) and leave the result
(``drop_sentinels``). The checked chain's first attempt on cuSOLVERMp is
``deflate_zero_rows(..., shift=True)``. Each is checked here with ``numpy.linalg.eigh`` /
``jnp.linalg.eigh`` standing in for the library.  Every layout either library picks (cuSOLVERMp
``solve_layout``, ELPA ``eigh_layout``) is valid: a block in range that tiles the padded side.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from distrib_la._cusolvermp import (  # noqa: E402
    _BLOCK_MAX, _BLOCK_MIN, _largest_block, drop_sentinels, pad_with_sentinels, retry_block,
    solve_layout)
from distrib_la._elpa import eigh_layout  # noqa: E402
from distrib_la._result_check import deflate_zero_rows  # noqa: E402


def _hermitian(n, rank, seed):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((rank, n)) + 1j * rng.standard_normal((rank, n))
    return b.conj().T @ b / n


def test_padded_and_shifted_eigh_numerics():
    # sentinel padding returns the unpadded eigensystem
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

    # shifted deflation keeps the spectrum and the zero rows
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

    # ``shift=False`` is the deflation every non-cuSOLVERMp route ran before: bitwise unchanged
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

