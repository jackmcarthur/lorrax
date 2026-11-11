"""The dense layouts, padding and shifted first attempt (CPU, seconds).

The padded rows carry sentinels below the spectrum (``pad_with_sentinels``) and leave the result
(``drop_sentinels``). The checked chain's first attempt on cuSOLVERMp is
``deflate_zero_rows(..., shift=True)``. Each is checked here with ``numpy.linalg.eigh`` /
``jnp.linalg.eigh`` standing in for the library.

Every layout the service derives from (n, mesh) is valid or refused, for every n LORRAX can ask
for (to 200k) on every mesh with P <= 64: the cuSOLVERMp eigh (``solve_layout``, the retry
``retry_block``, the polar dilation at 2n), the ELPA eigh (``eigh_layout``, both dtypes), the
cuSOLVERMp LU and its right-hand sides (``lu_layout``, ``rhs_layout``), and the panel widths of
the face GEMM and the flattened-mesh column panels.
"""
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from distrib_la._column_panels import _panel_width  # noqa: E402
from distrib_la._cusolvermp import (  # noqa: E402
    _BLOCK_MAX, _BLOCK_MIN, _LU_RELABEL_LOCAL_MIN, _largest_block, drop_sentinels, lu_layout,
    pad_with_sentinels, retry_block, rhs_layout, solve_layout)
from distrib_la._elpa import _TILE, eigh_layout  # noqa: E402
from distrib_la._panel_matmul import _interleaved_width  # noqa: E402
from distrib_la._result_check import deflate_zero_rows  # noqa: E402
from distrib_la.resolve import _check_geometry  # noqa: E402


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



# ----------------------------------------------------------------------------- layout properties
N_MAX = 200_000                          # the largest n LORRAX asks a dense layout for
_SQUARE = range(1, 9)                    # p x p meshes, P = p^2 <= 64
_MESHES = [(px, py) for px in range(1, 65) for py in range(1, 65) if px * py <= 64]


def _primes(limit):
    sieve = np.ones(limit + 1, bool)
    sieve[:2] = False
    for q in range(2, int(limit ** 0.5) + 1):
        if sieve[q]:
            sieve[q * q::q] = False
    return np.flatnonzero(sieve)


_PRIMES = _primes(N_MAX)


def _tiles(t_max, dense=4096, sample=1500):
    """Every tile up to ``dense``, the band around the LU relabel threshold, and above it a fixed
    sample: primes and two/three/six times a prime (the tiles with no block in [128, 256]),
    multiples of 128 and their neighbours, and uniform draws, all up to ``t_max``."""
    rng = np.random.default_rng(t_max)
    out = set(range(1, min(dense, t_max) + 1))
    out |= set(range(_LU_RELABEL_LOCAL_MIN - 200, _LU_RELABEL_LOCAL_MIN + 250))
    for mult in (1, 2, 3, 6):
        cand = mult * _PRIMES[(mult * _PRIMES > dense) & (mult * _PRIMES <= t_max)]
        if cand.size:
            out |= set(rng.choice(cand, size=min(sample // 4, cand.size), replace=False).tolist())
    m = 128 * rng.integers(dense // 128, t_max // 128 + 2, size=sample // 4)
    out |= {int(v + d) for v in m for d in (-1, 0, 1)}
    out |= set(rng.integers(dense, t_max + 1, size=sample // 4).tolist())
    out |= {3954, 7912, 8198, 10002, 15013, 2 * 4099}    # tiles that broke a layout before
    return sorted(t for t in out if 1 <= t <= t_max)


def _eigh_problems():
    """cuSOLVERMp eigh (and the polar dilation's 2n): ``solve_layout`` keeps tiles up to 256 whole
    and otherwise a block in [128, 256] that divides the solve tile, padding under 128 rows per
    rank; ``retry_block`` is None or a divisor of that tile in [8, block/2]."""
    bad = []
    for p in _SQUARE:
        for t in _tiles(N_MAX // p, dense=4096 // p, sample=1500 // p + 300):
            for n in (p * t, 2 * p * t):
                side, mb = solve_layout(n, p)
                tile, local = side // p, n // p
                ok = side % p == 0 and side >= n and tile % mb == 0
                ok &= ((side, mb) == (n, local) if local <= _BLOCK_MAX
                       else _BLOCK_MIN <= mb <= _BLOCK_MAX and tile - local < _BLOCK_MIN)
                rb = retry_block(n, p)
                ok &= rb is None or (8 <= rb <= mb // 2 and tile % rb == 0)
                if not ok:
                    bad.append(("eigh", p, n, side, mb, rb))
    return bad


def _elpa_problems():
    """ELPA eigh: a power-of-two block no larger than the dtype's preferred tile that divides
    the solve tile; padding under one block per rank."""
    bad = []
    for dtype in (np.float64, np.complex128):
        pref = _TILE[np.dtype(dtype).name]
        for p in _SQUARE:
            for t in _tiles(N_MAX // p):
                n = p * t
                side, nb = eigh_layout(n, p, dtype)
                tile = side // p
                ok = (side % p == 0 and side >= n and nb >= 1 and nb & (nb - 1) == 0
                      and nb <= pref and tile % nb == 0 and tile - t < nb)
                ok &= t < pref or nb == pref or tile == t
                if not ok:
                    bad.append(("elpa", np.dtype(dtype).name, p, n, side, nb))
    return bad


def _lu_problems():
    """cuSOLVERMp LU: one tile per rank only on a non-square mesh, at 1x1, for tiles up to 256,
    or on 2x2/3x3 up to the relabel threshold; otherwise a square block in [128, 256] that
    divides the padded tile, padding under 128 rows per rank.  A refusal is valid.  One tile
    above the threshold on a non-square mesh or at 1x1 is the planner branch's
    (claude/lu-workspace-pricing: price the LU workspace, refuse the non-square one-tile)."""
    bad = []
    for px, py in _MESHES:
        step = px * py // math.gcd(px, py)
        sample = _tiles(N_MAX // step, dense=4096 // px if px == py else 256,
                        sample=1500 // px + 300 if px == py else 120)
        for s in sample:
            n = step * s
            local = n // px
            try:
                side, mb, nb = lu_layout(n, px, py)
            except (ValueError, RuntimeError):
                continue
            if (side, mb, nb) == (n, local, n // py):
                if not (px != py or px == 1 or local <= _BLOCK_MAX
                        or (px < 4 and local <= _LU_RELABEL_LOCAL_MIN)):
                    bad.append(("lu one tile", px, py, n))
                continue
            tile = side // px
            if not (px == py and mb == nb and _BLOCK_MIN <= mb <= _BLOCK_MAX and side % px == 0
                    and tile % mb == 0 and 0 <= tile - local < _BLOCK_MIN):
                bad.append(("lu", px, py, n, side, mb, nb))
    return bad


def _rhs_problems():
    """LU right-hand sides: one block per process column unless relabeled with more than 256
    local columns; then a block in [128, 256] that divides the padded width (under 128 more)."""
    bad = []
    for py in range(1, 65):
        for s in _tiles(N_MAX // py, dense=max(256, 2048 // py), sample=400):
            nrhs = py * s
            for relabeled in (False, True):
                cols, nb_b = rhs_layout(nrhs, py, relabeled)
                if not relabeled or s <= _BLOCK_MAX:
                    ok = (cols, nb_b) == (nrhs, s)
                else:
                    width = cols // py
                    ok = (cols % py == 0 and width % nb_b == 0 and _BLOCK_MIN <= nb_b <= _BLOCK_MAX
                          and 0 <= width - s < _BLOCK_MIN)
                if not ok:
                    bad.append(("rhs", py, nrhs, relabeled, cols, nb_b))
    return bad


def _panel_problems():
    """Panel widths: the face GEMM's interleaved panels (``panel_matmul``: at least one column,
    within the two-panel limit and K/p^2 local columns, evened out with less than one panel of
    pad) and the flattened-mesh column panels (``project_out``: the fewest panels within the
    byte cap, none empty)."""
    bad = []
    for p in _SQUARE:
        for kl in _tiles(N_MAX // p, dense=1024, sample=200):
            for limit in (1, 2, 7, 64, 1000, 10 ** 6):
                w = _interleaved_width(p * kl, p, limit)
                n_panel = -(-kl // w)
                cap = max(1, min(limit // (2 * p), kl // p if p > 1 else kl))
                if not (1 <= w <= cap and n_panel * w >= kl and n_panel * w - kl < w
                        and n_panel == -(-kl // cap)):
                    bad.append(("face gemm", p, kl, limit, w))
    for n_loc in _tiles(N_MAX, dense=1024, sample=200):
        for column, panel in ((16, 1), (16, 16), (16 * 49536, 1 << 30), (1, 7), (8, 10 ** 9)):
            w, n_panels = _panel_width(n_loc, column, panel)
            cap = max(1, min(n_loc, panel // column))
            if not (1 <= w <= cap and w * n_panels >= n_loc and w * (n_panels - 1) < n_loc
                    and n_panels == -(-n_loc // cap)):
                bad.append(("column panels", n_loc, column, panel, w, n_panels))
    return bad


def _geometry_problems():
    """Mesh-geometry refusals (resolve guard 5): exactly the meshes a library cannot run.  The
    cuSOLVERMp and ELPA eigh and the cuSOLVERMp Cholesky need a square mesh, ScaLAPACK a square or
    1-D one; nothing else is refused here (the 1-D cuSOLVERMp LU/Cholesky guard precedes it)."""
    rules = {("eigh", "cusolvermp"): lambda px, py: px != py,
             ("eigh", "elpa"): lambda px, py: px != py,
             ("cholesky", "cusolvermp"): lambda px, py: px != py,
             ("solve_lu", "cusolvermp"): lambda px, py: False,
             ("eigh", "scalapack"): lambda px, py: px > 1 and py > 1 and px != py,
             ("solve_lu", "scalapack"): lambda px, py: px > 1 and py > 1 and px != py}
    bad = []
    for (op, backend), refuse in rules.items():
        for px, py in _MESHES:
            try:
                _check_geometry(op, backend, px, py)
                refused = False
            except ValueError:
                refused = True
            if refused != refuse(px, py):
                bad.append(("geometry", op, backend, px, py, refused))
    return bad


@pytest.mark.parametrize("family", ["eigh", "elpa", "lu", "rhs", "panels", "geometry"])
def test_every_layout_is_valid_or_refused(family):
    bad = {"eigh": _eigh_problems, "elpa": _elpa_problems, "lu": _lu_problems,
           "rhs": _rhs_problems, "panels": _panel_problems, "geometry": _geometry_problems}[family]()
    assert not bad, f"{len(bad)} invalid {family} layouts, first: {bad[:10]}"
