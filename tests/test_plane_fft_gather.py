"""The plane FFT door (``ffi.fft.make_plane_fft_gather``, mathdx mode 10) on cpu.

1. ``plane_fft_split``: every QE-style axis either splits into coprime
   cuFFTDx thread-FFT factors (<= 40) or is refused (prime powers above 40).
2. A NumPy model of mode 10's passes (gather the occupied rows, Good-Thomas
   row passes on those rows only, column passes reading dead rows as zero,
   the slot-permuted store) equals ``np.fft.fft2`` of the filled plane on
   every split size, square and rectangular, disk and random supports.
   This pins the index maps the kernel uses; the GPU parity gate is
   ``tests/multi_device/plane_fft_gather_p4.py``.
3. The cpu door is the XLA route and equals the same reference.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ffi.fft import make_plane_fft_gather, plane_fft_split

QE = (24, 25, 27, 30, 36, 40, 45, 48, 54, 60, 64, 72, 75, 80, 90, 96, 100, 108, 120, 125,
      128, 144, 150, 160, 180, 216, 240, 250)


def _support(nb, nc, frac, rng, kind):
    """(nb*nc,) plane_from_col for a centred (wrapped) disk or a random support."""
    if kind == "disk":
        hb = np.fft.fftfreq(nb) * nb
        hc = np.fft.fftfreq(nc) * nc
        r2 = hb[:, None] ** 2 / nb ** 2 + hc[None, :] ** 2 / nc ** 2
        occ = (r2 <= np.quantile(r2, frac)).ravel()
    else:
        occ = rng.random(nb * nc) < frac
    cols = np.flatnonzero(occ)
    pfc = np.full(nb * nc, cols.size, np.int64)
    pfc[cols] = np.arange(cols.size)
    return pfc, cols.size


def _mode10_model(F, pfc, n_col, nb, nc):
    """NumPy image of the kernel: the same passes, index maps and zero rules."""
    (b1, b2), (c1, c2) = plane_fft_split(nb), plane_fft_split(nc)
    occ = (pfc < n_col).reshape(nb, nc)
    rows = np.flatnonzero(occ.any(1))
    gidx = np.where(occ[rows], pfc.reshape(nb, nc)[rows], -1)
    live = occ.any(1)
    buf = np.full(F.shape[:-1] + (nb, nc), np.nan + 0j)       # dead rows never written
    buf[..., rows, :] = np.where(gidx >= 0, F[..., np.maximum(gidx, 0)], 0)

    def passes(x, n, n1, n2, axis, first_live=None):
        for first in (True, False):
            m, step = (n1, n2) if first else (n2, n1)
            if m == 1:
                continue
            for i in range(n2 if first else n1):
                off = (n1 if first else n2) * i
                p = (step * np.arange(m) + off) % n
                v = np.take(x, p, axis=axis)
                if first and first_live is not None:
                    shape = [1] * v.ndim
                    shape[axis] = m
                    v = np.where(first_live[p].reshape(shape), v, 0)
                y = np.fft.fft(v, axis=axis)
                idx = [slice(None)] * x.ndim
                idx[axis] = p
                x[tuple(idx)] = y
        return x

    buf[..., rows, :] = passes(buf[..., rows, :], nc, c1, c2, -1)
    buf = passes(buf, nb, b1, b2, -2, first_live=live)
    sb = (b2 * (np.arange(nb) % b1) + b1 * (np.arange(nb) % b2)) % nb
    sc = (c2 * (np.arange(nc) % c1) + c1 * (np.arange(nc) % c2)) % nc
    return buf[..., sb[:, None], sc[None, :]]


def _reference(F, pfc, n_col, nb, nc):
    Fz = np.concatenate([F, np.zeros(F.shape[:-1] + (1,), F.dtype)], -1)
    return np.fft.fft2(Fz[..., np.minimum(pfc, n_col)].reshape(F.shape[:-1] + (nb, nc)))


def test_split_covers_qe_axes_and_refuses_large_prime_powers():
    for n in QE:
        s = plane_fft_split(n)
        if n in (64, 125, 128, 250):
            assert s is None, (n, s)
            continue
        n1, n2 = s
        assert n1 * n2 == n and math.gcd(n1, n2) == 1 and 2 <= n1 <= 40 and n2 <= 40, (n, s)


@pytest.mark.parametrize("nb,nc", [(n, n) for n in QE if plane_fft_split(n)] + [(54, 45), (72, 80), (24, 100)])
def test_mode10_model_equals_fft2(nb, nc):
    rng = np.random.default_rng(nb * 1000 + nc)
    for frac, kind in ((0.2, "disk"), (0.45, "disk"), (0.3, "random")):
        pfc, n_col = _support(nb, nc, frac, rng, kind)
        F = rng.standard_normal((3, n_col)) + 1j * rng.standard_normal((3, n_col))
        ref = _reference(F, pfc, n_col, nb, nc)
        got = _mode10_model(F, pfc, n_col, nb, nc)
        assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref)), (nb, nc, frac, kind)


def test_cpu_door_is_the_xla_route():
    import jax
    from jax.sharding import Mesh
    mesh = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    rng = np.random.default_rng(7)
    for nb, nc in ((54, 54), (64, 64), (45, 54)):
        pfc, n_col = _support(nb, nc, 0.2, rng, "disk")
        F = rng.standard_normal((2, 3, n_col)) + 1j * rng.standard_normal((2, 3, n_col))
        got = np.asarray(jax.jit(make_plane_fft_gather(mesh, pfc, n_col, (nb, nc)))(F))
        ref = _reference(F, pfc, n_col, nb, nc)
        assert np.max(np.abs(got - ref)) <= 1e-13 * np.max(np.abs(ref))
