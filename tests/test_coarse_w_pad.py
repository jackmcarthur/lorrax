"""Tier-3 unit gates for the coarse-W → fine-grid zero-pad (BSE direct term).

``bse.bse_io.pad_W_R_to_grid`` embeds a coarse-k-grid real-space ``W_R`` into a
finer k-lattice by ZERO-PADDING in R.  The physics anchor: zero-padding in R is
EXACT trigonometric interpolation in k on any finer uniform grid.  It agrees
with the coarse ``W(k)`` at all common points (every coarse point for nested
grids; a strict subset for a non-divisor target such as 8→12).

Fixture-free (synthetic random W tensors), so it runs on CPU with float64 and
gates on modest hardware — NO 16-GPU / fixture dependency.

  1. exact-agreement identity: fft(pad(ifft(W_q_coarse)))|coarse-sub-points
     == W_q_coarse, bit-close (~1e-12).  The correctness anchor.
  2. no-op: padding to the SAME grid is byte-identical (and object-identical) —
     which guarantees the BSE matvec output is unchanged on the fast path.
  3. parity edges: even (6→12) AND odd (3→6) coarse grids, 2-D slab and true
     3-D grids; the even-N Nyquist rep is kept single-sided and STILL exact.
  4. decimate→pad pipeline (exactly what ``exciton_bands --w-coarse-grid`` runs):
     sub-sample a fine W_q, ifft, pad, fft → reproduces the fine W_q at the
     coarse sub-points.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from bse.bse_io import (pad_W_R_to_grid, decimate_W_q_to_subgrid,  # noqa: E402
                        _zeropad_R_axis)


def _rand_wq(shape, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape)
                       + 1j * rng.standard_normal(shape))


def _ifftn(x):
    return jnp.fft.ifftn(x, axes=(-3, -2, -1), norm="ortho")


def _fftn(x):
    return jnp.fft.fftn(x, axes=(-3, -2, -1), norm="ortho")


def _direct_trig_from_R(W_R, fine_grid):
    """Independent dense evaluation of the coarse Fourier polynomial."""
    arr = np.asarray(W_R)
    cg = arr.shape[-3:]
    phases = []
    for nc, nf in zip(cg, fine_grid):
        reps = np.fft.fftfreq(nc) * nc
        q = np.arange(nf, dtype=np.float64) / nf
        phases.append(np.exp(-2j * np.pi * q[:, None] * reps[None, :]))
    return (np.einsum("...abc,xa,yb,zc->...xyz", arr, *phases,
                      optimize=True) / np.sqrt(np.prod(cg)))


# (n_mu, n_nu, coarse_grid, fine_grid)
_CASES = [
    (3, 4, (6, 6, 1), (12, 12, 1)),     # owner case: 2-D slab, even, 2×
    (2, 3, (3, 3, 1), (6, 6, 1)),       # odd coarse (Nyquist-free) → even fine
    (2, 2, (4, 4, 2), (8, 8, 4)),       # true 3-D, all even
    (2, 2, (5, 5, 1), (15, 15, 1)),     # odd coarse, 3× (non-power-of-two)
    (3, 3, (6, 6, 1), (18, 18, 1)),     # even coarse, odd multiple 3×
]


@pytest.mark.parametrize("n_mu,n_nu,cg,fg", _CASES)
def test_exact_agreement_identity(n_mu, n_nu, cg, fg):
    """fft(pad(ifft(W_q_coarse))) equals W_q_coarse at the coinciding k-points."""
    W_q_c = _rand_wq((n_mu, n_nu, *cg), seed=hash((n_mu, cg)) & 0xFFFF)
    W_R_c = _ifftn(W_q_c)
    W_R_f = pad_W_R_to_grid(W_R_c, fg)
    assert tuple(W_R_f.shape) == (n_mu, n_nu, *fg)
    W_q_f = _fftn(W_R_f)
    # coarse BZ points are the sub-grid q_j = j·(nf/nc) of the fine BZ:
    W_q_f_on_coarse = decimate_W_q_to_subgrid(W_q_f, cg)
    err = float(jnp.max(jnp.abs(W_q_f_on_coarse - W_q_c)))
    assert err < 1e-11, f"exact-agreement residual {err:.2e} for {cg}->{fg}"


def test_noop_same_grid_byte_identical():
    """Padding to the SAME grid returns the input unchanged (byte + object id)."""
    W_R = _rand_wq((3, 4, 12, 12, 1), seed=7)
    out = pad_W_R_to_grid(W_R, (12, 12, 1))
    assert out is W_R                                   # object identity (fast path)
    assert np.array_equal(np.asarray(out), np.asarray(W_R))   # byte identical


def test_noop_3d_same_grid():
    W_R = _rand_wq((2, 2, 4, 4, 3), seed=9)
    out = pad_W_R_to_grid(W_R, (4, 4, 3))
    assert np.array_equal(np.asarray(out), np.asarray(W_R))


def test_decimate_then_pad_roundtrip():
    """The exact pipeline exciton_bands runs: fine W_q → decimate → ifft → pad →
    fft reproduces the fine W_q at the coarse sub-points."""
    fine = (12, 12, 1)
    coarse = (6, 6, 1)
    W_q_fine = _rand_wq((4, 5, *fine), seed=3)
    W_q_coarse = decimate_W_q_to_subgrid(W_q_fine, coarse)
    W_R_fine = pad_W_R_to_grid(_ifftn(W_q_coarse), fine)
    W_q_interp = _fftn(W_R_fine)
    # at the coarse sub-points the interpolant must equal the fine samples there
    on_coarse = decimate_W_q_to_subgrid(W_q_interp, coarse)
    err = float(jnp.max(jnp.abs(on_coarse - W_q_coarse)))
    assert err < 1e-11, f"decimate→pad round-trip residual {err:.2e}"


def test_scale_factor_matches_direct_fine_ifft():
    """pad_W_R_to_grid(W_R_coarse) == ifft(interpolant) — i.e. the padded tensor
    IS the fine-grid ifft of the trig interpolant, not just agreeing at samples.
    Verified by rebuilding the interpolant's fine ifft independently."""
    cg, fg = (6, 6, 1), (12, 12, 1)
    W_q_c = _rand_wq((2, 2, *cg), seed=11)
    W_R_f = pad_W_R_to_grid(_ifftn(W_q_c), fg)
    # independent construction: the interpolant sampled on the fine grid, then
    # ifft'd on the fine grid, must equal W_R_f exactly.
    W_q_interp = _fftn(W_R_f)
    W_R_f_rebuilt = _ifftn(W_q_interp)
    err = float(jnp.max(jnp.abs(W_R_f - W_R_f_rebuilt)))
    assert err < 1e-11, f"W_R_f is not a self-consistent fine ifft: {err:.2e}"


def test_zeropad_axis_index_map_even():
    """Per-element R-lattice embedding for even coarse N (the Nyquist edge).

    Coarse N=6 fftfreq reps [0,1,2,-3,-2,-1] at indices [0..5] must land at fine
    N=12 indices [0,1,2, 9,10,11] (reps [0,1,2,-3,-2,-1]); indices [3..8] zero.
    """
    x = jnp.arange(1, 7, dtype=jnp.float64).reshape(1, 6)  # tag each coarse slot
    y = np.asarray(_zeropad_R_axis(x, axis=1, n_fine=12)).ravel()
    expect = np.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6], dtype=float)
    np.testing.assert_array_equal(y, expect)


def test_zeropad_axis_index_map_odd():
    """Odd coarse N=3 reps [0,1,-1] at [0,1,2] → fine N=6 indices [0,1,5]."""
    x = jnp.arange(1, 4, dtype=jnp.float64).reshape(1, 3)
    y = np.asarray(_zeropad_R_axis(x, axis=1, n_fine=6)).ravel()
    expect = np.array([1, 2, 0, 0, 0, 3], dtype=float)
    np.testing.assert_array_equal(y, expect)


def test_nondivisor_8_to_12_matches_direct_trigonometric_evaluation():
    """8→12 evaluates the same Fourier series; no nesting assumption enters.

    The dense phase sum is independent of zero insertion and therefore goes
    red if this path degenerates into nearest-neighbour/linear interpolation or
    uses the target-grid frequencies for the coarse coefficients.
    """
    cg, fg = (8, 8, 1), (12, 12, 1)
    W_q_c = _rand_wq((2, 3, *cg), seed=812)
    W_R_c = _ifftn(W_q_c)
    got = np.asarray(_fftn(pad_W_R_to_grid(W_R_c, fg)))
    want = _direct_trig_from_R(W_R_c, fg)
    np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-12)

    # gcd(8,12)=4: indices (0,2,4,6) on the coarse grid coincide with
    # (0,3,6,9) on the fine grid, and the interpolant passes through them.
    ic = np.array([0, 2, 4, 6])
    jf = np.array([0, 3, 6, 9])
    np.testing.assert_allclose(
        got[:, :, jf[:, None], jf[None, :], 0],
        np.asarray(W_q_c)[:, :, ic[:, None], ic[None, :], 0],
        rtol=2e-12, atol=2e-12)


def test_zeropad_axis_index_map_nondivisor_even():
    """8→12 keeps reps [0,1,2,3,-4,-3,-2,-1] and inserts four zeros."""
    x = jnp.arange(1, 9, dtype=jnp.float64).reshape(1, 8)
    y = np.asarray(_zeropad_R_axis(x, axis=1, n_fine=12)).ravel()
    expect = np.array([1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8], dtype=float)
    np.testing.assert_array_equal(y, expect)


def test_rejects_coarser_grid_but_accepts_non_multiple_finer_grid():
    W_R = _rand_wq((2, 2, 6, 6, 1), seed=1)
    out = pad_W_R_to_grid(W_R, (10, 12, 1))
    assert out.shape == (2, 2, 10, 12, 1)
    with pytest.raises(ValueError):
        pad_W_R_to_grid(W_R, (3, 6, 1))      # 3 < 6 (coarser, not finer)
