"""Unit tests for the σ band-window mesh pad/strip pair.

``gw.ppm_sigma.pad_sigma_window`` / ``strip_sigma_window`` are the fix
the reduce-scatter divisibility guard in ``ppm_tau_kernel`` prescribes:
m is padded to a multiple of p_x and n to a multiple of p_y
INDEPENDENTLY (the p_x·p_y product rule is denounced as up-to-3.16×
waste in the pad's own docstring), the pad block of Σ is exactly zero,
and the strip is the single seam where the padded extent stops.

QUALITY_PATTERNS class 2 lists the σ-window divisibility among the
scale-threshold failures: the default suite runs a 1×1 mesh where the
pad is structurally unreachable, and until now the only gate was a
manual run cited in commit 9c687a5.  These tests exercise the shape
math WITHOUT a multi-device mesh — ``pad_sigma_window`` reads only
``mesh_xy.shape['x'/'y']``, so a shape-only stand-in drives the exact
production code path (rectangular 8×10 mesh, window 70: the MoS2 12×12
configuration that first fired the guard).  No MPI, no big arrays.

The one-axis-padded strip case pins the bug the strip docstring names:
"Testing only the last axis would have returned an m-padded Σ
untouched."
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pytest

from gw.ppm_sigma import pad_sigma_window, strip_sigma_window


def _mesh(p_x: int, p_y: int):
    """Shape-only mesh stand-in — pad_sigma_window reads mesh_xy.shape
    only, so the pad math is testable on a single CPU device."""
    return SimpleNamespace(shape={"x": p_x, "y": p_y})


def _psi_pair(nk: int, nb: int, ns: int, nmu: int):
    """(psi_xr, psi_yn) with distinct, reproducible values.

    Layouts match the production call site (_run_sigma_branch):
      psi_xr : (nk, m, s, mu)   band axis 1
      psi_yn : (nk, s, mu, n)   band axis 3
    """
    rng = np.random.default_rng(7)
    xr = rng.standard_normal((nk, nb, ns, nmu)) \
        + 1j * rng.standard_normal((nk, nb, ns, nmu))
    yn = rng.standard_normal((nk, ns, nmu, nb)) \
        + 1j * rng.standard_normal((nk, ns, nmu, nb))
    return jnp.asarray(xr), jnp.asarray(yn)


# ---------------------------------------------------------------------------
# pad_sigma_window — per-axis extents
# ---------------------------------------------------------------------------

def test_pad_extents_rectangular_mesh_one_axis_only():
    # MoS2 12x12 configuration: window 70 on an 8x10 mesh.
    # m -> 72 (next multiple of p_x=8); n stays 70 (70 % 10 == 0).
    xr, yn = _psi_pair(nk=2, nb=70, ns=1, nmu=3)
    xr_p, yn_p, nb_real = pad_sigma_window(xr, yn, _mesh(8, 10))
    assert nb_real == 70
    assert xr_p.shape == (2, 72, 1, 3)
    assert yn_p.shape == (2, 1, 3, 70)
    # The divisible axis is an identity passthrough — same buffer.
    assert yn_p is yn
    # m_pad != n_pad: the independent per-axis contract, NOT the
    # p_x*p_y product rule (which would demand 80x80 here).
    assert int(xr_p.shape[1]) != int(yn_p.shape[3])


def test_pad_extents_square_mesh_both_axes():
    xr, yn = _psi_pair(nk=1, nb=70, ns=1, nmu=2)
    xr_p, yn_p, nb_real = pad_sigma_window(xr, yn, _mesh(8, 8))
    assert nb_real == 70
    assert xr_p.shape[1] == 72 and yn_p.shape[3] == 72
    # 72x72 = 5184, not the product rule's 128x128 = 16384 (3.16x).


def test_pad_noop_when_divisible():
    xr, yn = _psi_pair(nk=1, nb=8, ns=1, nmu=2)
    xr_p, yn_p, nb_real = pad_sigma_window(xr, yn, _mesh(4, 2))
    assert nb_real == 8
    # Full identity: both buffers returned untouched.
    assert xr_p is xr and yn_p is yn


def test_pad_block_exactly_zero_and_real_block_untouched():
    xr, yn = _psi_pair(nk=2, nb=5, ns=2, nmu=3)
    xr_p, yn_p, nb_real = pad_sigma_window(xr, yn, _mesh(3, 4))
    assert (xr_p.shape[1], yn_p.shape[3]) == (6, 8)
    # Pad rows/cols are EXACT zeros (the Σ pad block being exactly zero
    # is bilinear in these).
    assert np.all(np.asarray(xr_p)[:, nb_real:] == 0.0)
    assert np.all(np.asarray(yn_p)[..., nb_real:] == 0.0)
    # The real block is bit-identical to the input.
    np.testing.assert_array_equal(np.asarray(xr_p)[:, :nb_real],
                                  np.asarray(xr))
    np.testing.assert_array_equal(np.asarray(yn_p)[..., :nb_real],
                                  np.asarray(yn))


# ---------------------------------------------------------------------------
# strip_sigma_window — the single un-pad seam
# ---------------------------------------------------------------------------

def test_strip_one_axis_padded_m_only():
    # The docstring's named bug: m padded, n at the real extent.  A strip
    # that tested only the LAST axis would return this untouched.
    nb = 70
    sigma = jnp.asarray(np.arange(2 * 3 * 72 * 70, dtype=np.float64)
                        .reshape(2, 3, 72, 70))
    out = strip_sigma_window(sigma, nb)
    assert out.shape == (2, 3, 70, 70)
    np.testing.assert_array_equal(np.asarray(out),
                                  np.asarray(sigma)[..., :nb, :nb])


def test_strip_both_axes_padded():
    nb = 5
    sigma = jnp.asarray(np.random.default_rng(3)
                        .standard_normal((1, 6, 8)))
    out = strip_sigma_window(sigma, nb)
    assert out.shape == (1, 5, 5)
    np.testing.assert_array_equal(np.asarray(out),
                                  np.asarray(sigma)[..., :nb, :nb])


def test_strip_noop_and_none():
    nb = 4
    sigma = jnp.zeros((2, nb, nb))
    # Unpadded input: identity (same object, no slice copy).
    assert strip_sigma_window(sigma, nb) is sigma
    assert strip_sigma_window(None, nb) is None


# ---------------------------------------------------------------------------
# pad -> (bilinear zero pad block) -> strip round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_x,p_y,nb", [(8, 10, 70), (8, 8, 70), (3, 4, 5)])
def test_pad_strip_round_trip(p_x, p_y, nb):
    """A Σ-shaped bilinear contraction of the padded ψ pair has an
    exactly-zero pad block, and strip recovers exactly the unpadded
    contraction — the guarantee _run_sigma_branch relies on."""
    xr, yn = _psi_pair(nk=1, nb=nb, ns=1, nmu=2)
    xr_p, yn_p, nb_real = pad_sigma_window(xr, yn, _mesh(p_x, p_y))

    def _sigma(a_xr, a_yn):
        # Σ[k,m,n] = Σ_{s,μ,s',μ'} ψ*_m σ ψ_n with σ = identity coupling:
        # every output element is an independent contraction (the
        # exactness argument in pad_sigma_window's docstring).
        return jnp.einsum('kmsx,ktyn->kmn', jnp.conj(a_xr), a_yn)

    padded = _sigma(xr_p, yn_p)
    assert padded.shape == (1, xr_p.shape[1], yn_p.shape[3])
    # Pad block exactly zero (rows m >= nb and cols n >= nb).
    pb = np.asarray(padded)
    assert np.all(pb[:, nb_real:, :] == 0.0)
    assert np.all(pb[:, :, nb_real:] == 0.0)
    # Strip == the unpadded contraction, bit-identical (pure data
    # movement: appending zero bands perturbs no existing element).
    stripped = strip_sigma_window(padded, nb_real)
    np.testing.assert_array_equal(np.asarray(stripped),
                                  np.asarray(_sigma(xr, yn)))
