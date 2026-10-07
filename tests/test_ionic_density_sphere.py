"""Independent Fourier support and zero-mode controls for analytic ionic fields."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from psp.ionic_gspace import _ionic_gspace_jit, build_fft_G_data, build_ionic_and_core


@pytest.mark.parametrize("cutoff", [1.0, 3.3, 9.1])
def test_analytic_fields_have_density_sphere_support_and_preserve_gzero(cutoff):
    shape = (5, 7, 9)
    b = np.asarray([[1., .2, 0.], [0., 1.1, .1], [0., 0., .9]])
    g, norm = build_fft_G_data(shape, b, 1.)
    n = int(np.prod(shape))
    # One atom at the origin with constant radial coefficients. The exact
    # real fields are the direct sum over the retained plane waves, so
    # this checks support, inverse-FFT normalization and both G=0 values.
    out = _ionic_gspace_jit(
        jnp.asarray(g), jnp.asarray(norm), jnp.zeros((1, 1, 3)), jnp.asarray([1]),
        jnp.ones((1, 2)), jnp.full((1, 2), 2.), jnp.ones(1),
        jnp.ones(1), jnp.zeros(1), jnp.asarray(b), 0., 100., 1., np.sqrt(n), float(n), cutoff,
        max_atoms=1, nx=shape[0], ny=shape[1], nz=shape[2], truncation_2d=False)
    # Independent metric selection; no production G-norm/mask reused.
    keep = np.einsum("gi,ij,gj->g", g, b @ b.T, g) <= cutoff
    xyz = np.stack(np.meshgrid(*(np.arange(s) / s for s in shape), indexing="ij"), axis=-1)
    exact_core = np.cos(2 * np.pi * np.einsum("...a,ga->...g", xyz, g[keep])).sum(axis=-1)
    vion, core, coreg = map(np.asarray, out)
    np.testing.assert_allclose(core, exact_core, atol=8e-14, rtol=1e-14)
    np.testing.assert_allclose(vion, 2. * exact_core / np.sqrt(n), atol=2e-14, rtol=1e-14)
    np.testing.assert_allclose(coreg.ravel(), n * keep, atol=0, rtol=0)
    np.testing.assert_allclose(np.mean(core), 1., atol=1e-14, rtol=0)
    np.testing.assert_allclose(np.mean(vion), 2. / np.sqrt(n), atol=1e-14, rtol=0)


@pytest.mark.parametrize("value", [None, 0., -1., np.nan, np.inf])
def test_missing_or_invalid_density_cutoff_refuses_before_radial_build(value):
    mf = SimpleNamespace() if value is None else SimpleNamespace(ecutrho=value)
    with pytest.raises(ValueError, match="ecutrho"):
        build_ionic_and_core(mf, {}, (3, 3, 3))
