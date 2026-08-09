"""The non-TDA KPM seed applies its pad mask by BROADCAST, not by a copy.

``make_bse_random_vector``'s non-TDA branch used to build
``mask_full = jnp.stack([mask, mask], axis=0)`` and capture it in the
returned closure — a materialised ``(2, 1, nc_pad, nv_pad, nk)`` duplicate
held resident for the whole KPM run, on top of the ``(1, nc_pad, nv_pad,
nk)`` ``mask`` it was built from, for a window that is IDENTICAL on the two
stack halves.  Broadcasting applies the same window with no buffer.

These gates pin the two facts that make the removal safe:

1. The two forms are BIT-identical.  The pad mask is pure 0/1 (it is built
   by ``pad_zone_mask_np`` from index counts, not by thresholding an
   energy), so ``x * mask`` and ``x * stack([mask, mask])`` agree exactly —
   this is an equality claim, not a tolerance claim.
2. The seed really is zero on the pad block, on BOTH halves.  That is the
   property the mask exists for: an unmasked start vector has support in a
   block whose eigenvalues are the ``PAD_EPS_GUARD_RY`` sentinel, and a
   Lanczos sweep started there returns ~1e3 Ry as e_max.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax

from bse.bse_io import pad_zone_mask_np


NK, NC, NV, NC_PAD, NV_PAD = 5, 3, 2, 4, 4


@pytest.fixture
def bundle():
    return {
        "nkx": NK, "nky": 1, "nkz": 1,
        "n_cond": NC, "n_val": NV,
        "n_cond_pad": NC_PAD, "n_val_pad": NV_PAD,
        "eps_c": jnp.zeros((1,), dtype=jnp.float64),
    }


def test_broadcast_mask_is_bit_identical_to_the_stacked_copy(bundle):
    from bse.bse_kpm import make_bse_random_vector

    fn = make_bse_random_vector(bundle, use_tda=False)
    x = fn(jax.random.PRNGKey(7))

    mask = jnp.asarray(pad_zone_mask_np(NC, NV, NC_PAD, NV_PAD, NK),
                       dtype=jnp.float64)
    mask_full = jnp.stack([mask, mask], axis=0)

    # Rebuild the raw Rademacher stack exactly as the factory does, then
    # apply the OLD (stacked) mask to it.
    k_x, k_y = jax.random.split(jax.random.PRNGKey(7))
    shape_1 = (1, NC_PAD, NV_PAD, NK)
    x0 = 2.0 * jax.random.bernoulli(k_x, shape=shape_1).astype(jnp.float64) - 1.0
    x1 = 2.0 * jax.random.bernoulli(k_y, shape=shape_1).astype(jnp.float64) - 1.0
    raw = jnp.stack([x0, x1], axis=0)
    old = (raw * mask_full).astype(jnp.complex128)

    assert x.shape == old.shape == (2, 1, NC_PAD, NV_PAD, NK)
    assert np.array_equal(np.asarray(x), np.asarray(old))


def test_the_pad_mask_is_pure_zero_one():
    """A non-binary 'mask' would be a WEIGHT, and broadcasting it would still
    be fine — but the bit-identity argument above rests on 0/1, so pin it."""
    m = pad_zone_mask_np(NC, NV, NC_PAD, NV_PAD, NK)
    assert set(np.unique(m).tolist()) <= {0.0, 1.0}


def test_seed_is_exactly_zero_on_the_pad_block_on_both_halves(bundle):
    from bse.bse_kpm import make_bse_random_vector

    x = np.asarray(make_bse_random_vector(bundle, use_tda=False)(
        jax.random.PRNGKey(3)))
    assert np.all(x[:, :, NC:, :, :] == 0), "conduction pad not zeroed"
    assert np.all(x[:, :, :, NV:, :] == 0), "valence pad not zeroed"
    # RED TWIN: the physical block must NOT be all zero, or the gate above
    # would pass on a vector that is zero everywhere.
    assert np.count_nonzero(x[:, :, :NC, :NV, :]) > 0


def test_tda_branch_still_masks(bundle):
    from bse.bse_kpm import make_bse_random_vector

    x = np.asarray(make_bse_random_vector(bundle, use_tda=True)(
        jax.random.PRNGKey(3)))
    assert x.shape == (1, NC_PAD, NV_PAD, NK)
    assert np.all(x[:, NC:, :, :] == 0)
    assert np.all(x[:, :, NV:, :] == 0)
    assert np.count_nonzero(x[:, :NC, :NV, :]) > 0
