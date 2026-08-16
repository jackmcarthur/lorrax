"""Star covariance of the tetrahedron Fermi-surface weights.

The quadrature in ``gw.fermi_surface`` splits every grid cell into the six
Kuhn tetrahedra that share the ONE hardcoded ``(1,1,1)`` body diagonal
(``_TETRA_OFFSETS``).  That partition is not invariant under the crystal
point group, so the weight table it returns is not star covariant, and the
head's Drude tensor -- ``sum_kn w_kn v_a v_b`` -- comes out anisotropic on a
cubic crystal, where it must be exactly ``(tr D / 3) delta_ab``.

The discriminating cell is :func:`test_drude_tensor_isotropy_on_a_cubic_deck`:
the raw weights FAIL isotropy and the star-symmetrized weights PASS it.  A
test that only checked the symmetrized side would pass against a no-op.

Pure numpy + the module under test.  No device, no container, no fixtures.

Measured counterpart on the real deck (bcc Na 8x8x8, 48 ops, converged
metallic MPA-QSGW state): 4 of 48 ops leave the raw table invariant, the
Drude tensor is 2.68 percent anisotropic, and after symmetrization it is
isotropic to 7e-13 with the trace unchanged to the last bit.
"""
import itertools

import numpy as np
import pytest

from gw.fermi_surface import star_symmetrize_weights, tetrahedron_delta_weights


# --- a synthetic simple-cubic deck -----------------------------------------
# Reciprocal lattice = identity, so fractional and Cartesian axes coincide and
# the point group is the 48 signed axis permutations.  The band is isotropic
# in k, so any deviation from an isotropic Drude tensor is the quadrature's.
_GRID = (8, 8, 8)


def _cubic_ops():
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3), dtype=np.int64)
            for i, j in enumerate(perm):
                m[i, j] = signs[i]
            ops.append(m)
    return ops


def _deck():
    """Return (kpoints_crystal, energies_kn, velocities_akn, star_index).

    The band is the simple-cubic nearest-neighbour one,
    ``E(k) = 3 - sum_a cos(2 pi k_a)``, with ``v_a = 2 pi sin(2 pi k_a)``.
    It is PERIODIC, which a free-electron ``|k|^2`` model is not: the latter
    needs a branch cut to fold k into ``[-1/2,1/2)``, and at a zone-boundary
    grid point (``k_a = 1/2`` on an even grid) the two branches give
    ``v_a`` of opposite sign.  That makes the model's own velocity field
    non-covariant and the test then fails for a reason that has nothing to
    do with the quadrature.  Measured while writing this test: it produced a
    symmetrized tensor with equal diagonals and equal NONZERO off-diagonals.
    """
    g = np.asarray(_GRID, dtype=np.int64)
    idx = np.array(list(np.ndindex(*(int(x) for x in g))), dtype=np.int64)
    kfrac = idx / g[None, :]
    energies = (3.0 - np.sum(np.cos(2.0 * np.pi * kfrac), axis=1))[:, None]
    velocities = (2.0 * np.pi * np.sin(2.0 * np.pi * kfrac)).T[:, :, None]

    # exact orbits under the 48 cubic ops, built from the ops themselves
    key = {tuple(r): i for i, r in enumerate(np.mod(idx, g))}
    label = np.full(idx.shape[0], -1, dtype=np.int64)
    nstar = 0
    for i, r in enumerate(np.mod(idx, g)):
        if label[i] >= 0:
            continue
        for m in _cubic_ops():
            label[key[tuple(np.mod(m @ r, g))]] = nstar
        nstar += 1
    assert np.all(label >= 0)
    return kfrac, energies, velocities, label


def _drude(weights_kn, velocities_akn):
    return np.einsum("akn,kn,bkn->ab", velocities_akn, weights_kn,
                     velocities_akn, optimize=True)


def _anisotropy(d):
    return float(np.max(np.abs(d - (np.trace(d) / 3.0) * np.eye(3)))
                 / (np.trace(d) / 3.0))


@pytest.fixture(scope="module")
def deck():
    return _deck()


@pytest.fixture(scope="module")
def weights(deck):
    kfrac, energies, _, _ = deck
    mu = 3.0                                  # the half-filled sc surface
    w = tetrahedron_delta_weights(energies, kfrac, _GRID, mu)
    assert float(np.sum(w)) > 0.0, "the test's mu must cut the band"
    return w


def test_raw_weights_are_not_star_covariant(deck, weights):
    """The defect itself: star partners get different weights."""
    _, _, _, label = deck
    worst = 0.0
    for s in np.unique(label):
        m = np.flatnonzero(label == s)
        worst = max(worst, float(np.max(weights[m]) - np.min(weights[m])))
    assert worst > 1.0e-6 * float(np.max(weights)), (
        "the Kuhn partition is expected to break star covariance; a zero "
        "spread here means this test is no longer discriminating")


def test_star_symmetrized_weights_are_star_covariant(deck, weights):
    _, _, _, label = deck
    wbar = star_symmetrize_weights(weights, label)
    for s in np.unique(label):
        m = np.flatnonzero(label == s)
        assert np.allclose(wbar[m], wbar[m][0], rtol=0.0, atol=1.0e-15)


def test_symmetrization_preserves_the_integral(deck, weights):
    """It moves weight only between star partners: N(E_F) is untouched."""
    _, _, _, label = deck
    wbar = star_symmetrize_weights(weights, label)
    assert float(np.sum(wbar)) == pytest.approx(float(np.sum(weights)),
                                                rel=0.0, abs=1.0e-14)


def test_drude_tensor_isotropy_on_a_cubic_deck(deck, weights):
    """THE discriminating cell: raw FAILS isotropy, symmetrized PASSES.

    Both halves are asserted in one test on purpose.  Checking only the
    symmetrized half would pass against a symmetrization that did nothing.
    """
    _, _, velocities, label = deck
    d_raw = _drude(weights, velocities)
    d_sym = _drude(star_symmetrize_weights(weights, label), velocities)

    assert _anisotropy(d_raw) > 1.0e-6, (
        "the raw Kuhn-partition Drude tensor is expected to be anisotropic "
        f"on a cubic deck; got {_anisotropy(d_raw):.3e}")
    assert _anisotropy(d_sym) < 1.0e-12, (
        f"star-symmetrized Drude tensor is not isotropic: {d_sym!r}")
    assert np.trace(d_sym) == pytest.approx(np.trace(d_raw), rel=1.0e-14)


def test_star_symmetrize_refuses_a_mismatched_label_array(weights):
    with pytest.raises(ValueError, match="star_index must be"):
        star_symmetrize_weights(weights, np.zeros(weights.shape[0] + 1,
                                                  dtype=np.int64))
