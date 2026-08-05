"""Step-occupation Fermi level: gapped, metallic, degenerate, k-weighted.

RUNS IN THE CONTAINER, NOT ON A LOGIN NODE.  ``gw.efermi`` is a jax.jit
kernel now, so this imports it normally instead of by path and runs with
the container gates: ``/scratch2/08271/jackmc/dsc_demo/sc_gates.sbatch``
runs it alongside ``test_scissor_weights`` / ``test_sc_band_window`` /
``test_layering``.

The degenerate refusal and the occupation dtype are checked with E on a
device as well as on the host — that is the path the SC loop takes, and
the refusal is decoded from a kernel flag rather than raised where it is
found.
"""
import numpy as np
import pytest

from gw.efermi import (fermi_level_step, occupied_band_count,
                       step_occupations)


def _roundtrip(E, w, n):
    return occupied_band_count(step_occupations(E, fermi_level_step(E, w, n)), w)


def test_gapped_uniform_lands_midgap():
    """VBM=1, CBM=2 at every k ⇒ E_F is the gap midpoint and the count is exact."""
    E = np.array([[0.0, 1.0, 2.0, 3.0]] * 4)
    w = np.full(4, 0.25)
    assert fermi_level_step(E, w, 2.0) == pytest.approx(1.5)
    assert _roundtrip(E, w, 2.0) == pytest.approx(2.0)


def test_gapped_with_nonuniform_ibz_weights():
    """The IBZ case: weights differ per k and the count must still be exact.

    A fixed-band-index cut cannot express this in general; a weighted
    cumulative can.
    """
    E = np.array([[0.0, 1.0, 2.0, 3.0],
                  [0.1, 1.1, 2.1, 3.1],
                  [0.2, 0.9, 2.2, 3.2]])
    w = np.array([0.0625, 0.125, 0.125])
    w = w / w.sum()
    assert _roundtrip(E, w, 2.0) == pytest.approx(2.0)


def test_metal_shortfall_is_bounded_by_the_largest_weight():
    """A step function realises only partial sums of the k-weights.

    The realised count is therefore AT MOST the target and short of it by
    less than ``max(w)``.  This is the documented, inherent limit of step
    occupations — not a defect — and it is the reason the fractional
    treatment is wanted.
    """
    rng = np.random.default_rng(0)
    E = np.sort(rng.standard_normal((10, 600)), axis=1)
    w = rng.random(10)
    w = w / w.sum()
    got = _roundtrip(E, w, 208.0)
    assert got <= 208.0 + 1e-12
    assert 208.0 - got < w.max() + 1e-12


def test_degenerate_manifold_is_refused_not_approximated():
    """E_F inside a degenerate set has no step representation — refuse.

    ``E < E_F`` takes every state at that energy or none, so any returned
    number would surface only as a wrong electron count much later.
    """
    import jax.numpy as jnp

    E = np.array([[0.0, 1.0, 1.0, 1.0, 5.0]] * 2)
    w = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="degenerate manifold"):
        fermi_level_step(E, w, 3.0)
    # And with E on a device: the condition is decoded from the kernel's
    # returned 3-vector there, and that is the path the SC loop takes.
    with pytest.raises(ValueError, match="degenerate manifold"):
        fermi_level_step(jnp.asarray(E), w, 3.0)


def test_agrees_with_sc_iteration_midgap_on_a_real_gap():
    """No second convention: on the case they share, the two agree exactly.

    ``sc_iteration._diagonalize_and_get_efermi`` uses ``0.5·(vbm+cbm)``
    with a fixed band cut.  That is right when a genuine gap separates
    band ``nocc-1`` from ``nocc`` at every k, and there this must match it
    to the last bit.  (They legitimately differ when bands overlap across
    k — the fixed cut is not the general answer, which is the reason this
    module exists.)
    """
    rng = np.random.default_rng(0)
    nk, nb, nocc = 6, 20, 8
    E = np.sort(rng.standard_normal((nk, nb)), axis=1)
    E[:, nocc:] += 10.0                      # force a real gap at the cut
    w = np.full(nk, 1.0 / nk)
    vbm = E[:, :nocc].max()
    cbm = E[:, nocc:].min()
    assert fermi_level_step(E, w, float(nocc)) == pytest.approx(
        0.5 * (vbm + cbm), abs=1e-12)
    assert _roundtrip(E, w, float(nocc)) == pytest.approx(float(nocc))


def test_kweights_shape_mismatch_refuses():
    """Full-BZ weights with IBZ energies silently rescales the count."""
    E = np.zeros((4, 3))
    with pytest.raises(ValueError, match="kweights must be"):
        fermi_level_step(E, np.full(16, 1 / 16), 1.0)


def test_occupations_are_float64_for_the_fractional_successor():
    """The dtype is the contract ρ contracts against; keep it float64.

    ``rho_from_wfns`` takes these as a WEIGHT, so the finite-temperature
    successor swaps in a Fermi–Dirac factor with no consumer changing.
    The device operand must keep the dtype: it is the one that reaches ρ.
    """
    import jax.numpy as jnp

    E = np.array([[0.0, 1.0, 2.0]])
    occ = step_occupations(E, 1.5)
    assert occ.dtype == np.float64
    assert occ.tolist() == [[1.0, 1.0, 0.0]]
    occ_dev = step_occupations(jnp.asarray(E), 1.5)
    assert occ_dev.dtype == np.float64
    assert occ_dev.tolist() == [[1.0, 1.0, 0.0]]
