"""The complex-z chi0 assembly: kernel, convention, and the whole sum.

These are the checks that do NOT need a deck.  The check that does --
that this route reproduces ``w_isdf.compute_chi0`` at z = 0 to the
minimax quadrature's own error -- is the bridge gate, and it is measured
on the real Si deck rather than asserted here; see
``~/lorrax_service_phase/SI_MPA_FIRST_LIGHT.md``.  What is here is
everything that can go wrong WITHOUT a device: the kernel's closed form,
the k-shift convention, the FFT-normalisation residue, and the claim
that the blocked/scanned implementation computes the same sum as a
transparent triple loop.
"""

import numpy as np
import pytest

from gw.mpa import chi0_resolvent as cres
from gw.mpa import evaluator, sample_plan


def _reference_chi0(psi, enk, val, cond, z_values, q_int, kgrid):
    """The sum, written out.  Slow, obvious, and the thing under test.

    ``chi0_q(m, n) = C * sum_{k,c,v} K_z(D) M(m) conj(M(n))`` with
    ``M = sum_s conj(psi_c[k-q]) psi_v[k]`` and ``D = eps_c[k-q] -
    eps_v[k]`` -- ``bse_w_exact.build_finite_q_data``'s convention,
    spelled out with explicit loops so that a reader can check it against
    that docstring without decoding an einsum.
    """

    n_k, _, _, n_mu = psi.shape
    psi_c = cres.roll_k_axis(psi[:, cond], q_int, kgrid)
    eps_c = cres.roll_k_axis(enk[:, cond], q_int, kgrid)
    psi_v, eps_v = psi[:, val], enk[:, val]
    out = np.zeros((len(z_values), n_mu, n_mu), dtype=np.complex128)
    for k in range(n_k):
        for c in range(psi_c.shape[1]):
            for v in range(psi_v.shape[1]):
                M = np.einsum("sm,sm->m", np.conj(psi_c[k, c]), psi_v[k, v])
                d = eps_c[k, c] - eps_v[k, v]
                for j, z in enumerate(z_values):
                    out[j] += evaluator.damped_kernel(z, d) * np.outer(
                        M, np.conj(M))
    return out * cres.chi0_ortho_norm(n_k)


@pytest.fixture
def toy():
    """A 2x2x2 deck's worth of psi/enk.  Deterministic, tiny, complex."""

    rng = np.random.default_rng(20260808)
    kgrid = (2, 2, 2)
    n_k, n_b, n_s, n_mu = 8, 5, 2, 6
    psi = (rng.normal(size=(n_k, n_b, n_s, n_mu))
           + 1j * rng.normal(size=(n_k, n_b, n_s, n_mu)))
    # A gapped spectrum, so no transition energy lands on a sample point.
    enk = np.sort(rng.normal(size=(n_k, n_b)), axis=1)
    enk[:, 2:] += 3.0
    return psi, enk, slice(0, 2), slice(2, n_b), kgrid


def test_the_kernel_is_the_evaluator_s_closed_form(toy):
    """The device weight and the host oracle are one function.

    ``chi0_resolvent`` inlines ``K_z(Delta)`` because ``damped_kernel``
    is host numpy, so the two forms can drift.  A one-transition, one-mu
    system makes chi0 proportional to the kernel and pins them together.
    """

    psi = np.ones((1, 2, 1, 1), dtype=np.complex128)
    enk = np.array([[0.0, 2.0]])
    z = np.array([0.0 + 0.0j, 0.5 + 0.1j, 0.0 + 1.0j, 2.5 + 1.0j])
    got = np.asarray(cres.chi0_resolvent(
        psi, enk, type("S", (), {"val": slice(0, 1), "cond": slice(1, 2)}),
        z, (0, 0, 0), (1, 1, 1)))
    want = evaluator.damped_kernel(z, 2.0) * cres.chi0_ortho_norm(1)
    np.testing.assert_allclose(got[:, 0, 0], want, rtol=0, atol=1e-14)


def test_the_ortho_norm_is_the_production_kernel_s(toy):
    """``n_k**-0.5``, and it is not decoration.

    Three ``norm='ortho'`` FFTs leave exactly this residue; getting it
    wrong scales every W silently.  The RED TWIN is the value a reader
    would guess -- 1, or ``1/n_k`` -- and the assertion is that neither
    is what the production kernel leaves behind.
    """

    assert cres.chi0_ortho_norm(64) == pytest.approx(0.125)
    assert cres.chi0_ortho_norm(1) == pytest.approx(1.0)
    assert cres.chi0_ortho_norm(64) != pytest.approx(1.0 / 64)


def test_the_roll_is_bse_w_exact_s_shift(toy):
    """``out[k] = arr[k - q]``, the SAME shift the BSE resolvent uses.

    Duplicated code, so it needs a test that pins it to its source
    rather than to itself.  ``bse_w_exact._roll_k_axis_host`` is the
    source and ``test_bse_w0_resolvent`` is what makes that source
    trustworthy against a real stored W_q tile.
    """

    from bse.bse_w_exact import _roll_k_axis_host

    rng = np.random.default_rng(7)
    arr = rng.normal(size=(8, 3, 2))
    for q in [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)]:
        np.testing.assert_array_equal(
            cres.roll_k_axis(arr, q, (2, 2, 2)),
            _roll_k_axis_host(arr, q, 2, 2, 2))


@pytest.mark.parametrize("q_int", [(0, 0, 0), (0, 1, 0), (1, 1, 1)])
def test_the_blocked_sum_equals_the_written_out_sum(toy, q_int):
    """The scanned, k-blocked implementation IS the triple loop.

    Exercised at every q of the toy grid because the k-shift is the one
    place a finite-q bug can hide -- at q = 0 the roll is the identity
    and a wrong shift would pass.
    """

    psi, enk, val, cond, kgrid = toy
    z = np.array([0.0 + 0.0j, 0.4 + 0.1j, 1.7 + 0.1j, 0.0 + 1.0j,
                  1.7 + 1.0j])
    slices = type("S", (), {"val": val, "cond": cond})
    want = _reference_chi0(psi, enk, val, cond, z, q_int, kgrid)
    for k_block in (1, 3, 8):
        got = np.asarray(cres.chi0_resolvent(
            psi, enk, slices, z, q_int, kgrid, k_block=k_block))
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_chi0_at_z_zero_is_real_symmetric(toy):
    """z = 0 gives a Hermitian chi0; a strip sample does not.

    Not a formality: the static chi0 the production kernel returns is
    Hermitian, and a conjugation slip in the pair amplitude would break
    that while leaving the magnitudes plausible.  The strip half of the
    assertion is the control -- it says the test can tell the two apart.
    """

    psi, enk, val, cond, kgrid = toy
    slices = type("S", (), {"val": val, "cond": cond})
    out = np.asarray(cres.chi0_resolvent(
        psi, enk, slices, np.array([0.0 + 0.0j, 1.7 + 0.1j]), (0, 1, 0),
        kgrid))
    np.testing.assert_allclose(out[0], out[0].conj().T, rtol=1e-12,
                               atol=1e-12)
    assert not np.allclose(out[1], out[1].conj().T, rtol=1e-6, atol=1e-6)


def test_x64_is_required(toy, monkeypatch):
    """Single precision is refused BY NAME, not tolerated.

    A strip sample sits near its own pole; in float32 it loses the
    imaginary part that makes it well posed, and the failure downstream
    is a bad fit rather than an exception.
    """

    import jax

    monkeypatch.setattr(jax.config, "read", lambda name: False)
    psi, enk, val, cond, kgrid = toy
    slices = type("S", (), {"val": val, "cond": cond})
    with pytest.raises(RuntimeError, match="GATE x64_enabled"):
        cres.chi0_resolvent(psi, enk, slices, np.array([0.0 + 0.0j]),
                            (0, 0, 0), kgrid)


def test_the_kernel_factor_is_shared_with_sample_plan():
    """-2 lives in one place.

    ``sample_plan.KERNEL_FACTOR`` is the 2x2 table's own statement of
    what ``K_z`` is; this module reads it rather than restating it, and
    the evaluator's oracle does the same.
    """

    assert sample_plan.KERNEL_FACTOR == -2.0
    d = np.array([1.5])
    np.testing.assert_allclose(
        evaluator.damped_kernel(0.0 + 0.0j, d),
        sample_plan.KERNEL_FACTOR / d, rtol=0, atol=1e-15)


def test_cost_model_is_arithmetically_consistent():
    """The cost report's parts multiply up to its totals."""

    c = cres.cost_model(n_k=64, n_v=8, n_c=92, n_mu=1128, n_z=16, n_q=8)
    assert c["n_transitions_per_q"] == 64 * 8 * 92
    assert c["flops_per_sample_per_q"] == pytest.approx(
        8.0 * 1128 ** 2 * c["n_transitions_per_q"])
    assert c["flops_total"] == pytest.approx(
        c["flops_per_sample_per_q"] * 16 * 8)
    assert c["n_logical_outputs"] == 16 * 8 == c["n_dyson_solves"]
    assert c["store_bytes"] == pytest.approx(16.0 * 16 * 8 * 1128 ** 2)
