"""The crossing window's one-sided-τ completion, and where it has to happen.

The defect this file pins (KNOWN_FAILURES, found 2026-08-09 by the MPA
three-way-table lane): the crossing (HGL, ``project="imag"``) consumer in
``gw.ppm_accumulators`` completed its one-sided τ grid with an ELEMENTWISE
imaginary part — ``Re(c)·S_I + Im(c)·S_R``, which is ``Im[c·σ(μ,ν)]`` taken
scalar by scalar before the ψ projection.  The sum it stands in for is a SINE
sum, ``G(u) ≈ Σ_l α_l sin(τ_l u)``, and ``sin(τu) = (e^{+iτu} − e^{−iτu})/2i``
carries a second exponential that is the adjoint of the (μ, ν) PAIR, not the
conjugate of each scalar.  The two coincide exactly where σ^τ is
complex-symmetric, which under time reversal holds only at k ≡ −k mod G — so
a check taken at a TRIM k alone comes back green and says NOTHING about the
other k.  That discriminator is pinned here on purpose
(:func:`test_kstar_relation_is_green_at_trim_for_BOTH_forms`).

Two independent things are gated, because the fix has two halves:

**The algebra.**  ``(Z − Z†)/2i`` with ``Z = Σ_τ coeff·X`` is the completion.
Each gate ships with the arm in which it returns FALSE: Hermiticity (red
twin: the elementwise form), the TRIM reduction (red twins: the
conjugate-crossed form ``(c̄X − cX†)/2i``, and ``−Im_op``), and the k-star
relation itself.

**The placement.**  ``(Z − Z†)`` pairs band element (i, j) with (j, i), and
Σ_c tiles are sharded ``P(None, None, 'x', 'y')`` — m over ``'x'``, n over
``'y'`` — so the partner of every element on rank (a, b) lives on rank (b, a).
A ``swapaxes(-1, -2)`` applied inside the per-τ, per-shard projector is
therefore NOT the band adjoint: on a square mesh the shapes match and it
returns a wrong answer in silence.  The last two tests are that measurement
on a 2×2 emulated mesh, and they are the reason the completion is a
window-level operator in ``_TauAccumulator._finish_window`` rather than a
line in ``_project_tau_onto_omega_np``.

The algebra half is pure numpy — no device, no FFI, no deck.  The placement
half needs four devices::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_ENABLE_X64=1 \
        python -m pytest tests/test_ppm_crossing_completion.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.ppm_accumulators import (
    _TauAccumulator,
    _complete_one_sided_tau,
    _project_tau_onto_omega_np,
)


RNG = np.random.default_rng(20260809)

OMEGA = np.linspace(-1.5, 1.5, 5)
T_NODE = 0.37 + 0.0j          # crossing windows carry a REAL t = τ/ξ
ALPHA_EFF = 0.83 - 0.29j
OMEGA_SIGN = 1.0
PREF = 0.6180339887


def _rand(*shape):
    return RNG.standard_normal(shape)


def _coeff(omega_vec=OMEGA):
    """The consumer's own ω-kernel coefficient, reproduced for the oracles."""
    return (PREF * ALPHA_EFF) * np.exp(1j * OMEGA_SIGN * omega_vec * T_NODE)


def _adj(a):
    """Band adjoint on the trailing (i, j) axes."""
    return np.conj(np.swapaxes(a, -1, -2))


def _one_sided(S_R, S_I, omega_vec=OMEGA):
    """Call the production crossing consumer (project_code=1)."""
    return _project_tau_onto_omega_np(
        S_R, S_I, omega_vec, complex(T_NODE), complex(ALPHA_EFF),
        float(OMEGA_SIGN), float(PREF), 1)


def _complete(Z):
    """The window-level completion on an unsplit band matrix (numpy path)."""
    return _complete_one_sided_tau([Z], None, None)[0]


def _elementwise_im_form(S_R, S_I, omega_vec=OMEGA):
    """The PRE-FIX consumer body, verbatim: Re(c)·S_I + Im(c)·S_R."""
    c = _coeff(omega_vec)
    cr = np.real(c).reshape(-1, 1, 1, 1)
    ci = np.imag(c).reshape(-1, 1, 1, 1)
    return cr * S_I[None, ...] + ci * S_R[None, ...]


def _channels(psi, sig_r, sig_i):
    """(S_R, S_I) as the two-channel kernel builds them: ψ†(Re σ)ψ, ψ†(Im σ)ψ."""
    S_R = np.einsum("kmi,kmn,knj->kij", psi.conj(), sig_r, psi)
    S_I = np.einsum("kmi,kmn,knj->kij", psi.conj(), sig_i, psi)
    return S_R, S_I


def _make_channels(nk=2, nb=5, nmu=11, *, complex_symmetric):
    """σ^τ either complex-symmetric (the TRIM situation) or generic.

    ``complex_symmetric=True`` makes σ_R and σ_I real SYMMETRIC, i.e. σ^τ
    complex-symmetric — which is exactly the condition under which S_R and
    S_I come out Hermitian and the elementwise form is right.
    """
    psi = _rand(nk, nmu, nb) + 1j * _rand(nk, nmu, nb)
    sig_r, sig_i = _rand(nk, nmu, nmu), _rand(nk, nmu, nmu)
    if complex_symmetric:
        sig_r = 0.5 * (sig_r + np.swapaxes(sig_r, -1, -2))
        sig_i = 0.5 * (sig_i + np.swapaxes(sig_i, -1, -2))
    return _channels(psi, sig_r, sig_i)


# ---------------------------------------------------------------------------
#  The derivation: the missing exponential is the PAIR adjoint.
# ---------------------------------------------------------------------------

def test_missing_half_of_the_sine_sum_is_the_pair_adjoint():
    """``Z'_μν = B_μν e^{+iτu_μν}`` equals ``Z†`` for Hermitian B, symmetric Ω.

    This is the step the whole fix rests on, checked as arithmetic rather
    than believed.  The GN-PPM fit produces ``B_q† = B_q`` and ``Ω_q`` real
    symmetric (``DERIVATION_channel_hermiticity.md`` §1.3), and the τ kernel
    builds ``σ^τ_μν ∝ B_μν e^{−iτ(E+Ω_μν)}``; the second exponential of
    ``sin(τu)`` is then exactly the band adjoint of the first.  The
    elementwise conjugate is a DIFFERENT matrix — asserted on the same
    sample, so a degenerate draw cannot make this test vacuous.
    """
    n = 7
    B = _rand(n, n) + 1j * _rand(n, n)
    B = 0.5 * (B + B.conj().T)                    # Hermitian amplitudes
    Om = _rand(n, n)
    Om = 0.5 * (Om + Om.T)                        # real symmetric pole matrix
    tau, w = 0.83, 1.21
    u = w - Om
    Z = B * np.exp(-1j * tau * u)
    Z_missing = B * np.exp(+1j * tau * u)

    assert np.abs(Z_missing - Z.conj().T).max() < 1e-13 * np.abs(Z).max(), (
        "the second exponential of sin(τu) is not the pair adjoint — the "
        "derivation behind the crossing completion is broken")
    assert np.abs(Z_missing - np.conj(Z)).max() > 1e-2 * np.abs(Z).max(), (
        "sample degenerated: Z came out symmetric, so the elementwise red "
        "half is not being exercised")


# ---------------------------------------------------------------------------
#  The per-τ consumer now returns the one-sided half, and nothing else.
# ---------------------------------------------------------------------------

def test_crossing_consumer_returns_the_one_sided_half():
    """project_code=1 must hand back ``coeff·X``, X = S_R + i·S_I.

    The completion is deliberately NOT here (it is not elementwise in
    (i, j), so it cannot be — see the sharding tests at the bottom).  The
    red arm is the pre-fix body, which this must no longer equal.
    """
    S_R, S_I = _make_channels(complex_symmetric=False)
    out = _one_sided(S_R, S_I)
    want = _coeff().reshape(-1, 1, 1, 1) * (S_R + 1j * S_I)[None, ...]
    assert out.dtype == np.complex128
    assert np.abs(out - want).max() < 1e-14 * np.abs(want).max()

    red = _elementwise_im_form(S_R, S_I)
    assert np.abs(out - red).max() > 1e-2 * np.abs(red).max(), (
        "the crossing consumer still returns the elementwise form")


def test_dispatch_guards_still_refuse_cross_pairing():
    """MPA's merged crossing carrier works; a Laplace pair still refuses."""
    S_R, S_I = _make_channels(nk=1, nb=2, nmu=3, complex_symmetric=False)
    merged = _project_tau_onto_omega_np(
        S_R + 1j * S_I, None, OMEGA,
        0.1 + 0.0j, 1.0 + 0.0j, 1.0, 1.0, 1)
    want = np.exp(1j * OMEGA * 0.1).reshape(-1, 1, 1, 1) * (
        S_R + 1j * S_I)[None]
    np.testing.assert_allclose(merged, want, rtol=2e-15, atol=2e-15)
    with pytest.raises(ValueError, match="Laplace"):
        _project_tau_onto_omega_np(
            S_R, S_I, OMEGA, 0.1 + 0.0j, 1.0 + 0.0j, 1.0, 1.0, 0)


# ---------------------------------------------------------------------------
#  The completion operator: Hermiticity, the TRIM reduction, and two twins.
# ---------------------------------------------------------------------------

def test_completed_window_is_hermitian_at_generic_k():
    """Gate 1: ``(Z − Z†)/2i`` is Hermitian for a GENERIC σ^τ.

    Σ_c must be Hermitian in the band pair: every Laplace window contributes
    a REAL coefficient times a Hermitian X (σ^τ Hermitian on the imaginary-t
    axis), and with this fix every crossing window contributes a manifestly
    Hermitian object.  The arm in which it returns FALSE is the pre-fix
    elementwise form on the same operands — 8.8–30.3 eV of non-Hermiticity
    at the five non-TRIM k of the Si arm-b deck.
    """
    S_R, S_I = _make_channels(complex_symmetric=False)
    out = _complete(_one_sided(S_R, S_I))
    assert np.abs(out - _adj(out)).max() < 1e-13 * np.abs(out).max()

    red = _elementwise_im_form(S_R, S_I)
    assert np.abs(red - _adj(red)).max() > 1e-2 * np.abs(red).max(), (
        "the elementwise red twin came out Hermitian on a generic sample — "
        "the Hermiticity gate is not discriminating anything")


def test_completion_reduces_to_the_elementwise_form_at_trim():
    """Gate 2: where σ^τ is complex-symmetric, the fix is a no-op.

    σ_R, σ_I real symmetric ⇒ S_R, S_I Hermitian ⇒
    ``(cX − (cX)†)/2i = Im(c)·S_R + Re(c)·S_I``, the old body exactly.  That
    is why every TRIM k of every frozen reference is preserved — ALGEBRAICALLY.
    It is NOT bitwise: the new form associates one complex multiply and a
    subtract where the old associated two real-scaled adds, so the agreement
    is at fp roundoff (measured below), not at zero ulp.
    """
    S_R, S_I = _make_channels(complex_symmetric=True)
    out = _complete(_one_sided(S_R, S_I))
    old = _elementwise_im_form(S_R, S_I)
    resid = np.abs(out - old).max() / np.abs(old).max()
    assert resid < 1e-14, f"TRIM reduction broke: relative residual {resid:.3e}"
    assert resid > 0.0, (
        "bit-identity at TRIM would be a surprise, not a relief — if this "
        "ever fires, the arithmetic changed and the claim above needs a "
        "re-measurement, not a looser tolerance")


def test_conjugate_crossed_twin_breaks_the_trim_reduction():
    """Red twin for gate 2: ``(c̄X − cX†)/2i`` — the conjugation on the WRONG
    operand — is Hermitian too, so Hermiticity alone cannot reject it.  The
    TRIM reduction does.  (Measured on the deck: the crossed twin is
    Hermitian AND star-spread-0, while moving TRIM eqp0 rows by 7.4–19.7 eV.)
    """
    S_R, S_I = _make_channels(complex_symmetric=True)
    c = _coeff().reshape(-1, 1, 1, 1)
    X = (S_R + 1j * S_I)[None, ...]
    crossed = (np.conj(c) * X - c * _adj(X)) / 2j
    assert np.abs(crossed - _adj(crossed)).max() < 1e-13 * np.abs(crossed).max()
    old = _elementwise_im_form(S_R, S_I)
    assert np.abs(crossed - old).max() > 1e-2 * np.abs(old).max()


def test_sign_flipped_twin_breaks_the_trim_reduction():
    """Red twin for the sign: ``−Im_op`` is Hermitian and star-covariant, and
    only the TRIM reduction rejects it.  A spread statistic alone would
    accept a global sign error."""
    S_R, S_I = _make_channels(complex_symmetric=True)
    out = _complete(_one_sided(S_R, S_I))
    old = _elementwise_im_form(S_R, S_I)
    assert np.abs((-out) - old).max() > 1e-2 * np.abs(old).max()


def test_real_scalar_channel_is_preserved():
    """REAL (S_R, S_I) at nb = 1: the completion degenerates to the
    elementwise Im, so the scalar τ-loop analogue (``test_mpa_sigma_pass``
    feeds ``np.real(sig)`` / ``np.imag(sig)`` — real 1×1 channels) cannot
    move.  Pinned so that statement is a measurement rather than a hope."""
    S_R, S_I = _rand(3, 1, 1), _rand(3, 1, 1)
    out = _complete(_one_sided(S_R, S_I))
    old = _elementwise_im_form(S_R, S_I)
    assert np.abs(out - old).max() < 1e-15 * max(np.abs(old).max(), 1.0)


# ---------------------------------------------------------------------------
#  The k-star gate.
#
#  THE RELATION, stated: Σ_c(ω, k, i, j) is a BAND-INDEX quantity, and a band
#  index is symmetry-inert (symmetry_maps.maps, the note above star_select).
#  So for a pair of k related by time reversal, KStarMap.broadcast fills the
#  TRS member by CONJUGATION —
#
#       Σ_c(−k) = conj( Σ_c(k) ),
#
#  which is the relation KStarMap.spread/spread_rel measures the residual of,
#  and the one the deck reported 43.85 eV of.  It factors into two halves:
#  the TRS operand relation X_{−k} = X_k^T (no conjugation; derivation §3.3,
#  from ψ_{−k} = ψ_k^* and σ^τ_{−k} = σ^{τ,T}_k), and Hermiticity Σ = Σ†.
#  The elementwise form satisfies the FIRST and violates the SECOND — which
#  is why the star residual, not the transpose relation, is what went red.
# ---------------------------------------------------------------------------

def _trs_pair(*, trim, nb=5, nmu=11):
    """(S_R, S_I) at +k and at −k, built from an explicit TRS partner.

    ``ns=1``, so the literal deck gauge is ψ_{−k} = ψ_k^*, and the operand
    relation the derivation gives is σ^τ_{−k} = σ^{τ,T}_k.  ``trim=True``
    is the self-paired case −k ≡ k: the partner construction must then
    return the SAME point, which forces σ^τ complex-symmetric AND ψ real.
    """
    psi = _rand(nmu, nb) + 1j * _rand(nmu, nb)
    sig_r, sig_i = _rand(nmu, nmu), _rand(nmu, nmu)
    if trim:
        psi = np.real(psi) + 0j                       # ψ_k = ψ_k^*
        sig_r = 0.5 * (sig_r + sig_r.T)               # σ^τ = σ^{τ,T}
        sig_i = 0.5 * (sig_i + sig_i.T)
    psi_m = np.conj(psi)
    plus = _channels(psi[None], sig_r[None], sig_i[None])
    minus = _channels(psi_m[None], sig_r.T[None], sig_i.T[None])
    # The operand-level half of the relation, checked before it is used.
    assert np.abs(minus[0] - np.swapaxes(plus[0], -1, -2)).max() < 1e-11
    assert np.abs(minus[1] - np.swapaxes(plus[1], -1, -2)).max() < 1e-11
    return plus, minus


def _star_residual(sig_plus, sig_minus):
    """max |Σ_c(−k) − conj(Σ_c(k))|, relative — the KStarMap.spread_rel form."""
    scale = max(np.abs(sig_plus).max(), 1e-300)
    return float(np.abs(sig_minus - np.conj(sig_plus)).max() / scale)


def test_kstar_relation_holds_after_the_fix_at_a_non_trim_k():
    """Gate 3 (the k-star gate): Σ_c(−k) = conj(Σ_c(k)) at a generic k.

    Green with the completion; the arm in which it returns FALSE is the
    pre-fix elementwise form on the SAME ±k pair.
    """
    (SRp, SIp), (SRm, SIm) = _trs_pair(trim=False)
    fixed_p = _complete(_one_sided(SRp, SIp))
    fixed_m = _complete(_one_sided(SRm, SIm))
    assert _star_residual(fixed_p, fixed_m) < 1e-12

    red_p = _elementwise_im_form(SRp, SIp)
    red_m = _elementwise_im_form(SRm, SIm)
    assert _star_residual(red_p, red_m) > 1e-2, (
        "the elementwise form satisfied the star relation at a non-TRIM k — "
        "this gate has stopped reproducing the defect it exists for")


def test_kstar_relation_is_green_at_trim_for_BOTH_forms():
    """The discriminator, pinned: at k ≡ −k the pre-fix form passes too.

    This is why a TRIM-only check must never be read as coverage — it is
    green on the defect.  KNOWN_FAILURES carries the same sentence; this is
    the executable half of it.
    """
    (SRp, SIp), (SRm, SIm) = _trs_pair(trim=True)
    fixed_p = _complete(_one_sided(SRp, SIp))
    fixed_m = _complete(_one_sided(SRm, SIm))
    assert _star_residual(fixed_p, fixed_m) < 1e-12

    red_p = _elementwise_im_form(SRp, SIp)
    red_m = _elementwise_im_form(SRm, SIm)
    assert _star_residual(red_p, red_m) < 1e-12, (
        "the TRIM construction is not actually self-paired — the "
        "'coincide only at k = −k' discriminator is not being demonstrated")


# ---------------------------------------------------------------------------
#  The placement: the adjoint is NOT a per-shard operation.
# ---------------------------------------------------------------------------

#: FOUR DEVICES, and the suite is now able to supply them.  This was a
#: ``skipif(jax.device_count() < 4)``, which SKIPPED in every suite run
#: whatever the node had: tests/conftest.py pins each test process to one
#: GPU, so ``device_count()`` is 1 by construction and the placement half of
#: this file was never exercised except by hand.  The marker states the
#: requirement and lets the conftest satisfy it (real GPUs on a >=4-GPU
#: node, emulated devices when the caller supplied them, skip only when the
#: hardware is genuinely absent).
pytest_sharded = pytest.mark.mesh(4)


class _RecordingSink:
    """A _WindowSink that assembles the global Σ from whatever it is handed."""

    def __init__(self, gshape):
        self._g = np.zeros(gshape, dtype=np.complex128)

    def consume_window(self, win_shards, shard_index, shard_devices) -> None:
        for tile, ix in zip(win_shards, shard_index):
            self._g[(slice(None),) + tuple(ix)] += tile

    def result(self):
        return self._g


def _crossing_window():
    return SimpleNamespace(omega_sign=OMEGA_SIGN, prefactor=PREF, project_code=1)


def _run_accumulator(S_R_g, S_I_g, mesh, *, n_tau=3):
    """Drive the production accumulator over ``n_tau`` τ on ``mesh``."""
    nk, nb, _ = S_R_g.shape
    band = NamedSharding(mesh, P(None, 'x', 'y'))
    sr = jax.device_put(S_R_g, band)
    si = jax.device_put(S_I_g, band)
    sink = _RecordingSink((OMEGA.size, nk, nb, nb))
    acc = _TauAccumulator(omega_vec=np.asarray(OMEGA, dtype=np.complex128),
                          sink=sink)
    acc.begin_window(_crossing_window())
    for _ in range(n_tau):
        acc.add_tau(sr, si, complex(T_NODE), complex(ALPHA_EFF))
    acc.end_window()
    return acc.finalize()


def _reference(S_R_g, S_I_g, n_tau=3):
    """(Z − Z†)/2i with Z = Σ_τ coeff·X, computed globally in numpy."""
    Z = n_tau * (_coeff().reshape(-1, 1, 1, 1)
                 * (S_R_g + 1j * S_I_g)[None, ...])
    return (Z - _adj(Z)) / 2j


@pytest_sharded
def test_sharded_band_axes_give_the_same_completion_as_one_device():
    """The whole point: the answer must not depend on the mesh.

    m is reduce-scattered over ``'x'`` and n over ``'y'``, so on a 2×2 mesh
    each rank holds a band BLOCK and the (j, i) partner of its elements is
    on another rank.  Running the production accumulator on 1×1 and on 2×2
    must give the same Σ, and both must equal the numpy oracle.
    """
    nk, nb = 2, 6
    S_R_g, S_I_g = _make_channels(nk=nk, nb=nb, nmu=9, complex_symmetric=False)
    want = _reference(S_R_g, S_I_g)
    devs = jax.devices()

    one = Mesh(np.array(devs[:1]).reshape(1, 1), ('x', 'y'))
    got_1 = _run_accumulator(S_R_g, S_I_g, one)
    assert np.abs(got_1 - want).max() < 1e-11 * np.abs(want).max()

    four = Mesh(np.array(devs[:4]).reshape(2, 2), ('x', 'y'))
    got_4 = _run_accumulator(S_R_g, S_I_g, four)
    assert np.abs(got_4 - want).max() < 1e-11 * np.abs(want).max(), (
        "the crossing completion changed answer when the band axes were "
        "split — the adjoint is being taken inside a shard")


@pytest_sharded
def test_per_shard_local_adjoint_is_a_different_object_on_a_split_mesh():
    """Red twin for the PLACEMENT, and the reason this fix is not a one-liner.

    Applying ``(t − conj(swapaxes(t)))/2j`` to each per-rank tile — the
    natural, and wrong, place for it — agrees with the global completion
    when the band axes are unsplit (which is why a ``-G=1`` deck leg cannot
    see the difference) and DISAGREES at O(1) the moment they are split.  On
    a non-square mesh it would not even be shape-legal.
    """
    nk, nb = 2, 6
    S_R_g, S_I_g = _make_channels(nk=nk, nb=nb, nmu=9, complex_symmetric=False)
    want = _reference(S_R_g, S_I_g)
    Z = 3 * (_coeff().reshape(-1, 1, 1, 1) * (S_R_g + 1j * S_I_g)[None, ...])

    def per_shard(p_m, p_n):
        out = np.zeros_like(Z)
        bm, bn = nb // p_m, nb // p_n
        for a in range(p_m):
            for b in range(p_n):
                sl = (slice(None), slice(None),
                      slice(a * bm, (a + 1) * bm), slice(b * bn, (b + 1) * bn))
                t = Z[sl]
                out[sl] = (t - np.conj(np.swapaxes(t, -1, -2))) / 2j
        return out

    scale = np.abs(want).max()
    assert np.abs(per_shard(1, 1) - want).max() < 1e-13 * scale, (
        "the per-shard form must agree when the band axes are UNSPLIT — "
        "otherwise this twin is measuring something other than the sharding")
    assert np.abs(per_shard(2, 2) - want).max() > 1e-2 * scale, (
        "the per-shard local adjoint agreed with the global one on a split "
        "band mesh — the sharding hazard this fix exists to avoid is not "
        "being reproduced")
