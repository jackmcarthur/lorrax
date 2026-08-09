"""The crossing window's one-sided-τ completion is the OPERATOR imaginary part.

The defect this file pins (found 2026-08-09 on the Si 4×4×4 arm-b deck,
evidence `/pscratch/sd/j/jackm/sigc_star_0809/`): the crossing (HGL,
``project="imag"``) consumer in ``gw.ppm_accumulators`` completed its
one-sided τ grid with an ELEMENTWISE imaginary part, ``Im[c·σ]`` scalar by
scalar.  The sine sum it stands in for evaluates ``sin(τ·u)`` at the COMPLEX
pole argument ``u_μν = ω̃ − E ∓ Ω_μν``; the missing exponential is
``B_μν e^{+iτu_μν} = (Z†)_μν`` — the adjoint of the (μ, ν) PAIR — so the
correct completion is ``Im_op[Z] = (Z − Z†)/2i``.  The two coincide exactly
where σ^τ is complex-symmetric, which under time reversal holds only at
k ≡ −k mod G.  On the arm-b deck the elementwise form left Σ_c exactly
Hermitian at the three TRIM k and non-Hermitian by 8.8–30.3 eV at the five
non-TRIM k (Σ_c star spread 43.85 eV diag; suppressing the crossing window
took it to exactly 0), which is registered task #16's eqp0 anomaly.

Structure of this file, and why each red twin exists:

* the PHYSICS test derives the completion from an explicit Hermitian pole
  model — the missing half IS the pair adjoint, and is NOT the elementwise
  conjugate (the red half of the same assertion);
* the CONSUMER tests pin ``_project_tau_onto_omega_np``'s crossing branch to
  ``Im_op[c·X]`` through three gates, each shipped with the arm in which it
  returns FALSE: Hermiticity (red twin: the elementwise form), TRIM
  reduction (red twin: the conjugate-crossed form ``(c̄X − cX†)/2i``), and
  the sign (red twin: ``−Im_op``).  A completion that passed all three by
  accident would have to be ``Im_op`` itself.

Pure numpy — no device, no FFI, no deck.
"""

from __future__ import annotations

import numpy as np
import pytest

from gw.ppm_accumulators import _project_tau_onto_omega_np


RNG = np.random.default_rng(20260809)


def _rand(*shape, rng=RNG):
    return rng.standard_normal(shape)


def _elementwise_im_form(sigma_re, sigma_im, coeff):
    """The pre-fix consumer body, verbatim: Re(c)·S_I + Im(c)·S_R."""
    coeff_re = np.real(coeff).reshape(-1, 1, 1, 1)
    coeff_im = np.imag(coeff).reshape(-1, 1, 1, 1)
    return coeff_re * sigma_im[None, ...] + coeff_im * sigma_re[None, ...]


def _crossing(sigma_re, sigma_im, omega_vec, *, t_node=0.37 + 0.11j,
              alpha_eff=0.83 - 0.29j, omega_sign=1.0, pref=0.6180339887):
    """Call the production crossing consumer (project_code=1)."""
    return _project_tau_onto_omega_np(
        sigma_re, sigma_im, omega_vec, complex(t_node), complex(alpha_eff),
        float(omega_sign), float(pref), 1)


def _coeff(omega_vec, *, t_node=0.37 + 0.11j, alpha_eff=0.83 - 0.29j,
           omega_sign=1.0, pref=0.6180339887):
    """The consumer's own ω-kernel coefficient, reproduced for the oracles."""
    return (pref * alpha_eff) * np.exp(1j * omega_sign * omega_vec * t_node)


def _adj(a):
    """Band adjoint on the trailing (i, j) axes."""
    return np.conj(np.swapaxes(a, -1, -2))


# ---------------------------------------------------------------------------
#  The physics: the missing exponential is the PAIR adjoint, elementwise
#  conjugation is a different (wrong) object.
# ---------------------------------------------------------------------------

def test_missing_half_of_the_sine_sum_is_the_pair_adjoint():
    """Z' with Z'_μν = B_μν e^{+iτu_μν} equals Z† for Hermitian B, Ω.

    This is the derivation the fix rests on, checked as arithmetic rather
    than believed: for the PPM pole model σ^τ_μν = B_μν e^{−iτu_μν} with
    u_μν = w − Ω_μν, B† = B, Ω† = Ω (elementwise-Hermitian pole matrices,
    the production gate on B_q measures 8.6e-10 on the arm-b deck), the
    second exponential of sin(τu) is exactly the band adjoint of the first.
    The elementwise conjugate Z* is NOT that object unless Z is symmetric —
    asserted, not assumed, on the same sample.
    """
    n = 7
    B = _rand(n, n) + 1j * _rand(n, n)
    B = 0.5 * (B + B.conj().T)                       # Hermitian amplitudes
    Om = _rand(n, n) + 1j * _rand(n, n)
    Om = 0.5 * (Om + Om.conj().T)                    # Hermitian pole matrix
    tau, w = 0.83, 1.21
    u = w - Om
    Z = B * np.exp(-1j * tau * u)
    Z_missing = B * np.exp(+1j * tau * u)

    assert np.abs(Z_missing - Z.conj().T).max() < 1e-13 * np.abs(Z).max(), (
        "the second exponential of sin(τu) is not the pair adjoint — the "
        "derivation behind the crossing completion is broken")
    # Red half: the elementwise conjugate is a DIFFERENT matrix here (Z is
    # not symmetric for a generic Hermitian Ω), so a consumer built on Z*
    # is not completing the sine sum.
    assert np.abs(Z_missing - np.conj(Z)).max() > 1e-2 * np.abs(Z).max(), (
        "sample degenerated: Z came out symmetric, the elementwise red half "
        "is not being exercised")


# ---------------------------------------------------------------------------
#  Consumer gates on the production function.
# ---------------------------------------------------------------------------

def _make_channels(nk=2, nb=5, nmu=11, *, symmetric):
    """(S_R, S_I) as the two-channel kernel builds them: ψ†(Re σ)ψ, ψ†(Im σ)ψ.

    ``symmetric=True`` makes σ complex-symmetric — the TRIM situation, where
    S_R and S_I are Hermitian; ``False`` is the generic non-TRIM k.
    """
    psi = _rand(nk, nmu, nb) + 1j * _rand(nk, nmu, nb)
    sig_r = _rand(nk, nmu, nmu)
    sig_i = _rand(nk, nmu, nmu)
    if symmetric:
        sig_r = 0.5 * (sig_r + np.swapaxes(sig_r, -1, -2))
        sig_i = 0.5 * (sig_i + np.swapaxes(sig_i, -1, -2))
    S_R = np.einsum("kmi,kmn,knj->kij", psi.conj(), sig_r, psi)
    S_I = np.einsum("kmi,kmn,knj->kij", psi.conj(), sig_i, psi)
    return S_R, S_I


OMEGA = np.linspace(-1.5, 1.5, 9)


def test_crossing_contrib_is_hermitian_at_every_k_and_omega():
    """Gate 1: Im_op output is Hermitian for GENERIC σ^τ.

    The arm in which this returns FALSE is the pre-fix elementwise form on
    the same operands — asserted alongside, so a regression to it (or to
    anything elementwise-shaped) turns this file red rather than silent.
    """
    S_R, S_I = _make_channels(symmetric=False)
    out = _crossing(S_R, S_I, OMEGA)
    scale = np.abs(out).max()
    assert np.abs(out - _adj(out)).max() < 1e-13 * scale

    red = _elementwise_im_form(S_R, S_I, _coeff(OMEGA))
    assert np.abs(red - _adj(red)).max() > 1e-2 * np.abs(red).max(), (
        "the elementwise red twin came out Hermitian on a generic sample — "
        "the Hermiticity gate above is not discriminating anything")


def test_crossing_contrib_reduces_to_elementwise_at_trim():
    """Gate 2: at complex-symmetric σ^τ (TRIM), Im_op == the old elementwise
    form, algebraically — so every TRIM k of every frozen reference is
    preserved by the fix.  This is the invariance half of the arm-b
    measurement (the three TRIM k were correct before and after).
    """
    S_R, S_I = _make_channels(symmetric=True)
    out = _crossing(S_R, S_I, OMEGA)
    old = _elementwise_im_form(S_R, S_I, _coeff(OMEGA))
    scale = np.abs(old).max()
    assert np.abs(out - old).max() < 1e-13 * scale


def test_conjugate_crossed_twin_breaks_the_trim_reduction():
    """Red twin for gate 2: (c̄·X − c·X†)/2i — the conjugation on the WRONG
    operand — is Hermitian too, but does NOT reduce to the elementwise form
    at TRIM (it gives Re(c)·S_I − Im(c)·S_R there).  This is the arm that
    proves the TRIM-reduction gate discriminates conjugation conventions
    rather than passing any Hermitian-by-construction completion.
    """
    S_R, S_I = _make_channels(symmetric=True)
    c = _coeff(OMEGA).reshape(-1, 1, 1, 1)
    X = S_R + 1j * S_I
    crossed = (np.conj(c) * X[None] - c * _adj(X)[None]) / 2j
    # it IS Hermitian — Hermiticity alone cannot reject it...
    assert np.abs(crossed - _adj(crossed)).max() < 1e-13 * np.abs(crossed).max()
    # ...and the TRIM gate does.
    old = _elementwise_im_form(S_R, S_I, _coeff(OMEGA))
    assert np.abs(crossed - old).max() > 1e-2 * np.abs(old).max(), (
        "conjugate-crossed twin agreed at TRIM — the coefficient sample has "
        "no imaginary part and the reduction gate is a tautology")


def test_sign_flipped_twin_breaks_the_trim_reduction():
    """Red twin for the sign: −Im_op is Hermitian and equivariant, and the
    TRIM gate rejects it.  (A star-spread measurement alone would accept a
    global sign error; this is the gate that does not.)"""
    S_R, S_I = _make_channels(symmetric=True)
    out = _crossing(S_R, S_I, OMEGA)
    old = _elementwise_im_form(S_R, S_I, _coeff(OMEGA))
    assert np.abs((-out) - old).max() > 1e-2 * np.abs(old).max()


def test_scalar_channel_unchanged():
    """REAL (S_R, S_I) at nb = 1: Im_op degenerates to the elementwise Im,
    so the scalar τ-loop analogue (test_mpa_sigma_pass feeds
    ``np.real(sig)`` / ``np.imag(sig)`` — real 1×1 channels) is preserved
    to fp roundoff and its frozen numbers cannot move.  Pinned here so that
    statement is a measurement.  NOTE the realness is load-bearing: a
    COMPLEX 1×1 S_R encodes a non-symmetric σ, where the elementwise form
    was wrong even at nb = 1 and the two forms must NOT agree.
    """
    S_R = _rand(3, 1, 1)
    S_I = _rand(3, 1, 1)
    out = _crossing(S_R, S_I, OMEGA)
    old = _elementwise_im_form(S_R, S_I, _coeff(OMEGA))
    scale = np.abs(old).max()
    assert np.abs(out - old).max() < 1e-15 * max(scale, 1.0)


def test_dispatch_guards_still_refuse_cross_pairing():
    """The load-bearing dispatch guards survive the fix: a merged tile at
    project_code=1 and a two-channel pair at project_code=0 both raise."""
    S_R, S_I = _make_channels(nk=1, nb=2, nmu=3, symmetric=False)
    with pytest.raises(ValueError, match="crossing"):
        _project_tau_onto_omega_np(
            S_R + 1j * S_I, None, OMEGA, 0.1 + 0.1j, 1.0 + 0.0j, 1.0, 1.0, 1)
    with pytest.raises(ValueError, match="Laplace"):
        _project_tau_onto_omega_np(
            S_R, S_I, OMEGA, 0.1 + 0.1j, 1.0 + 0.0j, 1.0, 1.0, 0)
