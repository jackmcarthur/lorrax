"""The q != 0 Coulomb head-channel placement (``mc_average_placement``).

Four things are asserted here, and the first is the one the whole feature
rests on:

1. **The identity.**  ``gw.head_channel`` computes BerkeleyGW's placement
   ``W_head = eps_c^-1 <v>`` by taking a REAL scalar combination of two
   single-V Dyson solves, ``W = W_body0 + r (W_bare - W_body0)`` with
   ``r = <v>/v_c``.  The claim that this equals the head-channel Schur form
   ``gw.experimental.head_wing_schur`` implements — the same body solve,
   the same wings, and the mini-BZ average on the head SCALAR only — is
   checked against an independent numpy construction of that Schur form,
   at rank 1 and at the tie multiplicities 2 and 4 that Si's 4x4x4 grid
   actually produces.  If this test is green the two-solve arithmetic path
   and the Schur algebraic path are the same object.

2. **Hermiticity is structural.**  ``r`` is real, so the combination of two
   congruence-Hermitian solves is Hermitian to round-off, at every ``r``.
   This is the property a two-tile ``(I - V_bare chi)^-1 V_avg`` form
   breaks and ``test_alpha_herm_head_slot_link``'s ``ALPHA_HERM_RTOL``
   asserts; it is checked here on the same synthetic problems.

3. **The default is inert, by construction.**  ``r = 1`` (Gamma, a cut head
   slot, or ``<v> == v_c``) returns the un-rescaled solve, and the mode
   string ``off`` is what an absent deck key resolves to.

4. **The knob refuses rather than no-ops.**  An unknown spelling and the
   unimplemented ``schur_avg`` both raise with the valid list in the
   message — the ``vhead`` parser lesson (a validation knob that quietly
   does nothing is worse than no knob).

numpy only: no jax, no mesh, no fixture deck.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from gw.head_channel import (  # noqa: E402
    PLACEMENT_BGW,
    PLACEMENT_OFF,
    PLACEMENT_SCHUR_AVG,
    HeadChannel,
    normalize_placement,
    refuse_if_unimplemented,
)


# ---------------------------------------------------------------------------
# Reference constructions
# ---------------------------------------------------------------------------

def _hermitian(rng, n):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (A + A.conj().T) / (2.0 * np.sqrt(n))


def _problem(rng, *, n_mu=48, k=1):
    """A Hermitian body V, a Hermitian chi, and a rank-k head channel.

    ``G`` is ``(n_mu, k)`` — the centroid-basis columns of the tied head
    slots.  All tied slots share one ``v`` value, exactly as
    ``vcoul.base.v_qG_table``'s tied-set mean makes them.
    """
    A = (rng.normal(size=(n_mu, n_mu)) + 1j * rng.normal(size=(n_mu, n_mu)))
    A /= np.sqrt(n_mu)
    V_body = A @ A.conj().T + 0.05 * np.eye(n_mu)
    G = (rng.normal(size=(n_mu, k)) + 1j * rng.normal(size=(n_mu, k)))
    G /= np.sqrt(n_mu)
    B = (rng.normal(size=(n_mu, n_mu)) + 1j * rng.normal(size=(n_mu, n_mu)))
    B /= np.sqrt(n_mu)
    chi = -0.02 * (B @ B.conj().T)          # Hermitian, negative-definite
    v_c = float(rng.uniform(0.5, 3.0))      # bare v at the head slot
    eta = float(rng.uniform(0.005, 0.08))   # the measured mini-BZ range
    return V_body, G, chi, v_c, eta


def _solve(V, chi):
    return np.linalg.solve(np.eye(V.shape[0]) - V @ chi, V)


def _schur_reference(V_body, G, chi, v_c, v_avg):
    """The head-channel Schur form, built term for term from its own pieces.

    This is ``head_wing_schur``'s chain (``extract_V_body`` ->
    ``solve_W_body0`` -> ``schur_reductions`` -> ``W_head_scalar_per_q`` ->
    ``assemble_W``) generalised from a rank-1 head channel to the rank-k
    tied set, with the mini-BZ average placed on the head SCALAR only:
    the numerator carries ``<v>`` while the denominator stays at the bare
    ``v_c``, which is BerkeleyGW's ``eps_c^-1 <v>``.
    """
    n = V_body.shape[0]
    W_body0 = _solve(V_body, chi)
    chi_wing = chi @ np.conj(G)                 # (n, k)
    chi_wingp = G.T @ chi                       # (k, n)
    chi_head = G.T @ chi @ np.conj(G)           # (k, k)
    A_wing = W_body0 @ chi_wing                 # (n, k)
    A_wingp = chi_wingp @ W_body0               # (k, n)
    chi_eff = chi_head + chi_wingp @ A_wing     # (k, k)
    S = v_avg * np.linalg.inv(np.eye(len(chi_eff)) - v_c * chi_eff)
    left = np.conj(G) + A_wing                  # (n, k)
    right = G.T + A_wingp                       # (k, n)
    assert left.shape == (n, S.shape[0])
    return W_body0 + left @ S @ right


# ---------------------------------------------------------------------------
# 1. The identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 4])
def test_two_solve_form_equals_head_channel_schur(k):
    """``W_body0 + r (W_bare - W_body0)`` IS the Schur head-channel rescale.

    Both sides are built from the same ``(V_body, G, chi, v_c, <v>)``; the
    left side never forms a wing and the right side never forms a second
    Dyson operator, so agreement is a statement about the algebra rather
    than about a shared implementation.
    """
    rng = np.random.default_rng(20260809 + k)
    for _ in range(3):
        V_body, G, chi, v_c, eta = _problem(rng, k=k)
        v_avg = v_c * (1.0 + eta)
        P = np.conj(G) @ G.T                    # sum_j conj(g_j) (x) g_j
        V_bare = V_body + v_c * P

        W_body0 = _solve(V_body, chi)
        W_bare = _solve(V_bare, chi)
        r = v_avg / v_c
        W_two_solve = W_body0 + r * (W_bare - W_body0)

        W_schur = _schur_reference(V_body, G, chi, v_c, v_avg)

        rel = np.max(np.abs(W_two_solve - W_schur)) / np.max(np.abs(W_schur))
        assert rel < 1e-9, (
            f"k={k}: two-solve form differs from the head-channel Schur "
            f"form by {rel:.3e} relative")


@pytest.mark.parametrize("k", [1, 2, 4])
def test_r_equal_one_recovers_the_plain_solve(k):
    """``<v> == v_c`` must return the untouched single-V Dyson solution.

    This is the structural statement behind "Gamma is untouched" and
    behind the ``mc_average_vcoul_body = false`` arm: wherever the head
    channel carries no enhancement, the rescale is the identity, not
    something that happens to be small.
    """
    rng = np.random.default_rng(7 * k)
    V_body, G, chi, v_c, _ = _problem(rng, k=k)
    V_bare = V_body + v_c * (np.conj(G) @ G.T)
    W_body0 = _solve(V_body, chi)
    W_bare = _solve(V_bare, chi)
    W = W_body0 + 1.0 * (W_bare - W_body0)
    rel = np.max(np.abs(W - W_bare)) / np.max(np.abs(W_bare))
    assert rel < 1e-13, f"r=1 moved W by {rel:.3e} relative"


def test_zero_head_channel_is_a_structural_no_op():
    """A masked head channel (Gamma) leaves both arms equal to V itself."""
    rng = np.random.default_rng(11)
    V_body, G, chi, v_c, eta = _problem(rng, k=4)
    G0 = np.zeros_like(G)                        # the Gamma mask: all-zero
    P = np.conj(G0) @ G0.T
    assert np.array_equal(P, np.zeros_like(P))
    V_bare = V_body + v_c * P
    assert np.array_equal(V_bare, V_body)
    W_body0 = _solve(V_body, chi)
    W_bare = _solve(V_bare, chi)
    assert np.array_equal(W_body0, W_bare)
    del eta


# ---------------------------------------------------------------------------
# 2. Hermiticity is structural
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 4])
def test_rescaled_W_stays_hermitian(k):
    """W is Hermitian at every r, because r is real and both arms are.

    ``(I - V chi)^-1 V`` with a single V is ``V^1/2 (I - V^1/2 chi V^1/2)^-1
    V^1/2``, Hermitian by congruence; a real combination of two such is
    Hermitian.  The tolerance is the arithmetic floor of the two dense
    solves, not a physics tolerance.
    """
    rng = np.random.default_rng(101 + k)
    V_body, G, chi, v_c, eta = _problem(rng, k=k)
    V_bare = V_body + v_c * (np.conj(G) @ G.T)
    W_body0 = _solve(V_body, chi)
    W_bare = _solve(V_bare, chi)
    for r in (1.0, 1.0 + eta, 1.5, 6.0):
        W = W_body0 + r * (W_bare - W_body0)
        asym = np.max(np.abs(W - W.conj().T)) / np.max(np.abs(W))
        assert asym < 1e-12, f"k={k}, r={r}: W asymmetry {asym:.3e}"


# ---------------------------------------------------------------------------
# 3 + 4. The knob
# ---------------------------------------------------------------------------

def test_placement_default_and_aliases():
    assert normalize_placement(None) == PLACEMENT_OFF
    for spelling in ("off", "OFF", " off ", "false", "none", "dyson"):
        assert normalize_placement(spelling) == PLACEMENT_OFF
    assert normalize_placement("bgw") == PLACEMENT_BGW
    assert normalize_placement("BGW") == PLACEMENT_BGW
    assert normalize_placement("schur_avg") == PLACEMENT_SCHUR_AVG


def test_placement_refuses_unknown_spelling():
    with pytest.raises(ValueError) as exc:
        normalize_placement("bwg")           # the obvious typo
    msg = str(exc.value)
    assert "bwg" in msg and "bgw" in msg and "off" in msg


def test_schur_avg_is_refused_with_a_reason():
    refuse_if_unimplemented(PLACEMENT_OFF)
    refuse_if_unimplemented(PLACEMENT_BGW)
    with pytest.raises(NotImplementedError) as exc:
        refuse_if_unimplemented(PLACEMENT_SCHUR_AVG)
    msg = str(exc.value)
    assert "schur_avg" in msg
    assert "bgw" in msg                       # names the usable alternative


# ---------------------------------------------------------------------------
# HeadChannel bookkeeping
# ---------------------------------------------------------------------------

def _head_channel(v_bare, v_avg, v_in_V, k=1):
    n_q = len(v_bare)
    return HeadChannel(
        g_head=np.zeros((n_q, k, 4), dtype=np.complex128),
        v_bare=np.asarray(v_bare, dtype=np.float64),
        v_avg=np.asarray(v_avg, dtype=np.float64),
        v_in_V=np.asarray(v_in_V, dtype=np.float64),
        mult=np.ones(n_q, dtype=np.int32),
        len2=np.ones(n_q),
        mode=PLACEMENT_BGW,
    )


def test_head_ratio_is_one_where_there_is_no_head_slot():
    """Gamma (v_bare = 0) must give r = 1, not a divide-by-zero."""
    hc = _head_channel([0.0, 10.0, 20.0], [0.0, 10.5, 20.4],
                       [0.0, 10.5, 20.4])
    r = hc.head_ratio()
    assert r[0] == 1.0
    assert np.isclose(r[1], 1.05)
    assert np.isclose(r[2], 1.02)
    assert np.all(np.isfinite(r))


def test_subtract_scalars_track_the_mc_average_flag():
    """The two V arms are correct with the flag on AND off.

    Flag ON:  V carries <v>, so the body arm removes <v> and the bare arm
              removes the enhancement <v> - v_c.
    Flag OFF: V already carries v_c, so the bare arm removes NOTHING —
              that arm has to be V itself, bitwise, or the placement mode
              would perturb the arm it is supposed to leave alone.
    """
    on = _head_channel([0.0, 10.0], [0.0, 10.5], [0.0, 10.5])
    assert np.allclose(on.body_subtract(), [0.0, 10.5])
    assert np.allclose(on.bare_subtract(), [0.0, 0.5])

    off = _head_channel([0.0, 10.0], [0.0, 10.5], [0.0, 10.0])
    assert np.allclose(off.body_subtract(), [0.0, 10.0])
    assert np.allclose(off.bare_subtract(), [0.0, 0.0])
    # ...and the head RATIO is the same either way: the placement is a
    # property of the physics, not of which tile V happens to carry.
    assert np.allclose(on.head_ratio(), off.head_ratio())


def test_both_mc_average_arms_reach_the_same_W():
    """The boundary case: mode 'bgw' is insensitive to mc_average_vcoul_body.

    With the flag on, V's head slots carry ``<v>`` and the body arm removes
    ``<v>``; with it off they carry ``v_c`` and the body arm removes
    ``v_c``.  Either way ``V_body`` is the same matrix and ``V_bare`` is the
    same matrix, so W is the same — which is the spec: the placement mode
    decides where the average is applied, and the flag decides only what V
    hands to Sigma_X.
    """
    rng = np.random.default_rng(4242)
    V_body, G, chi, v_c, eta = _problem(rng, k=2)
    v_avg = v_c * (1.0 + eta)
    P = np.conj(G) @ G.T

    V_flag_on = V_body + v_avg * P
    V_flag_off = V_body + v_c * P

    on = _head_channel([v_c], [v_avg], [v_avg])
    off = _head_channel([v_c], [v_avg], [v_c])

    body_on = V_flag_on - on.body_subtract()[0] * P
    body_off = V_flag_off - off.body_subtract()[0] * P
    assert np.allclose(body_on, body_off)
    assert np.allclose(body_on, V_body)

    bare_on = V_flag_on - on.bare_subtract()[0] * P
    bare_off = V_flag_off - off.bare_subtract()[0] * P
    assert np.allclose(bare_on, bare_off)
    assert np.allclose(bare_off, V_flag_off)   # bitwise-identity arm

    W_on = (_solve(body_on, chi)
            + on.head_ratio()[0] * (_solve(bare_on, chi) - _solve(body_on, chi)))
    W_off = (_solve(body_off, chi)
             + off.head_ratio()[0] * (_solve(bare_off, chi) - _solve(body_off, chi)))
    rel = np.max(np.abs(W_on - W_off)) / np.max(np.abs(W_off))
    assert rel < 1e-12, f"the two mc_average arms disagree by {rel:.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
