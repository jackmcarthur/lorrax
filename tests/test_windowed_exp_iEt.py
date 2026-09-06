"""Gates for ``windowed_exp_iEt`` — the non-materialised energy window.

THE PATTERN.  A band/energy window on the INTEGRAND used to be spelled as a
boolean array the shape of ``E``, built once, kept alive as a jit operand and
multiplied (or ``where``-ed) into the phases.  ``windowed_exp_iEt`` recomputes
the predicate from ``E`` at the point of use so it fuses with the ``exp`` into
one elementwise loop and never becomes a buffer.

WHAT MAKES THE REPLACEMENT SAFE, and what these gates assert:

* The mask it replaces was PURE 0/1.  For such a mask ``m·x`` and
  ``where(m, x, 0)`` agree BIT-FOR-BIT (``1·x == x`` and ``0·x == 0``
  exactly, for finite ``x``), so this is a bit-identity claim, not a
  tolerance claim, and it is gated with ``np.array_equal``.  A "mask" that
  is NOT pure 0/1 is a weight in disguise and is out of scope for this
  helper — gated below so the assumption cannot rot.
* ``where`` is strictly BETTER than the multiply on one class of lane: an
  EXCLUDED band whose ``exp`` overflows gives ``0·inf = nan`` under the
  multiply and exact zero under ``where``.
* The window is HALF-OPEN AND CLOSED AT THE TOP, ``(E_min, E_max]``, so
  abutting windows tile an energy axis without double-counting a band on a
  boundary AND assign such a band downward, into the pane whose supremum it
  is.  The direction is the decided one (peer decision 2026-08-09,
  ``bin_convention``): every certified rule is built at max(Γ) over its pane,
  so a pane must contain its own supremum.  This is also the Σ B-side
  convention in ``ppm_windows.window_mask_B_bounds``, and the two are gated
  against each other below rather than against a copied bound — the helper
  landed as ``[lo, hi)`` and was flipped here to end that mismatch.
* ``e_ref`` moves the phase origin ONLY.  It must never move the window
  support — the bounds are in the same reference as ``E``.

``build_G_tau`` is gated as a bit-exact NO-OP for every pre-existing caller
(chi0's unmasked call, Σ's band-identity ``mask_A`` call, and the ``(1,nk,nb)``
1×1-mesh reshape path that once crashed GN-PPM on one GPU).
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from gw.greens_function_kernel import build_G, build_G_tau, windowed_exp_iEt

NK, NB = 32, 64
E_MIN, E_MAX = -0.3, 1.1

# Abutting panes for the tiling gate; the interior edges are also the
# boundary-pole positions the red twins below plant a band on.
EDGES = [-2.0, -0.5, 0.4, 1.3, 3.0]

# real t -> imaginary-time evolution (chi0); pure-i t -> real-time (Sigma_c).
TIMES = [("chi0_real_t", 0.37), ("sigma_imag_t", 1j * 0.37), ("mixed_t", 0.11 + 0.42j)]


@pytest.fixture(scope="module")
def enk():
    rng = np.random.default_rng(20260809)
    return jnp.asarray(rng.uniform(-1.5, 2.5, size=(NK, NB)))


def _materialised(E, t, e_ref, lo, hi):
    """The form being replaced: a mask array multiplied into the phases."""
    mask = ((E > lo) & (E <= hi)).astype(jnp.complex128)
    return mask * jnp.exp(-t * (E - e_ref))


@pytest.mark.parametrize("tag,t", TIMES, ids=[x[0] for x in TIMES])
@pytest.mark.parametrize("e_ref", [0.0, 0.65])
def test_bit_identical_to_the_materialised_mask(enk, tag, t, e_ref):
    """where(pred, exp, 0) == mask*exp, bit for bit, for a 0/1 mask."""
    new = windowed_exp_iEt(enk, t, E_MIN, E_MAX, e_ref=e_ref)
    old = _materialised(enk, t, e_ref, E_MIN, E_MAX)
    assert np.array_equal(np.asarray(new), np.asarray(old))


def test_the_mask_it_replaces_is_pure_zero_one(enk):
    """If this ever fails, the 'mask' is a WEIGHT and must not be fused here."""
    m = np.asarray(((enk > E_MIN) & (enk <= E_MAX)).astype(jnp.float64))
    assert set(np.unique(m).tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("lo,hi", [(E_MIN, None), (None, E_MAX)])
def test_one_sided_windows(enk, lo, hi):
    got = windowed_exp_iEt(enk, 1j * 0.3, lo, hi)
    pred = (enk > lo) if hi is None else (enk <= hi)
    ref = jnp.where(pred, jnp.exp(-1j * 0.3 * enk),
                    jnp.asarray(0.0 + 0.0j, jnp.complex128))
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_no_bounds_is_the_bare_phase(enk):
    got = windowed_exp_iEt(enk, 1j * 0.3, None, None, e_ref=0.4)
    assert np.array_equal(np.asarray(got),
                          np.asarray(jnp.exp(-1j * 0.3 * (enk - 0.4))))


def test_half_open_windows_tile_the_axis_exactly(enk):
    """Abutting windows sum to the unwindowed phase — no gap, no double-count.

    The array under test has bands sitting EXACTLY on the interior edges and
    on the closed top edge, which is the only configuration in which a tiling
    claim can fail.  ``EDGES[0]`` is deliberately absent: the cover is
    ``(EDGES[0], EDGES[-1]]``, so a band exactly at the bottom edge is outside
    it by construction, not by accident (see the red twin below).
    """
    E = jnp.concatenate([jnp.reshape(enk, (-1,)),
                         jnp.asarray(EDGES[1:], dtype=enk.dtype)])
    acc = jnp.zeros_like(E, dtype=jnp.complex128)
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        acc = acc + windowed_exp_iEt(E, 1j * 0.3, a, b)
    assert np.array_equal(np.asarray(acc), np.asarray(jnp.exp(-1j * 0.3 * E)))


# ------------------------------------------------------------------ red twins

def test_red_twin_boundary_band_is_included_at_E_max_excluded_at_E_min():
    """``(lo, hi]``: closed at the top, open at the bottom.

    This is the assertion that changed direction when the helper flipped from
    ``[lo, hi)``.  The pre-flip predicate turns it red on both lines, which is
    the point: it pins the DIRECTION of the boundary assignment, not merely
    that some boundary rule exists.
    """
    at_max = windowed_exp_iEt(jnp.asarray([[E_MAX]]), 1j * 0.3, E_MIN, E_MAX)
    at_min = windowed_exp_iEt(jnp.asarray([[E_MIN]]), 1j * 0.3, E_MIN, E_MAX)
    assert np.asarray(at_max)[0, 0] != 0.0 + 0.0j
    assert np.asarray(at_min)[0, 0] == 0.0 + 0.0j


def test_red_twin_boundary_pole_lands_in_the_pane_below_it_exactly_once():
    """THE boundary-pole twin: a pole sitting exactly on a shared pane edge.

    For each interior edge of an abutting cover, the pole at that edge must be
    carried by the pane BELOW it — the pane whose supremum it is, and so the
    pane whose certified rule was built at max(Γ) = that pole — and by exactly
    one pane overall.  Under the pre-flip ``[lo, hi)`` the count is still one,
    but it is the pane ABOVE that carries it, and the pole is then evaluated
    under a rule certified on an interval whose max(Γ) is a different number.
    """
    panes = list(zip(EDGES[:-1], EDGES[1:]))
    for i, edge in enumerate(EDGES[1:-1]):          # interior edges only
        E = jnp.asarray([[edge]])
        carried = [j for j, (a, b) in enumerate(panes)
                   if np.asarray(windowed_exp_iEt(E, 1j * 0.3, a, b))[0, 0] != 0]
        assert carried == [i], (
            f"pole at {edge} carried by panes {carried}, expected the pane "
            f"below it ({panes[i]}) and only that one")


def test_red_twin_the_two_sides_of_sigma_agree_on_a_boundary_pole():
    """Certified where consumed: agreement with the Σ B-side selector itself.

    ``ppm_windows.window_mask_B_bounds`` turns the ``le_t``/``gt_t`` modes into
    the scalar bounds the Ω predicate is rebuilt from, and it has always been
    ``(lo, hi]``.  Feeding ITS bounds — not a copy of them — through this
    helper is what pins the two sides of Σ to one convention: a pole exactly at
    the threshold belongs to ``le_t`` and not to ``gt_t``, on both sides.
    Under the pre-flip helper this is red, which is the mismatch the flip
    closed.
    """
    from types import SimpleNamespace

    from gw.ppm_windows import window_mask_B_bounds

    T = 0.75
    lo_le, hi_le = window_mask_B_bounds(
        SimpleNamespace(
            mask_B_mode="le_t", mask_B_threshold=T,
            B_lo=None, B_hi=None))
    lo_gt, hi_gt = window_mask_B_bounds(
        SimpleNamespace(
            mask_B_mode="gt_t", mask_B_threshold=T,
            B_lo=None, B_hi=None))

    at_T = jnp.asarray([[T]])
    in_le = np.asarray(windowed_exp_iEt(at_T, 1j * 0.3, lo_le, hi_le))[0, 0]
    in_gt = np.asarray(windowed_exp_iEt(at_T, 1j * 0.3, lo_gt, hi_gt))[0, 0]
    assert in_le != 0.0 + 0.0j
    assert in_gt == 0.0 + 0.0j

    # ...and the B side's own predicate says the same thing about that pole,
    # so the gate cannot pass by both sides being wrong together.
    Om = np.asarray([T])
    assert bool(((Om > lo_le) & (Om <= hi_le))[0])
    assert not bool(((Om > lo_gt) & (Om <= hi_gt))[0])


def test_red_twin_bounds_are_live_and_the_window_is_non_trivial(enk):
    """A shifted window must NOT match, and the window must select a
    PROPER subset — otherwise the identity gates above could pass on an
    all-in or all-out window that exercises nothing."""
    right = windowed_exp_iEt(enk, 1j * 0.3, E_MIN, E_MAX)
    wrong = windowed_exp_iEt(enk, 1j * 0.3, E_MIN + 0.25, E_MAX)
    assert not np.array_equal(np.asarray(right), np.asarray(wrong))
    n_in = int(np.count_nonzero(np.asarray(right)))
    assert 0 < n_in < int(np.asarray(enk).size)


def test_red_twin_e_ref_moves_the_phase_never_the_support(enk):
    s0 = np.asarray(windowed_exp_iEt(enk, 1j * 0.3, E_MIN, E_MAX, e_ref=0.0)) != 0
    s1 = np.asarray(windowed_exp_iEt(enk, 1j * 0.3, E_MIN, E_MAX, e_ref=0.9)) != 0
    assert np.array_equal(s0, s1)


def test_where_returns_exact_zero_where_the_multiply_would_give_nan():
    """The one lane class where the two forms differ, and ``where`` wins:
    an EXCLUDED band whose exp overflows.  0*inf = nan; where = 0."""
    E = jnp.asarray([[1.0e4]])
    new = windowed_exp_iEt(E, -1000.0, -1.0, 1.0)
    mul = ((E > -1.0) & (E <= 1.0)).astype(jnp.complex128) * jnp.exp(1000.0 * E)
    assert np.asarray(new)[0, 0] == 0.0 + 0.0j
    assert np.isnan(np.asarray(mul)[0, 0])


# --------------------------------------------- build_G_tau is an exact no-op

@pytest.fixture
def gemm():
    """Isolate phase and selector algebra from the native distributed GEMM backend."""
    import jax
    from jax.sharding import Mesh
    plan = lambda left, right: left @ right
    plan.mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    return plan


def _build_G_tau_pre_refactor(psi_xn, psi_yr, enk, t, *, gemm, e_ref=0.0, mask=None):
    """Materialize the phase and identity mask before the same canonical contraction."""
    phases = jnp.exp(-t * (enk - e_ref))
    if mask is not None:
        mask = jnp.reshape(mask, enk.shape)
        phases = jnp.where(mask, phases,
                           jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    return build_G(psi_xn, psi_yr, phases=phases, gemm=gemm)


@pytest.fixture(scope="module")
def psis():
    rng = np.random.default_rng(11)
    s, mu = 1, 12
    xn = jnp.asarray(rng.normal(size=(NK, s, mu, NB))
                     + 1j * rng.normal(size=(NK, s, mu, NB)))
    yr = jnp.asarray(rng.normal(size=(NK, NB, s, mu))
                     + 1j * rng.normal(size=(NK, NB, s, mu)))
    return xn, yr


@pytest.mark.parametrize("tag,t", TIMES[:2], ids=[x[0] for x in TIMES[:2]])
def test_build_G_tau_unmasked_is_a_bit_exact_no_op(psis, enk, tag, t, gemm):
    """chi0's call signature."""
    xn, yr = psis
    new = build_G_tau(xn, yr, enk, t, gemm=gemm, e_ref=0.5)
    old = _build_G_tau_pre_refactor(xn, yr, enk, t, gemm=gemm, e_ref=0.5)
    assert np.array_equal(np.asarray(new), np.asarray(old))


@pytest.mark.parametrize("tag,t", TIMES[:2], ids=[x[0] for x in TIMES[:2]])
def test_build_G_tau_band_identity_mask_is_a_bit_exact_no_op(psis, enk, tag, t, gemm):
    """Σ's ``mask_A`` call signature, and the ``(1,nk,nb)`` 1×1-mesh reshape
    path (the nspin axis a 2×2 mesh squeezes and a 1×1 mesh does not — an
    unreshaped mask broadcast phases to 3-D and crashed GN-PPM on one GPU)."""
    xn, yr = psis
    rng = np.random.default_rng(5)
    mA = jnp.asarray(rng.random((NK, NB)) > 0.4)
    old = _build_G_tau_pre_refactor(xn, yr, enk, t, gemm=gemm, e_ref=0.5, mask=mA)
    assert np.array_equal(
        np.asarray(build_G_tau(xn, yr, enk, t, gemm=gemm, e_ref=0.5, mask=mA)),
        np.asarray(old))
    assert np.array_equal(
        np.asarray(build_G_tau(xn, yr, enk, t, gemm=gemm, e_ref=0.5,
                               mask=jnp.reshape(mA, (1, NK, NB)))),
        np.asarray(old))


def test_build_G_tau_zero_weight_gates_an_overflowing_unselected_band(gemm):
    """Sparse delivered selectors cannot turn an excluded overflow into NaN."""
    energies = jnp.asarray([[0.1, 1000.0]])
    weight = jnp.asarray([[1.0, 0.0]])
    psi_xn = jnp.ones((1, 1, 1, 2), dtype=jnp.complex128)
    psi_yr = jnp.ones((1, 2, 1, 1), dtype=jnp.complex128)

    got = np.asarray(build_G_tau(
        psi_xn, psi_yr, energies, -1.0, band_weight=weight, gemm=gemm))

    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, np.exp(0.1), rtol=1.0e-15, atol=0.0)


@pytest.mark.parametrize("tag,t", TIMES[:2], ids=[x[0] for x in TIMES[:2]])
def test_build_G_tau_bounds_route_equals_the_materialised_mask_route(psis, enk, tag, t, gemm):
    """The conversion itself: E_min/E_max reproduces the mask array it replaces."""
    xn, yr = psis
    new = build_G_tau(xn, yr, enk, t, gemm=gemm, e_ref=0.5, E_min=E_MIN, E_max=E_MAX)
    old = _build_G_tau_pre_refactor(xn, yr, enk, t, gemm=gemm, e_ref=0.5,
                                    mask=((enk > E_MIN) & (enk <= E_MAX)))
    assert np.array_equal(np.asarray(new), np.asarray(old))


def test_build_G_rejects_unknown_layout():
    with pytest.raises(ValueError, match="layout"):
        build_G(jnp.zeros((1, 1, 1, 1)), jnp.zeros((1, 1, 1, 1)),
               layout="bogus")


def test_build_G_face_requires_a_gemm_plan():
    with pytest.raises(ValueError, match="gemm"):
        build_G(jnp.zeros((1, 1, 1, 1)), jnp.zeros((1, 1, 1, 1)),
               layout="face")


def test_build_G_face_refuses_dense_Gij_by_name():
    """Report obstacle #4's named escape hatch: a dense (non-diagonal)
    band operator is not ported for face layout — refuse rather than
    silently build a legacy-shaped contraction this layout cannot run.
    A sentinel non-None ``gemm`` is enough to reach the refusal; it is
    never called (Gij is checked first)."""
    with pytest.raises(NotImplementedError, match="Gij"):
        build_G(jnp.zeros((1, 1, 1, 1)), jnp.zeros((1, 1, 1, 1)),
               Gij=jnp.zeros((1, 1, 1)), layout="face",
               gemm=lambda *a, **k: None)


def test_build_G_tau_forwards_layout_and_gemm_to_build_G():
    """build_G_tau's own layout dispatch: the SAME refusals as build_G,
    reached through the phase-computation wrapper — pinning that the
    forward is wired, not merely that build_G itself refuses."""
    enk = jnp.zeros((2, 3))
    with pytest.raises(ValueError, match="layout"):
        build_G_tau(jnp.zeros((1,)), jnp.zeros((1,)), enk, 1.0,
                   layout="bogus")
    with pytest.raises(ValueError, match="gemm"):
        build_G_tau(jnp.zeros((2, 1, 1, 3)), jnp.zeros((2, 3, 1, 1)),
                   enk, 1.0, layout="face")


def test_build_G_and_tau_apply_parent_unfold_after_the_only_contraction(gemm):
    class ParentRows:
        @staticmethod
        def unfold_operator(operator):
            return operator[jnp.asarray([1, 0, 1])]

    xn = jnp.asarray(np.arange(2 * 1 * 2 * 3).reshape(2, 1, 2, 3)
                     + 1j)
    yr = jnp.asarray(np.arange(2 * 3 * 1 * 2).reshape(2, 3, 1, 2)
                     - 2j)
    parent = build_G(xn, yr, gemm=gemm)
    got = build_G(xn, yr, gemm=gemm, k_unfold_plan=ParentRows())
    np.testing.assert_array_equal(np.asarray(got),
                                  np.asarray(parent)[[1, 0, 1]])

    energy = jnp.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    parent_tau = build_G_tau(xn, yr, energy, 0.7, gemm=gemm)
    got_tau = build_G_tau(
        xn, yr, energy, 0.7, gemm=gemm, k_unfold_plan=ParentRows())
    np.testing.assert_array_equal(np.asarray(got_tau),
                                  np.asarray(parent_tau)[[1, 0, 1]])
