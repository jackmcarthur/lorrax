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
* The window is HALF-OPEN ``[E_min, E_max)`` so abutting windows tile an
  energy axis without double-counting a band on a boundary.
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

# real t -> imaginary-time evolution (chi0); pure-i t -> real-time (Sigma_c).
TIMES = [("chi0_real_t", 0.37), ("sigma_imag_t", 1j * 0.37), ("mixed_t", 0.11 + 0.42j)]


@pytest.fixture(scope="module")
def enk():
    rng = np.random.default_rng(20260809)
    return jnp.asarray(rng.uniform(-1.5, 2.5, size=(NK, NB)))


def _materialised(E, t, e_ref, lo, hi):
    """The form being replaced: a mask array multiplied into the phases."""
    mask = ((E >= lo) & (E < hi)).astype(jnp.complex128)
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
    m = np.asarray(((enk >= E_MIN) & (enk < E_MAX)).astype(jnp.float64))
    assert set(np.unique(m).tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("lo,hi", [(E_MIN, None), (None, E_MAX)])
def test_one_sided_windows(enk, lo, hi):
    got = windowed_exp_iEt(enk, 1j * 0.3, lo, hi)
    pred = (enk >= lo) if hi is None else (enk < hi)
    ref = jnp.where(pred, jnp.exp(-1j * 0.3 * enk),
                    jnp.asarray(0.0 + 0.0j, jnp.complex128))
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_no_bounds_is_the_bare_phase(enk):
    got = windowed_exp_iEt(enk, 1j * 0.3, None, None, e_ref=0.4)
    assert np.array_equal(np.asarray(got),
                          np.asarray(jnp.exp(-1j * 0.3 * (enk - 0.4))))


def test_half_open_windows_tile_the_axis_exactly(enk):
    """Abutting windows sum to the unwindowed phase — no gap, no double-count."""
    edges = [-2.0, -0.5, 0.4, 1.3, 3.0]
    acc = jnp.zeros_like(enk, dtype=jnp.complex128)
    for a, b in zip(edges[:-1], edges[1:]):
        acc = acc + windowed_exp_iEt(enk, 1j * 0.3, a, b)
    assert np.array_equal(np.asarray(acc), np.asarray(jnp.exp(-1j * 0.3 * enk)))


# ------------------------------------------------------------------ red twins

def test_red_twin_boundary_band_is_excluded_at_E_max_included_at_E_min():
    """Half-open. A ``<=`` upper predicate turns this red."""
    at_max = windowed_exp_iEt(jnp.asarray([[E_MAX]]), 1j * 0.3, E_MIN, E_MAX)
    at_min = windowed_exp_iEt(jnp.asarray([[E_MIN]]), 1j * 0.3, E_MIN, E_MAX)
    assert np.asarray(at_max)[0, 0] == 0.0 + 0.0j
    assert np.asarray(at_min)[0, 0] != 0.0 + 0.0j


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
    mul = ((E >= -1.0) & (E < 1.0)).astype(jnp.complex128) * jnp.exp(1000.0 * E)
    assert np.asarray(new)[0, 0] == 0.0 + 0.0j
    assert np.isnan(np.asarray(mul)[0, 0])


# --------------------------------------------- build_G_tau is an exact no-op

def _build_G_tau_pre_refactor(psi_xn, psi_yr, enk, t, *, e_ref=0.0, mask=None):
    """Verbatim body as of d25d8d6a, kept as the parity reference."""
    phases = jnp.exp(-t * (enk - e_ref))
    if mask is not None:
        mask = jnp.reshape(mask, enk.shape)
        phases = jnp.where(mask, phases,
                           jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    return build_G(psi_xn, psi_yr, phases=phases)


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
def test_build_G_tau_unmasked_is_a_bit_exact_no_op(psis, enk, tag, t):
    """chi0's call signature."""
    xn, yr = psis
    new = build_G_tau(xn, yr, enk, t, e_ref=0.5)
    old = _build_G_tau_pre_refactor(xn, yr, enk, t, e_ref=0.5)
    assert np.array_equal(np.asarray(new), np.asarray(old))


@pytest.mark.parametrize("tag,t", TIMES[:2], ids=[x[0] for x in TIMES[:2]])
def test_build_G_tau_band_identity_mask_is_a_bit_exact_no_op(psis, enk, tag, t):
    """Σ's ``mask_A`` call signature, and the ``(1,nk,nb)`` reshape path (an
    unreshaped mask broadcast phases to 3-D and crashed GN-PPM on one GPU).

    That leading axis is the PROCESS axis ``process_allgather(tiled=False)``
    prepends, not an nspin axis — see the comment at the reshape and
    ``ppm_windows._already_on_host``.  This cell pins the guard, which is why
    it still passes a rank-3 mask on purpose.
    """
    xn, yr = psis
    rng = np.random.default_rng(5)
    mA = jnp.asarray(rng.random((NK, NB)) > 0.4)
    old = _build_G_tau_pre_refactor(xn, yr, enk, t, e_ref=0.5, mask=mA)
    assert np.array_equal(
        np.asarray(build_G_tau(xn, yr, enk, t, e_ref=0.5, mask=mA)),
        np.asarray(old))
    assert np.array_equal(
        np.asarray(build_G_tau(xn, yr, enk, t, e_ref=0.5,
                               mask=jnp.reshape(mA, (1, NK, NB)))),
        np.asarray(old))


@pytest.mark.parametrize("tag,t", TIMES[:2], ids=[x[0] for x in TIMES[:2]])
def test_build_G_tau_bounds_route_equals_the_materialised_mask_route(psis, enk, tag, t):
    """The conversion itself: E_min/E_max reproduces the mask array it replaces."""
    xn, yr = psis
    new = build_G_tau(xn, yr, enk, t, e_ref=0.5, E_min=E_MIN, E_max=E_MAX)
    old = _build_G_tau_pre_refactor(xn, yr, enk, t, e_ref=0.5,
                                    mask=((enk >= E_MIN) & (enk < E_MAX)))
    assert np.array_equal(np.asarray(new), np.asarray(old))
