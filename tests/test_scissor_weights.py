"""The scissor fit is a REDUCTION over k, so it needs k weights.

Four properties, each of which failed at least once in the IBZ
self-consistency work (``docs/dev/ibz_self_consistency_scaffold.md`` §8):

1. ``fit_scissor`` has NO unweighted spelling.  Omitting ``k_weights`` is
   a ``TypeError`` at the call site, not a silently different fit.
2. Uniform weights reproduce the pre-weighting arithmetic BIT FOR BIT.
   The unweighted formulas are transcribed here as ``_legacy_ols`` (from
   ``scissor.py`` before this change) and compared with ``==``, not
   ``allclose`` — the full-BZ default is the production path and must not
   move by one ulp.
3. A weighted fit on star representatives equals an unweighted fit on the
   unfolded full BZ.  This is the actual bug: 6 of the 10 MoS₂ 4×4 stars
   have multiplicity 2, so the full-BZ arm saw them twice.
4. ``k_star_weights`` returns the multiplicities in ``select``'s own row
   order, and ones on an identity map.

``src/gw/scissor.py`` is loaded FROM ITS PATH, not as ``gw.scissor``:
``gw/__init__.py`` imports ``common.meta``, which imports jax, and jax is
not importable on a login node.  The module itself needs only numpy, so
loading it directly keeps this suite runnable everywhere — the same
reason ``test_layering.py`` is pure AST.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_SCISSOR = (pathlib.Path(__file__).resolve().parents[1]
            / "src" / "gw" / "scissor.py")
_spec = importlib.util.spec_from_file_location("_lorrax_scissor", _SCISSOR)
scissor = importlib.util.module_from_spec(_spec)
# Register before exec: @dataclass resolves annotations through
# ``sys.modules[cls.__module__]`` on Python 3.7 and raises without it.
sys.modules[_spec.name] = scissor
_spec.loader.exec_module(scissor)

ScissorFit = scissor.ScissorFit
fit_scissor = scissor.fit_scissor
full_bz_k_weights = scissor.full_bz_k_weights
k_star_weights = scissor.k_star_weights


# ---------------------------------------------------------------------------
# The pre-weighting arithmetic, transcribed verbatim for property 2
# ---------------------------------------------------------------------------

def _legacy_ols(x, y):
    """``scissor._ols_line`` as it stood before k weights (git b0d58f7)."""
    n = int(x.size)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return 0.0, float(y[0]), 0.0
    xm = float(x.mean())
    ym = float(y.mean())
    dx = x - xm
    denom = float(np.dot(dx, dx))
    if denom < 1.0e-30:
        return 0.0, ym, float(np.sqrt(np.mean((y - ym) ** 2)))
    slope = float(np.dot(dx, y - ym) / denom)
    intercept = float(ym - slope * xm)
    resid = y - (slope * x + intercept)
    return slope, intercept, float(np.sqrt(np.mean(resid * resid)))


def _legacy_fit_scissor(E_dft_kn_ev, E_qp_kn_ev, valence_mask_kn,
                        fit_mask_kn):
    """``scissor.fit_scissor`` as it stood before k weights (git b0d58f7)."""
    E_dft = np.asarray(E_dft_kn_ev, dtype=np.float64)
    E_qp = np.real(np.asarray(E_qp_kn_ev, dtype=np.complex128))
    vm = np.asarray(valence_mask_kn, dtype=bool)
    fm = np.asarray(fit_mask_kn, dtype=bool)
    nk = E_dft.shape[0]
    rows = np.arange(nk)[:, None]
    order_dft = np.argsort(E_dft, axis=1)
    order_qp = np.argsort(E_qp, axis=1)
    E_dft_sorted = E_dft[rows, order_dft]
    E_qp_sorted = E_qp[rows, order_qp]
    vm_sorted = vm[rows, order_dft]
    fm_sorted = fm[rows, order_dft]
    mask_v = vm_sorted & fm_sorted
    mask_c = (~vm_sorted) & fm_sorted
    alpha_v, beta_v, _ = _legacy_ols(E_dft_sorted[mask_v], E_qp_sorted[mask_v])
    alpha_c, beta_c, _ = _legacy_ols(E_dft_sorted[mask_c], E_qp_sorted[mask_c])
    resid_v = (E_qp_sorted - E_dft_sorted)[mask_v] - (
        (alpha_v - 1.0) * E_dft_sorted[mask_v] + beta_v)
    resid_c = (E_qp_sorted - E_dft_sorted)[mask_c] - (
        (alpha_c - 1.0) * E_dft_sorted[mask_c] + beta_c)
    rmse_v = float(np.sqrt(np.mean(resid_v * resid_v))) if resid_v.size else 0.0
    rmse_c = float(np.sqrt(np.mean(resid_c * resid_c))) if resid_c.size else 0.0
    return (alpha_v, beta_v, alpha_c, beta_c,
            int(mask_v.sum()), int(mask_c.sum()), rmse_v, rmse_c)


# ---------------------------------------------------------------------------
# A KStarMap stand-in.  Duck-typed on ``irr_idx`` + ``select``, which is
# all ``k_star_weights`` uses, so this file needs no jax.
# ---------------------------------------------------------------------------

class _FakeKStar:
    """Mirrors ``symmetry_maps.star_select``'s row order exactly."""

    def __init__(self, irr_idx):
        self.irr_idx = np.asarray(irr_idx, dtype=np.int32)

    def select(self, A_full):
        _, first = np.unique(self.irr_idx, return_index=True)
        return np.asarray(A_full)[np.sort(first)]


def _random_deck(rng, nk, nb):
    """(E_DFT, E_QP, valence mask, fit mask) with a realistic shape."""
    e_dft = np.sort(rng.normal(0.0, 6.0, size=(nk, nb)), axis=1)
    e_qp = e_dft + 0.25 * e_dft + rng.normal(0.8, 0.1, size=(nk, nb))
    vm = np.zeros((nk, nb), dtype=bool)
    vm[:, : nb // 2] = True
    fm = np.zeros((nk, nb), dtype=bool)
    fm[:, nb // 4: 3 * nb // 4] = True
    return e_dft, e_qp, vm, fm


# ---------------------------------------------------------------------------
# 1.  No unweighted spelling
# ---------------------------------------------------------------------------

def test_fit_scissor_refuses_to_run_without_k_weights():
    rng = np.random.default_rng(0)
    e_dft, e_qp, vm, fm = _random_deck(rng, 6, 16)
    with pytest.raises(TypeError):
        fit_scissor(e_dft, e_qp, valence_mask_kn=vm, fit_mask_kn=fm)


def test_fit_scissor_refuses_weights_from_a_different_k_set():
    """The IBZ-vs-full-BZ mix-up, caught by shape rather than by a wrong eV."""
    rng = np.random.default_rng(1)
    e_dft, e_qp, vm, fm = _random_deck(rng, 10, 16)
    with pytest.raises(ValueError, match="k_weights has shape"):
        fit_scissor(e_dft, e_qp, valence_mask_kn=vm, fit_mask_kn=fm,
                    k_weights=full_bz_k_weights(16))


def test_fit_scissor_refuses_a_zero_weight():
    rng = np.random.default_rng(2)
    e_dft, e_qp, vm, fm = _random_deck(rng, 4, 16)
    w = full_bz_k_weights(4)
    w[2] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        fit_scissor(e_dft, e_qp, valence_mask_kn=vm, fit_mask_kn=fm,
                    k_weights=w)


# ---------------------------------------------------------------------------
# 2.  Uniform weights are byte-identical to the pre-weighting arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(24))
def test_uniform_weights_are_bit_identical_to_the_unweighted_fit(seed):
    rng = np.random.default_rng(1000 + seed)
    nk = int(rng.integers(1, 33))
    nb = int(rng.integers(8, 129))
    e_dft, e_qp, vm, fm = _random_deck(rng, nk, nb)
    got = fit_scissor(e_dft, e_qp, valence_mask_kn=vm, fit_mask_kn=fm,
                      k_weights=full_bz_k_weights(nk))
    want = _legacy_fit_scissor(e_dft, e_qp, vm, fm)
    assert (got.alpha_v, got.beta_v_ev, got.alpha_c, got.beta_c_ev,
            got.n_fit_v, got.n_fit_c, got.rmse_v_ev, got.rmse_c_ev) == want
    assert got.w_fit_v == float(got.n_fit_v)
    assert got.w_fit_c == float(got.n_fit_c)


def test_the_bit_identity_check_can_fail():
    """Red twin: the comparison above is exact, so a real change trips it."""
    rng = np.random.default_rng(7)
    e_dft, e_qp, vm, fm = _random_deck(rng, 8, 32)
    w = full_bz_k_weights(8)
    w[3] = 2.0
    got = fit_scissor(e_dft, e_qp, valence_mask_kn=vm, fit_mask_kn=fm,
                      k_weights=w)
    want = _legacy_fit_scissor(e_dft, e_qp, vm, fm)
    assert got.alpha_v != want[0]


# ---------------------------------------------------------------------------
# 3.  Weighted stars == unweighted full BZ.  The bug itself.
# ---------------------------------------------------------------------------

def test_weighted_ibz_fit_equals_unweighted_full_bz_fit():
    """MoS2 4x4 shape: 16 full-BZ k -> 10 stars, 6 of multiplicity 2.

    Star members carry identical energies by symmetry (the measured star
    spread of the real Sigma+V_H is 7.6e-12 relative, job 7889373), so the
    exact statement is: an unweighted fit over the unfolded rows equals a
    multiplicity-weighted fit over the representatives.
    """
    rng = np.random.default_rng(3)
    nb = 128
    irr_idx = np.array([0, 1, 2, 1, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5])
    ks = _FakeKStar(irr_idx)
    nk_irr = int(np.unique(irr_idx).size)
    assert nk_irr == 10 and irr_idx.size == 16

    e_dft_i, e_qp_i, vm_i, fm_i = _random_deck(rng, nk_irr, nb)
    # Unfold: every full-BZ row is a verbatim copy of its representative.
    _, first = np.unique(irr_idx, return_index=True)
    labels = irr_idx[np.sort(first)]
    take = np.array([int(np.where(labels == v)[0][0]) for v in irr_idx])
    e_dft_f, e_qp_f = e_dft_i[take], e_qp_i[take]
    vm_f, fm_f = vm_i[take], fm_i[take]

    full = fit_scissor(e_dft_f, e_qp_f, valence_mask_kn=vm_f,
                       fit_mask_kn=fm_f,
                       k_weights=full_bz_k_weights(16))
    ibz = fit_scissor(e_dft_i, e_qp_i, valence_mask_kn=vm_i,
                      fit_mask_kn=fm_i,
                      k_weights=k_star_weights(ks))

    # Point counts differ; total weight and the fit itself do not.
    assert ibz.n_fit_v < full.n_fit_v
    assert ibz.w_fit_v == full.w_fit_v
    assert ibz.w_fit_c == full.w_fit_c
    for a, b in ((ibz.alpha_v, full.alpha_v), (ibz.beta_v_ev, full.beta_v_ev),
                 (ibz.alpha_c, full.alpha_c), (ibz.beta_c_ev, full.beta_c_ev)):
        assert abs(a - b) <= 1.0e-12 * max(abs(b), 1.0)

    # And the unweighted IBZ fit -- what the code did before -- does not
    # reproduce the full-BZ fit.  Without this the test above would pass
    # on a deck where weighting happened not to matter.
    unweighted = fit_scissor(e_dft_i, e_qp_i, valence_mask_kn=vm_i,
                             fit_mask_kn=fm_i,
                             k_weights=full_bz_k_weights(nk_irr))
    assert abs(unweighted.beta_c_ev - full.beta_c_ev) > 1.0e-9


def test_predicted_correction_agrees_between_the_arms():
    """The quantity that actually reaches the carry is ``predict``."""
    rng = np.random.default_rng(4)
    nb = 64
    irr_idx = np.array([0, 1, 2, 1, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5])
    ks = _FakeKStar(irr_idx)
    e_dft_i, e_qp_i, vm_i, fm_i = _random_deck(rng, 10, nb)
    _, first = np.unique(irr_idx, return_index=True)
    labels = irr_idx[np.sort(first)]
    take = np.array([int(np.where(labels == v)[0][0]) for v in irr_idx])

    full = fit_scissor(e_dft_i[take], e_qp_i[take],
                       valence_mask_kn=vm_i[take], fit_mask_kn=fm_i[take],
                       k_weights=full_bz_k_weights(16))
    ibz = fit_scissor(e_dft_i, e_qp_i, valence_mask_kn=vm_i,
                      fit_mask_kn=fm_i, k_weights=k_star_weights(ks))
    d_full = full.predict(e_dft_i, vm_i)
    d_ibz = ibz.predict(e_dft_i, vm_i)
    assert float(np.abs(d_full - d_ibz).max()) <= 1.0e-11


# ---------------------------------------------------------------------------
# 4.  k_star_weights
# ---------------------------------------------------------------------------

def test_k_star_weights_are_multiplicities_in_select_row_order():
    irr_idx = np.array([0, 1, 2, 1, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5])
    w = k_star_weights(_FakeKStar(irr_idx))
    # Row order is first occurrence: labels 0,1,2,4,5,6,7,8,9,10.
    assert w.tolist() == [1., 2., 1., 1., 2., 2., 2., 2., 2., 1.]
    assert float(w.sum()) == float(irr_idx.size)


def test_k_star_weights_on_an_identity_map_are_ones():
    w = k_star_weights(_FakeKStar(np.arange(16)))
    assert w.tolist() == [1.0] * 16


def test_k_star_weights_survive_non_monotone_first_occurrence():
    """Row order comes from ``select``, not from sorted label value."""
    irr_idx = np.array([5, 5, 0, 2, 0, 2, 2])
    w = k_star_weights(_FakeKStar(irr_idx))
    # first occurrences at positions 0 (label 5), 2 (label 0), 3 (label 2)
    assert w.tolist() == [2.0, 2.0, 3.0]


def test_scissor_fit_fields_are_all_named():
    """A positional ScissorFit construction would silently swap w and rmse."""
    f = ScissorFit(alpha_v=1.0, beta_v_ev=0.0, alpha_c=1.0, beta_c_ev=0.0,
                   n_fit_v=1, n_fit_c=1, rmse_v_ev=0.0, rmse_c_ev=0.0,
                   w_fit_v=1.0, w_fit_c=1.0)
    assert f.w_fit_v == 1.0
