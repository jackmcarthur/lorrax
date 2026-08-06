"""Cell-by-cell gate for the ONE bare-Coulomb formula (``gw.coulomb.kernel``).

``v(q+G)`` used to be spelled out ~10 times across the tree.  ``kernel.v_qG``
is the single formula those call sites collapse onto.  This module holds a
VERBATIM frozen copy of every incumbent it replaces and asserts the merged
kernel against it over the full ``(sys_dim x channel x units x zero_tol)``
product, on a real production ``(q+G)`` table (the Si 4x4x4 regression WFN
G-sphere; falls back to a synthetic table if the fixture is unavailable).

Parity classes asserted here (measured on Frontera 2026-08-05, jobs
7890613 / 7890630 — see docs/architecture/decisions.md):

* ``compute_v_q_per_G``      -> BIT-EXACT   (byte equality, both sys_dim)
* ``_minibz_kernel_bare``    -> BIT-EXACT   (all four kinds)
* ``Slab2D._vq_sobol``        -> BIT-EXACT
* ``Bulk3D._vq_isotropic``   -> <= 3 ULP  (``einsum`` vs ``sum`` on the
  points-first ``(N, 3)`` layout; measured job 7890630, flat in N)
* ``v_slab_on_set`` / ``make_eval_vq._body`` -> <= 2 ULP (the incumbents
  spell the volume factor ``x / V`` where the merged kernel spells
  ``x * (1/V)``; measured max 2 ULP / 3.9e-16 relative)

Both allowances are MEASUREMENTS, not guesses.  ``x/V`` vs ``x*(1/V)`` is
exactly 1 ULP on this table and the LR/SR Gaussian factor carries it to 2.
``np.linalg.norm(K[:, :2])`` vs ``sqrt(x**2 + y**2)`` is bit-exact.
``sum(K*K, axis)`` vs ``einsum('ij,ij->i')`` is bit-exact on the
components-first ``(3, nG)`` layout and 3 ULP on the points-first
``(N, 3)`` one — the layout, not the spelling, is what decides.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from gw.coulomb.kernel import TOL_MC_NAN, TOL_QG_ZERO, v_qG

REPO = Path(__file__).resolve().parents[1]
SI_WFN = REPO / "tests" / "regression" / "si_cohsex_debug" / "WFN.h5"

# Measured ceilings, in ULPs of the incumbent (not relative epsilons).
# Frontera jobs 7890613 (volume factor) and 7890630 (einsum reduction).
# Raising either of these silently would hide a real regression -- if a
# number here needs to move, re-measure and say which job measured it.
ULP_VOLUME_RESPELL = 2.0     # x / V   vs   x * (1/V)
ULP_EINSUM_RESPELL = 3.0     # einsum('ij,ij->i')  vs  sum(K*K, axis=1)


# ---------------------------------------------------------------------------
# the (q+G) table
# ---------------------------------------------------------------------------
def _table():
    """(bvec, celvol, q-list, G-set (3, nG)) — production if available."""
    if SI_WFN.exists():
        import h5py
        with h5py.File(SI_WFN, "r") as f:
            blat = float(np.asarray(f["mf_header/crystal/blat"]).ravel()[0])
            bvec = blat * np.asarray(f["mf_header/crystal/bvec"],
                                     dtype=np.float64)
            celvol = float(np.asarray(
                f["mf_header/crystal/celvol"]).ravel()[0])
            kgrid = np.asarray(f["mf_header/kpoints/kgrid"], dtype=int).ravel()
            G = np.asarray(f["mf_header/gspace/components"],
                           dtype=np.int64).T.astype(np.float64)
    else:                                    # pragma: no cover - fixture-less
        bvec = 2.0 * np.pi * np.array([[1.0, 0.0, 0.0],
                                       [-0.31, 0.95, 0.0],
                                       [0.0, 0.0, 0.27]])
        celvol = (2.0 * np.pi) ** 3 / abs(np.linalg.det(bvec))
        kgrid = np.array([4, 4, 4])
        rng = np.random.RandomState(0)
        G = rng.randint(-6, 7, (3, 2000)).astype(np.float64)
        G[:, 0] = 0.0
    nk = kgrid.astype(np.float64)
    qs = []
    for i in range(int(nk[0])):
        for j in range(int(nk[1])):
            for k in range(int(nk[2])):
                qw = np.array([i, j, k], dtype=np.float64)
                qs.append(np.where(qw > nk / 2, qw - nk, qw) / nk)
    return bvec, celvol, np.asarray(qs), G


BVEC, CELVOL, QLIST, GSET = _table()
ZC = float(np.pi / BVEC[2, 2])
ALPHA = 0.5


def _K(qf):
    """Cartesian (3, nG) q+G, the components-first convention."""
    return BVEC.T @ (np.asarray(qf)[:, None] + GSET)


# ---------------------------------------------------------------------------
# FROZEN incumbents — verbatim copies, do not "clean up"
# ---------------------------------------------------------------------------
def inc_compute_v_q_per_G(qf, sys_dim):
    """gw/compute_vcoul.py::compute_v_q_per_G, per-q body (no head/cutoff)."""
    fact = 1.0 / float(CELVOL)
    qG_cart = BVEC.T @ (qf[:, None] + GSET)
    denom = np.sum(qG_cart * qG_cart, axis=0)
    denom_zero = denom < 1e-12
    denom_safe = np.where(denom_zero, 1.0, denom)
    if sys_dim == 3:
        v_reg = 8.0 * np.pi / denom_safe
        return np.where(denom_zero, 0.0, v_reg * fact)
    zc = float(np.pi / float(BVEC[2, 2]))
    kxy = np.sqrt(qG_cart[0] ** 2 + qG_cart[1] ** 2)
    f2d = 1.0 - np.exp(-zc * kxy) * np.cos(qG_cart[2] * zc)
    v_reg = (8.0 * np.pi / denom_safe) * f2d
    return np.where(denom_zero, 0.0, v_reg * fact)


def inc_v_slab_on_set(qf, kind):
    """bse/vq_interp.py::v_slab_on_set."""
    K = BVEC.T @ (np.asarray(qf)[:, None] + GSET)
    K2 = np.sum(K * K, axis=0)
    zero = K2 < 1e-12
    K2s = np.where(zero, 1.0, K2)
    zc = np.pi / BVEC[2, 2]
    f2d = 1.0 - np.exp(-zc * np.sqrt(K[0] ** 2 + K[1] ** 2)) * np.cos(K[2] * zc)
    v = 8.0 * np.pi / K2s * f2d / CELVOL
    if kind == "slab_lr":
        v = v * np.exp(-K2 / (4.0 * ALPHA ** 2))
    elif kind == "slab_sr":
        v = v * (-np.expm1(-K2 / (4.0 * ALPHA ** 2)))
    return np.where(zero, 0.0, v)


def inc_minibz_kernel_bare(shift_cart, dq_cart, kind):
    """gw/coulomb/base.py::_minibz_kernel_bare."""
    K = np.asarray(shift_cart, dtype=np.float64)[None, :] + np.asarray(dq_cart)
    len2 = np.sum(K * K, axis=1)
    len2s = np.where(len2 < 1e-24, 1.0, len2)
    v = 8.0 * np.pi / len2s
    if kind in ("slab", "slab_lr"):
        kxy = np.linalg.norm(K[:, :2], axis=1)
        v = v * (1.0 - np.exp(-ZC * kxy) * np.cos(K[:, 2] * ZC))
    if kind in ("bulk_3d_lr", "slab_lr"):
        v = v * np.exp(-len2 / (4.0 * ALPHA ** 2))
    return np.where(len2 < 1e-24, 0.0, v)


def inc_vq_isotropic(qcart):
    """gw/coulomb/bulk_3d.py::Bulk3D._vq_isotropic (jnp -> np transcription)."""
    return 8.0 * np.pi / np.einsum("ij,ij->i", qcart, qcart)


def inc_vq_sobol(rq):
    """gw/coulomb/slab_2d.py::Slab2D.q0_average._vq_sobol."""
    base = 8.0 * np.pi / np.einsum("ij,ij->i", rq, rq)
    kxy = np.linalg.norm(rq[:, :2], axis=1)
    return base * (1.0 - np.exp(-ZC * kxy) * np.cos(rq[:, 2] * ZC))


def inc_eval_vq_body_v(qf):
    """bse/vq_interp.py::make_eval_vq._body, the ``v`` (LR slab) lines."""
    K = BVEC.T @ (np.asarray(qf)[:, None] + GSET)
    K2 = np.sum(K * K, axis=0)
    zero = K2 < 1e-12
    K2s = np.where(zero, 1.0, K2)
    f2d = 1.0 - np.exp(-ZC * np.sqrt(K[0] ** 2 + K[1] ** 2)) * np.cos(K[2] * ZC)
    v = 8.0 * np.pi / K2s * f2d / CELVOL * np.exp(-K2 / (4.0 * ALPHA ** 2))
    return np.where(zero, 0.0, v)


# ---------------------------------------------------------------------------
def max_ulp(a, b):
    """max |a-b| in ULPs of b, over the nonzero entries of b."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    nz = np.abs(b) > 0.0
    if not nz.any():
        return 0.0
    return float(np.max(np.abs(a[nz] - b[nz]) / np.spacing(np.abs(b[nz]))))


# ===========================================================================
# BIT-EXACT cells
# ===========================================================================
@pytest.mark.parametrize("sys_dim", [2, 3])
def test_compute_v_q_per_G_is_bit_exact(sys_dim):
    """GW per-sphere builder: byte equality, every q on the production list."""
    for qf in QLIST:
        ref = inc_compute_v_q_per_G(qf, sys_dim)
        got = v_qG(_K(qf), axis=0, sys_dim=sys_dim, channel="full",
                   units="per_volume", celvol=CELVOL, zc=ZC,
                   zero_tol=TOL_QG_ZERO)
        assert np.array_equal(got, ref), (
            f"sys_dim={sys_dim} q={qf}: max ULP {max_ulp(got, ref)}")


@pytest.mark.parametrize("kind,sys_dim,channel", [
    ("bulk_3d", 3, "full"),
    ("bulk_3d_lr", 3, "lr"),
    ("slab", 2, "full"),
    ("slab_lr", 2, "lr"),
])
def test_minibz_kernel_bare_is_bit_exact(kind, sys_dim, channel):
    """MC kernel, bare units, the 1e-24 NaN guard: byte equality."""
    rng = np.random.RandomState(1)
    dq = rng.uniform(-0.05, 0.05, (200_000, 3))
    shift = np.array([0.03, -0.02, 0.011])
    ref = inc_minibz_kernel_bare(shift, dq, kind)
    got = v_qG(shift[None, :] + dq, axis=1, sys_dim=sys_dim, channel=channel,
               units="bare", alpha=ALPHA, zc=ZC, zero_tol=TOL_MC_NAN)
    assert np.array_equal(got, ref), f"{kind}: max ULP {max_ulp(got, ref)}"


def test_q0_average_slab_point_kernel_is_bit_exact():
    """Slab2D.q0_average._vq_sobol: byte equality.

    Its samples have ``rq[:, 2] == 0`` exactly (the sampler zeroes qz), and
    with a zero third component ``einsum('ij,ij->i')`` and
    ``sum(K*K, axis=1)`` reduce identically.
    """
    rng = np.random.RandomState(2)
    rq = rng.uniform(-0.08, 0.08, (100_000, 3))
    rq[:, 2] = 0.0
    got = v_qG(rq, axis=1, sys_dim=2, channel="full", units="bare",
               zc=ZC, zero_tol=TOL_MC_NAN)
    assert np.array_equal(got, inc_vq_sobol(rq))


def test_q0_average_bulk_point_kernel_within_measured_ulp():
    """Bulk3D._vq_isotropic: 3 ULP, MEASURED, not guessed.

    The incumbent spells |q|^2 as ``einsum('ij,ij->i', q, q)``; the shared
    kernel spells it ``sum(K*K, axis=1)``.  On the components-first
    ``(3, nG)`` layout those are bit-identical; on the points-first
    ``(N, 3)`` C-contiguous layout numpy takes a different reduction path
    and they are not.  Frontera job 7890630 swept N in {1e5, 1e6} x seeds
    {2, 7, 99}: max 3.000 ULP in every cell, mean 0.24, max relative
    5.1e-16.  The bound is flat in N, so 3 is the bound and not a
    sample-size artefact.  This is a q->0 head MC estimator whose own
    statistical error is ~1e-3 relative; 3 ULP is 13 orders below it.
    """
    worst = 0.0
    for seed in (2, 7, 99):
        rng = np.random.RandomState(seed)
        rq = rng.uniform(-0.08, 0.08, (100_000, 3))
        got = v_qG(rq, axis=1, sys_dim=3, channel="full", units="bare",
                   zero_tol=TOL_MC_NAN)
        worst = max(worst, max_ulp(got, inc_vq_isotropic(rq)))
    assert worst <= ULP_EINSUM_RESPELL, f"_vq_isotropic: {worst} ULP"


# ===========================================================================
# VALUE-LEVEL cells (measured <= 2 ULP from the volume-factor respelling)
# ===========================================================================
@pytest.mark.parametrize("kind,channel", [
    ("slab", "full"), ("slab_lr", "lr"), ("slab_sr", "sr"),
])
def test_v_slab_on_set_within_measured_ulp(kind, channel):
    worst = 0.0
    for qf in QLIST:
        ref = inc_v_slab_on_set(qf, kind)
        got = v_qG(_K(qf), axis=0, sys_dim=2, channel=channel,
                   units="per_volume", celvol=CELVOL, alpha=ALPHA, zc=ZC,
                   zero_tol=TOL_QG_ZERO)
        worst = max(worst, max_ulp(got, ref))
    assert worst <= ULP_VOLUME_RESPELL, f"{kind}: {worst} ULP"


def test_eval_vq_body_within_measured_ulp():
    worst = 0.0
    for qf in QLIST:
        ref = inc_eval_vq_body_v(qf)
        got = v_qG(_K(qf), axis=0, sys_dim=2, channel="lr",
                   units="per_volume", celvol=CELVOL, alpha=ALPHA, zc=ZC,
                   zero_tol=TOL_QG_ZERO)
        worst = max(worst, max_ulp(got, ref))
    assert worst <= ULP_VOLUME_RESPELL, f"eval_vq body: {worst} ULP"


# ===========================================================================
# the full product, and the contract
# ===========================================================================
@pytest.mark.parametrize("sys_dim", [2, 3])
@pytest.mark.parametrize("channel", ["full", "lr", "sr"])
def test_sr_plus_lr_reconstructs_full(sys_dim, channel):
    """{bulk, slab} x {full, lr, sr} all exist and SR+LR == full."""
    qf = QLIST[3]
    kw = dict(axis=0, sys_dim=sys_dim, units="bare", alpha=ALPHA, zc=ZC)
    got = v_qG(_K(qf), channel=channel, **kw)
    assert got.shape == (GSET.shape[1],)
    lr = v_qG(_K(qf), channel="lr", **kw)
    sr = v_qG(_K(qf), channel="sr", **kw)
    full = v_qG(_K(qf), channel="full", **kw)
    assert np.allclose(lr + sr, full, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("xp_name", ["numpy", "jax.numpy"])
def test_one_formula_two_array_modules(xp_name):
    """xp=numpy and xp=jax.numpy are the same source line, same answer."""
    if xp_name == "numpy":
        xp = np
    else:
        jnp = pytest.importorskip("jax.numpy")
        import jax
        jax.config.update("jax_enable_x64", True)
        xp = jnp
    qf = QLIST[5]
    got = v_qG(xp.asarray(_K(qf)), axis=0, sys_dim=2, channel="lr",
               units="per_volume", celvol=CELVOL, alpha=ALPHA, zc=ZC,
               zero_tol=TOL_QG_ZERO, xp=xp)
    ref = v_qG(_K(qf), axis=0, sys_dim=2, channel="lr", units="per_volume",
               celvol=CELVOL, alpha=ALPHA, zc=ZC, zero_tol=TOL_QG_ZERO)
    assert np.allclose(np.asarray(got), ref, rtol=1e-15, atol=0.0)


def test_units_has_no_silent_default():
    """Volume convention is a decision, never an inherited default."""
    with pytest.raises(TypeError):
        v_qG(_K(QLIST[0]), axis=0, sys_dim=3)          # units missing
    with pytest.raises(ValueError):
        v_qG(_K(QLIST[0]), axis=0, sys_dim=3, units="Rydberg-per-what")
    with pytest.raises(ValueError):
        # per_volume without a volume is a refusal, not a 1.0 fallback
        v_qG(_K(QLIST[0]), axis=0, sys_dim=3, units="per_volume")


def test_the_two_guard_constants_are_distinct_and_ordered():
    """TOL_QG_ZERO identifies a lattice slot; TOL_MC_NAN guards a 0/0 draw."""
    assert TOL_QG_ZERO == 1e-12
    assert TOL_MC_NAN == 1e-24
    assert TOL_MC_NAN < TOL_QG_ZERO


def test_G0_at_finite_q_is_NOT_zeroed():
    """REFUSAL, measured: zeroing v(G=0) at q!=0 moves makeVq-vs-disk from
    ~1e-9 to 0.33 (bse/vq_interp.py:325-328).  Only |q+G|^2 < TOL_QG_ZERO --
    the q=G=0 lattice slot -- is zeroed.  A newcomer "tidying" this into
    "zero the G=0 column" breaks the body."""
    g0 = int(np.argmin(np.sum(np.abs(GSET), axis=0)))
    assert np.all(GSET[:, g0] == 0.0)
    qf = QLIST[1]                                   # a finite q on the grid
    assert np.dot(qf @ BVEC, qf @ BVEC) > TOL_QG_ZERO
    v = v_qG(_K(qf), axis=0, sys_dim=3, units="per_volume", celvol=CELVOL)
    assert v[g0] > 0.0, "v(G=0) at finite q must be the finite body value"
    v0 = v_qG(_K(np.zeros(3)), axis=0, sys_dim=3, units="per_volume",
              celvol=CELVOL)
    assert v0[g0] == 0.0, "the q=G=0 lattice slot must be zeroed"


def test_mc_tolerance_keeps_near_singular_draws():
    """TOL_MC_NAN must not be raised to TOL_QG_ZERO: MC draws between the two
    carry real weight and zeroing them biases the estimator."""
    K = np.array([[1e-7, 0.0, 0.0]])                # |K|^2 = 1e-14
    v_mc = v_qG(K, axis=1, sys_dim=3, units="bare", zero_tol=TOL_MC_NAN)
    v_lat = v_qG(K, axis=1, sys_dim=3, units="bare", zero_tol=TOL_QG_ZERO)
    assert v_mc[0] == pytest.approx(8.0 * math.pi / 1e-14, rel=1e-14)
    assert v_lat[0] == 0.0
