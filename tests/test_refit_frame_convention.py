"""The per-Q ζ refit transforms ζ(r) → ζ(q+G) in the PRODUCER's frame.

WHAT WENT WRONG.  ``bse.vq_interp.refit_vq`` turned its refitted ζ'(r) into
sphere coefficients with a phase-free FFT followed by a per-centroid winding
phase::

    ZG_μ(G)  =  e^{−2πi q·s_μ} · Σ_r e^{−2πi G·r} ζ_μ(r)          (WRONG)

The ζ writer does something else, and it is not a convention — it is a
different transform::

    ZG_μ(G)  =  Σ_r e^{−2πi (q+G)·r} ζ_μ(r)                        (RIGHT)

(``gw.isdf_fitting`` hands ``qvec_frac`` to
``common.wfn_transforms.accumulate_rchunk_to_gflat``, whose per-q Bloch factor
multiplies the r grid BEFORE the FFT; ``vq_interp.to_sphere`` / ``recon`` are
the same statement in host numpy.)  Replacing an r-dependent factor by a
per-μ constant is exact only for a ζ_μ that is a delta at s_μ, and ζ_μ is a
cardinal interpolation function with support across the whole cell.  So the
two differ in the G-CHANNEL STRUCTURE, not by a phase — and both agree
identically at q = 0, which is the entire reason this survived.

WHAT IT COST.  ``refit_ongrid_null`` on ``dp2628n20`` read **4.688e-06 at Γ
and 1.11–1.17 at every finite q**, with no stored tile reproduced by any of
the 64 (``tests/known_failures/2026-08-11-two-window-contract-lands-and-the-\
sixth-wall-is-finite-q.md`` §5).  Measured with ``m_leg="stored"``, ζ' against
the STORED ζ on the matched sphere: the old spelling 1.02–1.10, the winding
dropped 1.42–1.76, and this transform **1.4e-06 … 4.6e-06**.

WHAT THIS FILE ASSERTS, in the order that makes it mean anything.  The
production gate is the tile null on a real bundle and it costs a GPU leg;
these are the cells that hold the same fact in a second on CPU.

1. THE INSTRUMENT — at a finite q the two spellings DISAGREE at O(1) on a
   random ζ.  Without this the rest could pass on an accident of the fixture
   and the file would be measuring nothing.
2. ``zeta_r_to_sphere_q`` agrees with :func:`vq_interp.to_sphere` — the
   module's already-pinned host transform, the one ``recon`` inverts — to
   round-off, at finite q, on the stored sphere.
3. At q = 0 all three spellings coincide exactly, which is why Γ never saw
   this and why the Γ null does not move under the fix.
4. ``refit_vq`` routes through the helper: the winding phase is gone from the
   module and cannot come back unnoticed.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

SRC_VQ = os.path.join(os.path.dirname(__file__), "..", "src", "bse",
                      "vq_interp.py")

#: Small enough to be instant on CPU, large enough that a centroid winding
#: phase and an r-space Bloch factor cannot coincide by symmetry.
_FFT = (6, 6, 4)
_N_MU = 5
_NGK = 17


def _synthetic_zx(qfrac):
    """A ``zx`` carrying only what the two transforms read.

    One stored q slot at ``qfrac`` with a sphere of ``_NGK`` distinct Miller
    triples, the FFT grid, ``rfrac`` and ``rmu_frac`` — the same keys
    ``load_zeta_coarse`` binds, spelled the same way.
    """
    nx, ny, nz = _FFT
    n_rtot = nx * ny * nz
    rg = np.meshgrid(np.arange(nx) / nx, np.arange(ny) / ny,
                     np.arange(nz) / nz, indexing="ij")
    rng = np.random.default_rng(20260811)
    # A sphere of DISTINCT Millers around the origin, small enough to sit
    # inside the box so no wrap aliases two columns onto one flat slot.
    cand = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
            for k in (-1, 0, 1)][:_NGK]
    gvec = np.asarray(cand, dtype=np.int64).T                  # (3, ngk)
    zx = {
        "nx": nx, "ny": ny, "nz": nz, "n_rtot": n_rtot,
        "rfrac": np.stack([g.ravel() for g in rg], 1),
        "rmu_frac": rng.random((_N_MU, 3)),
        "n_mu": _N_MU, "nq": 1, "ngkmax": _NGK,
        "qfr": np.asarray([qfrac], dtype=np.float64),
        "gvec": gvec[None, :, :],
        "ngk": np.asarray([_NGK], dtype=int),
    }
    zeta = (rng.standard_normal((_N_MU, n_rtot))
            + 1j * rng.standard_normal((_N_MU, n_rtot)))
    return zx, zeta


def _old_spelling(vqi, zx, zeta, qw, fi):
    """The pre-fix transform, verbatim: winding phase on a phase-free FFT."""
    z = np.fft.fftn(zeta.reshape(_N_MU, *_FFT), axes=(1, 2, 3),
                    norm="backward").reshape(_N_MU, zx["n_rtot"])[:, fi]
    return np.exp(-2j * np.pi * (zx["rmu_frac"] @ np.asarray(qw)))[:, None] * z


def _relF(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# A finite q ON a coarse grid the fixture never has to own — the transforms
# take q as a number, and 1/4, 1/2 are the two the tile null failed at.
_Q_FINITE = np.array([0.25, 0.5, 0.5])


def test_old_and_new_spellings_disagree_at_finite_q():
    """THE INSTRUMENT.  Without this the rest could pass on an accident."""
    vqi = pytest.importorskip("bse.vq_interp")
    zx, zeta = _synthetic_zx(_Q_FINITE)
    fi = vqi.flat_idx(zx, zx["gvec"][0])
    new = np.asarray(vqi.zeta_r_to_sphere_q(zx, zeta, _Q_FINITE, fi))
    old = _old_spelling(vqi, zx, zeta, _Q_FINITE, fi)
    assert _relF(old, new) > 0.5, (
        "the pre-fix spelling and the producer's transform agree on this "
        "fixture, so every other cell in this file is vacuous")


def test_refit_transform_is_the_producers_at_finite_q():
    """THE FIX.  ``zeta_r_to_sphere_q`` == ``to_sphere``, the pinned one."""
    vqi = pytest.importorskip("bse.vq_interp")
    zx, zeta = _synthetic_zx(_Q_FINITE)
    fi = vqi.flat_idx(zx, zx["gvec"][0])
    new = np.asarray(vqi.zeta_r_to_sphere_q(zx, zeta, _Q_FINITE, fi))
    ref = vqi.to_sphere(zx, zeta, 0)[:, :_NGK]
    assert _relF(new, ref) < 1e-12, (
        f"refit r→G transform differs from vq_interp.to_sphere by "
        f"{_relF(new, ref):.3e}; they are the same frame and must agree to "
        f"round-off")


def test_gamma_is_where_the_two_spellings_coincide():
    """WHY IT SURVIVED, and why the Γ null does not move under the fix.

    The exactness claim is about the two PHASE FACTORS, and it is asserted
    where it is exact: at q = 0 both ``e^{−2πi q·r}`` and ``e^{−2πi q·s_μ}``
    evaluate to 1.0 + 0.0j in every slot, so neither transform is doing
    anything and the tile cannot move.  The two SPELLINGS are then compared
    at round-off rather than at zero, because ``zeta_r_to_sphere_q`` goes
    through ``local_fftn3`` (jax, and on a GPU leg a GPU FFT) while the
    pre-fix spelling here is ``np.fft`` — two implementations of the same
    transform, measured 2.1e-16 apart on this fixture at four GPUs.  A
    bit-equality assertion there would be a claim about FFT backends, which
    is not what this file is for.
    """
    vqi = pytest.importorskip("bse.vq_interp")
    q0 = np.zeros(3)
    zx, zeta = _synthetic_zx(q0)
    fi = vqi.flat_idx(zx, zx["gvec"][0])
    assert np.all(np.exp(-2j * np.pi * (zx["rfrac"] @ q0)) == 1.0 + 0.0j)
    assert np.all(np.exp(-2j * np.pi * (zx["rmu_frac"] @ q0)) == 1.0 + 0.0j)
    new = np.asarray(vqi.zeta_r_to_sphere_q(zx, zeta, q0, fi))
    old = _old_spelling(vqi, zx, zeta, q0, fi)
    ref = vqi.to_sphere(zx, zeta, 0)[:, :_NGK]
    assert _relF(old, new) < 1e-12, "at q = 0 the two spellings coincide"
    assert _relF(new, ref) < 1e-12


def test_refit_vq_carries_no_centroid_winding_phase():
    """The winding phase is out of the module and cannot drift back in."""
    src = open(SRC_VQ, encoding="utf-8").read()
    body = src[src.index("def refit_vq("):]
    live = [ln for ln in body.splitlines()
            if "rmu_frac" in ln and not ln.lstrip().startswith("#")]
    assert not live, (
        f"refit_vq reads zx['rmu_frac'] again on {len(live)} live line(s): "
        f"{live}.  The centroid winding phase belongs to the F-scheme "
        f"interpolation model, where it is an APPROXIMATION taken out of a "
        f"stored ζ on purpose; in the refit it stood in for the producer's "
        f"r-space Bloch factor and cost every finite-q tile.")
    assert "def zeta_r_to_sphere_q" in src
    assert "zeta_r_to_sphere_q(zx, zeta, qw, fi)" in body
