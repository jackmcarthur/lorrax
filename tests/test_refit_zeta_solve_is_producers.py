"""The per-Q ζ refit solves the system THE PRODUCER solved, not a lookalike.

WHAT WENT WRONG.  ``bse.vq_interp._refit_kernels._solve_zeta`` ran a plain
Cholesky with a fixed ``1e-14·|tr C|`` ridge, under a comment claiming it
followed ``isdf.core._ridged_chol`` — a symbol that has never existed in this
tree.  The production charge path is ``replicated_rank_truncate``: a
rank-revealing ``eigh`` pseudo-inverse that DROPS every direction with
λ < ``zeta_rcond``·λ_max.  A ridged Cholesky KEEPS those directions and inverts
through them, so ζ' differed from ζ in precisely the near-null subspace the
producer had discarded — and ``V_Q = Σ_G conj(ζ(G)) v(q+G) ζ(G)`` is QUADRATIC
in ζ.

The consequence was measured on five parents and is monotone in the DISCARDED
FRACTION, and in nothing else: 4.7 % dropped → tile null 3.289, 7.9 % → 16.0,
39.4 % → 50.9, 58.6 % → 139.9, against a 5.0e-02 bracket, while the htransform
Galerkin residual moved eight orders (4.3e-15 … 4.5e-07) across the same five
and the tile null did not follow it.  Nothing had ever executed that gate, so
nothing had ever said so
(``tests/known_failures/2026-08-11-narrowed-zeta-window-clears-fh-and-the-tile-\
null-still-refuses.md`` §4).

WHAT THIS FILE ASSERTS, in the order that makes it mean anything.  The
production gate is the tile null on a real bundle and it costs a GPU leg; these
are the cells that can hold the same fact in a second on CPU:

1. THE INSTRUMENT — on a deliberately rank-deficient C, the pre-fix solve and
   the producer's solve DISAGREE by orders of magnitude.  Without this the rest
   could pass on a well-conditioned matrix, where every solve agrees, and the
   file would be measuring nothing.
2. The refit's kernel now agrees with the producer's ``isdf.core.solve_zeta``
   to round-off, at the same ``zeta_rcond`` — the same comparison, one level
   below the tile.
3. ``zeta_rcond`` is read from the BUNDLE'S provenance, not from the deck.
4. A refit that cannot name the fit it is reproducing REFUSES.
5. The dead ``_ridged_chol`` citation is gone and cannot come back.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

SRC_VQ = os.path.join(os.path.dirname(__file__), "..", "src", "bse",
                      "vq_interp.py")

#: A ζ conditioning cutoff and the matrix built to straddle it.  ``N`` is the
#: μ extent, ``KEEP`` the directions above the cut, and the discarded block
#: sits at 1e-13·λ_max — six decades under a 1e-8 rcond, i.e. exactly the
#: near-null structure an over-complete centroid set produces (κ~1e13).
_N, _KEEP, _RCOND = 24, 16, 1e-8


def _rank_deficient_system(seed: int = 20260811):
    """``(C, Z, n_drop)``: Hermitian-SPD C with a near-null block, and a RHS.

    C = U diag(λ) Uᴴ with λ spanning [1e-13, 1] — the producer keeps the 16
    directions above ``_RCOND``·λ_max and drops 8.  Z is dense in ALL of them,
    which is what makes the two solves disagree: a solve that inverts through
    the null block amplifies Z's component there by up to 1e13.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((_N, _N)) + 1j * rng.standard_normal((_N, _N))
    U, _ = np.linalg.qr(A)
    lam = np.concatenate([
        np.logspace(-13, -11, _N - _KEEP),        # under the cut
        np.logspace(-3, 0, _KEEP),                # kept
    ])
    C = (U * lam[None, :]) @ np.conj(U).T
    C = 0.5 * (C + np.conj(C).T)
    Z = (rng.standard_normal((_N, 6))
         + 1j * rng.standard_normal((_N, 6))).astype(np.complex128)
    return C, Z, _N - _KEEP


def _prefix_solve(C, Z):
    """THE SOLVE THAT WAS THERE — verbatim, so the twin cannot drift from the
    thing it is a twin of.  Cholesky of C + 1e-14·|tr C|·I, two triangular
    solves."""
    import jax.numpy as jnp
    import jax.scipy.linalg as jsl
    ridge = 1e-14 * jnp.abs(jnp.trace(C))
    L = jnp.linalg.cholesky(C + ridge * jnp.eye(C.shape[0], dtype=C.dtype))
    y = jsl.solve_triangular(L, Z, lower=True)
    return np.asarray(jsl.solve_triangular(jnp.conj(L).T, y, lower=False))


# ---------------------------------------------------------------------------
# (1) THE INSTRUMENT: the two solves are not the same solve
# ---------------------------------------------------------------------------

def test_the_two_solves_disagree_on_a_rank_deficient_C():
    """Establish that this system CAN tell them apart before asking whether
    the fix made them agree.  A green comparison on a well-conditioned matrix
    would be a green that measured nothing."""
    pytest.importorskip("jax")
    from isdf.core import solve_zeta_charge_dense

    C, Z, n_drop = _rank_deficient_system()
    prod = np.asarray(solve_zeta_charge_dense(
        C, Z, charge_zeta_solve="rank_truncate", zeta_rcond=_RCOND,
        rank_log=False))
    old = _prefix_solve(C, Z)
    rel = np.linalg.norm(prod - old) / np.linalg.norm(prod)
    assert n_drop == _N - _KEEP
    assert rel > 1e3, (
        f"the pre-fix ridged Cholesky and the producer's rank-truncated "
        f"pseudo-inverse differ by only {rel:.2e} on this system, so this "
        f"fixture cannot distinguish them and every cell below is vacuous")


def test_the_producers_solve_is_the_truncated_pseudo_inverse():
    """``ζ = C⁺Z`` with the sub-rcond block ANNIHILATED, not amplified: the
    defining property, checked against the spectrum directly rather than
    against another implementation."""
    pytest.importorskip("jax")
    from isdf.core import solve_zeta_charge_dense

    C, Z, n_drop = _rank_deficient_system()
    lam, V = np.linalg.eigh(C)
    keep = lam > _RCOND * lam.max()
    assert int((~keep).sum()) == n_drop
    want = (V[:, keep] / lam[keep][None, :]) @ (np.conj(V[:, keep]).T @ Z)
    got = np.asarray(solve_zeta_charge_dense(
        C, Z, charge_zeta_solve="rank_truncate", zeta_rcond=_RCOND,
        rank_log=False))
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-10


# ---------------------------------------------------------------------------
# (2) the refit's kernel IS the producer's kernel
# ---------------------------------------------------------------------------

def test_refit_kernel_matches_the_producer_solve_path():
    """One level below the tile null: the refit's own jitted ``_solve_zeta``
    against ``isdf.core.solve_zeta``'s rank-truncate back-solve, driven
    through ``isdf.factor_c_q`` exactly as the ζ fit drives it."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh

    from bse import vq_interp
    from isdf import factor_c_q, solve_zeta

    C, Z, _ = _rank_deficient_system()
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                axis_names=("x", "y"))
    with mesh:
        L = factor_c_q(jnp.asarray(C)[None], mesh, n_rmu_logical=_N,
                       solver_kind="replicated_rank_truncate",
                       zeta_rcond=_RCOND)
        ref = np.asarray(jax.device_get(solve_zeta(
            L, jnp.asarray(Z)[None], mesh,
            solver_kind="replicated_rank_truncate", n_rmu_logical=_N)))[0]
        _, refit_solve = vq_interp._refit_kernels(
            1, 1, 1, _N, ("rank_truncate", _RCOND, 0.0))
        got = np.asarray(jax.device_get(refit_solve(jnp.asarray(C),
                                                    jnp.asarray(Z))))
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-10, (
        "the refit's ζ solve no longer reproduces the producer's — ζ' then "
        "differs from ζ in the near-null subspace and V_Q is quadratic in it")


def test_the_kernel_cache_keys_on_the_solve_not_only_the_shapes():
    """Two bundles of the same shape at different ``zeta_rcond`` are two
    different programs.  A key that omitted the solve would hand the second
    bundle of a process the first bundle's compiled truncation."""
    pytest.importorskip("jax")
    from bse import vq_interp

    a = vq_interp._refit_kernels(
        1, 1, 1, _N, ("rank_truncate", 1e-8, 0.0))
    b = vq_interp._refit_kernels(
        1, 1, 1, _N, ("rank_truncate", 1e-4, 0.0))
    c = vq_interp._refit_kernels(
        1, 1, 1, _N, ("cholesky", 1e-8, 0.0))
    assert a[1] is not b[1] and a[1] is not c[1]
    assert vq_interp._refit_kernels(
        1, 1, 1, _N, ("rank_truncate", 1e-8, 0.0))[1] is a[1]


def test_the_cholesky_arm_is_still_reachable_and_is_the_old_arithmetic():
    """A deck that pinned ``charge_zeta_solve = cholesky`` gets a refit that
    solves ITS system — the fix is "the producer's solve", not "always
    rank-truncate"."""
    pytest.importorskip("jax")
    from isdf.core import solve_zeta_charge_dense

    C, Z, _ = _rank_deficient_system()
    got = np.asarray(solve_zeta_charge_dense(
        C, Z, charge_zeta_solve="cholesky", zeta_rcond=_RCOND))
    assert np.allclose(got, _prefix_solve(C, Z), rtol=1e-9, atol=0)


def test_a_transverse_family_is_refused_by_name():
    pytest.importorskip("jax")
    from isdf.core import solve_zeta_charge_dense

    C, Z, _ = _rank_deficient_system()
    with pytest.raises(ValueError, match="not a charge-channel solve"):
        solve_zeta_charge_dense(C, Z, charge_zeta_solve="ridge",
                                zeta_rcond=_RCOND)


# ---------------------------------------------------------------------------
# (3) THE BUNDLE IS THE TRUTH: rcond comes off the ζ file, not the deck
# ---------------------------------------------------------------------------

def _prov(**over):
    p = {"charge_zeta_solve": "rank_truncate", "zeta_rcond": "1e-10",
         "zeta_ridge": "0.0", "band_range_left": [0, 52],
         "band_range_right": [0, 52]}
    p.update(over)
    return p


def test_the_solve_triple_comes_from_the_fit_provenance():
    pytest.importorskip("jax")
    from bse import vq_interp

    kind, rcond, ridge = vq_interp._zeta_solve_of(_prov(), "zeta_q.h5")
    assert (kind, rcond, ridge) == ("rank_truncate", 1e-10, 0.0)
    # the EFFECTIVE value: an env-overridden fit records the raw env string,
    # and it has to parse the same way (isdf.core.deprecated_env_record).
    assert vq_interp._zeta_solve_of(
        _prov(zeta_rcond="1e-06"), "zeta_q.h5")[1] == 1e-6


@pytest.mark.parametrize("bad", [
    {"charge_zeta_solve": "ridge"},          # a transverse family
    {"charge_zeta_solve": ""},               # absent
])
def test_a_solve_this_refit_cannot_reproduce_is_refused_by_name(bad):
    pytest.importorskip("jax")
    from bse import vq_interp

    with pytest.raises(SystemExit, match="charge_zeta_solve"):
        vq_interp._zeta_solve_of(_prov(**bad), "zeta_q.h5")


def test_an_unparseable_rcond_is_refused_rather_than_defaulted():
    """There is no default for this.  A refit that guessed the cutoff would
    reproduce a fit that never happened."""
    pytest.importorskip("jax")
    from bse import vq_interp

    with pytest.raises(SystemExit, match="zeta_rcond"):
        vq_interp._zeta_solve_of(_prov(zeta_rcond=None), "zeta_q.h5")


def test_the_zeta_fit_window_is_read_from_the_stamp():
    """The other half of "the bundle is the truth": on a ``zeta_nband``
    -decoupled bundle the ζ-fit window is a strict sub-window of the stored
    band axis, and only the stamp knows which."""
    pytest.importorskip("jax")
    from bse import vq_interp

    assert vq_interp._zeta_fit_window_of(_prov()) == (0, 52)
    # asymmetric left/right — the refit fits ONE band axis, so it cannot
    # reproduce that ζ and must not pretend to.
    assert vq_interp._zeta_fit_window_of(
        _prov(band_range_left=[0, 52], band_range_right=[4, 60])) is None
    assert vq_interp._zeta_fit_window_of({}) is None


def test_provenance_reader_round_trips_a_real_isdf_header(tmp_path):
    """Through the real writer, not a hand-built group: this is the format
    contract between ``gw.gw_init`` and the refit."""
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("jax")
    from bse import vq_interp
    from file_io.isdf_header import (IsdfHeader, stamp_fit_provenance,
                                     write_isdf_header)

    path = tmp_path / "zeta_q.h5"
    with h5py.File(path, "w"):
        pass
    write_isdf_header(str(path), IsdfHeader(
        density="charge", vertex_mu_L=0,
        r_mu_fft_idx=np.zeros((3, 3), dtype=np.int32),
        r_mu_crystal=np.zeros((3, 3), dtype=np.float64)), mode="a")
    stamp_fit_provenance(str(path), json.dumps(_prov()))
    got = vq_interp.read_zeta_fit_provenance(str(path))
    assert got["charge_zeta_solve"] == "rank_truncate"
    assert vq_interp._zeta_solve_of(got, str(path))[1] == 1e-10


def test_an_unstamped_zeta_file_reads_as_absent_not_as_a_default(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("jax")
    from bse import vq_interp
    from file_io.isdf_header import IsdfHeader, write_isdf_header

    path = tmp_path / "zeta_q.h5"
    with h5py.File(path, "w"):
        pass
    write_isdf_header(str(path), IsdfHeader(
        density="charge", vertex_mu_L=0,
        r_mu_fft_idx=np.zeros((3, 3), dtype=np.int32),
        r_mu_crystal=np.zeros((3, 3), dtype=np.float64)), mode="a")
    assert vq_interp.read_zeta_fit_provenance(str(path)) is None
    assert vq_interp.read_zeta_fit_provenance(str(tmp_path / "nope.h5")) is None


# ---------------------------------------------------------------------------
# (4) a refit that cannot name its fit refuses
# ---------------------------------------------------------------------------

def test_refit_vq_refuses_without_the_producers_solve():
    """The old default — ridged Cholesky, silently — is the defect.  There is
    no fallback: a state that cannot name the fit it reproduces produces no
    tile."""
    pytest.importorskip("jax")
    from bse import vq_interp

    with pytest.raises(SystemExit, match="zeta_solve"):
        vq_interp.refit_vq(
            {"nk": 1, "nb": 1, "ns": 1, "n_mu": 4},
            {"rank": 1, "r_chunk": 8, "n_rp": 8}, (0.0, 0.0, 0.0), None)


# ---------------------------------------------------------------------------
# (5) the dead citation stays dead
# ---------------------------------------------------------------------------

def test_no_source_cites_a_symbol_that_does_not_exist():
    """``isdf.core._ridged_chol`` was cited in two places for months and never
    existed.  A comment naming a nonexistent function is worse than no comment:
    it is what let a second, DIFFERENT solve read as the producer's."""
    src_root = os.path.join(os.path.dirname(__file__), "..", "src")
    hits = []
    for root, _dirs, files in os.walk(src_root):
        for name in files:
            if not name.endswith(".py"):
                continue
            p = os.path.join(root, name)
            with open(p, encoding="utf8") as fh:
                for i, line in enumerate(fh, 1):
                    if "_ridged_chol" in line:
                        hits.append(f"{os.path.relpath(p, src_root)}:{i}")
    assert not hits, (
        f"_ridged_chol is cited at {hits} and there is no such symbol under "
        f"src/.  The production charge path is isdf.core.solve_zeta in its "
        f"replicated_rank_truncate mode; the whole-tile entry point is "
        f"isdf.core.solve_zeta_charge_dense.")
