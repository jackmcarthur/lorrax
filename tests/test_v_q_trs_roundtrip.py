"""Synthetic V_q IBZ→full-BZ round-trip test that exposes the TRS bug.

Background
----------
``_unfold_v_q_ibz_to_full`` (``gw/v_q_tile.py``) uses the centroid
permutation ``sym_perm`` (length ``ntran``, built from spatial-only
syms) but indexes it with ``full_to_irr_sym`` (length ``2·ntran``,
from the TRS-augmented ``SymMaps.sym_mats_k``).  When the
``full_to_irr_sym`` value is ``>= ntran`` JAX gathers under
``mode='promise_in_bounds'`` clamp to the last valid index, silently
producing the wrong V_q at every TRS-mapped q.

The fix:
  1. compute ``s_spatial = full_to_irr_sym % ntran`` and use that when
     indexing ``sym_perm`` (or its inverse);
  2. take the complex conjugate of V_ibz when ``full_to_irr_sym >= ntran``.

This test builds a tiny synthetic system where some q-pairs are
related by spatial-only ops (e.g. (0,2,0)←(0,1,0) under σ_y) and
others ONLY by TRS-augmented ops (e.g. (2,0,0)←(1,0,0) under -I).
It then feeds a hand-constructed ζ_irr through both the "correct"
math (in this test file) and the codebase's ``_unfold_v_q_ibz_to_full``
and verifies they agree element-wise.

On the pre-fix HEAD (``c796420`` of ``agent/zeta-bc-scan-shardmap``)
the TRS q's disagree by O(1) absolute (= eV-scale on a physical V_q
matrix element); on the fix branch agreement is <1e-12 at every q.

Self-contained: no WFN, no real centroid file, runs on a single
CPU/GPU in seconds.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from centroid.orbit_syms import compute_centroid_sym_perm
from common.symmetry_maps import SymMaps
from common.symmetry_maps import unfold_v_q


# ===========================================================================
# Geometry helpers
# ===========================================================================

def _build_geometry():
    """3x3x1 q-grid, ntran=2 ({I, σ_y}), 6x6x1 FFT grid, orbit-closed
    centroid set, small G-sphere closed under the full TRS-augmented
    group.
    """
    kgrid = np.array([3, 3, 1], dtype=np.int64)
    fft_grid = np.array([6, 6, 1], dtype=np.int64)

    I3 = np.eye(3, dtype=np.int64)
    sigma_y = np.diag([1, -1, 1]).astype(np.int64)
    sym_matrices = np.stack([I3, sigma_y], axis=0)            # (ntran, 3, 3)
    ntran = sym_matrices.shape[0]
    translations = np.zeros((ntran, 3), dtype=np.float64)

    # TRS-augmented (matches symmetry_maps.py:117-130 logic).
    sym_mats_k_spatial = sym_matrices.transpose(0, 2, 1).copy()
    sym_mats_k = np.concatenate(
        [sym_mats_k_spatial, -sym_mats_k_spatial], axis=0)

    # Centroids: orbit-closed under {I, σ_y}.
    seeds = np.array([
        [1, 1, 0], [2, 0, 0], [3, 2, 0], [4, 3, 0], [0, 0, 0],
    ], dtype=np.int64)
    Rinv = np.rint(np.linalg.inv(sym_matrices)).astype(np.int64)
    all_imgs = set()
    for r in seeds:
        for s in range(ntran):
            img = tuple(((Rinv[s] @ r) % fft_grid).tolist())
            all_imgs.add(img)
    cent_idx = np.array(sorted(all_imgs), dtype=np.int64)

    # ``extend_trs=True`` (TRS-aware sym_perm of length 2·ntran) is the
    # post-fix contract; fall back to the legacy length-``ntran`` form
    # on the pre-fix HEAD where the kwarg doesn't exist yet.  Tests on
    # the pre-fix tree are EXPECTED to fail at the assertion below; the
    # fall-back here just ensures the test even runs.
    try:
        sym_perm = compute_centroid_sym_perm(
            cent_idx.astype(np.int32),
            sym_matrices, 2.0 * np.pi * translations,
            fft_grid.astype(np.int32), validate=True,
            extend_trs=True,
        )
    except TypeError:
        sym_perm = compute_centroid_sym_perm(
            cent_idx.astype(np.int32),
            sym_matrices, 2.0 * np.pi * translations,
            fft_grid.astype(np.int32), validate=True,
        )
    return {
        "kgrid": kgrid, "fft_grid": fft_grid,
        "sym_matrices": sym_matrices, "sym_mats_k": sym_mats_k,
        "ntran": ntran, "translations": translations,
        "cent_idx": cent_idx, "n_rmu": int(cent_idx.shape[0]),
        "sym_perm": sym_perm,
    }


def _reduce_q_to_ibz(geom):
    obj = object.__new__(SymMaps)
    kg = geom["kgrid"]
    kx, ky, kz = np.meshgrid(
        np.arange(kg[0]), np.arange(kg[1]), np.arange(kg[2]),
        indexing='ij')
    obj.kvecs_asints = np.stack(
        [kx.flatten(), ky.flatten(), kz.flatten()], axis=1).astype(np.int64)
    obj.sym_mats_k = geom["sym_mats_k"]
    from common.symmetry_maps import find_irreducible_bz_points
    idx, sym, q_irr = find_irreducible_bz_points(
        obj.kvecs_asints, obj.sym_mats_k, irr_kgrid_int=None)
    return obj.kvecs_asints.copy(), q_irr, idx, sym


def _build_g_sphere(sym_mats_k):
    seeds = [
        np.array([0, 0, 0], dtype=np.int64),
        np.array([1, 0, 0], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int64),
        np.array([1, 1, 0], dtype=np.int64),
        np.array([1, -1, 0], dtype=np.int64),
        np.array([2, 1, 0], dtype=np.int64),
    ]
    n_sym = sym_mats_k.shape[0]
    all_gs = set()
    for g in seeds:
        for s in range(n_sym):
            all_gs.add(tuple((sym_mats_k[s] @ g).tolist()))
    G_set = np.array(sorted(all_gs), dtype=np.int64)
    G_perm = np.zeros((n_sym, G_set.shape[0]), dtype=np.int64)
    lookup = {tuple(g.tolist()): i for i, g in enumerate(G_set)}
    for s in range(n_sym):
        for ig, g in enumerate(G_set):
            G_perm[s, ig] = lookup[tuple((sym_mats_k[s] @ g).tolist())]
    return G_set, G_perm


def _q_to_wrapped_frac(q_int, kgrid):
    qf = q_int.astype(np.float64) / kgrid
    return np.where(qf > 0.5, qf - 1.0, qf)


def _build_v_per_G(q_frac_wrapped, G_set):
    qG = q_frac_wrapped[:, None, :] + G_set[None, :, :].astype(np.float64)
    norm2 = np.einsum('qgi,qgi->qg', qG, qG)
    return 1.0 / np.maximum(norm2, 1e-6)


def _unfold_zeta_proper(zeta_irr, *, irr_idx, sym_idx,
                        sym_perm, ntran, G_perm):
    """Build the canonical full-BZ ζ via the correct TRS-aware rule.

    Spatial (s < ntran): ζ_{q_full, π_s(ν)}(G) = ζ_{q_irr, ν}(S^{-1} G).
    TRS (s ≥ ntran, spatial part s0): ζ_{q_full, π_{s0}(ν)}(G) =
        conj(ζ_{q_irr, ν}(-S0^{-1} G)).  The G-back lookup uses
        ``sym_mats_k`` (which already carries the -1 factor for TRS),
        so we use ``G_perm[s]`` unchanged for both cases.
    """
    n_q_full = irr_idx.shape[0]
    n_rmu = zeta_irr.shape[1]
    n_g = zeta_irr.shape[2]
    zeta_full = np.zeros((n_q_full, n_rmu, n_g), dtype=np.complex128)
    G_perm_inv = np.argsort(G_perm, axis=-1)
    inv_sym_perm = np.argsort(sym_perm, axis=-1)
    for iq in range(n_q_full):
        irr_i = int(irr_idx[iq])
        s = int(sym_idx[iq])
        s_spatial = s % ntran
        is_trs = s >= ntran
        for nu in range(n_rmu):
            nu_irr = int(inv_sym_perm[s_spatial, nu])
            slice_irr = zeta_irr[irr_i, nu_irr, G_perm_inv[s]]
            zeta_full[iq, nu, :] = np.conj(slice_irr) if is_trs else slice_irr
    return zeta_full


def _compute_V(zeta, v_per_G):
    return np.einsum(
        'qmg,qng->qmn', np.conj(zeta), zeta * v_per_G[:, None, :])


# ===========================================================================
# Single-device JAX mesh (works on CPU or 1 GPU)
# ===========================================================================

def _make_mesh():
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    return Mesh(devices, ('x', 'y'))


# ===========================================================================
# Tests
# ===========================================================================

def test_geometry_has_trs_required_qpoints():
    """Sanity check: the geometry contains q's that fold only via TRS."""
    geom = _build_geometry()
    _, _, _, full_to_irr_sym = _reduce_q_to_ibz(geom)
    ntran = geom["ntran"]
    n_trs = int(np.sum(full_to_irr_sym >= ntran))
    n_spatial = int(np.sum(full_to_irr_sym < ntran))
    assert n_trs > 0, (
        f"Geometry must contain TRS-required q's; got n_trs={n_trs}")
    assert n_spatial > 0, (
        f"Geometry must also contain spatial-only q's; got "
        f"n_spatial={n_spatial}")


def test_v_q_trs_roundtrip():
    """V_q IBZ→full-BZ unfold must agree element-wise at every q.

    PRE-FIX: this test FAILS at the TRS-required q's by O(1) absolute
             magnitude (corresponds to O(eV) on a real V_q matrix
             element), while passing at the spatial-only q's.
    POST-FIX: passes at every q with max|ΔV| < 1e-12.
    """
    geom = _build_geometry()
    qs_full, q_irr_int, full_to_irr_idx, full_to_irr_sym = _reduce_q_to_ibz(geom)
    G_set, G_perm = _build_g_sphere(geom["sym_mats_k"])

    n_qpt_irr = q_irr_int.shape[0]
    n_q_full = qs_full.shape[0]
    n_rmu = geom["n_rmu"]
    n_g = G_set.shape[0]
    kgrid = geom["kgrid"]
    ntran = geom["ntran"]

    # v(q+G) at IBZ q's (the ONLY v table the production V_q kernel
    # builds — it never directly references v(q_full)).
    q_irr_frac = _q_to_wrapped_frac(q_irr_int, kgrid)
    v_per_G_ibz = _build_v_per_G(q_irr_frac, G_set)

    # At each full q, the contracted G runs over the full-BZ ζ's G axis;
    # the per-G weight should come from v(q_irr + G_back) where
    # G_back = S^{-1} G.  Reorder using G_perm_inv to align with the
    # full-q ζ.
    v_per_G_full = np.zeros((n_q_full, n_g), dtype=np.float64)
    G_perm_inv = np.argsort(G_perm, axis=-1)
    for iq in range(n_q_full):
        v_per_G_full[iq] = v_per_G_ibz[
            int(full_to_irr_idx[iq]), G_perm_inv[int(full_to_irr_sym[iq])]]

    # Random complex ζ at IBZ q's.
    rng = np.random.default_rng(seed=7)
    zeta_irr = (rng.standard_normal((n_qpt_irr, n_rmu, n_g))
                + 1j * rng.standard_normal((n_qpt_irr, n_rmu, n_g)))

    V_q_ibz = _compute_V(zeta_irr, v_per_G_ibz)

    # Reference: unfold ζ properly, then contract.
    zeta_full = _unfold_zeta_proper(
        zeta_irr,
        irr_idx=full_to_irr_idx, sym_idx=full_to_irr_sym,
        sym_perm=geom["sym_perm"], ntran=ntran, G_perm=G_perm)
    V_ref = _compute_V(zeta_full, v_per_G_full)

    # Codebase path.
    mesh_xy = _make_mesh()
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    V_q_ibz_j = jax.device_put(V_q_ibz.astype(np.complex128), V_sh)
    # ``n_sym_spatial`` (added in the trs-aware-sym-fix patch) tells
    # the unfolder which half of sym_perm/sym_mats_k is TRS.  Pre-fix
    # signatures don't accept it; fall back gracefully.
    unfold_kwargs = dict(
        irr_idx=full_to_irr_idx,
        sym_idx=full_to_irr_sym,
        sym_perm=geom["sym_perm"],
        mesh_xy=mesh_xy,
    )
    try:
        V_codebase = np.asarray(jax.device_get(
            unfold_v_q(
                V_q_ibz_j, n_sym_spatial=ntran, **unfold_kwargs)))
    except TypeError:
        V_codebase = np.asarray(jax.device_get(
            unfold_v_q(V_q_ibz_j, **unfold_kwargs)))

    # Per-q assertion.
    max_diff_per_q = np.max(np.abs(V_codebase - V_ref).reshape(n_q_full, -1),
                            axis=-1)
    ref_norm_per_q = np.linalg.norm(V_ref.reshape(n_q_full, -1), axis=-1)
    rel_per_q = max_diff_per_q / np.maximum(ref_norm_per_q, 1e-30)

    # Build a small report.
    rows = []
    for iq in range(n_q_full):
        s = int(full_to_irr_sym[iq])
        rows.append({
            "q_full": qs_full[iq].tolist(),
            "irr": int(full_to_irr_idx[iq]),
            "sym_idx": s,
            "is_trs": s >= ntran,
            "max_abs": float(max_diff_per_q[iq]),
            "rel": float(rel_per_q[iq]),
        })

    msg = "\nPer-q V_q IBZ-unfold error:\n"
    msg += f"{'q_full':>12} {'irr':>3} {'sym':>3} {'TRS':>4} "
    msg += f"{'max|ΔV|':>12} {'rel':>10}\n"
    for r in rows:
        msg += (f"  ({r['q_full'][0]},{r['q_full'][1]},{r['q_full'][2]:>1})"
                f"   {r['irr']:>2}   {r['sym_idx']:>2}   "
                f"{'T' if r['is_trs'] else 'F':>2}   "
                f"{r['max_abs']:>12.3e} {r['rel']:>10.2e}\n")

    # Tolerance: 1e-12 relative.
    bad = rel_per_q > 1e-12
    assert not bad.any(), (
        f"V_q IBZ→full-BZ unfold disagrees with reference at "
        f"{int(bad.sum())}/{n_q_full} q-points.{msg}")


if __name__ == "__main__":
    # Allow running outside pytest for quick eyeballing.
    test_geometry_has_trs_required_qpoints()
    print("geometry check OK")
    try:
        test_v_q_trs_roundtrip()
        print("V_q TRS round-trip PASS")
    except AssertionError as e:
        print("V_q TRS round-trip FAIL:")
        print(e)
