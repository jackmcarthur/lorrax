"""The manual-mode local unfold body, its spin-block coefficient, the
O(n_rtot) orbit labels and the left-only basis reorder.

These four names exist for the parent-k ζ fit (``isdf.core._z_q_face_parent``):
a kernel that already stands inside a manual ``shard_map`` cannot call the
top-level ``unfold_isdf_operator`` jit, must hold ONE output spin block at a
time, and changes its right-endpoint (real-grid tile) tables on every call.
Each cell here pins the new spelling to the established one on the same
numbers:

* :func:`unfold_operator_local` == :func:`unfold_isdf_operator` (rectangular,
  axis-local, ``trs_rule="conj"``) with the tile tables passed as RUNTIME
  operands, and with no collective in the lowered HLO;
* :func:`open_spin_block_coefficient` reproduces one ``(a, b)`` block of
  ``_rotate_open_spin_centroid_operator`` on a rectangular operator;
* :func:`real_space_orbit_labels` induces the same partition as the union-find
  of :func:`fft_grid_pullback_perm`, on symmorphic and nonsymmorphic groups;

Geometry, mesh and hand reference are the emulated-mesh suite's own
(``test_symmetry_maps_emulated_mesh``); nothing is re-derived here.
"""

from __future__ import annotations

import numpy as np
import pytest

from lxkit.testing import require_devices
from symmetry_maps import (
    centroid_source_map_and_wrap,
    fft_grid_pullback_perm,
    open_spin_block_coefficient,
    permutation_orbit_labels,
    real_space_orbit_labels,
    unfold_isdf_operator,
    unfold_operator_local,
)

from test_symmetry_maps_emulated_mesh import (  # noqa: E402  (suite helpers)
    _FFT, _IRR, _NTRAN, _Q_IRR, _SEEDS_12, _SYM, _SYMS, _geometry, _mesh)


def _grid_endpoint_geometry():
    """A second orbit-closed set playing the real-grid tile: 8 points."""
    perm, L, n = _geometry(((1, 3), (2, 5), (4, 4), (7, 9)))
    assert n == 8 and n % 4 == 0, f"expected 8 grid points, got {n}"
    return perm, L, n


def _packed(perm, L, n, mesh_shape):
    """Orbit-pack one endpoint: (packed_perm, packed_L, local_perm, layout)."""
    from common.grouped_layout import build_square_grouped_shard_layout
    square = build_square_grouped_shard_layout(
        permutation_orbit_labels(perm), mesh_shape)
    layout = square.axis
    return (layout.pack_permutations_host(perm),
            layout.pack_host(L, axis=1, fill_value=0),
            square.pack_axis_local_permutations_host(perm),
            layout)


def test_local_body_matches_the_top_level_rectangular_unfold():
    """Same numbers, runtime tables, zero collectives."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.shard_map import shard_map
    from functools import partial

    mesh = _mesh(2, 2)
    lp, lL, n_left = _geometry(_SEEDS_12)
    rp, rL, n_right = _grid_endpoint_geometry()
    l_perm, l_L, l_local, l_layout = _packed(lp, lL, n_left, (2, 2))
    r_perm, r_L, r_local, r_layout = _packed(rp, rL, n_right, (2, 2))
    n_lp, n_rp = l_layout.n_padded, r_layout.n_padded

    rng = np.random.default_rng(20260905)
    V = (rng.standard_normal((3, n_left, n_right))
         + 1j * rng.standard_normal((3, n_left, n_right)))
    V_packed = l_layout.pack_host(r_layout.pack_host(V, axis=2), axis=1)
    sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    V_dev = jax.device_put(jnp.asarray(V_packed), sharding)

    expected = np.asarray(unfold_isdf_operator(
        V_dev, irr_idx=_IRR, sym_idx=_SYM,
        sym_perm=l_perm, L_table=l_L,
        right_sym_perm=r_perm, right_L_table=r_L,
        q_irr_frac=_Q_IRR, mesh_xy=mesh, n_sym_spatial=_NTRAN,
        trs_rule="conj",
        axis_local_sym_perm=l_local, right_axis_local_sym_perm=r_local))

    @partial(shard_map, mesh=mesh,
             in_specs=(P(None, 'x', 'y'), P(None, None), P(None, None, None)),
             out_specs=P(None, 'x', 'y'), check_vma=False)
    def _body(V_local, right_local_perm, right_L):
        return unfold_operator_local(
            V_local, irr_idx=_IRR, sym_idx=_SYM, q_irr_frac=_Q_IRR,
            left_local_perm=l_local, left_L_table=l_L,
            right_local_perm=right_local_perm, right_L_table=right_L,
            n_sym_spatial=_NTRAN)

    run = jax.jit(_body)
    got = np.asarray(run(
        V_dev, jnp.asarray(r_local, dtype=jnp.int32),
        jnp.asarray(r_L, dtype=jnp.float64)))
    assert got.shape == (len(_IRR), n_lp, n_rp)
    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)

    # The pad rows of an orbit-packed endpoint are fixed points; the table
    # must have kept them at exactly zero.
    assert np.all(got[:, ~l_layout.active_mask, :] == 0.0)
    assert np.all(got[:, :, ~r_layout.active_mask] == 0.0)
    # The antiunitary rows are live in this map, so a check that only
    # covered unitary rows would pass with the conjugation removed.
    assert np.any(_SYM >= _NTRAN)
    unconj = got.copy()
    trs_rows = np.flatnonzero(_SYM >= _NTRAN)
    unconj[trs_rows] = np.conj(unconj[trs_rows])
    assert np.max(np.abs(unconj - expected)) > 0.1 * np.max(np.abs(expected))

    hlo = run.lower(
        V_dev, jnp.asarray(r_local, dtype=jnp.int32),
        jnp.asarray(r_L, dtype=jnp.float64)).compiler_ir(
            dialect="hlo").as_hlo_text().lower()
    forbidden = ("all-to-all(", "all-gather(", "collective-permute(",
                 "reduce-scatter(", "all-reduce(")
    assert not [name for name in forbidden if name in hlo]


def test_open_spin_block_coefficient_is_one_block_of_the_rotation():
    """sum_{c,d} coef[a,b][c,d] O[c,:,d,:] == (U O U†)[a,:,b,:], rectangular."""
    import jax.numpy as jnp
    from symmetry_maps.maps import _rotate_open_spin_centroid_operator

    rng = np.random.default_rng(20260905)
    nk, ns, m, n = 5, 2, 6, 9
    spatial = (rng.standard_normal((nk, ns, m, ns, n))
               + 1j * rng.standard_normal((nk, ns, m, ns, n)))
    U = np.empty((nk, ns, ns), dtype=np.complex128)
    for k in range(nk):
        U[k], _ = np.linalg.qr(
            rng.standard_normal((ns, ns)) + 1j * rng.standard_normal((ns, ns)))
    rotated = np.asarray(_rotate_open_spin_centroid_operator(
        jnp.asarray(spatial), U))
    for a in range(ns):
        for b in range(ns):
            coef = np.asarray(open_spin_block_coefficient(U, a, b))
            assert coef.shape == (nk, ns, ns)
            block = sum(
                coef[:, c, d][:, None, None] * spatial[:, c, :, d, :]
                for c in range(ns) for d in range(ns))
            np.testing.assert_allclose(
                block, rotated[:, a, :, b, :], rtol=2.0e-13, atol=2.0e-13)


@pytest.mark.parametrize("tnp", [
    np.zeros((2, 3)),
    np.array([[0.0, 0.0, 0.0], [0.0, np.pi, 0.0]]),   # σ_y with a half glide
])
def test_orbit_labels_match_the_pullback_union_find(tnp):
    """Same partition as permutation_orbit_labels(fft_grid_pullback_perm)."""
    labels = real_space_orbit_labels(_SYMS, tnp, _FFT)
    reference = permutation_orbit_labels(
        fft_grid_pullback_perm(_SYMS, tnp, _FFT))
    n_rtot = int(np.prod(_FFT))
    assert labels.shape == (n_rtot,)
    # A label is the orbit's smallest flat index, so it names its own orbit.
    assert np.all(labels[labels] == labels)
    # Partitions agree: the label pairs are in bijection.
    pairs = {(int(a), int(b)) for a, b in zip(labels, reference)}
    assert len({a for a, _ in pairs}) == len(pairs) == len({b for _, b in pairs})
    # Nontrivial: σ_y (with or without the glide) pairs up most points.
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]
    assert sizes.max() == 2 and (sizes == 2).sum() > 50


def test_unavailable_unused_rows_preserve_canonical_trs_unfold():
    """Canonical row2 is antiunitary for n_spatial2 and conjugates the complex operator."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    mesh = _mesh(2, 2)
    operator = np.array([[[1, 2j, 0, 0], [-2j, 3, 0, 0],
                          [0, 0, 4, 1j], [0, 0, -1j, 5]]])
    perm = np.array([[0, 1, 2, 3], [-1, -1, -1, -1],
                     [0, 1, 2, 3], [-1, -1, -1, -1]])
    local = np.where(perm < 0, -1, perm % 2)
    dev = jax.device_put(jnp.asarray(operator), NamedSharding(mesh, P(None, 'x', 'y')))
    result = unfold_isdf_operator(
        dev, irr_idx=np.array([0, 0]), sym_idx=np.array([0, 2]),
        sym_perm=perm, L_table=np.zeros((4, 4, 3)),
        q_irr_frac=np.array([[0.25, 0, 0]]), mesh_xy=mesh, n_sym_spatial=2,
        axis_local_sym_perm=local)
    np.testing.assert_array_equal(result, np.concatenate([operator, operator.conj()]))
    with pytest.raises(ValueError, match="selected centroid action is unavailable"):
        unfold_isdf_operator(
            dev, irr_idx=np.array([0, 0]), sym_idx=np.array([0, 3]),
            sym_perm=perm, L_table=np.zeros((4, 4, 3)),
            q_irr_frac=np.array([[0.25, 0, 0]]), mesh_xy=mesh, n_sym_spatial=2)


def test_identity_shortcut_refuses_an_unavailable_identity_action():
    """Even a q axis requiring no expansion must authenticate its centroid identity."""
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="selected centroid action is unavailable"):
        unfold_isdf_operator(
            jnp.eye(4)[None], irr_idx=np.array([0]), sym_idx=np.array([0]),
            sym_perm=-np.ones((2, 4), dtype=int), L_table=np.zeros((2, 4, 3)),
            q_irr_frac=np.zeros((1, 3)), mesh_xy=_mesh(2, 2), n_sym_spatial=1)
