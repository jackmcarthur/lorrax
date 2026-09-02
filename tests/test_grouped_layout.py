"""Contracts for orbit-local runtime views.

The layout is generic and host-only.  These tests pin the facts that a future
X/Y communication kernel is allowed to assume without rechecking them inside
JIT: whole groups, static equal shard shapes, reversible canonical identity,
and shard-local conjugated symmetry gathers.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest


def _labels_from_sizes(sizes, *, spelling=None):
    labels = np.repeat(np.arange(len(sizes), dtype=np.int32), sizes)
    if spelling is not None:
        labels = np.asarray(spelling, dtype=np.int32)[labels]
    return labels


def test_grouped_layout_balances_whole_groups_and_round_trips():
    from common.grouped_layout import (
        GroupedShardLayout,
        SquareGroupedShardLayout,
        build_grouped_shard_layout,
    )

    with pytest.raises(TypeError):
        GroupedShardLayout()
    with pytest.raises(TypeError):
        SquareGroupedShardLayout()

    sizes = np.asarray([11, 9, 7, 5, 4, 3, 2, 1], dtype=np.int32)
    labels = _labels_from_sizes(sizes, spelling=[81, 4, 77, 9, 15, 2, 40, 8])
    layout = build_grouped_shard_layout(labels, 4)

    assert layout.n_logical == int(sizes.sum())
    assert layout.n_padded == layout.n_shards * layout.shard_size
    assert layout.n_pad == layout.n_padded - int(sizes.sum())
    np.testing.assert_array_equal(
        np.sort(layout.packed_to_canonical[layout.active_mask]),
        np.arange(layout.n_logical))
    for group in range(layout.n_groups):
        rows = np.flatnonzero(layout.packed_group_id == group)
        assert rows.size == layout.group_size[group]
        np.testing.assert_array_equal(
            rows, np.arange(rows[0], rows[0] + rows.size))
        assert np.unique(rows // layout.shard_size).size == 1
        assert rows[0] // layout.shard_size == layout.group_owner[group]

    canonical = np.arange(layout.n_logical * 3).reshape(3, -1)
    packed = layout.pack_host(canonical, axis=1, fill_value=-700)
    assert packed.shape == (3, layout.n_padded)
    assert np.all(packed[:, ~layout.active_mask] == -700)
    np.testing.assert_array_equal(layout.unpack_host(packed, axis=1), canonical)

    # Input label names are not scientific identity.  Renaming them leaves
    # every derived permutation and load exactly unchanged.
    renamed = build_grouped_shard_layout(
        labels + np.int32(1003), layout.n_shards)
    np.testing.assert_array_equal(
        renamed.packed_to_canonical, layout.packed_to_canonical)
    np.testing.assert_array_equal(renamed.shard_load, layout.shard_load)
    with pytest.raises(ValueError):
        layout.packed_to_canonical[0] = 99


@pytest.mark.parametrize(
    ("layout_package", "selector_package"),
    [
        ("common", "common"),
        ("src.common", "src.common"),
        ("common", "src.common"),
        ("src.common", "common"),
    ],
)
def test_layout_receipt_crosses_supported_import_styles(
        layout_package, selector_package):
    """Receipt validation is structural across Python's two module names."""
    jax = pytest.importorskip("jax")
    from jax.sharding import Mesh

    layout_module = importlib.import_module(f"{layout_package}.grouped_layout")
    selector_module = importlib.import_module(
        f"{selector_package}.pivoted_cholesky")
    labels = _labels_from_sizes([2, 1])
    layout = layout_module.build_grouped_shard_layout(labels, 1)
    if layout_package != selector_package:
        assert not isinstance(layout, selector_module.GroupedShardLayout)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    selector = selector_module.make_sharded_group_panel_pivoted_cholesky_select(
        mesh, 2, layout, mesh_axis=("x", "y"))
    assert callable(selector)


def test_structural_receipt_validator_refuses_drifting_inverse_metadata():
    from common.grouped_layout import (
        build_grouped_shard_layout,
        validate_grouped_shard_layout,
    )

    layout = build_grouped_shard_layout(_labels_from_sizes([2, 3]), 2)
    assert validate_grouped_shard_layout(layout) is layout
    fields = dict(vars(layout))
    fields["__lorrax_grouped_layout_schema__"] = (
        layout.__lorrax_grouped_layout_schema__)
    wrong_version = SimpleNamespace(**fields)
    wrong_version.__lorrax_grouped_layout_schema__ = (
        "lorrax.grouped-shard-layout.v999")
    with pytest.raises(TypeError, match="build_grouped_shard_layout"):
        validate_grouped_shard_layout(wrong_version)

    bad_inverse = layout.canonical_to_packed.copy()
    bad_inverse[[0, 1]] = bad_inverse[[1, 0]]
    bad_inverse.setflags(write=False)
    fields["canonical_to_packed"] = bad_inverse
    drifted = SimpleNamespace(**fields)
    with pytest.raises(ValueError, match="inverse"):
        validate_grouped_shard_layout(drifted)

    def frozen(values, dtype):
        out = np.asarray(values, dtype=dtype)
        out.setflags(write=False)
        return out

    missing = dict(vars(layout))
    missing["__lorrax_grouped_layout_schema__"] = (
        layout.__lorrax_grouped_layout_schema__)
    missing["n_groups"] = layout.n_groups + 1
    packed_group = layout.packed_group_id.copy()
    packed_group[~layout.active_mask] = missing["n_groups"]
    missing["packed_group_id"] = frozen(packed_group, np.int32)
    missing["group_owner"] = frozen(
        np.append(layout.group_owner, 0), np.int32)
    missing["group_start"] = frozen(
        np.append(layout.group_start, 0), np.int64)
    missing["group_size"] = frozen(
        np.append(layout.group_size, 0), np.int32)
    missing_group = SimpleNamespace(**missing)
    with pytest.raises(ValueError, match="nonempty"):
        validate_grouped_shard_layout(missing_group)


def test_one_axis_layout_serves_identical_square_views_without_xy_locality():
    from common.grouped_layout import build_square_grouped_shard_layout

    labels = _labels_from_sizes([9, 7, 5, 3])
    square = build_square_grouped_shard_layout(labels, (2, 2))
    layout = square.axis

    assert layout.n_shards == 2
    assert square.axis_shard_size == layout.shard_size
    assert layout.n_padded % 4 == 0
    np.testing.assert_array_equal(
        layout.unpack_host(layout.pack_host(np.arange(layout.n_logical))),
        np.arange(layout.n_logical))
    crosses_fine = False
    fine_size = layout.n_padded // 4
    for group in range(layout.n_groups):
        rows = np.flatnonzero(layout.packed_group_id == group)
        axis_owners = np.unique(rows // layout.shard_size)
        assert axis_owners.tolist() == [int(square.axis_group_owner[group])]
        crosses_fine |= np.unique(rows // fine_size).size > 1
    assert crosses_fine, "the fixture must prove XxY locality is not imposed"

    with pytest.raises(ValueError, match="square"):
        build_square_grouped_shard_layout(labels, (2, 4))


def test_finest_layout_allows_all_pad_shards_without_splitting_groups():
    from common.grouped_layout import build_grouped_shard_layout

    layout = build_grouped_shard_layout(_labels_from_sizes([5, 3]), 4)
    assert np.count_nonzero(layout.shard_load == 0) == 2
    for group in range(layout.n_groups):
        rows = np.flatnonzero(layout.packed_group_id == group)
        assert np.unique(rows // layout.shard_size).size == 1


def test_grouped_layout_refuses_nonpositive_shard_multiple():
    from common.grouped_layout import build_grouped_shard_layout

    with pytest.raises(ValueError, match="shard_size_multiple"):
        build_grouped_shard_layout(
            np.asarray([0, 1], dtype=np.int32), 2,
            shard_size_multiple=0)


def test_symmetry_maps_become_shard_local_after_packing():
    from common.grouped_layout import build_grouped_shard_layout
    from symmetry_maps import permutation_orbit_labels

    sizes = [3, 4, 2, 5]
    starts = np.cumsum([0] + sizes[:-1])
    n = sum(sizes)
    identity = np.arange(n, dtype=np.int32)
    rotate = identity.copy()
    reverse = identity.copy()
    for start, size in zip(starts, sizes):
        rows = np.arange(start, start + size)
        rotate[rows] = np.roll(rows, -1)
        reverse[rows] = rows[::-1]
    permutations = np.stack([identity, rotate, reverse])

    labels = permutation_orbit_labels(permutations)
    np.testing.assert_array_equal(np.bincount(labels), sizes)
    layout = build_grouped_shard_layout(labels, 3)
    packed_perm = layout.pack_permutations_host(permutations)
    local_perm = layout.pack_local_permutations_host(permutations)

    owner = np.arange(layout.n_padded) // layout.shard_size
    assert np.all(owner[None, :] == owner[packed_perm])
    np.testing.assert_array_equal(
        local_perm, packed_perm % layout.shard_size)
    values = np.arange(n, dtype=np.int64) * 17
    packed_values = layout.pack_host(values, fill_value=-1)
    for operation in range(permutations.shape[0]):
        got = layout.unpack_host(packed_values[packed_perm[operation]])
        np.testing.assert_array_equal(got, values[permutations[operation]])


def test_nonclosed_group_partition_refuses_a_fake_local_gather():
    from common.grouped_layout import build_grouped_shard_layout

    singleton_layout = build_grouped_shard_layout(
        np.arange(8, dtype=np.int32), 2)
    # Singleton LPT alternates canonical rows between the two owners, so a
    # one-row rotation necessarily crosses (a four-row rotation preserves
    # parity and would be an accidentally local negative control).
    crosses = np.asarray([np.roll(np.arange(8), 1)], dtype=np.int32)
    with pytest.raises(ValueError, match="cross|not closed|moves canonical row"):
        singleton_layout.pack_permutations_host(crosses)


def test_panel_selector_matches_rank_one_reference_in_canonical_order():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from common.pivoted_cholesky import (
        group_block_pivoted_cholesky_select,
        make_sharded_group_panel_pivoted_cholesky_select,
    )
    from common.grouped_layout import build_grouped_shard_layout

    labels = _labels_from_sizes([3, 5, 2, 4, 3])
    n = int(labels.size)
    budget = 12
    rng = np.random.default_rng(908)
    features = (rng.standard_normal((n, 2 * n))
                + 1j * rng.standard_normal((n, 2 * n)))
    gram = features @ features.conj().T
    gram = 0.5 * (gram + gram.conj().T)
    reference = group_block_pivoted_cholesky_select(
        jnp.asarray(gram), budget, jnp.asarray(labels),
        n_groups=int(labels.max()) + 1, tol_rel=1e-13)

    layout = build_grouped_shard_layout(labels, 1)
    packed = layout.pack_host(layout.pack_host(gram, axis=0), axis=1)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    with pytest.raises(TypeError, match="build_grouped_shard_layout"):
        make_sharded_group_panel_pivoted_cholesky_select(
            mesh, budget, object(), mesh_axis=("x", "y"))
    with pytest.raises(ValueError, match="2 fine shards"):
        make_sharded_group_panel_pivoted_cholesky_select(
            mesh, budget, build_grouped_shard_layout(labels, 2),
            mesh_axis=("x", "y"))
    selector = make_sharded_group_panel_pivoted_cholesky_select(
        mesh, budget, layout,
        mesh_axis=("x", "y"), tol_rel=1e-13)
    panel = selector(jnp.asarray(packed))

    np.testing.assert_array_equal(np.asarray(panel[0]), np.asarray(reference[0]))
    assert int(panel[2]) == int(reference[2])
    np.testing.assert_allclose(
        layout.unpack_host(np.asarray(panel[1]), axis=0), np.asarray(reference[1]),
        rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        layout.unpack_host(np.asarray(panel[3])), np.asarray(reference[3]),
        rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        np.asarray(panel[4]), np.asarray(reference[4]),
        rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        np.asarray(panel[5]), np.asarray(reference[5]),
        rtol=2e-11, atol=2e-11)


def test_panel_selector_preserves_rank_floor_and_post_floor_delivery():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from common.pivoted_cholesky import (
        group_block_pivoted_cholesky_select,
        make_sharded_group_panel_pivoted_cholesky_select,
    )
    from common.grouped_layout import build_grouped_shard_layout

    labels = _labels_from_sizes([4, 3, 5, 2, 4])
    n, budget = int(labels.size), 14
    rng = np.random.default_rng(191)
    features = (rng.standard_normal((n, 7))
                + 1j * rng.standard_normal((n, 7)))
    gram = features @ features.conj().T
    gram = 0.5 * (gram + gram.conj().T)
    reference = group_block_pivoted_cholesky_select(
        jnp.asarray(gram), budget, jnp.asarray(labels),
        n_groups=int(labels.max()) + 1, tol_rel=1e-10)

    layout = build_grouped_shard_layout(labels, 1)
    packed = layout.pack_host(layout.pack_host(gram, axis=0), axis=1)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    selector = make_sharded_group_panel_pivoted_cholesky_select(
        mesh, budget, layout,
        mesh_axis=("x", "y"), tol_rel=1e-10)
    panel = selector(jnp.asarray(packed))

    assert int(reference[2]) == 7 == int(panel[2])
    ref_piv, panel_piv = np.asarray(reference[0]), np.asarray(panel[0])
    # Certified directions retain exact order.  After the rank floor every
    # factor column is zero; panel arithmetic may swap zero-factor members of
    # the SAME already-admitted group.  Production emission restores
    # canonical candidate order, so the invariant there is selected groups
    # and rows, not a meaningless within-group post-floor order.
    np.testing.assert_array_equal(panel_piv[:7], ref_piv[:7])
    np.testing.assert_array_equal(
        np.sort(panel_piv[panel_piv >= 0]),
        np.sort(ref_piv[ref_piv >= 0]))
    def group_order(piv):
        sequence = labels[piv[piv >= 0]]
        return sequence[np.sort(np.unique(sequence, return_index=True)[1])]
    np.testing.assert_array_equal(group_order(panel_piv), group_order(ref_piv))
    np.testing.assert_allclose(
        np.asarray(panel[4]), np.asarray(reference[4]),
        rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(
        np.asarray(panel[5]), np.asarray(reference[5]),
        rtol=3e-10, atol=3e-10)
    assert np.count_nonzero(np.asarray(panel[4])) == 7
    assert np.count_nonzero(panel_piv >= 0) == 14 > int(panel[2]), (
        "rank floor must zero factor columns, not stop whole-group delivery")
    assert panel_piv[-1] >= 0 and ref_piv[-1] >= 0


def test_group_select_sentinel_means_no_complete_group_fits():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from common.pivoted_cholesky import (
        group_block_pivoted_cholesky_select,
        make_sharded_group_panel_pivoted_cholesky_select,
    )
    from common.grouped_layout import build_grouped_shard_layout

    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    gram = np.diag(np.asarray([4., 3., 2., 1.]))
    expected = np.asarray([0, 1, -1], dtype=np.int32)
    reference = group_block_pivoted_cholesky_select(
        jnp.asarray(gram), 3, jnp.asarray(labels), n_groups=2,
        tol_rel=1e-10)
    np.testing.assert_array_equal(np.asarray(reference[0]), expected)

    layout = build_grouped_shard_layout(labels, 1)
    packed = layout.pack_host(layout.pack_host(gram, axis=0), axis=1)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    panel = make_sharded_group_panel_pivoted_cholesky_select(
        mesh, 3, layout, mesh_axis=("x", "y"), tol_rel=1e-10)(
            jnp.asarray(packed))
    np.testing.assert_array_equal(np.asarray(panel[0]), expected)


def test_group_select_does_not_reopen_a_group_at_the_exact_rank_floor():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from common.pivoted_cholesky import (
        group_block_pivoted_cholesky_select,
        make_sharded_group_panel_pivoted_cholesky_select,
    )
    from common.grouped_layout import build_grouped_shard_layout

    labels = np.asarray([0, 1, 2, 2, 2, 3, 3], dtype=np.int32)
    gram = np.diag(np.asarray([9., 0., 8., 7., 6., 5., 4.]))
    expected_piv = np.asarray([0, 2, 3, 4, 1], dtype=np.int32)
    expected_taken = np.asarray([9., 8., 7., 6., 0.])
    reference = group_block_pivoted_cholesky_select(
        jnp.asarray(gram), 5, jnp.asarray(labels), n_groups=4,
        tol_rel=1e-10)
    np.testing.assert_array_equal(np.asarray(reference[0]), expected_piv)
    np.testing.assert_array_equal(np.asarray(reference[4]), expected_taken)
    assert int(reference[2]) == 4

    layout = build_grouped_shard_layout(labels, 1)
    packed = layout.pack_host(layout.pack_host(gram, axis=0), axis=1)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    panel = make_sharded_group_panel_pivoted_cholesky_select(
        mesh, 5, layout, mesh_axis=("x", "y"), tol_rel=1e-10)(
            jnp.asarray(packed))
    np.testing.assert_array_equal(np.asarray(panel[0]), expected_piv)
    np.testing.assert_array_equal(np.asarray(panel[4]), expected_taken)
    assert int(panel[2]) == 4


@pytest.mark.parametrize(
    "bad",
    [np.asarray([[0, 0, 2]], dtype=np.int32),
     np.asarray([[0, 1, 3]], dtype=np.int32)],
)
def test_permutation_orbit_labels_refuses_nonpermutations(bad):
    from symmetry_maps import permutation_orbit_labels

    with pytest.raises(ValueError, match="not a permutation"):
        permutation_orbit_labels(bad)
