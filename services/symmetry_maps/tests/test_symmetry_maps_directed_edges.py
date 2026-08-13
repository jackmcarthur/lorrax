"""Synthetic gates for directed k-edge unfolding and band-matrix actions."""

from __future__ import annotations

import numpy as np
import pytest

from symmetry_maps import (
    apply_band_matrix_symmetry,
    directed_edge_orbit_table,
    star_broadcast,
)


pytestmark = [pytest.mark.services, pytest.mark.symmetry_maps]

_AXES = np.eye(3, dtype=np.int32)


def _grid_rows(kgrid):
    return np.stack(
        np.meshgrid(*(np.arange(n) for n in kgrid), indexing="ij"), axis=-1,
    ).reshape(-1, 3)


def _flat(rows, kgrid):
    rows = np.asarray(rows) % np.asarray(kgrid)[None, :]
    return ((rows[:, 0] * kgrid[1] + rows[:, 1]) * kgrid[2]
            + rows[:, 2]).astype(np.int32)


def _permutations(kgrid, shift, syms):
    rows = _grid_rows(kgrid)
    coords = (rows + np.asarray(shift)[None, :]) / np.asarray(kgrid)[None, :]
    out = []
    for sym in syms:
        mapped = np.rint(
            (coords @ sym.T) * np.asarray(kgrid)[None, :]
            - np.asarray(shift)[None, :],
        ).astype(np.int32)
        out.append(_flat(mapped, kgrid))
    return np.asarray(out)


def _all_points_point_map(kgrid, shift, syms, sym_choice):
    """Valid pure-array point map with every full point a source row."""
    perms = _permutations(kgrid, shift, syms)
    nk = int(np.prod(kgrid))
    source_full_ids = np.arange(nk, dtype=np.int32)
    sym_idx = np.full(nk, int(sym_choice), dtype=np.int32)
    inverse = np.empty(nk, dtype=np.int32)
    inverse[perms[sym_choice]] = np.arange(nk, dtype=np.int32)
    return inverse, sym_idx, source_full_ids, perms


def _table(kgrid, shift, syms, sym_choice, *, source_steps=_AXES,
           target_steps=_AXES, n_sym_spatial=None):
    irr, sidx, source_ids, perms = _all_points_point_map(
        kgrid, shift, syms, sym_choice)
    if n_sym_spatial is None:
        n_sym_spatial = len(syms)
    table = directed_edge_orbit_table(
        kgrid=kgrid,
        kgrid_shift=shift,
        sym_mats_k=syms,
        irr_idx_k=irr,
        sym_idx_k=sidx,
        source_full_ids=source_ids,
        source_steps=source_steps,
        target_steps=target_steps,
        n_sym_spatial=n_sym_spatial,
    )
    return table, perms


def test_c4_negative_direction_uses_explicit_adjoint_and_endpoints():
    """C4 maps stored +y to -x, so requested +x uses the neighbor adjoint."""
    c4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int32)
    syms = np.asarray([np.eye(3, dtype=np.int32), c4, c4 @ c4,
                       c4 @ c4 @ c4])
    table, perms = _table((3, 3, 2), (0, 0, 0), syms, 1)

    # Target +x is the reverse of a stored +y edge under this C4 row.
    assert np.all(table["reverse"][:, 0])
    np.testing.assert_array_equal(table["source_direction"][:, 0], 1)
    np.testing.assert_array_equal(
        table["source_start_full"][:, 0], table["stored_end_full"][:, 0])
    np.testing.assert_array_equal(
        table["source_end_full"][:, 0], table["stored_start_full"][:, 0])

    # The explicitly oriented endpoints, not a modulo-difference guess, map
    # to the requested target endpoints even across a boundary.
    s = table["sym_idx"][:, 0]
    start_image = perms[s, table["source_start_full"][:, 0]]
    end_image = perms[s, table["source_end_full"][:, 0]]
    np.testing.assert_array_equal(start_image, table["target_start_full"][:, 0])
    np.testing.assert_array_equal(end_image, table["target_end_full"][:, 0])


def test_trs_nonhermitian_link_is_conjugated_not_transposed():
    """Antiunitarity and edge reversal are independent explicit flags."""
    eye = np.eye(3, dtype=np.int32)
    syms = np.asarray([eye, -eye])
    source_steps = np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              dtype=np.int32)
    table, _ = _table(
        (3, 2, 2), (0, 0, 0), syms, 1,
        source_steps=source_steps, n_sym_spatial=1,
    )
    assert np.all(table["antiunitary"])
    assert not np.any(table["reverse"][:, 0])

    link = np.asarray([[1 + 2j, 3 + 5j], [7 + 11j, 13 + 17j]])
    got = apply_band_matrix_symmetry(link, antiunitary=True)
    np.testing.assert_array_equal(got, np.conj(link))
    assert not np.array_equal(got, np.conj(link).T)


def test_anisotropic_half_shifted_mesh_is_an_exact_permutation():
    """The affine shift is part of the grid action, including at wraps."""
    eye = np.eye(3, dtype=np.int32)
    syms = np.asarray([eye, -eye])
    kgrid = (2, 3, 4)
    shift = (0.5, 0.0, 0.5)
    table, _ = _table(kgrid, shift, syms, 0)

    assert table["source_row"].shape == (24, 3)
    np.testing.assert_array_equal(table["target_steps"], _AXES)
    # +z from the last z plane wraps to z=0 without changing x/y.
    row = np.ravel_multi_index((1, 2, 3), kgrid)
    end = np.ravel_multi_index((1, 2, 0), kgrid)
    assert table["target_end_full"][row, 2] == end


def test_c3_non_elementary_step_orbit_refuses():
    """Hexagonal C3 sends one primitive axis to a two-axis combination."""
    eye = np.eye(3, dtype=np.int32)
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=np.int32)
    syms = np.asarray([eye, c3, c3 @ c3])
    nk = 18
    with pytest.raises(
            ValueError,
            match="PT-EDGE-NONPERMUTATION.*elementary-step basis closed"):
        directed_edge_orbit_table(
            kgrid=(3, 3, 2),
            kgrid_shift=(0, 0, 0),
            sym_mats_k=syms,
            irr_idx_k=np.arange(nk),
            sym_idx_k=np.zeros(nk, dtype=np.int32),
            source_full_ids=np.arange(nk),
            source_steps=_AXES,
            n_sym_spatial=3,
        )


def test_missing_direction_refuses_instead_of_nearest_fallback():
    eye = np.eye(3, dtype=np.int32)[None, :, :]
    nk = 8
    with pytest.raises(ValueError, match="PT-EDGE-INCOMPLETE.*missing compact"):
        directed_edge_orbit_table(
            kgrid=(2, 2, 2),
            kgrid_shift=(0, 0, 0),
            sym_mats_k=eye,
            irr_idx_k=np.arange(nk),
            sym_idx_k=np.zeros(nk, dtype=np.int32),
            source_full_ids=np.arange(nk),
            source_steps=np.asarray([[1, 0, 0]]),
            target_steps=np.asarray([[0, 1, 0]]),
            n_sym_spatial=1,
        )


def test_conflicting_duplicate_image_is_a_red_twin():
    """A last-write-wins table would pass this and silently pick one link."""
    eye = np.eye(3, dtype=np.int32)[None, :, :]
    nk = 8
    with pytest.raises(ValueError, match="PT-EDGE-CONFLICT.*duplicate"):
        directed_edge_orbit_table(
            kgrid=(2, 2, 2),
            kgrid_shift=(0, 0, 0),
            sym_mats_k=eye,
            irr_idx_k=np.arange(nk),
            sym_idx_k=np.zeros(nk, dtype=np.int32),
            source_full_ids=np.arange(nk),
            source_steps=np.asarray([[1, 0, 0], [1, 0, 0]]),
            n_sym_spatial=1,
        )


def test_nontrivial_endpoint_sewings_and_reverse_swap():
    link = np.asarray([[1 + 1j, 2 - 3j], [4 + 2j, -1 + 5j]])
    b0 = np.asarray([[0, 1j], [1, 0]])
    b1 = np.asarray([[1, 0], [0, -1j]])

    got = apply_band_matrix_symmetry(
        link, sewing_start=b0, sewing_end=b1)
    np.testing.assert_allclose(got, b0 @ link @ np.conj(b1).T)

    got_reverse = apply_band_matrix_symmetry(
        link, reverse=True, sewing_start=b0, sewing_end=b1)
    expected_reverse = b1 @ np.conj(link).T @ np.conj(b0).T
    np.testing.assert_allclose(got_reverse, expected_reverse)


def test_batched_flags_keep_endpoint_sewings_paired_with_stored_links():
    """Production consumes the table flags in batches, not one Python row."""
    links = np.asarray([
        [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
        [[2 - 1j, 4 - 3j], [6 - 5j, 8 - 7j]],
    ])
    b0 = np.asarray([
        [[1, 0], [0, 1j]],
        [[0, 1], [1j, 0]],
    ])
    b1 = np.asarray([
        [[0, -1j], [1, 0]],
        [[1j, 0], [0, -1]],
    ])
    antiunitary = np.asarray([True, False])
    reverse = np.asarray([False, True])

    got = apply_band_matrix_symmetry(
        links,
        antiunitary=antiunitary,
        reverse=reverse,
        sewing_start=b0,
        sewing_end=b1,
    )
    expected = np.stack([
        apply_band_matrix_symmetry(
            links[i], antiunitary=bool(antiunitary[i]),
            reverse=bool(reverse[i]), sewing_start=b0[i], sewing_end=b1[i])
        for i in range(2)
    ])
    np.testing.assert_allclose(got, expected)


def test_component_mixing_uses_an_explicit_nonband_axis():
    matrices = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    got = apply_band_matrix_symmetry(
        matrices, component_mix=rotation, component_axis=0)
    expected = np.einsum("oi,iab->oab", rotation, matrices)
    np.testing.assert_array_equal(got, expected)


def test_identity_sewing_scalar_path_matches_existing_star_broadcast():
    """Parity gate: the existing scalar/star behavior remains bit-exact."""
    irr = np.asarray([0, 0, 1, 1], dtype=np.int32)
    sidx = np.asarray([0, 1, 0, 1], dtype=np.int32)
    values = np.asarray([1 + 2j, 3 + 5j])
    old_door = star_broadcast(
        values, irr, sidx, 1, trs_reference="ibz_slab")
    same_action = apply_band_matrix_symmetry(
        values[irr], antiunitary=(sidx >= 1))
    np.testing.assert_array_equal(old_door, same_action)
