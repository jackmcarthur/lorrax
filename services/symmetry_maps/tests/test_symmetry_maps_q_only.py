"""The explicit q-only group leaves a raw full-k WFN map untouched."""

from __future__ import annotations

import types

import numpy as np
import pytest

from symmetry_maps import (
    SymMaps,
    derive_q_only_symmetry,
    q_symmetry_receipt_json,
)


def _raw_full_wfn(*, break_energy=False, break_atom=False):
    kgrid = np.asarray((6, 6, 1), dtype=np.int32)
    xyz = np.stack(np.meshgrid(
        np.arange(6), np.arange(6), np.arange(1), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    kpoints = xyz / kgrid[None, :]
    phase = 2.0 * np.pi * kpoints
    band0 = np.cos(phase[:, 0]) + 0.3 * np.cos(phase[:, 1])
    energies = np.stack((band0, band0 + 2.0), axis=-1)[None, :, :]
    if break_energy:
        energies[0, 1, 0] += 1.0e-4
    atoms = np.asarray(((0.13, 0.21, 0.0), (0.87, 0.79, 0.0)))
    if break_atom:
        atoms[1, 0] += 0.02
    return types.SimpleNamespace(
        kpoints=kpoints,
        kgrid=kgrid,
        shift=np.zeros(3),
        nkpts=int(kpoints.shape[0]),
        ntran=1,
        sym_matrices=np.eye(3, dtype=np.int32)[None, :, :],
        translations=np.zeros((1, 3), dtype=np.float64),
        avec=np.eye(3),
        atom_types=np.asarray((1, 1), dtype=np.int32),
        atom_crys=atoms,
        trs_holds=False,
        energies=energies,
    )


def _inversion_group():
    return (
        np.asarray((np.eye(3), -np.eye(3)), dtype=np.int32),
        np.zeros((2, 3), dtype=np.float64),
    )


def test_q_only_inversion_reduces_36_to_20_and_preserves_every_base_attr():
    wfn = _raw_full_wfn()
    matrices, translations = _inversion_group()
    base = SymMaps(wfn)
    derived = derive_q_only_symmetry(
        base, wfn, spatial_matrices=matrices, translations=translations,
        source="unit-test:explicit-inversion")

    assert derived is not base
    assert base.q_irr_full_idx.shape == (36,)
    assert derived.q_irr_full_idx.shape == (20,)
    assert np.any(np.asarray(derived.sym_idx_q) == 1)
    assert int(np.max(derived.sym_idx_q)) < 2  # inversion is unitary, not TRS
    assert derived.trs_allowed is False
    assert derived.q_symmetry_source == "unit-test:explicit-inversion"
    assert len(derived.q_symmetry_digest) == 64
    assert derived.q_symmetry_energy_residual < 1.0e-12
    receipt_json = q_symmetry_receipt_json(derived)
    assert derived.q_symmetry_receipt["source"] == (
        "unit-test:explicit-inversion")
    assert derived.q_symmetry_receipt["q_irr_full_idx"] == (
        np.asarray(derived.q_irr_full_idx).tolist())
    assert derived.q_symmetry_digest == __import__("hashlib").sha256(
        receipt_json.encode("utf-8")).hexdigest()
    assert q_symmetry_receipt_json(base) == q_symmetry_receipt_json(base)
    assert '"source":"wfn"' in q_symmetry_receipt_json(base)

    replaced = set((
        "trs_allowed", "sym_matrices", "sym_mats_k",
        "_sym_mats_k_search", "translations", "R_grid", "Rinv_grid",
        "R_cart", "U_spinor", "R_proper", "irr_idx_q", "sym_idx_q",
        "q_irr_kgrid_int", "q_irr_full_idx",
    ))
    for name, value in vars(base).items():
        if name in replaced:
            continue
        got = getattr(derived, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(got, value, err_msg=name)
            assert not np.shares_memory(got, value), name
        else:
            assert got == value, name


@pytest.mark.parametrize(
    "kind, match",
    (
        ("energy", "band-energy covariance"),
        ("atom", "atom/species closure"),
        ("group", "do not close"),
    ),
)
def test_q_only_group_refuses_failed_authentication(kind, match):
    wfn = _raw_full_wfn(
        break_energy=(kind == "energy"), break_atom=(kind == "atom"))
    base = SymMaps(wfn)
    matrices, translations = _inversion_group()
    if kind == "group":
        matrices = np.asarray((
            np.eye(3),
            ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ), dtype=np.int32)
    with pytest.raises(ValueError, match=match):
        derive_q_only_symmetry(
            base, wfn, spatial_matrices=matrices,
            translations=translations, source=f"negative:{kind}")


def test_q_only_group_requires_raw_identity_k_maps_without_mutating_base():
    wfn = _raw_full_wfn()
    base = SymMaps(wfn)
    before = np.array(base.irr_idx_k, copy=True)
    base.irr_idx_k[3] = 0
    matrices, translations = _inversion_group()
    with pytest.raises(ValueError, match="exact identity k maps"):
        derive_q_only_symmetry(
            base, wfn, spatial_matrices=matrices,
            translations=translations, source="negative:k-map")
    assert base.irr_idx_k[3] == 0
    assert before[3] == 3
