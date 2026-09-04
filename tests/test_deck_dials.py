"""Contracts for the compact user-facing execution dials."""

from dataclasses import dataclass

import pytest

from gw.gw_config import (
    linalg_resolution,
    read_lorrax_input,
    resolve_mpa_sampling_alpha,
)


def _deck(tmp_path, body: str):
    path = tmp_path / "gw.in"
    path.write_text("[cohsex]\n" + body, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "layout,w_solver,zeta,lu,eigh,sc_eigh",
    [
        ("local", "local", "auto", "auto", "auto", "auto"),
        ("distributed", "distributed", "distributed", "distributed",
         "distributed", "distributed"),
    ],
)
def test_linalg_dial_resolves_one_complete_profile(
        tmp_path, layout, w_solver, zeta, lu, eigh, sc_eigh):
    params = read_lorrax_input(_deck(tmp_path, f"linalg = {layout}\n"))
    resolved = linalg_resolution(params)
    assert resolved.layout == layout
    assert resolved.provenance == "deck"
    assert resolved.w_dyson_solver == w_solver
    assert resolved.distributed_zeta_solve == zeta
    assert resolved.distributed_cholesky == "auto"
    assert resolved.distributed_lu == lu
    assert resolved.batched_route == (
        "batch_reshard" if layout == "local" else "auto")
    assert resolved.eigh_backend == eigh
    assert resolved.sc_eigh == sc_eigh


def test_linalg_default_is_local_with_default_provenance(tmp_path):
    resolved = linalg_resolution(read_lorrax_input(_deck(tmp_path, "")))
    assert resolved.layout == "local"
    assert resolved.provenance == "default"


@pytest.mark.parametrize("value", ["replicated", "fast", "2d"])
def test_linalg_rejects_unknown_values(tmp_path, value):
    with pytest.raises(ValueError, match="expected local or distributed"):
        read_lorrax_input(_deck(tmp_path, f"linalg = {value}\n"))


@pytest.mark.parametrize(
    "key",
    [
        "distributed_zeta_solve",
        "distributed_cholesky",
        "distributed_lu",
        "w_dyson_solver",
        "distrib_la_batched_route",
        "charge_zeta_solve",
        "transverse_zeta_solve",
        "eigh_backend",
        "sc_eigh",
        "use_low_mem_eigh",
    ],
)
def test_retired_linalg_keys_refuse_by_name_with_migration_hint(
        tmp_path, key):
    with pytest.raises(ValueError) as exc:
        read_lorrax_input(_deck(tmp_path, f"{key} = auto\n"))
    message = str(exc.value)
    assert f"Input key '{key}' is retired" in message
    assert "linalg = local | distributed" in message


def test_band_chunk_size_is_folded_under_low_mem_bands(tmp_path):
    with pytest.raises(ValueError, match="band_chunk_size.*retired") as exc:
        read_lorrax_input(_deck(tmp_path, "band_chunk_size = 8\n"))
    assert "low_mem_bands = true | false" in str(exc.value)


def test_strict_keys_is_retired_and_unknown_keys_always_refuse(tmp_path):
    with pytest.raises(ValueError, match="strict_keys.*retired"):
        read_lorrax_input(_deck(tmp_path, "strict_keys = true\n"))
    with pytest.raises(ValueError, match="unrecognized deck key") as exc:
        read_lorrax_input(_deck(tmp_path, "linag = local\n"))
    assert "linag" in str(exc.value)


@dataclass(frozen=True)
class _MPA:
    sampling_alpha: int | None
    sampling_alpha_provenance: str = "unresolved"


@dataclass(frozen=True)
class _Config:
    mpa: _MPA
    raw_input_keys: frozenset


@pytest.mark.parametrize(
    "material,expected,provenance",
    [
        ("insulator", 1, "default for nonmetal"),
        ("metal", 2, "default for metal"),
    ],
)
def test_mpa_sampling_alpha_default_follows_material_class(
        material, expected, provenance):
    lines = []
    resolved = resolve_mpa_sampling_alpha(
        _Config(_MPA(None), frozenset()), material, print_fn=lines.append)
    assert resolved.mpa.sampling_alpha == expected
    assert resolved.mpa.sampling_alpha_provenance == provenance
    assert lines == [
        f"  [config provenance] mpa_sampling_alpha = {expected} "
        f"({provenance})"]


def test_explicit_mpa_sampling_alpha_wins_with_deck_provenance():
    lines = []
    resolved = resolve_mpa_sampling_alpha(
        _Config(_MPA(1), frozenset({"mpa_sampling_alpha"})),
        "metal", print_fn=lines.append)
    assert resolved.mpa.sampling_alpha == 1
    assert resolved.mpa.sampling_alpha_provenance == "deck"
    assert lines == ["  [config provenance] mpa_sampling_alpha = 1 (deck)"]
