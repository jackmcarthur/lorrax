"""Cheap core checks for the minimax and Coulomb service doors."""
from __future__ import annotations

import numpy as np
import pytest

import minimax
import vcoul
from gw.gw_config import LorraxConfig
from lxkit.deck_doctor import required_input_paths
from wfn_loader import WfnLoader


def test_one_certified_minimax_rule_keeps_its_payload_contract():
    rule = minimax.lookup(
        family="noncrossing", target="inverse",
        range_value=212.23793639387773,
        error_bound=3.4533298639725701e-8, n_max=64,
    )
    assert rule.provenance.certified is True
    assert rule.node_count <= 64
    assert rule.max_error < 3.4533298639725701e-8
    assert rule.provenance.one_line().endswith("CERTIFIED")


def test_fixture_a_bulk_vq_matches_the_analytic_gamma_limit(core_fixtures):
    root = core_fixtures / "A"
    with WfnLoader(root / "WFN.h5", backend="eager") as loader:
        geometry = vcoul.CoulombGeometry.from_wfn(loader)
        gvec = np.asarray(loader.gvecs(k=[0]), dtype=np.float64).transpose(0, 2, 1)
        ngk = int(loader.ngk_valid(k=[0])[0])
    got = vcoul.v_qG_table(
        vcoul.get_kernel(3), np.zeros((1, 3)), gvec,
        geometry=geometry,
    )[0, :ngk]
    gcart = geometry.bvec.T @ gvec[0, :, :ngk]
    g2 = np.einsum("ig,ig->g", gcart, gcart)
    expected = np.zeros_like(g2)
    np.divide(
        8.0 * np.pi, g2 * geometry.cell_volume,
        out=expected, where=g2 >= 1e-12,
    )
    np.testing.assert_allclose(got, expected, rtol=2e-14, atol=2e-14)
    assert got[np.argmin(g2)] == 0.0


def test_retired_slab_transport_key_is_one_strict_refusal(tmp_path):
    deck = tmp_path / "retired.in"
    deck.write_text(
        "[cohsex]\n"
        "nval = 2\n"
        "ncond = 2\n"
        "nband = 8\n"
        "memory_per_device_gb = 4.0\n"
        "slab_io = auto\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be removed") as exc:
        LorraxConfig.from_input_file(str(deck), print_fn=lambda *_a: None)
    assert "slab_io" in str(exc.value)


def test_deck_doctor_input_closure_accepts_both_tiny_systems(core_fixtures):
    """Strict-parse both production inputs through the doctor's path owner."""
    cases = (("A", "cohsex.in", 21, False),
             ("B", "mpa_sc1.in", 13, True))
    for label, name, n_rmu, wants_restart in cases:
        deck = core_fixtures / label / name
        config = LorraxConfig.from_input_file(
            str(deck), runtime_platform="gpu", resolve_hardware=False,
            print_fn=lambda *_a: None,
        )
        rows = required_input_paths(config, deck, n_rmu=n_rmu)
        assert rows and all(row.path.is_file() for row in rows)
        roles = {row.role for row in rows}
        assert {"DFT wavefunctions", "ISDF centroids",
                "mean-field Hamiltonian"} <= roles
        assert ("ISDF restart tensors" in roles) is wants_restart
