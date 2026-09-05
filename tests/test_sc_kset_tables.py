"""The SC map retains one k-set for every Sigma-derived table."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from file_io.sigma_output import (
    K_STORAGE_ATTR, K_STORAGE_IBZ, SPREAD_ATTR_PREFIX,
    extract_and_stamp_k_irr)
from gw.dynamic_sigma import OmegaCoverage
from gw.sc_iteration import SCExactHartree, _sc_output_tables_on_loop_kset
from gw.sigma_dispatch import (
    DFT_BASIS_FIELDS, ROTATED_TO_DFT_FIELDS, SIGMA_BASIS_FIELDS,
    SIGMA_KSET_STAR_WEDGE, SIGMA_RESULT_K_AXES, SigmaResult)
from symmetry_maps import KStarMap


SC_SOURCE = Path(__file__).resolve().parents[1] / "src" / "gw" / "sc_iteration.py"


def _rows(nk, *tail, offset=0):
    size = int(nk * np.prod(tail))
    return (np.arange(size, dtype=np.float64) + offset).reshape((nk,) + tail)


def test_every_band_indexed_sigma_field_declares_its_k_axis():
    assert set(SIGMA_RESULT_K_AXES) == (
        set(ROTATED_TO_DFT_FIELDS)
        | set(SIGMA_BASIS_FIELDS)
        | set(DFT_BASIS_FIELDS))


def test_sc_output_seam_selects_every_retained_sigma_table():
    """A Si-4x4x4-sized nontrivial star map leaves no full-BZ table."""
    nk_full, nk_loop, nb, nw = 64, 8, 2, 3
    irr_idx = np.repeat(np.arange(nk_loop, dtype=np.int32), nk_full // nk_loop)
    kstar = KStarMap(
        irr_idx, np.zeros(nk_full, dtype=np.int32), n_sym_spatial=1)
    kept = np.arange(0, nk_full, nk_full // nk_loop)

    matrix = _rows(nk_full, nb, nb)
    diag = _rows(nk_full, nb, offset=1000)
    cube = np.moveaxis(_rows(nk_full, nw, nb, nb, offset=2000), 1, 0)
    head = np.moveaxis(_rows(nk_full, nw, nb, offset=3000), 1, 0)
    sectors = np.moveaxis(_rows(nk_full, 3, nb, nb, offset=4000), 1, 0)
    photon = np.moveaxis(
        _rows(nk_full, 3, 3, nb, offset=5000), 0, 2)
    mask = (_rows(nk_full, nb).astype(np.int64) % 3) != 0
    coverage = OmegaCoverage(
        mask_kn=mask,
        n_uncovered=int(mask.size - np.count_nonzero(mask)),
        fraction_uncovered=float(1.0 - np.count_nonzero(mask) / mask.size),
        omega_min_ev=-5.0,
        omega_max_ev=5.0,
        policy="clamp",
    )
    sigma_full = SigmaResult(
        v_h_kij_ry=matrix,
        v_h_scalar_kij_ry=matrix + 10,
        h_transverse_kij_ry=matrix + 20,
        sigma_x_kij_ry=matrix + 30,
        sigma_xc_kij_ry=matrix + 40,
        sigma_xc_kij_ry_unextrap=matrix + 50,
        sigma_sx_kij_ry=matrix + 60,
        sigma_coh_kij_ry=matrix + 70,
        sigma_lorentz_skij_ry=sectors,
        photon_head_sigma_diag_tskn_ry=photon,
        sigma_c_omega_kij_ry=cube,
        sigma_c_at_dft_diag_ev=diag,
        sigma_c_odd_at_dft_diag_ev=diag + 10,
        omega_dft_rel_ev=diag + 20,
        e_eval_ev=diag + 30,
        head_sigma_diag_w_kn_ry=head,
        omega_coverage=coverage,
    )
    U_full = matrix.astype(np.complex128) + 1j
    exact_full = SCExactHartree(
        scalar_dft=matrix + 80,
        transverse_dft=matrix + 90,
        efermi_ry=0.25,
    )

    delta_full = matrix + 100
    delta_unextrap_full = matrix + 110
    (sigma_loop, U_loop, exact_loop, delta_loop,
     delta_unextrap_loop) = _sc_output_tables_on_loop_kset(
        sigma_full, delta_full, delta_unextrap_full, U_full, exact_full,
        kstar)

    assert sigma_loop.kset == SIGMA_KSET_STAR_WEDGE
    for name, k_axis in SIGMA_RESULT_K_AXES.items():
        value = getattr(sigma_loop, name)
        assert value is not None, f"fixture forgot retained table {name}"
        if name == "omega_coverage":
            value = value.mask_kn
        assert np.shape(value)[k_axis] == nk_loop, name

    np.testing.assert_array_equal(
        np.asarray(sigma_loop.sigma_c_at_dft_diag_ev), diag[kept])
    np.testing.assert_array_equal(
        np.asarray(sigma_loop.sigma_c_omega_kij_ry), cube[:, kept])
    np.testing.assert_array_equal(
        np.asarray(sigma_loop.sigma_lorentz_skij_ry), sectors[:, kept])
    np.testing.assert_array_equal(
        np.asarray(sigma_loop.photon_head_sigma_diag_tskn_ry),
        photon[:, :, kept])
    np.testing.assert_array_equal(np.asarray(U_loop), U_full[kept])
    np.testing.assert_array_equal(
        np.asarray(exact_loop.scalar_dft), exact_full.scalar_dft[kept])
    np.testing.assert_array_equal(
        np.asarray(exact_loop.transverse_dft),
        exact_full.transverse_dft[kept])
    np.testing.assert_array_equal(np.asarray(delta_loop), delta_full[kept])
    np.testing.assert_array_equal(
        np.asarray(delta_unextrap_loop), delta_unextrap_full[kept])
    selected_mask = mask[kept]
    np.testing.assert_array_equal(
        sigma_loop.omega_coverage.mask_kn, selected_mask)
    assert sigma_loop.omega_coverage.n_uncovered == int(
        selected_mask.size - np.count_nonzero(selected_mask))


def test_terminal_h5_boundary_stamps_preselected_rows_without_reselecting():
    irr_idx = np.repeat(np.arange(3, dtype=np.int32), 3)
    sym_idx = np.zeros(9, dtype=np.int32)
    selected = _rows(3, 2, 2)
    payload, attrs_for, compact, _ = extract_and_stamp_k_irr(
        {"sigma_xc_qsgw_kij_ev": selected},
        (irr_idx, sym_idx, 1),
        star_already_selected=True,
    )
    np.testing.assert_array_equal(
        payload["sigma_xc_qsgw_kij_ev"], selected)
    attrs = attrs_for("sigma_xc_qsgw_kij_ev")
    assert attrs[K_STORAGE_ATTR] == K_STORAGE_IBZ
    assert not any(name.startswith(SPREAD_ATTR_PREFIX) for name in attrs)
    assert compact.shape == (9,)


def _function(tree, name):
    matches = [n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1
    return matches[0]


def test_retained_sigma_tables_have_no_second_select_or_broadcast():
    """Bounded source guard: only the named SC output seam owns selection."""
    tree = ast.parse(SC_SOURCE.read_text(encoding="utf-8"))
    seam = _function(tree, "_sc_output_tables_on_loop_kset")
    selector_calls = [
        n for n in ast.walk(seam)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "sigma_result_on_kset"
        and any(kw.arg == "select_rows" for kw in n.keywords)
    ]
    assert len(selector_calls) == 1

    retained_aliases = {
        "sigma", "sigma_result", "sigma_on_shell", "sigma_now", "cube",
        "e_eval", "z_factor_full",
    }
    offenders = []
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n is not seam):
        for call in ast.walk(fn):
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr in {"select", "broadcast"}):
                continue
            names = {n.id for n in ast.walk(call) if isinstance(n, ast.Name)}
            attrs = {n.attr for n in ast.walk(call)
                     if isinstance(n, ast.Attribute)}
            if names & retained_aliases or attrs & set(SIGMA_RESULT_K_AXES):
                offenders.append((fn.name, call.lineno, ast.unparse(call)))
    assert not offenders, (
        "retained Sigma tables may be selected only in "
        f"_sc_output_tables_on_loop_kset; found {offenders}")

    for consumer in (
            "_sc_map_gain_for_call", "_write_sc_eqp_snapshot",
            "_dump_sc_rotation", "dump_sigma_omega_h5_final"):
        fn = _function(tree, consumer)
        calls = [n.func.attr for n in ast.walk(fn)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr in {"select", "broadcast"}]
        assert not calls, f"{consumer} performs a second k-set operation"
