"""QE operation typing, raw-axis orientation, and magnetic-row gates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from symmetry_maps import (
    QESymmetryBinding,
    SymMaps,
    bind_qe_symmetry_receipt,
    build_qgrid_trs_policy,
    qe_xml_seitz_to_bgw,
    read_qe_symmetry_receipt,
    tau_phase_row,
    unfold_file_wedge_polar_matrix,
)


_ORDER3 = np.asarray([
    [0, -1, 0],
    [1, -1, 0],
    [0, 0, 1],
], dtype=np.int32)
_TAU_QE = np.asarray([1.0 / 3.0, 2.0 / 3.0, 0.0])
_REAL_QE_SCHEMA = Path(__file__).with_name("data") / "qe_7_2_si_schema_min.xml"


def _matrix_text(matrix) -> str:
    return "\n".join(" ".join(str(int(v)) for v in row) for row in matrix)


def _schema_text(*, second_matrix=_ORDER3, second_tr=True,
                 second_tau=_TAU_QE, kpoints=((0, 0, 0), (0, 1 / 3, 0)),
                 nosym=False, noinv=True, no_t_rev=False,
                 do_magnetization=False) -> str:
    krows = "\n".join(
        f'<ks_energies><k_point weight="0.5">{x} {y} {z}</k_point></ks_energies>'
        for x, y, z in kpoints)
    return f"""<?xml version="1.0"?>
<espresso>
  <input>
    <symmetry_flags>
      <nosym>{str(nosym).lower()}</nosym>
      <noinv>{str(noinv).lower()}</noinv>
      <no_t_rev>{str(no_t_rev).lower()}</no_t_rev>
      <force_symmorphic>false</force_symmorphic>
    </symmetry_flags>
  </input>
  <output>
    <basis_set><reciprocal_lattice>
      <b1>1 0 0</b1><b2>0 1 0</b2><b3>0 0 1</b3>
    </reciprocal_lattice></basis_set>
    <symmetries>
      <symmetry><info time_reversal="false">crystal_symmetry</info>
        <rotation>{_matrix_text(np.eye(3, dtype=np.int32))}</rotation>
        <fractional_translation>0 0 0</fractional_translation>
      </symmetry>
      <symmetry><info time_reversal="{str(second_tr).lower()}">crystal_symmetry</info>
        <rotation>{_matrix_text(second_matrix)}</rotation>
        <fractional_translation>{second_tau[0]} {second_tau[1]} {second_tau[2]}</fractional_translation>
      </symmetry>
    </symmetries>
    <band_structure><noncolin>true</noncolin>
      <starting_k_points><monkhorst_pack nk1="1" nk2="3" nk3="1"/></starting_k_points>
      {krows}
    </band_structure>
    <magnetization>
      <noncolin>true</noncolin>
      <do_magnetization>{str(do_magnetization).lower()}</do_magnetization>
    </magnetization>
  </output>
</espresso>
"""


def _write_schema(tmp_path, **kwargs):
    save = tmp_path / "typed.save"
    save.mkdir()
    schema = save / "data-file-schema.xml"
    schema.write_text(_schema_text(**kwargs), encoding="utf-8")
    return schema


def _binding_stub(receipt):
    return SimpleNamespace(
        kgrid=np.asarray([1, 3, 1], dtype=np.int32),
        nspinor=2,
        nkpts=2,
        kpoints=np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0 / 3.0, 0.0]]),
        ntran=int(receipt.sym_matrices.shape[0]),
        sym_matrices=receipt.sym_matrices,
        translations=receipt.translations,
    )


def test_xml_raw_orientation_and_seitz_translation_are_distinct(tmp_path):
    receipt = read_qe_symmetry_receipt(_write_schema(tmp_path))
    np.testing.assert_array_equal(receipt.sym_matrices[1], _ORDER3)
    assert not np.array_equal(_ORDER3, _ORDER3.T)
    expected_tnp = 2.0 * np.pi * (np.linalg.inv(_ORDER3) @ _TAU_QE)
    np.testing.assert_allclose(receipt.translations[1], expected_tnp,
                               atol=2.0e-15)
    # Reciprocal action transposes once, after the raw ingress boundary.
    k = np.asarray([1, 2, 0], dtype=np.int32)
    np.testing.assert_array_equal(receipt.sym_matrices[1].T @ k,
                                  np.asarray([2, -3, 0]))
    assert not np.array_equal(receipt.sym_matrices[1] @ k,
                              receipt.sym_matrices[1].T @ k)


def test_minimized_real_qe_schema_pins_storage_order_and_translation_units():
    receipt = read_qe_symmetry_receipt(_REAL_QE_SCHEMA)
    matrix = np.asarray([[-1, -1, -1], [0, 0, 1], [0, 1, 0]])
    tau_qe = np.asarray([-0.5, 2.775557561562891e-17,
                         -2.775557561562891e-17])

    assert receipt.nspinor == 1
    assert receipt.no_t_rev
    np.testing.assert_array_equal(receipt.sym_matrices[1], matrix)
    np.testing.assert_allclose(
        receipt.translations[1],
        2.0 * np.pi * (np.linalg.inv(matrix) @ tau_qe), atol=2.0e-15)
    assert receipt.antiunitary.tolist() == [False, False]


def test_inactive_lattice_symmetry_rows_are_not_bound_to_the_wfn(tmp_path):
    lattice = np.diag([-1, 1, -1]).astype(np.int32)
    inactive = f"""
      <symmetry><info time_reversal="false">lattice_symmetry</info>
        <rotation>{_matrix_text(lattice)}</rotation>
      </symmetry>
    """
    text = _schema_text().replace(
        "    <symmetries>",
        "    <symmetries><nsym>2</nsym><nrot>3</nrot>",
    ).replace("    </symmetries>", inactive + "    </symmetries>")
    save = tmp_path / "lattice-candidates.save"
    save.mkdir()
    schema = save / "data-file-schema.xml"
    schema.write_text(text, encoding="utf-8")

    receipt = read_qe_symmetry_receipt(schema)
    assert receipt.sym_matrices.shape == (2, 3, 3)
    assert receipt.antiunitary.tolist() == [False, True]
    np.testing.assert_array_equal(receipt.sym_matrices[1], _ORDER3)


def test_declared_active_symmetry_count_mismatch_refuses(tmp_path):
    text = _schema_text().replace(
        "    <symmetries>", "    <symmetries><nsym>3</nsym><nrot>3</nrot>")
    save = tmp_path / "bad-active-count.save"
    save.mkdir()
    schema = save / "data-file-schema.xml"
    schema.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="declared nsym=3.*found 2"):
        read_qe_symmetry_receipt(schema)


def test_binding_refuses_transposed_major_minor_axis_twin(tmp_path):
    receipt = read_qe_symmetry_receipt(_write_schema(tmp_path))
    wfn = _binding_stub(receipt)
    wfn.sym_matrices = np.asarray(wfn.sym_matrices).transpose(0, 2, 1)
    with pytest.raises(ValueError, match="major/minor-axis bug"):
        bind_qe_symmetry_receipt(wfn, receipt)


def test_binding_authenticates_k_rows_and_nonsymmorphic_phase(tmp_path):
    receipt = read_qe_symmetry_receipt(_write_schema(tmp_path))
    binding = bind_qe_symmetry_receipt(_binding_stub(receipt), receipt)
    assert binding.antiunitary.tolist() == [False, True]
    assert not binding.qe_permitted_pure_time_reversal
    gvec = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.int32)
    phase = tau_phase_row(
        -receipt.sym_matrices[1].T, receipt.translations[1], gvec)
    expected = np.exp(-1j * ((-receipt.sym_matrices[1].T @ gvec.T).T
                             @ receipt.translations[1]))
    np.testing.assert_allclose(phase, expected, atol=2.0e-15)


@pytest.mark.parametrize(
    ("noinv", "no_t_rev", "do_magnetization", "permitted"),
    ((False, False, False, True),
     (False, True, False, True),
     (False, False, True, False),
     (False, True, True, False),
     (True, False, False, False),
     (True, True, False, False)),
)
def test_pure_time_reversal_follows_qe_noinv_and_magnetic_symmetry(
        tmp_path, noinv, no_t_rev, do_magnetization, permitted):
    receipt = read_qe_symmetry_receipt(_write_schema(
        tmp_path,
        second_tr=False,
        noinv=noinv,
        no_t_rev=no_t_rev,
        do_magnetization=do_magnetization,
    ))
    binding = bind_qe_symmetry_receipt(_binding_stub(receipt), receipt)
    assert binding.qe_permitted_pure_time_reversal is permitted


def test_missing_noncollinear_magnetization_receipt_fails_closed(tmp_path):
    text = _schema_text(second_tr=False, noinv=False).replace(
        "      <do_magnetization>false</do_magnetization>\n", "")
    save = tmp_path / "old.save"
    save.mkdir()
    schema = save / "data-file-schema.xml"
    schema.write_text(text, encoding="utf-8")
    receipt = read_qe_symmetry_receipt(schema)
    assert receipt.do_magnetization is None
    binding = bind_qe_symmetry_receipt(_binding_stub(receipt), receipt)
    assert not binding.qe_permitted_pure_time_reversal


def test_nosym_keeps_only_unitary_identity_but_can_retain_pure_tr(tmp_path):
    receipt = read_qe_symmetry_receipt(_write_schema(
        tmp_path,
        second_tr=False,
        nosym=True,
        noinv=False,
        no_t_rev=False,
    ))
    np.testing.assert_array_equal(
        receipt.sym_matrices, np.eye(3, dtype=np.int32)[None])
    assert receipt.antiunitary.tolist() == [False]
    binding = bind_qe_symmetry_receipt(_binding_stub(receipt), receipt)
    assert binding.qe_permitted_pure_time_reversal


def _typed_symmaps_stub(binding):
    # I and antiunitary C2_y.  Its reciprocal action is -S = diag(1,-1,1),
    # so stored y=1/3 reaches y=2/3 without authorizing global k<->-k.
    matrices = np.stack([
        np.eye(3, dtype=np.int32),
        np.diag([-1, 1, -1]).astype(np.int32),
    ])
    return SimpleNamespace(
        ntran=2,
        sym_matrices=matrices,
        translations=np.zeros((2, 3), dtype=np.float64),
        kpoints=np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0 / 3.0, 0.0]]),
        kgrid=np.asarray([1, 3, 1], dtype=np.int32),
        shift=np.zeros(3),
        nkpts=2,
        nspinor=2,
        avec=np.eye(3),
        atom_types=np.asarray([1], dtype=np.int32),
        atom_crys=np.zeros((1, 3)),
        trs_holds=False,
        qe_symmetry_binding=binding,
        qe_symmetry_diagnostic="synthetic authenticated receipt",
    )


def test_broken_global_trs_keeps_only_qe_typed_antiunitary_rows():
    binding = QESymmetryBinding(
        schema_path="synthetic/data-file-schema.xml",
        schema_sha256="a" * 64,
        antiunitary=np.asarray([False, True]),
        qe_permitted_pure_time_reversal=False,
    )
    with pytest.warns(RuntimeWarning, match="operation-specific antiunitary"):
        sym = SymMaps(_typed_symmaps_stub(binding))
    assert sym.active_symmetry_rows.tolist() == [0, 3]
    assert int(sym.sym_idx_k[2]) == 3
    assert int(sym.sym_idx_q[2]) == 3
    assert not sym.trs_allowed

    policy = build_qgrid_trs_policy(
        trs_measured=False,
        irr_idx_q=sym.irr_idx_q,
        sym_idx_q=sym.sym_idx_q,
        q_irr_full_idx=sym.q_irr_full_idx,
        kgrid=(1, 3, 1),
        n_sym_spatial=2,
        active_symmetry_rows=sym.active_symmetry_rows,
        context="typed-test",
    )
    np.testing.assert_array_equal(policy.unfold_sym_idx, sym.sym_idx_q)
    assert "operation-specific antiunitary" in policy.announcement()


def test_parent_covariance_uses_typed_antiunitary_little_group_action():
    policy = build_qgrid_trs_policy(
        trs_measured=False,
        irr_idx_q=np.asarray([0], dtype=np.int32),
        sym_idx_q=np.asarray([3], dtype=np.int32),
        q_irr_full_idx=np.asarray([0], dtype=np.int32),
        kgrid=(1, 1, 1),
        n_sym_spatial=2,
        active_symmetry_rows=np.asarray([0, 3], dtype=np.int32),
        context="typed-little-group",
    )
    spatial = np.stack([
        np.eye(3, dtype=np.int32),
        np.diag([-1, 1, -1]).astype(np.int32),
    ])
    sym_mats_k = np.concatenate([spatial, -spatial], axis=0)
    permutation = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]],
                             dtype=np.int32)
    wraps = np.zeros((4, 2, 3), dtype=np.int32)

    # P conj(V) P = V for the authorized antiunitary row 3, while P V P != V.
    covariant = np.asarray([[[1.0, 1.0j], [-1.0j, 1.0]]],
                           dtype=np.complex128)
    good = policy.measure_covariance(
        covariant,
        q_irr_frac=np.zeros((1, 3)),
        q_irr_full_idx=np.asarray([0], dtype=np.int32),
        sym_mats_k=sym_mats_k,
        sym_perm=permutation,
        L_table=wraps,
    )
    assert good["worst_sym"] == 3
    assert good["max_rel"] < 1.0e-14

    noncovariant = np.asarray([[[1.0, 0.0], [0.0, 2.0]]],
                              dtype=np.complex128)
    bad = policy.measure_covariance(
        noncovariant,
        q_irr_frac=np.zeros((1, 3)),
        q_irr_full_idx=np.asarray([0], dtype=np.int32),
        sym_mats_k=sym_mats_k,
        sym_perm=permutation,
        L_table=wraps,
    )
    assert bad["worst_sym"] == 3
    assert bad["max_rel"] > 0.4


def test_file_wedge_polar_matrix_uses_typed_antiunitary_row_once():
    binding = QESymmetryBinding(
        schema_path="synthetic/data-file-schema.xml",
        schema_sha256="a" * 64,
        antiunitary=np.asarray([False, True]),
        qe_permitted_pure_time_reversal=False,
    )
    with pytest.warns(RuntimeWarning, match="operation-specific antiunitary"):
        sym = SymMaps(_typed_symmaps_stub(binding))
    rng = np.random.default_rng(9)
    wedge = (rng.standard_normal((sym.nk_red, 3, 2, 2))
             + 1j * rng.standard_normal((sym.nk_red, 3, 2, 2)))
    got = unfold_file_wedge_polar_matrix(sym, wedge)
    expected = np.empty_like(got)
    ntran = int(np.asarray(sym.sym_mats_k).shape[0]) // 2
    actions = sym.cartesian_action(
        np.arange(2 * ntran, dtype=np.int32), axial=False, time_odd=True)
    for ik in range(sym.nk_tot):
        row = int(sym.sym_idx_k[ik])
        parent = wedge[int(sym.irr_idx_k[ik])]
        if row >= ntran:
            parent = np.conj(parent)
        expected[ik] = np.einsum("oi,iab->oab", actions[row], parent)
    np.testing.assert_allclose(got, expected, atol=2.0e-15)
    assert int(sym.sym_idx_k[2]) == 3
    # The row-3 Cartesian table already includes velocity's TR-odd minus.
    np.testing.assert_array_equal(actions[3], -actions[1])


def test_file_wedge_polar_matrix_uses_forward_not_inverse_rotation():
    forward = np.asarray([[0.0, -1.0, 0.0],
                          [1.0, -1.0, 0.0],
                          [0.0, 0.0, 1.0]])
    sym = SimpleNamespace(
        nk_red=1,
        nk_tot=1,
        irr_idx_k=np.asarray([0], dtype=np.int32),
        sym_idx_k=np.asarray([0], dtype=np.int32),
        sym_mats_k=np.stack([np.eye(3), -np.eye(3)]),
        cartesian_action=lambda rows, *, axial, time_odd: np.stack(
            [forward, -forward])[np.asarray(rows, dtype=np.int32)],
    )
    wedge = np.asarray([[[[1.0]], [[2.0]], [[3.0]]]], dtype=np.complex128)
    got = unfold_file_wedge_polar_matrix(sym, wedge)
    expected = np.einsum("oi,kiab->koab", forward, wedge)
    wrong = np.einsum("oi,kiab->koab", forward.T, wedge)
    np.testing.assert_array_equal(got, expected)
    assert not np.array_equal(got, wrong)


def test_allow_trs_override_is_retired_before_missing_schema_fallback():
    stub = _typed_symmaps_stub(None)
    del stub.qe_symmetry_binding
    with pytest.raises(ValueError, match="retired_SymMaps_allow_trs"):
        SymMaps(stub, allow_trs=True)


def test_missing_trs_verdict_refuses_instead_of_defaulting_true():
    stub = _typed_symmaps_stub(None)
    del stub.trs_holds
    with pytest.raises(ValueError, match="SymMaps_needs_measured_trs"):
        SymMaps(stub)


def test_missing_schema_warning_says_results_can_be_wrong():
    stub = _typed_symmaps_stub(None)
    del stub.qe_symmetry_binding
    with pytest.warns(RuntimeWarning, match="RESULTS WILL BE WRONG"):
        with pytest.raises(ValueError, match=r"1 of 3 full-BZ k-points"):
            SymMaps(stub)
