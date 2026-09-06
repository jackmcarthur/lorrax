"""Nonclosed centroids use identity plans over already loader-unfolded states."""
from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh


def _source_symmetry():
    from symmetry_maps import SymMaps
    header = SimpleNamespace(
        trs_holds=True, ntran=2, nkpts=2,
        sym_matrices=np.array([np.eye(3), -np.eye(3)], dtype=np.int32),
        translations=np.zeros((2, 3)), kgrid=np.array([3, 1, 1]),
        shift=np.zeros(3), kpoints=np.array([[0., 0., 0.], [1/3, 0., 0.]]),
        avec=np.eye(3), atom_types=np.array([1]), atom_crys=np.zeros((1, 3)))
    return SymMaps(header)


def test_trivial_view_preserves_loader_group_and_admits_nonclosed_centroids():
    """Restricting the computational group changes neither loader actions nor child states."""
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    source = _source_symmetry()
    original = source.irr_idx_k.copy(), source.sym_idx_k.copy()
    view = source.trivial_view()
    assert source.parent_k_domain == "ibz" and view.parent_k_domain == "full_bz"
    assert source.nk_red == 2 and view.nk_red == view.nk_tot == 3
    assert source.trs_allowed and not view.trs_allowed
    np.testing.assert_array_equal(source.irr_idx_k, original[0])
    np.testing.assert_array_equal(source.sym_idx_k, original[1])
    for rows in (view.irr_idx_k, view.kirr_fullids, view.irr_idx_q, view.q_irr_full_idx):
        np.testing.assert_array_equal(rows, [0, 1, 2])
    np.testing.assert_array_equal(view.sym_idx_k, [0, 0, 0])
    np.testing.assert_array_equal(view.sym_idx_q, [0, 0, 0])
    np.testing.assert_array_equal(view.active_symmetry_rows, [0])
    np.testing.assert_array_equal(view.spinor_action([0], nspinor=4), np.eye(4)[None])
    np.testing.assert_array_equal(view.lorentz_action([0]), np.eye(4)[None])
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))
    plan = build_centroid_k_unfold_plan(view, np.array([[1, 0, 0]]),
                                      (3, 1, 1), mesh, nspinor=4)
    assert plan.n_parent == plan.n_full == 3
    np.testing.assert_array_equal(plan.parent_full_rows, [0, 1, 2])
    np.testing.assert_array_equal(plan.sym_perm[0], [0])
    np.testing.assert_array_equal(plan.L_table[0], [[0, 0, 0]])

    from common.wfn_transforms import get_enk_bandrange
    wfn = SimpleNamespace(symmetry=lambda: source, nspinor=1, efermi=0.,
        energies=np.array([[[-2., -1., 1., 2.], [-3., -.5, 2., 3.]]]))
    energies, _ = get_enk_bandrange(wfn, view, (0, 4), (1, 3))
    np.testing.assert_array_equal(energies,
        [[-2., -1., 1., 2.], [-3., -.5, 2., 3.], [-3., -.5, 2., 3.]])


def test_unreduced_file_output_keeps_authenticated_file_rows():
    """Full-k computational parents do not widen files defined on the original WFN wedge."""
    from gw.gw_output import sigma_table_to_file_wedge, SIGMA_KSET_FULL_BZ
    source = _source_symmetry()
    view = source.trivial_view()
    values = np.array([[10., 20.], [30., 40.], [50., 60.]])
    result = sigma_table_to_file_wedge(view, values,
        source_kset=SIGMA_KSET_FULL_BZ, file_sym=source)
    np.testing.assert_array_equal(result, [[10., 20.], [30., 40.]])
