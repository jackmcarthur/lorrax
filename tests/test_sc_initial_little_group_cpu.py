"""SC seed projection respects the computational, not source-file, group."""
from types import SimpleNamespace as NS

import numpy as np
import pytest


def _sym(active, *, translation=None):
    matrices = np.stack((np.eye(3, dtype=int), -np.eye(3, dtype=int)))
    shifts = np.zeros((2, 3))
    if translation is not None:
        shifts[0] = translation
    active = np.asarray(active, np.int32)
    def operation_rows(rows):
        row = np.asarray(rows)
        return matrices[row % 2], shifts[row % 2], row >= 2
    return NS(active_symmetry_rows=active, operation_rows=operation_rows,
              sym_matrices=matrices, translations=shifts,
              irr_idx_k=np.arange(64, dtype=np.int32),
              sym_idx_k=np.zeros(64, np.int32))


def test_full64_identity_view_does_not_read_original_file10_or_project(monkeypatch):
    from gw.sc_iteration import _project_on_little_groups
    import file_io.qp_wfn as qp
    rng = np.random.default_rng(60053348)
    h = rng.normal(size=(64, 3, 3))+1j*rng.normal(size=(64, 3, 3))
    h = h+h.conj().swapaxes(-1, -2)
    original = h.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError('computational full rows must not become raw file parents')
    monkeypatch.setattr(qp, 'little_group_band_representations', forbidden)
    inputs = NS(sym=_sym([0]), wfn=NS(nkpts=10, load=forbidden), mesh_xy=None,
                band_slices=NS(sigma_range=(0, 3)))
    for ks in (None, NS(is_identity=True)):
        result, moved = _project_on_little_groups(inputs, h, ks)
        np.testing.assert_array_equal(result, original)
        assert result is not h and moved == 0.
    np.testing.assert_array_equal(h, original)


@pytest.mark.parametrize('active,translation', [([0, 1], None), ([0, 2], None),
                                               ([2], None), ([0], [.5, 0, 0])])
def test_nontrivial_antiunitary_and_translated_groups_keep_existing_projection(monkeypatch, active, translation):
    from gw.sc_iteration import _project_on_little_groups
    import file_io.qp_wfn as qp
    calls = []
    unit = np.eye(2, dtype=complex)
    d = np.diag([1., -1.]).astype(complex)
    group = [(0, unit, False), (1, d, bool(max(active) >= 2))]
    def representations(wfn, **kwargs):
        calls.append(kwargs['parent'])
        return group
    monkeypatch.setattr(qp, 'little_group_band_representations', representations)
    sym = _sym(active, translation=translation)
    sym.irr_idx_k = np.array([0, 1, 0, 1])
    sym.sym_idx_k = np.zeros(4, np.int32)
    h = np.array([[[2., 1+2j], [1-2j, 3.]], [[4., 3-1j], [3+1j, 5.]]])
    inputs = NS(sym=sym, wfn=NS(nkpts=2), mesh_xy=None,
                band_slices=NS(sigma_range=(0, 2)))
    ks = NS(is_identity=False, rows=np.array([0, 1]), labels=np.array([0, 1]))
    result, moved = _project_on_little_groups(inputs, h, ks)
    expected = []
    for a in h:
        terms = [D.conj().T@a@D for _, D, _ in group]
        if group[1][2]:
            terms[1] = terms[1].conj()
        expected.append(sum(terms)/len(terms))
    np.testing.assert_array_equal(result, expected)
    assert moved == np.max(abs(result-h)) > 0
    assert calls == [0, 1]
