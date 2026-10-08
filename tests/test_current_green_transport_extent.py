"""Protected OWN3 transport bands never become physical Green bands."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('transport_bands', [120, 128])
def test_augmented_current_binds_green_prefix_and_retains_fit_faces(monkeypatch, transport_bands):
    from gw import gw_init, isdf_fitting
    from gw.wavefunction_bundle import BandSlices

    slices = BandSlices.from_band_edges(0, 0, 36, 64, 120, b4_logical=120)
    rng = np.random.default_rng(51923)
    nmu = np.zeros((2, transport_bands, 4, 8), complex)
    nmu[:, :120] = rng.normal(size=(2, 120, 4, 8)) + 1j*rng.normal(size=(2, 120, 4, 8))
    mun = nmu.conj().transpose(0, 2, 3, 1).copy()
    before = (nmu.copy(), mun.copy())
    plan = object()
    source = object()
    current = dict(parent_faces=(nmu, mun), parent_psi=source)
    called = []

    def bind_green(*args, **kwargs):
        faces = kwargs['faces']
        assert args[6] is slices and kwargs['plan'] is plan
        assert faces[0].shape == (2, 120, 4, 8)
        assert faces[1].shape == (2, 4, 8, 120)
        np.testing.assert_array_equal(faces[0], before[0][:, :120])
        np.testing.assert_array_equal(faces[1], before[1][..., :120])
        called.append('green')
        return dict(green_parent=SimpleNamespace(plan=plan, psi_nmu=faces[0], psi_mun=faces[1]))

    class ReachedFit(Exception):
        pass

    def fit(**kwargs):
        assert called == ['green']
        assert kwargs['psi_nmu_parent'] is nmu and kwargs['psi_mun_parent'] is mun
        assert kwargs['parent_psi'] is source and kwargs['current_augmentation'] is current
        assert kwargs['band_range_left'] == (0, 64) and kwargs['band_range_right'] == (0, 120)
        np.testing.assert_array_equal(nmu, before[0])
        np.testing.assert_array_equal(mun, before[1])
        raise ReachedFit

    monkeypatch.setattr(gw_init, '_transverse_wfn_data', bind_green)
    monkeypatch.setattr(isdf_fitting, 'fit_zeta_to_h5', fit)
    chunks = dict(band_chunk=16, centroid_k_chunk=1, k_unfold_plan=plan, mubatch=object())
    cfg = SimpleNamespace(paths=SimpleNamespace(centroids_file_current='current'),
                          write_restart_tensors=False,
                          backend=SimpleNamespace(distrib_la_batched_route='lu'))
    with pytest.raises(ReachedFit):
        gw_init._fit_transverse_zeta_channels(
            None, np.zeros((8, 3)), chunks, SimpleNamespace(current_basis_rows=None), None,
            (False,)*3, None, True, {mu: str(mu) for mu in (1, 2, 3)}, 320.,
            (0, 64), (0, 120), slices, cfg, None, lambda *args: None,
            SimpleNamespace(current_lift=None), object(), object(), object(),
            current_augmentation=current)


def test_reused_unaugmented_current_keeps_existing_face_load(monkeypatch):
    from gw import gw_init
    result = object()

    def bind_green(*args, **kwargs):
        assert kwargs['faces'] is None and kwargs['plan'] is None
        return result

    monkeypatch.setattr(gw_init, '_transverse_wfn_data', bind_green)
    returned = gw_init._fit_transverse_zeta_channels(
        None, None, None, None, None, (True,)*3, None, True,
        {mu: str(mu) for mu in (1, 2, 3)}, 320., (0, 64), (0, 120),
        SimpleNamespace(nb_full=120),
        SimpleNamespace(paths=SimpleNamespace(centroids_file_current='current')),
        None, lambda *args: None, object(), object(), object(),
        SimpleNamespace(loader_band_chunk=16, loader_k_chunk=1))
    assert returned == (result, None)
