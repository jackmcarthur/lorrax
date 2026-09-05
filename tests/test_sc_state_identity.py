"""Crossings and gauge rotations must not manufacture SC convergence/motion."""
import numpy as np
import pytest

from gw.sc_state_identity import assign_qp_identity
from gw.sc_iteration import protected_band_convergence


def test_trusted_state_crosses_scissored_sorted_column():
    e0 = np.array([[0., 1., 2., 4.]])
    u0 = np.eye(4)[None].astype(complex)
    order = [0, 2, 3, 1]
    e = np.array([[0., 2., 4., 5.]])
    mask = np.array([True, True, False, False])
    idx, aligned, _, _ = assign_qp_identity(
        u0, e0, u0[:, :, order], e, mask, degeneracy_tol_ev=1e-4)
    assert idx[0, 1] == 3
    assert aligned[0, 1] == 5
    assert np.isnan(aligned[0, 2:]).all()


def test_equal_sorted_spectra_can_hide_nonzero_identity_residual():
    e = np.array([[0., 1., 3.]])
    u = np.eye(3)[None].astype(complex)
    mask = np.ones(3, bool)
    _, aligned, _, _ = assign_qp_identity(
        u, e, u[:, :, [0, 2, 1]], e, mask, degeneracy_tol_ev=1e-4)
    assert protected_band_convergence(e, e, mask, mask, .01).converged
    verdict = protected_band_convergence(aligned, e, mask, mask, .01)
    assert not verdict.converged
    assert verdict.max_abs_ev == 2.


def test_doublet_triplet_exchange_is_blockwise_and_gauge_invariant():
    rng = np.random.default_rng(15)
    u = np.eye(6, dtype=complex)
    e0 = np.array([[0., 1., 1., 2., 2., 2.]])
    order = [0, 3, 4, 5, 1, 2]
    enow = np.array([[0., 1.8, 1.8, 1.8, 2.1, 2.1]])
    reference = u.copy()
    current = u[:, order].copy()
    mask = np.ones(6, bool)
    for _ in range(4):
        for columns in ([1, 2], [3, 4, 5]):
            q, _ = np.linalg.qr(rng.normal(size=(len(columns), len(columns))) +
                                1j*rng.normal(size=(len(columns), len(columns))))
            reference[:, columns] = reference[:, columns] @ q
        for columns in ([1, 2, 3], [4, 5]):
            q, _ = np.linalg.qr(rng.normal(size=(len(columns), len(columns))) +
                                1j*rng.normal(size=(len(columns), len(columns))))
            current[:, columns] = current[:, columns] @ q
        idx, aligned, blocks, weights = assign_qp_identity(
            reference[None], e0, current[None], enow, mask,
            degeneracy_tol_ev=1e-4)
        assert set(idx[0, 1:3]) == {4, 5}
        assert set(idx[0, 3:]) == {1, 2, 3}
        np.testing.assert_allclose(aligned, [[0., 2.1, 2.1, 1.8, 1.8, 1.8]])
        np.testing.assert_allclose(weights, 1., atol=1e-14)
        np.testing.assert_array_equal(blocks, [[0, 1, 1, 3, 3, 3]])


def test_reference_multiplet_cut_refuses():
    with pytest.raises(ValueError, match='cuts multiplet'):
        assign_qp_identity(np.eye(3)[None], [[0., 1., 1.]], np.eye(3)[None],
                           [[0., 1., 1.]], [True, True, False],
                           degeneracy_tol_ev=1e-4)


def test_three_way_partition_diagonal_and_criterion_agree():
    import jax.numpy as jnp
    from gw.band_partition import apply_band_partition
    # Protected outside range, unprotected in range, and scissored.
    prot = np.array([True, False, False])
    inr = np.array([False, True, False])
    raw = np.array([[[1., .2, .3], [.2, 2., .4], [.3, .4, 3.]]])
    h = np.asarray(apply_band_partition(
        jnp.asarray(raw), protected_mask=jnp.asarray(prot),
        in_range_mask=jnp.asarray(inr), scissor_E_qp_kn=jnp.array([[8., 9., 10.]])))
    np.testing.assert_array_equal(h, np.diag([1., 2., 10.])[None])
    for band in range(3):
        new = np.zeros((1, 3)); new[0, band] = 1.
        verdict = protected_band_convergence(new, np.zeros_like(new), prot, inr, .1)
        assert verdict.converged == (band == 2)


def test_map_readout_freezes_output_reference_and_does_not_change_carry(monkeypatch):
    from types import SimpleNamespace
    from gw import sc_iteration as sc
    from common import collectives
    from gw.band_partition import BandPartition
    u = np.eye(3)[None].astype(complex)
    e = np.array([[0., 1., 3.]])
    mask = np.ones(3, bool)
    partition = BandPartition(mask, mask)
    inputs = SimpleNamespace(mesh_xy=None, config=SimpleNamespace(
        sc=SimpleNamespace(exact_degeneracy_tol_ev=1e-4)))
    logs = []
    monkeypatch.setattr(collectives, 'gather_to_host', np.asarray)
    monkeypatch.setattr(sc, '_record_sc', lambda inputs, text: logs.append(text))
    output_rotation = [u]
    monkeypatch.setattr(sc, '_kshard_eigh_kernels',
                        lambda *a: (lambda h: (e, output_rotation[0]), None))
    outputs = sc.SCOutputs(None, u, None, None, None)
    state = sc.SCState(np.diag(e[0])[None], 1, partition, outputs=outputs)
    history = {}
    v0, _ = sc._sc_identity_for_call(inputs, state, e, e, history, cutoff_ev=.01)
    assert v0.converged
    output_rotation[0] = u[:, :, [0, 2, 1]]
    verdict, updated = sc._sc_identity_for_call(
        inputs, state, e, e, history, cutoff_ev=.01)
    assert not verdict.converged and verdict.max_abs_ev == 2.
    assert verdict.rms_all_ev == 0.  # explicitly the legacy sorted diagnostic
    assert updated.H_qp_dft is state.H_qp_dft
    np.testing.assert_array_equal(history['u'], u)
    np.testing.assert_array_equal(updated.outputs.identity['motion_ev'], [[0., 2., -2.]])
    assert 'sorted-index value 0.000000000e+00' in logs[-1]
