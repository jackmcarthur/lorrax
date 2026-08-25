"""Focused algebra/source gates for the immutable centroid-WFN receipt."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from file_io.isdf_header import (
    CENTROID_TABLE_FINGERPRINT_SCHEME,
    WavefunctionBasisReceipt,
    centroid_table_md5,
)


CENTROIDS = np.asarray([
    [0, 0, 0],
    [1, 2, 3],
    [3, 1, 0],
], dtype=np.int32)


def _wfn(*, shift: float = 0.0):
    return SimpleNamespace(
        nbands=6,
        nelec=2,
        nspinor=2,
        energies=np.asarray([[[0.1 + shift, 0.2, 0.3, 0.4, 0.5, 0.6]]]),
        kpoints=np.asarray([[0.0, 0.0, 0.0]]),
    )


def _receipt(*, role="transverse", pad=4, centroids=CENTROIDS,
             grid=(4, 4, 4), band_interval=(1, 5), wfn=None):
    table = np.asarray(centroids, dtype=np.int32)
    return WavefunctionBasisReceipt.from_source(
        wfn=_wfn() if wfn is None else wfn,
        role=role,
        band_interval=band_interval,
        fft_grid=grid,
        centroid_fft_idx=table,
        n_rmu_logical=table.shape[0],
        n_rmu_padded=pad,
    )


def test_receipt_is_immutable_and_reuses_the_restart_centroid_digest():
    receipt = _receipt()
    assert receipt.centroid_fingerprint_scheme == \
        CENTROID_TABLE_FINGERPRINT_SCHEME
    assert receipt.centroid_table_md5 == centroid_table_md5(CENTROIDS)
    with pytest.raises(FrozenInstanceError):
        receipt.role = "charge"


def test_fresh_and_restart_face_builders_propagate_one_receipt_object():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from gw.wavefunction_bundle import (
        BandSlices,
        build_wavefunctions_face,
        wavefunctions_face_from_restart,
    )

    receipt = _receipt()
    slices = BandSlices.from_band_edges(1, 1, 2, 3, 5)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    psi_y = jnp.zeros((1, 4, 2, 4), dtype=jnp.complex128)
    psi_t_x = jnp.zeros((1, 4, 4, 2), dtype=jnp.complex128)
    enk = jnp.zeros((1, 4), dtype=jnp.float64)

    fresh = build_wavefunctions_face(
        psi_y, psi_t_x, enk_full=enk, slices=slices, mesh_xy=mesh,
        basis_receipt=receipt)
    restart = wavefunctions_face_from_restart(
        fresh.psi_nmu, fresh.psi_mun, enk_full=enk, slices=slices,
        mesh_xy=mesh, basis_receipt=receipt)
    assert fresh.basis_receipt is receipt
    assert restart.basis_receipt is receipt


def test_physical_source_identity_is_layout_and_device_count_independent():
    p4 = _receipt(pad=4)
    p16 = _receipt(pad=16)
    assert p4.source_identity == p16.source_identity
    p4.assert_same_source(p16, where="cross-device receipt")
    assert p4 != p16
    with pytest.raises(ValueError, match="padded centroid extents"):
        p4.assert_same_carrier(p16, where="same-runtime carrier")


def test_charge_and_transverse_are_distinct_even_on_the_same_points():
    charge = _receipt(role="charge")
    transverse = _receipt(role="transverse")
    with pytest.raises(ValueError, match="role"):
        charge.assert_same_source(transverse, where="Lorentz channel")


@pytest.mark.parametrize(
    "changed,field",
    [
        (_receipt(wfn=_wfn(shift=1.0e-3)), "wfn_fingerprint"),
        (_receipt(band_interval=(0, 4)), "band_interval"),
        (_receipt(grid=(5, 4, 4)), "fft_grid"),
        (_receipt(centroids=CENTROIDS[::-1]), "centroid_table_md5"),
        (_receipt(centroids=CENTROIDS[:2], pad=4), "n_rmu_logical"),
    ],
)
def test_every_physical_source_field_has_a_red_twin(changed, field):
    with pytest.raises(ValueError, match=field):
        _receipt().assert_same_source(changed, where="stale source")


def test_wavefunctions_authenticates_receipt_against_live_mu_carrier():
    import jax.numpy as jnp
    from gw.wavefunction_bundle import BandSlices, Wavefunctions

    slices = BandSlices.from_band_edges(1, 1, 2, 3, 5)
    psi_nmu = jnp.zeros((1, 4, 2, 4), dtype=jnp.complex128)
    psi_mun = jnp.zeros((1, 2, 4, 4), dtype=jnp.complex128)
    wfns = Wavefunctions(
        enk=jnp.zeros((1, 4)), occ=jnp.zeros((1, 4)), slices=slices,
        psi_nmu=psi_nmu, psi_mun=psi_mun, layout="face",
        basis_receipt=_receipt(role="charge"),
    )
    assert wfns.basis_receipt.role == "charge"

    with pytest.raises(ValueError, match="centroid extent"):
        Wavefunctions(
            enk=jnp.zeros((1, 4)), occ=jnp.zeros((1, 4)), slices=slices,
            psi_nmu=psi_nmu[..., :3], psi_mun=psi_mun[:, :, :3],
            layout="face", basis_receipt=_receipt(role="charge"),
        )
