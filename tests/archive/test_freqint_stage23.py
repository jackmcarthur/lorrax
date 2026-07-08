"""Parity checks for staged freqint engine (stage 2/3)."""

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from common.meta import Meta
from gw_isdf.get_windows import WindowInfo, WindowPair
from gw_isdf.wavefunction_bundle import BandSlices, WavefunctionBundle
from gw_isdf.w_isdf_dynamic import frequency_integration_from_bundle as legacy_frequency_integration
from gw_isdf.freqint import chi_from_bundle


jax.config.update("jax_enable_x64", True)


def _toy_bundle():
    slices = BandSlices.from_band_edges(0, 0, 2, 2, 4)
    # (nk=1, nb=4, nspinor=1, nmu=1)
    psi_y = jnp.asarray(np.array([[[[0.9]], [[1.1]], [[1.3]], [[0.8]]]], dtype=np.float64))
    psi_x = jnp.asarray(np.transpose(np.array([[[[0.9]], [[1.1]], [[1.3]], [[0.8]]]], dtype=np.float64), (0, 2, 3, 1)))
    enk = jnp.asarray(np.array([[-0.6, -0.3, 0.2, 0.8]], dtype=np.float64))
    occ = jnp.asarray(np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float64))
    wf = WavefunctionBundle(psi_y=psi_y, psi_x=psi_x, enk=enk, occ=occ, slices=slices)

    val_win = WindowInfo(-0.6, -0.3, index=0, count=1)
    cond_win = WindowInfo(0.2, 0.8, index=0, count=1)
    wins = [WindowPair(val_win, cond_win, epsq=1e-4)]

    meta = Meta(
        rank=0,
        n_proc=1,
        b_id_0=0,
        b_id_1=0,
        b_id_2=0,
        b_id_3=0,
        b_id_4=0,
        fft_grid=(1, 1, 1),
        n_rtot=1,
        n_rmu=1,
        npol=1,
        nfreq=1,
        nspin=1,
        nspinor=1,
        nspinor_wfnfile=1,
        nkx=1,
        nky=1,
        nkz=1,
        nk_tot=1,
    )
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    return wf, wins, meta, mesh


def test_freqint_policies_match_legacy_scalar():
    wf, wins, meta, mesh = _toy_bundle()
    omegas = np.array([-1.4, -0.5, 0.0, 0.5, 1.6], dtype=np.float64)

    for policy in ("per_omega", "shared_conservative"):
        chi_new = chi_from_bundle(wf, wins, omegas, meta, mesh, policy=policy)
        tol = 1e-8 if policy == "per_omega" else 1e-6
        for i, om in enumerate(omegas):
            chi_old = legacy_frequency_integration(wf, wins, float(om), meta, mesh)
            rel = np.linalg.norm(np.asarray(chi_new[i] - chi_old)) / max(1e-14, np.linalg.norm(np.asarray(chi_old)))
            assert rel < tol, f"policy={policy} omega={om} rel={rel}"


def test_freqint_scalar_shape_matches_legacy():
    wf, wins, meta, mesh = _toy_bundle()
    omega = 0.7
    chi_new = chi_from_bundle(wf, wins, omega, meta, mesh)
    chi_old = legacy_frequency_integration(wf, wins, omega, meta, mesh)
    assert chi_new.shape == chi_old.shape
    rel = np.linalg.norm(np.asarray(chi_new - chi_old)) / max(1e-14, np.linalg.norm(np.asarray(chi_old)))
    assert rel < 1e-8
