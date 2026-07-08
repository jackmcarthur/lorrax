"""Restart-file μ-pad roundtrip: disk stores LOGICAL, memory re-pads.

Disk contract (SHARDING_RULES §2 / PADDING_AUDIT item 1): in-memory
restart tensors carry the P-dependent padded μ extent
(``Meta.n_rmu_padded``, zero pad rows by construction), but the restart
file must store the LOGICAL extent so a restart written at one device
count is readable and bit-correct at any other.  Before the fix the
writers persisted the padded extent verbatim — the ROOT_CAUSE.md defect
class one hop downstream (bse_io recovered "n_rmu" from the dataset
shape, so pad rows masqueraded as physical centroids).

This test forces a μ pad at fixed P=1 (the same knob mechanism as
``test_mu_pad_invariance``), writes the restart state, checks every
on-disk μ extent is logical, re-reads through the production loader,
and requires the roundtrip to be bit-identical to the original padded
arrays.
"""

import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh


N_LOG = 6      # logical μ extent
MU_PAD = 2     # forced extra pad rows (LORRAX_EXTRA_MU_PAD on read)
N_PAD = N_LOG + MU_PAD
NQ = 3
NK, NB, NS = 2, 4, 1


def _mesh_1x1():
    dev = np.asarray(jax.devices()[:1]).reshape(1, 1)
    return Mesh(dev, axis_names=("x", "y"))


def _padded_state(rng):
    """Synthetic restart tensors at the PADDED μ extent, zero pad rows."""
    def _c(*shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(np.complex128)

    V = _c(NQ, N_PAD, N_PAD)
    V[:, N_LOG:, :] = 0.0
    V[:, :, N_LOG:] = 0.0
    G0 = _c(N_PAD)
    G0[N_LOG:] = 0.0
    psi = _c(NK, NB, NS, N_PAD)
    psi[..., N_LOG:] = 0.0
    enk = rng.standard_normal((NK, NB)).astype(np.float64)
    W0 = _c(NQ, N_PAD, N_PAD)
    W0[:, N_LOG:, :] = 0.0
    W0[:, :, N_LOG:] = 0.0
    return V, G0, psi, enk, W0


def test_restart_mu_pad_roundtrip(tmp_path, monkeypatch):
    import h5py
    from file_io import (
        load_restart_state_from_h5,
        write_restart_state_to_h5,
        write_w0_qmunu_to_h5,
    )

    mesh = _mesh_1x1()
    rng = np.random.default_rng(20260708)
    V, G0, psi, enk, W0 = _padded_state(rng)
    path = str(tmp_path / "isdf_tensors_test.h5")

    # Writers receive PADDED in-memory arrays + the logical extent
    # (exactly the gw_init / gw_jax call pattern).
    monkeypatch.delenv("LORRAX_EXTRA_MU_PAD", raising=False)
    write_restart_state_to_h5(
        path, n_rmu_logical=N_LOG,
        V_qmunu=jnp.asarray(V), G0_mu_nu=jnp.asarray(G0),
        enk_full=jnp.asarray(enk), init_W0=True,
        mesh=mesh, mode="w", kgrid=(NQ, 1, 1),
    )
    write_restart_state_to_h5(
        path, n_rmu_logical=N_LOG,
        psi_full_y=jnp.asarray(psi), mesh=mesh, mode="a",
    )
    write_w0_qmunu_to_h5(path, jnp.asarray(W0), n_rmu_logical=N_LOG,
                         mesh=mesh)

    # ── Disk contract: every μ extent on disk is LOGICAL ──────────────
    with h5py.File(path, "r") as f:
        assert f["V_qmunu"].shape == (NQ, N_LOG, N_LOG)
        assert f["G0_mu_nu"].shape == (N_LOG,)
        assert f["psi_full_y"].shape == (NK, NB, NS, N_LOG)
        assert f["W0_qmunu"].shape == (NQ, N_LOG, N_LOG)
        assert bool(f["W0_qmunu"].attrs["W0_ready"])
        # Logical block reaches disk bit-exact.
        np.testing.assert_array_equal(
            f["V_qmunu"][:], V[:, :N_LOG, :N_LOG])
        np.testing.assert_array_equal(
            f["W0_qmunu"][:], W0[:, :N_LOG, :N_LOG])
        np.testing.assert_array_equal(f["G0_mu_nu"][:], G0[:N_LOG])
        np.testing.assert_array_equal(
            f["psi_full_y"][:], psi[..., :N_LOG])
        np.testing.assert_array_equal(f["enk_full"][:], enk)

    # ── Re-read with a FORCED pad (simulates a bigger device count):
    # loader must re-pad every μ axis with exact zeros ─────────────────
    monkeypatch.setenv("LORRAX_EXTRA_MU_PAD", str(MU_PAD))
    rs = load_restart_state_from_h5(path, mesh)

    np.testing.assert_array_equal(np.asarray(rs.V_qmunu), V)
    np.testing.assert_array_equal(np.asarray(rs.G0_mu_nu), G0)
    np.testing.assert_array_equal(np.asarray(rs.psi_rmu_Y), psi)
    np.testing.assert_array_equal(np.asarray(rs.enk_full), enk)
    # psi_rmuT_X is conj(ψ) with (nb, s, μ) → (μ, nb, s) transpose.
    np.testing.assert_array_equal(
        np.asarray(rs.psi_rmuT_X), np.conj(psi).transpose(0, 3, 1, 2))

    # ── Re-read with NO pad (simulates P where the extent divides):
    # loader hands back the logical extent untouched ───────────────────
    monkeypatch.delenv("LORRAX_EXTRA_MU_PAD", raising=False)
    rs_log = load_restart_state_from_h5(path, mesh)
    assert rs_log.V_qmunu.shape == (NQ, N_LOG, N_LOG)
    np.testing.assert_array_equal(
        np.asarray(rs_log.V_qmunu), V[:, :N_LOG, :N_LOG])
    np.testing.assert_array_equal(np.asarray(rs_log.G0_mu_nu), G0[:N_LOG])
