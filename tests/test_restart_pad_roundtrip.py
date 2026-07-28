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
    # Scalar restart: no transverse channel on disk → None fields.
    assert rs_log.psi_rmu_Y_transverse is None
    assert rs_log.psi_rmuT_X_transverse is None
    assert rs_log.n_rmu_transverse_disk is None


def test_restart_transverse_psi_roundtrip(tmp_path, monkeypatch):
    """Bispinor per-channel ψ round-trip: the TRANSVERSE dataset has its
    own (smaller) logical μ extent, its own disk clip, and its own
    re-pad on read — independent of the charge ``n_rmu_logical``."""
    import h5py
    from file_io import (
        load_restart_state_from_h5,
        write_restart_state_to_h5,
    )

    N_LOG_T = 5          # transverse centroid count (≠ N_LOG, non-dividing)
    NS_T = 4             # bispinor-lifted spinor components

    mesh = _mesh_1x1()
    rng = np.random.default_rng(20260727)
    V, G0, psi, enk, _ = _padded_state(rng)

    def _c(*shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(np.complex128)

    psi_T = _c(NK, NB, NS_T, N_LOG_T + MU_PAD)
    psi_T[..., N_LOG_T:] = 0.0

    path = str(tmp_path / "isdf_tensors_bispinor_test.h5")
    monkeypatch.delenv("LORRAX_EXTRA_MU_PAD", raising=False)
    write_restart_state_to_h5(
        path, n_rmu_logical=N_LOG,
        V_qmunu=jnp.asarray(V), G0_mu_nu=jnp.asarray(G0),
        enk_full=jnp.asarray(enk),
        mesh=mesh, mode="w", kgrid=(NQ, 1, 1),
    )
    # The gw_init call pattern: charge ψ + transverse ψ in one append.
    write_restart_state_to_h5(
        path, n_rmu_logical=N_LOG,
        psi_full_y=jnp.asarray(psi),
        psi_full_y_transverse=jnp.asarray(psi_T),
        n_rmu_transverse_logical=N_LOG_T,
        mesh=mesh, mode="a",
    )

    # Disk contract: transverse μ extent is its OWN logical count.
    with h5py.File(path, "r") as f:
        assert f["psi_full_y"].shape == (NK, NB, NS, N_LOG)
        assert f["psi_full_y_transverse"].shape == (NK, NB, NS_T, N_LOG_T)
        np.testing.assert_array_equal(
            f["psi_full_y_transverse"][:], psi_T[..., :N_LOG_T])
        assert int(np.asarray(f["n_rmu_transverse_logical"])[()]) == N_LOG_T

    # Forced pad on read: transverse ψ re-pads at its own extent.
    monkeypatch.setenv("LORRAX_EXTRA_MU_PAD", str(MU_PAD))
    rs = load_restart_state_from_h5(path, mesh)
    assert rs.n_rmu_transverse_disk == N_LOG_T
    np.testing.assert_array_equal(
        np.asarray(rs.psi_rmu_Y_transverse), psi_T)
    np.testing.assert_array_equal(
        np.asarray(rs.psi_rmuT_X_transverse),
        np.conj(psi_T).transpose(0, 3, 1, 2))

    # No pad: logical extent untouched.
    monkeypatch.delenv("LORRAX_EXTRA_MU_PAD", raising=False)
    rs_log = load_restart_state_from_h5(path, mesh)
    assert rs_log.psi_rmu_Y_transverse.shape == (NK, NB, NS_T, N_LOG_T)
    np.testing.assert_array_equal(
        np.asarray(rs_log.psi_rmu_Y_transverse), psi_T[..., :N_LOG_T])

    # Contract guard: the transverse dataset REQUIRES its logical count.
    with pytest.raises(ValueError, match="n_rmu_transverse_logical"):
        write_restart_state_to_h5(
            path, n_rmu_logical=N_LOG,
            psi_full_y_transverse=jnp.asarray(psi_T),
            mesh=mesh, mode="a",
        )
