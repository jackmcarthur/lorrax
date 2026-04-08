"""Canonical restart-state I/O for GW/BSE workflows.

This module reads/writes HDF5 restart files in the v2 format used by gw_jax.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import h5py
from jax.sharding import NamedSharding, PartitionSpec as P


def write_restart_state_to_h5(
    filename,
    V_qmunu,
    psi_full_y,
    enk_full=None,
    S_qmunu=None,
    V0_noG0_munu=None,
    G0_mu_nu=None,
    W0_qmunu=None,
    init_W0: bool = False,
):
    """Write canonical restart state with a single full-band wavefunction dataset."""
    from jax.experimental import multihost_utils as _mh

    def _to_host(a):
        if isinstance(a, (jax.Array,)):
            try:
                return _mh.process_allgather(a, tiled=True)
            except Exception:
                return jax.device_get(a)
        return a

    V_qmunu_h = _to_host(V_qmunu)
    psi_full_y_h = _to_host(psi_full_y)
    enk_full_h = _to_host(enk_full) if enk_full is not None else None
    S_qmunu_h = _to_host(S_qmunu) if S_qmunu is not None else None
    V0_noG0_h = _to_host(V0_noG0_munu) if V0_noG0_munu is not None else None
    G0_mu_nu_h = _to_host(G0_mu_nu) if G0_mu_nu is not None else None
    W0_qmunu_h = _to_host(W0_qmunu) if W0_qmunu is not None else None

    if jax.process_index() == 0:
        with h5py.File(filename, "w") as f:
            f.attrs["restart_format_version"] = 2
            f.create_dataset("V_qmunu", data=np.asarray(V_qmunu_h))
            if S_qmunu_h is not None:
                f.create_dataset("S_qmunu", data=np.asarray(S_qmunu_h))
            if V0_noG0_h is not None:
                f.create_dataset("V0_noG0_munu", data=np.asarray(V0_noG0_h))
            if G0_mu_nu_h is not None:
                f.create_dataset("G0_mu_nu", data=np.asarray(G0_mu_nu_h))
            if W0_qmunu_h is not None:
                dset = f.create_dataset("W0_qmunu", data=np.asarray(W0_qmunu_h))
                dset.attrs["W0_ready"] = True
            elif init_W0:
                v_shape = np.asarray(V_qmunu_h).shape
                v_dtype = np.asarray(V_qmunu_h).dtype
                fill = np.zeros((), dtype=v_dtype)
                dset = f.create_dataset("W0_qmunu", shape=v_shape, dtype=v_dtype, fillvalue=fill)
                dset.attrs["W0_ready"] = False
            f.create_dataset("psi_full_y", data=np.asarray(psi_full_y_h))
            if enk_full_h is not None:
                f.create_dataset("enk_full", data=np.asarray(enk_full_h))


def write_w0_qmunu_to_h5(filename, W0_qmunu):
    """Overwrite or append the W0_qmunu dataset in an existing restart file."""
    from jax.experimental import multihost_utils as _mh

    def _to_host(a):
        if isinstance(a, (jax.Array,)):
            try:
                return _mh.process_allgather(a, tiled=True)
            except Exception:
                return jax.device_get(a)
        return a

    W0_qmunu_h = _to_host(W0_qmunu)
    if jax.process_index() == 0:
        with h5py.File(filename, "a") as f:
            if "W0_qmunu" in f:
                del f["W0_qmunu"]
            dset = f.create_dataset("W0_qmunu", data=np.asarray(W0_qmunu_h))
            dset.attrs["W0_ready"] = True


def read_restart_state_from_h5(filename):
    """Read canonical restart state from HDF5 (restart format v2)."""
    with h5py.File(filename, "r") as f:
        if "psi_full_y" not in f:
            raise ValueError(
                f"Restart file {filename} is missing canonical psi_full_y dataset. "
                "Regenerate restart tensors with current gw_jax."
            )

        V_qmunu = jnp.asarray(f["V_qmunu"][:])
        S_qmunu = jnp.asarray(f["S_qmunu"][:]) if "S_qmunu" in f else None
        V0_noG0_munu = jnp.asarray(f["V0_noG0_munu"][:]) if "V0_noG0_munu" in f else None
        G0_mu_nu = jnp.asarray(f["G0_mu_nu"][:]) if "G0_mu_nu" in f else None
        psi_full_y = jnp.asarray(f["psi_full_y"][:])
        enk_full = jnp.asarray(f["enk_full"][:]) if "enk_full" in f else None

    return V_qmunu, S_qmunu, psi_full_y, enk_full, V0_noG0_munu, G0_mu_nu


def load_restart_state_from_h5(filename, mesh_xy, band_slices=None):
    """Load canonical restart arrays with explicit X/Y shardings for full wavefunctions."""
    del band_slices  # retained for call-site compatibility
    V_qmunu, S_qmunu, psi_full_y_raw, enk_full, V0_noG0_munu, G0_mu_nu = read_restart_state_from_h5(filename)

    x6y7_8 = NamedSharding(mesh_xy, P(None, None, None, None, None, None, "x", "y"))
    x3y4_5 = NamedSharding(mesh_xy, P(None, None, None, "x", "y"))
    y3_4 = NamedSharding(mesh_xy, P(None, None, None, "y"))
    x2_4 = NamedSharding(mesh_xy, P(None, None, "x", None))
    replicated_2 = NamedSharding(mesh_xy, P(None, None))

    V_qmunu = jax.lax.with_sharding_constraint(V_qmunu, x6y7_8)
    if S_qmunu is not None:
        S_qmunu = jax.lax.with_sharding_constraint(S_qmunu, x3y4_5)
    if V0_noG0_munu is not None:
        V0_noG0_munu = jax.lax.with_sharding_constraint(V0_noG0_munu, NamedSharding(mesh_xy, P("x", "y")))
    if G0_mu_nu is not None:
        # G0 should be (n_rmu,) for head corrections. If stored as 2D
        # (e.g. (nqz, n_rmu) from an old code version), extract q=0 row.
        if G0_mu_nu.ndim > 1:
            G0_mu_nu = G0_mu_nu[0]
        G0_mu_nu = jax.lax.with_sharding_constraint(G0_mu_nu, NamedSharding(mesh_xy, P("y")))

    psi_full_y = jax.lax.with_sharding_constraint(psi_full_y_raw, y3_4)
    psi_full_x = jax.lax.with_sharding_constraint(psi_full_y.transpose(0, 2, 3, 1), x2_4)
    if enk_full is not None:
        enk_full = jax.lax.with_sharding_constraint(enk_full, replicated_2)

    return V_qmunu, S_qmunu, psi_full_x, psi_full_y, enk_full, V0_noG0_munu, G0_mu_nu


def _mesh_coords_for_local_process(mesh_xy):
    devices_2d = np.array(mesh_xy.devices)
    local = list(jax.local_devices())
    local.sort(key=lambda d: d.id)
    target = local[0]
    coord = tuple(np.argwhere(devices_2d == target)[0])
    return coord


def save_restart_state_per_proc(
    prefix: str,
    V_qmunu,
    S_qmunu,
    psi_full_y,
    enk_full,
    meta,
    mesh_xy,
    V0_noG0_munu=None,
):
    """Save local per-process shards for canonical restart state."""
    del meta
    cx, cy = _mesh_coords_for_local_process(mesh_xy)
    rank = jax.process_index()
    fname = f"{prefix}.rank{rank}.x{cx}.y{cy}.h5"
    devices_2d = np.array(mesh_xy.devices)
    grid_x, grid_y = devices_2d.shape

    def _block_slice(n, parts, idx):
        start = (n * idx) // parts
        end = (n * (idx + 1)) // parts
        return int(start), int(end)

    vx0, vx1 = _block_slice(int(V_qmunu.shape[-2]), grid_x, cx)
    vy0, vy1 = _block_slice(int(V_qmunu.shape[-1]), grid_y, cy)
    V_local = jax.device_get(V_qmunu[..., vx0:vx1, vy0:vy1])

    V0_local = None
    if V0_noG0_munu is not None:
        vx0_V0, vx1_V0 = _block_slice(int(V0_noG0_munu.shape[-2]), grid_x, cx)
        vy0_V0, vy1_V0 = _block_slice(int(V0_noG0_munu.shape[-1]), grid_y, cy)
        V0_local = jax.device_get(V0_noG0_munu[vx0_V0:vx1_V0, vy0_V0:vy1_V0])

    S_local = None
    if S_qmunu is not None:
        sx0, sx1 = _block_slice(int(S_qmunu.shape[-2]), grid_x, cx)
        sy0, sy1 = _block_slice(int(S_qmunu.shape[-1]), grid_y, cy)
        S_local = jax.device_get(S_qmunu[..., sx0:sx1, sy0:sy1])

    py0, py1 = _block_slice(int(psi_full_y.shape[-1]), grid_y, cy)
    psi_full_local = jax.device_get(psi_full_y[..., py0:py1])

    def _to_np(a):
        try:
            return jax.device_get(a)
        except Exception:
            return a.get() if hasattr(a, "get") else np.asarray(a)

    with h5py.File(fname, "w") as f:
        f.attrs["restart_format_version"] = 2
        f.attrs["global_V_shape"] = np.array(V_qmunu.shape, dtype=np.int64)
        f.attrs["global_S_shape"] = (
            np.array(S_qmunu.shape, dtype=np.int64)
            if S_qmunu is not None
            else np.array([-1], dtype=np.int64)
        )
        f.attrs["global_psi_full_shape"] = np.array(psi_full_y.shape, dtype=np.int64)
        if enk_full is not None:
            f.attrs["global_enk_full_shape"] = np.array(enk_full.shape, dtype=np.int64)
        if V0_noG0_munu is not None:
            f.attrs["global_V0_shape"] = np.array(V0_noG0_munu.shape, dtype=np.int64)
        f.attrs["grid_x"] = int(grid_x)
        f.attrs["grid_y"] = int(grid_y)
        f.attrs["coord_x"] = int(cx)
        f.attrs["coord_y"] = int(cy)
        f.create_dataset("V_local", data=V_local)
        if S_local is not None:
            f.create_dataset("S_local", data=S_local)
        if V0_local is not None:
            f.create_dataset("V0_noG0_local", data=V0_local)
        f.create_dataset("psi_full_local", data=psi_full_local)
        if enk_full is not None:
            f.create_dataset("enk_full", data=_to_np(enk_full))
