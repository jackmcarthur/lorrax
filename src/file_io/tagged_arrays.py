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
    *,
    V_qmunu=None,
    psi_full_y=None,
    enk_full=None,
    S_qmunu=None,
    V0_noG0_munu=None,
    G0_mu_nu=None,
    W0_qmunu=None,
    init_W0: bool = False,
    mesh=None,
    use_ffi_io: bool = False,
    mode: str = "w",
):
    """Write (subset of) canonical restart state via SlabIO.

    All array arguments are optional — only the provided ones are
    written, so this function can be called multiple times to flush
    pieces of the restart state as they become available.  With
    ``mode="w"`` the file is truncated first (and the format-version
    attribute written); with ``mode="a"`` the file is opened for
    append / overwrite of the named datasets.

    ``init_W0=True`` pre-allocates an all-zeros W0_qmunu dataset sized
    from ``V_qmunu``; the ``W0_ready`` attr on that dataset is set to
    False so downstream readers (bse_io) know to treat it as a
    placeholder.  Passing ``W0_qmunu`` directly flips ``W0_ready`` to
    True.
    """
    from .slab_io import SlabIO

    with SlabIO(filename, mode=mode, mesh=mesh, use_ffi_io=use_ffi_io) as io:
        if mode == "w":
            io.write_attr("restart_format_version", np.int64(2))

        def _write(name, arr):
            if arr is None:
                return
            shape = tuple(arr.shape)
            io.create_dataset(name, shape=shape, dtype=arr.dtype)
            io.write_slab(name, arr, global_shape=shape)

        _write("V_qmunu",      V_qmunu)
        _write("S_qmunu",      S_qmunu)
        _write("V0_noG0_munu", V0_noG0_munu)
        _write("G0_mu_nu",     G0_mu_nu)
        _write("psi_full_y",   psi_full_y)
        _write("enk_full",     enk_full)

        # W0_qmunu: either write the real data or pre-allocate an
        # all-zeros placeholder.
        w0_touched = W0_qmunu is not None or init_W0
        w0_ready = False
        if W0_qmunu is not None:
            shape = tuple(W0_qmunu.shape)
            io.create_dataset("W0_qmunu", shape=shape, dtype=W0_qmunu.dtype)
            io.write_slab("W0_qmunu", W0_qmunu, global_shape=shape)
            w0_ready = True
        elif init_W0:
            if V_qmunu is None:
                raise ValueError("init_W0=True requires V_qmunu to size the placeholder")
            v_shape = tuple(V_qmunu.shape)
            v_dtype = V_qmunu.dtype
            io.create_dataset("W0_qmunu", shape=v_shape, dtype=v_dtype)

    # bse_io.py reads W0_ready as an HDF5 attr on the W0_qmunu dataset.
    # Set it rank-0-only after SlabIO has released the file, to stay
    # compatible with that reader.
    if w0_touched and jax.process_index() == 0:
        with h5py.File(filename, "a") as f:
            f["W0_qmunu"].attrs["W0_ready"] = w0_ready
    try:
        from jax.experimental import multihost_utils as _mh
        _mh.sync_global_devices("restart_W0_ready_flag")
    except Exception:
        pass


def write_w0_qmunu_to_h5(
    filename, W0_qmunu, mesh=None, use_ffi_io: bool = False,
):
    """Overwrite or append the W0_qmunu dataset in an existing restart file."""
    from .slab_io import SlabIO

    shape = tuple(W0_qmunu.shape)
    with SlabIO(filename, mode="a", mesh=mesh, use_ffi_io=use_ffi_io) as io:
        io.create_dataset("W0_qmunu", shape=shape, dtype=W0_qmunu.dtype)
        io.write_slab("W0_qmunu", W0_qmunu, global_shape=shape)

    # W0_ready flag is a per-dataset attr read by bse_io.py.
    if jax.process_index() == 0:
        with h5py.File(filename, "a") as f:
            f["W0_qmunu"].attrs["W0_ready"] = True
    try:
        from jax.experimental import multihost_utils as _mh
        _mh.sync_global_devices("restart_W0_ready_flag")
    except Exception:
        pass


def write_head_scalars_to_h5(
    filename: str,
    *,
    vhead: complex | None = None,
    whead: np.ndarray | jnp.ndarray | None = None,
    omega_grid: np.ndarray | jnp.ndarray | None = None,
):
    """Persist q=0 Coulomb head scalars to the restart file.

    Stored alongside ``G0_mu_nu``; consumed by ``bse_io._load_ring_subset``
    (and any future Σ-builder) via ``head_correction.apply_q0_head_rank1``.

    - ``vhead``: scalar v(q→0, G=G'=0) in Ry, BGW convention.
    - ``whead``: shape ``(n_omega,)``. Length 1 for static COHSEX,
      length 2 for GN-PPM (static, iω_p).
    - ``omega_grid``: optional ``(n_omega,)`` array of the ω values
      (in Ry) corresponding to ``whead`` — written as an attribute on
      the ``whead`` dataset for consumer interpretation.

    Rank-0-only write (these are tiny; no MPI-IO needed).
    """
    if jax.process_index() != 0:
        try:
            from jax.experimental import multihost_utils as _mh
            _mh.sync_global_devices("restart_head_scalars")
        except Exception:
            pass
        return
    with h5py.File(filename, "a") as f:
        if vhead is not None:
            if "vhead" in f:
                del f["vhead"]
            f.create_dataset("vhead", data=np.complex128(vhead))
        if whead is not None:
            if "whead" in f:
                del f["whead"]
            arr = np.asarray(whead, dtype=np.complex128).reshape(-1)
            ds = f.create_dataset("whead", data=arr)
            if omega_grid is not None:
                ds.attrs["omega_grid"] = np.asarray(omega_grid, dtype=np.float64).reshape(-1)
    try:
        from jax.experimental import multihost_utils as _mh
        _mh.sync_global_devices("restart_head_scalars")
    except Exception:
        pass


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
    """Load canonical restart state and reshape wavefunctions into the
    two arrays expected by :func:`gw.wavefunction_bundle.build_wavefunctions`.

    Returns a ``SimpleNamespace`` with fields:

      V_qmunu, S_qmunu, V0_noG0_munu, G0_mu_nu, enk_full
      psi_rmu_Y   (nk, nb, ns, n_rmu)   P(None, None, None, 'y')
                  un-conjugated ψ.
      psi_rmuT_X  (nk, n_rmu, nb, ns)   P(None, 'x', None, None)
                  conjugated ψ* (matches the pair-density convention
                  ``load_centroids_band_chunked`` uses).

    The x-sharded psi copy is derived from the y-sharded one with a
    single y→x all-to-all on the μ axis; this is the only reshard on
    the restart path.
    """
    del band_slices  # retained for call-site compatibility
    from types import SimpleNamespace
    V_qmunu, S_qmunu, psi_full_y_raw, enk_full, V0_noG0_munu, G0_mu_nu = read_restart_state_from_h5(filename)

    x6y7_8 = NamedSharding(mesh_xy, P(None, None, None, None, None, None, "x", "y"))
    x3y4_5 = NamedSharding(mesh_xy, P(None, None, None, "x", "y"))
    y3_psi_Y = NamedSharding(mesh_xy, P(None, None, None, "y"))
    x1_psi_X = NamedSharding(mesh_xy, P(None, "x", None, None))
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

    # psi_rmu_Y: stored layout (un-conjugated ψ), just pin to Y-sharding.
    psi_rmu_Y = jax.lax.with_sharding_constraint(psi_full_y_raw, y3_psi_Y)
    # psi_rmuT_X: conj + transpose(nb↔μ) then y→x reshard on μ.
    psi_rmuT_X = jax.lax.with_sharding_constraint(
        jnp.conj(psi_rmu_Y).transpose(0, 3, 1, 2),
        x1_psi_X,
    )
    if enk_full is not None:
        enk_full = jax.lax.with_sharding_constraint(enk_full, replicated_2)

    return SimpleNamespace(
        V_qmunu=V_qmunu, S_qmunu=S_qmunu, V0_noG0_munu=V0_noG0_munu,
        G0_mu_nu=G0_mu_nu, enk_full=enk_full,
        psi_rmu_Y=psi_rmu_Y, psi_rmuT_X=psi_rmuT_X,
    )


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
