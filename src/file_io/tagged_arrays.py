"""Canonical restart-state I/O for GW/BSE workflows.

This module reads/writes HDF5 restart files in the v2 format used by gw_jax.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import h5py
from jax.sharding import NamedSharding, PartitionSpec as P


# --- File layout (restart_format_version = 3) -------------------------------
#
# V and W are bosonic two-point tensors with two Lorentz-channel indices
# (μ_L, ν_L) and two centroid indices (μ_X, ν_Y).  Non-bispinor runs use
# npol=1 (one channel only, ``(0,0)``); bispinor runs use npol=4 with up to
# 16 channels (10 non-zero + 6 zero by Coulomb gauge for V).  μ_X and ν_Y
# can have DIFFERENT centroid counts when the (μ_L, ν_L) tile is built on
# different centroid sets (charge n_rmu_0 vs current n_rmu_curr in the
# bispinor pipeline) — so each slot is its OWN dataset with its own
# (n_rmu_X, n_rmu_Y) shape, written via the SAME ``write_slab`` machinery
# as a non-bispinor V_qmunu would be.
#
#   restart.h5/
#       V_qmunu/                            HDF5 group
#           pol_{μ_L}_{ν_L}/                 (nkx, nky, nkz, n_rmu_X, n_rmu_Y) c128
#                                            sharded P(None,None,None,'x','y')
#           pol_..../...
#           [and similarly for W0_qmunu, S_qmunu, V0_noG0_munu]
#
# Same shape rules apply for any other bosonic (μ_L, ν_L) tensor: pass a
# ``dict[(int, int), Array]`` to ``write_restart_state_to_h5`` (one slot
# per non-zero channel, omit zeros) and the file lands with one dataset
# per slot.  The reader returns a dict keyed by (μ_L, ν_L).
#
# Backwards-compatibility: ``V_qmunu = <Array>`` (legacy, format v2) still
# works and gets stored at the legacy ``V_qmunu`` dataset path.  The
# reader detects either format.

def _normalize_block_dict(arg, name: str):
    """Accept either an Array (single (0,0) slot) or a dict[(μ_L,ν_L),Array]
    and return a normalised dict.  ``None`` passes through.  Used by the
    writer to unify the legacy and per-slot interfaces."""
    if arg is None:
        return None
    if isinstance(arg, dict):
        return {(int(k[0]), int(k[1])): v for k, v in arg.items()}
    # Treat a single Array as the (0, 0) Lorentz slot (npol=1).
    return {(0, 0): arg}


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
    backend=None,
    use_ffi_io: bool | None = None,
    mode: str = "w",
):
    """Write (subset of) canonical restart state via SlabIO.

    All array arguments are optional — only the provided ones are
    written, so this function can be called multiple times to flush
    pieces of the restart state as they become available.  With
    ``mode="w"`` the file is truncated first (and the format-version
    attribute written); with ``mode="a"`` the file is opened for
    append / overwrite of the named datasets.

    ``V_qmunu``, ``S_qmunu``, ``W0_qmunu`` may be either a single Array
    (legacy / non-bispinor — treated as the (0,0) Lorentz slot) OR a
    ``dict[(μ_L, ν_L), Array]`` of per-Lorentz-channel slabs (bispinor;
    typically 10 non-zero entries from
    ``v_q_lorentz.compute_all_V_q_lorentz_sharded``).  Each slot writes
    a separate ``<name>/pol_{μ_L}_{ν_L}`` dataset of shape
    ``(nkx, nky, nkz, n_rmu_X, n_rmu_Y)`` sharded
    ``P(None,None,None,'x','y')`` — the SAME write_slab path used for
    legacy V_qmunu.  The 6 (0,i)/(i,0) gauge-zero tiles can simply be
    omitted from the dict.

    ``init_W0=True`` pre-allocates an all-zeros W0_qmunu dataset sized
    from ``V_qmunu``; the ``W0_ready`` attr on that dataset is set to
    False so downstream readers (bse_io) know to treat it as a
    placeholder.  Passing ``W0_qmunu`` directly flips ``W0_ready`` to
    True.
    """
    from .slab_io import SlabIO

    V_blocks  = _normalize_block_dict(V_qmunu,  "V_qmunu")
    S_blocks  = _normalize_block_dict(S_qmunu,  "S_qmunu")
    W0_blocks = _normalize_block_dict(W0_qmunu, "W0_qmunu")

    with SlabIO(filename, mode=mode, mesh=mesh,
                backend=backend, use_ffi_io=use_ffi_io) as io:
        if mode == "w":
            io.write_attr("restart_format_version", np.int64(3))

        def _write(name, arr):
            if arr is None:
                return
            shape = tuple(arr.shape)
            io.create_dataset(name, shape=shape, dtype=arr.dtype)
            io.write_slab(name, arr, global_shape=shape)

        def _write_blocks(group_name, blocks: dict | None):
            """Write a dict-of-(μ_L,ν_L)-blocks under ``<group_name>_pol_X_Y``.
            Each slot uses the SAME ``write_slab`` path as the legacy
            single-tensor write — just looped over slots.  Flat naming
            (no nested HDF5 groups) because the phdf5 FFI's H5Dcreate
            doesn't accept paths through intermediate groups.
            """
            if blocks is None:
                return
            for (mu_L, nu_L), arr in sorted(blocks.items()):
                _write(f"{group_name}_pol_{int(mu_L)}_{int(nu_L)}", arr)

        _write_blocks("V_qmunu", V_blocks)
        _write_blocks("S_qmunu", S_blocks)
        _write("V0_noG0_munu", V0_noG0_munu)
        _write("G0_mu_nu",     G0_mu_nu)
        _write("psi_full_y",   psi_full_y)
        _write("enk_full",     enk_full)

        # W0_qmunu: either write the real data or pre-allocate an
        # all-zeros placeholder per (μ_L, ν_L) slot.
        w0_touched = W0_qmunu is not None or init_W0
        w0_ready = False
        if W0_blocks is not None:
            _write_blocks("W0_qmunu", W0_blocks)
            w0_ready = True
        elif init_W0:
            if V_blocks is None:
                raise ValueError("init_W0=True requires V_qmunu to size the placeholder")
            for (mu_L, nu_L), arr in sorted(V_blocks.items()):
                shape = tuple(arr.shape)
                io.create_dataset(
                    f"W0_qmunu_pol_{int(mu_L)}_{int(nu_L)}",
                    shape=shape, dtype=arr.dtype)

    # bse_io.py reads W0_ready as an HDF5 attr.  v3 layout has flat
    # ``W0_qmunu_pol_X_Y`` datasets; we attach the attr to the (0,0)
    # slot (always present).  Legacy v2 files (single ``W0_qmunu``
    # dataset) keep the attr on the dataset.
    if w0_touched and jax.process_index() == 0:
        with h5py.File(filename, "a") as f:
            target = "W0_qmunu_pol_0_0" if "W0_qmunu_pol_0_0" in f else "W0_qmunu"
            f[target].attrs["W0_ready"] = w0_ready
    try:
        from jax.experimental import multihost_utils as _mh
        _mh.sync_global_devices("restart_W0_ready_flag")
    except Exception:
        pass


def write_w0_qmunu_to_h5(
    filename, W0_qmunu, mesh=None,
    backend=None, use_ffi_io: bool | None = None,
):
    """Overwrite or append the W0_qmunu group in an existing restart file.

    ``W0_qmunu`` may be a single Array (legacy / non-bispinor — written
    as the (0,0) slot) or a ``dict[(μ_L, ν_L), Array]`` of per-Lorentz-
    channel slabs.  Each slot writes to ``W0_qmunu_pol_X_Y`` via the
    same SlabIO machinery.
    """
    from .slab_io import SlabIO

    W0_blocks = _normalize_block_dict(W0_qmunu, "W0_qmunu")

    with SlabIO(filename, mode="a", mesh=mesh,
                backend=backend, use_ffi_io=use_ffi_io) as io:
        for (mu_L, nu_L), arr in sorted(W0_blocks.items()):
            shape = tuple(arr.shape)
            name = f"W0_qmunu_pol_{int(mu_L)}_{int(nu_L)}"
            io.create_dataset(name, shape=shape, dtype=arr.dtype)
            io.write_slab(name, arr, global_shape=shape)

    # W0_ready flag (attr on the (0,0) slot for v3, or the single
    # dataset for legacy v2) is read by bse_io.py.
    if jax.process_index() == 0:
        with h5py.File(filename, "a") as f:
            target = "W0_qmunu_pol_0_0" if "W0_qmunu_pol_0_0" in f else "W0_qmunu"
            f[target].attrs["W0_ready"] = True
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


def _read_blocks_or_legacy(f, name):
    """Read either v3 per-pol datasets named ``<name>_pol_{μL}_{νL}``
    returning a ``dict[(μ_L, ν_L), Array]``, or a legacy v2 single
    dataset at ``<name>`` returning ``{(0, 0): Array}``.  Returns
    ``None`` if neither form is present.

    Flat naming (no nested HDF5 groups) because the phdf5 FFI's
    ``H5Dcreate`` requires top-level dataset paths.
    """
    prefix = f"{name}_pol_"
    out: dict[tuple[int, int], jnp.ndarray] = {}
    for ds_name in f.keys():
        if not ds_name.startswith(prefix):
            continue
        suffix = ds_name[len(prefix):]
        try:
            mu_s, nu_s = suffix.split("_", 1)
            out[(int(mu_s), int(nu_s))] = jnp.asarray(f[ds_name][:])
        except ValueError:
            continue
    if out:
        return out
    if name in f and isinstance(f[name], h5py.Dataset):  # legacy v2
        return {(0, 0): jnp.asarray(f[name][:])}
    return None


def read_restart_state_from_h5(filename):
    """Read canonical restart state from HDF5.

    Returns ``V_blocks`` and ``S_blocks`` as ``dict[(μ_L, ν_L), Array]``
    (a single-entry dict for non-bispinor / legacy v2 files; up to 16
    entries for bispinor v3 files).  Other quantities are unchanged.
    """
    with h5py.File(filename, "r") as f:
        if "psi_full_y" not in f:
            raise ValueError(
                f"Restart file {filename} is missing canonical psi_full_y dataset. "
                "Regenerate restart tensors with current gw_jax."
            )

        V_blocks = _read_blocks_or_legacy(f, "V_qmunu")
        S_blocks = _read_blocks_or_legacy(f, "S_qmunu")
        V0_noG0_munu = jnp.asarray(f["V0_noG0_munu"][:]) if "V0_noG0_munu" in f else None
        G0_mu_nu = jnp.asarray(f["G0_mu_nu"][:]) if "G0_mu_nu" in f else None
        psi_full_y = jnp.asarray(f["psi_full_y"][:])
        enk_full = jnp.asarray(f["enk_full"][:]) if "enk_full" in f else None

    return V_blocks, S_blocks, psi_full_y, enk_full, V0_noG0_munu, G0_mu_nu


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
    V_blocks, S_blocks, psi_full_y_raw, enk_full, V0_noG0_munu, G0_mu_nu = read_restart_state_from_h5(filename)

    x3y4_5 = NamedSharding(mesh_xy, P(None, None, None, "x", "y"))
    y3_psi_Y = NamedSharding(mesh_xy, P(None, None, None, "y"))
    x1_psi_X = NamedSharding(mesh_xy, P(None, "x", None, None))
    replicated_2 = NamedSharding(mesh_xy, P(None, None))

    # Per-pol slabs are 5-D ``(nkx, nky, nkz, n_rmu_X, n_rmu_Y)`` regardless
    # of whether the file held a v2 single-tensor ``V_qmunu`` (now wrapped
    # as {(0,0): ...} by the reader, but with a leading 1×npol×npol stub
    # layout — strip those if present) or a v3 per-pol group.
    def _ensure_5d(arr):
        # Legacy v2 stored (1, npol, npol, nkx, nky, nkz, μ, μ); strip
        # the three leading replicated/identity axes and keep the
        # 5-D core.  v3 already stores 5-D per-pol slabs directly.
        if arr.ndim == 8 and arr.shape[0] == 1:
            arr = arr[0, 0, 0]  # take (0,0) Lorentz slot
        return arr

    if V_blocks is not None:
        V_blocks = {k: jax.lax.with_sharding_constraint(_ensure_5d(v), x3y4_5)
                    for k, v in V_blocks.items()}
    if S_blocks is not None:
        S_blocks = {k: jax.lax.with_sharding_constraint(_ensure_5d(v), x3y4_5)
                    for k, v in S_blocks.items()}
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
        V_blocks=V_blocks,
        S_blocks=S_blocks,
        # Legacy aliases — populated only when the file holds a single
        # (0,0) slot (v2 or non-bispinor v3) so callers that haven't yet
        # been updated to consume the dict still see a single tensor.
        V_qmunu=(V_blocks[(0, 0)]
                 if V_blocks is not None and len(V_blocks) == 1 and (0, 0) in V_blocks
                 else None),
        S_qmunu=(S_blocks[(0, 0)]
                 if S_blocks is not None and len(S_blocks) == 1 and (0, 0) in S_blocks
                 else None),
        V0_noG0_munu=V0_noG0_munu,
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
    """Save local per-process shards for canonical restart state.

    ``V_qmunu`` and ``S_qmunu`` may be either a single Array (legacy /
    non-bispinor — treated as the (0,0) Lorentz slot) OR a
    ``dict[(μ_L, ν_L), Array]`` of per-Lorentz-channel slabs.  Each slot
    is sliced INDEPENDENTLY at its own (n_rmu_X, n_rmu_Y) extent and
    saved at ``V_local_pol_{μ_L}_{ν_L}`` (group + dataset).  This
    matches the v3 file layout in :func:`write_restart_state_to_h5`
    and avoids the broadcast-view rematerialisation OOM that hits when
    the bispinor V_qmunu is a 16-fold-replicated view.
    """
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

    V_blocks = _normalize_block_dict(V_qmunu, "V_qmunu")
    S_blocks = _normalize_block_dict(S_qmunu, "S_qmunu")

    def _slice_block(arr):
        """Slice a (..., n_rmu_X, n_rmu_Y) shard down to this rank's
        (cx, cy)-local block on the trailing two axes.  For a properly-
        materialised sharded array this is a local op (no NCCL); for a
        broadcast view it forces materialisation — the new format
        always passes real per-pol slabs so this stays cheap."""
        ax0 = int(arr.shape[-2]); ay0 = int(arr.shape[-1])
        x0, x1 = _block_slice(ax0, grid_x, cx)
        y0, y1 = _block_slice(ay0, grid_y, cy)
        return jax.device_get(arr[..., x0:x1, y0:y1])

    V_local_blocks: dict[tuple[int, int], np.ndarray] | None = None
    if V_blocks is not None:
        V_local_blocks = {(mu, nu): _slice_block(arr)
                          for (mu, nu), arr in V_blocks.items()}

    S_local_blocks: dict[tuple[int, int], np.ndarray] | None = None
    if S_blocks is not None:
        S_local_blocks = {(mu, nu): _slice_block(arr)
                          for (mu, nu), arr in S_blocks.items()}

    V0_local = None
    if V0_noG0_munu is not None:
        vx0_V0, vx1_V0 = _block_slice(int(V0_noG0_munu.shape[-2]), grid_x, cx)
        vy0_V0, vy1_V0 = _block_slice(int(V0_noG0_munu.shape[-1]), grid_y, cy)
        V0_local = jax.device_get(V0_noG0_munu[vx0_V0:vx1_V0, vy0_V0:vy1_V0])

    py0, py1 = _block_slice(int(psi_full_y.shape[-1]), grid_y, cy)
    psi_full_local = jax.device_get(psi_full_y[..., py0:py1])

    def _to_np(a):
        try:
            return jax.device_get(a)
        except Exception:
            return a.get() if hasattr(a, "get") else np.asarray(a)

    with h5py.File(fname, "w") as f:
        f.attrs["restart_format_version"] = 3
        f.attrs["global_psi_full_shape"] = np.array(psi_full_y.shape, dtype=np.int64)
        if enk_full is not None:
            f.attrs["global_enk_full_shape"] = np.array(enk_full.shape, dtype=np.int64)
        if V0_noG0_munu is not None:
            f.attrs["global_V0_shape"] = np.array(V0_noG0_munu.shape, dtype=np.int64)
        f.attrs["grid_x"] = int(grid_x)
        f.attrs["grid_y"] = int(grid_y)
        f.attrs["coord_x"] = int(cx)
        f.attrs["coord_y"] = int(cy)

        def _save_blocks(group_name: str,
                         local_blocks: dict | None,
                         global_blocks: dict | None):
            """Write per-(μ_L,ν_L) shards under <group_name>/pol_X_Y and
            stash each slot's global shape as an attribute on the
            dataset (so the loader can reconstruct the full layout)."""
            if local_blocks is None or global_blocks is None:
                return
            grp = f.create_group(group_name)
            grp.attrs["pol_keys"] = np.array(
                [[int(k[0]), int(k[1])] for k in sorted(local_blocks)],
                dtype=np.int64)
            for (mu, nu), shard in sorted(local_blocks.items()):
                ds = grp.create_dataset(f"pol_{int(mu)}_{int(nu)}", data=shard)
                ds.attrs["global_shape"] = np.array(
                    global_blocks[(mu, nu)].shape, dtype=np.int64)

        _save_blocks("V_local", V_local_blocks, V_blocks)
        _save_blocks("S_local", S_local_blocks, S_blocks)
        if V0_local is not None:
            f.create_dataset("V0_noG0_local", data=V0_local)
            f["V0_noG0_local"].attrs["global_shape"] = np.array(
                V0_noG0_munu.shape, dtype=np.int64)
        f.create_dataset("psi_full_local", data=psi_full_local)
        if enk_full is not None:
            f.create_dataset("enk_full", data=_to_np(enk_full))
