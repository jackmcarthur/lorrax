"""Reading a GW restart into a BSE bundle: the gates, the transport, the loaders.

AUTHORITY RULE — a restart tensor is refused unless the FILE says its data was
persisted, and the q-set it is stored on is asked once and answered once.
``W0_qmunu`` carries ``W0_ready`` and ``V_qmunu`` carries ``V_ready``; ABSENT
MEANS READY, so a file written before those attrs existed loads exactly as it
always did and only a file that positively claims "not persisted" is refused.
Whether a tensor is stored on the IBZ q wedge is ``is_q_wedge``'s question and
``restart_munu_full_bz``'s answer, and every h5py-side reader here goes through
them rather than subscripting a dataset — a wedge read under a full-BZ q index
passes every shape check downstream.

Which bytes move is settled independently of how they move.  The SlabIO
transport and the serial h5py tile readers return identical global shapes,
identical PartitionSpecs and identical per-rank tiles, so the parity bar
between them is BIT EQUALITY; the choice is made once per load, is a pure
function of the platform probes, and is announced when it declines to the slow
path.  Nothing here materialises more than one rank's (μ, ν) tile and there is
no allgather on any arm.

``load_bse_data_from_restart_sharded`` is the production path at any process
count; ``_load_ring_subset`` is the single-device full-file reader and refuses
at P>1.  Both publish the resolved window under the same four names, take the
band window from the same guard, and inject the q=0 head through the same
helper, so a request cannot mean two different things depending on which one
served it.

ONE ORDERING IS LOAD-BEARING: whether a coarse→fine densification is pending is
resolved BEFORE the head injection, because the answer changes what the
injection does (``bse_head``'s ``defer_whead``, ``bse_densify``'s
re-attachment).
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from runtime.padding import pad_axis, padded_mu_extent
from common.band_degeneracy import (DEFAULT_MODE, DEGENERACY_TOL_RY,
                                    resolve_band_window)

from .bse_densify import (_interpolate_bse_data_to_grid,
                          _read_lorrax_input_quietly, _resolve_bse_k_grid,
                          resolve_w_head_densify)
from .bse_head import _inject_q0_head, _resolve_head_params
from .bse_serial import compute_pair_amplitude
from .bse_window import (PAD_EPS_GUARD_RY, _log0,
                         _parse_wfn_path, apply_eqp_corrections, resolve_n_occ)


_BARE_V_FALLBACK_WARNING = (
    "\n" + "=" * 72 + "\n"
    "BSE WARNING: W0_qmunu not ready (missing or W0_ready=False) -- falling\n"
    "back to BARE COULOMB V for the screened interaction W. This is NOT a\n"
    "physical BSE calculation: excitonic binding from screening is entirely\n"
    "absent. Re-run GW screening to produce a ready W0_qmunu before trusting\n"
    "these results.\n"
    + "=" * 72
)


def _refuse_unpersisted(dset, name: str, restart_file: str) -> None:
    """Refuse a restart tensor whose file says its data was never written.

    The ``V_ready`` counterpart of W0's ``W0_ready`` gate; the writer is
    ``file_io.tagged_arrays``.  ABSENT MEANS READY — a file written before
    the attr existed carries nothing here and loads byte-for-byte as it
    always did, so only a file that positively claims "not persisted" is
    refused.  That claim is the only way to catch the one state the numbers
    cannot betray: a full-size zero placeholder passes every shape check.
    """
    if dset.attrs.get("V_ready", True):
        return
    raise ValueError(
        f"{restart_file}: {name} is present and correctly shaped but its "
        f"V_ready attr is False — the file says this tensor's data was "
        f"never persisted.  Reading it would hand the BSE a tensor of "
        f"zeros that passes every shape check, which is exactly the "
        f"mechanism behind the all-zero-screening incident.  Re-run the "
        f"GW leg that writes it.")


def is_q_wedge(dset) -> bool:
    """Is this restart tensor stored on the IBZ q wedge?  ATTRS ONLY.

    A one-line wrapper because the ANSWER must have exactly one
    implementation — ``symmetry_maps.dataset_q_storage``, which also owns
    the refusal on a partially-stamped file — and because the service-path
    bootstrap belongs at one probe site, not at every one.  ``False`` for
    every pre-q_irr file and every ``restart_q_storage = full`` file, which
    is what keeps those readers on the byte path they have always had.
    """
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import dataset_q_storage
    return dataset_q_storage(dset) == "ibz"


def restart_munu_full_bz(dset, name: str, restart_file: str):
    """THE ONE PLACE A RESTART V/W TENSOR BECOMES A FULL-BZ HOST ARRAY.

    Returns a numpy ``(…, μ, ν)`` array on the FULL BZ, whatever q-set the
    file stores it on.  Every h5py-side restart reader in this module and in
    ``bse.vq_interp`` asks this instead of subscripting the dataset, so the
    question "is this file a q wedge" is asked once and answered once.

    THE LEGACY PATH IS THE SAME BYTES IT ALWAYS WAS: a dataset with no
    ``qirr_*`` attrs goes through ``dset[()]`` and nothing else happens.
    A wedge unfolds once, all at once, through
    ``symmetry_maps.unfold_isdf_operator`` driven by the tables stored IN
    THE FILE, so what comes back is the same function of the same inputs
    the producing run used rather than a re-derivation that could drift.

    THE MESH IS THIS MODULE'S CHOICE, NOT THE FORMAT'S: the unfold is a
    sharded double-gather and needs one, but "which devices" is a run-level
    decision the format layer must not take.  Callers here are on the host
    single-device path, so ``collectives.single_device_mesh()`` is named at
    the site.  That makes this a stopgap — it materialises the full BZ once
    per caller per rank.  The sharded transport does not come through here;
    ``_MunuSlabPlan`` refuses a wedge.

    THE μ PAD IS NOT RE-APPLIED HERE.  The file stores the LOGICAL extent
    (SHARDING_RULES §2) and every caller pads on its own axis afterwards
    from its own ``n_rmu_pad``.  Padding here would bake THIS process's
    device count into the array — the one device-count-dependent quantity
    the on-disk format exists to keep out.
    """
    if not is_q_wedge(dset):
        return np.asarray(dset[()])
    from symmetry_maps import read_tensor
    from common.collectives import single_device_mesh
    grp = dset.parent
    full, header = read_tensor(grp, name, mesh_xy=single_device_mesh())
    if not header.was_unfolded:                       # pragma: no cover
        raise AssertionError(
            f"{restart_file}: {name} probed as a q wedge and read back as "
            f"{header.q_storage!r}; the attr and the shape disagree, which "
            f"qirr_store.read_tensor should already have refused.")
    return np.asarray(full)


def _pad_last_axis(x: jax.Array, target: int) -> jax.Array:
    pad = target - x.shape[-1]
    if pad <= 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, pad)
    return jnp.pad(x, pad_width, mode="constant")


def _pad_last_two_axes(x: jax.Array, target: int) -> jax.Array:
    pad0 = target - x.shape[-2]
    pad1 = target - x.shape[-1]
    if pad0 <= 0 and pad1 <= 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[-2] = (0, max(0, pad0))
    pad_width[-1] = (0, max(0, pad1))
    return jnp.pad(x, pad_width, mode="constant")


def _pad_first_two_axes(x: jax.Array, target: int) -> jax.Array:
    pad0 = target - x.shape[0]
    pad1 = target - x.shape[1]
    if pad0 <= 0 and pad1 <= 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[0] = (0, max(0, pad0))
    pad_width[1] = (0, max(0, pad1))
    return jnp.pad(x, pad_width, mode="constant")


def _get_local_mesh_coords(
    mesh_xy: Mesh, *, origin: str = "bse_loading._get_local_mesh_coords",
) -> tuple[list[tuple[int, int]], int, int]:
    """This process's (x, y) coords in ``mesh_xy``, plus the grid shape.

    Refuses up front (``collectives._require_addressable``) a mesh on which
    THIS process owns no device; without it
    ``make_array_from_process_local_data`` raises a bare
    ``StopIteration('')`` naming neither rank nor mesh.

    THE GUARD BELONGS HERE, not at the three
    ``make_array_from_process_local_data`` call sites: the very next line —
    ``np.argwhere(...)[0]`` over a device this mesh does not contain —
    would raise an equally anonymous ``IndexError`` before a guard placed
    lower down could speak.
    """
    from common.collectives import _require_addressable

    _require_addressable(mesh_xy, origin=origin)
    devices_2d = np.asarray(mesh_xy.devices)
    grid_x, grid_y = devices_2d.shape
    local_devices = list(jax.local_devices())
    local_coords = [tuple(np.argwhere(devices_2d == d)[0]) for d in local_devices]
    local_coords = sorted(local_coords, key=lambda c: c[0] * grid_y + c[1])
    return local_coords, grid_x, grid_y


def _get_local_axis_coords(local_coords: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    local_x = sorted({coord[0] for coord in local_coords})
    local_y = sorted({coord[1] for coord in local_coords})
    return local_x, local_y


def _assert_local_block(local_coords: list[tuple[int, int]], local_x: list[int], local_y: list[int]) -> None:
    expected = {(x, y) for x in local_x for y in local_y}
    actual = set(local_coords)
    if actual != expected:
        raise ValueError(
            "Local devices are not a full x/y block; shard-aware loader expects "
            "local device coords to form a Cartesian product."
        )


def _read_psi_mu_sharded(
    dset: h5py.Dataset,
    band_indices: np.ndarray,
    mu_per_shard: int,
    axis: str,
    mesh_xy: Mesh,
    n_rmu_pad: int,
    dtype: np.dtype = np.complex128,
    trim: bool = True,
) -> jax.Array:
    local_coords, grid_x, grid_y = _get_local_mesh_coords(
        mesh_xy, origin="bse_loading._read_psi_mu_sharded")
    local_x, local_y = _get_local_axis_coords(local_coords)
    _assert_local_block(local_coords, local_x, local_y)

    n_rmu = dset.shape[3]
    nk = dset.shape[0]
    nspinor = dset.shape[2]

    local_mu = mu_per_shard * (len(local_x) if axis == "x" else len(local_y))
    local_psi = np.zeros((nk, len(band_indices), nspinor, local_mu), dtype=dtype)

    if axis == "x":
        coords = local_x
    else:
        coords = local_y

    for i, coord in enumerate(coords):
        mu_start = coord * mu_per_shard
        mu_end = min(mu_start + mu_per_shard, n_rmu)
        if mu_start >= n_rmu:
            continue
        slab = dset[:, band_indices, :, mu_start:mu_end]
        if slab.shape[3] < mu_per_shard:
            pad_mu = mu_per_shard - slab.shape[3]
            slab = np.pad(slab, ((0, 0), (0, 0), (0, 0), (0, pad_mu)), mode="constant")
        mu_off = i * mu_per_shard
        local_psi[:, :, :, mu_off:mu_off + mu_per_shard] = slab

    global_shape = (nk, len(band_indices), nspinor, n_rmu_pad)
    psi_sharding = NamedSharding(mesh_xy, P(None, None, None, axis))
    local_psi_jax = jax.device_put(local_psi)
    psi_global = jax.make_array_from_process_local_data(psi_sharding, local_psi_jax, global_shape)
    if trim and n_rmu_pad > n_rmu:
        psi_global = psi_global[..., :n_rmu]
    return psi_global


def _resolve_munu_reader(
    dset: h5py.Dataset,
    kgrid: Optional[tuple[int, int, int]] = None,
):
    """Resolve ``(n_rmu, n_rnu, nkx, nky, nkz, read_slab, read_q_slab)``.

    Single source of the on-disk layout shim both sharded readers need.
    Handles three layouts:

      * 8-D legacy ``(1, npol, npol, nkx, nky, nkz, μ, ν)`` — kgrid from shape.
      * 6-D transitional ``(1, npol, npol, nq, μ, ν)``.
      * 3-D flat-q ``(nq, μ, ν)``.

    For the 6-D/3-D layouts the caller must pass ``kgrid=(nkx, nky, nkz)`` (or
    the dataset must carry a ``'kgrid'`` attribute).  ``read_slab(mu0, mu1,
    nu0, nu1)`` returns the ``(μ, ν, nkx, nky, nkz)`` slab for that μ/ν block;
    q=0 is the ``(0, 0, 0)`` k-slice.

    ``read_q_slab(q, mu0, mu1, nu0, nu1)`` returns the ``(μ, ν)`` tile at ONE
    flat q index.  ``read_slab`` reads the FULL q axis, so a single-q consumer
    (``_read_vq0_sharded``) that went through ``read_slab(...)[:, :, 0, 0, 0]``
    paid ``nq``× the bytes it used.  Element-for-element the two routes select
    the same numbers.
    """
    # THE q WEDGE, UNFOLDED BEFORE ANY LAYOUT QUESTION.  A q_irr file's q axis
    # is the IBZ, so every shape rule below — and every consumer's
    # ``nkx*nky*nkz == nq`` arithmetic — would be about a tensor that does not
    # exist yet.  A no-op on every legacy/full-BZ dataset, and the unfolded
    # host array slices identically, so the closures need no branch.
    if is_q_wedge(dset):
        dset = restart_munu_full_bz(dset, dset.name.lstrip("/"),
                                    dset.file.filename)
    if dset.ndim == 8:
        n_rmu = int(dset.shape[6])
        n_rnu = int(dset.shape[7])
        nkx, nky, nkz = (int(s) for s in dset.shape[3:6])
        read_slab = lambda mu0, mu1, nu0, nu1: np.transpose(
            dset[0, 0, 0, :, :, :, mu0:mu1, nu0:nu1], (3, 4, 0, 1, 2))

        def read_q_slab(q, mu0, mu1, nu0, nu1, _n=(nkx, nky, nkz)):
            qx, qy, qz = np.unravel_index(int(q), _n)
            return np.asarray(dset[0, 0, 0, qx, qy, qz, mu0:mu1, nu0:nu1])

        return n_rmu, n_rnu, nkx, nky, nkz, read_slab, read_q_slab
    if dset.ndim == 6:
        n_rmu = int(dset.shape[-2])
        n_rnu = int(dset.shape[-1])
        nq = int(dset.shape[3])
        _flat = lambda mu0, mu1, nu0, nu1: np.asarray(
            dset[0, 0, 0, :, mu0:mu1, nu0:nu1])
        _flat_q = lambda q, mu0, mu1, nu0, nu1: np.asarray(
            dset[0, 0, 0, int(q), mu0:mu1, nu0:nu1])
    else:  # 3-D flat-q
        n_rmu = int(dset.shape[-2])
        n_rnu = int(dset.shape[-1])
        nq = int(dset.shape[0])
        _flat = lambda mu0, mu1, nu0, nu1: np.asarray(
            dset[:, mu0:mu1, nu0:nu1])
        _flat_q = lambda q, mu0, mu1, nu0, nu1: np.asarray(
            dset[int(q), mu0:mu1, nu0:nu1])
    if kgrid is not None:
        nkx, nky, nkz = (int(v) for v in kgrid)
    elif 'kgrid' in dset.attrs:
        nkx, nky, nkz = (int(v) for v in dset.attrs['kgrid'])
    else:
        raise ValueError(
            "_resolve_munu_reader: V/W dataset is flat-q but no kgrid was "
            "passed and no 'kgrid' attribute is present; the caller must "
            "resolve (nkx, nky, nkz) (from the top-level 'kgrid' dataset or "
            "the WFN) and pass it in.")
    if nkx * nky * nkz != nq:
        raise ValueError(f"kgrid {nkx}×{nky}×{nkz} ≠ nq={nq}")
    read_slab = (lambda mu0, mu1, nu0, nu1:
        _flat(mu0, mu1, nu0, nu1)
        .reshape(nkx, nky, nkz, mu1 - mu0, nu1 - nu0)
        .transpose(3, 4, 0, 1, 2))
    return n_rmu, n_rnu, nkx, nky, nkz, read_slab, _flat_q


def _read_vq0_sharded(
    dset: h5py.Dataset,
    mu_per_x: int,
    nu_per_y: int,
    mesh_xy: Mesh,
    n_rmu_pad: int,
    dtype: np.dtype = np.complex128,
    trim: bool = True,
    kgrid: Optional[tuple[int, int, int]] = None,
) -> jax.Array:
    local_coords, grid_x, grid_y = _get_local_mesh_coords(
        mesh_xy, origin="bse_loading._read_vq0_sharded")
    local_x, local_y = _get_local_axis_coords(local_coords)
    _assert_local_block(local_coords, local_x, local_y)
    (n_rmu, n_rnu, _nkx, _nky, _nkz,
     _read_slab, read_q_slab) = _resolve_munu_reader(dset, kgrid=kgrid)

    local_mu = mu_per_x * len(local_x)
    local_nu = nu_per_y * len(local_y)
    local_v = np.zeros((local_mu, local_nu), dtype=dtype)

    for ix, x_coord in enumerate(local_x):
        mu_start = x_coord * mu_per_x
        mu_end = min(mu_start + mu_per_x, n_rmu)
        if mu_start >= n_rmu:
            continue
        for iy, y_coord in enumerate(local_y):
            nu_start = y_coord * nu_per_y
            nu_end = min(nu_start + nu_per_y, n_rnu)
            if nu_start >= n_rnu:
                continue
            slab = read_q_slab(0, mu_start, mu_end, nu_start, nu_end)
            if slab.shape[0] < mu_per_x or slab.shape[1] < nu_per_y:
                pad_mu = mu_per_x - slab.shape[0]
                pad_nu = nu_per_y - slab.shape[1]
                slab = np.pad(slab, ((0, pad_mu), (0, pad_nu)), mode="constant")
            mu_off = ix * mu_per_x
            nu_off = iy * nu_per_y
            local_v[mu_off:mu_off + mu_per_x, nu_off:nu_off + nu_per_y] = slab

    global_shape = (n_rmu_pad, n_rmu_pad)
    v_sharding = NamedSharding(mesh_xy, P("x", "y"))
    local_v_jax = jax.device_put(local_v)
    v_global = jax.make_array_from_process_local_data(v_sharding, local_v_jax, global_shape)
    if trim and (global_shape[0] > n_rmu or global_shape[1] > n_rnu):
        v_global = v_global[:n_rmu, :n_rnu]
    return v_global


def _read_wq_sharded(
    dset: h5py.Dataset,
    mu_per_x: int,
    nu_per_y: int,
    mesh_xy: Mesh,
    n_rmu_pad: int,
    dtype: np.dtype = np.complex128,
    trim: bool = True,
    kgrid: Optional[tuple[int, int, int]] = None,
) -> jax.Array:
    local_coords, grid_x, grid_y = _get_local_mesh_coords(
        mesh_xy, origin="bse_loading._read_wq_sharded")
    local_x, local_y = _get_local_axis_coords(local_coords)
    _assert_local_block(local_coords, local_x, local_y)
    (n_rmu, n_rnu, nkx, nky, nkz,
     _read_munu_slab, _read_q_slab) = _resolve_munu_reader(dset, kgrid=kgrid)

    local_mu = mu_per_x * len(local_x)
    local_nu = nu_per_y * len(local_y)
    local_w = np.zeros((local_mu, local_nu, nkx, nky, nkz), dtype=dtype)

    for ix, x_coord in enumerate(local_x):
        mu_start = x_coord * mu_per_x
        mu_end = min(mu_start + mu_per_x, n_rmu)
        if mu_start >= n_rmu:
            continue
        for iy, y_coord in enumerate(local_y):
            nu_start = y_coord * nu_per_y
            nu_end = min(nu_start + nu_per_y, n_rnu)
            if nu_start >= n_rnu:
                continue
            slab = _read_munu_slab(mu_start, mu_end, nu_start, nu_end)
            if slab.shape[0] < mu_per_x or slab.shape[1] < nu_per_y:
                pad_mu = mu_per_x - slab.shape[0]
                pad_nu = nu_per_y - slab.shape[1]
                slab = np.pad(slab, ((0, pad_mu), (0, pad_nu), (0, 0), (0, 0), (0, 0)), mode="constant")
            mu_off = ix * mu_per_x
            nu_off = iy * nu_per_y
            local_w[mu_off:mu_off + mu_per_x, nu_off:nu_off + nu_per_y, :, :, :] = slab

    global_shape = (n_rmu_pad, n_rmu_pad, nkx, nky, nkz)
    w_sharding = NamedSharding(mesh_xy, P("x", "y", None, None, None))
    local_w_jax = jax.device_put(local_w)
    w_global = jax.make_array_from_process_local_data(w_sharding, local_w_jax, global_shape)
    if trim and (global_shape[0] > n_rmu or global_shape[1] > n_rnu):
        w_global = w_global[:n_rmu, :n_rnu, :, :, :]
    return w_global


# ---------------------------------------------------------------------------
# SlabIO transport for the sharded BSE loader
# ---------------------------------------------------------------------------
# The tile geometry below is DELIBERATELY the tile geometry the serial readers
# compute.  The memory contract is unchanged and is the point: per-rank tiles,
# nothing larger than one rank's tile materialised anywhere, no allgather (an
# allgather is a refusal, not a fallback — owner ruling 2026-08-05).  Only who
# moves the bytes changes.  The serial readers are memory-correct and ~17x
# slower (0.17 GiB/s at P=4, CLAIMS 76, against 2.919 GiB/s for the phdf5 tile
# path at 16 ranks, CLAIMS 69): serial POSIX h5py issues one rank's W_q tile as
# nq × μ/px short row-runs, one at a time, with no collective buffering.
#
# SlabIO needs the phdf5 FFI and refuses outright where it is unavailable —
# there is no router and no allgather tier left to hand back — so the serial
# readers stay reachable as the only fallback.


def _bse_slabio_usable(log_fn=print) -> bool:
    """Can this process move the restart tensors through SlabIO?

    A CAPABILITY QUESTION, asked of the stack and not of the input file —
    there is no deck key and no ``input_file`` parameter, because several
    BSE entry points and every unit test call the loader with a bare
    restart path and no deck.

    A ``False`` selects the h5py tile readers above, not a rank-0 gather:
    there is no such thing in this module.  Announced when it declines,
    because a silent 17x is indistinguishable from a hang.
    """
    from file_io.slab_io import probe_availability
    ok, stage, reason = probe_availability()
    if not ok:
        log_fn(f"BSE-sharded: SlabIO unavailable at probe stage '{stage}' "
               f"({reason}); reading the restart with the serial h5py tile "
               f"readers -- memory-correct (per-rank tiles, no allgather) "
               f"and ~17x slower (CLAIMS 76 vs 69).")
    return ok


class _MunuSlabPlan:
    """Where (μ, ν) and q live in a V/W dataset, for a SlabIO request.

    The same three on-disk layouts ``_resolve_munu_reader`` shims, stated
    as (offset, shape, spec) rather than as a closure over ``dset``,
    because that is what ``SlabIO.read_slab`` takes.
    ``_resolve_munu_reader`` stays the single source for the serial path
    and for the layout FACTS (which axes, which order); this only
    re-expresses them.  ``file_io.tagged_arrays._munu_slab_request`` is
    the GW-side statement of the same three layouts, kept separate because
    it never selects a single q and so never needs the kgrid.

    * 8-D legacy ``(1, npol, npol, nkx, nky, nkz, μ, ν)``
    * 6-D transitional ``(1, npol, npol, nq, μ, ν)``
    * 3-D flat-q ``(nq, μ, ν)``
    """

    def __init__(self, ds_shape, kgrid, *, wedge_tables=None):
        self.ds_shape = tuple(int(s) for s in ds_shape)
        self.tables = wedge_tables
        self.ndim = len(self.ds_shape)
        nkx, nky, nkz = (int(v) for v in kgrid)
        self.kgrid = (nkx, nky, nkz)
        self.nq = nkx * nky * nkz
        if self.ndim == 8:
            self.lead = 3
            self.q_axes = 3                      # (nkx, nky, nkz) separately
            self.q_extent = tuple(self.ds_shape[3:6])
            if self.q_extent != self.kgrid:
                raise ValueError(
                    f"_MunuSlabPlan: 8-D dataset k-axes {self.q_extent} "
                    f"disagree with kgrid {self.kgrid}")
        elif self.ndim == 6:
            self.lead = 3
            self.q_axes = 1
            self.q_extent = (int(self.ds_shape[3]),)
        elif self.ndim == 3:
            self.lead = 0
            self.q_axes = 1
            self.q_extent = (int(self.ds_shape[0]),)
        else:
            raise ValueError(
                f"_MunuSlabPlan: unsupported V/W dataset rank {self.ndim} "
                f"(shape {self.ds_shape}); expected 8-D, 6-D or 3-D flat-q.")
        # THE q_irr WEDGE IS READ, NOT REFUSED, since 2026-08-15.
        #
        # This used to raise on any q extent != nq, saying the SlabIO
        # transport "cannot unfold" a wedge.  THE STATED REASON WAS COST AND
        # THE COST WAS NEVER MEASURED; the inability claim was wrong in the
        # first place, and ``file_io.tagged_arrays._unfold_wedge`` says so at
        # length: the unfold is a ``shard_map`` over four ``lax.all_to_all``
        # collectives that take and return exactly the ``P(None,'x','y')``
        # this plan already produces, the collective happens AFTER the read
        # in jax, and nothing asks SlabIO to unfold as a hyperslab offset.
        # The GW leg has read wedges this way in production since 536cbac9.
        #
        # MEASURED 2026-08-15, 4xA100 NVLink, complex128 (JAX_ENABLE_X64=1 --
        # without it JAX silently truncates to complex64 and halves the bytes,
        # which would make the wedge look 2x better than it is):
        #
        #     mu     full-BZ tensor   unfold     B_net        vs 2.919 GiB/s disk
        #     2048   9.000 GiB        0.157 s    57.4 GiB/s   19.7x
        #     1024   2.250 GiB        0.041 s    54.3 GiB/s   18.6x
        #      512   0.562 GiB        0.015 s    38.5 GiB/s   13.2x
        #      256   0.141 GiB        0.007 s    20.7 GiB/s    7.1x
        #
        # The unfold moves C.nq_full.mu^2.16 bytes and the wedge saves reading
        # (nq_full - nq_ibz).mu^2.16, so MU-SQUARED CANCELS and the verdict is
        # a bandwidth ratio: the wedge wins iff B_net > [nq_full /
        # (nq_full - nq_ibz)] . B_disk, a bracket of 1.152 at the 7.6x
        # reduction of the mu=2406 anchor.  It wins by 6-17x at every size.
        # Worth ~23 GB at that anchor (26.7 GB -> ~3.5 GB).
        #
        # SCOPE: single node, NVLink -- which IS this transport's P=4 geometry,
        # but a multi-node BSE crosses Slingshot and was not measured.
        self.is_wedge = False
        if self.q_axes == 1 and int(self.q_extent[0]) != self.nq:
            t = self.tables
            n_ibz = int(self.q_extent[0])
            if (t is not None and int(t.n_q_ibz) == n_ibz
                    and int(t.n_q_full) == self.nq):
                self.is_wedge = True
            else:
                # Still a refusal, but now only for a file that genuinely
                # cannot be reconstructed: a truncated dataset, or a wedge
                # whose own unfold tables are missing or disagree.  A table
                # that reconstructs the tensor must be the table that
                # deconstructed it, so re-deriving it from this run's ``sym``
                # is not offered.
                _why = ("carries no q_irr unfold tables"
                        if t is None else
                        f"carries tables for {int(t.n_q_ibz)} -> "
                        f"{int(t.n_q_full)} q, which do not match")
                # THE ADVICE STAYS ATTACHED TO THE ARM IT APPLIES TO.  A
                # dataset with MORE q rows than the k-grid is not a wedge and
                # no restart_q_storage setting produces or fixes it; naming
                # the key there sends an operator to re-run a GW leg that was
                # never the problem.  Pinned by
                # test_an_oversized_q_extent_does_not_claim_to_be_a_wedge.
                _fix = ""
                if n_ibz < self.nq:
                    _fix = (
                        "  A q-WEDGE file IS readable here when its unfold "
                        "tables are present (restart_q_storage=auto|ibz "
                        "writes them); this one is not reconstructible, so it "
                        "is a truncated or mis-stamped file rather than a "
                        "wedge.")
                raise ValueError(
                    f"_MunuSlabPlan: kgrid {self.kgrid} (nq={self.nq}) "
                    f"disagrees with the dataset q extent {n_ibz}, and the "
                    f"file {_why}.{_fix}")
        self.n_rmu = int(self.ds_shape[-2])
        self.n_rnu = int(self.ds_shape[-1])

    def gamma_wedge_row(self):
        """The wedge row holding flat q=0, or ``None`` if it needs an unfold.

        Γ is its own orbit parent under every point group, so on a wedge the
        single-q ``V_q0`` read is still ONE hyperslab — the same bytes, at a
        different row — and needs no collective at all.  That is asserted
        against the file's OWN tables rather than assumed: the operation
        taking the wedge row to flat q=0 must be a spatial op (not TRS) whose
        centroid permutation is the identity and whose lattice wrap is zero.
        Anything else would need the real unfold, and this returns ``None`` so
        the caller refuses by name instead of reading a rotated block as if it
        were Γ.
        """
        t = self.tables
        if t is None or not self.is_wedge:
            return None
        s_idx = int(np.asarray(t.sym_idx_q)[0])
        row = int(np.asarray(t.irr_idx_q)[0])
        if s_idx >= int(t.n_sym_spatial):
            return None                              # time-reversed: conj
        perm = np.asarray(t.sym_perm)[s_idx]
        wrap = np.asarray(t.L_table)[s_idx]
        if not np.array_equal(perm, np.arange(perm.shape[0])):
            return None
        if np.any(wrap != 0):
            return None
        return row

    def request(self, n_rmu_pad, q_index=None):
        """``(offset, shape, partition_spec)`` for one read_slab call.

        ``q_index`` selects a single FLAT q (the ``V_q0`` case, q=0);
        ``None`` asks for every q, which is what the ``W_q`` consumer
        needs.  Leading layout axes are always taken at index 0 with
        extent 1 — the serial readers' ``[0, 0, 0]``.

        ON A WEDGE the full-q request reads ``n_q_ibz`` rows and
        :func:`_slabio_read_munu` unfolds them; the single-q request is
        remapped to Γ's wedge row by :meth:`gamma_wedge_row`.
        """
        offset = [0] * self.ndim
        shape = [1] * self.lead
        if q_index is None:
            shape.extend(int(v) for v in self.q_extent)
        elif self.q_axes == 3:
            qx, qy, qz = (int(v) for v in
                          np.unravel_index(int(q_index), self.kgrid))
            offset[3], offset[4], offset[5] = qx, qy, qz
            shape.extend((1, 1, 1))
        else:
            row = int(q_index)
            if self.is_wedge:
                if int(q_index) != 0:
                    raise ValueError(
                        f"_MunuSlabPlan: single-q read of flat q={q_index} on "
                        f"a q-WEDGE dataset.  Only q=0 (Γ) is a hyperslab on "
                        f"a wedge, because Γ is its own orbit parent; any "
                        f"other q is a rotated image and must come from the "
                        f"full-q read, which unfolds.")
                g = self.gamma_wedge_row()
                if g is None:
                    raise ValueError(
                        "_MunuSlabPlan: this wedge reaches flat q=0 by a "
                        "non-identity operation (a rotation, a non-zero "
                        "lattice wrap, or time reversal), so V_q0 is not one "
                        "hyperslab here.  Read the full q axis, which "
                        "unfolds, or re-run with restart_q_storage=full.")
                row = g
            offset[self.lead] = row
            shape.append(1)
        shape.extend((int(n_rmu_pad), int(n_rmu_pad)))
        spec = P(*([None] * (len(shape) - 2) + ["x", "y"]))
        return tuple(offset), tuple(shape), spec


def _slabio_read_munu(io, name, plan, mesh_xy, n_rmu_pad, *,
                      q_index=None, dtype=np.complex128):
    """Read a V/W dataset through SlabIO into the BSE consumer layout.

    Returns ``(μ_pad, ν_pad)`` when ``q_index`` is given and
    ``(μ_pad, ν_pad, nkx, nky, nkz)`` otherwise — byte-for-byte the
    shapes ``_read_vq0_sharded`` / ``_read_wq_sharded`` return at
    ``trim=False``, with the same P('x','y',...) sharding.

    The on-disk order is q-major and the consumer wants μ-major, so the
    result is transposed.  That transpose is LOCAL — μ and ν stay on
    ('x', 'y') across it and every other axis is replicated — and its
    ``out_shardings`` is PINNED rather than inferred, because an inferred
    resharding here would be a silent all-to-all on an N_mu²-class object,
    the one thing this loader must never do.
    """
    offset, shape, spec = plan.request(n_rmu_pad, q_index=q_index)
    arr = io.read_slab(name, shape=shape, dtype=dtype, offset=offset,
                       mesh=mesh_xy, partition_spec=spec)
    # THE WEDGE UNFOLD, HERE AND NOWHERE ELSE: after the read, before the
    # μ-major transpose, because ``unfold_isdf_operator`` works q-major and
    # takes/returns the P(None,'x','y') spec the read already produced.  It
    # is the SAME helper the GW leg uses (file_io.tagged_arrays._unfold_wedge)
    # on the SAME tables the file carries, so producer and consumer evaluate
    # one function on one set of inputs rather than agreeing by a property.
    # A full-BZ file has ``plan.tables is None`` and this is a no-op.
    #
    # The single-q route never reaches this: on a wedge it is remapped to Γ's
    # own wedge row by ``plan.request``, which is one hyperslab and no
    # collective.  Applying an unfold to a single-q slab would be wrong —
    # ``unfold_isdf_operator`` wants the whole IBZ axis.
    if q_index is None and getattr(plan, "is_wedge", False):
        from file_io.tagged_arrays import _unfold_wedge
        arr = _unfold_wedge(jnp.reshape(arr, (int(plan.tables.n_q_ibz),
                                              int(n_rmu_pad), int(n_rmu_pad))),
                            plan.tables, n_rmu_pad, mesh_xy)
    nkx, nky, nkz = plan.kgrid
    if q_index is not None:
        out_spec = P("x", "y")
        target = (int(n_rmu_pad), int(n_rmu_pad))
        fn = lambda a: jnp.reshape(a, target)
    else:
        # (nkx, nky, nkz) is the FULL BZ either way: a full-BZ file read it
        # directly, a wedge file was just unfolded to it above.
        out_spec = P("x", "y", None, None, None)
        mid = (nkx, nky, nkz, int(n_rmu_pad), int(n_rmu_pad))
        fn = lambda a: jnp.transpose(jnp.reshape(a, mid), (3, 4, 0, 1, 2))
    return jax.jit(
        fn, out_shardings=NamedSharding(mesh_xy, out_spec))(arr)


def _slabio_read_psi(io, name, psi_shape, band_indices, axis, mesh_xy,
                     n_rmu_pad, *, dtype=np.complex128):
    """Read ψ's band window through SlabIO, μ on ``axis``.

    Returns ``(nk, nb_sel, nspinor, μ_pad)`` with
    ``P(None, None, None, axis)`` — what ``_read_psi_mu_sharded``
    returns at ``trim=False``.

    ``band_indices`` must be a CONTIGUOUS ascending range; the caller's
    two windows (``arange(n_occ-n_val, n_occ)`` and
    ``arange(n_occ, n_occ+n_cond)``) always are.  A hyperslab is an offset
    and an extent, so a gap would silently read the wrong bands — hence a
    refusal.  A caller that ever needs a ragged window must ask for the
    serial reader for the whole load, so that the transport choice stays
    one decision per load.
    """
    b = np.asarray(band_indices)
    if b.size == 0 or not np.array_equal(b, np.arange(int(b[0]),
                                                      int(b[0]) + b.size)):
        raise ValueError(
            f"_slabio_read_psi: band_indices must be a contiguous ascending "
            f"range to become an HDF5 hyperslab; got {b[:8]}… (size {b.size}).")
    nk, _nb, nspinor = (int(psi_shape[0]), int(psi_shape[1]),
                        int(psi_shape[2]))
    offset = (0, int(b[0]), 0, 0)
    shape = (nk, int(b.size), nspinor, int(n_rmu_pad))
    spec = P(None, None, None, axis)
    return io.read_slab(name, shape=shape, dtype=dtype, offset=offset,
                        mesh=mesh_xy, partition_spec=spec)


def _read_bse_tensors(
    restart_file: str,
    *,
    vq_key: str,
    wq_key: str,
    vq_shape,
    wq_shape,
    psi_shape,
    val_indices,
    cond_indices,
    mu_per_x: int,
    nu_per_y: int,
    n_rmu_pad: int,
    mesh_xy: Mesh,
    kgrid,
    load_v_full: bool,
    input_file: Optional[str],
    log_fn=print,
):
    """``(psi_v_X, psi_c_X, V_q0, W_q, V_q_full)`` — one transport decision.

    THE seam between "which bytes" and "how they move".  Both branches
    below return identical global shapes, identical PartitionSpecs,
    identical per-rank tiles and the same elements from the same datasets,
    so the parity bar for the SlabIO branch is BIT EQUALITY, not a
    tolerance: it is an element-SELECTION change, not a reduction-order
    one, and no eps floor applies.

    The transport choice is per-rank but is a pure function of the
    platform probes, so every rank reaches the same answer and the
    collective SlabIO open is well-formed.  Stated because a divergence
    would present as a hang inside ``H5Fopen`` rather than as an error.
    """
    if not _bse_slabio_usable(log_fn=log_fn):
        with h5py.File(restart_file, "r") as f:
            vq_dset = f[vq_key]
            wq_dset = f[wq_key]
            psi_dset = f["psi_full_y"]
            psi_v_X = _read_psi_mu_sharded(psi_dset, val_indices, mu_per_x,
                                           "x", mesh_xy, n_rmu_pad, trim=False)
            psi_c_X = _read_psi_mu_sharded(psi_dset, cond_indices, mu_per_x,
                                           "x", mesh_xy, n_rmu_pad, trim=False)
            V_q0 = _read_vq0_sharded(vq_dset, mu_per_x, nu_per_y, mesh_xy,
                                     n_rmu_pad, trim=False, kgrid=kgrid)
            W_q = _read_wq_sharded(wq_dset, mu_per_x, nu_per_y, mesh_xy,
                                   n_rmu_pad, trim=False, kgrid=kgrid)
            # V read with the SAME (μ, ν, nkx, nky, nkz) reader as W_q, so
            # ``V_q_full[:, :, 0, 0, 0] == V_q0`` (both head-less) — a
            # self-check the finite-q harness asserts.
            V_q_full = (_read_wq_sharded(vq_dset, mu_per_x, nu_per_y, mesh_xy,
                                         n_rmu_pad, trim=False, kgrid=kgrid)
                        if load_v_full else None)
        return psi_v_X, psi_c_X, V_q0, W_q, V_q_full

    from file_io.slab_io import SlabIO
    from file_io.tagged_arrays import _qirr_wedge_tables
    # Kilobytes, read on the serial handle BEFORE the collective SlabIO open,
    # which is the same ordering rule tagged_arrays follows: the two handles
    # must not overlap.  Empty for every full-BZ file, so those files take the
    # byte path they always had.
    with h5py.File(restart_file, "r") as _f:
        _wedge = _qirr_wedge_tables(_f)
    vq_plan = _MunuSlabPlan(vq_shape, kgrid, wedge_tables=_wedge.get(vq_key))
    wq_plan = _MunuSlabPlan(wq_shape, kgrid, wedge_tables=_wedge.get(wq_key))
    if vq_plan.is_wedge or wq_plan.is_wedge:
        log_fn(f"BSE-sharded: q-WEDGE restart, unfolding "
               f"{int((vq_plan.tables or wq_plan.tables).n_q_ibz)} -> "
               f"{int((vq_plan.tables or wq_plan.tables).n_q_full)} q "
               f"through the symmetry service")
    log_fn(f"BSE-sharded: restart tensors via SlabIO "
           f"({os.path.basename(restart_file)})")
    with SlabIO(restart_file, mode="r", mesh=mesh_xy) as io:
        psi_v_X = _slabio_read_psi(io, "psi_full_y", psi_shape, val_indices,
                                   "x", mesh_xy, n_rmu_pad)
        psi_c_X = _slabio_read_psi(io, "psi_full_y", psi_shape, cond_indices,
                                   "x", mesh_xy, n_rmu_pad)
        V_q0 = _slabio_read_munu(io, vq_key, vq_plan, mesh_xy, n_rmu_pad,
                                 q_index=0)
        W_q = _slabio_read_munu(io, wq_key, wq_plan, mesh_xy, n_rmu_pad)
        V_q_full = (_slabio_read_munu(io, vq_key, vq_plan, mesh_xy, n_rmu_pad)
                    if load_v_full else None)
    return psi_v_X, psi_c_X, V_q0, W_q, V_q_full


def load_bse_data_from_restart_sharded(
    restart_file: str,
    n_val: int = 4,
    n_cond: int = 4,
    fermi_energy: float = 0.0,
    mesh_xy: Optional[Mesh] = None,
    pad_bands: bool = True,
    use_nohead: bool = False,
    *,
    input_file: Optional[str] = None,
    cell_volume: Optional[float] = None,
    n_occ: Optional[int] = None,
    inject_head: bool = True,
    load_v_full: bool = False,
    bse_k_grid=None,
    w_head_densify=None,
    w_head_gamma_cell: str = "fine",
    degeneracy_mode: str = DEFAULT_MODE,
    degeneracy_tol_ry: float = DEGENERACY_TOL_RY,
    distrib_la_batched_route: str | None = None,
    htransform_a_band: int | None = None,
) -> dict:
    """Load BSE tensors from canonical gw_jax restart state (psi_full_y/enk_full).

    ``bse_k_grid`` (``(nx,ny,nz)`` / ``"nx ny nz"`` / ``None``) turns on
    fine-grid densification: when set and DIFFERENT from the coarse restart
    grid, the ENTIRE bundle (ψ, QP ε, V_Q exchange, W direct) is interpolated
    onto that grid via :func:`_interpolate_bse_data_to_grid` BEFORE returning,
    so every downstream solver runs on the fine grid transparently.  When
    ``None`` the value is read from the cohsex.in ``bse_k_grid`` key (via
    ``input_file``); unset or == the coarse grid → the coarse bundle is
    returned byte-identically (fast path untouched).

    ``inject_head=False`` returns the head-LESS V_q0 / W_q bodies exactly as
    stored on disk (the rank-1 q=0 head from vhead/whead is NOT added). Used by
    body-vs-body diagnostics such as the W(0) resolvent cross-check, where both
    sides must be head-less (bse_w_exact ``--compare-w0``).

    ``load_v_full=True`` additionally reads the FULL exchange tensor
    ``V_qmunu`` at every q as ``data['V_q_full']`` (μ, ν, nkx, nky, nkz),
    P('x','y',None,None,None) — same layout as ``W_q``.  The finite-q W_q
    resolvent (``bse_w_exact --compare-wq``) picks the tile
    ``V_q_full[:, :, qx, qy, qz]`` (NO head at q≠0) as the screening V and as
    the comparison target ``W_q[...,q] - V_q_full[...,q]``.  Default False keeps
    the q=0 path byte-identical.
    """
    if mesh_xy is None:
        raise ValueError("mesh_xy is required for sharded load")

    with h5py.File(restart_file, "r") as f:
        vq_key = "V_qmunu_nohead" if use_nohead and "V_qmunu_nohead" in f else "V_qmunu"
        if use_nohead and vq_key == "V_qmunu":
            print("Warning: requested --nohead but V_qmunu_nohead not found; using V_qmunu.")
        vq_dset = f[vq_key]
        _refuse_unpersisted(vq_dset, vq_key, restart_file)
        wq_key = None
        if use_nohead and "W0_qmunu_nohead" in f and bool(f["W0_qmunu_nohead"].attrs.get("W0_ready", False)):
            wq_key = "W0_qmunu_nohead"
        elif "W0_qmunu" in f and bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            wq_key = "W0_qmunu"
        # The BSE reads W off the GW restart rather than computing it, so it
        # has no Coulomb config of its own to check the stored policy against;
        # disclosing the stamp is the honest substitute.
        if jax.process_index() == 0:
            from file_io import describe_coulomb_policy_stamp
            print(describe_coulomb_policy_stamp(restart_file))
        # CAPTURED BEFORE THE FALLBACK ALIASES ``wq_key``: after the else
        # branch below, the key alone can no longer say whether the head
        # injection is adding whead to a SCREENED W or to bare V.  Same
        # question, same spelling, as ``_load_ring_subset``'s
        # ``w0_ready = W0_qmunu is not None``.
        w0_ready = wq_key is not None
        if wq_key is not None:
            wq_dset = f[wq_key]
        else:
            if use_nohead:
                print("Warning: requested --nohead but W0_qmunu_nohead not found/ready.")
            print(_BARE_V_FALLBACK_WARNING)
            # NAME the fallback, don't only alias the handle: the readers
            # below are given dataset NAMES (SlabIO opens by name), so a
            # ``None`` key would mean "no W dataset" rather than "W is bare V".
            wq_key = vq_key
            wq_dset = vq_dset
        if "psi_full_y" not in f or "enk_full" not in f:
            raise ValueError(
                f"{restart_file} is missing canonical psi_full_y/enk_full datasets. "
                "Regenerate restart tensors with current gw_jax."
            )
        psi_full_dset = f["psi_full_y"]
        enk_full = np.asarray(f["enk_full"][:])

        # The same three on-disk layouts ``_resolve_munu_reader`` shims, asked
        # here for the GEOMETRY only — this must not unfold a wedge, which is
        # why it reads the dataset's own shape/attrs rather than calling that.
        if vq_dset.ndim == 8:
            nkx, nky, nkz = (int(s) for s in vq_dset.shape[3:6])
            n_rmu = int(vq_dset.shape[6])
            n_rnu = int(vq_dset.shape[7])
        else:
            n_rmu = int(vq_dset.shape[-2])
            n_rnu = int(vq_dset.shape[-1])
            if 'kgrid' in vq_dset.attrs:
                nkx, nky, nkz = (int(v) for v in vq_dset.attrs['kgrid'])
            elif 'kgrid' in f:
                kgrid_vals = np.asarray(f['kgrid'][:]).reshape(-1)
                nkx, nky, nkz = (int(kgrid_vals[0]), int(kgrid_vals[1]),
                                  int(kgrid_vals[2]))
            elif input_file is not None:
                from ffi import _services
                _services.ensure_on_path()
                from wfn_loader import WfnLoader
                _wfn = WfnLoader(_parse_wfn_path(input_file))
                nkx, nky, nkz = (int(_wfn.kgrid[0]), int(_wfn.kgrid[1]),
                                  int(_wfn.kgrid[2]))
            else:
                raise ValueError(
                    "load_bse_data_from_restart_sharded: flat-q V_qmunu "
                    "needs kgrid; restart file has no top-level 'kgrid' "
                    "dataset and no input_file passed to fall back to WFN.")
        if n_rmu != n_rnu:
            raise ValueError("Expected square μ/ν dimensions in V_qmunu")

        n_occ = resolve_n_occ(
            enk_full, n_occ=n_occ, input_file=input_file,
            fermi_energy=fermi_energy if fermi_energy != 0.0 else None,
        )
        nb_total = int(enk_full.shape[1])
        n_val_available = int(n_occ)
        n_cond_available = nb_total - int(n_occ)
        if n_val > n_val_available:
            print(f"Warning: requested {n_val} valence bands but only {n_val_available} available; using {n_val_available}")
        if n_cond > n_cond_available:
            print(f"Warning: requested {n_cond} conduction bands but only {n_cond_available} available; using {n_cond_available}")
        n_val = min(n_val, n_val_available)
        n_cond = min(n_cond, n_cond_available)
        if n_val == 0 or n_cond == 0:
            raise ValueError(
                f"No valence ({n_val_available}) or conduction ({n_cond_available}) bands "
                f"resolved (n_occ={n_occ}, total={nb_total})."
            )
        # THE degeneracy choke point, placed before the index arrays exist and
        # therefore before the ψ hyperslab read is sized: a snap here resizes
        # the read, where a guard after it could only complain.
        n_val, n_cond = resolve_band_window(
            enk_full, n_occ, n_val, n_cond,
            tol_ry=degeneracy_tol_ry, mode=degeneracy_mode,
            where="load_bse_data_from_restart_sharded", log=_log0)
        val_indices = np.arange(n_occ - n_val, n_occ)
        cond_indices = np.arange(n_occ, n_occ + n_cond)

        eps_v = jnp.asarray(enk_full[:, val_indices])
        eps_c = jnp.asarray(enk_full[:, cond_indices])

        _, grid_x, grid_y = _get_local_mesh_coords(
            mesh_xy, origin="bse_loading.load_bse_data_from_restart_sharded")
        # Disk stores the LOGICAL μ extent; re-pad to the ONE in-memory
        # convention (mesh-product round-up, runtime.padding).
        n_rmu_pad = padded_mu_extent(n_rmu, grid_x * grid_y)
        mu_per_x = n_rmu_pad // grid_x
        nu_per_y = n_rmu_pad // grid_y

        # Shapes are all the tensor readers need from the open handle; the
        # bytes move AFTER this block closes the file, because SlabIO reopens
        # the same path under collective MPI-IO.
        vq_shape = tuple(int(s) for s in vq_dset.shape)
        wq_shape = tuple(int(s) for s in wq_dset.shape)
        psi_shape = tuple(int(s) for s in psi_full_dset.shape)

        # ── q=0 head: load G0_mu_nu, dual-shard X/Y, inject as rank-1 ────
        # On the (μ,ν)-sharded V_q0 and W_q tensors, the rank-1 update
        # ``conj(g0_X[μ_loc]) * g0_Y[ν_loc]`` is local on every proc when
        # g0 is held in TWO copies — one P("x") on μ, one P("y") on ν.
        # Source priority: cohsex.in overrides → restart vhead/whead.
        if "G0_mu_nu" in f:
            G0_full = np.asarray(f["G0_mu_nu"][:], dtype=np.complex128)
            if G0_full.size < n_rmu_pad:
                G0_pad = np.zeros((n_rmu_pad,), dtype=np.complex128)
                G0_pad[:G0_full.size] = G0_full
                G0_full = G0_pad
            # G0_full is host numpy read identically on every rank; a plain
            # device_put would fire the hidden assert_equal all-gather, twice
            # (AA.1).  LORRAX_CHECK_REPLICA=1 re-arms it.
            from common.collectives import device_put_process_local
            g0_X = device_put_process_local(G0_full,
                                            NamedSharding(mesh_xy, P("x")))
            g0_Y = device_put_process_local(G0_full,
                                            NamedSharding(mesh_xy, P("y")))
            vhead_restart = (complex(f["vhead"][()])
                             if "vhead" in f else None)
            whead_restart = (jnp.asarray(f["whead"][:], dtype=jnp.complex128)
                             if "whead" in f else None)
        else:
            g0_X = g0_Y = None
            vhead_restart = whead_restart = None

    # ── The big tensors, on whichever transport this stack has ──────────
    # Outside the h5py handle on purpose (see the shape capture above), and
    # ONE decision for all four: mixing transports would open the SlabIO file
    # twice and make any throughput number unattributable.
    psi_v_X, psi_c_X, V_q0, W_q, V_q_full = _read_bse_tensors(
        restart_file, vq_key=vq_key, wq_key=wq_key, vq_shape=vq_shape,
        wq_shape=wq_shape, psi_shape=psi_shape, val_indices=val_indices,
        cond_indices=cond_indices, mu_per_x=mu_per_x, nu_per_y=nu_per_y,
        n_rmu_pad=n_rmu_pad, mesh_xy=mesh_xy, kgrid=(nkx, nky, nkz),
        load_v_full=load_v_full, input_file=input_file)

    if pad_bands:
        # ψ pad = 0 (bilinear ⇒ inert, and it is what decouples the pad
        # block); ε pad = signed sentinel (diagonal of a diagonalisation).
        _v = pad_axis(psi_v_X, grid_y, axis=1)
        _c = pad_axis(psi_c_X, grid_x, axis=1)
        # ``.padded`` by name: n_val_pad / n_cond_pad are the MESH-ROUNDED
        # extents (bse_ring_comm asserts ``n_cond_pad % px == 0``), never
        # the logical band counts.
        psi_v_X, n_val_pad = _v.array, _v.padded
        psi_c_X, n_cond_pad = _c.array, _c.padded
        eps_v = pad_axis(eps_v, grid_y, axis=1, fill=-PAD_EPS_GUARD_RY).array
        eps_c = pad_axis(eps_c, grid_x, axis=1, fill=PAD_EPS_GUARD_RY).array
    else:
        n_val_pad = int(psi_v_X.shape[1])
        n_cond_pad = int(psi_c_X.shape[1])
    psi_v_Y = jax.lax.with_sharding_constraint(psi_v_X, NamedSharding(mesh_xy, P(None, None, None, "y")))
    psi_c_Y = jax.lax.with_sharding_constraint(psi_c_X, NamedSharding(mesh_xy, P(None, None, None, "y")))

    # ── Is a coarse→fine densification pending?  Resolved HERE, before the
    # head injection, because C1 (the default) hands the densifier the
    # head-EXCLUDED body and re-attaches the head per fine q afterwards, so on
    # a densifying run the rank-1 whead must NOT go on now.
    fine_grid = _resolve_bse_k_grid(bse_k_grid, input_file)
    densify_pending = (fine_grid is not None
                       and fine_grid != (nkx, nky, nkz))
    w_head_mode = resolve_w_head_densify(
        w_head_densify, _read_lorrax_input_quietly(input_file))
    defer_whead = densify_pending and w_head_mode == "c1"
    head_channel = None

    if g0_X is not None and inject_head:
        vhead, whead, cell_volume, head_src = _resolve_head_params(
            input_file, vhead_restart, whead_restart, cell_volume)

        if cell_volume is not None and (vhead is not None or whead is not None):
            # whead goes on a SCREENED W or nowhere — same gate, same helper,
            # as the single-device loader.
            V_q0, W_q, head_str = _inject_q0_head(
                V_q0, W_q, g0_X, g0_Y, vhead, whead, cell_volume,
                w0_ready=w0_ready, defer_whead=defer_whead)
            print(f"BSE-sharded: q=0 head injected (rank-1, dual-sharded G0, "
                  f"V_cell={cell_volume:.2f}): {head_str} "
                  f"[source: {head_src}]")
            if defer_whead and w0_ready and whead is not None:
                head_channel = {
                    "whead": float(complex(whead[0]).real),
                    "cell_volume": float(cell_volume),
                    "gamma_cell": w_head_gamma_cell,
                }
        else:
            # G0_mu_nu is present and inject_head is True, but the head cannot
            # be built.  The loader has no wfn/meta/sym/S_cart with which to
            # recompute <v>_mBZ, so the only honest move is to warn loudly and
            # name the fix; silence here leaves a head-LESS q=0 tile no trace.
            import warnings
            reasons = []
            if cell_volume is None:
                reasons.append("cell_volume unknown (WFN not passed)")
            if vhead is None and whead is None:
                reasons.append("vhead/whead both unresolved "
                               "(no cohsex.in override, no restart scalars)")
            msg = (
                "BSE q=0 head NOT injected though G0_mu_nu is present and "
                f"inject_head=True: {', '.join(reasons)}.  The q=0 exchange "
                "tile is HEAD-LESS (missing the rank-1 (vhead/V_cell)·conj(g0)g0 "
                "term) — exciton binding energies will be under-bound at the "
                "zone centre.  FIX: add ``vhead``/``whead_0freq`` to cohsex.in, "
                "or write ``vhead``/``whead`` datasets into the restart.  "
                "(Recompute-from-WFN is not available at the loader.)")
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            print(f"BSE-sharded: [WARN] {msg}")

    # Exchange pair amplitudes M(k,c,v,μ) = Σ_s conj(ψ_c) ψ_v, hoisted so the
    # per-iteration matvec receives them instead of rebuilding them from ψ:
    # the V-term decode (M_X, μ on x) and encode (M_Y, ν on y) vertices.
    # Peak-neutral; the between-matvec floor rises by ~2·M/p.
    M_X = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_X, psi_v_X),
        NamedSharding(mesh_xy, P(None, None, None, "x")))
    M_Y = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_Y, psi_v_Y),
        NamedSharding(mesh_xy, P(None, None, None, "y")))

    data = {
        "psi_c_X": psi_c_X,
        "psi_c_Y": psi_c_Y,
        "psi_v_X": psi_v_X,
        "psi_v_Y": psi_v_Y,
        "M_X": M_X,
        "M_Y": M_Y,
        "eps_c": eps_c,
        "eps_v": eps_v,
        "W_q": W_q,
        "V_q0": V_q0,
        "V_q_full": V_q_full,
        "g0_X": g0_X,
        "g0_Y": g0_Y,
        "nkx": nkx,
        "nky": nky,
        "nkz": nkz,
        "n_rmu": n_rmu,
        "n_rmu_pad": n_rmu_pad,
        "n_val": n_val,
        "n_cond": n_cond,
        "n_val_pad": n_val_pad,
        "n_cond_pad": n_cond_pad,
        "fermi_energy": fermi_energy,
    }

    # ── bse_k_grid coarse→fine densification ─────────────────────────────
    # Unset or == the coarse grid → the coarse bundle above is returned
    # UNTOUCHED.  That is the on-grid byte-identity guarantee, and it is
    # structural rather than measured.
    if densify_pending:
        if input_file is None:
            raise ValueError(
                "bse_k_grid densification needs input_file (cohsex.in) to run "
                "the htransform ψ/ε and vq_interp V_Q interpolation.")
        if w_head_mode == "legacy":
            print("BSE-sharded: [WARN] w_head_densify = legacy — W's Γ head "
                  "rides through the trigonometric interpolant as a Kronecker "
                  "delta.  That is the documented defect (gw.head_densify): "
                  "the interpolant of a delta is a Dirichlet kernel, so a "
                  "fraction of the head's ~10^3 meV prefactor is deposited at "
                  "fine q that should carry none of it, and the 1/q² rise "
                  "inside the coarse Γ cell is missing entirely.  This arm "
                  "exists to price the repair, not to be run for physics.")
        data = _interpolate_bse_data_to_grid(
            data, fine_grid, restart_file, input_file, mesh_xy,
            head_channel=head_channel,
            distrib_la_batched_route=distrib_la_batched_route,
            htransform_a_band=htransform_a_band)
    return data


def _find_restart_file(input_file: str) -> str:
    """Locate ``isdf_tensors_<n_rmu>.h5`` — loudly when the choice is ambiguous.

    ``isdf_tensors_*.h5`` is namespaced by centroid count, so a run
    directory used for a μ-sweep holds several, and lexicographic order is
    meaningless across them (``isdf_tensors_1194.h5`` sorts before
    ``isdf_tensors_276.h5``).  Picking the wrong one is SILENT: the BSE
    kernel is built from a different ISDF basis than Σ was, every stage
    reports success, and the exciton spectrum is quietly wrong.  So the
    NEWEST wins and the ambiguity is named in the log.
    ``gw.downfold_run.resolve_restart_file`` refuses the same ambiguity
    instead, deliberately — see its docstring for why the two differ.
    """
    input_dir = os.path.dirname(os.path.abspath(input_file))
    candidates = []
    candidates.extend(sorted(glob.glob(os.path.join(input_dir, "tmp", "isdf_tensors_*.h5"))))
    candidates.extend(sorted(glob.glob(os.path.join(input_dir, "isdf_tensors_*.h5"))))
    candidates = [p for p in candidates if os.path.exists(p)]
    if not candidates:
        raise FileNotFoundError(f"Could not find canonical restart file isdf_tensors_*.h5 in {input_dir}")
    if len(candidates) > 1:
        chosen = max(candidates, key=os.path.getmtime)
        others = ", ".join(os.path.basename(p) for p in candidates
                           if p != chosen)
        print(
            f"  *** LORRAX SANITY: {len(candidates)} restart tensor files "
            f"match isdf_tensors_*.h5 in {input_dir} — they hold DIFFERENT "
            f"ISDF bases (different centroid counts).  Using the newest, "
            f"{os.path.basename(chosen)}; ignoring: {others}.  If that is "
            f"not the basis this BSE's eqp/Σ inputs were built with, the "
            f"exciton spectrum will be wrong with no other symptom — pass "
            f"the file explicitly or clean the run directory. ***",
            flush=True)
        return chosen
    return candidates[0]


def _load_ring_subset(
    restart_file: str,
    n_val: int,
    n_cond: int,
    px: int,
    py: int,
    eqp_file: Optional[str] = None,
    n_occ: Optional[int] = None,
    input_file: Optional[str] = None,
    degeneracy_mode: str = DEFAULT_MODE,
    degeneracy_tol_ry: float = DEGENERACY_TOL_RY,
) -> dict:
    """Load a single-device BSE subset from canonical gw_jax restart state.

    FULL-FILE reader by construction: V_qmunu, W0_qmunu and psi_full_y are
    materialised whole, which is what its two callers need (the
    ``_preview_lanczos`` 1-device branch and the ring-matvec correctness
    check's single-device reference).  A multi-process run must NEVER come
    through here — every rank would h5py-read the full tensors before any
    sharding, and single-device arrays cannot represent one logical object
    across processes — so the P>1 guard below turns a future misrouting
    into a refusal rather than a silent per-rank full read.
    :func:`load_bse_data_from_restart_sharded` is the P>1 counterpart.
    """
    import jax as _jax
    if int(_jax.process_count()) > 1:
        raise RuntimeError(
            "_load_ring_subset is the single-device FULL-FILE reader; on "
            f"{int(_jax.process_count())} processes every rank would read "
            "the whole V_qmunu/W0_qmunu/psi_full_y. Use "
            "load_bse_data_from_restart_sharded (sharded hyperslab reads) "
            "as the P>1 preview/ring paths already do.")
    with h5py.File(restart_file, "r") as f:
        _refuse_unpersisted(f["V_qmunu"], "V_qmunu", restart_file)
        V_qmunu = jnp.asarray(
            restart_munu_full_bz(f["V_qmunu"], "V_qmunu", restart_file))
        from file_io import describe_coulomb_policy_stamp
        print(describe_coulomb_policy_stamp(restart_file))
        if "W0_qmunu" in f and bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            W0_qmunu = jnp.asarray(
                restart_munu_full_bz(f["W0_qmunu"], "W0_qmunu", restart_file))
        else:
            W0_qmunu = None
        # G0_mu_nu = ζ(q=0, μ, G=0), the rank-1 head projector.
        G0_mu_nu = jnp.asarray(f["G0_mu_nu"][:]) if "G0_mu_nu" in f else None
        vhead_restart = complex(f["vhead"][()]) if "vhead" in f else None
        whead_restart = (jnp.asarray(f["whead"][:], dtype=jnp.complex128)
                         if "whead" in f else None)
        if "psi_full_y" not in f or "enk_full" not in f:
            raise ValueError(
                f"{restart_file} is missing canonical psi_full_y/enk_full datasets. "
                "Regenerate restart tensors with current gw_jax."
            )
        psi_full = jnp.asarray(f["psi_full_y"][:])
        enk_full_np = np.asarray(f["enk_full"][:])

    if eqp_file is not None:
        enk_full_np = apply_eqp_corrections(enk_full_np, eqp_file, input_file=input_file)

    enk_full = jnp.asarray(enk_full_np)

    # The same three on-disk layouts ``_resolve_munu_reader`` shims, restated
    # for arrays that are already materialised: 8-D carries the kgrid in its
    # own shape, 6-D and 3-D flat-q need it from the WFN.
    if V_qmunu.ndim == 8:
        nkx, nky, nkz = V_qmunu.shape[3:6]
        n_rmu = int(V_qmunu.shape[-1])
        V_qmunu_flat = jnp.asarray(V_qmunu)[0, 0, 0].reshape(
            -1, n_rmu, n_rmu)
        if W0_qmunu is not None:
            W0_qmunu_flat = jnp.asarray(W0_qmunu)[0, 0, 0].reshape(
                -1, n_rmu, n_rmu)
        else:
            W0_qmunu_flat = None
    else:
        if input_file is None:
            raise ValueError(
                "BSE: flat-q V_qmunu requires input_file to resolve "
                "kgrid from the WFN")
        from ffi import _services
        _services.ensure_on_path()
        from wfn_loader import WfnLoader
        _wfn = WfnLoader(_parse_wfn_path(input_file))
        nkx, nky, nkz = (int(_wfn.kgrid[0]), int(_wfn.kgrid[1]), int(_wfn.kgrid[2]))
        n_rmu = int(V_qmunu.shape[-1])
        if V_qmunu.ndim == 6:
            V_qmunu_flat = jnp.asarray(V_qmunu)[0, 0, 0]
            W0_qmunu_flat = (jnp.asarray(W0_qmunu)[0, 0, 0]
                              if W0_qmunu is not None else None)
        else:
            V_qmunu_flat = jnp.asarray(V_qmunu)  # already (nq, μ, μ)
            W0_qmunu_flat = (jnp.asarray(W0_qmunu)
                              if W0_qmunu is not None else None)
    V_qmunu = V_qmunu_flat
    W0_qmunu = W0_qmunu_flat
    nk = nkx * nky * nkz
    # Same μ re-pad convention as the sharded loader above.
    n_rmu_pad = padded_mu_extent(n_rmu, px * py)

    n_bands_total = enk_full.shape[1]
    n_occ = resolve_n_occ(enk_full_np, n_occ=n_occ, input_file=input_file)
    n_val_available = int(n_occ)
    n_cond_available = n_bands_total - n_occ
    if n_val > n_val_available:
        print(f"Warning: requested {n_val} valence bands but only {n_val_available} available; using {n_val_available}")
    if n_cond > n_cond_available:
        print(f"Warning: requested {n_cond} conduction bands but only {n_cond_available} available; using {n_cond_available}")
    n_val = min(n_val, n_val_available)
    n_cond = min(n_cond, n_cond_available)
    # Same guard and same defaults as the sharded loader, so the two routes
    # cannot disagree about what window a given (n_val, n_cond) request means.
    n_val, n_cond = resolve_band_window(
        enk_full_np, n_occ, n_val, n_cond,
        tol_ry=degeneracy_tol_ry, mode=degeneracy_mode,
        where="_load_ring_subset", log=_log0)
    val_indices = jnp.arange(n_occ - n_val, n_occ)
    cond_indices = jnp.arange(n_occ, n_occ + n_cond)

    psi_v = psi_full[:, val_indices, :, :]
    psi_c = psi_full[:, cond_indices, :, :]
    eps_v = enk_full[:, val_indices]
    eps_c = enk_full[:, cond_indices]

    psi_v = _pad_last_axis(psi_v, n_rmu_pad)
    psi_c = _pad_last_axis(psi_c, n_rmu_pad)
    _v = pad_axis(psi_v, py, axis=1)
    _c = pad_axis(psi_c, px, axis=1)
    # ``.padded`` by name -- see the note at the sharded loader seam.
    psi_v, n_val_pad = _v.array, _v.padded
    psi_c, n_cond_pad = _c.array, _c.padded
    eps_v = pad_axis(eps_v, py, axis=1, fill=-PAD_EPS_GUARD_RY).array
    eps_c = pad_axis(eps_c, px, axis=1, fill=PAD_EPS_GUARD_RY).array
    V_q0 = V_qmunu[0]                       # flat-q (nq, μ, μ) post-shim
    V_q0 = _pad_last_two_axes(V_q0, n_rmu_pad)
    if W0_qmunu is not None:
        W_src = W0_qmunu
    else:
        print(_BARE_V_FALLBACK_WARNING)
        W_src = V_qmunu
    # THE ONE PLACE the 3-D-k form materialises inside BSE — everything else
    # keeps flat-q — because the downstream machinery consumes W_q as
    # ``(μ, μ, nkx, nky, nkz)``.
    W_q = W_src.reshape(nkx, nky, nkz, n_rmu, n_rmu).transpose(3, 4, 0, 1, 2)
    W_q = _pad_first_two_axes(W_q, n_rmu_pad)

    # ── q=0 head injection (rank-1 in (μ,ν) ISDF basis) ──────────────────
    # AFTER the layout shim, so it operates on the normalised q=0 slice.
    # ``compute_vcoul`` zeroes the G=G'=0 element of v(q=0) and this
    # reinstates the mini-BZ-averaged head from G0_mu_nu = ζ(0,μ,G=0).
    # Single device, so G0 serves as both the μ- and the ν-axis copy.
    if G0_mu_nu is not None:
        vhead, whead, cell_volume, head_src = _resolve_head_params(
            input_file, vhead_restart, whead_restart)
        if cell_volume is None:
            print("BSE: head injection skipped — could not resolve cell_volume "
                  "(input_file required)")
        elif vhead is not None or whead is not None:
            g0_pad = _pad_last_axis(G0_mu_nu, n_rmu_pad)
            w0_ready = W0_qmunu is not None
            V_q0, W_q, head_str = _inject_q0_head(
                V_q0, W_q, g0_pad, g0_pad, vhead, whead, cell_volume,
                w0_ready=w0_ready)
            print(f"BSE: q=0 head injected (rank-1 in μν, V_cell={cell_volume:.2f}): "
                  f"{head_str} [source: {head_src}]")

    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (1, n_cond_pad, n_val_pad, nk)) + 1j * jax.random.normal(
        key, (1, n_cond_pad, n_val_pad, nk)
    )

    # The RESOLVED window travels with the bundle under the same four names
    # ``load_bse_data_from_restart_sharded`` uses: ``n_val``/``n_cond`` are
    # post-clamp AND post-snap, ``*_pad`` are the mesh-rounded extents the
    # ψ/ε arrays actually carry.  A caller re-deriving either from
    # ``psi_c.shape[1]`` gets the padded number.
    return {
        "psi_c": psi_c,
        "psi_v": psi_v,
        "eps_c": eps_c,
        "eps_v": eps_v,
        "W_q": W_q,
        "V_q0": V_q0,
        "X": X,
        "n_val": n_val,
        "n_cond": n_cond,
        "n_val_pad": n_val_pad,
        "n_cond_pad": n_cond_pad,
        "nkx": nkx,
        "nky": nky,
        "nkz": nkz,
        "nk": nk,
        "n_rmu_pad": n_rmu_pad,
    }
