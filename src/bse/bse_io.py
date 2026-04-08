"""I/O and padding utilities for BSE ISDF data."""
from __future__ import annotations

import glob
import math
import os
from types import SimpleNamespace
from typing import Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


class BSEData(SimpleNamespace):
    """Container for BSE calculation data."""
    pass


def write_eigenvectors_stream(
    output_file: str,
    eigenvalues: jax.Array,
    eigenvectors: jax.Array,
    n_val: int,
    n_cond: int,
    nkx: int,
    nky: int,
    nkz: int,
    n_write: int,
) -> None:
    from .write_eigenvectors import generate_kpts_grid

    eigenvalues = np.asarray(jax.device_get(eigenvalues[:n_write]))
    kpts = generate_kpts_grid(nkx, nky, nkz)
    nk = kpts.shape[0]
    ns = 1
    nQ = 1
    flavor = 2
    spin_kernel = 3
    bse_hamiltonian_size = ns * nk * n_val * n_cond
    evec_sz = bse_hamiltonian_size

    kpts_fortran = kpts.T.copy()
    exciton_Q_shifts = np.zeros((1, 3), dtype=np.float64)

    with h5py.File(output_file, "w") as f:
        f.create_group("mf_header")
        f.create_group("eps_header")
        f.create_group("bse_header")

        exciton_header = f.create_group("exciton_header")
        exciton_header.create_dataset("version", data=1)
        exciton_header.create_dataset("flavor", data=flavor)

        params = exciton_header.create_group("params")
        params.create_dataset("bse_hamiltonian_size", data=bse_hamiltonian_size)
        params.create_dataset("evec_sz", data=evec_sz)
        params.create_dataset("spin_kernel", data=spin_kernel)
        params.create_dataset("nevecs", data=n_write)
        params.create_dataset("ns", data=ns)
        params.create_dataset("nc", data=n_cond)
        params.create_dataset("nv", data=n_val)
        params.create_dataset("use_tda", data=1)

        kpoints = exciton_header.create_group("kpoints")
        kpoints.create_dataset("nk", data=nk)
        kpoints.create_dataset("kpts", data=kpts_fortran)
        kpoints.create_dataset("nQ", data=nQ)
        kpoints.create_dataset("exciton_Q_shifts", data=exciton_Q_shifts.T)

        exciton_data = f.create_group("exciton_data")
        exciton_data.create_dataset("eigenvalues", data=eigenvalues)
        evec_dset = exciton_data.create_dataset(
            "eigenvectors",
            shape=(1, n_write, nk, n_cond, n_val, ns, 2),
            dtype=np.float64,
        )

        for i in range(n_write):
            vec = jax.device_get(eigenvectors[i])
            vec = np.transpose(vec, (2, 0, 1))  # (nk, nc, nv)
            vec = vec[..., None]  # (nk, nc, nv, ns)
            evec_dset[0, i, :, :, :, :, 0] = vec.real
            evec_dset[0, i, :, :, :, :, 1] = vec.imag

    print(f"Wrote {n_write} eigenvectors to {output_file}")


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


def _pad_axis_to_multiple(x: jax.Array, axis: int, multiple: int) -> tuple[jax.Array, int]:
    size = x.shape[axis]
    pad = (-size) % multiple
    if pad == 0:
        return x, size
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad)
    return jnp.pad(x, pad_width, mode="constant"), size


def _get_local_mesh_coords(mesh_xy: Mesh) -> tuple[list[tuple[int, int]], int, int]:
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
    local_coords, grid_x, grid_y = _get_local_mesh_coords(mesh_xy)
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


def _read_vq0_sharded(
    dset: h5py.Dataset,
    mu_per_x: int,
    nu_per_y: int,
    mesh_xy: Mesh,
    n_rmu_pad: int,
    dtype: np.dtype = np.complex128,
    trim: bool = True,
) -> jax.Array:
    local_coords, grid_x, grid_y = _get_local_mesh_coords(mesh_xy)
    local_x, local_y = _get_local_axis_coords(local_coords)
    _assert_local_block(local_coords, local_x, local_y)
    n_rmu = dset.shape[6]
    n_rnu = dset.shape[7]

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
            slab = dset[0, 0, 0, 0, 0, 0, mu_start:mu_end, nu_start:nu_end]
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
) -> jax.Array:
    local_coords, grid_x, grid_y = _get_local_mesh_coords(mesh_xy)
    local_x, local_y = _get_local_axis_coords(local_coords)
    _assert_local_block(local_coords, local_x, local_y)
    n_rmu = dset.shape[6]
    n_rnu = dset.shape[7]
    nkx, nky, nkz = dset.shape[3:6]

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
            slab = dset[0, 0, 0, :, :, :, mu_start:mu_end, nu_start:nu_end]
            slab = np.transpose(slab, (3, 4, 0, 1, 2))
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


def load_bse_data_from_restart_sharded(
    restart_file: str,
    n_val: int = 4,
    n_cond: int = 4,
    fermi_energy: float = 0.0,
    mesh_xy: Optional[Mesh] = None,
    pad_bands: bool = True,
) -> dict:
    """Load BSE tensors from canonical gw_jax restart state (psi_full_y/enk_full)."""
    if mesh_xy is None:
        raise ValueError("mesh_xy is required for sharded load")

    with h5py.File(restart_file, "r") as f:
        vq_dset = f["V_qmunu"]
        if "W0_qmunu" in f and bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            wq_dset = f["W0_qmunu"]
        else:
            wq_dset = vq_dset
        if "psi_full_y" not in f or "enk_full" not in f:
            raise ValueError(
                f"{restart_file} is missing canonical psi_full_y/enk_full datasets. "
                "Regenerate restart tensors with current gw_jax."
            )
        psi_full_dset = f["psi_full_y"]
        enk_full = np.asarray(f["enk_full"][:])

        nkx, nky, nkz = vq_dset.shape[3:6]
        n_rmu = int(vq_dset.shape[6])
        n_rnu = int(vq_dset.shape[7])
        if n_rmu != n_rnu:
            raise ValueError("Expected square μ/ν dimensions in V_qmunu")

        mean_enk_full = np.mean(enk_full, axis=0)
        val_mask = mean_enk_full < fermi_energy
        cond_mask = mean_enk_full > fermi_energy
        n_val_available = int(np.sum(val_mask))
        n_cond_available = int(np.sum(cond_mask))
        if n_val > n_val_available:
            print(f"Warning: requested {n_val} valence bands but only {n_val_available} available; using {n_val_available}")
        if n_cond > n_cond_available:
            print(f"Warning: requested {n_cond} conduction bands but only {n_cond_available} available; using {n_cond_available}")
        n_val = min(n_val, n_val_available)
        n_cond = min(n_cond, n_cond_available)
        if n_val == 0 or n_cond == 0:
            raise ValueError("No valence or conduction bands found for given Fermi energy")
        val_indices = np.argsort(np.where(val_mask, mean_enk_full, -np.inf))[-n_val:]
        cond_indices = np.argsort(np.where(cond_mask, mean_enk_full, np.inf))[:n_cond]

        eps_v = jnp.asarray(enk_full[:, val_indices])
        eps_c = jnp.asarray(enk_full[:, cond_indices])

        _, grid_x, grid_y = _get_local_mesh_coords(mesh_xy)
        lcm_xy = math.lcm(grid_x, grid_y)
        n_rmu_pad = ((n_rmu + lcm_xy - 1) // lcm_xy) * lcm_xy
        mu_per_x = n_rmu_pad // grid_x
        nu_per_y = n_rmu_pad // grid_y

        psi_v_X = _read_psi_mu_sharded(psi_full_dset, val_indices, mu_per_x, "x", mesh_xy, n_rmu_pad, trim=False)
        psi_c_X = _read_psi_mu_sharded(psi_full_dset, cond_indices, mu_per_x, "x", mesh_xy, n_rmu_pad, trim=False)

        if pad_bands:
            psi_v_X, n_val_pad = _pad_axis_to_multiple(psi_v_X, axis=1, multiple=grid_y)
            psi_c_X, n_cond_pad = _pad_axis_to_multiple(psi_c_X, axis=1, multiple=grid_x)
            eps_v, _ = _pad_axis_to_multiple(eps_v, axis=1, multiple=grid_y)
            eps_c, _ = _pad_axis_to_multiple(eps_c, axis=1, multiple=grid_x)
        else:
            n_val_pad = int(psi_v_X.shape[1])
            n_cond_pad = int(psi_c_X.shape[1])
        psi_v_Y = jax.lax.with_sharding_constraint(psi_v_X, NamedSharding(mesh_xy, P(None, None, None, "y")))
        psi_c_Y = jax.lax.with_sharding_constraint(psi_c_X, NamedSharding(mesh_xy, P(None, None, None, "y")))

        V_q0 = _read_vq0_sharded(vq_dset, mu_per_x, nu_per_y, mesh_xy, n_rmu_pad, trim=False)
        W_q = _read_wq_sharded(wq_dset, mu_per_x, nu_per_y, mesh_xy, n_rmu_pad, trim=False)

    return {
        "psi_c_X": psi_c_X,
        "psi_c_Y": psi_c_Y,
        "psi_v_X": psi_v_X,
        "psi_v_Y": psi_v_Y,
        "eps_c": eps_c,
        "eps_v": eps_v,
        "W_q": W_q,
        "V_q0": V_q0,
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


def read_bgw_eqp(eqp_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a BerkeleyGW eqp1.dat file.

    BGW eqp1.dat format (per k-point):
        kx  ky  kz  n_bands
        spin  band_index  E_DFT_eV  E_QP_eV
        ...

    Returns:
        kpts_ibz: (n_kpts_ibz, 3) crystal coordinates
        e_dft_ibz: (n_kpts_ibz, max_band) DFT energies in eV
        e_qp_ibz: (n_kpts_ibz, max_band) QP energies in eV
    """
    kpts = []
    e_dft_blocks = []
    e_qp_blocks = []

    with open(eqp_file) as f:
        while True:
            header = f.readline()
            if not header.strip():
                break
            parts = header.split()
            if len(parts) < 4:
                break
            kx, ky, kz = float(parts[0]), float(parts[1]), float(parts[2])
            n_bands = int(parts[3])
            kpts.append([kx, ky, kz])

            e_dft_k = []
            e_qp_k = []
            for _ in range(n_bands):
                line = f.readline()
                cols = line.split()
                e_dft_k.append(float(cols[2]))
                e_qp_k.append(float(cols[3]))
            e_dft_blocks.append(e_dft_k)
            e_qp_blocks.append(e_qp_k)

    kpts_ibz = np.array(kpts)
    max_band = max(len(b) for b in e_dft_blocks)
    n_kpts = len(kpts)

    e_dft_ibz = np.full((n_kpts, max_band), np.nan)
    e_qp_ibz = np.full((n_kpts, max_band), np.nan)
    for i in range(n_kpts):
        nb = len(e_dft_blocks[i])
        e_dft_ibz[i, :nb] = e_dft_blocks[i]
        e_qp_ibz[i, :nb] = e_qp_blocks[i]

    return kpts_ibz, e_dft_ibz, e_qp_ibz


def apply_eqp_corrections(
    enk_full: np.ndarray,
    eqp_file: str,
    ry_to_ev: float = 13.6056980659,
    tol_ev: float = 0.01,
) -> np.ndarray:
    """Apply BGW eqp corrections to full-BZ DFT eigenvalues.

    Maps IBZ k-points from eqp1.dat to full-BZ k-points in enk_full by
    matching DFT eigenvalues (symmetry-equivalent k-points have identical
    DFT eigenvalues). For matched bands, replaces DFT with QP energies.

    Args:
        enk_full: (nk_full, n_band) DFT eigenvalues in Rydberg
        eqp_file: path to BGW eqp1.dat
        ry_to_ev: Rydberg to eV conversion
        tol_ev: tolerance for DFT energy matching in eV

    Returns:
        enk_qp: (nk_full, n_band) eigenvalues with QP corrections applied (Ry)
    """
    kpts_ibz, e_dft_ibz, e_qp_ibz = read_bgw_eqp(eqp_file)
    nk_ibz, nb_eqp = e_dft_ibz.shape
    nk_full, nb_full = enk_full.shape

    enk_full_ev = enk_full * ry_to_ev
    enk_qp = enk_full.copy()

    matched = np.zeros(nk_full, dtype=bool)

    for ik_full in range(nk_full):
        best_ibz = -1
        best_err = np.inf
        for ik_ibz in range(nk_ibz):
            # Compare DFT energies for the bands covered by eqp
            n_compare = min(nb_eqp, nb_full)
            mask = ~np.isnan(e_dft_ibz[ik_ibz, :n_compare])
            if not np.any(mask):
                continue
            err = np.max(np.abs(
                enk_full_ev[ik_full, :n_compare][mask] - e_dft_ibz[ik_ibz, :n_compare][mask]
            ))
            if err < best_err:
                best_err = err
                best_ibz = ik_ibz

        if best_ibz >= 0 and best_err < tol_ev:
            matched[ik_full] = True
            for ib in range(min(nb_eqp, nb_full)):
                if not np.isnan(e_qp_ibz[best_ibz, ib]):
                    enk_qp[ik_full, ib] = e_qp_ibz[best_ibz, ib] / ry_to_ev

    n_matched = int(np.sum(matched))
    print(f"EQP: matched {n_matched}/{nk_full} full-BZ k-points to {nk_ibz} IBZ k-points")
    if n_matched < nk_full:
        unmatched = np.where(~matched)[0]
        print(f"  WARNING: {nk_full - n_matched} k-points unmatched (indices: {unmatched[:10]}...)")

    return enk_qp


def _find_restart_file(input_file: str) -> str:
    input_dir = os.path.dirname(os.path.abspath(input_file))
    candidates = []
    candidates.extend(sorted(glob.glob(os.path.join(input_dir, "tmp", "isdf_tensors_*.h5"))))
    candidates.extend(sorted(glob.glob(os.path.join(input_dir, "isdf_tensors_*.h5"))))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find canonical restart file isdf_tensors_*.h5 in {input_dir}")


def _load_ring_subset(
    restart_file: str,
    n_val: int,
    n_cond: int,
    px: int,
    py: int,
    eqp_file: Optional[str] = None,
    n_occ: Optional[int] = None,
) -> dict:
    """Load a single-device BSE subset from canonical gw_jax restart state."""
    with h5py.File(restart_file, "r") as f:
        V_qmunu = jnp.asarray(f["V_qmunu"][:])
        if "W0_qmunu" in f and bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            W0_qmunu = jnp.asarray(f["W0_qmunu"][:])
        else:
            W0_qmunu = None
        if "psi_full_y" not in f or "enk_full" not in f:
            raise ValueError(
                f"{restart_file} is missing canonical psi_full_y/enk_full datasets. "
                "Regenerate restart tensors with current gw_jax."
            )
        psi_full = jnp.asarray(f["psi_full_y"][:])
        enk_full_np = np.asarray(f["enk_full"][:])

    if eqp_file is not None:
        print(f"Applying EQP corrections from {eqp_file}")
        enk_full_np = apply_eqp_corrections(enk_full_np, eqp_file)

    enk_full = jnp.asarray(enk_full_np)

    nkx, nky, nkz = V_qmunu.shape[3:6]
    nk = nkx * nky * nkz
    n_rmu = int(V_qmunu.shape[-1])
    lcm_xy = math.lcm(px, py)
    n_rmu_pad = ((n_rmu + lcm_xy - 1) // lcm_xy) * lcm_xy

    n_bands_total = enk_full.shape[1]
    if n_occ is not None:
        # Explicit occupied band count: top n_val of occupied, bottom n_cond of unoccupied
        n_val_available = n_occ
        n_cond_available = n_bands_total - n_occ
        print(f"Band split: {n_occ} occupied, {n_cond_available} unoccupied (explicit n_occ)")
    else:
        # Fallback: detect from mean energy sign (may fail for non-Fermi-level-referenced energies)
        mean_enk_full = jnp.mean(enk_full, axis=0)
        n_val_available = int(jnp.sum(mean_enk_full < 0.0))
        n_cond_available = int(jnp.sum(mean_enk_full > 0.0))
        n_occ = n_val_available
        if n_val_available < n_val:
            print(f"WARNING: auto-detected only {n_val_available} valence bands from mean energy < 0.")
            print(f"  Consider using --n-occ to specify the number of occupied bands explicitly.")
    if n_val > n_val_available:
        print(f"Warning: requested {n_val} valence bands but only {n_val_available} available; using {n_val_available}")
    if n_cond > n_cond_available:
        print(f"Warning: requested {n_cond} conduction bands but only {n_cond_available} available; using {n_cond_available}")
    n_val = min(n_val, n_val_available)
    n_cond = min(n_cond, n_cond_available)
    # Select top n_val occupied bands and bottom n_cond unoccupied bands
    val_indices = jnp.arange(n_occ - n_val, n_occ)
    cond_indices = jnp.arange(n_occ, n_occ + n_cond)

    psi_v = psi_full[:, val_indices, :, :]
    psi_c = psi_full[:, cond_indices, :, :]
    eps_v = enk_full[:, val_indices]
    eps_c = enk_full[:, cond_indices]

    psi_v = _pad_last_axis(psi_v, n_rmu_pad)
    psi_c = _pad_last_axis(psi_c, n_rmu_pad)
    psi_v, n_val_pad = _pad_axis_to_multiple(psi_v, axis=1, multiple=py)
    psi_c, n_cond_pad = _pad_axis_to_multiple(psi_c, axis=1, multiple=px)
    eps_v, _ = _pad_axis_to_multiple(eps_v, axis=1, multiple=py)
    eps_c, _ = _pad_axis_to_multiple(eps_c, axis=1, multiple=px)
    V_q0 = V_qmunu[0, 0, 0, 0, 0, 0, :, :]
    V_q0 = _pad_last_two_axes(V_q0, n_rmu_pad)
    W_src = W0_qmunu if W0_qmunu is not None else V_qmunu
    W_q = W_src[0, 0, 0, :, :, :, :, :].transpose(3, 4, 0, 1, 2)
    W_q = _pad_first_two_axes(W_q, n_rmu_pad)

    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (1, n_cond_pad, n_val_pad, nk)) + 1j * jax.random.normal(
        key, (1, n_cond_pad, n_val_pad, nk)
    )

    return {
        "psi_c": psi_c,
        "psi_v": psi_v,
        "eps_c": eps_c,
        "eps_v": eps_v,
        "W_q": W_q,
        "V_q0": V_q0,
        "X": X,
        "nkx": nkx,
        "nky": nky,
        "nkz": nkz,
        "nk": nk,
        "n_rmu_pad": n_rmu_pad,
    }
