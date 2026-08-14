"""Execute an MPA Sigma plan with the established GN spatial kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.units import RYD_TO_EV
from file_io.mpa_store import read_poles, validate_fit_store
from gw.ppm_accumulators import DeviceOmegaAccumulator
from gw.ppm_sigma import SigmaOmegaResult, pad_sigma_window, strip_sigma_window
from gw.ppm_tau_kernel import get_shared_sigma_tau_kernel
from gw.ppm_windows import _iter_branches

from .sigma_windows import (build_shared_sigma_windows,
                            summarize_sigma_poles)


def _bounded_pole_batch_size(value):
    size = int(value)
    if not 1 <= size <= 4:
        raise ValueError("MPA pole_batch_size must be in [1, 4]")
    return size


def _batch_rows(row, batch):
    local = {int(p): i for i, p in enumerate(batch)}
    keep = [i for i, p in enumerate(row.pole_indices) if int(p) in local]
    if not keep:
        return None
    return (
        np.asarray([local[int(row.pole_indices[i])] for i in keep], np.int32),
        np.asarray(row.bounds[keep], np.float64),
        np.asarray(row.phase_real[keep], bool),
    )


def execution_census(plan, n_poles, pole_batch_size=4):
    """Return physical tau dispatches after memory batching."""
    pole_batch_size = _bounded_pole_batch_size(pole_batch_size)
    batches = [range(lo, min(lo + pole_batch_size, n_poles))
               for lo in range(0, n_poles, pole_batch_size)]
    sweeps = nodes = 0
    for batch in batches:
        for row in plan:
            if _batch_rows(row, batch) is not None:
                sweeps += 1
                nodes += row.window.n_tau
    return {"n_sweeps": sweeps, "n_tau": nodes}


def integrate_sigma_windows(
    wfns,
    Omega_poles,
    B_poles,
    plan,
    omega_grid_ry,
    meta,
    mesh_xy,
    *,
    pole_batch_size=4,
    print_fn=print,
):
    """Compute ``Sigma_c(omega)`` from resident pole fields (test adapter)."""
    if not plan:
        raise ValueError("MPA Sigma needs pole fields and a nonempty plan")
    if (Omega_poles.ndim != 4 or B_poles.shape != Omega_poles.shape
            or not int(Omega_poles.shape[0])):
        raise ValueError("Omega_poles and B_poles must share (p,q,mu,mu)")
    batch_size = _bounded_pole_batch_size(pole_batch_size)

    n_poles = int(Omega_poles.shape[0])
    batches = ((lo, Omega_poles[lo:lo + batch_size],
                B_poles[lo:lo + batch_size])
               for lo in range(0, n_poles, batch_size))
    return _integrate_sigma_batches(
        wfns, batches, n_poles, plan, omega_grid_ry, meta, mesh_xy,
        pole_batch_size=batch_size, print_fn=print_fn)


def _integrate_sigma_batches(
    wfns,
    batches,
    n_poles,
    plan,
    omega_grid_ry,
    meta,
    mesh_xy,
    *,
    pole_batch_size,
    print_fn,
):
    """One spatial executor for resident arrays and streamed fit slabs."""
    omega = np.asarray(omega_grid_ry, np.float64)
    if omega.ndim != 1 or not omega.size:
        raise ValueError("omega_grid_ry must be a nonempty vector")

    s = wfns.slices
    psi_coh_xn, psi_coh_yr = wfns.xn(s.full), wfns.yr(s.full)
    psi_proj_xr, psi_proj_yn = wfns.xr(s.sigma), wfns.yn(s.sigma)
    psi_proj_xr, psi_proj_yn, nb_real = pad_sigma_window(
        psi_proj_xr, psi_proj_yn, mesh_xy)
    shape = (omega.size, int(psi_proj_xr.shape[0]),
             int(psi_proj_xr.shape[1]), int(psi_proj_yn.shape[3]))
    output_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y"))
    accumulator = DeviceOmegaAccumulator(
        omega, shape=shape, sharding=output_sharding)
    tau_kernel = get_shared_sigma_tau_kernel(
        mesh_xy=mesh_xy,
        kgrid=(int(meta.nkx), int(meta.nky), int(meta.nkz)))
    small = NamedSharding(mesh_xy, P())

    n_sweeps = n_tau = 0
    batch_size = int(pole_batch_size)
    for lo, Omega, B in batches:
        width = int(Omega.shape[0])
        batch = tuple(range(int(lo), int(lo) + width))
        for row in plan:
            selected = _batch_rows(row, batch)
            if selected is None:
                continue
            pole_indices, bounds, phase_real = (
                jax.device_put(x, small) for x in selected)
            win = row.window
            accumulator.begin_window(
                win.nodes.t, win.nodes.alpha,
                omega_sign=win.omega_sign, prefactor=win.prefactor,
                e_ref_sum=win.E_ref_A + win.E_ref_B,
                antihermitian=(win.project_code == 1),
                omega_indices=row.omega_idx, omega_values=row.omega_abs)
            for t in np.asarray(jax.device_get(win.nodes.t), np.complex128):
                sigma_tau = tau_kernel(
                    psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                    row.E_A, jnp.asarray(win.mask_A), B, Omega,
                    pole_indices, bounds, phase_real,
                    jnp.asarray(win.E_ref_A), jnp.asarray(win.E_ref_B),
                    jnp.asarray(t, dtype=jnp.complex128))
                accumulator.add_tau(sigma_tau)
                n_tau += 1
            accumulator.end_window()
            n_sweeps += 1
        del B, Omega

    sigma = strip_sigma_window(accumulator.finalize(), nb_real)
    print_fn(f"  MPA Sigma: {n_tau} tau dispatches in {n_sweeps} sweeps "
             f"({n_poles} poles, batches of {batch_size})")
    return SigmaOmegaResult(
        omega_ry=omega,
        omega_ev=np.asarray(omega * RYD_TO_EV, np.float64),
        sigma_c_kij=sigma)


def integrate_sigma_store(
    wfns,
    fit_src,
    n_poles,
    plan,
    omega_grid_ry,
    meta,
    mesh_xy,
    *,
    pole_batch_size=4,
    print_fn=print,
):
    """Read, unfold, consume, and release one pole range at a time."""
    batch_size = _bounded_pole_batch_size(pole_batch_size)

    def batches():
        for lo in range(0, int(n_poles), batch_size):
            hi = min(lo + batch_size, int(n_poles))
            Omega, B = read_poles(
                fit_src, pole_slice=slice(lo, hi), mesh_xy=mesh_xy,
                unfold=True, return_sharded=True, to_unit="Ry")
            yield lo, Omega, B

    return _integrate_sigma_batches(
        wfns, batches(), int(n_poles), plan, omega_grid_ry, meta, mesh_xy,
        pole_batch_size=batch_size, print_fn=print_fn)


def _branches(wfns, omega, efermi_ry):
    """The four causal branches, with occupation and energy kept separate."""
    omega = np.asarray(omega, np.float64)
    idx_pos, idx_neg = np.where(omega >= 0.0)[0], np.where(omega < 0.0)[0]
    energy = wfns.enk[:, wfns.slices.full] - float(efermi_ry)
    occupied = wfns.occ[:, wfns.slices.full] > 0.5
    # Do not clip these distances at zero.  In a small-gap or inverted
    # system an unoccupied state may sit below E_F (or an occupied state
    # above it); occupation still chooses the band sum.  The current internal
    # planner refuses a nominally sign-definite cell that then crosses zero.
    # Public MPA remains disabled until that cell is split and rerouted.
    return _iter_branches(
        omega_pos=omega[idx_pos], idx_pos=idx_pos,
        omega_neg_abs=-omega[idx_neg], idx_neg=idx_neg,
        E_cond=energy, H_val=-energy,
        cond_mask=~occupied, val_mask=occupied)


def compute_sigma_c_mpa_omega_grid(
    wfns,
    fit_src,
    meta,
    mesh_xy,
    *,
    omega_grid_ry,
    efermi_ry,
    regularization_width_ry,
    edge_factor=1.5,
    target_error=1.0e-4,
    max_rank=96,
    crossing_max_nodes=500,
    pole_batch_size=4,
    fit_identity=None,
    print_fn=print,
):
    """Read a fitted MPA store, derive its windows, and compute Sigma_c.

    Pole tensors are read collectively in their native sharding.  A first
    four-pole walk retains only scalar geometry for planning; the spatial
    executor rereads and releases the same four-pole ranges.  No complete
    pole axis exists on host or device.
    """
    ledger = validate_fit_store(fit_src, expected_identity=fit_identity)
    n_poles = int(ledger["n_p"])
    pole_batch_size = _bounded_pole_batch_size(pole_batch_size)
    branches = _branches(wfns, omega_grid_ry, efermi_ry)
    summaries = []
    for lo in range(0, n_poles, int(pole_batch_size)):
        hi = min(lo + int(pole_batch_size), n_poles)
        Omega, B = read_poles(
            fit_src, pole_slice=slice(lo, hi), mesh_xy=mesh_xy,
            unfold=True, return_sharded=True, to_unit="Ry")
        summaries.extend(summarize_sigma_poles(
            Omega, branches,
            regularization_width_ry=regularization_width_ry,
            edge_factor=edge_factor, pole_offset=lo))
        del Omega, B
    plan, geometry = build_shared_sigma_windows(
        None, branches, pole_summaries=summaries,
        regularization_width_ry=regularization_width_ry,
        edge_factor=edge_factor, target_error=target_error,
        max_rank=max_rank, crossing_max_nodes=crossing_max_nodes)
    physical = execution_census(plan, n_poles, pole_batch_size)
    print_fn(
        f"  MPA windows: eta={geometry['eta_ry'] * RYD_TO_EV:.4f} eV, "
        f"{geometry['n_windows']} logical windows, "
        f"{physical['n_tau']} physical tau dispatches")
    return integrate_sigma_store(
        wfns, fit_src, n_poles, plan, omega_grid_ry, meta, mesh_xy,
        pole_batch_size=pole_batch_size, print_fn=print_fn)


__all__ = [
    "compute_sigma_c_mpa_omega_grid",
    "execution_census",
    "integrate_sigma_store",
    "integrate_sigma_windows",
]
