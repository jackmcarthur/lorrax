"""Execute an MPA Sigma plan with the established GN spatial kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.units import RYD_TO_EV
from file_io.mpa_store import read_pole_slices
from gw.ppm_accumulators import DeviceOmegaAccumulator
from gw.ppm_sigma import SigmaOmegaResult, pad_sigma_window, strip_sigma_window
from gw.ppm_tau_kernel import get_shared_sigma_tau_kernel
from gw.ppm_windows import _iter_branches

from .sigma_windows import build_shared_sigma_windows


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
    """Compute ``Sigma_c(omega)`` from sharded ``(Omega_p, B_p)`` fields.

    Four poles are stacked at a time by default to bound HBM.  The grouping is
    purely a memory schedule: every window selects poles from their actual
    complex frequency, and a rule spanning two batches is evaluated twice.
    Only one ``W(t)``, ``G(t)``, and ``Sigma(t)`` tile exists per dispatch.
    """
    if not plan:
        raise ValueError("MPA Sigma needs pole fields and a nonempty plan")
    if (Omega_poles.ndim != 4 or B_poles.shape != Omega_poles.shape
            or not int(Omega_poles.shape[0])):
        raise ValueError("Omega_poles and B_poles must share (p,q,mu,mu)")
    batch_size = int(pole_batch_size)
    if batch_size < 1:
        raise ValueError("pole_batch_size must be positive")

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
    n_poles = int(Omega_poles.shape[0])
    for lo in range(0, n_poles, batch_size):
        batch = tuple(range(lo, min(lo + batch_size, n_poles)))
        Omega, B = Omega_poles[lo:lo + batch_size], B_poles[lo:lo + batch_size]
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


def _branches(wfns, omega, efermi_ry):
    """The four causal branches, with occupation and energy kept separate."""
    omega = np.asarray(omega, np.float64)
    idx_pos, idx_neg = np.where(omega >= 0.0)[0], np.where(omega < 0.0)[0]
    energy = wfns.enk[:, wfns.slices.full] - float(efermi_ry)
    occupied = wfns.occ[:, wfns.slices.full] > 0.5
    # Do not clip these distances at zero.  In a small-gap or inverted
    # system an unoccupied state may sit below E_F (or an occupied state
    # above it); the occupation chooses the branch and the window planner
    # decides from the actual denominator whether it is crossing.
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
    hgl_target_error=1.0e-6,
    hgl_max_nodes=200,
    pole_batch_size=4,
    print_fn=print,
):
    """Read a fitted MPA store, derive its windows, and compute Sigma_c.

    Pole tensors are read collectively in their native sharding through
    :mod:`file_io.mpa_store`; no full ``(p,q,mu,mu)`` host copy exists.
    """
    Omega, B = read_pole_slices(
        fit_src, mesh_xy=mesh_xy, unfold=True, return_sharded=True,
        to_unit="Ry")
    branches = _branches(wfns, omega_grid_ry, efermi_ry)
    plan, geometry = build_shared_sigma_windows(
        Omega, branches,
        regularization_width_ry=regularization_width_ry,
        edge_factor=edge_factor, target_error=target_error,
        max_rank=max_rank, hgl_target_error=hgl_target_error,
        hgl_max_nodes=hgl_max_nodes)
    physical = execution_census(plan, int(Omega.shape[0]), pole_batch_size)
    print_fn(
        f"  MPA windows: xi={geometry['xi_ry'] * RYD_TO_EV:.4f} eV, "
        f"{geometry['n_windows']} logical windows, "
        f"{physical['n_tau']} physical tau dispatches")
    return integrate_sigma_windows(
        wfns, Omega, B, plan, omega_grid_ry, meta, mesh_xy,
        pole_batch_size=pole_batch_size, print_fn=print_fn)


__all__ = [
    "compute_sigma_c_mpa_omega_grid",
    "execution_census",
    "integrate_sigma_windows",
]
