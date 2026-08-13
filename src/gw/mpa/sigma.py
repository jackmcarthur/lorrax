"""Execute an MPA Sigma plan with the established GN spatial kernel."""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.units import RYD_TO_EV
from gw.ppm_accumulators import DeviceOmegaAccumulator
from gw.ppm_sigma import SigmaOmegaResult, pad_sigma_window, strip_sigma_window
from gw.ppm_tau_kernel import get_shared_sigma_tau_kernel


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


@lru_cache(maxsize=8)
def _stack_on_device(sharding):
    out = NamedSharding(sharding.mesh, P(None, None, "x", "y"))
    return jax.jit(lambda *xs: jnp.stack(xs), out_shardings=out)


def integrate_sigma_windows(
    wfns,
    pole_fields,
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
    fields = tuple(pole_fields)
    if not fields or not plan:
        raise ValueError("MPA Sigma needs pole fields and a nonempty plan")
    if any(len(pair) != 2 for pair in fields):
        raise ValueError("pole_fields entries must be (Omega_p, B_p) pairs")
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
    for lo in range(0, len(fields), batch_size):
        batch = tuple(range(lo, min(lo + batch_size, len(fields))))
        Omega = _stack_on_device(fields[0][0].sharding)(
            *(fields[p][0] for p in batch))
        B = _stack_on_device(fields[0][1].sharding)(
            *(fields[p][1] for p in batch))
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
             f"({len(fields)} poles, batches of {batch_size})")
    return SigmaOmegaResult(
        omega_ry=omega,
        omega_ev=np.asarray(omega * RYD_TO_EV, np.float64),
        sigma_c_kij=sigma)


__all__ = ["execution_census", "integrate_sigma_windows"]
