"""Execute an MPA Sigma plan with the established GN spatial kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common.units import RYD_TO_EV
from file_io.mpa_store import PoleReader, open_pole_reader, validate_fit_store
from gw.ppm_accumulators import DeviceOmegaAccumulator
from gw.ppm_sigma import SigmaOmegaResult, pad_sigma_window, strip_sigma_window
from gw.ppm_tau_kernel import get_shared_sigma_tau_kernel
from gw.ppm_windows import branches_for_omega_grid

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
    """One spatial executor for streamed fit slabs."""
    omega = np.asarray(omega_grid_ry, np.float64)
    if omega.ndim != 1 or not omega.size:
        raise ValueError("omega_grid_ry must be a nonempty vector")

    s = wfns.slices
    # ``sigma_sum``, not ``full`` — the Σ band sum, not the loaded extent.
    # Identical on an unsplit deck.  UNVERIFIED on a split one: the public
    # MPA Σ still refuses to run (gw_config.ComputeMode), so this line is
    # wired for consistency and has never executed under a split.
    psi_coh_xn, psi_coh_yr = wfns.xn(s.sigma_sum), wfns.yr(s.sigma_sum)
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
                device_put_process_local(x, small) for x in selected)
            win = row.window
            weight = getattr(row, "band_weight", None)
            if weight is None:
                # Incumbent bool selector — the kernel's mask path, bit-exact.
                selector = jnp.asarray(win.mask_A)
            else:
                # Metallic: fold support × fractional weight into one float
                # operand; the kernel dtype-dispatches it onto build_G_tau's
                # band_weight seam.  Never clipped.
                selector = (jnp.asarray(win.mask_A, jnp.float64)
                            * jnp.reshape(jnp.asarray(weight, jnp.float64),
                                          np.asarray(win.mask_A).shape))
            accumulator.begin_window(
                win.nodes.t, win.nodes.alpha,
                omega_sign=win.omega_sign, prefactor=win.prefactor,
                e_ref_sum=win.E_ref_A + win.E_ref_B,
                antihermitian=(win.project_code == 1),
                omega_indices=row.omega_idx, omega_values=row.omega_abs)
            for t in np.asarray(jax.device_get(win.nodes.t), np.complex128):
                sigma_tau = tau_kernel(
                    psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                    row.E_A, selector, B, Omega,
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
    """Read, unfold, consume, and release one pole range at a time.

    ``fit_src`` is a store path, or a live
    :class:`~file_io.mpa_store.PoleReader` whose collective handle the
    caller owns — which is what
    :func:`compute_sigma_c_mpa_omega_grid` passes, so the census walk and
    this executor walk share ONE open handle for the whole iteration
    instead of opening the store once per pole batch (audit A1).  Given a
    path, this function owns a reader for the length of its own walk.
    """
    batch_size = _bounded_pole_batch_size(pole_batch_size)

    def batches(reader):
        for lo in range(0, int(n_poles), batch_size):
            hi = min(lo + batch_size, int(n_poles))
            Omega, B = reader.read(
                slice(lo, hi), unfold=True, return_sharded=True,
                to_unit="Ry")
            yield lo, Omega, B

    def run(reader):
        return _integrate_sigma_batches(
            wfns, batches(reader), int(n_poles), plan, omega_grid_ry, meta,
            mesh_xy, pole_batch_size=batch_size, print_fn=print_fn)

    if isinstance(fit_src, PoleReader):
        return run(fit_src)
    with open_pole_reader(fit_src, mesh_xy=mesh_xy) as reader:
        return run(reader)


def _branches(wfns, omega, efermi_ry, occupation_state=None):
    """The four causal branches, with occupation and energy kept separate.

    Band axis is ``slices.sigma_sum`` -- the Sigma band count, not the
    chi one -- for the same reason as the psi slices above: these index
    the SAME band axis the causal branches sum over.  That choice is
    orthogonal to the occupation one below: the slice says WHICH bands
    are summed, the weights say with what amplitude.

    ``occupation_state=None`` is the incumbent insulating semantics,
    bit-exact: bool occ>0.5 masks, distances signed against ``efermi_ry``.
    With a state (duck-typed: ``.f_kn``, ``.mu_ry``), the branches carry the
    exact fractional supports and weights: the val branch sums EVERY band
    with f≠0 at weight f, the cond branch every band with f≠1 at weight
    1−f — only exact 0/1 weights are dropped, nothing is clipped, and MP
    overshoot (f<0 or f>1) rides through unchanged
    (docs/theory/finite-occupation-screening.md).
    """
    if occupation_state is None:
        energy = wfns.enk[:, wfns.slices.sigma_sum] - float(efermi_ry)
        occupied = wfns.occ[:, wfns.slices.sigma_sum] > 0.5
        # Do not clip these distances at zero.  In a small-gap or inverted
        # system an unoccupied state may sit below E_F (or an occupied state
        # above it); occupation still chooses the band sum.  A cell whose
        # rectangle then crosses zero is rerouted through the crossing core
        # by the planner's excursion-deepened edge (sigma_windows._geometry).
        return branches_for_omega_grid(
            omega, E_cond=energy, H_val=-energy,
            cond_mask=~occupied, val_mask=occupied)
    mu = float(occupation_state.mu_ry)
    if abs(float(efermi_ry) - mu) > 1.0e-12:
        raise ValueError(
            "MPA Sigma got efermi_ry inconsistent with its occupation "
            f"state: efermi_ry={float(efermi_ry):.12g} Ry vs "
            f"occupation_state.mu_ry={mu:.12g} Ry.  One chemical potential "
            "per iteration — pass the state's own mu.")
    f = jnp.reshape(jnp.asarray(occupation_state.f_kn),
                    wfns.enk.shape)[:, wfns.slices.sigma_sum]
    energy = wfns.enk[:, wfns.slices.sigma_sum] - mu
    return branches_for_omega_grid(
        omega, E_cond=energy, H_val=-energy,
        cond_mask=(f != 1.0), val_mask=(f != 0.0),
        cond_weight=1.0 - f, val_weight=f)


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
    target_error,
    crossing_target_error=None,
    max_rank,
    crossing_max_nodes,
    omega_cluster_gap_ry=1.0,
    pole_batch_size=4,
    fit_identity=None,
    expected_screening_diagrams=None,
    occupation_state=None,
    print_fn=print,
):
    """Read a fitted MPA store, derive its windows, and compute Sigma_c.

    ``occupation_state`` (duck-typed ``gw.efermi.OccupationState``): None is
    the incumbent insulating semantics, bit-exact.  With a state, the causal
    branches carry exact fractional supports and (f, 1−f) weights, and
    ``efermi_ry`` must equal ``occupation_state.mu_ry``.

    Pole tensors are read collectively in their native sharding.  A first
    four-pole walk retains only scalar geometry for planning; the spatial
    executor rereads and releases the same four-pole ranges.  No complete
    pole axis exists on host or device.
    """
    ledger = validate_fit_store(
        fit_src, expected_identity=fit_identity,
        expected_screening_diagrams=expected_screening_diagrams)
    n_poles = int(ledger["n_p"])
    pole_batch_size = _bounded_pole_batch_size(pole_batch_size)
    branches = _branches(wfns, omega_grid_ry, efermi_ry,
                         occupation_state=occupation_state)
    summaries = []
    # ONE collective handle for the census walk, the planner, and the
    # executor walk — the whole Σ stage of this iteration.  The reader
    # does its h5py reads (ledger, unfold tables) before that handle
    # exists and none after, so no serial-h5py open on this store
    # overlaps or interleaves with the FFI one anywhere inside a Σ stage
    # (audit A1; hdf5_owner enforces it).  The context manager is the
    # release path: a refusal from the planner or the executor must still
    # close the handle on every rank.
    with open_pole_reader(fit_src, mesh_xy=mesh_xy) as reader:
        for lo in range(0, n_poles, int(pole_batch_size)):
            hi = min(lo + int(pole_batch_size), n_poles)
            Omega, B = reader.read(
                slice(lo, hi), unfold=True, return_sharded=True,
                to_unit="Ry")
            summaries.extend(summarize_sigma_poles(
                Omega, B, branches,
                regularization_width_ry=regularization_width_ry,
                edge_factor=edge_factor, pole_offset=lo))
            del Omega, B
        plan, geometry = build_shared_sigma_windows(
            summaries, branches,
            regularization_width_ry=regularization_width_ry,
            edge_factor=edge_factor, target_error=target_error,
            crossing_target_error=crossing_target_error,
            max_rank=max_rank, crossing_max_nodes=crossing_max_nodes,
            omega_cluster_gap_ry=omega_cluster_gap_ry)
        print_fn(
            f"  MPA windows: eta={geometry['eta_ry'] * RYD_TO_EV:.4f} eV, "
            f"{geometry['n_windows']} logical windows")
        return integrate_sigma_store(
            wfns, reader, n_poles, plan, omega_grid_ry, meta, mesh_xy,
            pole_batch_size=pole_batch_size, print_fn=print_fn)


def assert_head_body_occupation_match(head_attrs, occupation_state):
    """Refuse when the head fit and the body Sigma disagree about occupations.

    ``head_attrs`` is the stamp dict a head-fit reader returns.  One
    occupation state per iteration is the rule (ARCHITECTURE W2.d/W3); the
    head's stamped ``occ_hash``/``mu_ry`` must equal the body's.  A metal
    run with an UNSTAMPED head fit refuses too — an unverifiable stamp is
    not a pass.  Insulating runs (state None) skip the check.
    """
    if occupation_state is None:
        return
    stamped_hash = head_attrs.get("occ_hash")
    stamped_mu = head_attrs.get("mu_ry")
    if stamped_hash is None or stamped_mu is None:
        raise ValueError(
            "metallic MPA Sigma requires an occupation-stamped head fit "
            "(occ_hash + mu_ry attrs); this store has "
            f"occ_hash={stamped_hash!r}, mu_ry={stamped_mu!r}.  Refit the "
            "head with the current iteration's occupation state.")
    if (str(stamped_hash) != str(occupation_state.occ_hash)
            or abs(float(stamped_mu) - float(occupation_state.mu_ry))
            > 1.0e-12):
        raise ValueError(
            "head fit and Sigma body carry different occupation states: "
            f"head (occ_hash={stamped_hash}, mu={float(stamped_mu):.12g}) "
            f"vs body (occ_hash={occupation_state.occ_hash}, "
            f"mu={float(occupation_state.mu_ry):.12g}).  One state per "
            "iteration; rebuild the stale artifact.")


__all__ = [
    "assert_head_body_occupation_match",
    "compute_sigma_c_mpa_omega_grid",
    "integrate_sigma_store",
]
