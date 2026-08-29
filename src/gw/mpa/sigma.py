"""Execute an MPA Sigma plan with the established GN spatial kernel."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common.units import RYD_TO_EV
from file_io.mpa_store import PoleReader, open_pole_reader, validate_fit_store
from gw.ppm_accumulators import DeviceOmegaAccumulator
from gw.ppm_sigma import (SigmaOmegaResult,
                          assert_sharded_sigma_window_divides_mesh,
                          pad_sigma_window, strip_sigma_window)
from gw.ppm_tau_kernel import get_shared_sigma_tau_kernel
from gw.ppm_windows import branches_for_omega_grid
from gw.wavefunction_bundle import face_kernel_kwargs

from .sigma_windows import (OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
                            build_shared_sigma_windows,
                            summarize_sigma_poles)


def _build_delivered_sigma_windows(*args, **kwargs):
    """Lazy service/solver import; the default panes path pays none of it."""
    from .delivered_windows import build_delivered_sigma_windows
    return build_delivered_sigma_windows(*args, **kwargs)


def resolve_sigma_plan_builder():
    """Resolve ``LORRAX_SIGMA_PLAN`` to its planning pathway.

    The exact grammar is ``panes`` or ``delivered`` after stripping and
    lower-casing.  Unset and blank both preserve the incumbent panes path;
    every other spelling refuses rather than silently running another arm.
    """
    raw = os.environ.get("LORRAX_SIGMA_PLAN", "panes").strip().lower()
    mode = raw or "panes"
    if mode == "panes":
        return mode, build_shared_sigma_windows
    if mode == "delivered":
        return mode, _build_delivered_sigma_windows
    raise ValueError(
        "LORRAX_SIGMA_PLAN must be 'panes' or 'delivered'; "
        f"got {raw!r}")


def _bounded_pole_batch_size(value):
    size = int(value)
    if not 1 <= size <= 8:
        raise ValueError("MPA pole_batch_size must be in [1, 8]")
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
    face_kwargs = face_kernel_kwargs(wfns)
    if wfns.layout == "legacy":
        # ``sigma_sum``, not ``full`` — the Σ band sum, not the loaded
        # extent.  Identical on an unsplit deck.  UNVERIFIED on a split
        # one: the public MPA Σ still refuses to run (gw_config.
        # ComputeMode), so this line is wired for consistency and has
        # never executed under a split.
        psi_coh_xn, psi_coh_yr = wfns.xn(s.sigma_sum), wfns.yr(s.sigma_sum)
        psi_proj_xr, psi_proj_yn = wfns.xr(s.sigma), wfns.yn(s.sigma)
        # THE SAME precondition the PPM sharded branch owns.  This executor
        # accumulates into a P(None,None,'x','y') array and then strips the pad
        # block off both trailing axes, which on an indivisible window leaves a
        # sharded array whose declared spec no longer divides its own shape.
        # Before 2026-08-22 there was no divisibility check anywhere in this
        # module while ppm_sigma refused the same case by name -- two contracts
        # at one seam.
        assert_sharded_sigma_window_divides_mesh(
            int(psi_proj_xr.shape[1]), mesh_xy, ansatz="compute_mode = mpa")
        psi_proj_xr, psi_proj_yn, nb_real = pad_sigma_window(
            psi_proj_xr, psi_proj_yn, mesh_xy)
        shape = (omega.size, int(psi_proj_xr.shape[0]),
                 int(psi_proj_xr.shape[1]), int(psi_proj_yn.shape[3]))
    else:
        # Face carrier (2026-08-22, mechanical port sharing
        # ppm_tau_kernel's face dispatch — see gw.ppm_sigma._run_sigma_
        # branch's identically-shaped docstring): psi_mun/psi_nmu are used
        # UNSLICED for both roles — the accumulator BUILDS at the mesh-
        # divisible nb_full extent regardless of nb_sigma, and always
        # will: contract_bands.contract_bands_block_reshard's face arm
        # (_face_project_kernel) builds its two distrib_la.gemm_plans
        # EAGERLY at the fixed face_shape width (nb_full), shared with
        # this same kernel's "coh"/G-build plan, so narrowing psi_proj's
        # INPUT width would desync it from a plan compiled for nb_full —
        # not this executor's call to make without reopening
        # contract_bands.py's shared GEMM-plan contract (report §5: "do
        # not fork ... a second Sigma projector").
        #
        # THE FIX (2026-08-23): the part that genuinely must land at
        # nb_sigma is the OUTPUT — Sigma_c(omega,k,m,n)'s own (m,n) axes
        # — not the input.  strip_sigma_window's device-array arm now
        # applies wavefunction_bundle.pack_band_window's OWN mechanism
        # (jax.lax.slice_in_dim + jax.lax.with_sharding_constraint) to
        # those trailing axes in place of the numpy-style slice that is
        # illegal on a mesh-sharded axis — the output-side analog of that
        # primitive's input-side repack, reusing its idiom rather than
        # inventing a second one.  That mechanism is legal ONLY when the
        # target extent already divides the mesh (with_sharding_
        # constraint requires it); an indivisible Σ window is the
        # genuinely impossible sub-case and is refused by name, by the
        # SAME shared owner the legacy branch above calls
        # (assert_sharded_sigma_window_divides_mesh — "ONE owner for a
        # contract two ansaetze reach at the same seam").  Reachability:
        # low_mem_bands_dynamic_ppm_unported still refuses every
        # compute_mode this function serves pending its own end-to-end
        # gate (gw_config.py), so this arm is exercised by the parity
        # tests and that gate, not yet by a general production path.
        nb_real = int(s.nb_sigma)
        assert_sharded_sigma_window_divides_mesh(
            nb_real, mesh_xy, ansatz="compute_mode = mpa")
        psi_coh_xn, psi_coh_yr = wfns.psi_mun, wfns.psi_nmu
        psi_proj_xr, psi_proj_yn = wfns.psi_nmu, wfns.psi_mun
        shape = (omega.size, int(meta.nk_tot), int(s.nb_full), int(s.nb_full))
    output_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y"))
    accumulator = DeviceOmegaAccumulator(
        omega, shape=shape, sharding=output_sharding)
    tau_kernel = get_shared_sigma_tau_kernel(
        mesh_xy=mesh_xy,
        kgrid=(int(meta.nkx), int(meta.nky), int(meta.nkz)),
        **face_kwargs)
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

    sigma = strip_sigma_window(
        accumulator.finalize(), nb_real, mesh_xy=mesh_xy)
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


def _branches(wfns, omega, efermi_ry, occupation_state=None,
              occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT):
    """The four causal branches, with occupation and energy kept separate.

    Band axis is ``slices.sigma_sum`` -- the Sigma band count, not the
    chi one -- for the same reason as the psi slices above: these index
    the SAME band axis the causal branches sum over.  That choice is
    orthogonal to the occupation one below: the slice says WHICH bands
    are summed, the weights say with what amplitude.

    ``occupation_state=None`` is the incumbent insulating semantics,
    bit-exact: bool occ>0.5 masks, distances signed against ``efermi_ry``.
    With a state (duck-typed: ``.f_kn``, ``.mu_ry``), the branches carry the
    fractional supports and weights: the val branch sums every band whose
    weight f clears the occupancy window at weight f, the cond branch every
    band whose weight 1−f clears it at weight 1−f.  Nothing is clipped and MP
    overshoot (f<0 or f>1) rides through unchanged
    (docs/theory/finite-occupation-screening.md).

    ``occupation_window_threshold`` sets that window;
    ``branches_for_omega_grid`` applies it.  1.0 restores the historical
    ``f != 1`` / ``f != 0`` supports bit-for-bit.  Applying it here rather
    than only in the planner keeps ONE support: ``sigma_windows._a_space``
    re-applies the same floor to the same weights, so the two agree by
    construction instead of by review.
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
        cond_weight=1.0 - f, val_weight=f,
        occupation_window_threshold=occupation_window_threshold)


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
    occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
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

    ``occupation_window_threshold`` is the OCCUPANCY below which a band is
    still counted in a branch; the cut is ``|weight| > 1 - it``.  It is
    forwarded from this one value to the BRANCH BUILD and to BOTH planner
    entry points, which is what keeps the branch supports, the pole census
    and the window build on one support.

    Pole tensors are read collectively in their native sharding.  The panes
    pathway's first configured-batch walk retains only scalar geometry; the
    delivered pathway reads the complete pole axis once and reduces each
    addressable spatial shard to a bounded measure.  The spatial executor
    then rereads and releases its configured pole ranges in either mode.
    """
    ledger = validate_fit_store(
        fit_src, expected_identity=fit_identity,
        expected_screening_diagrams=expected_screening_diagrams)
    n_poles = int(ledger["n_p"])
    pole_batch_size = _bounded_pole_batch_size(pole_batch_size)
    branches = _branches(
        wfns, omega_grid_ry, efermi_ry,
        occupation_state=occupation_state,
        occupation_window_threshold=occupation_window_threshold)
    plan_mode, plan_builder = resolve_sigma_plan_builder()
    # ONE collective handle for the census walk, the planner, and the
    # executor walk — the whole Σ stage of this iteration.  The reader
    # does its h5py reads (ledger, unfold tables) before that handle
    # exists and none after, so no serial-h5py open on this store
    # overlaps or interleaves with the FFI one anywhere inside a Σ stage
    # (audit A1; hdf5_owner enforces it).  The context manager is the
    # release path: a refusal from the planner or the executor must still
    # close the handle on every rank.
    with open_pole_reader(fit_src, mesh_xy=mesh_xy) as reader:
        if plan_mode == "panes":
            # The incumbent path is intentionally kept byte-for-byte in its
            # own arm: same configured census batches, summaries, builder
            # arguments, and executor plan.
            summaries = []
            for lo in range(0, n_poles, int(pole_batch_size)):
                hi = min(lo + int(pole_batch_size), n_poles)
                Omega, B = reader.read(
                    slice(lo, hi), unfold=True, return_sharded=True,
                    to_unit="Ry")
                summaries.extend(summarize_sigma_poles(
                    Omega, B, branches,
                    regularization_width_ry=regularization_width_ry,
                    edge_factor=edge_factor, pole_offset=lo,
                    occupation_window_threshold=occupation_window_threshold))
                del Omega, B
            plan, geometry = plan_builder(
                summaries, branches,
                regularization_width_ry=regularization_width_ry,
                edge_factor=edge_factor, target_error=target_error,
                crossing_target_error=crossing_target_error,
                max_rank=max_rank, crossing_max_nodes=crossing_max_nodes,
                omega_cluster_gap_ry=omega_cluster_gap_ry,
                occupation_window_threshold=occupation_window_threshold)
        else:
            # The delivered measure needs the residues, not only rectangle
            # extrema.  Pole fields stay sharded; the planner reduces each
            # addressable shard to bounded cells before any cross-rank gather.
            Omega, B = reader.read(
                slice(0, n_poles), unfold=True, return_sharded=True,
                to_unit="Ry")
            plan, geometry = plan_builder(
                [Omega] * len(branches), [B] * len(branches), branches,
                omega_grid_ry,
                regularization_width_ry=regularization_width_ry,
                target_error=target_error,
                max_nodes=max(int(max_rank), int(crossing_max_nodes)))
            del Omega, B
        if plan_mode == "panes":
            print_fn(
                f"  MPA windows: eta={geometry['eta_ry'] * RYD_TO_EV:.4f} eV, "
                f"{geometry['n_windows']} logical windows")
        else:
            print_fn(
                f"  MPA windows [delivered]: "
                f"eta={geometry['eta_ry'] * RYD_TO_EV:.4f} eV, "
                f"{geometry['n_windows']} logical windows, "
                f"{geometry['n_tau']} total nodes")
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
