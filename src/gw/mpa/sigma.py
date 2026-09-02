"""Execute an MPA Sigma plan with the established GN spatial kernel."""

from __future__ import annotations

import gc

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
from gw.sigma_box_plan import plan_sigma_windows
from gw.sigma_plan import resolve_sigma_plan
from gw.wavefunction_bundle import face_kernel_kwargs

from .sigma_windows import (OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
                            CROSSING_NODE_FLOOR,
                            build_shared_sigma_windows,
                            summarize_sigma_poles)


# The pane route is an immutable comparison instrument, not a production
# accuracy policy.  Freezing its historical target here lets old/new box-rule
# comparisons keep the same control while retiring the measured-sector deck
# dial from the production path.
_PANE_CONTROL_TARGET_ERROR = 6.5e-4


def _bounded_pole_batch_size(value):
    size = int(value)
    if not 1 <= size <= 8:
        raise ValueError("MPA pole_batch_size must be in [1, 8]")
    return size


def _batch_rows(row, batch):
    """Relocalize one window's pole ranges into a fixed batch-width carrier.

    The tau kernel's executable signature must not depend on how many poles a
    particular pane/product window selects.  Inactive rows therefore occupy
    the remaining batch slots with an impossible ``a`` interval.  All windows
    over a resident batch then call the same jitted callable with identical
    shapes, dtypes, and shardings; only the selector values change.
    """
    batch = tuple(int(p) for p in batch)
    local = {int(p): i for i, p in enumerate(batch)}
    keep = [i for i, p in enumerate(row.pole_indices) if int(p) in local]
    if not keep:
        return None
    capacity = len(batch)
    if len(keep) > capacity:
        raise ValueError(
            "one MPA window selects a resident pole more than once; "
            f"{len(keep)} selector rows exceed batch width {capacity}")
    count = len(keep)
    pole_indices = np.zeros(capacity, dtype=np.int32)
    pole_indices[:count] = [local[int(row.pole_indices[i])] for i in keep]
    # ``a > +inf`` is false for every finite pole.  Keeping all six values
    # finite-or-infinite (never NaN) also makes the inactive path harmless
    # under XLA predicate motion.
    bounds = np.broadcast_to(
        np.asarray((np.inf, -np.inf, np.inf, np.inf, -np.inf, -np.inf),
                   np.float64),
        (capacity, 6),
    ).copy()
    bounds[:count] = np.asarray(row.bounds[keep], np.float64)
    phase_real = np.zeros(capacity, dtype=bool)
    phase_real[:count] = np.asarray(row.phase_real[keep], bool)
    return (
        pole_indices,
        bounds,
        phase_real,
        None,
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
    brackets=None,
    band_counts=None,
    print_fn,
):
    """One spatial executor for streamed fit slabs."""
    omega = np.asarray(omega_grid_ry, np.float64)
    if omega.ndim != 1 or not omega.size:
        raise ValueError("omega_grid_ry must be a nonempty vector")

    s = wfns.slices
    bracketed = brackets is not None
    if bracketed:
        brackets = tuple(
            (int(lo), None if hi is None else int(hi))
            for lo, hi in brackets)
        if not brackets:
            raise ValueError("MPA Sigma band-bracket plan must be nonempty")
    face_kwargs = face_kernel_kwargs(wfns)
    if wfns.layout == "legacy":
        # ``sigma_sum``, not ``full`` — the Σ band sum, not the loaded
        # extent.  Identical on an unsplit deck.  UNVERIFIED on a split
        # one: the public MPA Σ still refuses to run (gw_config.
        # ComputeMode), so this line is wired for consistency and has
        # never executed under a split.
        state_slice = s.full if bracketed else s.sigma_sum
        psi_coh_xn, psi_coh_yr = wfns.xn(state_slice), wfns.yr(state_slice)
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
        spatial_shape = (int(psi_proj_xr.shape[0]),
                         int(psi_proj_xr.shape[1]),
                         int(psi_proj_yn.shape[3]))
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
        if bracketed and len(brackets) > 1:
            from gw.wavefunction_bundle import pack_band_window
            packed = [pack_band_window(wfns, lo, hi, mesh_xy=mesh_xy)
                      for lo, hi in brackets]
            psi_coh_xn = tuple(pair[0] for pair in packed)
            psi_coh_yr = tuple(pair[1] for pair in packed)
            pack_brackets = True
        else:
            psi_coh_xn, psi_coh_yr = wfns.psi_mun, wfns.psi_nmu
            pack_brackets = False
        psi_proj_xr, psi_proj_yn = wfns.psi_nmu, wfns.psi_mun
        spatial_shape = (int(meta.nk_tot), int(s.nb_full), int(s.nb_full))
    if wfns.layout == "legacy":
        pack_brackets = False
    if bracketed:
        shape = (len(brackets), omega.size, *spatial_shape)
        output_sharding = NamedSharding(
            mesh_xy, P(None, None, None, "x", "y"))
        sigma_shape = (len(brackets), *spatial_shape)
        sigma_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y"))
    else:
        shape = (omega.size, *spatial_shape)
        output_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y"))
        sigma_shape = spatial_shape
        sigma_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))
    accumulator = DeviceOmegaAccumulator(
        omega, shape=shape, sharding=output_sharding, omega_axis=0)
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    tau_kernel = get_shared_sigma_tau_kernel(
        mesh_xy=mesh_xy,
        kgrid=kgrid,
        brackets=brackets,
        pack_brackets=pack_brackets,
        **face_kwargs)
    small = NamedSharding(mesh_xy, P())

    n_sweeps = n_tau = 0
    logical_tau_pairs = 0
    batch_size = int(pole_batch_size)
    sweep_started = False
    for lo, Omega, B in batches:
        width = int(Omega.shape[0])
        batch = tuple(range(int(lo), int(lo) + width))
        for row in plan:
            selected = _batch_rows(row, batch)
            if selected is None:
                continue
            pole_indices, bounds, phase_real, _states = selected
            pole_indices, bounds, phase_real = (
                device_put_process_local(x, small)
                for x in (pole_indices, bounds, phase_real))
            win = row.window
            weight = getattr(row, "band_weight", None)
            if weight is None:
                selector = jnp.asarray(win.mask_A)
            else:
                selector = (jnp.asarray(win.mask_A, jnp.float64)
                            * jnp.reshape(
                                jnp.asarray(weight, jnp.float64),
                                np.asarray(win.mask_A).shape))
            if not sweep_started:
                first_t = np.asarray(
                    jax.device_get(win.nodes.t), np.complex128)[0]
                prewarm_args = (
                    psi_coh_xn, psi_coh_yr,
                    psi_proj_xr, psi_proj_yn,
                    row.E_A, selector, B, Omega,
                    pole_indices, bounds, phase_real,
                    jnp.asarray(win.E_ref_A),
                    jnp.asarray(win.E_ref_B),
                    jnp.asarray(first_t, dtype=jnp.complex128))
                if hasattr(tau_kernel, "lower"):
                    tau_kernel.lower(*prewarm_args).compile()
                else:
                    # The stage-split diagnostic is a Python dispatcher over
                    # separately-jitted stages.  Execute one real-shape call
                    # to prewarm the same kernels the timed sweep will use.
                    jax.block_until_ready(tau_kernel(*prewarm_args))
                accumulator.precompile_tau_add(
                    sigma_shape=sigma_shape,
                    sigma_sharding=sigma_sharding)
                print_fn(
                    "  MPA Sigma sweep begin: shared pane tau kernel "
                    "prewarmed")
                sweep_started = True
            accumulator.begin_window(
                win.nodes.t, win.nodes.alpha,
                omega_sign=win.omega_sign, prefactor=win.prefactor,
                e_ref_sum=win.E_ref_A + win.E_ref_B,
                antihermitian=(win.project_code == 1),
                omega_indices=row.omega_idx,
                omega_values=row.omega_abs)
            for t in np.asarray(
                    jax.device_get(win.nodes.t), np.complex128):
                sigma_tau = tau_kernel(
                    psi_coh_xn, psi_coh_yr,
                    psi_proj_xr, psi_proj_yn,
                    row.E_A, selector, B, Omega,
                    pole_indices, bounds, phase_real,
                    jnp.asarray(win.E_ref_A),
                    jnp.asarray(win.E_ref_B),
                    jnp.asarray(t, dtype=jnp.complex128))
                accumulator.add_tau(sigma_tau)
                n_tau += 1
            accumulator.end_window()
            n_sweeps += 1
            logical_tau_pairs += win.n_tau
        del B, Omega
        gc.collect()

    sigma = strip_sigma_window(
        accumulator.finalize(), nb_real, mesh_xy=mesh_xy)
    if bracketed:
        sigma = jax.jit(
            lambda values: jnp.cumsum(values, axis=0),
            out_shardings=sigma.sharding)(sigma)
        if band_counts is None:
            band_counts = tuple(
                int(s.nb_sigma_sum) if hi is None else int(hi)
                for _lo, hi in brackets)
        else:
            band_counts = tuple(int(count) for count in band_counts)
        if len(band_counts) != len(brackets):
            raise ValueError(
                "MPA Sigma band_counts must align with band brackets")
    transform_saving = int(logical_tau_pairs - n_tau)
    print_fn(
        f"  MPA Sigma: {n_tau} tau dispatches in {n_sweeps} sweeps "
        f"({n_poles} poles, batches of {batch_size}); "
        f"{transform_saving} undispatched logical tau; "
        f"panes and product windows used one shared tau kernel")
    return SigmaOmegaResult(
        omega_ry=omega,
        omega_ev=np.asarray(omega * RYD_TO_EV, np.float64),
        sigma_c_kij=sigma,
        band_counts=(() if band_counts is None else tuple(band_counts)))


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
    brackets=None,
    band_counts=None,
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

    ``brackets`` optionally partitions the intermediate-state band sum into
    disjoint slices.  The spatial kernel then returns a leading bracket axis;
    this executor inserts omega behind it and cumulatively sums the brackets
    before returning.  ``None`` preserves the ordinary MPA rank-4 result.
    """
    batch_size = _bounded_pole_batch_size(pole_batch_size)

    def batches(reader):
        for lo in range(0, int(n_poles), batch_size):
            hi = min(lo + batch_size, int(n_poles))
            Omega, B = reader.read(
                slice(lo, hi), unfold=True, return_sharded=True,
                to_unit="Ry")
            yield lo, Omega, B
            del Omega, B
            gc.collect()

    def run(reader):
        return _integrate_sigma_batches(
            wfns, batches(reader), int(n_poles), plan, omega_grid_ry, meta,
            mesh_xy, pole_batch_size=batch_size, brackets=brackets,
            band_counts=band_counts, print_fn=print_fn)

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
    quadrature_eps,
    quadrature_reduction_seconds,
    pair_ceiling,
    quadrature_cache_dir,
    omega_grid_step_ry,
    occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
    pole_batch_size=4,
    fit_identity=None,
    expected_screening_diagrams=None,
    occupation_state=None,
    sigma_branches=None,
    band_brackets=None,
    band_counts=None,
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

    Pole tensors are read collectively in their native sharding.  The box
    planner retains only exact live extrema from each configured pole batch;
    the pane control consumes the same bounded census.  The spatial executor
    then rereads and releases the pole ranges through one shared tau kernel.
    ``sigma_branches`` is an optional already-resolved set of causal branches;
    it lets another pole model retain its established occupation/Fermi policy
    while sharing this planner and executor exactly.  ``band_brackets`` and
    ``band_counts`` similarly carry the optional disjoint band-convergence
    partition.  They change neither pole interpretation nor window planning.
    """
    ledger = validate_fit_store(
        fit_src, expected_identity=fit_identity,
        expected_screening_diagrams=expected_screening_diagrams)
    n_poles = int(ledger["n_p"])
    pole_batch_size = _bounded_pole_batch_size(pole_batch_size)
    branches = (_branches(
        wfns, omega_grid_ry, efermi_ry,
        occupation_state=occupation_state,
        occupation_window_threshold=occupation_window_threshold)
        if sigma_branches is None else tuple(sigma_branches))
    plan_mode = resolve_sigma_plan()
    # ONE collective handle for the census walk, the planner, and the
    # executor walk — the whole Σ stage of this iteration.  The reader
    # does its h5py reads (ledger, unfold tables) before that handle
    # exists and none after, so no serial-h5py open on this store
    # overlaps or interleaves with the FFI one anywhere inside a Σ stage
    # (audit A1; hdf5_owner enforces it).  The context manager is the
    # release path: a refusal from the planner or the executor must still
    # close the handle on every rank.
    with open_pole_reader(fit_src, mesh_xy=mesh_xy) as reader:
        # One bounded extrema census serves both routes.  In particular, the
        # production route does not read residues into a host histogram and
        # never constructs a sampled state-pole lattice.
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
            gc.collect()
        if plan_mode == "panes":
            plan, geometry = build_shared_sigma_windows(
                summaries, branches,
                regularization_width_ry=regularization_width_ry,
                edge_factor=edge_factor,
                target_error=_PANE_CONTROL_TARGET_ERROR,
                max_rank=int(pair_ceiling),
                crossing_max_nodes=max(
                    CROSSING_NODE_FLOOR, int(pair_ceiling)),
                omega_grid_step_ry=omega_grid_step_ry,
                occupation_window_threshold=occupation_window_threshold)
        else:
            plan, geometry = plan_sigma_windows(
                summaries, branches, omega_grid_ry,
                regularization_width_ry,
                eps=quadrature_eps,
                reduction_seconds=quadrature_reduction_seconds,
                pair_ceiling=pair_ceiling,
                cache_dir=quadrature_cache_dir,
                print_fn=print_fn, edge_factor=edge_factor)
        if plan_mode == "panes":
            print_fn(
                f"  MPA windows: eta={geometry['eta_ry'] * RYD_TO_EV:.4f} eV, "
                f"{geometry['n_windows']} logical windows")
        else:
            print_fn(
                f"  MPA windows [box]: "
                f"eta={geometry['eta_ry'] * RYD_TO_EV:.4f} eV, "
                f"eps={geometry['eps']:.3g}, "
                f"certificate={geometry['rule_eps']:.3g}, "
                f"{geometry['n_windows']} logical windows, "
                f"{geometry['window_tau_pairs']} (window,tau) pairs, "
                f"{geometry['distinct_tau_count']} branch-distinct tau, "
                f"cache={geometry['cache_dir'] or 'off'}")
            for branch in geometry["branches"]:
                for window in branch["windows"]:
                    print_fn(
                        f"    {window['name']}: n_tau={window['node_count']}, "
                        f"box={tuple(window['box_ry'])} Ry, "
                        f"sup={window['sup_error']:.6g}/"
                        f"{window['eps']:.6g} ({window['criterion']}), "
                        f"kappa_max={window['kappa_max']:.6g}, "
                        f"noise={window['runtime_noise_bound']:.6g}/"
                        f"{window['runtime_noise_budget']:.6g}")
        return integrate_sigma_store(
            wfns, reader, n_poles, plan, omega_grid_ry, meta, mesh_xy,
            pole_batch_size=pole_batch_size, brackets=band_brackets,
            band_counts=band_counts, print_fn=print_fn)


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
