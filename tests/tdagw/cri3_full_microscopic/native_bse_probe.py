"""Campaign BSE check on an authenticated, live accepted SC input frame.

Called collectively by the frozen-state native GW probe after its W producer
returns. This assembles the existing BSE payload, not another BSE kernel or
screening solve. A nested coarse bosonic grid uses its explicit ordinary-q
quadrature comb, not W interpolation, on the fine electronic BSE grid.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time


def source_fit_g0(source):
    """Pack the exact canonical source-fit Gamma vector, including inert pads.

    The native Coulomb owner returns canonical logical OR suffix-padded
    metadata, not orbit-packed metadata. Only the basis owner packs it.
    No vector is inferred from V, refitted, or reconstructed from a head.
    """
    import numpy as np
    meta = source["meta"]
    basis = meta.mu_basis
    canonical = np.asarray(source.get("G0_canonical"))
    logical, carrier = int(basis.n_logical), int(basis.n_canonical)
    if (canonical.ndim != 1 or logical != int(meta.n_rmu)
            or canonical.size not in (logical, carrier)
            or not np.isfinite(canonical).all()
            or np.any(canonical[logical:] != 0)):
        raise ValueError("BSE needs the exact source-fit G0 vector: canonical logical or "
                         "basis suffix-padded extent with exact-zero padding; "
                         f"got shape {canonical.shape}, logical={logical}, canonical={carrier}")
    receipt = source.get("G0_receipt")
    if receipt is not None:
        if (receipt.get("canonical_value_sha256") != hashlib.sha256(
                np.ascontiguousarray(canonical).tobytes()).hexdigest()
                or receipt.get("canonical_centroids_sha256") != hashlib.sha256(
                    np.ascontiguousarray(basis.canonical_indices).tobytes()).hexdigest()
                or receipt.get("logical_extent") != logical
                or receipt.get("canonical_extent") != carrier
                or receipt.get("packed_extent") != int(basis.n_packed)
                or receipt.get("order") != "canonical_suffix_padded"
                or receipt.get("shape") != list(canonical.shape)
                or receipt.get("dtype") != str(canonical.dtype)
                or receipt.get("producer") != "gw.gw_init._finalize_vq_views"
                or receipt.get("q_fractional") != [0., 0., 0.]):
            raise ValueError("BSE source-fit G0 producer/basis/value receipt mismatch")
    packed = basis.pack_host(canonical)
    if (packed.shape != (int(basis.n_packed),)
            or np.any(packed[~np.asarray(basis.active_mask, bool)] != 0)):
        raise ValueError("BSE source-fit G0 packed extent/pad mismatch")
    return packed


def bright_subgap_summary(energies_ev, residual_ev, dipoles, *, gap_ev, nk):
    """Identify resolved bright bound states; strengths are bohr² per cell."""
    import numpy as np
    from bse.absorption_common import inplane_helicity
    energy, residual = np.asarray(energies_ev), np.asarray(residual_ev)
    d = np.asarray(dipoles)
    if (energy.ndim != 1 or residual.shape != energy.shape
            or d.shape != (energy.size, 3) or nk <= 0 or gap_ev <= 0
            or not all(np.isfinite(a).all() for a in (energy, residual, d))):
        raise ValueError("BSE summary needs finite matched roots, residuals and dipoles")
    ip, im, _ = inplane_helicity(d, nk)
    strength = np.abs(d) ** 2 / nk
    # These are report/target-selection thresholds, not solver tolerances.
    ok = ((energy > 0) & (energy + residual < gap_ev)
          & (residual <= 1.e-3) & (strength[:, :2].sum(axis=1) > 1.e-8))
    eligible = np.flatnonzero(ok)
    target = (None if not eligible.size else int(eligible[
        np.argmax(strength[eligible, :2].sum(axis=1))]))
    return dict(gap_ev=float(gap_ev), energies_ev=energy.tolist(),
        residual_norm_ev=residual.tolist(), strength_xyz_bohr2_cell=strength.tolist(),
        strength_sigma_plus_bohr2_cell=ip.tolist(),
        strength_sigma_minus_bohr2_cell=im.tolist(),
        bright_subgap_root_indices=eligible.tolist(), target_root_index=target,
        target_energy_ev=None if target is None else float(energy[target]),
        scope="finite-window TDA BSE target check, not a converged optical spectrum",
        optical_approximation="full-protected U^H v_DFT U divided by current SC gap")


def band_pair_summary(pair_power, *, nocc):
    """Small physical-band weights after summing native eigenvector power over k.

    Input is [root, conduction, valence], with padding already excluded.
    Weights diagnose containment within THIS finite-window BSE solution;
    they are not a larger-window convergence or optical-oscillator certificate.
    """
    import numpy as np
    power = np.asarray(pair_power, float)
    if (power.ndim != 3 or power.size > 65536 or not np.isfinite(power).all()
            or np.any(power < 0) or power.shape[-1] > int(nocc)):
        raise ValueError("BSE band weights require bounded finite nonnegative physical pair power")
    norm = power.sum(axis=(1, 2))
    if np.any(norm <= 0):
        raise ValueError("BSE band weights require nonzero physical eigenvectors")
    nc, nv = power.shape[1:]
    c, v = np.arange(nocc, nocc+nc), np.arange(nocc-nv, nocc)
    weights = power/norm[:, None, None]
    core = ((c < nocc+2)[:, None] & (v >= nocc-2)[None, :])
    return dict(conduction_bands_zero_based=c.tolist(), valence_bands_zero_based=v.tolist(),
        physical_norm=norm.tolist(), normalized_pair_weights=weights.tolist(),
        central_2v2c_weight=(weights*core[None]).sum(axis=(1, 2)).tolist(),
        scope="k-summed eigenvector power in this finite BSE window; not a larger-window convergence test")


def run_bse_probe(source, *, out, nval=4, ncond=4, nroots=24, maxiter=200):
    """Solve and check the native TDA BSE on the same full-q accepted SC state.

    ``source`` is the frozen-state probe's live bundle. Large ψ/V/W stay
    distributed on its (x,y) mesh. E, occupations, U and the protected velocity
    are small band metadata; no centroid-square array is gathered to a host.
    All energies and velocities entering native owners are Ry and Ry bohr.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host, device_put_process_local, rank0_transaction
    from common.band_degeneracy import resolve_band_window
    from common.units import RYD_TO_EV
    from runtime.padding import pad_axis
    from symmetry_maps import QirrOperator
    from gw.ppm_sigma import fresh_shared_pole_capacity
    from gw.static_screening import build_static_screened_w
    from bse.bse_head import _inject_q0_head
    from bse.bse_preconditioner import compute_pair_amplitude
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_stack_matvec import build_bse_stack_matvec
    from bse.absorption_common import (build_dipole_vector_bse,
                                       exciton_dipoles_distributed)
    from bse import bse_lanczos

    start = time.perf_counter()
    mesh, meta, config = source["mesh"], source["meta"], source["config"]
    # Refuse a missing/mis-associated vector BEFORE W0 or BSE allocations.
    packed_g0 = source_fit_g0(source)
    grid = tuple(int(x) for x in meta.kgrid)
    qgrid = tuple(int(x) for x in (getattr(meta, "qgrid", None) or grid))
    if qgrid != grid:
        from common.fft_helpers import make_bosonic_q_comb
        comb_program = make_bosonic_q_comb(mesh, grid, qgrid)
        qsym = getattr(meta, "q_sym", None)
        nq = int(np.prod(qgrid))
        if (qsym is None or not np.array_equal(qsym.q_irr_full_idx, np.arange(nq))
                or not np.array_equal(qsym.q_irr_kgrid_int,
                    np.indices(qgrid).reshape(3, -1).T)):
            raise ValueError("BSE coarse quadrature needs canonical unshifted full-q rows")
    if not source["frame"].get("reference_ready", False):
        raise ValueError("BSE target requires an authenticated converged input reference")
    wfns, energy, f = source["wfns"], np.asarray(source["energy_ry"]), np.asarray(source["occupations"])
    nk = int(np.prod(grid))
    if (energy.shape != f.shape or energy.shape[0] != nk
            or not np.all((f == 0) | (f == 1))):
        raise ValueError("BSE probe requires the insulating full-k SC energy/occupation frame")
    nocc = int(f[0].sum())
    if not np.all(f[:, :nocc] == 1) or not np.all(f[:, nocc:] == 0):
        raise ValueError("BSE occupied boundary must be common to every SC k row")
    U = np.asarray(gather_to_host(source["U_active"]))
    if U.ndim != 3 or U.shape[:2] != (nk, U.shape[-1]):
        raise ValueError("BSE needs the whole protected SC rotation")
    nval, ncond = resolve_band_window(energy[:, :U.shape[-1]], nocc,
        nval, ncond, mode="snap", where="accepted SC BSE probe")
    if nocc+ncond > U.shape[-1]:
        raise ValueError("BSE window exceeds the protected SC eigensystem")
    carrier = wfns.green_parent
    if (carrier is None or int(carrier.plan.n_parent) != nk
            or not np.array_equal(np.asarray(carrier.plan.irr_idx), np.arange(nk))):
        raise ValueError("BSE probe requires the reconstructed full-k native carrier")
    # The carrier ALREADY contains the full protected rotation. Never apply U
    # again, nor slice DFT orbitals before rotating their whole protected block.
    psi = carrier.projection_faces()[0]
    sh = make_bse_shardings(mesh)
    nvpad = ((nval+int(mesh.shape['y'])-1)//int(mesh.shape['y']))*int(mesh.shape['y'])
    ncpad = ((ncond+int(mesh.shape['x'])-1)//int(mesh.shape['x']))*int(mesh.shape['x'])
    pv = jax.jit(lambda p: pad_axis(p[:, nocc-nval:nocc], int(mesh.shape['y']), axis=1).array,
                 out_shardings=sh.psi_x)(psi)
    pc = jax.jit(lambda p: pad_axis(p[:, nocc:nocc+ncond], int(mesh.shape['x']), axis=1).array,
                 out_shardings=sh.psi_x)(psi)
    pvy = jax.jit(lambda p: p, out_shardings=sh.psi_y)(pv)
    pcy = jax.jit(lambda p: p, out_shardings=sh.psi_y)(pc)
    V = QirrOperator.of(source["V_q"])
    if not V.is_whole_zone() or V.n_full != nk:
        raise ValueError("BSE probe needs full fine-grid bare V")
    head = source["W_by_role"]["iteration_head"]
    interaction = V if qgrid == grid else QirrOperator.of(source["V_screen"])
    if not interaction.is_whole_zone() or interaction.n_full != int(np.prod(qgrid)):
        raise ValueError("BSE W0 needs the matching full bosonic-grid bare interaction")
    wm = fresh_shared_pole_capacity(meta, mesh)
    # All ranks bind the real allocator room while the source's full-band
    # faces/V/velocity remain live. This ledger prices static W synthesis,
    # not the later BSE payload/workspaces or a measured material peak.
    wm.shared_pole_capacity.at_phase()
    W0, hs = build_static_screened_w(None, interaction, config=config, meta=wm,
        mesh_xy=mesh, shared_pole=source["W_by_role"]["shared_pole"], head_resolver=head)
    wvalues = QirrOperator.of(W0).unfold(mesh)
    W = jax.jit(lambda w: jnp.moveaxis(w, 0, -1).reshape(w.shape[1], w.shape[2], *qgrid),
                out_shardings=sh.W)(wvalues)
    v0 = jax.jit(lambda v: v[0], out_shardings=sh.V)(V.values)
    # G0 is ALREADY small replicated host metadata from its native producer.
    # Use its host packing owner: the large-array staged conversion requires
    # a second slab axis and is not the vector-metadata path.
    gx = device_put_process_local(packed_g0, NamedSharding(mesh, P('x')))
    gy = device_put_process_local(packed_g0, NamedSharding(mesh, P('y')))
    # direct_at defaults explicitly to the ELECTRONIC cell (vcoul adapter),
    # even when meta.qgrid is coarse; hs is the BOSONIC W-cell sample.
    bare_sample = hs if qgrid == grid else source["head_resolver"].direct_at(0.0j)
    v0, W, head_log = _inject_q0_head(v0, W, gx, gy, bare_sample.vc0,
        jnp.asarray([hs.wcoul0]), float(meta.cell_volume), w0_ready=True)
    comb_receipt = None
    if qgrid != grid:
        # Inject BEFORE embedding: both W body and its single coarse q0 head
        # acquire Nk/Nq. Bare fine-grid exchange is neither tiled nor scaled.
        flat = jax.jit(lambda w: jnp.moveaxis(w.reshape(w.shape[0], w.shape[1], -1), -1, 0),
            out_shardings=NamedSharding(mesh, P(None, 'x', 'y')))(W)
        embed = jax.jit(lambda w: jnp.moveaxis(comb_program(w), 0, -1).reshape(
            w.shape[1], w.shape[2], *grid), out_shardings=sh.W)
        compiled = embed.lower(flat).compile()
        stats = compiled.memory_analysis()
        if stats is None:
            raise ValueError("BSE coarse q-comb needs its actual compiled memory price")
        from runtime.aot_memory import compiled_new_bytes
        new_bytes = compiled_new_bytes(compiled)
        wm.shared_pole_capacity.at_phase()
        resident = max(0, int(stats.output_size_in_bytes)-int(stats.alias_size_in_bytes))
        comb_receipt = wm.shared_pole_capacity.reserve("bse.coarse_q_comb",
            resident_bytes_per_rank=resident,
            workspace_bytes_per_rank=max(0, new_bytes-resident))
        W = compiled(flat)
        del flat
    ev = pad_axis(energy[:, nocc-nval:nocc], int(mesh.shape['y']), axis=1, fill=-1.e6).array
    ec = pad_axis(energy[:, nocc:nocc+ncond], int(mesh.shape['x']), axis=1, fill=1.e6).array
    data = dict(psi_c_X=pc, psi_c_Y=pcy, psi_v_X=pv, psi_v_Y=pvy,
        eps_c=device_put_process_local(np.asarray(ec), sh.eps),
        eps_v=device_put_process_local(np.asarray(ev), sh.eps), V_q0=v0, W_q=W,
        M=jax.jit(compute_pair_amplitude, out_shardings=sh.M)(pc, pv),
        nkx=grid[0], nky=grid[1], nkz=grid[2], n_cond=ncond, n_val=nval,
        n_cond_pad=ncpad, n_val_pad=nvpad)
    # Observe the existing sole donated IFFT return, retaining the SAME WR
    # buffer for a true post-solve residual. No second transform or W copy.
    held = {}
    original = bse_lanczos.get_donated_kfft_kminor
    def observe_ifft(*args, **kwargs):
        fn = original(*args, **kwargs)
        def apply(w):
            held['wr'] = fn(w)
            return held['wr']
        return apply
    bse_lanczos.get_donated_kfft_kminor = observe_ifft
    try:
        values, vectors, iterations = bse_lanczos.solve_bse_sharded(data, mesh,
            n_eig=nroots, max_iter=maxiter, include_W=True, block_size=1,
            rtol=1.e-8, check_every=4, n_reorth=maxiter, tda=True)
    finally:
        bse_lanczos.get_donated_kfft_kminor = original
    matvec = build_bse_stack_matvec(mesh, *grid, kernel='bse')
    args = (pc, pcy, pv, pvy, data['eps_c'], data['eps_v'], held['wr'], v0, data['M'])
    residuals = []
    for i in range(nroots):
        x = vectors[i, 0][None]
        hx = matvec(x, *args)
        residuals.append(jnp.linalg.norm(hx-values[i]*x))
    residual_ev = np.asarray(gather_to_host(jnp.stack(residuals)))*RYD_TO_EV
    # Exact native head DFT velocity, already authenticated/distributed by SC.
    nb = U.shape[-1]
    vd = jax.jit(lambda v: v[:, :, :nb, :nb],
                 out_shardings=NamedSharding(mesh, P()))(source['velocity_dft_cart'])
    vd = np.asarray(gather_to_host(vd))
    uc, uv = U[:, :, nocc:nocc+ncond], U[:, :, nocc-nval:nocc]
    velocity = np.einsum('kmi,akmn,knj->akij', uc.conj(), vd, uv, optimize=True)
    de = energy[:, nocc:nocc+ncond, None]-energy[:, None, nocc-nval:nocc]
    d = device_put_process_local(
        build_dipole_vector_bse(velocity/de[None], ncpad, nvpad), sh.X)
    dipoles = exciton_dipoles_distributed(vectors, d, nroots)
    gap = (energy[:, nocc:].min()-energy[:, :nocc].max())*RYD_TO_EV
    report = bright_subgap_summary(np.asarray(gather_to_host(values))*RYD_TO_EV,
        residual_ev, dipoles, gap_ev=gap, nk=nk)
    pair_power = jax.jit(lambda x: jnp.sum(
        jnp.abs(x[:, 0, :ncond, :nval])**2, axis=-1))(vectors)
    report["band_pair_weights"] = band_pair_summary(
        np.asarray(gather_to_host(pair_power)), nocc=nocc)
    report.update(nval=int(nval), ncond=int(ncond), nk=nk,
        bosonic_grid=list(qgrid), electronic_grid=list(grid),
        response_qgrid=list(qgrid), electronic_kgrid=list(grid),
        response_window=getattr(meta, "qgrid_response_window", "sharp"),
        q_quadrature="ordinary coarse full-BZ q sum; NOT WS interpolation",
        q_comb_weight=float(nk/np.prod(qgrid)), q_comb_capacity=comb_receipt,
        bare_head_grid=list(grid), screened_head_grid=list(qgrid),
        head_applications=dict(bare=1, screened=1),
        iterations=int(iterations), maxiter=maxiter, head=head_log,
        screening_model=dict(source['W_by_role']['shared_pole']),
        frame=source['frame'], elapsed_s=time.perf_counter()-start)
    # Host-only metadata commit on rank0; every device/collective above ran on all ranks.
    destination = Path(out)
    def publish():
        destination.mkdir(parents=True, exist_ok=False)
        with (destination/'bright_subgap.json').open('x') as stream:
            json.dump(report, stream, indent=2, default=str, allow_nan=False)
    rank0_transaction(destination, stage='native_bse.bright_subgap_report', write=publish)
    return report
