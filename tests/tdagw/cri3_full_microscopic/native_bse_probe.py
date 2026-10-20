"""Campaign BSE check on an authenticated, live accepted SC input frame.

Called collectively by the frozen-state native GW probe after its W producer
returns. This assembles the existing BSE payload, not another BSE kernel or
screening solve. It initially admits only equal electronic and bosonic grids.
"""
from __future__ import annotations

import json
from pathlib import Path
import time


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
    grid = tuple(int(x) for x in meta.kgrid)
    if tuple(getattr(meta, "qgrid", None) or grid) != grid:
        raise ValueError("BSE probe presently requires qgrid=kgrid; no implicit W interpolation")
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
    wm = fresh_shared_pole_capacity(meta, mesh)
    # All ranks bind the real allocator room while the source's full-band
    # faces/V/velocity remain live. This ledger prices static W synthesis,
    # not the later BSE payload/workspaces or a measured material peak.
    wm.shared_pole_capacity.at_phase()
    W0, hs = build_static_screened_w(None, V, config=config, meta=wm,
        mesh_xy=mesh, shared_pole=source["W_by_role"]["shared_pole"], head_resolver=head)
    wvalues = QirrOperator.of(W0).unfold(mesh)
    W = jax.jit(lambda w: jnp.moveaxis(w, 0, -1).reshape(w.shape[1], w.shape[2], *grid),
                out_shardings=sh.W)(wvalues)
    v0 = jax.jit(lambda v: v[0], out_shardings=sh.V)(V.values)
    # Native Coulomb owner captured this exact small G0 before returning V.
    canonical = np.asarray(source["G0_canonical"])
    basis = meta.mu_basis
    if canonical.ndim != 1 or canonical.size != int(meta.n_rmu):
        raise ValueError("BSE needs the exact source-fit G0 vector")
    # G0 is ALREADY small replicated host metadata from its native producer.
    # Use its host packing owner: the large-array staged conversion requires
    # a second slab axis and is not the vector-metadata path.
    packed_g0 = basis.pack_host(canonical)
    gx = device_put_process_local(packed_g0, NamedSharding(mesh, P('x')))
    gy = device_put_process_local(packed_g0, NamedSharding(mesh, P('y')))
    v0, W, head_log = _inject_q0_head(v0, W, gx, gy, hs.vc0,
        jnp.asarray([hs.wcoul0]), float(meta.cell_volume), w0_ready=True)
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
    report.update(nval=int(nval), ncond=int(ncond), nk=nk,
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
