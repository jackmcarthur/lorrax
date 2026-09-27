"""SPCOST scratch instrument, not production. Ry operators, eV diagnostics.

Fixed DFT P/R masks; vectorized JAX matrix edits on the existing 2-D band
layout. Only bounded (k,band) diagnostics are gathered. The Sigma consumer
still computes its square cube: rectangular costs are counts, not timings.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import device_put_process_local, gather_to_host
from common.units import RYD_TO_EV

CANDIDATE = None
CONTEXT = None


def begin(inputs, state, ks, indices, energies, U, classes):
    global CONTEXT
    if CANDIDATE is None:
        return
    from .qp_support import sigma_window_states, quasiparticle_mask
    from .scissor import k_star_weights
    from .qsgw_utils import static_sigma_diag_to_host
    e = np.asarray(inputs.e_dft_active_kn_ry)
    val = np.asarray(inputs.valence_mask_active_kn)
    if not ks.is_identity:
        e, val = ks.select(e), ks.select(val)
    p = sigma_window_states((e-float(inputs.wfn.efermi))*RYD_TO_EV,
                            CANDIDATE['window_ev'], tol_ev=1e-4)
    p[:, :int(inputs.config.sc.frozen_core_bands)] = False
    if classes is not None:
        val, cross = classes.masks(e.shape)
    else:
        cross = np.zeros_like(p)
    z = np.ones_like(e) if state.tail_z_kn is None else np.asarray(state.tail_z_kn)
    fit = p & ~val & ~cross & quasiparticle_mask(z)
    weights = k_star_weights(ks)[:, None] * np.where(fit, z, 0.)
    pcol = np.zeros(p.shape, dtype=float)
    np.put_along_axis(pcol, indices, p.astype(float), axis=1)
    pfull = pcol if ks.is_identity else pcol[np.asarray(ks.take)]
    rep = NamedSharding(inputs.mesh_xy, P(None, None))
    pj = device_put_process_local(p.astype(float), rep)
    leak_sorted = _leak_kernel(inputs.mesh_xy)(U, pj)
    leak = np.take_along_axis(np.asarray(gather_to_host(leak_sorted)), indices, axis=1)
    diag_in = static_sigma_diag_to_host(state.H_qp_dft, inputs.mesh_xy).real
    r_empty = ~p & ~val & ~cross
    wr = k_star_weights(ks)[:, None] * r_empty
    beta_prev = float(np.sum(wr*(diag_in-e))/max(np.sum(wr), 1e-300))
    CONTEXT = dict(p=p, pfull=pfull, e=e, delta=energies/RYD_TO_EV-e,
                   weights=weights, leak=leak, beta_prev=beta_prev,
                   empty=~val & ~cross, iteration=int(state.iteration))


@lru_cache(None)
def _leak_kernel(mesh):
    rep = NamedSharding(mesh, P(None, None))
    @jax.jit
    def calc(U, p):
        result = jnp.sum(jnp.abs(U)**2 * (1-p)[:, :, None], axis=1)
        return jax.lax.with_sharding_constraint(result, rep)
    return calc


@lru_cache(None)
def _edit_kernel(mesh, coupling, lowdin):
    sh = NamedSharding(mesh, P(None, 'x', 'y'))
    rep = NamedSharding(mesh, P(None, None))
    @jax.jit
    def edit(H, p, target):
        diag = jnp.real(jnp.diagonal(H, axis1=-2, axis2=-1))
        pp = p[:, :, None]*p[:, None, :]
        pr = p[:, :, None]*(1-p)[:, None, :]
        keep = pp if coupling == 'none' else pp+pr+jnp.swapaxes(pr,-1,-2)
        denom = target[:, None, :] - diag[:, :, None]
        safe = jnp.where(pr > 0, denom, 1.)
        # A genuine resonance is returned for the host refusal below.
        min_denom = jnp.min(jnp.where(pr > 0, jnp.abs(denom), jnp.inf))
        ct = jnp.sum(jnp.where(pr > 0, jnp.abs(H)**2 / safe, 0.), axis=1) if lowdin else jnp.zeros_like(diag)
        desired = jnp.where(p > 0, diag, target-ct)
        eye = jnp.eye(H.shape[-1])[None]
        result = H * keep * (1-eye) + desired[:, :, None]*eye
        return (jax.lax.with_sharding_constraint(result, sh),
                jax.lax.with_sharding_constraint(ct, rep), min_denom)
    return edit


def finish(H, inputs):
    if CANDIDATE is None:
        return H
    from .qsgw_utils import static_sigma_diag_to_host
    c = CONTEXT
    p,e,w = c['p'],c['e'],c['weights']
    model = CANDIDATE['scissor']
    coupling = CANDIDATE['coupling']
    if model not in ('s1','s2','s3','s4','s5') or coupling not in ('none','energy','zero'):
        raise ValueError('SPCOST: unknown candidate')
    diag = static_sigma_diag_to_host(H, inputs.mesh_xy).real
    delta = diag-e if model in ('s3','s4') else c['delta']
    if model == 's2':
        if np.any((w > 0) & (1-c['leak'] < 1e-8)):
            raise ValueError('SPCOST leak correction has no protected weight')
        delta = (delta-c['leak']*c['beta_prev']) / np.maximum(1-c['leak'], 1e-8)
    total = max(float(w.sum()), 1e-300)
    beta = float(np.sum(w*delta)/total)
    center = float(np.sum(w*e)/total)
    slope = 0.
    if model == 's5':
        var = float(np.sum(w*(e-center)**2))
        slope = float(np.sum(w*(e-center)*(delta-beta))/var) if var > 1e-20 else 0.
    target = e + np.where(c['empty'], beta+slope*(e-center), 0.)
    rep = NamedSharding(inputs.mesh_xy, P(None, None))
    out, ct, min_denom = _edit_kernel(inputs.mesh_xy, coupling, model=='s4')(
        H, device_put_process_local(p.astype(float), rep),
        device_put_process_local(target, rep))
    min_denom = float(min_denom)*RYD_TO_EV
    if model == 's4' and min_denom < 1e-6:
        raise ValueError(f'SPCOST Lowdin resonance: denominator {min_denom} eV')
    ct = np.asarray(gather_to_host(ct))
    receipt = dict(iteration=c['iteration'], candidate=CANDIDATE,
                   beta_ev=beta*RYD_TO_EV, slope=slope, nfit=int((w>0).sum()),
                   leak_max=float(np.max(np.where(w>0,c['leak'],0))),
                   counterterm_max_ev=float(np.max(np.abs(ct)))*RYD_TO_EV,
                   protected_count=int(p.sum()), total_count=int(p.size),
                   pp_elements=int(np.sum(p.sum(axis=1)**2)),
                   p_all_elements=int(np.sum(p.sum(axis=1)*p.shape[1])),
                   square_elements=int(p.shape[0]*p.shape[1]**2))
    if jax.process_index() == 0:
        Path(f'spcost_map{c["iteration"]:04d}.json').write_text(json.dumps(receipt, indent=2)+'\n')
        np.savez(f'spcost_map{c["iteration"]:04d}.npz', e_dft_ev=e*RYD_TO_EV,
                 protected=p, diagonal_ev=diag*RYD_TO_EV, leak=c['leak'],
                 target_ev=target*RYD_TO_EV, weights=w, counterterm_ev=ct*RYD_TO_EV)
    return out
