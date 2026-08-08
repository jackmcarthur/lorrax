"""Certify the three converted call sites against the per-k local plan.

``common.mtxel_sweep`` now serves the three sweeps that used to call
``collectives.gather_k_blocks``:

  ⟨mk|V_H|nk⟩        ``gw.kin_ion_io.compute_hartree_matrix``
  ⟨mk|T+V_loc+V_NL|nk⟩  ``gw.kin_ion_io.main``
  ⟨mk|v|nk⟩          ``psp.get_dipole_mtxels.main``

This gate rebuilds each one BOTH ways on a real deck — the local plan
(whole-band FFT box on one rank, the route being replaced) and the
2-D band-sharded k-scan — and compares them PER SHARD at 1e-12
relative.  Per shard because a global ``device_get`` would gather the
very tile the sweep exists to remove, and a rank-0-only check passes a
plan that puts the right answer on the wrong rank.

Bit-identity is not the criterion: the band sharding reassociates the G
sum, and ``sum_operators`` moves one scalar normalisation from after the
G sum to before it.  Owner tolerance is 1e-12 relative.

Five questions:

  1. V_H         — the FFT round trip, wired to the loader's ψ + D10 table.
  2. kin+ion     — T + V_loc + V_NL summed ON THE KET in one sweep.  Also
                   the answer to "did V_NL fit the Operator protocol".
  3. dipole      — 2(k+G)ψ − ∂V_NL/∂K ψ, three Cartesian components on
                   one sweep, with the component axis replicated.
  4. the boundary — ``blocks_to_host`` (replicated and owner_only) against
                   the same reference, including the band-pad trim.
  5. end to end  — ``compute_hartree_matrix`` itself, the edited function.

Plus: one compile for the whole k sweep, and per-rank VmHWM — a plan
that merely RELOCATED a peak looks clean on rank 0 alone.

Env: MTX_DECK (input file), MTX_NB.
"""
import os
import sys
import time

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from common import Meta, symmetry_maps                        # noqa: E402
from common.collectives import (process_count, process_rank,   # noqa: E402
                                resolve_mesh)
from common.jax_compile_cache import _STATE as CC_STATE        # noqa: E402
from common.mtxel_sweep import (SweepGeometry, band_sphere_spec,  # noqa: E402
                                blocks_to_host, dipole_operator,
                                kinetic_operator,
                                local_potential_operator, sum_operators,
                                sweep_matrix_elements, vnl_operator)
from common.wfn_transforms import load_kpoint_fftbox_local     # noqa: E402
from ffi import _services      # noqa: F401,E402  (path bootstrap; dies
                                 # with the owner's workspace fix)
_services.ensure_on_path()
from wfn_loader import WfnLoader                               # noqa: E402
from gw.gw_config import read_lorrax_input                     # noqa: E402
from gw.kin_ion_io import (build_valence_density_distributed,   # noqa: E402
                           compute_hartree_matrix, get_kin_ion_k)
from psp.dft_operators import padded_gvectors                  # noqa: E402
from psp.get_DFT_mtxels import (build_hartree_potential,        # noqa: E402
                                compute_local_V_k,
                                spin_degeneracy_factor)
from psp.get_dipole_mtxels import (compute_p_operator_k,        # noqa: E402
                                   compute_vnl_velocity_cart)
from psp.pseudos import (build_atom_pp_assignments,             # noqa: E402
                         load_pseudopotentials)
from psp.radial.build_projectors_qe import (                    # noqa: E402
    build_local_ionic_potential_on_G_total)
import psp.vnl_ops as vnl_ops                                  # noqa: E402

DECK = os.environ.get(
    "MTX_DECK", "/scratch2/08271/jackmc/mos2_4x4_test/deck_b300.in")
NB = int(os.environ.get("MTX_NB", "28"))
RTOL = 1e-12


def vmhwm_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / (1024.0 * 1024.0)
    return float("nan")


def shard_vs_reference(H_sharded, ref, label, p0):
    """Per-shard max relative delta.  Each rank checks ITS OWN block."""
    worst = 0.0
    for sh in H_sharded.addressable_shards:
        blk = np.asarray(sh.data)
        r = ref[sh.index]
        scale = max(float(np.abs(r).max()), 1e-300)
        worst = max(worst, float(np.abs(blk - r).max()) / scale)
    p0(f"[mtxel] {label:<38} per-shard max rel = {worst:.3e}   "
       f"{'PASS' if worst <= RTOL else 'FAIL'}")
    return worst


def host_vs_reference(A, ref, label, p0):
    if A is None:                       # owner_only peer: nothing to check
        return 0.0
    scale = max(float(np.abs(ref).max()), 1e-300)
    d = float(np.abs(np.asarray(A) - ref).max()) / scale
    p0(f"[mtxel] {label:<38} host     max rel = {d:.3e}   "
       f"{'PASS' if d <= RTOL else 'FAIL'}")
    return d


def main():
    rank, world = process_rank(), process_count()
    p0 = print if rank == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    px, py = (int(s) for s in mesh.devices.shape)

    # ---- deck ------------------------------------------------------------
    deck_dir = os.path.dirname(os.path.abspath(DECK))
    params = read_lorrax_input(DECK)
    wfn_path = params.get("wfn_file", "WFN.h5")
    if not os.path.isabs(wfn_path):
        wfn_path = os.path.join(deck_dir, wfn_path)
    wfn = WfnLoader(wfn_path)
    wfn.adopt_mesh(mesh)
    sym = symmetry_maps.SymMaps(wfn)
    nval = int(params.get("nval", 5))
    ncond = int(params.get("ncond", 5))
    bispinor = bool(params.get("bispinor", False))
    truncation_2d = int(params.get("sys_dim", 3)) == 2
    nb = min(NB, int(wfn.nbands))
    meta = Meta.from_system(wfn, sym, nval, ncond, nb, 0, bispinor)
    nk = int(sym.nk_tot)
    grid = tuple(int(s) for s in meta.fft_grid)
    p0(f"[mtxel] world={world} mesh=({px},{py}) deck={os.path.basename(DECK)} "
       f"nk={nk} nb={nb} ns={int(wfn.nspinor)} grid={grid} "
       f"nelec={int(wfn.nelec)} bispinor={bispinor}")

    pseudos = load_pseudopotentials(deck_dir)
    if not pseudos:
        raise SystemExit(f"no .upf in {deck_dir}")

    # ---- k-independent potentials, exactly as the drivers build them -----
    assignments = build_atom_pp_assignments(
        jnp.asarray(np.asarray(wfn.atom_crys, dtype=float)),
        jnp.asarray(np.asarray(wfn.atom_types, dtype=int)), pseudos)
    species_tmp = {}
    for ap in assignments:
        if ap.pseudo is None:
            continue
        e = species_tmp.setdefault(id(ap.pseudo),
                                   {"pseudo": ap.pseudo, "positions": []})
        e["positions"].append(np.asarray(ap.position, dtype=float))
    species_payload = [
        (e["pseudo"], np.asarray(e["positions"], dtype=float)
         if e["positions"] else np.zeros((0, 3), dtype=float))
        for e in species_tmp.values()]
    V_loc_r = jnp.asarray(build_local_ionic_potential_on_G_total(
        assignments=[{"pseudo": ap.pseudo,
                      "position": np.asarray(ap.position, dtype=float)}
                     for ap in assignments],
        species_groups=species_payload, fft_grid=grid,
        bdot=np.asarray(wfn.bdot, dtype=float),
        cell_volume=float(wfn.cell_volume),
        bvec=np.asarray(wfn.bvec, dtype=float), blat=float(wfn.blat),
        truncation_2d=truncation_2d), dtype=jnp.float64)
    vnl_setup = vnl_ops.build_vnl_setup(wfn, sym, meta, pseudos,
                                        nspinor=int(wfn.nspinor))

    nocc = int(wfn.nelec)
    f_spin = spin_degeneracy_factor(wfn)
    rho_np = build_valence_density_distributed(
        wfn, sym, meta, nocc, nk=nk, mesh=mesh, print_fn=p0)
    V_H_r = build_hartree_potential(
        jnp.asarray(rho_np), wfn, truncation_2d=truncation_2d,
        expected_electrons=f_spin * float(nocc), print_fn=p0)
    V_H_r = jnp.asarray(np.asarray(V_H_r, dtype=np.float64),
                        dtype=jnp.float64)
    del rho_np

    gtab = padded_gvectors(wfn, k="full_bz")
    kvecs = np.asarray(sym.unfolded_kpts)

    # ---- the LOCAL PLAN reference: one k's whole band block at a time ----
    # This is the route being replaced, including its wall: the per-k
    # full-band FFT box, nb·ns·nx·ny·nz·16 B, alive on EVERY rank.
    hwm0 = vmhwm_gib()
    vh_ref = np.zeros((nk, nb, nb), dtype=np.complex128)
    ki_ref = np.zeros((nk, nb, nb), dtype=np.complex128)
    dip_ref = np.zeros((nk, 3, nb, nb), dtype=np.complex128)
    t0 = time.time()
    for ik in range(nk):
        box = load_kpoint_fftbox_local(wfn, meta, ik, nb, bispinor=bispinor)
        G_pad, g_mask = gtab.at(ik)
        kpt = jnp.asarray(kvecs[ik], dtype=jnp.float64)
        vh_ref[ik] = np.asarray(compute_local_V_k(
            box, G_pad, V_H_r, wfn.cell_volume, g_mask=g_mask))
        ki_ref[ik] = np.asarray(get_kin_ion_k(
            box, G_pad, kvecs[ik], V_loc_r, vnl_setup, wfn, g_mask=g_mask))
        p_cart = compute_p_operator_k(
            box, G_pad, kpt, jnp.asarray(wfn.bdot, dtype=jnp.float64),
            jnp.asarray(wfn.bvec, dtype=jnp.float64), float(wfn.blat),
            g_mask=g_mask)
        v_nl = compute_vnl_velocity_cart(box, G_pad, kpt, vnl_setup,
                                         g_mask=g_mask)
        dip_ref[ik] = np.asarray(p_cart - v_nl)
        del box
    t_local = time.time() - t0
    hwm_local = vmhwm_gib()
    p0(f"[mtxel] local plan  {t_local:7.3f}s   VmHWM {hwm0:.3f} -> "
       f"{hwm_local:.3f} GiB (the per-k full-band box)")

    # ---- the SWEEP, constructed exactly as the three call sites do -------
    psi_G = wfn.load(bands=(0, nb), k="full_bz",
                     sharding=band_sphere_spec(), bispinor=bispinor)
    bidx = wfn.box_index(k="full_bz")
    geom = SweepGeometry(mesh=mesh, fft_grid=grid,
                         ngkmax=int(psi_G.shape[3]), nb=nb,
                         ns=int(psi_G.shape[2]), nk=nk,
                         cell_volume=float(wfn.cell_volume))
    p0(f"[mtxel] psi_G {tuple(psi_G.shape)} spec={psi_G.sharding.spec}  "
       f"nb_logical={geom.nb_logical} nb_padded={geom.nb} "
       f"p_prod={geom.p_prod}")
    kw = dict(geom=geom, gvecs=gtab.gvecs, gmask=gtab.mask,
              box_index=bidx, kvecs=kvecs)

    def pad_ref(ref):
        """Reference at the sweep's PADDED band extent; pad entries stay
        exact zeros, so comparing against them also asserts the pad bands
        contribute nothing at all rather than merely something small."""
        if geom.nb == nb:
            return ref
        w = [(0, 0)] * ref.ndim
        w[-1] = w[-2] = (0, geom.nb - nb)
        return np.pad(ref, tuple(w))

    c0 = CC_STATE.compiles
    t0 = time.time()
    H_vh = sweep_matrix_elements(
        psi_G, operator=local_potential_operator(geom, V_H_r), **kw)
    H_vh.block_until_ready()
    t_vh, c_vh = time.time() - t0, CC_STATE.compiles - c0

    terms = [kinetic_operator(geom, np.asarray(wfn.bdot, dtype=float)),
             local_potential_operator(geom, V_loc_r),
             vnl_operator(geom, vnl_setup)]
    c0 = CC_STATE.compiles
    t0 = time.time()
    H_ki = sweep_matrix_elements(psi_G, operator=sum_operators(*terms), **kw)
    H_ki.block_until_ready()
    t_ki, c_ki = time.time() - t0, CC_STATE.compiles - c0

    c0 = CC_STATE.compiles
    t0 = time.time()
    H_dip = sweep_matrix_elements(
        psi_G, operator=dipole_operator(geom, bvec=wfn.bvec, blat=wfn.blat,
                                        vnl_setup=vnl_setup), **kw)
    H_dip.block_until_ready()
    t_dip, c_dip = time.time() - t0, CC_STATE.compiles - c0
    hwm_sweep = vmhwm_gib()

    p0(f"[mtxel] sweep V_H {t_vh:7.3f}s  kin+ion {t_ki:7.3f}s  "
       f"dipole {t_dip:7.3f}s   VmHWM {hwm_sweep:.3f} GiB")
    p0(f"[mtxel] compiles per sweep (nk={nk}; ONE lowering is the D10 "
       f"claim): V_H={c_vh} kin_ion={c_ki} dipole={c_dip}")
    p0(f"[mtxel] specs: V_H {H_vh.sharding.spec} {H_vh.shape}   "
       f"dipole {H_dip.sharding.spec} {H_dip.shape}")

    # ---- 1-3. per shard --------------------------------------------------
    worst = [shard_vs_reference(H_vh, pad_ref(vh_ref), "V_H", p0),
             shard_vs_reference(H_ki, pad_ref(ki_ref), "kin+ion (T+V_loc+V_NL)",
                                p0),
             shard_vs_reference(H_dip, pad_ref(dip_ref), "dipole (p - dV_NL/dK)",
                                p0)]

    # ---- 4. the boundary -------------------------------------------------
    worst.append(host_vs_reference(blocks_to_host(H_vh, nb=nb), vh_ref,
                                   "blocks_to_host V_H replicated", p0))
    ki_h = blocks_to_host(H_ki, nb=nb, owner_only=True)
    if rank == 0:
        worst.append(host_vs_reference(ki_h, ki_ref,
                                       "blocks_to_host kin+ion owner", p0))
    elif ki_h is not None:
        p0("")
        print(f"[mtxel] rank={rank:03d} FAIL owner_only returned an array "
             f"on a peer", flush=True)
        worst.append(1.0)
    dip_h = blocks_to_host(H_dip, nb=nb, owner_only=True)
    if rank == 0:
        worst.append(host_vs_reference(dip_h, dip_ref,
                                       "blocks_to_host dipole owner", p0))

    # ---- 5. end to end: the edited compute_hartree_matrix -----------------
    vh_e2e = compute_hartree_matrix(
        wfn, sym, meta, truncation_2d=truncation_2d, nb=nb, mesh=mesh,
        print_fn=p0)
    worst.append(host_vs_reference(vh_e2e, vh_ref,
                                   "compute_hartree_matrix e2e", p0))

    print(f"[mtxel] rank={rank:03d} VmHWM local {hwm_local:.3f} GiB, "
          f"after sweep {hwm_sweep:.3f} GiB", flush=True)
    ok = max(worst) <= RTOL
    p0(f"[mtxel] VERDICT {'PASS' if ok else 'FAIL'} "
       f"(tol {RTOL:.0e} relative, per shard)")
    return 0 if ok else 1


if __name__ == "__main__":
    # finalize_process() ends the process with os._exit and DOES NOT RETURN,
    # so a bare `finally: finalize_process()` swallows the exception and
    # exits 0 — a gate that reports PASS on a crash (job 7888809).
    import traceback
    rc = 1
    try:
        rc = main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        rc = 1
    finalize_process(rc)
