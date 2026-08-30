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
  3. dipole      — 2(k+G)ψ ± ∂V_NL/∂K ψ, three Cartesian components on
                   one sweep, with the component axis replicated.  BOTH
                   ARMS OF THE SIGN, for the reason below.
  4. the boundary — ``blocks_to_host`` (replicated and owner_only) against
                   the same reference, including the band-pad trim.
  5. end to end  — ``compute_hartree_matrix`` itself, the edited function.

Plus: one compile for the whole k sweep, and per-rank VmHWM — a plan
that merely RELOCATED a peak looks clean on rank 0 alone.

WHY CHECK 3 SWEEPS TWICE, AND WHY THE ONE-ARM VERSION WAS A DEFECT
------------------------------------------------------------------
``dipole_operator`` assembles ``p ± ∂V_NL/∂K`` under a
``vnl_velocity_sign`` knob, and its DEFAULT flipped from ``-1`` to
``+1`` on 2026-08-09 (``6b3ffc1f``, an owner ruling measured against
BerkeleyGW's q→0 head).  This file was written on 2026-08-04 and its
local-plan reference hard-coded ``p_cart - v_nl``; ``6b3ffc1f`` did not
touch it.  From that day the two sides of check 3 were DIFFERENT
OPERATORS, and check 3 reported ``4.364e-01`` — the whole of which is
``max|2·∂V_NL/∂K| / max|p − ∂V_NL/∂K|`` and none of which is a sweep
defect.  Nobody saw it until 2026-08-17 because two unrelated bit-rot
defects — the stale ``common.symmetry_maps`` import and the
``psi_G.sharding.spec`` print, each commented at its own line — had made
this file an ImportError since 2026-08-07, two days BEFORE the ruling
that stranded it.  It has never been possible to observe.

The repair is not to re-pin the reference to today's default — that
just re-arms the same trap for the next ruling.  ``p`` and
``∂V_NL/∂K`` are accumulated SEPARATELY by the local plan, both arms
are assembled from them, and BOTH are swept and compared.  That is also
the property the call site needs: ``psp.get_dipole_mtxels.main``
resolves the sign at run time from a CLI flag or a deck key, so a gate
that certifies one arm certifies half its runs.  Running the two arms
in one process additionally exercises ``_operator_key``'s sign entry on
a real deck — two arms that hashed the same would silently share a
compiled program, which is the hazard
``tests/test_dipole_vnl_velocity_sign.py`` gates synthetically.

Env: MTX_DECK (input file), MTX_NB, and optional MTX_PSEUDO_DIR (defaults
to the deck directory, useful when a read-only WFN fixture shares another
fixture's identical species pseudopotentials).
"""
import os
import sys
import time

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import PartitionSpec as P                   # noqa: E402

from common import Meta                                       # noqa: E402
from common.collectives import (process_count, process_rank,   # noqa: E402
                                resolve_mesh)
from common.jax_compile_cache import _STATE as CC_STATE        # noqa: E402
from common.mtxel_sweep import (VNL_VELOCITY_SIGN_FLIPPED,      # noqa: E402
                                VNL_VELOCITY_SIGN_SHIPPED,
                                SweepGeometry, blocks_to_host, dipole_operator,
                                kinetic_operator,
                                local_potential_operator, sum_operators,
                                sweep_matrix_elements, vnl_operator)
from common.wfn_layout import band_sphere_spec                   # noqa: E402
from common.wfn_transforms import load_kpoint_fftbox_local     # noqa: E402
from ffi import _services      # noqa: F401,E402  (path bootstrap; dies
                                 # with the owner's workspace fix)
_services.ensure_on_path()
# THE SERVICE, not ``from common import symmetry_maps``.  That shim was
# deleted with the rest of the phase-wide forwarding shims and this line
# went stale with it, which made the whole gate an ImportError at line
# 49 — so the one gate ``compute_hartree_matrix``'s own comment names as
# its certifier could not be run at all.  Found 2026-08-17.
import symmetry_maps                                           # noqa: E402
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

    pseudo_dir = os.environ.get("MTX_PSEUDO_DIR", deck_dir)
    pseudos = load_pseudopotentials(pseudo_dir)
    if not pseudos:
        raise SystemExit(f"no .upf in {pseudo_dir}")

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
        wfn, sym, meta, nk=nk, mesh=mesh, print_fn=p0)
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
    # THE TWO HALVES OF THE VELOCITY, KEPT APART.  Storing the assembled
    # ``p ± ∂V_NL/∂K`` here would bake one arm of ``vnl_velocity_sign``
    # into the reference, which is exactly the defect this file carried
    # from 2026-08-09 to 2026-08-17 (module docstring).  The arms are
    # assembled below, one per sweep.
    p_ref = np.zeros((nk, 3, nb, nb), dtype=np.complex128)
    vnl_ref = np.zeros((nk, 3, nb, nb), dtype=np.complex128)
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
        p_ref[ik] = np.asarray(p_cart)
        vnl_ref[ik] = np.asarray(v_nl)
        del box
    t_local = time.time() - t0
    hwm_local = vmhwm_gib()
    p0(f"[mtxel] local plan  {t_local:7.3f}s   VmHWM {hwm0:.3f} -> "
       f"{hwm_local:.3f} GiB (the per-k full-band box)")

    def dip_ref_at(sign):
        """The local plan's velocity at one arm of ``vnl_velocity_sign``.

        Written as a BRANCH and not as ``p + sign*v``, mirroring
        ``dipole_operator``'s own reason: a complex array times a real
        ``-1.0`` goes through the full complex product and turns ``+0.0``
        into ``-0.0``, which is numerically nothing and is not bit
        identity.  The reference has to execute the same expression the
        operator does or the gate is measuring its own arithmetic.
        """
        return p_ref + vnl_ref if sign > 0.0 else p_ref - vnl_ref

    # ---- the SWEEP, constructed exactly as the three call sites do -------
    psi_G = wfn.load(bands=(0, nb), k="full_bz",
                     sharding=band_sphere_spec(), bispinor=bispinor)
    bidx = wfn.box_index(k="full_bz")
    geom = SweepGeometry(mesh=mesh, fft_grid=grid,
                         ngkmax=int(psi_G.shape[3]), nb=nb,
                         ns=int(psi_G.shape[2]), nk=nk,
                         cell_volume=float(wfn.cell_volume))
    # ``getattr``, because at P=1 the loader hands back a
    # ``SingleDeviceSharding``, which carries no ``.spec`` — so this
    # DIAGNOSTIC LINE was an AttributeError that killed the gate before
    # any of its five checks ran.  Second bit-rot found in this file on
    # 2026-08-17, after the stale ``common.symmetry_maps`` import above.
    p0(f"[mtxel] psi_G {tuple(psi_G.shape)} "
       f"spec={getattr(psi_G.sharding, 'spec', psi_G.sharding)}  "
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

    # BOTH ARMS, default first.  ``_operator_key`` carries the sign, so
    # these are two compiled programs; if they were ever one, the second
    # arm below would reproduce the first and BOTH comparisons could not
    # pass at once.
    ARMS = ((VNL_VELOCITY_SIGN_FLIPPED, "p + dV_NL/dK  [default]"),
            (VNL_VELOCITY_SIGN_SHIPPED, "p - dV_NL/dK  [legacy]"))
    H_dip, t_dip, c_dip = {}, {}, {}
    for sign, _label in ARMS:
        c0 = CC_STATE.compiles
        t0 = time.time()
        H = sweep_matrix_elements(
            psi_G, operator=dipole_operator(geom, bvec=wfn.bvec,
                                            blat=wfn.blat,
                                            vnl_setup=vnl_setup,
                                            vnl_velocity_sign=sign), **kw)
        H.block_until_ready()
        H_dip[sign] = H
        t_dip[sign] = time.time() - t0
        c_dip[sign] = CC_STATE.compiles - c0
    H_dip_def = H_dip[VNL_VELOCITY_SIGN_FLIPPED]
    hwm_sweep = vmhwm_gib()

    p0(f"[mtxel] sweep V_H {t_vh:7.3f}s  kin+ion {t_ki:7.3f}s  "
       f"dipole {t_dip[VNL_VELOCITY_SIGN_FLIPPED]:7.3f}s (+arm) "
       f"{t_dip[VNL_VELOCITY_SIGN_SHIPPED]:7.3f}s (-arm)   "
       f"VmHWM {hwm_sweep:.3f} GiB")
    p0(f"[mtxel] compiles per sweep (nk={nk}; ONE lowering is the D10 "
       f"claim): V_H={c_vh} kin_ion={c_ki} "
       f"dipole={c_dip[VNL_VELOCITY_SIGN_FLIPPED]}"
       f"/{c_dip[VNL_VELOCITY_SIGN_SHIPPED]}")
    p0(f"[mtxel] specs: V_H {getattr(H_vh.sharding, 'spec', H_vh.sharding)} "
       f"{H_vh.shape}   dipole "
       f"{getattr(H_dip_def.sharding, 'spec', H_dip_def.sharding)} "
       f"{H_dip_def.shape}")

    # ---- 1-3. per shard --------------------------------------------------
    worst = [shard_vs_reference(H_vh, pad_ref(vh_ref), "V_H", p0),
             shard_vs_reference(H_ki, pad_ref(ki_ref), "kin+ion (T+V_loc+V_NL)",
                                p0)]
    for sign, label in ARMS:
        worst.append(shard_vs_reference(H_dip[sign],
                                        pad_ref(dip_ref_at(sign)),
                                        f"dipole {label}", p0))
    # The two arms must also DIFFER by 2·∂V_NL/∂K, or the sign never
    # reached the kernel and both comparisons passed against the same
    # array.  This is the "parsed, stored, never read" failure mode.
    arm_gap = float(np.abs(np.asarray(2.0 * vnl_ref)).max()) / max(
        float(np.abs(dip_ref_at(VNL_VELOCITY_SIGN_SHIPPED)).max()), 1e-300)
    p0(f"[mtxel] {'arm separation 2|dV_NL/dK| / |p-dV_NL/dK|':<38} "
       f"         {arm_gap:.3e}   "
       f"{'OK' if arm_gap > 1e-6 else 'DEGENERATE — the arms are the same'}")

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
    dip_h = blocks_to_host(H_dip_def, nb=nb, owner_only=True)
    if rank == 0:
        worst.append(host_vs_reference(
            dip_h, dip_ref_at(VNL_VELOCITY_SIGN_FLIPPED),
            "blocks_to_host dipole owner", p0))

    # ---- 5. end to end: the edited compute_hartree_matrix -----------------
    vh_e2e = compute_hartree_matrix(
        wfn, sym, meta, truncation_2d=truncation_2d, nb=nb, mesh=mesh,
        print_fn=p0)
    worst.append(host_vs_reference(vh_e2e, vh_ref,
                                   "compute_hartree_matrix e2e", p0))
    vh_e2e_sharded = compute_hartree_matrix(
        wfn, sym, meta, truncation_2d=truncation_2d, nb=nb, mesh=mesh,
        print_fn=p0, return_sharded=True)
    worst.append(shard_vs_reference(
        vh_e2e_sharded, vh_ref, "compute_hartree_matrix device e2e", p0))
    expected_spec = P(None, "x", "y")
    spec_ok = vh_e2e_sharded.sharding.spec == expected_spec
    p0(f"[mtxel] {'device e2e output layout':<38} "
       f"{vh_e2e_sharded.sharding.spec!s:<20} "
       f"{'PASS' if spec_ok else f'FAIL want {expected_spec}'}")
    worst.append(0.0 if spec_ok else 1.0)

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
