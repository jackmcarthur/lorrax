"""Does the sweep's V_NL projector build scale with P, or is it replicated?

THE CLAIM UNDER TEST is in ``common.mtxel_sweep``'s own module docstring:
"THE ONLY WAY A RANK GOES IDLE is ``nb_logical < P`` ... there is no other
idle case at any ``P``, ``nk`` or mesh shape."

``vnl_operator`` and ``dipole_operator`` call
``vnl_ops.build_vnl_kdata_traced`` INSIDE the scan body.  Z is
``(total_R, ngkmax)`` REPLICATED and its build depends on k but not on the
band index, so every rank builds the same Z for every k — nk builds per rank
where the k-partitioned plan it replaced did nk/P.  If that term is
significant the quoted claim is false.  It has never been measured.

THE INSTRUMENT.  Five operators through the same sweep skeleton; the
difference between neighbours is the term named:

    kin       T ∘ ψ, no projectors            the skeleton + a diagonal
    vnl_z     build Z, then ψ·Σ Z             THE REPLICATED BUILD, alone
    vnl       Z E Z† ψ  (production)          build + the band-sharded apply
    dip_z     build Z and dZ, then ψ·Σ dZ     the 4x build (Z + 3 dZ)
    dip       production dipole operator      build + apply, 3 components

``vnl_z``/``dip_z`` reduce Z to a scalar, so the whole projector build is
live and nothing else is: XLA cannot DCE any of it (a two-element
dependence would, since Z is elementwise in G — hence the full sum).

READ IT AS A SCALING TABLE, NOT A ROW.  ``vnl_z - kin`` is the replicated
term.  Run the same shapes at several P: a term that does not fall is
replicated work, and its share of ``vnl`` at large P is the size of the
error in the quoted claim.

Env: MTX_DECK, MTX_NB, MTX_ARMS, MTX_REPS.
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
from common.mtxel_sweep import (SweepGeometry, Operator,        # noqa: E402
                                band_sphere_spec, dipole_operator,
                                kinetic_operator, sweep_matrix_elements,
                                vnl_operator)
from file_io import WfnLoader                                  # noqa: E402
from gw.gw_config import read_lorrax_input                     # noqa: E402
from psp.dft_operators import padded_gvectors                  # noqa: E402
from psp.pseudos import (build_atom_pp_assignments,            # noqa: E402
                         load_pseudopotentials)
import psp.vnl_ops as vnl_ops                                  # noqa: E402

DECK = os.environ.get(
    "MTX_DECK", "/scratch2/08271/jackmc/mos2_4x4_test/deck_b300.in")
NB = int(os.environ.get("MTX_NB", "128"))
REPS = int(os.environ.get("MTX_REPS", "3"))
ARMS = [a for a in os.environ.get(
    "MTX_ARMS", "kin,vnl_z,vnl,dip_z,dip").split(",") if a.strip()]


def build_probe(geom, vnl_setup, kind, bdot, bvec, blat):
    """``vnl_z`` / ``dip_z``: the projector BUILD with no apply."""
    def op_z(psi_n, gvec, gmask, bidx, kvec):
        kd = vnl_ops.build_vnl_kdata_traced(kvec, gvec, vnl_setup)
        return psi_n * jnp.sum(kd.Z)

    def op_dz(psi_n, gvec, gmask, bidx, kvec):
        kd = vnl_ops.build_vnl_kdata_traced(kvec, gvec, vnl_setup,
                                            compute_dZ=True)
        return psi_n * (jnp.sum(kd.Z) + jnp.sum(kd.dZ))

    if kind == 'vnl_z':
        return Operator(apply=op_z, key=('probe_vnl_z', geom.ngkmax,
                                         id(vnl_setup)))
    return Operator(apply=op_dz, key=('probe_dip_z', geom.ngkmax,
                                      id(vnl_setup)))


def main():
    rank, world = process_rank(), process_count()
    p0 = print if rank == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    px, py = (int(s) for s in mesh.devices.shape)

    deck_dir = os.path.dirname(os.path.abspath(DECK))
    params = read_lorrax_input(DECK)
    wfn_path = params.get("wfn_file", "WFN.h5")
    if not os.path.isabs(wfn_path):
        wfn_path = os.path.join(deck_dir, wfn_path)
    wfn = WfnLoader(wfn_path)
    wfn.adopt_mesh(mesh)
    sym = symmetry_maps.SymMaps(wfn)
    nb = min(NB, int(wfn.nbands))
    meta = Meta.from_system(wfn, sym, int(params.get("nval", 5)),
                            int(params.get("ncond", 5)), nb, 0,
                            bool(params.get("bispinor", False)))
    nk = int(sym.nk_tot)
    grid = tuple(int(s) for s in meta.fft_grid)

    pseudos = load_pseudopotentials(deck_dir)
    if not pseudos:
        raise SystemExit(f"no .upf in {deck_dir}")
    build_atom_pp_assignments(
        jnp.asarray(np.asarray(wfn.atom_crys, dtype=float)),
        jnp.asarray(np.asarray(wfn.atom_types, dtype=int)), pseudos)
    vnl_setup = vnl_ops.build_vnl_setup(wfn, sym, meta, pseudos,
                                        nspinor=int(wfn.nspinor))

    gtab = padded_gvectors(wfn, k="full_bz")
    kvecs = np.asarray(sym.unfolded_kpts)
    psi = wfn.load(bands=(0, nb), k="full_bz", sharding=band_sphere_spec())
    bidx = np.asarray(wfn.box_index(k="full_bz"))
    ngkmax = int(psi.shape[3])
    ns = int(psi.shape[2])

    geom = SweepGeometry(mesh=mesh, fft_grid=grid, ngkmax=ngkmax, nb=nb,
                         ns=ns, nk=nk, cell_volume=float(wfn.cell_volume))
    # Z's own extent, so the replicated term has a SIZE and not just a wall.
    Z0 = vnl_ops.build_vnl_kdata_traced(
        jnp.asarray(kvecs[0]), jnp.asarray(gtab.gvecs[0]), vnl_setup).Z
    total_R = int(Z0.shape[0])
    p0(f"[vnlp] world={world} mesh=({px},{py}) deck={os.path.basename(DECK)} "
       f"nk={nk} nb={nb} ns={ns} ngkmax={ngkmax} grid={grid}")
    p0(f"[vnlp] Z is ({total_R}, {ngkmax}) c128 = "
       f"{total_R*ngkmax*16/2**20:.1f} MiB REPLICATED, rebuilt on every rank "
       f"for every one of {nk} k; dZ is 3x that")

    bdot = np.asarray(wfn.bdot, dtype=np.float64)
    kw = dict(geom=geom, gvecs=gtab.gvecs, gmask=gtab.mask, box_index=bidx,
              kvecs=kvecs)
    ops = {
        'kin': lambda: kinetic_operator(geom, bdot),
        'vnl_z': lambda: build_probe(geom, vnl_setup, 'vnl_z', bdot, None, 0),
        'vnl': lambda: vnl_operator(geom, vnl_setup),
        'dip_z': lambda: build_probe(geom, vnl_setup, 'dip_z', bdot, None, 0),
        'dip': lambda: dipole_operator(
            geom, bvec=np.asarray(wfn.bvec, dtype=np.float64),
            blat=float(wfn.blat), vnl_setup=vnl_setup),
    }
    res = {}
    for arm in ARMS:
        op = ops[arm]()
        sweep_matrix_elements(psi, operator=op, **kw).block_until_ready()
        t = []
        for _ in range(REPS):
            t0 = time.time()
            sweep_matrix_elements(psi, operator=op,
                                  **kw).block_until_ready()
            t.append(time.time() - t0)
        res[arm] = float(np.median(t))
        p0(f"[vnlp] arm={arm:<6} median={res[arm]:.3f}s  best={min(t):.3f}s",
           flush=True)

    p0(f"[vnlp] ---- P={world} attribution ----")
    if 'kin' in res and 'vnl_z' in res:
        p0(f"[vnlp]   replicated Z build   {res['vnl_z']-res['kin']:+.3f} s "
           f"(vnl_z - kin)")
    if 'kin' in res and 'vnl' in res:
        p0(f"[vnlp]   whole V_NL operator  {res['vnl']-res['kin']:+.3f} s "
           f"(vnl - kin)")
    if 'vnl' in res and 'vnl_z' in res and 'kin' in res:
        num, den = res['vnl_z'] - res['kin'], max(res['vnl'] - res['kin'],
                                                  1e-9)
        p0(f"[vnlp]   build share of V_NL  {100.0*num/den:.1f} %")
    if 'kin' in res and 'dip_z' in res:
        p0(f"[vnlp]   replicated Z+dZ build {res['dip_z']-res['kin']:+.3f} s "
           f"(dip_z - kin)")
    if 'kin' in res and 'dip' in res:
        p0(f"[vnlp]   whole dipole operator {res['dip']-res['kin']:+.3f} s "
           f"(dip - kin)")
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            p0(f"[vnlp] rank={rank:03d} "
               f"VmHWM={float(line.split()[1])/2**20:.3f} GiB")
    return 0


if __name__ == "__main__":
    # finalize_process() ends with os._exit and DOES NOT RETURN, so a bare
    # `finally: finalize_process()` swallows the exception and exits 0.
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
