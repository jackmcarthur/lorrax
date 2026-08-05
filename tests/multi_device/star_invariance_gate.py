"""Is a band-index matrix the SAME at every member of a k-star?

THE PREMISE THE IBZ SELF-CONSISTENT UPDATE RESTS ON.  ``WfnLoader.load(
k='full_bz')`` builds the full-BZ ψ by unfolding the IBZ ψ IN G-SPACE —
rotating G-vectors, applying the τ phase, the spinor rotation and the TRS
conjugation.  So band n at a star member ``Sk̄`` is the unfold of band n at
``k̄``, and any operator commuting with S has the SAME matrix at every
member:

    ⟨m,Sk̄|O|n,Sk̄⟩ = ⟨m,k̄|𝒰†O𝒰|n,k̄⟩ = ⟨m,k̄|O|n,k̄⟩

If that holds, moving H or U between the IBZ and the full BZ is an INDEX
MAP — no umklapp phase, no centroid permutation, no spinor rotation — and
the update can be done on the IBZ alone.  If it does NOT hold, every one
of those omissions is a silent error.

This measures it on REAL matrix elements from a REAL deck (⟨m|T+V|n⟩ built
by the k-scan), not on synthetic ψ, because the thing under test IS the
loader's unfold convention.

The negative control matters as much as the check: a matrix that is NOT
symmetry-related must show a large spread, or the test would pass on a
deck whose stars are all singletons and prove nothing.

Env: STAR_NB, STAR_DECK (an input dir containing WFN.h5 + deck.in).
"""
import os
import sys

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from common.collectives import (process_count, process_rank,   # noqa: E402
                                resolve_mesh)
from common.symmetry_maps import (SymMaps, star_select,        # noqa: E402
                                  star_broadcast, star_spread)
from common.mtxel_sweep import (SweepGeometry, band_sphere_spec,  # noqa: E402
                                kinetic_operator,
                                sweep_matrix_elements,
                                blocks_to_host)
from file_io.wfn_loader import WfnLoader                       # noqa: E402
from psp.dft_operators import padded_gvectors                  # noqa: E402

DECK = os.environ.get("STAR_DECK", "").strip()
NB = int(os.environ.get("STAR_NB", "16"))
RTOL = 1e-10


def main():
    rank = process_rank()
    p0 = print if rank == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    if not DECK:
        p0("[star] STAR_DECK unset")
        return 1

    wfn = WfnLoader(os.path.join(DECK, "WFN.h5"))
    sym = SymMaps(wfn)
    irr = np.asarray(sym.irr_idx_k)
    nk = int(irr.shape[0])
    n_star = len(np.unique(irr))
    sizes = np.bincount(irr)
    p0(f"[star] nk_full={nk}  nk_irr={n_star}  "
       f"star sizes {sorted(set(sizes[sizes > 0].tolist()))}  "
       f"reduction {nk / max(n_star, 1):.2f}x  ntran={int(wfn.ntran)}")
    if n_star == nk:
        p0("[star] every star is a singleton on this deck — the check "
           "cannot distinguish a correct broadcast from a no-op.  "
           "REFUSING to report a pass.")
        return 1

    nb = min(NB, int(wfn.nbands))
    psi = wfn.load(bands=(0, nb), k="full_bz", sharding=band_sphere_spec())
    gtab = padded_gvectors(wfn, k="full_bz")
    bidx = np.asarray(wfn.box_index(k="full_bz"))
    grid = tuple(int(v) for v in wfn.fft_grid)
    geom = SweepGeometry(mesh=mesh, fft_grid=grid,
                         ngkmax=int(psi.shape[3]), nb=nb,
                         ns=int(psi.shape[2]), nk=nk,
                         cell_volume=float(wfn.cell_volume))

    # A genuine symmetry-commuting operator: the kinetic energy |k+G|^2.
    op = kinetic_operator(geom, np.asarray(wfn.bdot, dtype=float))
    T = blocks_to_host(sweep_matrix_elements(
        psi, operator=op, geom=geom, gvecs=gtab.gvecs, gmask=gtab.mask,
        box_index=bidx, kvecs=np.asarray(sym.unfolded_kpts)),
        nb=nb, owner_only=False)
    T = np.asarray(T)

    # ---- 0. WHICH RULE? equality or antiunitary conjugation -------------
    # Spatial S is unitary, so O(Sk) == O(k).  Time reversal is ANTIunitary
    # (Theta = i sigma_y K), and for Hermitian O commuting with Theta
    #     <Theta m, k|O|Theta n, k> = conj(<m,k|O|n,k>)
    # so a TRS-paired member carries the CONJUGATE, not the same matrix.
    # Measure both against every non-singleton pair and report which holds
    # per sym row, rather than assuming.
    sidx = np.asarray(sym.sym_idx_k)
    n_spatial = int(wfn.ntran)
    d_eq = d_cj = 0.0
    n_sp = n_tr = 0
    for v in np.unique(irr):
        mem = np.where(irr == v)[0]
        if mem.size < 2:
            continue
        base = T[mem[0]]
        for j in mem[1:]:
            e = float(np.abs(T[j] - base).max())
            c = float(np.abs(T[j] - np.conj(base)).max())
            if int(sidx[j]) >= n_spatial:
                d_cj = max(d_cj, c); n_tr += 1
            else:
                d_eq = max(d_eq, e); n_sp += 1
    sc0 = max(float(np.abs(T).max()), 1e-300)
    p0(f"[star] 0. spatial pairs={n_sp} max|O(Sk)-O(k)| rel "
       f"{d_eq / sc0:.3e} | TRS pairs={n_tr} max|O(-k)-conj(O(k))| rel "
       f"{d_cj / sc0:.3e}")

    # ---- 1. the premise -------------------------------------------------
    spread = star_spread(T, irr, sidx, n_spatial)
    scale = max(float(np.abs(T).max()), 1e-300)
    ok1 = (spread / scale) <= RTOL
    p0(f"[star] 1. star spread of <m|T|n>   {spread:.3e} abs, "
       f"{spread / scale:.3e} rel   {'PASS' if ok1 else 'FAIL'}")

    # ---- 2. negative control -------------------------------------------
    # Break the correspondence by permuting bands at ONE star member.  If
    # the spread does not move, check 1 is not measuring anything.
    T_bad = T.copy()
    victim = int(np.where(irr == irr[np.argmax(sizes[irr] > 1)])[0][-1])
    T_bad[victim] = T_bad[victim][::-1, :]
    bad = star_spread(T_bad, irr, sidx, n_spatial) / scale
    ok2 = bad > 1e-3
    p0(f"[star] 2. negative control (bands permuted at one member)  "
       f"{bad:.3e} rel   (must be LARGE) {'PASS' if ok2 else 'FAIL'}")

    # ---- 3. select -> broadcast is the identity -------------------------
    T_irr, labels = star_select(T, irr)
    T_back = star_broadcast(T_irr, irr, sidx, n_spatial, labels)
    d = float(np.abs(T_back - T).max()) / scale
    ok3 = d <= RTOL
    p0(f"[star] 3. select->broadcast round trip  {d:.3e}   "
       f"{'PASS' if ok3 else 'FAIL'}")
    p0(f"[star]    IBZ block {T_irr.shape} from full {T.shape} — the eigh "
       f"cost drops by {nk / max(n_star, 1):.2f}x")

    ok = ok1 and ok2 and ok3
    p0(f"[star] VERDICT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
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
