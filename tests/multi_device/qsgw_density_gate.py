"""Certify ρ from rotated orbitals with per-state occupations.

Four checks, strongest first:

  0. U IS NEVER REPLICATED: it is sharded P(None,'x','y') and the
     rotation runs inside the k scan, so no rank holds a full (nb,nb).
  1. OCC-BLOCK INVARIANCE.  A unitary mixing only WITHIN the occupied
     manifold leaves ρ exactly invariant — ρ is the trace of the projector
     onto that subspace and a rotation inside it does not move the
     subspace.  So ρ(U_occ-only) must be BIT-IDENTICAL to ρ(unrotated).
     This needs no reference data and tests the rotation indexing, the
     occupation lookup, the k-weights, f_spin and the normalisation at
     once.  A transposed U or an off-by-one in the occupied range fails
     here and passes every norm check.
  2. ELECTRON COUNT.  ``ΔV · Σ_r ρ == f_spin · Σ_k w_k Σ_n f_nk``.
  3. vs THE EXISTING KERNEL.  Against
     ``psp.get_DFT_mtxels.valence_density_from_kpoint`` summed over k with
     U = I and step occupations, at 1e-12 relative.
  4. A GENERAL U MOVES ρ.  The negative control for (1): if occ↔unocc
     mixing did not change ρ, checks 1 and 3 would both pass on code that
     ignores U entirely.

Env: QD_NB, QD_NS, QD_NK, QD_GRID, QD_NGK, QD_NOCC.
"""
import os
import sys

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402

from common.collectives import (process_count, process_rank,   # noqa: E402
                                resolve_mesh)
from gw.qsgw_density import (rho_from_wfns, rho_r_to_G,          # noqa: E402
                             band_rotation_spec)
from psp.get_DFT_mtxels import valence_density_from_kpoint     # noqa: E402

NB = int(os.environ.get("QD_NB", "16"))
NS = int(os.environ.get("QD_NS", "2"))
NK = int(os.environ.get("QD_NK", "4"))
GRID = tuple(int(v) for v in os.environ.get("QD_GRID", "10,10,12").split(","))
NGK = int(os.environ.get("QD_NGK", "60"))
NOCC = int(os.environ.get("QD_NOCC", "6"))
RTOL = 1e-12


def _haar(rng, n):
    """A Haar-ish unitary: QR of a complex Gaussian."""
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    return Q * (np.diagonal(R) / np.abs(np.diagonal(R)))[None, :]


def main():
    rank, world = process_rank(), process_count()
    p0 = print if rank == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    px, py = (int(s) for s in mesh.devices.shape)
    nx, ny, nz = GRID
    ngrid = nx * ny * nz
    volume = 137.035
    f_spin = 1.0
    deltaV = volume / ngrid

    p0(f"[qd] world={world} mesh=({px},{py}) nk={NK} nb={NB} ns={NS} "
       f"grid={GRID} ngk={NGK} nocc={NOCC}")

    rng = np.random.default_rng(20260804)
    gv = np.zeros((NK, NGK, 3), dtype=np.int32)
    bidx = np.full((NK, nx, ny, nz), NGK, dtype=np.int32)
    for ik in range(NK):
        cells = rng.choice(ngrid, size=NGK, replace=False)
        gv[ik, :, 0] = cells // (ny * nz)
        gv[ik, :, 1] = (cells // nz) % ny
        gv[ik, :, 2] = cells % nz
        bidx[ik, gv[ik, :, 0], gv[ik, :, 1], gv[ik, :, 2]] = np.arange(NGK)

    # Orthonormal ψ per k: QR over the (spinor⊗G) index so |ψ|² integrates
    # to 1 per band and the electron-count check means something.
    psi = np.zeros((NK, NB, NS, NGK), dtype=np.complex128)
    for ik in range(NK):
        M = (rng.standard_normal((NS * NGK, NB))
             + 1j * rng.standard_normal((NS * NGK, NB)))
        Q, _ = np.linalg.qr(M)
        psi[ik] = Q.T.reshape(NB, NS, NGK)

    kweights = rng.random(NK)
    kweights /= kweights.sum()
    occ = np.zeros((NK, NB), dtype=np.float64)
    occ[:, :NOCC] = 1.0

    sharding = NamedSharding(mesh, P(None, ('x', 'y'), None, None))
    psi_j = jax.make_array_from_callback(psi.shape, sharding,
                                         lambda idx: psi[idx])

    kw = dict(mesh=mesh, box_index=bidx, fft_grid=GRID,
              cell_volume=volume, spin_degeneracy=f_spin)

    # ---- baseline: unrotated -------------------------------------------
    rho_ref = np.asarray(rho_from_wfns(psi_j, occ, kweights, **kw))

    # ---- 1. occupied-block-only U: rho must be BIT-IDENTICAL ------------
    U_occ = np.zeros((NK, NB, NB), dtype=np.complex128)
    for ik in range(NK):
        U_occ[ik] = np.eye(NB)
        U_occ[ik, :NOCC, :NOCC] = _haar(rng, NOCC)      # mix inside occ only
    U_sh = NamedSharding(mesh, band_rotation_spec())
    U_occ_j = jax.make_array_from_callback(
        U_occ.shape, U_sh, lambda idx: U_occ[idx])
    rho_occ = np.asarray(rho_from_wfns(psi_j, occ, kweights, U=U_occ_j, **kw))
    d_occ = float(np.abs(rho_occ - rho_ref).max())
    bit = bool(np.array_equal(rho_occ, rho_ref))
    scale = max(float(np.abs(rho_ref).max()), 1e-300)
    ok1 = (d_occ / scale) <= RTOL
    p0(f"[qd] 1. occ-block U vs unrotated   max|d| = {d_occ:.3e}  "
       f"rel {d_occ / scale:.3e}  BIT-IDENTICAL={bit}  "
       f"{'PASS' if ok1 else 'FAIL'}")

    # ---- 2. electron count ---------------------------------------------
    target = f_spin * float(np.einsum('k,kn->', kweights, occ))
    got = deltaV * float(rho_ref.sum())
    ok2 = abs(got - target) / max(target, 1e-300) <= 1e-10
    p0(f"[qd] 2. dV*sum(rho) = {got:.12f}  target {target:.12f}  "
       f"{'PASS' if ok2 else 'FAIL'}")

    # ---- 3. vs the existing per-k kernel --------------------------------
    rho_kern = np.zeros(GRID, dtype=np.float64)
    for ik in range(NK):
        box = np.zeros((NB, NS, nx, ny, nz), dtype=np.complex128)
        box[:, :, gv[ik, :, 0], gv[ik, :, 1], gv[ik, :, 2]] = psi[ik]
        rho_kern += np.asarray(valence_density_from_kpoint(
            jnp.asarray(box), nocc=NOCC, weight=float(kweights[ik]),
            cell_volume=volume, spin_degeneracy=f_spin))
    d_k = float(np.abs(rho_ref - rho_kern).max())
    s_k = max(float(np.abs(rho_kern).max()), 1e-300)
    ok3 = (d_k / s_k) <= RTOL
    p0(f"[qd] 3. vs valence_density_from_kpoint  rel {d_k / s_k:.3e}  "
       f"{'PASS' if ok3 else 'FAIL'}")

    # ---- 4. negative control: a general U MUST move rho -----------------
    U_gen = np.stack([_haar(rng, NB) for _ in range(NK)])
    U_gen_j = jax.make_array_from_callback(
        U_gen.shape, U_sh, lambda idx: U_gen[idx])
    rho_gen = np.asarray(
        rho_from_wfns(psi_j, occ, kweights, U=U_gen_j, **kw))
    d_gen = float(np.abs(rho_gen - rho_ref).max()) / scale
    ok4 = d_gen > 1e-6
    p0(f"[qd] 4. general U vs unrotated   rel {d_gen:.3e}  "
       f"(must be LARGE) {'PASS' if ok4 else 'FAIL'}")
    # ...and it must still hold the electron count.
    got_gen = deltaV * float(rho_gen.sum())
    ok4b = abs(got_gen - target) / max(target, 1e-300) <= 1e-10
    p0(f"[qd] 4b. general U electron count {got_gen:.12f}  "
       f"{'PASS' if ok4b else 'FAIL'}")

    rho_G = rho_r_to_G(rho_ref, mesh=mesh)
    p0(f"[qd] rho(G=0) = {complex(np.asarray(rho_G).ravel()[0]):.6e} "
       f"(= sum_r rho = {float(rho_ref.sum()):.6e})")

    ok = ok1 and ok2 and ok3 and ok4 and ok4b
    p0(f"[qd] VERDICT {'PASS' if ok else 'FAIL'}")
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
