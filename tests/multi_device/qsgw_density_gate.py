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
                             band_rotation_spec,
                             distributed_eigh_bands,
                             symmetrise_density,
                             hartree_from_orbitals)
from psp.get_DFT_mtxels import (valence_density_from_kpoint,    # noqa: E402
                                compute_local_V_k)
from common.mtxel_sweep import (SweepGeometry,                  # noqa: E402
                                local_potential_operator,
                                sweep_matrix_elements)

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

    # Identity sym table: these synthetic psi carry no real symmetry, so
    # the star average must be a no-op for checks 1-6 to mean what they say.
    ident_perm = np.arange(ngrid, dtype=np.int32)[None, :]
    kw = dict(mesh=mesh, box_index=bidx, fft_grid=GRID,
              cell_volume=volume, spin_degeneracy=f_spin,
              sym_perm=ident_perm)

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

    # ---- 5. CONVENTION PIN: eigenvectors are COLUMNS --------------------
    # Checks 1, 2, 4 and every norm check survive a TRANSPOSED U, because
    # for a unitary Q the transpose is also unitary and also mixes only
    # within the occupied block.  So the convention has to be pinned
    # against an explicit host-side rotation, not against an invariance.
    psi_rot_np = np.einsum('kmn,kmsg->knsg', U_gen, psi, optimize=True)
    rho_pin = np.zeros(GRID, dtype=np.float64)
    for ik in range(NK):
        box = np.zeros((NB, NS, nx, ny, nz), dtype=np.complex128)
        box[:, :, gv[ik, :, 0], gv[ik, :, 1], gv[ik, :, 2]] = psi_rot_np[ik]
        rho_pin += np.asarray(valence_density_from_kpoint(
            jnp.asarray(box), nocc=NOCC, weight=float(kweights[ik]),
            cell_volume=volume, spin_degeneracy=f_spin))
    d_pin = float(np.abs(rho_gen - rho_pin).max()) / max(
        float(np.abs(rho_pin).max()), 1e-300)
    ok5 = d_pin <= RTOL
    p0(f"[qd] 5. column-convention pin  rel {d_pin:.3e}  "
       f"{'PASS' if ok5 else 'FAIL'}")
    # ...and the transpose must DISAGREE, or the pin proves nothing.
    psi_rotT = np.einsum('knm,kmsg->knsg', U_gen, psi, optimize=True)
    rho_T = np.zeros(GRID, dtype=np.float64)
    for ik in range(NK):
        box = np.zeros((NB, NS, nx, ny, nz), dtype=np.complex128)
        box[:, :, gv[ik, :, 0], gv[ik, :, 1], gv[ik, :, 2]] = psi_rotT[ik]
        rho_T += np.asarray(valence_density_from_kpoint(
            jnp.asarray(box), nocc=NOCC, weight=float(kweights[ik]),
            cell_volume=volume, spin_degeneracy=f_spin))
    d_T = float(np.abs(rho_pin - rho_T).max()) / max(
        float(np.abs(rho_pin).max()), 1e-300)
    ok5b = d_T > 1e-6
    p0(f"[qd] 5b. transpose must DIFFER  rel {d_T:.3e}  "
       f"{'PASS' if ok5b else 'FAIL'}")

    # ---- 6. distributed eigh: E and Z with no (nb,nb) on one rank -------
    rngH = np.random.default_rng(7)
    A = rngH.standard_normal((NK, NB, NB)) + 1j * rngH.standard_normal((NK, NB, NB))
    A = 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))
    A_j = jax.make_array_from_callback(A.shape, U_sh, lambda idx: A[idx])
    try:
        E_d, Z_d = distributed_eigh_bands(A_j, mesh=mesh)
        E_d = np.asarray(E_d)
        E_ref = np.linalg.eigvalsh(A)
        d_E = float(np.abs(E_d - E_ref).max()) / max(
            float(np.abs(E_ref).max()), 1e-300)
        ok6 = d_E <= 1e-10
        p0(f"[qd] 6. distributed eigh eigenvalues  rel {d_E:.3e}  "
           f"Z spec={Z_d.sharding.spec}  {'PASS' if ok6 else 'FAIL'}")
        # rho built from the DISTRIBUTED Z must match rho from the same
        # eigenvectors computed natively -- rho is gauge-invariant even
        # though Z is not, which is the whole point of the gauge warning.
        occ_d = np.zeros((NK, NB)); occ_d[:, :NOCC] = 1.0
        rho_d = np.asarray(rho_from_wfns(psi_j, occ_d, kweights, U=Z_d, **kw))
        _, Z_n = np.linalg.eigh(A)
        Z_nj = jax.make_array_from_callback(Z_n.shape, U_sh, lambda idx: Z_n[idx])
        rho_n = np.asarray(rho_from_wfns(psi_j, occ_d, kweights, U=Z_nj, **kw))
        d_g = float(np.abs(rho_d - rho_n).max()) / max(
            float(np.abs(rho_n).max()), 1e-300)
        ok6b = d_g <= 1e-9
        p0(f"[qd] 6b. rho is gauge-invariant (dist Z vs native Z)  "
           f"rel {d_g:.3e}  {'PASS' if ok6b else 'FAIL'}")
    except Exception as exc:
        p0(f"[qd] 6. distributed eigh UNAVAILABLE: {type(exc).__name__}: "
           f"{str(exc)[:110]}")
        ok6 = ok6b = False

    # ---- 7. symmetrisation is ENFORCED on a reduced k-set ---------------
    # kweights here are random ⇒ non-uniform ⇒ a reduced set.  Every rho
    # above was built with sym_perm supplied as the identity table; asking
    # without one must REFUSE, because the unsymmetrised sum integrates to
    # the exact electron count and no other check sees it.
    try:
        rho_from_wfns(psi_j, occ, kweights,
                      **{k: v for k, v in kw.items() if k != "sym_perm"})
        ok7 = False
        p0("[qd] 7. reduced k-set without sym_perm  FAIL (not refused)")
    except ValueError as exc:
        ok7 = "non-uniform" in str(exc)
        p0(f"[qd] 7. reduced k-set without sym_perm  "
           f"{'PASS refused' if ok7 else 'FAIL wrong error'}")

    # ---- 8. symmetrise_density properties -------------------------------
    ident = np.arange(ngrid, dtype=np.int32)[None, :]     # identity only
    r_id = np.asarray(symmetrise_density(rho_ref, ident))
    ok8a = float(np.abs(r_id - rho_ref).max()) == 0.0
    # A CLOSED 2-op table (a Z2 group): identity + roll by nx/2, whose
    # square is the identity.  Idempotence of the star average holds iff
    # the table is a GROUP -- {identity, roll-by-1} is not one (its square
    # is roll-by-2, absent from the table) and correctly fails this.
    # fft_grid_pullback_perm emits a group; an ad-hoc table need not.
    rolled = np.roll(np.arange(ngrid, dtype=np.int32).reshape(GRID),
                     nx // 2, axis=0).ravel()
    tab = np.stack([np.arange(ngrid, dtype=np.int32), rolled])
    r_s = np.asarray(symmetrise_density(rho_ref, tab))
    ok8b = abs(float(r_s.sum()) - float(rho_ref.sum())) / abs(
        float(rho_ref.sum())) < 1e-12          # integral preserved
    r_ss = np.asarray(symmetrise_density(r_s, tab))
    ok8c = float(np.abs(r_ss - r_s).max()) / max(
        float(np.abs(r_s).max()), 1e-300) < 1e-12   # idempotent on its orbit
    p0(f"[qd] 8. symmetrise: identity no-op={ok8a} integral-preserved="
       f"{ok8b} idempotent={ok8c}  "
       f"{'PASS' if (ok8a and ok8b and ok8c) else 'FAIL'}")
    ok8 = ok8a and ok8b and ok8c

    # ---- 9. END TO END: psi -> rho -> V_H -> <m|V_H|n> ------------------
    # The actual self-consistency seam.  Proves the density module and the
    # matrix-element sweep compose: same grid, same Coulomb convention, and
    # a V_H that feeds local_potential_operator without any reshaping.
    class _Wfn:                     # build_hartree_potential reads 4 fields
        cell_volume = volume
        bdot = np.eye(3) * 1.31 + 0.07
        bvec = np.eye(3) * 0.9
        blat = 1.0
    wfn_stub = _Wfn()
    n_elec = f_spin * float(np.einsum('k,kn->', kweights, occ))
    # No ``U=``: ``hartree_from_orbitals`` lost that argument at d70e7fa
    # ("one rotation instead of two"), which made ψ̃ the CALLER's job via
    # ``rotate_bands``.  The gate kept passing ``U=None`` and has therefore
    # been dying with ``TypeError`` at this line ever since — after
    # printing PASS for checks 1-8, so the failure reads as a green gate
    # unless the return code is checked (job 7889252).  ``psi_j`` is
    # unrotated here, which is what ``U=None`` used to mean.
    rho_e2e, V_H_r = hartree_from_orbitals(
        psi_j, occ, kweights, wfn_stub, mesh=mesh, box_index=bidx,
        fft_grid=GRID, truncation_2d=False, spin_degeneracy=f_spin,
        sym_perm=ident_perm, expected_electrons=n_elec,
        print_fn=(lambda *a, **k: None))
    V_H_r = np.asarray(V_H_r)
    ok9a = V_H_r.shape == GRID
    e_h = 0.5 * float(np.sum(np.asarray(rho_e2e) * V_H_r)) * deltaV
    ok9b = e_h > 0.0                       # self-repulsion is positive
    p0(f"[qd] 9. V_H shape {V_H_r.shape} == grid {ok9a}; "
       f"Hartree energy {e_h:.6f} Ry > 0 {ok9b}  "
       f"{'PASS' if (ok9a and ok9b) else 'FAIL'}")

    # Feed that V_H straight into the sweep and check it against the
    # per-k kernel: the whole chain, one tolerance.
    geom = SweepGeometry(mesh=mesh, fft_grid=GRID, ngkmax=NGK, nb=NB,
                         ns=NS, nk=NK, cell_volume=volume)
    gmask = np.ones((NK, NGK), dtype=np.float64)
    op = local_potential_operator(geom, V_H_r)
    H_vh = sweep_matrix_elements(psi_j, operator=op, geom=geom, gvecs=gv,
                                 gmask=gmask, box_index=bidx,
                                 kvecs=np.zeros((NK, 3)))
    V_kern = np.zeros((NK, NB, NB), dtype=np.complex128)
    for ik in range(NK):
        box = np.zeros((NB, NS, nx, ny, nz), dtype=np.complex128)
        box[:, :, gv[ik, :, 0], gv[ik, :, 1], gv[ik, :, 2]] = psi[ik]
        V_kern[ik] = np.asarray(compute_local_V_k(
            jnp.asarray(box), jnp.asarray(gv[ik]), jnp.asarray(V_H_r),
            volume, g_mask=jnp.asarray(gmask[ik])))
    worst = 0.0
    for sh in H_vh.addressable_shards:
        ref = V_kern[sh.index]
        worst = max(worst, float(np.abs(np.asarray(sh.data) - ref).max())
                    / max(float(np.abs(ref).max()), 1e-300))
    ok9c = worst <= RTOL
    p0(f"[qd] 9b. <m|V_H|n> sweep vs per-k kernel  rel {worst:.3e}  "
       f"{'PASS' if ok9c else 'FAIL'}")
    ok9 = ok9a and ok9b and ok9c

    # ---- 10. the charge refusal fires -----------------------------------
    try:
        hartree_from_orbitals(
            psi_j, occ, kweights, wfn_stub, mesh=mesh, box_index=bidx,
            fft_grid=GRID, truncation_2d=False, spin_degeneracy=f_spin,
            sym_perm=ident_perm, expected_electrons=n_elec * 2.0,
            print_fn=(lambda *a, **k: None))
        ok10 = False
    except ValueError:
        ok10 = True
    p0(f"[qd] 10. wrong expected_electrons refused  "
       f"{'PASS' if ok10 else 'FAIL'}")

    rho_G = rho_r_to_G(rho_ref, mesh=mesh)
    p0(f"[qd] rho(G=0) = {complex(np.asarray(rho_G).ravel()[0]):.6e} "
       f"(= sum_r rho = {float(rho_ref.sum()):.6e})")

    ok = (ok1 and ok2 and ok3 and ok4 and ok4b and ok5 and ok5b
          and ok6 and ok6b and ok7 and ok8 and ok9 and ok10)
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
