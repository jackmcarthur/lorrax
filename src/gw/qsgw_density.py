"""ρ(r) and ρ(G) from rotated orbitals with per-state occupations.

The piece that turns the fixed-density QSGW loop into a density-updating
one.  ``sc_iteration`` today rotates V_H as a BASIS CHANGE (U†V_H U) and
leaves ρ at its DFT value; this rebuilds ρ from the rotated orbitals, so
the Hartree potential of iteration n+1 comes from iteration n's density
rather than from the DFT one.

STRUCTURE: ONE SCAN, U NEVER REPLICATED
---------------------------------------
A single ``lax.scan`` over k.  Per k, in this order:

    ψ_m^X   ← reshard ψ[k] so the CONTRACTED band index m lies on 'x'
    ψ̃_n^Y  ← einsum('mn,msg->nsg', Z[k], ψ_m^X)     # Z is P('x','y')
    ψ̃_n^XY ← reshard onto the whole mesh
    box → ifft → ρ(r) += w_k · f_spin · Σ_n f_nk |ψ̃_nk(r)|²

``Z`` is sharded ``P(None, 'x', 'y')`` — m on 'x', n on 'y' — so no rank
ever holds a full ``(nb, nb)``.  That is the NATIVE layout of both
eigenvector producers (``ffi.linalg._scalapack.batched_distributed_eigh``
and a ``jnp.linalg.eigh`` constrained to it), and the contraction is
written to consume it as-is: transposing Z to put n on 'x' would swap the
sharding to ``P(None,'y','x')`` and cost an all-to-all on the largest
object in the loop, for nothing.  At nb=640/P=64 each rank carries
80×80×16 B = 102 kB per k against 6.5 MB replicated, and the replicated
form is the ``(nk, nb, nb)`` W2-class object that reaches 9.2 GB at
nb=2000/nk=144.  The contraction index m is on 'x' and the output index n
on 'y', so the sum over m is a reduction along 'x' ALONE, not a global
collective.

EIGENVECTORS ARE COLUMNS.  ``Z[k, m, n]`` is component m of eigenvector
n: ``A[k] @ Z[k] == Z[k] @ diag(W[k])``.  That is ScaLAPACK's convention,
``jnp.linalg.eigh``'s, and the one ``sc_iteration._rotate_to_dft_basis``
already assumes, so ψ̃_n = Σ_m Z[m,n] ψ_m.  The transpose of this is a
CONVENTION BUG THAT MOST TESTS CANNOT SEE: for a unitary Q, Qᵀ is also
unitary and also mixes only within the occupied block, so occupied-block
invariance, the electron count and the norm all survive it.  The gate
therefore pins the convention against an explicit host-side rotation
rather than relying on an invariance.

The third step — resharding ψ̃ from ``P('x',…)`` back onto ``('x','y')``
— is what makes the rest uniform.  Straight out of the rotation the bands
are split over 'x' and REPLICATED over 'y', so an FFT there would do px-
fold redundant work and the final reduction would have to know to sum over
'x' only (double-counting by py if it did not).  One cheap sphere-space
reshard buys: full-mesh FFT parallelism, and a reduction identical to the
unrotated path, so there is one reduction rule in this file rather than
two.

**The rotation happens on the sphere, never in r.**  Rotating in real
space would need every band of a k in the FFT box at once — the per-k
full-band box, 1.9 GB at nb=640 bispinor, which is the wall
``common.mtxel_sweep`` exists to avoid.  The sphere is ~200× smaller and
the rotation is diagonal in G, so it costs a GEMM and no transform.

WHY ALL BANDS ARE TRANSFORMED, NOT JUST THE OCCUPIED ONES
---------------------------------------------------------
``occ`` enters as a per-state WEIGHT, so bands with f = 0 contribute
exactly nothing and could be skipped.  They are not skipped, on purpose:

* the scan needs shapes uniform across iterations, and the occupied count
  varies per k in a metal;
* fractional occupations — the successor to the step function in
  ``gw.efermi`` — make every band contribute, so a code that slices to
  the occupied window would have to be rewritten rather than re-fed.

The price is transforming ``nb`` bands instead of ``n_occ``.  Measure it
before optimising it: at fixed occupations the fix is a mask to a fixed
band window, which is a change of one slice, not of this structure.

THE CHEAP CORRECTNESS GATE
--------------------------
A unitary that mixes only WITHIN the occupied manifold leaves ρ exactly
invariant — ρ is the trace of the projector onto that subspace, and a
rotation inside the subspace does not move the subspace.  So zeroing the
occupied↔unoccupied block of U must return ρ equal to the unrotated
density TO ROUND-OFF — measured 3.4e-16 relative (job 7888958), not
bit-identical: the rotation is a GEMM and reassociates the band sum, so
exact equality is not available and demanding it would be the same
mistake as demanding bit-identity across a resharding.

That single check tests the rotation indexing, the occupation lookup, the
k-weights, f_spin and the normalisation at once, needs no reference data,
and is what ``tests/multi_device/qsgw_density_gate.py`` checks.  It also states the physics: Δρ comes entirely from occ↔unocc
mixing, and is small exactly when that mixing is.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import timing
from common.wfn_transforms import _box_kernel, _cached_jit, _sharding_key
from runtime.padding import pad_axis_to, spec_divisor


__all__ = ["rho_from_wfns", "rho_r_to_G", "band_rotation_spec",
           "distributed_eigh_bands", "symmetrise_density",
           "hartree_from_orbitals", "rotate_bands"]


def _band_spec() -> P:
    """ψ layout: (n_k, nb, nspinor, ngkmax), bands over the whole mesh.

    Imported from ``common.mtxel_sweep`` rather than re-spelled: a second
    literal of the same PartitionSpec is exactly the drift that
    ``runtime.padding.spec_divisor`` was introduced to remove on the band
    DIVISOR, and it would be worse here — a spec that disagreed would not
    raise, it would silently insert a reshard between the loader, the
    matrix-element sweep and this density build.
    """
    from common.mtxel_sweep import band_sphere_spec
    return band_sphere_spec()


def band_rotation_spec() -> P:
    """``Z`` layout: ``(n_k, nb, nb)`` with m on 'x' and n on 'y'.

    The eigenvector matrix, columns, sharded so no rank holds a full
    ``(nb, nb)``.  It is exactly ``batched_distributed_eigh``'s declared
    output layout, so the FFI path needs no reshard at all; ask
    ``_make_kshard_eigh(..., u_spec=...)`` for the same layout from the
    native path rather than resharding a replicated U after the fact.
    """
    return P(None, "x", "y")


def symmetrise_density(rho_r, sym_perm):
    """ρ_sym[r] = (1/n_sym) Σ_s ρ[sym_perm[s, r]] — the star average.

    ``sym_perm`` is ``centroid.orbit_syms.compute_rgrid_sym_perm``'s table,
    ``sym_perm[s, r_new] = r_old`` with ``r_{r_new} ≡ S_s·r_{r_old} + τ_s``
    on the FFT grid, so the fractional translations are already handled and
    this is a pure gather.

    IDEMPOTENT IFF THE TABLE IS A GROUP.  The star average is a projector
    onto the symmetric subspace only when the permutations are closed
    under composition, which ``compute_rgrid_sym_perm`` emits and an
    ad-hoc table need not.  Not checked here: closure costs
    ``O(n_sym² · N_r)`` and the producer already guarantees it.

    WHY IT IS NOT OPTIONAL ON A REDUCED k-SET.  A k-weighted sum over the
    IBZ produces the density of the IBZ representatives, not of the
    crystal: the star members each contribute their own rotated copy and
    only their average is the true ρ.  The failure is quiet — an
    unsymmetrised IBZ ρ still integrates to the exact electron count,
    because the star average is a permutation average and preserves
    ``Σ_r`` — so the one check that guards everything else in this module
    is blind to it.  That is why :func:`rho_from_wfns` REFUSES a
    non-uniformly-weighted k-set unless it is given this table, rather
    than documenting the requirement and hoping.
    """
    rho = jnp.asarray(rho_r, dtype=jnp.float64)
    grid = rho.shape
    perm = jnp.asarray(sym_perm, dtype=jnp.int32)
    flat = rho.reshape(-1)
    return (jnp.mean(flat[perm], axis=0)).reshape(grid)


# FORCED SYNC, AND IT COSTS NOTHING HERE.  ρ(r) is ``(nx, ny, nz)`` f64 —
# 750 kB at 60×60×26 — and the very next statement in the only production
# caller (:func:`hartree_from_orbitals` → ``build_hartree_potential``) is
# ``float(jnp.sum(rho_r))``, a full host synchronisation for the charge
# check.  ``watch=True`` therefore moves the block a few Python statements
# earlier and changes nothing else about the pipeline; what it buys is that
# ρ's compute is charged to this row instead of to ``vh.poisson``.
@timing.timed("vh.rho", watch=True)
def rho_from_wfns(psi_G, occ, kweights, *, mesh: Mesh, box_index,
                  fft_grid, cell_volume: float, spin_degeneracy: float,
                  U=None, sym_perm=None):
    """ρ(r) = Σ_k w_k f_spin Σ_{n,s} f_nk |ψ̃_nks(r)|², scanned over k.

    Parameters
    ----------
    psi_G : (n_k, nb, ns, ngkmax) c128, band-sharded
    occ : (n_k, nb) float64
        Per-state occupations — ``gw.efermi.step_occupations`` today, a
        Fermi–Dirac factor later.  A WEIGHT, not a mask: nothing here
        assumes it is 0 or 1.  Indexed by the ROTATED band n when ``U`` is
        given, which is the band the eigenvalue E_nk belongs to.
    kweights : (n_k,) float64
        Weights of the SAME k-set as ``psi_G``.  On the IBZ these are
        ``WfnLoader.kweights`` and the result MUST be symmetrised over the
        star afterwards — a weighted IBZ sum is not the full-BZ density
        until it is.  See the SYMMETRISATION note below.
    U : (n_k, nb, nb) complex, sharded :func:`band_rotation_spec`, optional
        Eigenvectors as COLUMNS — ``U[k, m, n]`` is component m of
        rotated band n, so ψ̃_n = Σ_m U[m,n] ψ_m.  This is what
        ``ffi.linalg`` and ``jnp.linalg.eigh`` return and what
        ``sc_iteration`` already assumes; do not pass a transpose.
        ``None`` builds ρ from ``psi_G`` unrotated — the DFT density and
        the gate's baseline.

    Returns
    -------
    (nx, ny, nz) float64, replicated.

    NORMALISATION is term-for-term ``psp.get_DFT_mtxels.
    valence_density_from_kpoint``: ψ_r = ifftn(box, 'ortho')·√(N/Ω), so
    ``ΔV · Σ_r ρ = f_spin · Σ_k w_k Σ_n f_nk``.

    THE CARRY IS ρ(r) — ``(nx, ny, nz)`` f64, 750 kB at a 60×60×26 grid.
    The scan carries something negligible and the band reduction is folded
    into the accumulation, so there is ONE collective class for the whole
    build rather than one materialised ψ̃ per k.

    SYMMETRISATION IS ENFORCED, NOT DOCUMENTED.  ``sym_perm`` (from
    ``centroid.orbit_syms.compute_rgrid_sym_perm``) makes the result the
    star average; see :func:`symmetrise_density`.  A k-set with
    NON-UNIFORM weights is by construction reduced, and this routine
    REFUSES it without ``sym_perm`` rather than returning the
    unsymmetrised sum — which would integrate to the exact electron count
    and pass every other check in this module.  Uniform weights are taken
    to be the full BZ and need no table.

    The residual hole, stated rather than papered over: a REDUCED set
    whose stars all happen to have equal size also has uniform weights and
    would not be caught.  That is exotic on a Monkhorst–Pack grid but not
    impossible; pass ``sym_perm`` whenever the k-set is reduced, uniform
    weights or not.
    """
    from common.fft_helpers import make_sharded_ifftn_3d

    grid = tuple(int(s) for s in fft_grid)
    ngrid = int(np.prod(grid))
    scale = float(np.sqrt(ngrid / float(cell_volume)))
    f_spin = float(spin_degeneracy)

    psi = jnp.asarray(psi_G, dtype=jnp.complex128)
    p_prod = spec_divisor(mesh, _band_spec(), 1)
    psi, nb_logical = pad_axis_to(psi, p_prod, axis=1)
    nb_pad = int(psi.shape[1])
    ngkmax = int(psi.shape[3])

    occ_j = jnp.asarray(occ, dtype=jnp.float64)
    if int(occ_j.shape[1]) != nb_pad:
        occ_j = jnp.pad(occ_j, ((0, 0), (0, nb_pad - int(occ_j.shape[1]))))

    have_U = U is not None
    if have_U:
        U_j = jnp.asarray(U, dtype=jnp.complex128)
        if U_j.shape[1] != nb_pad or U_j.shape[2] != nb_pad:
            # ZERO pad, not identity.  A pad ROW of U would otherwise build
            # a rotated pad band out of physical ones; zeros keep ψ̃'s pad
            # bands exactly zero, matching ψ's and occ's.
            U_j = jnp.pad(U_j, ((0, 0),
                                (0, nb_pad - int(U_j.shape[1])),
                                (0, nb_pad - int(U_j.shape[2]))))
    else:
        U_j = jnp.zeros((1, 1, 1), dtype=jnp.complex128)   # unused operand

    w_np = np.asarray(kweights, dtype=np.float64)
    if sym_perm is None and w_np.size > 1 and not np.allclose(
            w_np, w_np[0], rtol=0, atol=1e-12):
        raise ValueError(
            f"rho_from_wfns: kweights are non-uniform "
            f"(min {w_np.min():.6g}, max {w_np.max():.6g}), so this k-set is "
            f"reduced, but sym_perm=None.  A weighted sum over a reduced set "
            f"is the density of the representatives, not of the crystal, and "
            f"it still integrates to the exact electron count — so no other "
            f"check here would catch it.  Pass "
            f"centroid.orbit_syms.compute_rgrid_sym_perm(...).")
    w_j = jnp.asarray(kweights, dtype=jnp.float64)
    bidx_j = jnp.asarray(box_index, dtype=jnp.int32)

    band_xy = NamedSharding(mesh, _band_spec())
    m_on_x = NamedSharding(mesh, P(None, "x", None, None))
    U_sh = NamedSharding(mesh, band_rotation_spec())
    box_spec = P(None, ("x", "y"), None, None, None, None)
    ifftn = make_sharded_ifftn_3d(mesh, box_spec, box_spec,
                                  norm="ortho", axes=(-3, -2, -1))
    rho_sharding = NamedSharding(mesh, P(None, None, None))

    def build():
        @jax.jit
        def fn(psi_, U_, occ_, w_, bidx_, sp_):
            # ONE ψ LAYOUT PER BRANCH, AND ONLY WHAT THE BODY READS.  The
            # scanned tuple used to carry BOTH layouts plus a U dummy in
            # both branches, while the body reads ψ_mx alone under
            # ``have_U`` and ψ_xy alone without it.  XLA already dropped
            # the unread ones, so this is a legibility change and not an
            # optimisation: at nk=16 / nb=128 / ngkmax=5120 / grid
            # 60×60×26 on a 2×2 mesh the lowered ``while`` carry is the
            # same tuple before and after —
            # ``(s64, f64[60,60,26], f64[16], c128[16,64,64],
            # c128[16,64,1,5120], s64, s32)``, ONE ψ and not two — with
            # the same four collectives and the same 171.4 MiB temp, and
            # ``vh.rho`` runs 1175.9 → 1165.3 ms under U and 969.9 →
            # 955.3 ms without it (jobs 7889425 → 7889426, inside the
            # run-to-run spread).  Production takes the ``have_U=False``
            # branch: ``hartree_from_orbitals`` rotates once with
            # :func:`rotate_bands` and passes ``U=None``.
            if have_U:
                # m on 'x' so the contraction reduces along 'x' alone.
                psi_s = jax.lax.with_sharding_constraint(psi_, m_on_x)
                U_x = jax.lax.with_sharding_constraint(U_, U_sh)
            else:
                psi_s = jax.lax.with_sharding_constraint(psi_, band_xy)

            def body(rho, xs):
                if have_U:
                    psi_k, U_k, occ_k, w_k, bidx_k = xs
                    # COLUMNS: psi~_n = sum_m Z[m,n] psi_m.  m is on 'x'
                    # so the sum reduces along 'x' alone; n lands on 'y'.
                    psi_t = jnp.einsum('mn,msg->nsg', U_k, psi_k,
                                       optimize=True)
                    # Back onto the WHOLE mesh: straight out of the
                    # rotation the bands sit on 'x' and are replicated on
                    # 'y', which would make the FFT px-fold redundant and
                    # the final reduction a different rule than the
                    # unrotated path's.  One cheap sphere reshard fixes
                    # both.
                    # Same two-step as rotate_bands: land on 'y' (the
                    # contraction's natural output) so the reduce is
                    # (nb/py, ns, ngkmax), then slice to ('x','y') for
                    # free.  One constraint straight to ('x','y')
                    # all-reduces the whole global psi_tilde.
                    psi_t = jax.lax.with_sharding_constraint(
                        psi_t[None],
                        NamedSharding(mesh, P(None, "y", None, None)))
                    psi_t = jax.lax.with_sharding_constraint(
                        psi_t, NamedSharding(mesh, _band_spec()))
                else:
                    psi_k, occ_k, w_k, bidx_k = xs
                    psi_t = psi_k[None]
                box = _box_kernel(psi_t, bidx_k[None], ngkmax=ngkmax)
                psi_r = ifftn(box) * scale
                dens = jnp.einsum('n,knsxyz->xyz', occ_k,
                                  jnp.abs(psi_r) ** 2, optimize=True)
                return rho + (w_k * f_spin) * dens, None

            rho0 = jnp.zeros(grid, dtype=jnp.float64)
            xs = ((psi_s, U_x, occ_, w_, bidx_) if have_U
                  else (psi_s, occ_, w_, bidx_))
            rho, _ = jax.lax.scan(body, rho0, xs)
            rho = jax.lax.with_sharding_constraint(rho, rho_sharding)
            if sym_perm is not None:
                rho = symmetrise_density(rho, sp_)
                rho = jax.lax.with_sharding_constraint(rho, rho_sharding)
            return rho
        return fn

    fn = _cached_jit(
        "rho_from_wfns",
        (psi.shape, tuple(np.shape(U_j)), grid, float(cell_volume), f_spin,
         have_U, None if sym_perm is None else tuple(np.shape(sym_perm)),
         _sharding_key(psi)),
        build)
    # sym_perm is an OPERAND, not a closure.  The cache key can only carry
    # its SHAPE, so two different permutation tables of the same
    # (n_sym, N_r) would otherwise reuse the first one's compiled kernel and
    # return a silently wrong star average -- the one failure the
    # symmetrisation refusal above cannot catch, since an unsymmetrised or
    # WRONGLY symmetrised rho still integrates to the exact electron count.
    # Same class as V_r baked in as a jit constant (audit 2026-08-05).
    sp_j = (jnp.zeros((1, 1), dtype=jnp.int32) if sym_perm is None
            else jnp.asarray(sym_perm, dtype=jnp.int32))
    return fn(psi, U_j, occ_j, w_j, bidx_j, sp_j)


def rho_r_to_G(rho_r, *, mesh: Mesh):
    """ρ(r) → ρ(G) on the full FFT box.  One transform, at the end.

    Separate from :func:`rho_from_wfns` because the r-space density is
    what the symmetrisation and the ∫ρ check want, and because a caller
    building ρ for the ISDF quadrature needs r-space only.
    """
    from common.fft_helpers import make_sharded_fftn_3d

    rho = jnp.asarray(rho_r, dtype=jnp.complex128)
    spec = P(None, None, None)
    fftn = make_sharded_fftn_3d(mesh, spec, spec, norm='backward',
                                axes=(-3, -2, -1))
    return fftn(rho)


# FORCED SYNC.  ψ̃ is a jit OUTPUT — the buffer is materialised whatever
# this row does — so ``watch=True`` waits, it does not transfer and it does
# not add residency.  Nothing is donated across the boundary (the ρ scan
# takes ψ̃ as a plain argument), so the block cannot cost a donation either.
# What it gives up is the overlap between this einsum's device work and the
# handful of Python statements that follow it in
# ``rebuild_hartree_dft_basis`` (shape checks and a ``_cached_jit`` lookup,
# µs).  Without the block this row would be the dispatch alone and the
# rotation would be charged to ``vh.rho``, which is the one split the
# breakdown exists to make.
@timing.timed("vh.rotate_bands", watch=True)
def rotate_bands(psi_G, U_qp, *, mesh: Mesh):
    """ψ̃_nk(G) = Σ_m U_qp[k,m,n] ψ_mk(G) — one sharded einsum, on the sphere.

    Eigenvectors are COLUMNS (``U_qp[k, m, n]`` = component m of QP band
    n), matching ``ffi.linalg``, ``jnp.linalg.eigh`` and
    ``sc_iteration._rotate_to_dft_basis``.

    ``U_qp`` is :func:`band_rotation_spec` — m on 'x', n on 'y' — so the
    sum over m reduces along 'x' alone and no rank holds a full (nb, nb).

    WHY RETURN ψ̃ RATHER THAN FOLD THE ROTATION INTO ρ.  ψ̃ is needed
    TWICE per iteration: once to build ρ, once to contract ⟨m|V_H|n⟩ in
    the QP basis.  Rotating once and reusing it replaces an ``nk·nb³``
    matrix rotation (U†VU) with nothing, and removes a basis change from
    a ~400 Ry term.  Materialising ψ̃ is not an extra cost — the matrix
    elements need it resident anyway.
    """
    psi = jnp.asarray(psi_G, dtype=jnp.complex128)
    p_prod = spec_divisor(mesh, _band_spec(), 1)
    psi, _ = pad_axis_to(psi, p_prod, axis=1)
    nb_pad = int(psi.shape[1])
    U = jnp.asarray(U_qp, dtype=jnp.complex128)
    if U.shape[1] != nb_pad or U.shape[2] != nb_pad:
        # ZERO pad: a pad ROW of U would build a rotated pad band out of
        # physical ones; zeros keep ψ̃'s pad bands exactly zero.
        U = jnp.pad(U, ((0, 0), (0, nb_pad - int(U.shape[1])),
                        (0, nb_pad - int(U.shape[2]))))
    m_on_x = NamedSharding(mesh, P(None, "x", None, None))
    band_xy = NamedSharding(mesh, _band_spec())
    U_sh = NamedSharding(mesh, band_rotation_spec())

    def build():
        @jax.jit
        def fn(psi_, U_):
            psi_mx = jax.lax.with_sharding_constraint(psi_, m_on_x)
            U_x = jax.lax.with_sharding_constraint(U_, U_sh)
            out = jnp.einsum('kmn,kmsg->knsg', U_x, psi_mx, optimize=True)
            # TWO CONSTRAINTS, AND THE ORDER IS THE WHOLE POINT.  The
            # contraction is over m ('x') and n comes from U's 'y', so the
            # NATURAL output is n on 'y', partial over 'x'.  Constraining
            # straight to ('x','y') forced XLA to finish the reduction
            # before it could re-split, i.e. to all-reduce the FULL global
            # psi_tilde across 'x' -- measured c128[16,640,2,11008],
            # 3.36 GiB/rank, 4.93 s at b600/P=64 (audit 2026-08-05).
            #
            # Landing on 'y' first completes the same sum at (nb/py, ns,
            # ngkmax) = 28 MB, and the second constraint is then FREE: a
            # rank holding its whole y-block replicated over 'x' takes its
            # own x-th sub-slice with no communication at all.
            out = jax.lax.with_sharding_constraint(
                out, NamedSharding(mesh, P(None, "y", None, None)))
            return jax.lax.with_sharding_constraint(out, band_xy)
        return fn

    fn = _cached_jit('rotate_bands',
                     (psi.shape, U.shape, _sharding_key(psi)), build)
    return fn(psi, U)


# FORCED SYNC, ALREADY IMPLIED BY THE CALLER.  ``watch=True`` blocks on
# both returned arrays (the section walks tuples).  ``gw_iteration_map``
# does ``np.asarray(E_qp_ry)`` on the next line for the Fermi level and
# ``np.asarray(U_qp)`` a few lines later for the k-star broadcast, so both
# were being synchronised immediately regardless; E and Z come out of one
# FFI call, so blocking on the pair is the same wait as blocking on either.
@timing.timed("sc.eigh", watch=True)
def distributed_eigh_bands(H, *, mesh: Mesh):
    """(E, U_qp) for every k, with NO ``(nb, nb)`` ever on one rank.

    ``H`` is ``(n_k, nb, nb)`` Hermitian at :func:`band_rotation_spec`;
    returns ``E`` ``(n_k, nb)`` replicated ascending and ``U_qp`` at the
    same 2-D layout, eigenvectors as COLUMNS.

    Backend-agnostic.  ``resolve_backend('eigh', 'distributed', mesh)``
    picks the platform's distributed eigh — ScaLAPACK ``pXheevd`` on a
    host mesh, cuSOLVERMp on CUDA — and ``backend_module`` hands out that
    module, which is the idiom every other FFI consumer uses
    (``isdf.core`` for getrf/getrs/solve_lu, ``common.eigh_block_sweep``
    for cusolvermp).  Resolving first is the point: a CPU-only backend on
    a CUDA mesh, an uncompiled handler, a rectangular mesh or an
    indivisible ``n`` are refused THERE with the guard named, instead of
    failing or deadlocking inside the call.

    Batching asymmetry (only ScaLAPACK has a batched entry; cuSOLVERMp
    and SLATE do not) is handled by ``ffi.linalg.dispatch_batched_eigh``,
    not here — it is a backend-capability question, not a physics one.

    ``pXheevd`` is the permanent CPU distributed eigh; the batched entry
    costs one collective-serialisation round for the whole k stack rather
    than one per k.  Chosen against the usual cost argument on purpose:
    ``ffi/linalg/dispatch.py`` records that the native path solves ndev
    matrices at once and wins by roughly ndev whenever a tile fits on one
    device, which at nb=640 (6.5 MB) it does.  At nb=10⁴ a tile is 1.6 GB
    and at 2·10⁴ it is 6.4 GB, on ONE device on top of ψ and the FFT
    boxes — the owner's ruling is robustness there over speed here
    (2026-08-04).

    GAUGE.  ``U_qp`` is not mesh-invariant (a degenerate eigenvalue leaves
    an arbitrary unitary in its subspace and the block-cyclic reduction
    picks a different representative per grid).  ρ is unaffected: it is
    ``U f(E) U†`` contracted with ψ, gauge-invariant whenever f is
    constant across a degenerate set — which a step occupation from
    :func:`gw.efermi.fermi_level_step` is, because that routine refuses to
    split a degenerate manifold.  Same guarantee, twice.
    """
    from ffi.linalg import dispatch_batched_eigh

    H_j = jnp.asarray(H)
    p_prod = spec_divisor(mesh, _band_spec(), 1)
    H_j, _ = pad_axis_to(H_j, p_prod, axis=1)
    H_j, _ = pad_axis_to(H_j, p_prod, axis=2)
    H_j = jax.lax.with_sharding_constraint(
        H_j, NamedSharding(mesh, band_rotation_spec()))
    # pXheevd reads ONE triangle, so a non-Hermitian input is silently
    # interpreted rather than refused.  Hermitise here.
    H_j = 0.5 * (H_j + jnp.conj(jnp.swapaxes(H_j, -1, -2)))
    return dispatch_batched_eigh(H_j, mesh, "distributed")


def hartree_from_orbitals(psi_G, occ, kweights, wfn, *, mesh: Mesh,
                          box_index, fft_grid, truncation_2d: bool,
                          spin_degeneracy: float, sym_perm=None,
                          expected_electrons=None, print_fn=print):
    """ψ → ρ(r) → V_H(r).  ``psi_G`` must ALREADY be the rotated orbitals.

    No ``U`` argument: rotate once with :func:`rotate_bands` and pass ψ̃
    here, so the same ψ̃ also builds ⟨m|V_H|n⟩ directly in the QP basis.
    Rotating inside would force the matrix elements to be built in the DFT
    basis and rotated afterwards — an ``nk·nb³`` U†VU and a second basis
    change on a ~400 Ry term, both avoidable.

    Composes :func:`rho_from_wfns` with
    ``psp.get_DFT_mtxels.build_hartree_potential``, which takes ρ in REAL
    space and does its own transform — so V_H lands on the ψ box grid,
    which is what ``common.mtxel_sweep.local_potential_operator`` needs,
    and no ρ(G) step enters.

    ``expected_electrons`` is ``f_spin · Σ_k w_k Σ_n f_nk``.  Pass it: it
    is the only check that a rotated, re-occupied density still holds the
    right charge, and V_H is a ~400 Ry term.
    """
    from psp.get_DFT_mtxels import build_hartree_potential

    rho_r = rho_from_wfns(psi_G, occ, kweights, mesh=mesh,
                          box_index=box_index, fft_grid=fft_grid,
                          cell_volume=float(wfn.cell_volume),
                          spin_degeneracy=spin_degeneracy,
                          U=None, sym_perm=sym_perm)
    # NO FORCED SYNC NEEDED — this stage is synchronous by construction.
    # ``build_hartree_potential`` computes ``float(jnp.sum(rho_r))`` for the
    # charge check on entry and ``float(jnp.sum(rho_r * V_H_r))`` for the
    # Hartree energy on exit; both are host readbacks, so the row already
    # ends after V_H(r) exists.  ρ's own compute is charged to ``vh.rho``,
    # which blocks before this section opens, so what is left here is the
    # 1/G² solve and its two reductions.
    with timing.section("vh.poisson"):
        V_H_r = build_hartree_potential(
            rho_r, wfn, truncation_2d=bool(truncation_2d),
            expected_electrons=expected_electrons, print_fn=print_fn)
    return rho_r, V_H_r
