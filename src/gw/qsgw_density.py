"""ρ(r) and ρ(G) from rotated orbitals with per-state occupations.

The SC map rebuilds scalar and current Hartree fields from the iteration
orbitals on raw IBZ rows. File k weights and typed symmetry projection
complete the density; full-BZ inputs remain available for numerical controls.

STRUCTURE: ONE SCAN, U NEVER REPLICATED
---------------------------------------
A single ``lax.scan`` over k.  Per k, in this order:

    ψ_m^X   ← reshard ψ[k] so the CONTRACTED band index m lies on 'x'
    ψ̃_n^Y  ← einsum('mn,msg->nsg', Z[k], ψ_m^X)     # Z is P('x','y')
    ψ̃_n^XY ← reshard onto the whole mesh
    box → ifft → ρ(r) += w_k · f_spin · Σ_n f_nk |ψ̃_nk(r)|²

``Z`` is sharded ``P(None, 'x', 'y')`` — m on 'x', n on 'y' — so no rank
ever holds a full ``(nb, nb)``.  That is the NATIVE layout of both
eigenvector producers (``distrib_la``'s ScaLAPACK ``batched_distributed_eigh``
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
* fractional occupations, including the signed MP1 weights used for metallic
  self-consistency, may make every band contribute, so a code that slices to
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
from runtime.padding import pad_axis, spec_divisor


__all__ = ["rho_from_wfns", "rho_r_to_G", "band_rotation_spec",
           "distributed_eigh_bands", "symmetrise_density",
           "hartree_from_orbitals", "rotate_bands",
           "rotate_band_axis", "rotate_band_matrix"]


def _band_spec() -> P:
    """ψ layout: (n_k, nb, nspinor, ngkmax), bands over the whole mesh.

    Imported from ``common.wfn_layout`` rather than re-spelled: a second
    literal of the same PartitionSpec is exactly the drift that
    ``runtime.padding.spec_divisor`` was introduced to remove on the band
    DIVISOR, and it would be worse here — a spec that disagreed would not
    raise, it would silently insert a reshard between the loader, the
    matrix-element sweep and this density build.
    """
    from common.wfn_layout import band_sphere_spec
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


# ---------------------------------------------------------------------------
# THE band-rotation primitive.  One operation, two uses.
# ---------------------------------------------------------------------------
#
# Every U consumer in the tree does the same thing: contract ONE index of
# ``U`` at :func:`band_rotation_spec` against ONE band axis of an operand.
#
#   (a) state rotation      psi~_n     = sum_m U[m,n] psi_m
#   (b) similarity transform A -> U A U^dagger  (and U^dagger A U)
#
# and (b) is (a) applied once per index.  :func:`rotate_band_axis` is (a);
# :func:`rotate_band_matrix` is (b) written as two calls to it.  Both are
# TRACED HELPERS, not jit boundaries -- the caller keeps its own jit and
# its own cache key, so the compiled graph is exactly the constraints and
# einsum written here, with nothing inserted between them.
#
# THE POINT IS WHERE U LIVES.  U stays at ``band_rotation_spec`` (m on
# 'x', n on 'y'); no rank ever holds a full (nb, nb).  The contracted
# index rides U's mesh axis and the free index lands on the OTHER one, so
# each application is one psum along one mesh axis.

def rotate_band_axis(X, U, *, mesh: Mesh, axis: int, to_qp: bool,
                     conj_u: bool = False):
    """Rotate ONE band axis of ``X`` through ``U``, U kept distributed.

    ``U[k, m, n] = <DFT_m | QP_n>`` -- eigenvectors are COLUMNS, the same
    convention as ``distrib_la``, ``jnp.linalg.eigh``,
    :func:`distributed_eigh_bands` and
    ``wavefunction_bundle.rotate_wavefunctions``.

    ``to_qp`` picks which index of U is contracted, i.e. which way the
    axis is carried:

    * ``True``  -- ``Y[..., n, ...] = sum_m U[k, m, n] X[..., m, ...]``.
      ``X``'s ``axis`` is a DFT band index and becomes a QP one.  The sum
      is over U's 'x' index, so the reduction is along 'x' and the free
      index lands on 'y'.
    * ``False`` -- ``Y[..., m, ...] = sum_n U[k, m, n] X[..., n, ...]``.
      ``X``'s ``axis`` is a QP band index and becomes a DFT one.  Mirror
      image: reduce along 'y', land on 'x'.

    ``conj_u`` uses ``conj(U)`` as the kernel.  A similarity transform
    needs it on EXACTLY ONE of its two applications -- the bra index
    carries the complex conjugate of the ket index's transformation.  Use
    :func:`rotate_band_matrix` rather than pairing the calls by hand.
    Of the two ways to pair them wrong, only one is loud: an unflipped
    ``conj_u`` breaks hermiticity (measured max|Y - Y^dagger| = 3.9 on a
    Hermitian operand at nk=8/nb=32, job 7889851), while the wrong
    ``to_qp`` -- ``U A U^dagger`` where ``U^dagger A U`` was meant -- is
    Hermitian, has the same trace and the same Frobenius norm, so no
    invariance check can see it.  The gate pins both against an explicit
    host rotation.

    SHARDING CONTRACT, and it is the whole reason this is one function:

    * ``U``  -> :func:`band_rotation_spec`, always.
    * ``X``  -> ``axis`` on the CONTRACTED mesh axis, every other axis
      replicated.  On an operand that arrives replicated this is a free
      slice; on one that arrives sharded elsewhere it is the reshard the
      contraction needs, emitted here instead of being inferred.
    * ``Y``  -> ``axis`` on the FREE mesh axis, every other axis
      replicated.

    Landing on U's own free axis rather than on the caller's final layout
    is deliberate and measured: constraining straight to a re-split layout
    forced XLA to finish the reduction before it could re-split, i.e. to
    all-reduce the FULL global result -- c128[16,640,2,11008], 3.36
    GiB/rank, 4.93 s at b600/P=64 (audit 2026-08-05, ``rotate_bands``).
    A caller that wants another layout constrains again afterwards; from
    the free axis that second step is local.

    ``axis`` must not be 0 -- axis 0 is k, the batch index U is indexed
    by.
    """
    axis = int(axis)
    nd = int(getattr(X, "ndim", np.ndim(X)))
    if axis <= 0 or axis >= nd:
        raise ValueError(
            f"rotate_band_axis: axis={axis} out of range for a {nd}-D "
            f"operand, and axis 0 is k (the batch index U is indexed by).")
    contract_ax, free_ax = ("x", "y") if to_qp else ("y", "x")
    in_spec = P(*[contract_ax if i == axis else None for i in range(nd)])
    out_spec = P(*[free_ax if i == axis else None for i in range(nd)])

    U = jax.lax.with_sharding_constraint(
        U, NamedSharding(mesh, band_rotation_spec()))
    X = jax.lax.with_sharding_constraint(X, NamedSharding(mesh, in_spec))

    # Subscripts: axis 0 is 'k'; the rotated axis takes U's contracted
    # letter on input and U's free letter on output.  'k', 'm', 'n' are
    # reserved for U, so the operand's other axes are lettered from 'a'.
    other = "abcdefghij"
    x_sub = ["k"] + [other[i] for i in range(1, nd)]
    j = x_sub[axis]
    u_sub = f"k{j}n" if to_qp else f"km{j}"
    o_sub = list(x_sub)
    o_sub[axis] = "n" if to_qp else "m"
    out = jnp.einsum(f"{u_sub},{''.join(x_sub)}->{''.join(o_sub)}",
                     jnp.conj(U) if conj_u else U, X, optimize=True)
    return jax.lax.with_sharding_constraint(out, NamedSharding(mesh, out_spec))


def rotate_band_matrix(A, U, *, mesh: Mesh, to_qp: bool):
    """``A_QP = U^dagger A_DFT U`` (``to_qp``) or ``A_DFT = U A_QP U^dagger``.

    ``A`` is ``(n_k, nb, nb)`` with axis 1 the bra index and axis 2 the
    ket index; ``U`` is at :func:`band_rotation_spec` and stays there.
    Two :func:`rotate_band_axis` calls, ket first, with ``conj_u``
    flipped between them -- see there for why that flip is the one thing
    no invariance check can catch.

    THE OUTPUT IS SHARDED, ``P(None, free_ax, None)``.  A caller that
    needs it replicated (the SC carry is read on the host) says so; this
    function does not, because that gather is the caller's cost and the
    point here is that ``U`` never pays it.

    Ket first is not arbitrary: it leaves the intermediate's free index
    on the mesh axis the second call's ``in_spec`` does NOT claim, so the
    reshard between them is an all-to-all of the TILE, not a gather.
    Census at nk=8/nb=32 on a 2x2 mesh (job 7889851), one full
    (nk, nb, nb) = 0.1250 MiB, both directions identical:

        all-reduce         c128[8,32,16]    0.0625 MiB   group=2
        all-to-all         c128[8,1,16,16]  0.0312 MiB   group=2
        collective-permute c128[8,16,32]    0.0625 MiB
        all-reduce         c128[8,16,32]    0.0625 MiB   group=2

    -- ``nb^2/(px*py)`` for the reshard, ``nb^2/px`` for the two psums,
    and nothing full-size.  The only full-size collective in the shipped
    seam is the CALLER's gather of the result to replicated.
    """
    A = rotate_band_axis(A, U, mesh=mesh, axis=2,
                         to_qp=to_qp, conj_u=not to_qp)
    return rotate_band_axis(A, U, mesh=mesh, axis=1,
                            to_qp=to_qp, conj_u=to_qp)


def diagonal_rotated_band_matrix(A, U, *, mesh: Mesh, to_qp: bool):
    """Diagonal of a basis-rotated band matrix, without the dense result.

    ``A`` is ``(n_k, ..., nb, nb)``: any axes between k and the final two
    band axes are batches.  The convention is identical to
    :func:`rotate_band_matrix`, but after the canonical ket-first rotation
    the bra transform is contracted directly onto its diagonal.  Thus the
    only dense ``nb x nb`` temporary is the same sharded half-rotation the
    full transform requires; no completed rotated matrix is materialised.

    The result is ``(n_k, ..., nb)`` with its final band axis sharded on the
    rotation's free mesh axis.  Gathering that small diagonal, when needed,
    remains the caller's explicit cost.
    """
    nd = int(getattr(A, "ndim", np.ndim(A)))
    if nd < 3:
        raise ValueError(
            "diagonal_rotated_band_matrix requires (nk,...,nb,nb); "
            f"got {nd} dimensions")
    if int(A.shape[-2]) != int(A.shape[-1]):
        raise ValueError(
            "diagonal_rotated_band_matrix requires square trailing band "
            f"axes; got {tuple(A.shape[-2:])}")

    half = rotate_band_axis(
        A, U, mesh=mesh, axis=nd - 1,
        to_qp=to_qp, conj_u=not to_qp)
    # The half rotation is (..., bra, ket_out).  Partition its bra on the
    # contraction mesh axis while retaining the output band on the free one;
    # the einsum then needs only the corresponding one-axis psum.
    contract_ax, free_ax = ("x", "y") if to_qp else ("y", "x")
    half_spec = P(
        None, *([None] * (nd - 3)), contract_ax, free_ax)
    half = jax.lax.with_sharding_constraint(
        half, NamedSharding(mesh, half_spec))
    U = jax.lax.with_sharding_constraint(
        U, NamedSharding(mesh, band_rotation_spec()))

    if to_qp:
        # half[...,m,n] = sum_j A[...,m,j] U[j,n]
        # diag[n] = sum_m conj(U[m,n]) half[...,m,n]
        diagonal = jnp.einsum(
            "kmn,k...mn->k...n", jnp.conj(U), half, optimize=True)
    else:
        # half[...,n,m] = sum_p A[...,n,p] conj(U[m,p])
        # diag[m] = sum_n U[m,n] half[...,n,m]
        diagonal = jnp.einsum(
            "kmn,k...nm->k...m", U, half, optimize=True)
    out_spec = P(None, *([None] * (nd - 3)), free_ax)
    return jax.lax.with_sharding_constraint(
        diagonal, NamedSharding(mesh, out_spec))


def symmetrise_density(rho_r, sym_perm):
    """ρ_sym[r] = (1/n_sym) Σ_s ρ[sym_perm[s, r]] — the star average.

    ``sym_perm`` is ``symmetry_maps.fft_grid_pullback_perm``'s table,
    ``sym_perm[s, r_new] = r_old`` with ``r_{r_new} ≡ S_s·r_{r_old} + τ_s``
    on the FFT grid, so the fractional translations are already handled and
    this is a pure gather.

    IDEMPOTENT IFF THE TABLE IS A GROUP.  The star average is a projector
    onto the symmetric subspace only when the permutations are closed
    under composition, which ``fft_grid_pullback_perm`` emits and an
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
                  U=None, sym_perm=None, sym=None, include_dirac_current: bool = False,
                  charge_nspinor: int | None = None,
                  return_spin_density_matrix: bool = False):
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
        ``distrib_la`` and ``jnp.linalg.eigh`` return and what
        ``sc_iteration`` already assumes; do not pass a transpose.
        ``None`` builds ρ from ``psi_G`` unrotated — the DFT density and
        the gate's baseline.

    ``include_dirac_current=True`` requires four-component orbitals and
    returns ``(rho,Jx,Jy,Jz)`` from the SAME inverse FFT and the SAME signed
    per-state occupations.  ``charge_nspinor`` may restrict only ``rho`` to
    a leading carrier block (the Pauli-reference model); all three currents
    continue to use the full four-spinor. On reduced k sets, ``sym_perm``
    projects charge and the required ``sym`` applies the typed polar current
    projection; scalar pullbacks never rotate current components. The local contraction is shared
    with :func:`psp.get_DFT_mtxels.valence_density_from_kpoint`.

    ``return_spin_density_matrix=True`` requires two-component Pauli
    spinors and returns raw
    ``rho_ab(r)=sum_kn w_k f_nk psi_kna(r) psi_knb(r)*`` from the same scan.
    It refuses
    ``sym_perm`` because a scalar pullback cannot supply the spin rotation,
    axial parity, or antiunitary action.  The symmetry service can apply
    those operations to this raw matrix without duplicating the density
    build.

    Returns
    -------
    ``(nx,ny,nz)`` or ``(4,nx,ny,nz)`` float64, or
    ``(2,2,nx,ny,nz)`` complex128 in spin-matrix mode; replicated.

    NORMALISATION is term-for-term ``psp.get_DFT_mtxels.
    valence_density_from_kpoint``: ψ_r = ifftn(box, 'ortho')·√(N/Ω), so
    ``ΔV · Σ_r ρ = f_spin · Σ_k w_k Σ_n f_nk``.

    THE CARRY IS ρ(r) — ``(nx, ny, nz)`` f64, 750 kB at a 60×60×26 grid.
    The scan carries something negligible and the band reduction is folded
    into the accumulation, so there is ONE collective class for the whole
    build rather than one materialised ψ̃ per k.

    SYMMETRISATION IS ENFORCED, NOT DOCUMENTED.  ``sym_perm`` (from
    ``symmetry_maps.fft_grid_pullback_perm``) makes the result the
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
    from psp.get_DFT_mtxels import density_components_from_psi_r

    grid = tuple(int(s) for s in fft_grid)
    ngrid = int(np.prod(grid))
    scale = float(np.sqrt(ngrid / float(cell_volume)))
    f_spin = float(spin_degeneracy)

    psi = jnp.asarray(psi_G, dtype=jnp.complex128)
    include_current = bool(include_dirac_current)
    return_spin_matrix = bool(return_spin_density_matrix)
    ns = int(psi.shape[2])
    charge_ns = ns if charge_nspinor is None else int(charge_nspinor)
    if not 0 < charge_ns <= ns:
        raise ValueError(
            "charge_nspinor must select a nonempty leading spinor block; "
            f"got {charge_nspinor} for nspinor={ns}")
    if include_current and ns != 4:
        raise ValueError(
            "rho_from_wfns: Dirac current requires four-component "
            f"bispinors; got nspinor={ns}")
    if include_current and sym_perm is not None and sym is None:
        raise ValueError(
            "rho_from_wfns: reduced four-current density requires SymMaps "
            "for the polar current projection as well as scalar sym_perm.")
    if return_spin_matrix and include_current:
        raise ValueError(
            "rho_from_wfns: return_spin_density_matrix and "
            "include_dirac_current are mutually exclusive")
    if return_spin_matrix and ns != 2:
        raise ValueError(
            "rho_from_wfns: spin-density matrix requires exactly "
            f"two-component Pauli spinors; got nspinor={ns}")
    if return_spin_matrix and charge_nspinor is not None:
        raise ValueError(
            "rho_from_wfns: charge_nspinor does not apply to the complete "
            "two-component spin-density matrix")
    if return_spin_matrix and sym_perm is not None:
        raise ValueError(
            "rho_from_wfns: sym_perm is a scalar-density projector and "
            "cannot act on rho_ab. Request the raw matrix with "
            "sym_perm=None, then apply the canonical spinor/axial/TR "
            "symmetry action.")
    p_prod = spec_divisor(mesh, _band_spec(), 1)
    _pad = pad_axis(psi, p_prod, axis=1)
    psi, nb_logical = _pad.array, _pad.logical   # LOGICAL, by name
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
    if (not return_spin_matrix and sym_perm is None and w_np.size > 1
            and not np.allclose(
                w_np, w_np[0], rtol=0, atol=1e-12)):
        raise ValueError(
            f"rho_from_wfns: kweights are non-uniform "
            f"(min {w_np.min():.6g}, max {w_np.max():.6g}), so this k-set is "
            f"reduced, but sym_perm=None.  A weighted sum over a reduced set "
            f"is the density of the representatives, not of the crystal, and "
            f"it still integrates to the exact electron count — so no other "
            f"check here would catch it.  Pass "
            f"symmetry_maps.fft_grid_pullback_perm(...).")
    w_j = jnp.asarray(kweights, dtype=jnp.float64)
    bidx_j = jnp.asarray(box_index, dtype=jnp.int32)

    band_xy = NamedSharding(mesh, _band_spec())
    m_on_x = NamedSharding(mesh, P(None, "x", None, None))
    U_sh = NamedSharding(mesh, band_rotation_spec())
    box_spec = P(None, ("x", "y"), None, None, None, None)
    ifftn = make_sharded_ifftn_3d(mesh, box_spec, box_spec,
                                  norm="ortho", axes=(-3, -2, -1))
    field_shape = ((2, 2, *grid) if return_spin_matrix else
                   ((4, *grid) if include_current else grid))
    field_spec = (P(None, None, None, None, None) if return_spin_matrix else
                  (P(None, None, None, None) if include_current
                   else P(None, None, None)))
    rho_sharding = NamedSharding(mesh, field_spec)

    def build():
        @jax.jit
        def fn(psi_, U_, occ_, w_, bidx_, sp_):
            # Keep the resident all-k sphere in its canonical two-axis band
            # layout.  The rotation needs m on x only for the current k
            # point, so reshard that singleton slice inside the scan instead
            # of replicating the full psi array over y for the scan lifetime.
            psi_s = jax.lax.with_sharding_constraint(psi_, band_xy)
            if have_U:
                U_x = jax.lax.with_sharding_constraint(U_, U_sh)

            def body(rho, xs):
                if have_U:
                    psi_k, U_k, occ_k, w_k, bidx_k = xs
                    # COLUMNS: psi~_n = sum_m Z[m,n] psi_m.  m is on 'x'
                    # so the sum reduces along 'x' alone; n lands on 'y'.
                    psi_k_x = jax.lax.with_sharding_constraint(
                        psi_k[None], m_on_x)[0]
                    psi_t = jnp.einsum('mn,msg->nsg', U_k, psi_k_x,
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
                dens = density_components_from_psi_r(
                    psi_r[0], occ_k,
                    include_dirac_current=include_current,
                    charge_nspinor=(None if return_spin_matrix else charge_ns),
                    return_spin_density_matrix=return_spin_matrix)
                return rho + (w_k * f_spin) * dens, None

            field_dtype = (jnp.complex128 if return_spin_matrix
                           else jnp.float64)
            rho0 = jnp.zeros(field_shape, dtype=field_dtype)
            xs = ((psi_s, U_x, occ_, w_, bidx_) if have_U
                  else (psi_s, occ_, w_, bidx_))
            rho, _ = jax.lax.scan(body, rho0, xs, unroll=1)
            rho = jax.lax.with_sharding_constraint(rho, rho_sharding)
            if sym_perm is not None:
                if include_current:
                    rho = rho.at[0].set(symmetrise_density(rho[0], sp_))
                else:
                    rho = symmetrise_density(rho, sp_)
                rho = jax.lax.with_sharding_constraint(rho, rho_sharding)
            return rho
        return fn

    fn = _cached_jit(
        "rho_from_wfns",
        (psi.shape, tuple(np.shape(U_j)), grid, float(cell_volume), f_spin,
         have_U, include_current, charge_ns, return_spin_matrix,
         None if sym_perm is None else tuple(np.shape(sym_perm)),
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
    result = fn(psi, U_j, occ_j, w_j, bidx_j, sp_j)
    if include_current and sym is not None:
        from symmetry_maps import project_polar_fft_field
        projected = project_polar_fft_field(np.asarray(result[1:]), sym)
        result = result.at[1:].set(jnp.asarray(projected.field))
    return result


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
    n), matching ``distrib_la``, ``jnp.linalg.eigh`` and
    ``sc_iteration._rotate_to_dft_basis``.

    ``U_qp`` is :func:`band_rotation_spec` — m on 'x', n on 'y' — so the
    sum over m reduces along 'x' alone and no rank holds a full (nb, nb).

    This is :func:`rotate_band_axis` on ψ's band axis; what is local here
    is the mesh padding (ψ's band axis is padded to the mesh divisor and
    U is zero-padded to match) and the second constraint that re-splits
    the result from U's free axis 'y' onto ``_band_spec``'s ('x','y').

    WHY RETURN ψ̃ RATHER THAN FOLD THE ROTATION INTO ρ.  ψ̃ is needed
    TWICE per iteration: once to build ρ, once to contract ⟨m|V_H|n⟩ in
    the QP basis.  Rotating once and reusing it replaces an ``nk·nb³``
    matrix rotation (U†VU) with nothing, and removes a basis change from
    a ~400 Ry term.  Materialising ψ̃ is not an extra cost — the matrix
    elements need it resident anyway.
    """
    psi = jnp.asarray(psi_G, dtype=jnp.complex128)
    p_prod = spec_divisor(mesh, _band_spec(), 1)
    psi = pad_axis(psi, p_prod, axis=1).array
    nb_pad = int(psi.shape[1])
    U = jnp.asarray(U_qp, dtype=jnp.complex128)
    if U.shape[1] != nb_pad or U.shape[2] != nb_pad:
        # ZERO pad: a pad ROW of U would build a rotated pad band out of
        # physical ones; zeros keep ψ̃'s pad bands exactly zero.
        U = jnp.pad(U, ((0, 0), (0, nb_pad - int(U.shape[1])),
                        (0, nb_pad - int(U.shape[2]))))
    band_xy = NamedSharding(mesh, _band_spec())

    def build():
        @jax.jit
        def fn(psi_, U_):
            # ``rotate_band_axis`` emits psi at P(None,'x',None,None), U at
            # band_rotation_spec, the einsum, and the landing constraint
            # P(None,'y',None,None) -- U's own free axis.
            out = rotate_band_axis(psi_, U_, mesh=mesh, axis=1, to_qp=True)
            # THE SECOND CONSTRAINT, AND THE ORDER IS THE WHOLE POINT.
            # Constraining straight to ('x','y') forced XLA to finish the
            # reduction before it could re-split, i.e. to all-reduce the
            # FULL global psi_tilde across 'x' -- measured
            # c128[16,640,2,11008], 3.36 GiB/rank, 4.93 s at b600/P=64
            # (audit 2026-08-05).  From the 'y' landing this step is FREE:
            # a rank holding its whole y-block replicated over 'x' takes
            # its own x-th sub-slice with no communication at all.
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
# Pad-diagonal sentinel for the band-axis pad below.  Ry; physical QP
# eigenvalues are O(1) Ry, so this is ~10 orders clear and the pad
# eigenvalues cannot interleave with the physical spectrum at any deck.
# Same spelling as the tree's other sentinel, psp/dft_operators.py:736.
_EIGH_PAD_SENTINEL_RY = 1e10


@timing.timed("sc.eigh", watch=True)
def distributed_eigh_bands(H, *, mesh: Mesh,
                           distrib_la_batched_route: str = "batch_reshard",
                           distrib_la_backend: str = "distributed"):
    """(E, U_qp) for every k through the ``distrib_la`` batched surface.

    ``H`` is ``(n_k, nb, nb)`` Hermitian at :func:`band_rotation_spec`;
    returns ``E`` ``(n_k, nb)`` replicated ascending and ``U_qp`` at the
    same 2-D layout, eigenvectors as COLUMNS — at the **logical** ``nb``,
    whatever the mesh's band divisor.  The default distributed backend
    spreads each tile over the mesh, so no rank holds a full ``(nb,nb)``.
    The explicit ``off`` + ``batch_reshard`` route instead gives each rank
    whole fit-size matrices from the k batch, which is faster while a tile
    fits its memory budget.  A band axis that the divisor does not divide
    is padded to it with a large diagonal SENTINEL and sliced back BY COUNT
    after the solve; callers never see a padded shape.

    PARITY, and its limit.  Padding is **reduction-order gauge, not
    bit-exactness**.  The pad modes are exactly inert — the block is
    block-diagonal with zero coupling, so physical eigenpairs are
    unchanged in exact arithmetic — but a wider array changes how XLA
    *groups* the nonzeros inside full-array reductions, so ``E`` at
    ``nb_pad > nb`` can differ from ``E`` at ``nb_pad == nb`` in the last
    few ulp.  Drift scales with reduction length and with the trajectory,
    not with a fixed ULP count: measured 0.2 eps at ``nk*nb^2 = 243`` and
    39.9 eps at 29768, a contracting run staying <= 8.3 eps while a
    stalled one reached 2.9e5.  What bounds it is the RESIDUAL, not eps:
    ``|dH| <= 6.1e-8`` of the per-element residual norm.  Do not gate this
    on a ULP count — that passes on a fixture and fails at production
    shapes.

    Backend-agnostic.  The default
    ``resolve_backend('eigh', 'distributed', mesh)`` picks the platform's
    distributed eigh — ScaLAPACK ``pXheevd`` on a host mesh, cuSOLVERMp on
    CUDA — and ``backend_module`` hands out that module, which is the idiom
    every other FFI consumer uses
    (``isdf.core`` for getrf/getrs/solve_lu, ``common.eigh_block_sweep``
    for cusolvermp).  Resolving first is the point: a CPU-only backend on
    a CUDA mesh, an uncompiled handler, a rectangular mesh or an
    indivisible ``n`` are refused THERE with the guard named, instead of
    failing or deadlocking inside the call.

    ``distrib_la_backend='off'`` with
    ``distrib_la_batched_route='batch_reshard'`` is the service-owned
    small-matrix route: stage the k batch over the mesh, run local native
    eighs concurrently, and stage the eigenvectors back to the same face
    layout.  It deliberately shares this padding and column-convention seam
    with the distributed route; callers do not grow a second eigensolver.

    Batching asymmetry (only ScaLAPACK has a batched entry; cuSOLVERMp
    and SLATE do not) is handled by ``distrib_la.dispatch_batched_eigh``,
    not here — it is a backend-capability question, not a physics one.

    ``pXheevd`` is the permanent CPU distributed eigh; the batched entry
    costs one collective-serialisation round for the whole k stack rather
    than one per k.  Chosen against the usual cost argument on purpose:
    ``distrib_la.dispatch`` records that the native path solves ndev
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
    from ffi import _services
    _services.ensure_on_path()
    from distrib_la import dispatch_batched_eigh

    H_j = jnp.asarray(H)
    nb = int(H_j.shape[1])
    p_prod = spec_divisor(mesh, _band_spec(), 1)
    H_j = pad_axis(H_j, p_prod, axis=1).array
    H_j = pad_axis(H_j, p_prod, axis=2).array
    nb_pad = int(H_j.shape[1])
    H_j = jax.lax.with_sharding_constraint(
        H_j, NamedSharding(mesh, band_rotation_spec()))
    # pXheevd reads ONE triangle, so a non-Hermitian input is silently
    # interpreted rather than refused.  Hermitise here.
    H_j = 0.5 * (H_j + jnp.conj(jnp.swapaxes(H_j, -1, -2)))
    if nb_pad != nb:
        # SENTINEL pad, then drop BY COUNT.  ``pad_axis`` zero-fills, so
        # the pad block is [H 0; 0 0] and the pad eigenvalues are exactly
        # 0.0 — which sort into the MIDDLE of a Ry spectrum whose occupied
        # states are negative.  Two things then go wrong, and only the
        # first is obvious: band order moves (so `E[:, :n_occ]` straddles
        # the wrong bands, and `_midgap_efermi`/`fermi_level_step`/
        # `step_occupations` all read a wrong number), and — the quieter
        # one — if H itself has an eigenvalue near 0 the degenerate
        # subspace MIXES physical and pad eigenvectors, so no post-hoc
        # rule can separate them.  An identity pad (λ = 1 Ry = 13.6 eV)
        # fixes neither: it still lands inside a realistic QP window.
        #
        # With a large sentinel on the pad diagonal the block stays exactly
        # block-diagonal with zero coupling, so the padded subspace
        # decouples EXACTLY: physical eigenvectors keep zero weight on pad
        # rows, physical eigenvalues are unchanged, and the sentinel
        # eigenvalues provably occupy the LAST nb_pad-nb slots of an
        # ascending spectrum.  That is what makes "drop by count" correct
        # rather than probably-correct — the slice below is exact, not a
        # threshold.  Same argument the transverse rank-truncate makes for
        # its zero pad (isdf/core.py:2699-2714): a decoupled pad block
        # deflates exactly; only the safe VALUE differs, because there the
        # pad modes are dropped by |lambda| and here by position.
        #
        # Sentinel value: 1e10, the tree's existing spelling
        # (psp/dft_operators.py:736,759). Physical eigenvalues here are
        # O(1) Ry, so it is ~10 orders clear. NOTE it is safe *because it
        # is dropped*: common/wfn_transforms.py:1606-1618 deliberately
        # chose a FINITE max(E)+1 Ry sentinel for band energies that are
        # KEPT, since those flow into PPM resolvents 1/(w - e + i.eta).
        # These do not survive this function, and a value that is absurd
        # on sight makes a future leak loud instead of silent.
        i = jnp.arange(nb_pad)[:, None]
        j = jnp.arange(nb_pad)[None, :]
        on_pad_diag = ((i == j) & (i >= nb))[None]
        H_j = jnp.where(on_pad_diag, _EIGH_PAD_SENTINEL_RY, H_j)
    E, U = dispatch_batched_eigh(
        H_j, mesh, distrib_la_backend,
        batched_route=distrib_la_batched_route)
    if nb_pad != nb:
        # THE SEAM. Drop by COUNT, never by value — same shape contract the
        # rCROP carry restores at sc_iteration.py:1509-1514. Callers get the
        # LOGICAL extent, so no band-indexed operand beside the carry has to
        # know a pad ever existed.
        E = E[:, :nb]
        U = U[:, :nb, :nb]
    return E, U


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
