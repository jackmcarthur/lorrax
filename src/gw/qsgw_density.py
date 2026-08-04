"""ρ(r) and ρ(G) from rotated orbitals with per-state occupations.

The piece that turns the fixed-density QSGW loop into a density-updating
one.  ``sc_iteration`` today rotates V_H as a BASIS CHANGE (U†V_H U) and
leaves ρ at its DFT value; this rebuilds ρ from the rotated orbitals, so
the Hartree potential of iteration n+1 comes from iteration n's density
rather than from the DFT one.

STRUCTURE: ONE SCAN, U NEVER REPLICATED
---------------------------------------
A single ``lax.scan`` over k.  Per k, in this order:

    ψ_m^Y   ← reshard ψ[k] so the CONTRACTED band index m lies on 'y'
    ψ̃_n^X  ← einsum('nm,msg->nsg', U[k], ψ_m^Y)     # U is P('x','y')
    ψ̃_n^XY ← reshard onto the whole mesh
    box → ifft → ρ(r) += w_k · f_spin · Σ_n f_nk |ψ̃_nk(r)|²

``U`` is sharded ``P(None, 'x', 'y')`` — n on 'x', m on 'y' — so no rank
ever holds a full ``(nb, nb)``.  At nb=640/P=64 each rank carries
80×80×16 B = 102 kB per k against 6.5 MB replicated, and the replicated
form is the ``(nk, nb, nb)`` W2-class object that reaches 9.2 GB at
nb=2000/nk=144.  The contraction index m is on 'y' and the output index n
on 'x', so the sum over m is a reduction along 'y' ALONE, not a global
collective.

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

from common.wfn_transforms import _box_kernel, _cached_jit, _sharding_key
from runtime.padding import pad_axis_to, spec_divisor


__all__ = ["rho_from_wfns", "rho_r_to_G", "band_rotation_spec"]


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
    """``U`` layout: ``(n_k, nb, nb)`` with n on 'x' and m on 'y'.

    The eigenvector matrix from ``sc_iteration``, sharded so no rank holds
    a full ``(nb, nb)``.  Ask ``_make_kshard_eigh(..., u_spec=...)`` for it
    rather than resharding a replicated U after the fact — the replicated
    intermediate is the whole cost being avoided.
    """
    return P(None, "x", "y")


def rho_from_wfns(psi_G, occ, kweights, *, mesh: Mesh, box_index,
                  fft_grid, cell_volume: float, spin_degeneracy: float,
                  U=None):
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
        ``U[k, n, m]`` takes DFT band m into rotated band n.  ``None``
        builds ρ from ``psi_G`` unrotated, which is the DFT density and
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

    SYMMETRISATION IS THE CALLER'S, AND THAT IS A KNOWN ROUGH EDGE.  Handed
    IBZ weights this returns the weighted IBZ sum, which is not the density
    until symmetrised over the star (``centroid.orbit_syms``).  Leaving a
    correctness step to the caller is the inverse of the SlabIO padding
    ruling and should be closed once the star machinery of
    ``docs/dev/ibz_self_consistency_scaffold.md`` exists; until then the
    electron-count check is what catches a caller who forgets, because an
    unsymmetrised IBZ ρ still integrates to the right number.
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

    w_j = jnp.asarray(kweights, dtype=jnp.float64)
    bidx_j = jnp.asarray(box_index, dtype=jnp.int32)

    band_xy = NamedSharding(mesh, _band_spec())
    m_on_y = NamedSharding(mesh, P(None, "y", None, None))
    U_sh = NamedSharding(mesh, band_rotation_spec())
    box_spec = P(None, ("x", "y"), None, None, None, None)
    ifftn = make_sharded_ifftn_3d(mesh, box_spec, box_spec,
                                  norm="ortho", axes=(-3, -2, -1))
    rho_sharding = NamedSharding(mesh, P(None, None, None))

    def build():
        @jax.jit
        def fn(psi_, U_, occ_, w_, bidx_):
            psi_xy = jax.lax.with_sharding_constraint(psi_, band_xy)
            if have_U:
                # m on 'y' so the contraction reduces along 'y' alone.
                psi_my = jax.lax.with_sharding_constraint(psi_, m_on_y)
                U_x = jax.lax.with_sharding_constraint(U_, U_sh)
            else:
                psi_my = psi_xy
                U_x = U_

            def body(rho, xs):
                psi_xy_k, psi_my_k, U_k, occ_k, w_k, bidx_k = xs
                if have_U:
                    # n from U's 'x', m summed along 'y'.
                    psi_t = jnp.einsum('nm,msg->nsg', U_k, psi_my_k,
                                       optimize=True)
                    # Back onto the WHOLE mesh: straight out of the
                    # rotation the bands sit on 'x' and are replicated on
                    # 'y', which would make the FFT px-fold redundant and
                    # the final reduction a different rule than the
                    # unrotated path's.  One cheap sphere reshard fixes
                    # both.
                    psi_t = jax.lax.with_sharding_constraint(
                        psi_t[None], NamedSharding(mesh, _band_spec()))
                else:
                    psi_t = psi_xy_k[None]
                box = _box_kernel(psi_t, bidx_k[None], ngkmax=ngkmax)
                psi_r = ifftn(box) * scale
                dens = jnp.einsum('n,knsxyz->xyz', occ_k,
                                  jnp.abs(psi_r) ** 2, optimize=True)
                return rho + (w_k * f_spin) * dens, None

            rho0 = jnp.zeros(grid, dtype=jnp.float64)
            U_xs = U_x if have_U else jnp.zeros(
                (psi_.shape[0], 1, 1), dtype=jnp.complex128)
            rho, _ = jax.lax.scan(
                body, rho0, (psi_xy, psi_my, U_xs, occ_, w_, bidx_))
            return jax.lax.with_sharding_constraint(rho, rho_sharding)
        return fn

    fn = _cached_jit(
        "rho_from_wfns",
        (psi.shape, tuple(np.shape(U_j)), grid, float(cell_volume), f_spin,
         have_U, _sharding_key(psi)),
        build)
    return fn(psi, U_j, occ_j, w_j, bidx_j)


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
