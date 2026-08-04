"""ρ(r) and ρ(G) from rotated orbitals with per-state occupations.

The piece that turns the fixed-density QSGW loop into a density-updating
one.  ``sc_iteration`` today rotates V_H as a BASIS CHANGE (U†V_H U) and
leaves ρ at its DFT value; this rebuilds ρ from the rotated orbitals, so
the Hartree potential of iteration n+1 comes from iteration n's density
rather than from the DFT one.

STRUCTURE, AND WHY IT IS IN THIS ORDER
--------------------------------------
Two stages, deliberately separate:

1. :func:`rotate_bands` — ψ'_nk(G) = Σ_m U_nm(k) ψ_mk(G).  ONE sharded
   einsum over the whole (k, band) block, done ONCE, on the G-SPHERE.
2. :func:`rho_from_wfns` — a ``lax.scan`` over k that boxes, transforms
   and accumulates ``Σ_n f_nk |ψ'_nk(r)|²`` into a carried ρ(r).

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


__all__ = ["rotate_bands", "rho_from_wfns", "rho_r_to_G"]


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


def rotate_bands(psi_G, U, *, mesh: Mesh):
    """ψ'_nk(G) = Σ_m U_nm(k) ψ_mk(G) — one sharded einsum, on the sphere.

    Parameters
    ----------
    psi_G : (n_k, nb, ns, ngkmax) c128, band-sharded
    U : (n_k, nb, nb) complex
        ``U[k, n, m]`` takes DFT band ``m`` into rotated band ``n`` — the
        eigenvector matrix ``sc_iteration`` gets from ``eigh(H_qp_dft)``,
        in that index order.  Its rows are the new orbitals.
    mesh : Mesh

    Returns
    -------
    (n_k, nb, ns, ngkmax) c128, band-sharded on the OUTPUT band axis ``n``.

    The contraction runs over ``m``, which is the sharded axis, so a
    collective is unavoidable — ``with_sharding_constraint`` on the output
    lets XLA choose it (a reduce-scatter) rather than hand-rolling one.
    That is the same delegation ``mtxel_sweep`` makes for its per-k
    reshard, and for the same reason: XLA sees the whole program and can
    place the collective; a hand-rolled ``psum`` cannot be moved.

    ``U`` is REPLICATED.  It is ``(n_k, nb, nb)`` — 105 MB at nb=640 /
    nk=16, but 9.2 GB at nb=2000 / nk=144, i.e. the W2-class object.  That
    residency is inherited from the self-consistency loop, which already
    carries ``H_qp_dft`` at that shape, so it is not introduced here; it
    is the reason the loop as a whole does not yet scale to a 12×12 deck.
    Recorded rather than hidden.
    """
    psi = jnp.asarray(psi_G, dtype=jnp.complex128)
    U_j = jnp.asarray(U, dtype=jnp.complex128)
    band_sharding = NamedSharding(mesh, _band_spec())

    def build():
        @jax.jit
        def fn(psi_, U_):
            out = jnp.einsum('knm,kmsg->knsg', U_, psi_, optimize=True)
            return jax.lax.with_sharding_constraint(out, band_sharding)
        return fn

    fn = _cached_jit('rotate_bands',
                     (psi.shape, U_j.shape, _sharding_key(psi)), build)
    return fn(psi, U_j)


def rho_from_wfns(psi_G, occ, kweights, *, mesh: Mesh, box_index,
                  fft_grid, cell_volume: float, spin_degeneracy: float):
    """ρ(r) = Σ_k w_k f_spin Σ_{n,s} f_nk |ψ_nks(r)|², scanned over k.

    Parameters
    ----------
    psi_G : (n_k, nb, ns, ngkmax) c128, band-sharded
        Already rotated if a rotation is wanted — this routine does not
        care whether the orbitals are DFT or QP.
    occ : (n_k, nb) float64
        Per-state occupations, ``gw.efermi.step_occupations`` today and a
        Fermi–Dirac factor later.  A WEIGHT, not a mask: nothing here
        assumes it is 0 or 1.
    kweights : (n_k,) float64
        Weights of the SAME k-set as ``psi_G``.  On the IBZ these are
        ``WfnLoader.kweights`` and the result MUST be symmetrised over the
        star afterwards (``centroid.orbit_syms``) — a weighted IBZ sum is
        not the full-BZ density until it is.
    box_index : (n_k, nx, ny, nz) int32
        Sphere→box map, ``WfnLoader.box_index``.
    cell_volume, spin_degeneracy
        ``psp.get_DFT_mtxels.spin_degeneracy_factor`` supplies the latter.

    Returns
    -------
    (nx, ny, nz) float64, replicated.

    NORMALISATION is term-for-term ``psp.get_DFT_mtxels.
    valence_density_from_kpoint``: ψ_r = ifftn(box, 'ortho')·√(N/Ω), so
    ``ΔV · Σ_r ρ = f_spin · Σ_k w_k Σ_n f_nk``.  With step occupations and
    an E_F from ``gw.efermi`` that is ``f_spin · n_occ_bands``, which is
    the electron count — the gate checks exactly this.

    THE CARRY IS ρ(r).  ``(nx, ny, nz)`` f64 is 750 kB at a 60×60×26 grid,
    so the scan carries something negligible and there is ONE collective
    for the whole build (the final psum), not one per k.  This is the
    reason the k loop is a scan and not a partition: a k-partitioned build
    would need a psum of ρ per rank-group anyway, and could not use more
    than n_k ranks.
    """
    from common.fft_helpers import make_sharded_ifftn_3d

    grid = tuple(int(s) for s in fft_grid)
    ngrid = int(np.prod(grid))
    scale = float(np.sqrt(ngrid / float(cell_volume)))
    f_spin = float(spin_degeneracy)

    psi = jnp.asarray(psi_G, dtype=jnp.complex128)
    p_prod = spec_divisor(mesh, _band_spec(), 1)
    psi, _ = pad_axis_to(psi, p_prod, axis=1)
    nb_pad = int(psi.shape[1])

    occ_j = jnp.asarray(occ, dtype=jnp.float64)
    if int(occ_j.shape[1]) != nb_pad:
        # Pad bands carry ψ = 0, so their occupation is irrelevant to the
        # arithmetic — but the scan needs the shapes to line up, and a
        # zero is the value that stays correct if the ψ pad ever changes.
        occ_j = jnp.pad(occ_j, ((0, 0), (0, nb_pad - int(occ_j.shape[1]))))

    w_j = jnp.asarray(kweights, dtype=jnp.float64)
    bidx_j = jnp.asarray(box_index, dtype=jnp.int32)

    box_spec = P(None, ("x", "y"), None, None, None, None)
    ifftn = make_sharded_ifftn_3d(mesh, box_spec, box_spec,
                                  norm='ortho', axes=(-3, -2, -1))
    band_sharding = NamedSharding(mesh, _band_spec())
    rho_sharding = NamedSharding(mesh, P(None, None, None))

    def build():
        @jax.jit
        def fn(psi_, occ_, w_, bidx_):
            psi_x = jax.lax.with_sharding_constraint(psi_, band_sharding)

            def body(rho, xs):
                psi_k, occ_k, w_k, bidx_k = xs
                box = _box_kernel(psi_k[None], bidx_k[None], ngkmax=psi_.shape[3])
                psi_r = ifftn(box) * scale
                # |ψ|² summed over the SPINOR axis (both components of a
                # spinor are the same state) and over bands with the
                # occupation as the weight.  Bands are sharded, so this
                # sum is rank-local and the cross-rank reduction happens
                # once, on ρ, after the scan.
                dens = jnp.einsum(
                    'n,knsxyz->xyz',
                    occ_k, jnp.abs(psi_r) ** 2, optimize=True)
                return rho + (w_k * f_spin) * dens, None

            rho0 = jnp.zeros(grid, dtype=jnp.float64)
            rho, _ = jax.lax.scan(body, rho0, (psi_x, occ_, w_, bidx_))
            # THE one collective: bands were sharded, so each rank holds a
            # partial ρ over its own bands.  Constraining to replicated is
            # the all-reduce, and XLA places it.
            return jax.lax.with_sharding_constraint(rho, rho_sharding)
        return fn

    fn = _cached_jit(
        'rho_from_wfns',
        (psi.shape, grid, float(cell_volume), f_spin, _sharding_key(psi)),
        build)
    return fn(psi, occ_j, w_j, bidx_j)


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
