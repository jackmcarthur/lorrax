"""The direct (Hartree) field of the occupied states, as band matrices.

    ρ(r) [, J(r)]  = Σ_k w_k f_spin Σ_n f_nk |ψ_nk(r)|² [, ψ† α ψ]
                     (``gw.qsgw_density.rho_from_wfns``; J is projected onto
                     the crystal symmetry once, there)
    V_H = v * ρ,  V_T = transverse field of J      (replicated FFT solves)
    ⟨mk|V_H [+ α·A]|nk⟩                             (one star-wedge k-scan,
                     ``common.mtxel_sweep``; full BZ by the star broadcast
                     of ``file_io.kin_ion``)

:func:`direct_field_matrices` is called live by the GW Hamiltonian
(``gw.sigma_dispatch``) and by ``psp.get_DFT_mtxels``; ``gw.kin_ion_io``
re-exports it as ``compute_hartree_matrix``.  The SC rebuild
(``gw.sc_iteration.rebuild_hartree_dft_basis``) is a second copy of this
chain and moves here in ARCH wave 3.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

import common.timing as timing
from common.collectives import process_rank_world, resolve_mesh
from file_io.kin_ion import broadcast_star_wedge

__all__ = [
    "ExactHartreeMatrices",
    "direct_field_matrices",
    "star_wedge_kspec",
    "wedge_density_occupations",
]


def wedge_density_occupations(wfn, sym, k_spec, nb_carrier: int,
                                band_stop: int):
    """``(occ, kweights)`` for the density on the sweep's own k rows.

    ``occ`` is ``(n_k, nb_carrier)``: the canonical physical occupations
    (unit below ``band_stop`` for an exact insulator) and exact zeros above.
    On the star wedge each row weighs its star's share of the full zone,
    ``|star| / nk_full`` -- the same partition ``star_wedge_rows`` hands the
    matrix sweep -- and the density scan's star average then makes the
    weighted wedge sum the crystal's density.  A full-BZ k-set is uniform.
    """
    stop = int(band_stop)
    if k_spec == "full_bz":
        nk = int(sym.nk_tot)
        weights = np.full(nk, 1.0 / nk)
        occ_rows = wfn.physical_density_occupations(
            k="full_bz", unit_as_none=True)
    else:
        from symmetry_maps import star_wedge_rows
        rows, irr_idx_wedge = star_wedge_rows(sym)
        nk = int(rows.size)
        counts = np.bincount(np.asarray(irr_idx_wedge, dtype=np.int64),
                             minlength=nk).astype(np.float64)
        weights = counts / counts.sum()
        occ_rows = wfn.physical_density_occupations(
            k="file", unit_as_none=True)
        if occ_rows is not None:
            occ_rows = np.asarray(occ_rows)[np.asarray(rows)]
    occ = np.zeros((nk, int(nb_carrier)), dtype=np.float64)
    occ[:, :stop] = 1.0 if occ_rows is None else np.asarray(
        occ_rows, dtype=np.float64)[:, :stop]
    return occ, weights


def star_wedge_kspec(wfn, sym):
    """``(k_spec, kvecs, n_k)`` for a matrix-element sweep over the wedge.

    The loader k-spec that selects one WFN row per symmetry orbit, that
    row order's k-vectors, and the trip count.  Returned as ``"ibz"``
    unchanged when the WFN's k-set already IS the star wedge, so every
    deck cut at a true IBZ takes byte-for-byte the path it always took —
    same loader cache key, same read, same scan.
    """
    from symmetry_maps import star_wedge_rows
    from wfn_loader import IBZRows

    if sym.parent_k_domain == "full_bz":
        return "full_bz", wfn.kvecs(k="full_bz"), int(sym.nk_tot)
    rows, _ = star_wedge_rows(sym)
    kpts = np.asarray(wfn.kpoints, dtype=np.float64)
    n_red = int(wfn.nkpts)
    if int(rows.size) == n_red and np.array_equal(rows, np.arange(n_red)):
        return "ibz", kpts, n_red
    return (IBZRows(tuple(int(r) for r in rows)),
            kpts[rows], int(rows.size))


# It is mesh-aware from 1×1 up to a production mesh.
#
# The plumbing itself — mesh, work partition, psum, gather, the pipelined
# per-k sweep — is ``common.collectives``; nothing here reaches under it.
#
# COMMUNICATION CONTRACT (this is the whole design, in four lines):
#
#   ψ                 ONE G-sphere load on the star wedge, band-sharded and
#                     resident (≈19 MB/rank at b600/P=64); it serves both
#                     the density and the matrix elements.
#   ρ(r)              the SC loop's density scan
#                     (``gw.qsgw_density.rho_from_wfns``): each rank sums
#                     its own bands, ONE reduction of the nx·ny·nz field,
#                     then the star average.  The one-shot and the SC
#                     Hartree have one density builder.
#   V_H(r) = Poisson  **REPLICATED BY DESIGN** — zero collectives; see the
#                     step-2 comment inside ``direct_field_matrices``.
#   ⟨mk|V_H|nk⟩       ONE k-tile scan (``common.mtxel_sweep``): k is a
#                     trip count; per tile ψ and V_H ψ are all-to-all'd to
#                     a G-split layout and the slab partial is
#                     reduce-scattered; the output stays sharded
#                     ``P(None,'x','y')`` and is gathered only at the
#                     boundary, by name.


class ExactHartreeMatrices(NamedTuple):
    """Exact charge/current direct matrices from one occupied-WFN sweep."""

    charge: object
    transverse: object
    current_g0_l2: float
    current_l2: float
    current_g0_relative: float
    current_symmetry_relative_movement: float
    current_symmetry_relative_residual: float
    current_symmetry_rotation_table_closure_defect: float
    current_symmetry_floating_point_residual_bound: float
    current_symmetry_relative_residual_tolerance: float
    current_symmetry_rows: int
    current_symmetry_antiunitary_rows: int
    tt_metric_sign: float


def direct_field_matrices(wfn, sym, meta, *, truncation_2d: bool,
                          nb: int, mesh=None,
                          include_transverse: bool = False,
                          charge_nspinor: int | None = None,
                          bispinor_lift: str = "raw",
                          print_fn=print,
                          return_sharded: bool = False):
    """The exact FFT-grid ⟨mk|V_H|nk⟩ for all k — **(nk, nb, nb) Ry**.

    Single distributed source for the live GW Hamiltonian and density-SC
    rebuild.

    Distribution: ONE resident ψ(G) sphere on the star wedge, band-sharded
    over every process, serves both halves.  ρ is the SC loop's own density
    scan (:func:`gw.qsgw_density.rho_from_wfns`: the wedge-weighted sum,
    star-averaged by the FFT-grid pullback, one reduction) — one density
    builder for the one-shot and the self-consistent Hartree; the Poisson
    solve is replicated; ⟨mk|V_H|nk⟩ is ONE k-scan over the same sphere
    (``common.mtxel_sweep``).  ``mesh`` is the collectives' device mesh —
    pass the run's own (the driver does) or leave it None and one is
    derived, 1×1 on a single device.

    Needs no pseudopotentials: ρ comes from ψ, V_H from the Poisson
    solve, and the matrix element from the same normalisation chain
    (``psp.get_DFT_mtxels.local_potential_scalars``) V_loc takes — the
    two plans call it, so they agree by construction and differ only in
    the reassociation the sharding forces on the G sum.
    ``truncation_2d`` MUST be the run's own convention (deck
    ``sys_dim``); mixing it with V_loc's is a large systematic error
    inside a ~500 eV cancellation.

    By default returns the full-BZ matrix as a host ``numpy`` array replicated
    on every rank for diagnostic callers. Live GW consumers pass
    ``return_sharded=True`` and retain ``P(None,'x','y')`` through the
    full-BZ star broadcast. The SC rebuild calls
    :func:`common.mtxel_sweep.sweep_matrix_elements` directly because it
    already owns the real-space potential and its DFT-basis contraction.
    """
    from psp.dft_operators import padded_gvectors
    from psp.get_DFT_mtxels import (build_hartree_potential,
                                    spin_degeneracy_factor)

    mesh = resolve_mesh(mesh)
    _, world = process_rank_world()
    nocc = int(wfn.nelec)
    exact_unit_occupations = bool(wfn.occupations_are_exact_integer)
    density_band_stop = int(wfn.physical_density_band_stop)
    nk = int(sym.nk_tot)
    with_transverse = bool(include_transverse)
    charge_ns = (int(meta.nspinor) if charge_nspinor is None
                 else int(charge_nspinor))
    if not 0 < charge_ns <= int(meta.nspinor):
        raise ValueError(
            "charge_nspinor must select a nonempty leading spinor block; "
            f"got {charge_ns} for meta.nspinor={int(meta.nspinor)}")
    if with_transverse and int(meta.nspinor) != 4:
        raise ValueError(
            "transverse direct Hartree requires the canonical four-component "
            f"kinetic-balance carrier; meta.nspinor={int(meta.nspinor)}")
    if exact_unit_occupations and nocc > nb:
        raise ValueError(
            f"V_H needs the {nocc} occupied bands but only {nb} were requested")

    # ---- 1. ρ(r): the SC density scan on the resident wedge sphere -------
    # ONE ψ(G) load serves the density AND the matrix sweep (step 3): the
    # star wedge, band-sharded, at the extent both need.  The wedge sum is
    # weighted by star size and star-averaged by the FFT-grid pullback,
    # exactly the SC loop's map-0 density (sc_iteration.
    # rebuild_hartree_dft_basis) at U = 1.
    from common.wfn_layout import band_sphere_spec
    from gw.qsgw_density import rho_from_wfns
    k_spec, _, nk_irr = star_wedge_kspec(wfn, sym)
    nb_load = max(int(nb), density_band_stop)
    psi_G = wfn.load(
        bands=(0, nb_load), k=k_spec, sharding=band_sphere_spec(),
        bispinor=(int(meta.nspinor) == 4),
        bispinor_lift=bispinor_lift)
    box_index = wfn.box_index(k=k_spec)
    density_label = ("occupied bands" if exact_unit_occupations else
                     "WFN bands with canonical fractional occupations")
    print_fn(f"\nBuilding valence density from {density_band_stop} "
             f"{density_label} (P={world}, {nk_irr} star-wedge k of {nk})...")
    f_spin = spin_degeneracy_factor(wfn)
    occ, kweights = wedge_density_occupations(
        wfn, sym, k_spec, nb_load, density_band_stop)
    grid = tuple(int(s) for s in meta.fft_grid)
    # The Dirac current is projected onto the crystal symmetry ONCE, inside
    # the density scan; its receipt (movement of the raw field, residual of
    # the projected one) is the one reported below.
    projections = []
    with timing.section("vh_rho"):
        rho_np = np.asarray(rho_from_wfns(
            psi_G, occ, kweights, mesh=mesh, box_index=box_index,
            fft_grid=grid, cell_volume=float(wfn.cell_volume),
            spin_degeneracy=f_spin,
            include_dirac_current=with_transverse,
            charge_nspinor=(None if charge_nspinor is None else charge_ns),
            sym=sym,
            sym_perm=sym.fft_grid_pullback(sym.active_symmetry_rows, grid),
            projection_receipt_fn=projections.append,
            print_fn=print_fn), dtype=np.float64)

    if with_transverse:
        fields = np.asarray(rho_np, dtype=np.float64)
        rho_np = fields[0]
        current_np = fields[1:]
        ngrid = int(np.prod(current_np.shape[-3:]))
        current_g0 = np.sum(current_np, axis=(-3, -2, -1)) / np.sqrt(ngrid)
        current_g0_l2 = float(np.linalg.norm(current_g0))
        current_l2 = float(np.linalg.norm(current_np))
        current_g0_relative = current_g0_l2 / max(
            current_l2, np.finfo(np.float64).tiny)
        print_fn(
            "    Dirac-current G=0 diagnostic (J=j/c; gauged V_NL current "
            "is absent): "
            f"||J0||={current_g0_l2:.6e}, ||J||={current_l2:.6e}, "
            f"ratio={current_g0_relative:.6e}; periodic TT sets G=0 to zero")
        if len(projections) != 1:
            raise RuntimeError(
                "direct_field_matrices: the density scan projected the Dirac "
                f"current {len(projections)} times; want exactly one "
                "(gw.qsgw_density.rho_from_wfns with sym)")
        current_projection = projections[0]
        print_fn(
            "    Dirac-current symmetry projection: "
            f"rows={current_projection.n_symmetry_rows} "
            f"(antiunitary={current_projection.n_antiunitary_rows}), "
            f"movement={current_projection.relative_movement:.6e}, "
            f"residual={current_projection.relative_residual:.6e} <= "
            f"{current_projection.relative_residual_tolerance:.6e} "
            f"(delta_R="
            f"{current_projection.rotation_table_closure_defect:.6e} + "
            f"floating="
            f"{current_projection.floating_point_residual_bound:.6e}); "
            "alpha.A matrix remains on the authenticated star wedge")
        from vcoul import COULOMB_GAUGE_TT_SIGN
        from psp.dft_operators import transverse_potential_from_current
        tt_metric_sign = float(COULOMB_GAUGE_TT_SIGN)
        with timing.section("vh_transverse_field"):
            V_T_r = transverse_potential_from_current(
                jnp.asarray(current_np, dtype=jnp.float64),
                jnp.asarray(wfn.bdot, dtype=jnp.float64),
                jnp.asarray(wfn.bvec, dtype=jnp.float64),
                float(wfn.blat), bool(truncation_2d),
                tt_metric_sign=tt_metric_sign)
            V_T_r = jnp.asarray(np.asarray(V_T_r, dtype=np.float64),
                                dtype=jnp.float64)
        del fields, current_np, current_g0
    else:
        V_T_r = None
        current_g0_l2 = current_l2 = current_g0_relative = 0.0
        current_projection = None

    # ---- 2. Poisson: REPLICATED BY DESIGN ------------------------------
    # Two 3-D FFTs on a 1.4 MB array: 3.1e7 flop against the sweep's
    # 1.2e12, i.e. 3e-5 of the work.  Sharding it would replace a free
    # duplicated computation with an all-to-all (a distributed 3-D FFT
    # is two transposes) and buy back 1.4 MB of memory per rank.  Every
    # rank therefore solves the same Poisson equation from the same
    # replicated ρ and gets bit-identical V_H(r) — which is also what
    # makes the k-partitioned matrix-element sweep below trivially
    # rank-invariant.  Revisit only above N_r ≈ 1e8.
    expected_electrons = (f_spin * float(nocc) if exact_unit_occupations
                          else float(wfn.num_electrons))
    V_H_r = build_hartree_potential(
        jnp.asarray(rho_np), wfn,
        truncation_2d=bool(truncation_2d),
        expected_electrons=expected_electrons,
        print_fn=print_fn,
    )
    # V_H(r) is replicated (step 2), so it is closed over by the operator
    # as a constant; nothing about it is per-k.
    V_H_r = jnp.asarray(np.asarray(V_H_r, dtype=np.float64),
                        dtype=jnp.float64)
    del rho_np

    # ---- 3. ⟨mk|V_H|nk⟩: ONE k-scan over the STAR WEDGE ----
    #
    # THE k-SET IS THE STAR WEDGE, and the full-BZ table is the star
    # broadcast of it (docs/architecture/symmetry_register.md §8, including
    # the conjugation the time-reversed rows need).
    # V_H is a local scalar potential, so it commutes with the space group
    # and with time reversal like any other term here.
    #
    # ρ ABOVE IS UNTOUCHED and stays a full-BZ sum: it is a sum over the
    # zone, not a per-k operator, and rebuilding it from the IBZ with
    # weights is a different (unsymmetrised) quadrature.  V_H is then a
    # local scalar potential on the FFT grid like V_loc, so it takes the
    # same symmetry argument as the kin+ion sweep and no other.
    #
    # ``tests/multi_device/mtxel_callsite_gate.py`` check 5 compares this
    # function against a full-BZ per-k local plan at 1e-12 relative and is
    # the one gate in the tree that would notice if it did not.
    #
    # Replaces the k-partitioned ``gather_k_blocks`` sweep.  That sweep
    # built each k's FULL-BAND FFT box on one rank (1.77 GB at b600
    # bispinor) and could not use more than ``nk`` ranks at all — each
    # rank took a whole k, so its wall was one full-band k however large
    # P was.  ``sweep_matrix_elements`` scans k tiles with every rank
    # holding a G slab of every band, so ``nk`` is a trip count and never a
    # parallel axis (its module docstring owns the plan and its costs).
    #
    # Fixed-shape G (owner decision D10, 2026-07-30): ``padded_gvectors``
    # hands over the loader's OWN ``(nk, ngkmax, 3)`` table, so the scan
    # body lowers ONCE for the whole k range.  The pad columns are made
    # inert by ``g_mask`` — mandatory, not tidy: pad rows hold ``(0,0,0)``,
    # a valid box index that ALIASES physical Γ, so a forgotten mask is
    # silently wrong inside H₀'s ~500 eV cancellation rather than loud.
    #
    # ψ arrives on the G-SPHERE, band-sharded, resident for all k: 1.2 GB
    # globally at b600 but ≈19 MB/rank at P=64, against the 1.77 GB ONE
    # box the route above materialised per k.  ``band_sphere_spec`` is the
    # single definition of that layout, shared with the loader, so no
    # reshard is inserted between the read and the scan.
    from common.mtxel_sweep import (SweepGeometry, blocks_to_host,
                                    four_current_potential_operator,
                                    local_potential_operator,
                                    sweep_matrix_elements)
    from common.wfn_layout import band_sphere_spec
    gtab = padded_gvectors(wfn, k=k_spec)
    # The four-current operator takes its charge block by charge_nspinor
    # from psi_G itself, so a spinor slice is needed only by the scalar
    # sweep; in transverse mode it was a dead second copy of the sphere.
    psi_charge = psi_G if (with_transverse
                           or charge_ns == int(psi_G.shape[2])) \
        else psi_G[:, :, :charge_ns, :]
    geom_matrix = SweepGeometry(
        mesh=mesh, fft_grid=meta.fft_grid,
        ngkmax=int(psi_G.shape[3]), nb=nb_load,
        ns=(int(psi_G.shape[2]) if with_transverse
            else int(psi_charge.shape[2])),
        nk=nk_irr, cell_volume=float(wfn.cell_volume))
    matrix_label = ("⟨mk|V_H|nk⟩ + <m|sum_i alpha_i A_i|n>"
                    if with_transverse else "⟨mk|V_H|nk⟩")
    print_fn(f"\n{matrix_label}: one k-scan over {nk_irr} STAR-WEDGE "
             f"k-points (broadcast to {nk} full-BZ k), "
             f"{geom_matrix.nb} bands sharded over P={world}; "
             f"charge nspinor={charge_ns}...")
    matrix_psi = psi_G if with_transverse else psi_charge
    matrix_operator = (
        four_current_potential_operator(
            geom_matrix, V_H_r, V_T_r, charge_nspinor=charge_ns)
        if with_transverse else local_potential_operator(geom_matrix, V_H_r))
    with timing.section("vh_matrix"):
        H_matrix = sweep_matrix_elements(
            matrix_psi, operator=matrix_operator, geom=geom_matrix,
            gvecs=gtab.gvecs, gmask=gtab.mask,
            box_index=box_index, kvecs=gtab.kvecs)
        if with_transverse:
            H_vh, H_vt = H_matrix[:, 0], H_matrix[:, 1]
        else:
            H_vh = H_matrix
        del H_matrix
        # Live GW keeps the matrix at P(None,'x','y') and star-broadcasts on
        # device. Diagnostic callers explicitly request the host boundary.
        if return_sharded:
            H_charge = broadcast_star_wedge(H_vh[:, :nb, :nb], sym)
        else:
            H_charge = broadcast_star_wedge(
                blocks_to_host(H_vh, nb=nb, owner_only=False), sym)
    del H_vh
    if not with_transverse:
        del psi_G, psi_charge
        return H_charge

    with timing.section("vh_transverse_matrix_boundary"):
        if return_sharded:
            H_transverse = broadcast_star_wedge(
                H_vt[:, :nb, :nb], sym)
        else:
            H_transverse = broadcast_star_wedge(
                blocks_to_host(H_vt, nb=nb, owner_only=False), sym)
    del H_vt, psi_G, psi_charge, V_T_r
    return ExactHartreeMatrices(
        charge=H_charge,
        transverse=H_transverse,
        current_g0_l2=current_g0_l2,
        current_l2=current_l2,
        current_g0_relative=current_g0_relative,
        current_symmetry_relative_movement=(
            current_projection.relative_movement),
        current_symmetry_relative_residual=(
            current_projection.relative_residual),
        current_symmetry_rotation_table_closure_defect=(
            current_projection.rotation_table_closure_defect),
        current_symmetry_floating_point_residual_bound=(
            current_projection.floating_point_residual_bound),
        current_symmetry_relative_residual_tolerance=(
            current_projection.relative_residual_tolerance),
        current_symmetry_rows=current_projection.n_symmetry_rows,
        current_symmetry_antiunitary_rows=(
            current_projection.n_antiunitary_rows),
        tt_metric_sign=tt_metric_sign,
    )
