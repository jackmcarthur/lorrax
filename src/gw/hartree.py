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

The on-site Breit field (design note ``ONSITE_INCODE.md`` section 2) is
assembled here too, by one function for the three callers (the one-shot
Hamiltonian, the SC rebuild and ``gw.onsite_breit_cli``)::

    P[k, a, n]  = <phi~_a(k)|psi_nk>                     projection table, full BZ
    D[a, b]     = sum_k w_k sum_i f_ik (P U)_ai conj((P U)_bi)
    h           = h^dir[D] + h^x[D]                        (keys of section 3)
    H_OS[k]     = P[k]^dagger h P[k]                       (Ry, DFT band basis)

(:func:`onsite_breit_setup_for`, :func:`onsite_breit_field`; the physics
library is ``psp.onsite_breit``).
"""
from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

import common.timing as timing
from common.log import get_logger

log = get_logger(__name__)

from common.collectives import process_rank_world, resolve_mesh
from file_io.kin_ion import broadcast_star_wedge

__all__ = [
    "ExactHartreeMatrices",
    "OnsiteBreitField",
    "direct_field_matrices",
    "onsite_breit_field",
    "onsite_breit_setup_for",
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


def _book_density_price(*, nk_w: int, nb_load: int, ns: int, ngkmax: int,
                        charge_ns: int, with_transverse: bool, grid,
                        n_sym: int, ranks: int) -> float:
    """Book the stage's sphere and density phase to the open section, BEFORE
    ψ is loaded, so the increment's entry reading is what precedes the stage
    (``lxkit.memory``).  Returns the density transient in bytes.

    Spread over P: the ψ(G) wedge sphere, band-padded to the mesh (and its
    charge-spinor slice when the scalar sweep takes a copy).  Replicated: the
    fields and the star pullback table.  Transient: one density band chunk's
    boxes (:func:`gw.qsgw_density.density_band_chunk`), bounded by the tile.
    """
    from lxkit import memory
    from gw.qsgw_density import DENSITY_BOX_COPIES, density_band_chunk

    c128, n_grid = 16.0, float(np.prod(grid))
    p = max(1, int(ranks))
    rank_bands = -(-int(nb_load) // p)
    sphere = float(nk_w * rank_bands * p * ngkmax) * c128
    ns_sphere = ns + (charge_ns if not with_transverse and charge_ns != ns
                      else 0)
    n_fields = 4 if with_transverse else 1
    fields = 3.0 * n_fields * n_grid * 8.0 + float(n_sym) * n_grid * 4.0
    density = (DENSITY_BOX_COPIES * ns * n_grid * c128 * density_band_chunk(
        n_loc=rank_bands, k_tile=1, ns=ns, n_grid=int(n_grid)))
    memory.book(memory.Price(
        operands=memory.Term(fields, sphere * ns_sphere,
                             "psi(G) wedge sphere + fields + star pullback"),
        xla=memory.Term(density, 0.0, "density band-chunk boxes"),
        ranks=p, label="Sigma Hartree density", increment=True))
    return density


def _book_sweep_price(geom, operator, *, nk_full: int, nb: int,
                      n_blocks: int, density_bytes: float, ranks: int):
    """Book the matrix sweep's part: its output blocks (spread over P) and
    the part of its transient beyond the density phase's, so the stage's
    transient is the larger of the two
    (:func:`common.mtxel_sweep.sweep_transient_bytes`)."""
    from lxkit import memory
    from common.mtxel_sweep import sweep_transient_bytes

    blocks = (n_blocks * (geom.nk * float(geom.nb) ** 2
                          + nk_full * float(nb) ** 2) * 16.0)
    sweep = sweep_transient_bytes(geom, operator)
    memory.book(memory.Price(
        operands=memory.Term(0.0, blocks, "V_H wedge + full-BZ blocks"),
        xla=memory.Term(max(0.0, sweep - density_bytes), 0.0,
                        "sweep k step + band-chunk boxes"),
        ranks=max(1, int(ranks)), label="Sigma Hartree matrix sweep",
        increment=True))


def direct_field_matrices(wfn, sym, meta, *, truncation_2d: bool,
                          nb: int, mesh=None,
                          include_transverse: bool = False,
                          transverse_truncation_2d: bool | None = None,
                          charge_nspinor: int | None = None,
                          bispinor_lift: str = "raw",
                          print_fn=None,
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
    ``truncation_2d`` MUST match the DFT ionic operator's QE convention;
    mixing it with V_loc's is a large systematic error inside a ~500 eV
    cancellation. The beyond-DFT transverse field uses
    ``transverse_truncation_2d`` (the GW boundary); None retains the shared
    convention for standalone diagnostic callers.

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
    grid = tuple(int(s) for s in meta.fft_grid)
    gtab = padded_gvectors(wfn, k=k_spec)
    density_bytes = _book_density_price(
        nk_w=nk_irr, nb_load=nb_load, ns=int(meta.nspinor),
        ngkmax=int(gtab.gvecs.shape[1]), charge_ns=charge_ns,
        with_transverse=with_transverse, grid=grid,
        n_sym=len(sym.active_symmetry_rows), ranks=int(mesh.devices.size))
    psi_G = wfn.load(
        bands=(0, nb_load), k=k_spec, sharding=band_sphere_spec(),
        bispinor=(int(meta.nspinor) == 4),
        bispinor_lift=bispinor_lift)
    box_index = wfn.box_index(k=k_spec)
    density_label = ("occupied bands" if exact_unit_occupations else
                     "WFN bands with canonical fractional occupations")
    log.info(f"\nBuilding valence density from {density_band_stop} "
             f"{density_label} (P={world}, {nk_irr} star-wedge k of {nk})...")
    f_spin = spin_degeneracy_factor(wfn)
    occ, kweights = wedge_density_occupations(
        wfn, sym, k_spec, nb_load, density_band_stop)
    # The Dirac current is projected onto the crystal symmetry ONCE, inside
    # the density scan; its receipt (movement of the raw field, residual of
    # the projected one) is the one reported below.
    projections = []
    sym_perm = sym.fft_grid_pullback(sym.active_symmetry_rows, grid)
    with timing.section("vh_rho"):
        rho_np = np.asarray(rho_from_wfns(
            psi_G, occ, kweights, mesh=mesh, box_index=box_index,
            fft_grid=grid, cell_volume=float(wfn.cell_volume),
            spin_degeneracy=f_spin,
            include_dirac_current=with_transverse,
            charge_nspinor=(None if charge_nspinor is None else charge_ns),
            sym=sym,
            sym_perm=sym_perm,
            projection_receipt_fn=projections.append), dtype=np.float64)
    del sym_perm

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
        log.debug(
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
        log.debug(
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
                float(wfn.blat), bool(truncation_2d if transverse_truncation_2d is None
                                      else transverse_truncation_2d),
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
    log.debug(f"\n{matrix_label}: one k-scan over {nk_irr} STAR-WEDGE "
             f"k-points (broadcast to {nk} full-BZ k), "
             f"{geom_matrix.nb} bands sharded over P={world}; "
             f"charge nspinor={charge_ns}...")
    matrix_psi = psi_G if with_transverse else psi_charge
    matrix_operator = (
        four_current_potential_operator(
            geom_matrix, V_H_r, V_T_r, charge_nspinor=charge_ns)
        if with_transverse else local_potential_operator(geom_matrix, V_H_r))
    _book_sweep_price(
        geom_matrix, matrix_operator, nk_full=nk, nb=nb,
        n_blocks=2 if with_transverse else 1, density_bytes=density_bytes,
        ranks=int(mesh.devices.size))
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


# ===========================================================================
#  On-site Breit field (ONSITE_INCODE.md sections 2-4)
# ===========================================================================
#
# ONE ASSEMBLY FOR THREE CALLERS.  The one-shot Hamiltonian
# (``gw.sigma_dispatch``), the SC rebuild (``gw.sc_iteration``) and the
# DFT-level driver (``gw.onsite_breit_cli``) call :func:`onsite_breit_field`;
# the physics (sidecar, kernels, projector, moments) is ``psp.onsite_breit``.
#
# WHY THE FULL BZ.  P is computed from the two-component DFT psi(G) the loader
# unfolds (``WfnLoader.load(k="full_bz")``: rotation, tau phase, spinor
# rotation and the time-reversal conjugation of every full-BZ row, by the
# symmetry service), so D is the plain zone sum and carries the magnetic
# group because the unfolded states do.  No second symmetry action exists.
# Row k of every array here is row k of ``sym.unfolded_kpts``, the order of
# the SC loop's full-BZ U and occupations (``rebuild_hartree_dft_basis``
# takes ``U_qp[sym.kirr_fullids]`` from the same table); the loader's own
# ``kvecs(k="full_bz")`` is compared with it before P is built.

class OnsiteBreitField(NamedTuple):
    """The on-site Breit field of one density matrix (:func:`onsite_breit_field`).

    ``matrix`` (nk_full, nb, nb) complex128, Ry: H_OS in the DFT band basis of
    the window, full BZ in ``sym.unfolded_kpts`` order, replicated
    ``P(None, None, None)``.  ``density_matrix`` and ``h_block`` (R_os, R_os)
    host complex128 in the shell basis (``h`` in Ry; only the shell-atom
    diagonal blocks are nonzero in ``h``).  ``pieces``: per shell-atom dicts
    ``dir_ae, dir_ps, x_ae, x_ps, total`` (Ry; a ``*_ps`` is ``None`` without
    a ``ps`` tensor) of ``psp.onsite_breit.onsite_potential``.
    ``projections`` (nk_full, R_os, nb) complex128, replicated: P.
    ``setup``: the ``psp.onsite_breit.OnsiteSetup``.  ``report``: the log lines.
    """

    matrix: object
    density_matrix: object
    h_block: object
    pieces: list
    projections: object
    setup: object
    report: list


#: (WFN fingerprint, sidecar (path, sha256) pairs, pseudo dir, projector,
#: double counting, exchange) -> OnsiteSetup.  By value, so every SC map of
#: a run (a fresh call, the same inputs) reuses one setup and one compiled
#: projection program (``psp.onsite_breit`` keys its kernels by setup).
_ONSITE_SETUPS: dict = {}
#: (WFN fingerprint, global band range, setup provenance, mesh) -> the
#: projection table P (nk_full, R_os, nb), replicated.  psi_DFT and the
#: projector are fixed for a run (only U and f move), so P is one pass.
_ONSITE_PROJECTIONS: dict = {}

#: Hermiticity budget of D and H_OS, relative to their largest entry.
ONSITE_HERMITICITY_RTOL = 1.0e-10
#: Slack on the eigenvalue range of each shell block of D: with orthonormal
#: projector rows and orthonormal states, D's eigenvalues lie inside the
#: occupation range [min(0, f), max(1, f)].
ONSITE_DENSITY_EIGEN_TOL = 1.0e-6


def _onsite_gate(rule, got, want, why, fix):
    return ValueError(
        f"GATE {rule}: {why.split(';')[0]}\n  got:  {got}\n  want: {want}\n"
        f"  why:  {why}\n  fix:  {fix}\n  doc:  src/gw/hartree.py (onsite_breit_field); "
        "ONSITE_INCODE.md")


def _file_sha256(path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _onsite_upf_paths(pseudo_dir: str) -> dict:
    """``{element: UPF path}`` of the pseudo dir, refusing an ambiguous one.

    ``psp.pseudos.load_pseudopotentials`` keys by element, so two UPFs of one
    element (a Gaunt-stamped and a plain one, say) would leave the projector
    to whichever sorts last; that refuses here instead.
    """
    import glob
    from psp.pseudos import load_pseudopotentials
    pseudos = load_pseudopotentials(pseudo_dir)
    files = (glob.glob(os.path.join(pseudo_dir, "*.upf"))
             or glob.glob(os.path.join(pseudo_dir, "*.UPF")))
    if not pseudos or len(files) != len(pseudos):
        raise _onsite_gate(
            "onsite_breit_projector",
            f"onsite_breit_pseudo_dir = {pseudo_dir}: {len(files)} UPF files, "
            f"{len(pseudos)} species ({sorted(pseudos)})",
            "exactly one readable *.upf per species of the WFN",
            "the ortho-atomic projector is built from the PP_PSWFC of every species; an "
            "absent or ambiguous UPF is another projector",
            "point onsite_breit_pseudo_dir at a directory with one UPF per species")
    return {el: str(p._source_path) for el, p in pseudos.items()}


def onsite_breit_setup_for(config, wfn, *, input_dir, tt_exchange: bool):
    """The on-site Breit setup of this run, or ``None`` when ``onsite_breit = false``.

    Parameters
    ----------
    config : gw.gw_config.LorraxConfig
        Reads ``onsite_breit``, ``onsite_breit_files``,
        ``onsite_breit_double_counting``, ``onsite_breit_exchange``,
        ``onsite_breit_projector``, ``onsite_breit_pseudo_dir`` and ``bispinor``.
    wfn : wfn_loader.WfnLoader
    input_dir : str
        The deck directory; sidecar files and the pseudo dir are relative to
        it (the pseudo dir defaults to it).
    tt_exchange : bool
        Whether this run's Sigma carries the bare transverse (TT) exchange
        on pseudo-orbitals (``exchange = paw`` needs it, ``full`` must not
        have it).  ``False`` at the DFT level.

    Returns
    -------
    psp.onsite_breit.OnsiteSetup or None
        Built once per (WFN, sidecar contents, pseudo dir, keys): a module
        cache keyed by value, so the SC maps share it.  Every species of the
        WFN needs a UPF in the pseudo dir (loaded with
        ``psp.pseudos.load_pseudopotentials``); sidecars are read with
        ``require_ps`` when either key is ``paw``.  Refusals: the keys
        (``GATE onsite_breit_keys``, through
        ``psp.onsite_breit.resolve_onsite_keys``), a missing sidecar
        (``GATE onsite_breit_sidecar``), an absent or ambiguous UPF
        (``GATE onsite_breit_projector``), and every refusal of
        ``psp.onsite_breit.build_onsite_setup``.
    """
    if not bool(getattr(config, "onsite_breit", False)):
        return None
    from common.parallel_transport import wfn_fingerprint
    from psp.onsite_breit import build_onsite_setup, read_sidecar, resolve_onsite_keys

    dc, ex = resolve_onsite_keys(
        config.onsite_breit_double_counting, config.onsite_breit_exchange,
        bispinor=bool(config.bispinor), tt_exchange=bool(tt_exchange))
    base = os.path.abspath(str(input_dir))
    files = tuple(os.path.normpath(os.path.join(base, f))
                  for f in str(config.onsite_breit_files).split())
    absent = [f for f in files if not os.path.isfile(f)]
    if not files or absent:
        raise _onsite_gate(
            "onsite_breit_sidecar", f"onsite_breit_files = {config.onsite_breit_files!r} "
            f"(missing: {absent})", "one existing <El>.onsite_breit.json per species, deck-relative",
            "the radial integrals of the operator come only from the sidecar files",
            "fix onsite_breit_files")
    pseudo_dir = os.path.normpath(os.path.join(
        base, str(config.onsite_breit_pseudo_dir or "").strip() or "."))
    projector = str(config.onsite_breit_projector or "").strip()
    key = (wfn_fingerprint(wfn), tuple((f, _file_sha256(f)) for f in files),
           pseudo_dir, projector, dc, ex)
    setup = _ONSITE_SETUPS.get(key)
    if setup is not None:
        return setup
    with timing.section("onsite_setup"):
        upf_paths = _onsite_upf_paths(pseudo_dir)
        need_ps = dc == "paw" or ex == "paw"
        sidecars = [read_sidecar(f, require_ps=need_ps) for f in files]
        setup = build_onsite_setup(wfn, upf_paths, sidecars, projector=projector,
                                   double_counting=dc, exchange=ex)
    _ONSITE_SETUPS[key] = setup
    log.info(f"onsite_breit setup: projector {setup.projector}, double_counting = {dc}, "
             f"exchange = {ex}, {len(setup.shell_atoms)} shell-atoms, R_os = {setup.R_os}, "
             f"UPFs from {pseudo_dir}")
    for sc in sidecars:
        log.info(f"  sidecar {sc['element']}: {sc['path']} sha256 {sc['sha256']}")
    for sa in setup.shell_atoms:
        log.info(f"  shell-atom atom {sa['atom']} {sa['element']} {sa['label']} (l={sa['l']}): "
                 f"rows [{sa['start']}, {sa['stop']}) of the on-site axis")
    if ex == "paw":
        log.info("  onsite_breit_exchange = paw assumes the on-site part of Sigma's TT exchange "
                 "equals h^x_PS: NOT VALIDATED (ONSITE_INCODE.md section 3)")
    log.info("  not included: i[r, H_OS] in the velocity (dipole, head, orbital moment); "
             "on-site spin-spin and orbit-orbit; exchange with other shells (ONSITE_INCODE.md "
             "section 8)")
    return setup


def _book_onsite_price(*, nk: int, nb_carrier: int, ngkmax: int, r_os: int,
                       ranks: int) -> None:
    """Book the projection pass to the open section before psi is loaded.

    Spread over P: the full-BZ two-component sphere ``(nk, nb_carrier, 2,
    ngkmax)`` c128 (``lxkit.memory``, the ``_book_density_price`` pattern).
    Replicated: the projection table ``(nk, R_os, nb_carrier)`` c128 and the
    G table with its mask.  The per-k Loewdin rows are a scan carry.
    """
    from lxkit import memory
    c128 = 16.0
    sphere = float(nk) * float(nb_carrier) * 2.0 * float(ngkmax) * c128
    table = float(nk) * float(r_os) * float(nb_carrier) * c128
    gtab = float(nk) * float(ngkmax) * (3 * 4.0 + 8.0)
    memory.book(memory.Price(
        operands=memory.Term(table + gtab, sphere,
                             "full-BZ psi(G) sphere + on-site projection table + G table"),
        ranks=max(1, int(ranks)), label="on-site Breit projection", increment=True))


def _onsite_projection_table(wfn, sym, setup, band_range, mesh):
    """``P[k, a, n] = <phi~_a(k)|psi_nk>`` (nk_full, R_os, nb), replicated, cached.

    psi is the two-component DFT sphere of the GLOBAL band range, unfolded to
    the full BZ by the loader and band-sharded at ``band_sphere_spec``; the
    loader pads the band axis to the mesh carrier with zero bands, whose
    projections are zero, and the table is stripped back to the logical nb
    columns by the carrier's receipt.  psi is dropped before returning.
    """
    from common.parallel_transport import wfn_fingerprint
    from common.wfn_layout import band_sphere_spec
    from psp.dft_operators import padded_gvectors
    from psp.onsite_breit import onsite_projections
    from runtime.padding import authenticate_axis, padded_axis, strip_axis

    key = (wfn_fingerprint(wfn), tuple(band_range), setup.provenance, mesh)
    hit = _ONSITE_PROJECTIONS.get(key)
    if hit is not None:
        return hit
    nk = int(sym.nk_tot)
    b_lo, b_hi = band_range
    gtab = padded_gvectors(wfn, k="full_bz")
    kv, k_sym = np.asarray(gtab.kvecs), np.asarray(sym.unfolded_kpts, dtype=np.float64)
    if kv.shape != (nk, 3) or not np.array_equal(kv, k_sym):
        raise _onsite_gate(
            "onsite_breit_k_order", f"loader full-BZ k table {kv.shape} vs sym.unfolded_kpts "
            f"{k_sym.shape} (equal: {kv.shape == k_sym.shape and np.array_equal(kv, k_sym)})",
            "WfnLoader.kvecs(k='full_bz') == sym.unfolded_kpts, row by row",
            "P, U and the occupations are paired by full-BZ row; another order pairs a state "
            "with another k's projections", "use the loader's own symmetry() for sym")
    band_axis = padded_axis(b_hi - b_lo, mesh, name="on-site projection band sphere",
                            spec=band_sphere_spec(), axis=1)
    _book_onsite_price(nk=nk, nb_carrier=band_axis.carrier,
                       ngkmax=int(gtab.gvecs.shape[1]), r_os=setup.R_os,
                       ranks=int(mesh.devices.size))
    psi_G = wfn.load(bands=(b_lo, b_hi), k="full_bz", sharding=band_sphere_spec(),
                     bispinor=False)
    authenticate_axis(psi_G, band_axis, axis=1, where="onsite_breit_field psi(G)")
    with timing.section("onsite_projections"):
        P_carrier = onsite_projections(psi_G, gtab.gvecs, gtab.mask, gtab.kvecs, setup,
                                       mesh=mesh)
    del psi_G
    P_table = strip_axis(P_carrier, band_axis, axis=2)
    _ONSITE_PROJECTIONS[key] = P_table
    return P_table


def _onsite_dft_occupations(wfn, nk: int, band_range, kweights):
    """The WFN's canonical full-BZ occupations on the window (nk, nb), host.

    ``wfn.physical_density_occupations(k="full_bz")`` (unit below the
    density band stop when it returns ``None``).  Occupied weight at or
    above the window top would be dropped from D: it refuses
    (``GATE onsite_breit_band_cut``, the budget of the SC density cut).
    """
    from gw.qsgw_density import DENSITY_TAIL_ELECTRON_TOL, density_tail_electrons
    from psp.get_DFT_mtxels import spin_degeneracy_factor
    b_lo, b_hi = band_range
    stop = int(wfn.physical_density_band_stop)
    rows = wfn.physical_density_occupations(k="full_bz")
    occ = np.zeros((nk, max(stop, b_hi)), dtype=np.float64)
    occ[:, :stop] = 1.0 if rows is None else np.asarray(rows, dtype=np.float64)[:, :stop]
    dropped = float(density_tail_electrons(occ, kweights, spin_degeneracy_factor(wfn))[b_hi])
    if not np.isfinite(dropped) or dropped > DENSITY_TAIL_ELECTRON_TOL:
        raise _onsite_gate(
            "onsite_breit_band_cut", f"bands at or above {b_hi} carry {dropped:.3e} electrons",
            f"<= {DENSITY_TAIL_ELECTRON_TOL:.0e} electrons above the window [{b_lo}, {b_hi})",
            "D is built from the window's states only; an occupied state above it would be "
            "silently dropped", "widen the window (ncond)")
    return occ[:, b_lo:b_hi]


def _onsite_density_gate(D, occ, setup) -> None:
    """Finite, Hermitian, and each shell block's spectrum inside the occupation range."""
    D = np.asarray(D)
    scale = max(float(np.abs(D).max()) if D.size else 0.0, 1.0)
    herm = float(np.abs(D - D.conj().T).max()) if D.size else 0.0
    if not np.all(np.isfinite(D)) or herm > ONSITE_HERMITICITY_RTOL * scale:
        raise _onsite_gate(
            "onsite_breit_density", f"finite {bool(np.all(np.isfinite(D)))}, max|D - D^+| = {herm:.2e}",
            f"finite and max|D - D^+| <= {ONSITE_HERMITICITY_RTOL:g} max(1, max|D|)",
            "D is a density matrix; a non-Hermitian or non-finite D is a broken projection "
            "or rotation", "check the projection table and U")
    f_lo = min(0.0, float(jnp.min(occ)))
    f_hi = max(1.0, float(jnp.max(occ)))
    for sa in setup.shell_atoms:
        sl = slice(sa["start"], sa["stop"])
        ev = np.linalg.eigvalsh(0.5 * (D[sl, sl] + D[sl, sl].conj().T))
        if ev.min() < f_lo - ONSITE_DENSITY_EIGEN_TOL or ev.max() > f_hi + ONSITE_DENSITY_EIGEN_TOL:
            raise _onsite_gate(
                "onsite_breit_density",
                f"atom {sa['atom']} {sa['element']} {sa['label']}: eigenvalues of D in "
                f"[{ev.min():+.6f}, {ev.max():+.6f}]",
                f"inside the occupation range [{f_lo:g}, {f_hi:g}] +- {ONSITE_DENSITY_EIGEN_TOL:g}",
                "orthonormal projector rows and orthonormal states bound D by the occupations; "
                "outside it the rows or the states are not orthonormal",
                "check the Loewdin rows (UPF, mask) and the normalisation of psi and U")


def onsite_breit_field(wfn, sym, setup, *, band_range, mesh, U=None, occupations=None,
                       print_fn=None) -> OnsiteBreitField:
    """D, h and H_OS on the full BZ for the bands ``[band_range[0], band_range[1])``.

    Parameters
    ----------
    wfn : wfn_loader.WfnLoader
    sym : symmetry_maps.SymMaps
        ``wfn.symmetry()``; its ``unfolded_kpts`` order is the k order of
        every array in and out.
    setup : psp.onsite_breit.OnsiteSetup
        :func:`onsite_breit_setup_for`.
    band_range : (int, int)
        GLOBAL band indices, like ``band_slices.sigma_range``; nb = b1 - b0.
    mesh : jax.sharding.Mesh
        The run's mesh.
    U : (nk_full, nb, nb) complex, optional
        ``U[k, n, i] = <DFT_n|QP_i>``, full-BZ rows in sym order, replicated
        or mesh-sharded (contracted at its own layout).
    occupations : (nk_full, nb) float, optional
        Occupations of the states i (the map's fixed-N state for a metal).
        ``U`` and ``occupations`` both ``None``: the DFT level, U = 1 and the
        WFN's canonical occupations.  ``U`` without ``occupations`` refuses.
    print_fn : callable, optional
        Where the report lines go (default ``log.info``; rank 0 by the
        session's handlers).  Every call reports.

    Returns
    -------
    OnsiteBreitField

    The physics, in order (design note section 2)::

        P[k, a, n] = <phi~_a(k)|psi_nk>                         (cached per run)
        w_k        = 1 / nk_full
        D[a, b]    = sum_k w_k sum_i f_ik (P U)_{k a i} conj((P U)_{k b i})
        h_I        = h^dir[D_I] + h^x[D_I]                       (section 3 keys; Ry)
        H_OS[k]    = P[k]^dagger h P[k]                          (Ry)

    Gates (``QUALITY_PATTERNS`` #7): D finite, Hermitian and bounded by the
    occupations (``GATE onsite_breit_density``); H_OS finite and Hermitian
    (``GATE onsite_breit_matrix``).
    """
    from psp.onsite_breit import (RY_TO_MEV, onsite_band_matrix, onsite_block_potential,
                                  onsite_density_matrix, onsite_report_lines)

    b_lo, b_hi = int(band_range[0]), int(band_range[1])
    nb, nk = b_hi - b_lo, int(sym.nk_tot)
    if nb <= 0 or b_lo < 0:
        raise ValueError(f"onsite_breit_field: empty or negative band range {band_range}")
    if U is not None and occupations is None:
        raise ValueError("onsite_breit_field: U needs the occupations of its states")
    emit = log.info if print_fn is None else print_fn
    with timing.section("onsite_field"):
        # P[k, a, n] = <phi~_a(k)|psi_nk>, full BZ in sym.unfolded_kpts order
        P_table = _onsite_projection_table(wfn, sym, setup, (b_lo, b_hi), mesh)
        # w_k = 1 / N_k on the full BZ
        kweights = np.full(nk, 1.0 / nk, dtype=np.float64)
        # f_ik: the map's occupations, or the WFN's at the DFT level (U = 1)
        occ = (_onsite_dft_occupations(wfn, nk, (b_lo, b_hi), kweights)
               if occupations is None else occupations)
        if tuple(np.shape(occ)) != (nk, nb if U is None else int(np.shape(U)[2])):
            raise ValueError(f"onsite_breit_field: occupations {np.shape(occ)} do not match "
                             f"(nk_full={nk}, {nb if U is None else np.shape(U)[2]})")
        if U is not None and tuple(np.shape(U))[:2] != (nk, nb):
            raise ValueError(f"onsite_breit_field: U {np.shape(U)} is not (nk_full={nk}, nb={nb}, .)")
        # D[a, b] = sum_k w_k sum_i f_ik (P U)_{k a i} conj((P U)_{k b i})
        D = onsite_density_matrix(P_table, kweights, occ, setup, U=U, mesh=mesh)
        _onsite_density_gate(D, occ, setup)
        # h_I = h^dir[D_I] + h^x[D_I] (block-diagonal over shell-atoms, Ry)
        h_block, pieces = onsite_block_potential(D, setup)
        # H_OS[k, n, n'] = sum_ab conj(P[k, a, n]) h[a, b] P[k, b, n'] (Ry)
        H_os = onsite_band_matrix(P_table, h_block, mesh=mesh)
        h_max = float(jnp.max(jnp.abs(H_os)))
        h_anti = float(jnp.max(jnp.abs(H_os - jnp.conj(jnp.swapaxes(H_os, -1, -2)))))
        if not np.isfinite(h_max) or h_anti > ONSITE_HERMITICITY_RTOL * max(h_max, 1e-300):
            raise _onsite_gate(
                "onsite_breit_matrix", f"max|H_OS| = {h_max:.3e} Ry, max|H - H^+| = {h_anti:.3e} Ry",
                f"finite and max|H - H^+| <= {ONSITE_HERMITICITY_RTOL:g} max|H_OS|",
                "H_OS = P^+ h P with Hermitian h is Hermitian; otherwise h or P is broken",
                "check the on-site potential pieces and the projection table")
    report = onsite_report_lines(D, pieces, setup) + [
        f"onsite_breit H_OS: bands [{b_lo}, {b_hi}) (global), {nk} full-BZ k, "
        f"D from {'the WFN occupations (U = 1)' if U is None else 'the map (U, f)'}, "
        f"double_counting = {setup.double_counting}, exchange = {setup.exchange}, "
        f"max|H_OS| = {h_max * RY_TO_MEV:.5f} meV, max|H - H^+| = {h_anti * RY_TO_MEV:.1e} meV"]
    for line in report:
        emit(line)
    return OnsiteBreitField(matrix=H_os, density_matrix=np.asarray(D), h_block=h_block,
                            pieces=pieces, projections=P_table, setup=setup, report=report)
