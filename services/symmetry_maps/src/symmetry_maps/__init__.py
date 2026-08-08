"""``symmetry_maps`` — the crystal-symmetry service: tables, unfolds, verdict.

Everything LORRAX knows about the space group of a deck, in one package:
the k-grid reduction and the IBZ⇄full-BZ tables (:class:`SymMaps`), the
band-index k-star map (:class:`KStarMap` and the ``star_*`` helpers), the
sharded q-axis unfolds (:func:`unfold_v_q`,
:func:`unfold_v_q_bispinor_lorentz`), the ψ-unfold antiunitary rule
(:func:`unfold_psi`, :func:`trs_augment_U`, :func:`tau_phase_row`), the
real-space orbit machinery the ISDF quadrature is built on
(:func:`build_real_space_syms`, :func:`compute_centroid_sym_perm`, …), and
the TRS *measurement* (:func:`check_density_symmetries`) that decides
whether the flags in the file may be believed at all.

THE PACKAGE IS THE DOOR.  There is no separate facade module: everything a
consumer needs is a top-level name here, and importing
``symmetry_maps.<submodule>`` from outside is a layering violation the
monorepo's ``tests/test_layering.py`` fails on.  The private names
(``_I_SIGMA_Y``, the ``_star_*`` helpers) are deliberately NOT re-exported:
they are the service's own tests' business.

TWO THINGS THAT LOOK ALIKE AND ARE NOT.  :func:`star_broadcast` moves a
BAND-INDEX object (a self-energy, an eigenvector matrix) from the IBZ to
the full BZ, and that is pure indexing plus a conjugation on
time-reversed rows.  :func:`unfold_v_q` moves a CENTROID-indexed object
and needs the double gather, the umklapp phase and the TRS conjugation,
because a symmetry moves r and does not move bands.  The two are not
variants of each other; the argument is written out above
:func:`star_select` in :mod:`symmetry_maps.maps`.

TIME REVERSAL IS A MEASUREMENT HERE, NOT A CONVENTION.
:func:`check_density_symmetries` builds ρ from the raw IBZ coefficients
and asks whether ``w_k m_k + w_{−k} m_{−k}`` vanishes; the verdict lands
on the loader as ``trs_holds`` and :class:`SymMaps` refuses to select
time-reversal rows when the density says TRS is broken, whatever the
flags in the file claim.  That is why the check is in this package and
not beside it: the thing it protects is the unfold.

The surface
-----------
``SymMaps(wfn)``
    The eager table builder.  Publishes ``irr_idx_k``/``sym_idx_k`` (the
    full-BZ → IBZ row and the operator that gets you there), the q-axis
    twins, the rotation tables (``R_grid``, ``Rinv_grid``, ``R_cart``,
    ``R_proper``, ``U_spinor``) and the k−q maps.  ``R_cart`` is the
    INVERSE Cartesian rotation; rotating a Cartesian index with it
    untransposed passes norms, traces and hermiticity while being wrong.
``KStarMap`` / ``star_select`` / ``star_broadcast`` / ``star_spread``
    IBZ⇄full-BZ for band-index objects, and the residual that checks the
    premise.
``unfold_v_q`` / ``unfold_v_q_bispinor_lorentz`` / ``slice_q_full_to_ibz``
    The sharded q-axis unfolds, ``shard_map`` over an ``('x','y')`` mesh
    with a 1× single-tile memory contract.
``unfold_psi`` / ``trs_augment_U`` / ``tau_phase_row``
    The ψ-unfold rule, and the single sources of the spinor TRS
    augmentation and the τ phase.
``kgrid_shift_map`` / ``find_irreducible_bz_points``
    Pure integer k-grid algebra: the C-order fold with its umklapp G, and
    the IBZ reduction (derive-IBZ and anchored-IBZ branches).
``build_real_space_syms`` / ``orbit_images`` / ``canonicalize_orbit`` /
``unfold_orbit_unique_with_id`` / ``compute_centroid_sym_perm`` /
``compute_rgrid_sym_perm`` / ``recover_symmorphic_density_point_group``
    Real space: the FFT-grid permutation tables the ISDF centroids and
    the ρ symmetrisation ride on, and the holohedry recovery for decks
    whose header symmetry list is a subset of the true point group.
``verify_centroid_orbit_closure`` / ``CentroidClosureVerdict``
    Orbit closure as a MEASUREMENT rather than an exception:
    ``compute_centroid_sym_perm`` refuses on a non-closed set, this says
    by how much and on which ops.  It is also **the one place in the
    service where ``tnp = 2π·τ`` is divided**, and it has no positional
    slot for the translations precisely so that no caller can pass the
    wrong convention silently.
``DensitySymmetryReport`` / ``check_density_symmetries`` /
``cached_density_symmetry_check`` / ``trs_check_mode``
    The measurement, its cached front door, and the ``LORRAX_TRS_CHECK``
    switch.

Op-selection policy: REGISTERED, NOT TOUCHED
--------------------------------------------
``SymMaps.find_symmetry_ops_simple`` has no ``break`` on its inner loop,
so the HIGHEST matching irreducible k wins and then the lowest symmetry
index for that k.  ``find_irreducible_bz_points``' anchored branch
reproduces the same policy independently and says so at the site.  Both
moved into this package VERBATIM.  The policy is not a bug to be tidied:
changing it was measured to move ``test_gw_jax_matches_reference[cohsex]``
past tolerance (360/1620 elements, max abs **15.9 eV** in the V_H column)
and to break ``test_fixed_point_frozen_qp_rotations``, while ``diag(V_H)``
stays bit-identical — so the 15.9 eV originates downstream and is not
localised.  Owner row.  The bit-equality of ``(irr_idx_k, sym_idx_k)`` on
all four in-tree decks against ``tests/data/star_tables_e9340d1.json`` is
the tripwire that says if it ever moves.
"""

from __future__ import annotations

from symmetry_maps.density_symmetry_check import (
    DensitySymmetryReport,
    cached_density_symmetry_check,
    check_density_symmetries,
    trs_check_mode,
)
from symmetry_maps.maps import (
    KStarMap,
    SymMaps,
    find_irreducible_bz_points,
    kgrid_shift_map,
    slice_q_full_to_ibz,
    star_broadcast,
    star_select,
    star_spread,
    tau_phase_row,
    trs_augment_U,
    unfold_psi,
    unfold_v_q,
    unfold_v_q_bispinor_lorentz,
)
from symmetry_maps.orbit_syms import (
    CLOSURE_TOL_DEFAULT,
    CentroidClosureVerdict,
    build_real_space_syms,
    canonicalize_orbit,
    compute_centroid_sym_perm,
    compute_rgrid_sym_perm,
    orbit_images,
    recover_symmorphic_density_point_group,
    unfold_orbit_unique_with_id,
    verify_centroid_orbit_closure,
)

__all__ = [
    # tables
    "SymMaps", "kgrid_shift_map", "find_irreducible_bz_points",
    # k-stars (band-index IBZ<->full BZ)
    "KStarMap", "star_select", "star_broadcast", "star_spread",
    # sharded q-axis unfolds
    "slice_q_full_to_ibz", "unfold_v_q", "unfold_v_q_bispinor_lorentz",
    # psi unfold / antiunitary rule
    "unfold_psi", "trs_augment_U", "tau_phase_row",
    # real-space orbits
    "build_real_space_syms", "orbit_images", "canonicalize_orbit",
    "unfold_orbit_unique_with_id", "compute_centroid_sym_perm",
    "compute_rgrid_sym_perm", "recover_symmorphic_density_point_group",
    # orbit closure: the measurement, its verdict, its tolerance
    "verify_centroid_orbit_closure", "CentroidClosureVerdict",
    "CLOSURE_TOL_DEFAULT",
    # the TRS measurement
    "DensitySymmetryReport", "check_density_symmetries",
    "cached_density_symmetry_check", "trs_check_mode",
]
