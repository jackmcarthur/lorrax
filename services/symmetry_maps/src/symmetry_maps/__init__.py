"""``symmetry_maps`` — the crystal-symmetry service: tables, unfolds, verdict.

Everything LORRAX knows about the space group of a deck, in one package:
the k-grid reduction and the IBZ⇄full-BZ tables (:class:`SymMaps`), the
band-index k-star map (:class:`KStarMap` and the ``star_*`` helpers), the
sharded q-axis unfolds (:func:`unfold_isdf_operator`,
:func:`mix_channels_by_proper_rotation`), the ψ-unfold antiunitary rule
(:func:`unfold_psi`, :func:`spinor_rotation_for_sym_row`,
:func:`tau_phase_row`), the real-space orbit machinery the ISDF
quadrature is built on (:func:`real_space_action_tables`,
:func:`centroid_source_map_and_wrap`, …), and
the TRS *measurement* (:func:`check_density_symmetries`) that decides
whether the flags in the file may be believed at all.

THE PACKAGE IS THE DOOR.  There is no separate facade module: everything a
consumer needs is a top-level name here, and importing
``symmetry_maps.<submodule>`` from outside is a layering violation the
monorepo's ``tests/test_layering.py`` fails on.  The private names
(``_I_SIGMA_Y``, the ``_star_*`` helpers) are deliberately NOT re-exported:
they are the service's own tests' business.

SIX NAMES WERE RENAMED ON 2026-08-08 and every old spelling still
resolves.  Operations are named for what they mathematically DO, not for
what they are applied to — ``unfold_v_q`` unfolded any ISDF-basis
operator and V was merely its most frequent customer, so it is
:func:`unfold_isdf_operator` now.  The pre-sweep spellings are bound as
call-through aliases here AND on their defining modules; the map, the
reason and the retirement gate are in :mod:`symmetry_maps._compat` and
in :data:`RENAMES`.  The pair that mattered most is
:func:`centroid_source_map_and_wrap` (a SOURCE map plus a lattice wrap)
against :func:`fft_grid_pullback_perm` (a PULL-BACK permutation): they
used to share a name shape and return opposite directions, which is the
recorded mechanism of a silent 4 eV gap on hex systems.

TWO THINGS THAT LOOK ALIKE AND ARE NOT.  :func:`star_broadcast` moves a
BAND-INDEX object (a self-energy, an eigenvector matrix) from the IBZ to
the full BZ, and that is pure indexing plus a conjugation on
time-reversed rows.  :func:`unfold_isdf_operator` moves a CENTROID-indexed
object and needs the double gather, the umklapp phase and the TRS conjugation,
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
``directed_edge_orbit_table`` / ``apply_band_matrix_symmetry``
    Pure-array signed directed-edge metadata, and the one endpoint-sewn
    unitary/antiunitary band-matrix action.  Nonsymmorphic phases enter through
    the endpoint sewing matrices; vector/covector mixing is an explicit
    caller-supplied component representation.
``unfold_file_wedge_to_full_bz`` / ``unfold_star_wedge_to_full_bz``
    THE TWO NAMED UNFOLDS, taking a ``SymMaps`` rather than index tables
    so a driver never holds one.  TWO, because there are two different
    IBZs and they are NOT the same size: the FILE wedge (``wfn.kpoints``,
    ``nk_red``, what every .dat is indexed by and what BerkeleyGW means)
    and the STAR wedge (``star_select``'s, one row per orbit).  Measured,
    they coincide on ``si_cohsex_debug`` (8 = 8) and diverge on
    ``cohsex_debug`` (4 vs 3) and ``gnppm_debug`` (9 vs 5) — so ONE name
    would have been right on the deck most gates run and silently wrong
    elsewhere.  Both are thin wrappers over ONE backend
    (``star_broadcast``); a THIRD means the parameterisation is wrong.
``reduce_full_bz_to_file_wedge``
    The one reduction: full BZ → the rows that ARE ``wfn.kpoints``, so a
    writer never holds ``kirr_fullids``.  Pure selection.  NOT the exact
    inverse of the file-wedge unfold — see its docstring.  There is no
    star-wedge twin because that is ``star_select``.
``unfold_isdf_operator`` / ``mix_channels_by_proper_rotation`` /
``slice_q_full_to_ibz``
    The sharded q-axis unfolds, ``shard_map`` over an ``('x','y')`` mesh
    with a 1× single-tile memory contract.
``unfold_psi`` / ``spinor_rotation_for_sym_row`` / ``tau_phase_row``
    The ψ-unfold rule, and the single sources of the spinor TRS
    augmentation and the τ phase.
``kgrid_shift_map`` / ``find_irreducible_bz_points``
    Pure integer k-grid algebra: the C-order fold with its umklapp G, and
    the IBZ reduction (derive-IBZ and anchored-IBZ branches).
``real_space_action_tables`` / ``orbit_images`` / ``canonicalize_orbit`` /
``unfold_orbit_unique_with_id`` / ``centroid_source_map_and_wrap`` /
``fft_grid_pullback_perm`` / ``recover_symmorphic_density_point_group``
    Real space: the FFT-grid permutation tables the ISDF centroids and
    the ρ symmetrisation ride on, and the holohedry recovery for decks
    whose header symmetry list is a subset of the true point group.
``write_qirr_tensor`` / ``read_tensor`` / ``read_tables`` /
``allocate_qirr_placeholder`` / ``QirrTables`` / ``QirrHeader``
    q_irr restart tensors: the PRE-UNFOLD wedge on disk with the tables
    that unfold it in the same file, unfolded on read.  The writer
    REFUSES against a non-closed centroid set; the reader takes the
    SHAPE as the primary discriminant and the ``q_storage`` attr as its
    cross-check, refuses on version / hash / table drift, and refuses a
    placeholder whose data was never persisted.  A file with no attrs is
    read as full-BZ, unchanged.  h5py is imported lazily inside the two
    functions that open a file, so ``import symmetry_maps`` still needs
    nothing but jax and numpy.
``verify_centroid_orbit_closure`` / ``CentroidClosureVerdict``
    Orbit closure as a MEASUREMENT rather than an exception:
    ``centroid_source_map_and_wrap`` refuses on a non-closed set, this says
    by how much and on which ops.  It has no positional slot for the
    translations, and enforces a mutually-exclusive ``tnp=``/``tau=``
    keyword pair, precisely so that no caller can pass the wrong
    convention silently.

    IT IS NOT "the one place the 2π is divided" — this doc used to say
    that, and it is false: ``orbit_syms`` divides at :192, :478, :887 and
    :1308, and multiplies back at :1224.  THE REAL RULE, which is what a
    caller needs:

      ``SymMaps.translations`` is RAW BGW ``tnp`` (= 2π·τ).
      G-space consumes it UNDIVIDED — :func:`tau_phase_row` wants ``tnp``.
      Real space DIVIDES by 2π — every ``orbit_syms`` entry point does.

    So ``tau_phase_row(S, sym.translations[s], G)`` and
    ``centroid_source_map_and_wrap(..., sym.translations)`` take the SAME
    array and mean DIFFERENT things by it.  Both are right today only
    because each happens to want the convention it gets; nothing checks
    it.  Passing one's argument to the other is a 2π error that no shape
    or dtype catches.
``resolve_qgrid_symmetry`` / ``QgridSymmetryResolution``
    The DECISION, taken once: verdict → mode (``"ibz"`` | ``"full_bz"``)
    → tables → reason, in one object.  Consumers of the q-axis unfold
    used to each wrap the table builder in their own
    ``except RuntimeError`` and silently drop the whole run to the full
    BZ; they now read ``use_ibz`` and announce the fallback exactly once.
    The resolution COMPOSES the announcement and does not print it —
    rank and once-per-run memory belong to the process running the deck,
    not to a table library.
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
from symmetry_maps.directed_edges import (
    apply_band_matrix_symmetry,
    directed_edge_orbit_table,
    q_stencil_orbit_table,
)
from symmetry_maps.maps import (
    KStarMap,
    SymMaps,
    find_irreducible_bz_points,
    kgrid_shift_map,
    slice_q_full_to_ibz,
    star_broadcast,
    reduce_full_bz_to_file_wedge,
    star_tables_of,
    unfold_file_wedge_to_full_bz,
    unfold_star_wedge_to_full_bz,
    star_select,
    star_spread,
    tau_phase_row,
    spinor_rotation_for_sym_row,
    unfold_psi,
    unfold_isdf_operator,
    mix_channels_by_proper_rotation,
)
# Pre-sweep spellings.  Imported from the modules that define them, so
# the door and the module bind the SAME object and cannot drift apart.
from symmetry_maps.maps import (              # noqa: F401  (compat surface)
    trs_augment_U,
    unfold_v_q,
    unfold_v_q_bispinor_lorentz,
)
from symmetry_maps.qirr_store import (
    QIRR_FORMAT_VERSION,
    QIRR_RANK_BY_VERSION,
    QIRR_TABLE_SUFFIX,
    QIRR_VERSION_ATTR,
    QirrDest,
    QirrHeader,
    QirrTables,
    allocate_qirr_placeholder,
    dataset_q_storage,
    qirr_attr_str,
    qirr_generator_commit,
    read_tables,
    read_tensor,
    stamp_qirr_tensor,
    validate_qirr_tables,
    write_qirr_tensor,
)
from symmetry_maps.orbit_syms import (
    CLOSURE_TOL_DEFAULT,
    FULL_BZ_CONSEQUENCE,
    CentroidClosureVerdict,
    QgridSymmetryResolution,
    real_space_action_tables,
    canonicalize_orbit,
    centroid_source_map_and_wrap,
    fft_grid_pullback_perm,
    grid_point_image_perm,
    orbit_images,
    r_action_forward,
    r_action_forward_one,
    snap_to_grid_and_split_wrap,
    recover_symmorphic_density_point_group,
    resolve_qgrid_symmetry,
    unfold_orbit_unique_with_id,
    verify_centroid_orbit_closure,
)
from symmetry_maps.orbit_syms import (        # noqa: F401  (compat surface)
    build_real_space_syms,
    compute_centroid_sym_perm,
    compute_rgrid_sym_perm,
)
from symmetry_maps._compat import RENAMES, RETIREMENT_GATE  # noqa: F401

__all__ = [
    # tables
    "SymMaps", "kgrid_shift_map", "find_irreducible_bz_points",
    # k-stars (band-index IBZ<->full BZ)
    "KStarMap", "star_select", "star_broadcast", "star_spread",
    # directed band-matrix edges: pure table + the one symmetry action
    "directed_edge_orbit_table", "q_stencil_orbit_table",
    "apply_band_matrix_symmetry",
    # the two named unfolds (file wedge vs star wedge)
    "unfold_file_wedge_to_full_bz", "unfold_star_wedge_to_full_bz",
    "reduce_full_bz_to_file_wedge", "star_tables_of",
    # sharded q-axis unfolds
    "slice_q_full_to_ibz", "unfold_isdf_operator",
    "mix_channels_by_proper_rotation",
    # psi unfold / antiunitary rule
    "unfold_psi", "spinor_rotation_for_sym_row", "tau_phase_row",
    # real-space orbits
    "real_space_action_tables", "orbit_images", "canonicalize_orbit",
    "unfold_orbit_unique_with_id", "centroid_source_map_and_wrap",
    "fft_grid_pullback_perm", "grid_point_image_perm",
    "r_action_forward", "r_action_forward_one",
    "snap_to_grid_and_split_wrap",
    "recover_symmorphic_density_point_group",
    # orbit closure: the measurement, its verdict, its tolerance
    "verify_centroid_orbit_closure", "CentroidClosureVerdict",
    "CLOSURE_TOL_DEFAULT",
    # orbit closure: the RESOLUTION the consumers branch on
    "resolve_qgrid_symmetry", "QgridSymmetryResolution",
    "FULL_BZ_CONSEQUENCE",
    # q_irr restart tensors: the wedge on disk, the unfold on read
    "write_qirr_tensor", "read_tensor", "read_tables",
    "dataset_q_storage", "stamp_qirr_tensor",
    "allocate_qirr_placeholder", "QirrTables", "QirrHeader",
    "QIRR_FORMAT_VERSION", "QIRR_RANK_BY_VERSION",
    # ...and the format's own PLUMBING, on the door rather than reached
    # for through ``symmetry_maps.qirr_store``.  A second store writing
    # this layout — the frequency-resolved W — opens the same handles,
    # reads the same attrs, stamps the same provenance and owes the
    # tables the same validation; underscoring these would not keep it
    # out, it would only make it copy them, and a copied validator is a
    # second answer to "do these tables describe this tensor".  Same
    # reason ``vcoul`` puts ``_minibz_kernel_bare`` on its door: a
    # consumer reaching the submodule for one of these is a
    # past-the-door edge and ``tests/test_layering.py`` rule 6 counts it.
    "QIRR_VERSION_ATTR", "QIRR_TABLE_SUFFIX", "QirrDest", "qirr_attr_str",
    "qirr_generator_commit", "validate_qirr_tables",
    # the TRS measurement
    "DensitySymmetryReport", "check_density_symmetries",
    "cached_density_symmetry_check", "trs_check_mode",
    # ---------------------------------------------------------------
    # PRE-SWEEP SPELLINGS.  Aliases, not definitions — see
    # ``symmetry_maps._compat`` for the map and the retirement gate
    # (the documentation pass).  They are in ``__all__`` because a name
    # that ``from symmetry_maps import *`` drops is a name a consumer
    # discovers is missing at run time rather than at import time, and
    # the whole point of the alias is that nothing discovers anything.
    "unfold_v_q", "unfold_v_q_bispinor_lorentz", "trs_augment_U",
    "compute_centroid_sym_perm", "compute_rgrid_sym_perm",
    "build_real_space_syms",
    "RENAMES", "RETIREMENT_GATE",
]
